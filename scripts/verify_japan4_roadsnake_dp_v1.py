"""Static, real-batch, BN-ownership and deployment audit for RoadSnake-DP-v1."""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import YOLO  # noqa: E402
from ultralytics.cfg import DEFAULT_CFG, get_cfg  # noqa: E402
from ultralytics.data.build import build_dataloader, build_yolo_dataset  # noqa: E402
from ultralytics.data.utils import check_det_dataset  # noqa: E402
from ultralytics.models.yolo.detect.train import DetectionTrainer  # noqa: E402
from ultralytics.nn.modules.head import Detect  # noqa: E402
from ultralytics.nn.roadsnake import RoadSnakeDualPathDetect  # noqa: E402
from ultralytics.nn.tasks import DetectionModel  # noqa: E402
from ultralytics.utils.roadsnake_dp_loss import RoadSnakeDualPathE2ELoss  # noqa: E402

BASELINE = ROOT / "ultralytics/cfg/models/26/yolo26.yaml"
CANDIDATE = ROOT / "ultralytics/cfg/models/26/yolo26n-japan4-roadsnake-dp-v1.yaml"


def make_real_batch(data_yaml: Path, imgsz: int, batch_size: int) -> dict:
    data = check_det_dataset(str(data_yaml))
    cfg = get_cfg(
        DEFAULT_CFG,
        {
            "mode": "train",
            "task": "detect",
            "imgsz": imgsz,
            "batch": batch_size,
            "workers": 0,
            "rect": False,
            "cache": False,
            "mosaic": 0.0,
            "mixup": 0.0,
            "copy_paste": 0.0,
        },
    )
    dataset = build_yolo_dataset(cfg, data["train"], batch_size, data, mode="train", rect=False, stride=32)
    loader = build_dataloader(dataset, batch_size, 0, shuffle=False, rank=-1, pin_memory=False)
    return next(iter(loader))


def shared_bn_state(head: RoadSnakeDualPathDetect) -> dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    output = {}
    for index, module in enumerate(head._shared_detect_batch_norms()):
        output[str(index)] = (
            module.running_mean.detach().clone(),
            module.running_var.detach().clone(),
            module.num_batches_tracked.detach().clone(),
        )
    return output


def assert_bn_state_equal(left, right) -> None:
    if left.keys() != right.keys():
        raise AssertionError("Detect BN module sets differ")
    for key in left:
        for lhs, rhs in zip(left[key], right[key]):
            torch.testing.assert_close(lhs, rhs, atol=0, rtol=0)


def build_models(weights: Path, device: torch.device):
    checkpoint = YOLO(str(weights), task="detect", verbose=False).model.float()
    baseline = DetectionModel(str(BASELINE), nc=4, ch=3, verbose=False).float()
    baseline.load(checkpoint, verbose=False)
    candidate = DetectionModel(str(CANDIDATE), nc=4, ch=3, verbose=False).float()
    source, target = baseline.state_dict(), candidate.state_dict()
    missing = [key for key, value in source.items() if key not in target or target[key].shape != value.shape]
    if missing:
        raise AssertionError(f"DP cannot inherit B0 tensors: {missing[:8]}")
    candidate.load(baseline, verbose=False)
    changed = [key for key, value in source.items() if not torch.equal(value, candidate.state_dict()[key])]
    if changed:
        raise AssertionError(f"DP changed inherited B0 tensors: {changed[:8]}")

    trainer = object.__new__(DetectionTrainer)
    trainer.data = {"nc": 4, "channels": 3}
    rebuilt = DetectionTrainer.get_model(trainer, cfg=str(CANDIDATE), weights=candidate, verbose=False).float()
    rebuild_changed = [
        key for key, value in candidate.state_dict().items() if not torch.equal(value, rebuilt.state_dict()[key])
    ]
    if rebuild_changed:
        raise AssertionError(f"Trainer reconstruction changed DP tensors: {rebuild_changed[:8]}")
    hyp = get_cfg(DEFAULT_CFG, {"box": 7.5, "cls": 0.5, "dfl": 1.5, "epochs": 30})
    for model in (baseline, rebuilt):
        model.args = hyp
        model.criterion = None
    return baseline.to(device), rebuilt.to(device), {
        "b0_state_items": len(source),
        "dp_state_items": len(target),
        "shared_missing_or_mismatched": len(missing),
        "shared_changed_after_load": len(changed),
        "trainer_rebuild_changed": len(rebuild_changed),
        "new_state_items": sorted(set(target) - set(source)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=ROOT / "yolo26n.pt")
    parser.add_argument("--data", type=Path, default=ROOT / "configs/japan4_clean_v3_local.yaml")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--imgsz", type=int, default=128)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--onnx", type=Path)
    parser.add_argument("--report", type=Path, default=ROOT / "reports/japan4_roadsnake_dp_v1_static.json")
    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    device = torch.device(args.device)
    baseline, candidate, transfer = build_models(args.weights.resolve(), device)
    head = candidate.model[-1]
    if not isinstance(head, RoadSnakeDualPathDetect):
        raise TypeError(type(head).__name__)
    if float(head.road_snake.gamma.detach().item()) != 0.0:
        raise AssertionError("DP must initialize at exact native identity")
    if candidate.stride.tolist() != [8.0, 16.0, 32.0]:
        raise AssertionError(candidate.stride.tolist())
    criterion = candidate.init_criterion()
    if not isinstance(criterion, RoadSnakeDualPathE2ELoss):
        raise TypeError(type(criterion).__name__)

    # Eval never executes the removable adapter and must equal B0 exactly.
    sample = torch.randn(1, 3, args.imgsz, args.imgsz, device=device)
    adapter_calls = 0

    def count_adapter(*_):
        nonlocal adapter_calls
        adapter_calls += 1

    hook = head.road_snake.register_forward_hook(count_adapter)
    baseline.eval()
    candidate.eval()
    with torch.inference_mode():
        b0_eval = baseline(sample)[0]
        dp_eval = candidate(sample)[0]
    hook.remove()
    if adapter_calls:
        raise AssertionError("RoadSnake executed in deployment/eval path")
    torch.testing.assert_close(dp_eval, b0_eval, atol=0, rtol=0)

    # Head-only probe: DP running buffers must equal exactly one native Detect forward.
    candidate.train()
    head = candidate.model[-1]
    reference_head = copy.deepcopy(head).to(device).train()
    features = [
        torch.randn(args.batch, channels, args.imgsz // stride, args.imgsz // stride, device=device)
        for channels, stride in zip((64, 128, 256), (8, 16, 32))
    ]
    reference_predictions = Detect.forward(reference_head, list(features))
    dual_predictions = head(list(features))
    if set(dual_predictions) != {"native", "snake"}:
        raise AssertionError(dual_predictions.keys())
    assert_bn_state_equal(shared_bn_state(head), shared_bn_state(reference_head))
    for route in ("one2many", "one2one"):
        for field in ("boxes", "scores"):
            torch.testing.assert_close(
                dual_predictions["native"][route][field], reference_predictions[route][field], atol=0, rtol=0
            )
            torch.testing.assert_close(
                dual_predictions["snake"][route][field], reference_predictions[route][field], atol=1e-6, rtol=1e-6
            )

    # Real detector loss must preserve the B0 scale at step zero and activate gamma.
    batch = make_real_batch(args.data.resolve(), args.imgsz, args.batch)
    if not len(batch["cls"]):
        raise AssertionError("selected real batch has no objects")
    batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
    batch["img"] = batch["img"].float().div(255.0)
    baseline.train().zero_grad(set_to_none=True)
    candidate.train().zero_grad(set_to_none=True)
    b0_loss, b0_items = baseline.loss(batch)
    dp_loss, dp_items = candidate.loss(batch)
    torch.testing.assert_close(dp_loss, b0_loss, atol=2e-4, rtol=2e-5)
    torch.testing.assert_close(dp_items, b0_items, atol=2e-5, rtol=2e-5)
    dp_loss.sum().backward()
    gamma_grad = candidate.model[-1].road_snake.gamma.grad
    if gamma_grad is None or not torch.isfinite(gamma_grad) or gamma_grad.abs() == 0:
        raise AssertionError("DP loss did not activate RoadSnake gamma")
    if any(parameter.grad is not None and not torch.isfinite(parameter.grad).all() for parameter in candidate.parameters()):
        raise AssertionError("non-finite DP gradient")

    # Once gamma leaves zero, all curved/local branches must receive finite gradients under AMP.
    branch_probe = copy.deepcopy(candidate).to(device).train()
    branch_probe.criterion = None
    with torch.no_grad():
        branch_probe.model[-1].road_snake.gamma.fill_(0.01)
    branch_probe.zero_grad(set_to_none=True)
    amp_enabled = device.type == "cuda"
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
        amp_loss, _ = branch_probe.loss(batch)
    if not torch.isfinite(amp_loss).all():
        raise AssertionError("non-finite AMP loss")
    amp_loss.sum().backward()
    adapter = branch_probe.model[-1].road_snake
    branch_parameters = {
        "offset": adapter.offset.weight,
        "horizontal_weight": adapter.horizontal_weight,
        "vertical_weight": adapter.vertical_weight,
        "local": adapter.local.conv.weight,
        "fuse": adapter.fuse.conv.weight,
    }
    branch_gradients = {}
    for name, parameter in branch_parameters.items():
        gradient = parameter.grad
        if gradient is None or not torch.isfinite(gradient).all() or gradient.abs().sum() == 0:
            raise AssertionError(f"missing/non-finite DP branch gradient: {name}")
        branch_gradients[name] = float(gradient.detach().abs().sum().cpu())

    onnx_path = None
    if args.onnx:
        import onnx

        args.onnx.parent.mkdir(parents=True, exist_ok=True)
        export_model = copy.deepcopy(candidate).eval()
        export_model.model[-1].export = True
        torch.onnx.export(
            export_model,
            sample,
            args.onnx,
            opset_version=17,
            input_names=["images"],
            output_names=["output"],
        )
        graph = onnx.load(args.onnx)
        onnx.checker.check_model(graph)
        forbidden = [node.op_type for node in graph.graph.node if "GridSample" in node.op_type]
        if forbidden:
            raise AssertionError(f"deployment ONNX retained RoadSnake sampling: {forbidden}")
        onnx_path = str(args.onnx)

    report = {
        "model": str(CANDIDATE),
        "data": str(args.data.resolve()),
        "transfer": transfer,
        "stride": candidate.stride.tolist(),
        "shared_detect_bn_modules": len(tuple(candidate.model[-1]._shared_detect_batch_norms())),
        "eval_adapter_calls": adapter_calls,
        "eval_max_abs_error_vs_b0": float((dp_eval - b0_eval).abs().max().cpu()),
        "b0_loss": float(b0_loss.detach().sum().cpu()),
        "dp_loss": float(dp_loss.detach().sum().cpu()),
        "loss_abs_error": float((dp_loss.detach() - b0_loss.detach()).abs().max().cpu()),
        "gamma_gradient_step0": float(gamma_grad.detach().cpu()),
        "amp_enabled": amp_enabled,
        "amp_loss": float(amp_loss.detach().sum().cpu()),
        "branch_gradient_l1_after_gate_probe": branch_gradients,
        "onnx": onnx_path,
        "criterion": type(candidate.criterion).__name__,
        "ok": True,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
