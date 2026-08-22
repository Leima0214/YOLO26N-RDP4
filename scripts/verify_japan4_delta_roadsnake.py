"""Static, real-batch, AMP and deployment audit for DeltaRoadSnake-P4."""

from __future__ import annotations

import argparse
import copy
import json
import random
import statistics
import sys
import time
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
from ultralytics.nn.roadsnake import DeltaRoadSnakeAdapter, DeltaRoadSnakeDetect  # noqa: E402
from ultralytics.nn.tasks import DetectionModel  # noqa: E402
from ultralytics.utils.torch_utils import get_flops, get_flops_with_torch_profiler, init_seeds  # noqa: E402

BASELINE = ROOT / "ultralytics/cfg/models/26/yolo26.yaml"
CANDIDATE = ROOT / "ultralytics/cfg/models/26/yolo26n-japan4-delta-roadsnake.yaml"


def prediction(output):
    return output[0] if isinstance(output, tuple) else output


def real_batch(data_yaml: Path, imgsz: int, batch_size: int) -> dict:
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
    return next(iter(build_dataloader(dataset, batch_size, 0, shuffle=False, rank=-1, pin_memory=False)))


def build_models(weights: Path, device: torch.device):
    init_seeds(42, deterministic=True)
    checkpoint = YOLO(str(weights), task="detect", verbose=False).model.float()
    baseline = DetectionModel(str(BASELINE), nc=4, ch=3, verbose=False).float()
    baseline.load(checkpoint, verbose=False)
    candidate = DetectionModel(str(CANDIDATE), nc=4, ch=3, verbose=False).float()
    source, target = baseline.state_dict(), candidate.state_dict()
    missing = [key for key, value in source.items() if key not in target or target[key].shape != value.shape]
    if missing:
        raise AssertionError(f"Cannot inherit B0 tensors: {missing[:8]}")
    candidate.load(baseline, verbose=False)
    changed = [key for key, value in source.items() if not torch.equal(value, candidate.state_dict()[key])]
    if changed:
        raise AssertionError(f"Inherited B0 tensors changed: {changed[:8]}")

    trainer = object.__new__(DetectionTrainer)
    trainer.data = {"nc": 4, "channels": 3}
    rebuilt = DetectionTrainer.get_model(trainer, cfg=str(CANDIDATE), weights=candidate, verbose=False).float()
    rebuilt_changed = [
        key for key, value in candidate.state_dict().items() if not torch.equal(value, rebuilt.state_dict()[key])
    ]
    if rebuilt_changed:
        raise AssertionError(f"Trainer reconstruction changed tensors: {rebuilt_changed[:8]}")
    for model in (baseline, rebuilt):
        model.args = get_cfg(DEFAULT_CFG, {"box": 7.5, "cls": 0.5, "dfl": 1.5, "epochs": 30})
        model.criterion = None
    return baseline.to(device), rebuilt.to(device), {
        "b0_state_items": len(source),
        "candidate_state_items": len(target),
        "shared_missing_or_mismatched": len(missing),
        "shared_changed_after_load": len(changed),
        "trainer_rebuild_changed": len(rebuilt_changed),
        "new_state_items": sorted(set(target) - set(source)),
    }


def paired_latency(reference: torch.nn.Module, candidate: torch.nn.Module, sample: torch.Tensor):
    if sample.device.type != "cuda":
        return None
    values = ([], [])
    with torch.inference_mode():
        for _ in range(20):
            reference(sample)
            candidate(sample)
        for index in range(100):
            order = ((reference, values[0]), (candidate, values[1]))
            if index % 2:
                order = reversed(order)
            for model, timings in order:
                torch.cuda.synchronize(sample.device)
                started = time.perf_counter_ns()
                model(sample)
                torch.cuda.synchronize(sample.device)
                timings.append((time.perf_counter_ns() - started) / 1e6)
    b0_ms, delta_ms = map(statistics.median, values)
    return {"b0_ms": b0_ms, "delta_ms": delta_ms, "relative_delta": delta_ms / b0_ms - 1.0}


def finite_nonzero_gradient(parameter: torch.nn.Parameter, name: str) -> float:
    gradient = parameter.grad
    if gradient is None or not torch.isfinite(gradient).all() or gradient.abs().sum() == 0:
        raise AssertionError(f"Missing/non-finite/zero gradient: {name}")
    return float(gradient.detach().abs().sum().cpu())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=ROOT / "yolo26n.pt")
    parser.add_argument("--data", type=Path, default=ROOT / "configs/japan4_clean_v3_remote.yaml")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--onnx", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    device = torch.device(args.device)
    baseline, candidate, transfer = build_models(args.weights.resolve(), device)
    if not isinstance(baseline.model[-1], Detect) or not isinstance(candidate.model[-1], DeltaRoadSnakeDetect):
        raise AssertionError("Unexpected Detect head types")
    adapter: DeltaRoadSnakeAdapter = candidate.model[-1].road_snake
    if hasattr(adapter, "gamma"):
        raise AssertionError("DeltaRoadSnake must not retain a zero residual gamma")
    if torch.count_nonzero(adapter.offset.weight) or torch.count_nonzero(adapter.offset.bias):
        raise AssertionError("Offset predictor must initialize at zero")
    if candidate.stride.tolist() != [8.0, 16.0, 32.0] or candidate.stride.tolist() != baseline.stride.tolist():
        raise AssertionError("Detect strides changed")

    sample = torch.randn(1, 3, args.imgsz, args.imgsz, device=device)
    baseline.eval()
    candidate.eval()
    captured = {}

    def capture_inputs(_module, inputs):
        captured["shapes"] = [list(value.shape) for value in inputs[0]]

    hook = candidate.model[-1].register_forward_pre_hook(capture_inputs)
    with torch.inference_mode():
        b0_output = prediction(baseline(sample))
        delta_output = prediction(candidate(sample))
    hook.remove()
    identity_error = float((delta_output - b0_output).abs().max().cpu())
    torch.testing.assert_close(delta_output, b0_output, atol=0, rtol=0)
    expected_shapes = [[1, 64, 80, 80], [1, 128, 40, 40], [1, 256, 20, 20]]
    if captured.get("shapes") != expected_shapes:
        raise AssertionError(f"Unexpected Detect inputs: {captured}")

    fused_b0 = copy.deepcopy(baseline).eval().fuse(verbose=False)
    fused_delta = copy.deepcopy(candidate).eval().fuse(verbose=False)
    with torch.inference_mode():
        fused_error = float((prediction(fused_delta(sample)) - prediction(fused_b0(sample))).abs().max().cpu())
    if fused_error != 0.0:
        raise AssertionError(f"Fused step-zero identity failed: {fused_error}")

    # Confirm the sampler keeps images isolated after a deterministic non-zero offset perturbation.
    feature = torch.randn(2, adapter.channels, args.imgsz // 16, args.imgsz // 16, device=device)
    saved_weight = adapter.offset.weight.detach().clone()
    with torch.no_grad():
        adapter.offset.weight.normal_(mean=0.0, std=1e-3)
        isolated = adapter(feature[:1]).clone()
        batched = adapter(feature)[:1].clone()
        adapter.offset.weight.copy_(saved_weight)
    batch_error = float((isolated - batched).abs().max().cpu())
    torch.testing.assert_close(isolated, batched, atol=2e-6, rtol=2e-6)

    batch = real_batch(args.data.resolve(), args.imgsz, args.batch)
    batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
    batch["img"] = batch["img"].float().div(255.0)
    if not len(batch["cls"]):
        raise AssertionError("Real batch contains no labels")

    candidate.train().zero_grad(set_to_none=True)
    loss, components = candidate.loss(batch)
    if not torch.isfinite(loss).all() or not torch.isfinite(components).all():
        raise AssertionError("Non-finite detection loss")
    loss.sum().backward()
    step0_offset_gradients = {
        "weight_l1": finite_nonzero_gradient(adapter.offset.weight, "offset.weight@step0"),
        "bias_l1": finite_nonzero_gradient(adapter.offset.bias, "offset.bias@step0"),
    }

    # Apply one tiny train-only offset step, then prove every shared branch can receive gradients.
    with torch.no_grad():
        scale = adapter.offset.weight.grad.detach().abs().mean().clamp_min(1e-12)
        adapter.offset.weight.add_(adapter.offset.weight.grad, alpha=-1e-5 / float(scale))
        adapter.offset.bias.add_(adapter.offset.bias.grad, alpha=-1e-5 / float(adapter.offset.bias.grad.abs().mean().clamp_min(1e-12)))
    candidate.zero_grad(set_to_none=True)
    probe_loss, _ = candidate.loss(batch)
    probe_loss.sum().backward()
    post_offset_gradients = {
        "horizontal_weight": finite_nonzero_gradient(adapter.horizontal_weight, "horizontal_weight"),
        "vertical_weight": finite_nonzero_gradient(adapter.vertical_weight, "vertical_weight"),
        "local": finite_nonzero_gradient(adapter.local.conv.weight, "local"),
        "fuse": finite_nonzero_gradient(adapter.fuse.conv.weight, "fuse"),
    }

    amp_model = copy.deepcopy(candidate).train()
    amp_model.zero_grad(set_to_none=True)
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
        amp_loss, amp_components = amp_model.loss(batch)
    if not torch.isfinite(amp_loss).all() or not torch.isfinite(amp_components).all():
        raise AssertionError("Non-finite AMP loss")
    amp_loss.sum().backward()
    amp_gradients_finite = all(
        parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in amp_model.parameters()
    )
    if not amp_gradients_finite:
        raise AssertionError("Non-finite AMP gradients")

    b0_params = sum(parameter.numel() for parameter in baseline.parameters())
    delta_params = sum(parameter.numel() for parameter in candidate.parameters())
    b0_flops = get_flops(baseline, args.imgsz) or get_flops_with_torch_profiler(baseline, args.imgsz)
    delta_flops = get_flops(candidate, args.imgsz) or get_flops_with_torch_profiler(candidate, args.imgsz)
    latency = paired_latency(fused_b0, fused_delta, sample)

    onnx_path = None
    if args.onnx:
        import onnx

        args.onnx.parent.mkdir(parents=True, exist_ok=True)
        export_model = copy.deepcopy(fused_delta).eval()
        export_model.model[-1].export = True
        torch.onnx.export(export_model, sample, args.onnx, opset_version=17, input_names=["images"], output_names=["output"])
        onnx.checker.check_model(onnx.load(args.onnx))
        onnx_path = str(args.onnx.resolve())

    report = {
        "model": str(CANDIDATE),
        "data": str(args.data.resolve()),
        "images": list(batch["im_file"]),
        "objects": int(len(batch["cls"])),
        "transfer": transfer,
        "detect_input_shapes": captured["shapes"],
        "initial_output_max_abs_error": identity_error,
        "fused_initial_output_max_abs_error": fused_error,
        "batch_isolation_max_abs_error": batch_error,
        "loss": float(loss.detach().sum().cpu()),
        "loss_components": [float(value) for value in components.detach().flatten().cpu()],
        "step0_offset_gradient_l1": step0_offset_gradients,
        "post_offset_branch_gradient_l1": post_offset_gradients,
        "amp_loss": float(amp_loss.detach().sum().cpu()),
        "amp_gradients_finite": amp_gradients_finite,
        "parameters": {"b0": b0_params, "delta": delta_params, "relative_delta": delta_params / b0_params - 1.0},
        "gflops": {"b0": b0_flops, "delta": delta_flops, "relative_delta": delta_flops / b0_flops - 1.0},
        "paired_fused_latency_ms": latency,
        "onnx": onnx_path,
        "ok": True,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
