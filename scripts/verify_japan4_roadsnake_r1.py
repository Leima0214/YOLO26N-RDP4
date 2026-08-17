"""Static and real-batch verification for the pretrained-preserving RoadSnake-R1 candidate."""

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
sys.path.insert(0, str(ROOT))

from ultralytics import YOLO  # noqa: E402
from ultralytics.cfg import DEFAULT_CFG, get_cfg  # noqa: E402
from ultralytics.data.build import build_dataloader, build_yolo_dataset  # noqa: E402
from ultralytics.data.utils import check_det_dataset  # noqa: E402
from ultralytics.models.yolo.detect.train import DetectionTrainer  # noqa: E402
from ultralytics.nn.modules.head import Detect  # noqa: E402
from ultralytics.nn.roadsnake import RoadSnakeAdapter, RoadSnakeDetect  # noqa: E402
from ultralytics.nn.tasks import DetectionModel  # noqa: E402
from ultralytics.utils.torch_utils import get_flops, get_flops_with_torch_profiler  # noqa: E402

BASELINE = ROOT / "ultralytics/cfg/models/26/yolo26.yaml"
CANDIDATE = ROOT / "ultralytics/cfg/models/26/yolo26n-japan4-roadsnake-r1.yaml"


def prediction(output):
    """Return the deployed prediction tensor."""
    return output[0] if isinstance(output, tuple) else output


def paired_latency(reference: torch.nn.Module, candidate: torch.nn.Module, sample: torch.Tensor) -> dict | None:
    """Measure alternating synchronized CUDA latency to reduce clock drift bias."""
    if sample.device.type != "cuda":
        return None
    with torch.inference_mode():
        for _ in range(30):
            reference(sample)
            candidate(sample)
        torch.cuda.synchronize(sample.device)
        values = ([], [])
        for index in range(200):
            order = ((reference, values[0]), (candidate, values[1]))
            if index % 2:
                order = reversed(order)
            for model, timings in order:
                torch.cuda.synchronize(sample.device)
                start = time.perf_counter_ns()
                model(sample)
                torch.cuda.synchronize(sample.device)
                timings.append((time.perf_counter_ns() - start) / 1e6)
    b0_ms, r1_ms = map(statistics.median, values)
    return {"b0_ms": b0_ms, "r1_ms": r1_ms, "delta": r1_ms / b0_ms - 1}


def make_real_batch(data_yaml: Path, imgsz: int, batch_size: int) -> tuple[dict, dict]:
    """Load a deterministic labeled train batch with stochastic augmentation disabled."""
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
    return data, next(iter(loader))


def build_models(weights: Path, device: torch.device) -> tuple[DetectionModel, DetectionModel, dict]:
    """Transfer official YOLO26n tensors and prove Trainer reconstruction preserves every tensor."""
    checkpoint = YOLO(str(weights), task="detect", verbose=False).model.float()
    baseline = DetectionModel(str(BASELINE), nc=4, ch=3, verbose=False).float()
    baseline.load(checkpoint, verbose=False)
    candidate = DetectionModel(str(CANDIDATE), nc=4, ch=3, verbose=False).float()

    source, target = baseline.state_dict(), candidate.state_dict()
    missing = [key for key, value in source.items() if key not in target or target[key].shape != value.shape]
    if missing:
        raise AssertionError(f"RoadSnake-R1 cannot inherit B0 tensors: {missing[:10]}")
    candidate.load(baseline, verbose=False)
    changed = [key for key, value in source.items() if not torch.equal(value, candidate.state_dict()[key])]
    if changed:
        raise AssertionError(f"RoadSnake-R1 changed inherited B0 tensors: {changed[:10]}")

    trainer = object.__new__(DetectionTrainer)
    trainer.data = {"nc": 4, "channels": 3}
    rebuilt = DetectionTrainer.get_model(trainer, cfg=str(CANDIDATE), weights=candidate, verbose=False).float()
    rebuilt_state = rebuilt.state_dict()
    rebuild_changed = [key for key, value in candidate.state_dict().items() if not torch.equal(value, rebuilt_state[key])]
    if rebuild_changed:
        raise AssertionError(f"Trainer reconstruction changed RoadSnake-R1: {rebuild_changed[:10]}")
    audit = {
        "b0_state_items": len(source),
        "candidate_state_items": len(target),
        "shared_missing_or_mismatched": len(missing),
        "shared_changed_after_load": len(changed),
        "trainer_rebuild_changed": len(rebuild_changed),
        "new_state_items": sorted(set(target) - set(source)),
    }
    for model in (baseline, rebuilt):
        model.args = get_cfg(DEFAULT_CFG, {"box": 7.5, "cls": 0.5, "dfl": 1.5, "epochs": 30})
        model.criterion = None
    return baseline.to(device), rebuilt.to(device), audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=ROOT / "yolo26n.pt")
    parser.add_argument("--data", type=Path, default=ROOT / "configs/japan4_clean_v3_local.yaml")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--imgsz", type=int, default=128)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--onnx", type=Path)
    parser.add_argument("--report", type=Path, default=ROOT / "reports/japan4_roadsnake_r1_static.json")
    args = parser.parse_args()
    if args.imgsz < 64 or args.imgsz % 32:
        raise ValueError("imgsz must be >=64 and divisible by 32")

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    device = torch.device(args.device)
    baseline, candidate, transfer = build_models(args.weights.resolve(), device)
    if not isinstance(baseline.model[-1], Detect) or not isinstance(candidate.model[-1], RoadSnakeDetect):
        raise AssertionError("unexpected baseline/candidate head types")
    adapter = candidate.model[-1].road_snake
    if adapter.gamma.item() != 0.0:
        raise AssertionError("RoadSnake-R1 must initialize as an exact identity")
    if candidate.stride.tolist() != baseline.stride.tolist() or candidate.stride.tolist() != [8.0, 16.0, 32.0]:
        raise AssertionError("Detect strides changed")

    sample = torch.randn(1, 3, args.imgsz, args.imgsz, device=device)
    baseline.eval()
    candidate.eval()
    with torch.inference_mode():
        baseline_output = prediction(baseline(sample))
        candidate_output = prediction(candidate(sample))
    torch.testing.assert_close(candidate_output, baseline_output, atol=0, rtol=0)

    # Freeze deployment copies before any training-mode probe updates BatchNorm statistics.
    fused_baseline = copy.deepcopy(baseline).eval().fuse(verbose=False)
    fused_candidate = copy.deepcopy(candidate).eval().fuse(verbose=False)
    with torch.inference_mode():
        fused_baseline_output = prediction(fused_baseline(sample))
        fused_candidate_output = prediction(fused_candidate(sample))
    torch.testing.assert_close(fused_candidate_output, fused_baseline_output, atol=0, rtol=0)

    # The rewritten sampler must not mix samples across the batch dimension.
    feature = torch.randn(2, adapter.channels, args.imgsz // 16, args.imgsz // 16, device=device)
    adapter.eval()
    with torch.inference_mode():
        adapter.gamma.fill_(0.01)
        isolated = adapter(feature[:1]).clone()
        batched = adapter(feature)[:1].clone()
        adapter.gamma.zero_()
    batch_isolation_error = float((isolated - batched).abs().max().cpu())
    torch.testing.assert_close(isolated, batched, atol=1e-6, rtol=1e-6)

    data, batch = make_real_batch(args.data.resolve(), args.imgsz, args.batch)
    if not len(batch["cls"]):
        raise AssertionError("the selected real batch has no labeled objects")
    batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
    batch["img"] = batch["img"].float().div(255.0)
    candidate.train().zero_grad(set_to_none=True)
    loss, components = candidate.loss(batch)
    if not torch.isfinite(loss).all() or not torch.isfinite(components).all():
        raise AssertionError("non-finite detection loss")
    loss.sum().backward()
    gamma_grad_step0 = adapter.gamma.grad
    if gamma_grad_step0 is None or not torch.isfinite(gamma_grad_step0) or gamma_grad_step0.abs() == 0:
        raise AssertionError("detection loss did not activate the RoadSnake residual gate")

    # Move only gamma for a probe step; then every curved branch must receive a finite non-zero gradient.
    with torch.no_grad():
        adapter.gamma.copy_(-0.01 * gamma_grad_step0.sign())
    candidate.zero_grad(set_to_none=True)
    probe_loss = adapter(feature).square().mean()
    probe_loss.backward()
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
            raise AssertionError(f"missing/non-finite RoadSnake gradient: {name}")
        branch_gradients[name] = float(gradient.detach().abs().sum().cpu())
    with torch.no_grad():
        adapter.gamma.zero_()

    baseline_params = sum(parameter.numel() for parameter in baseline.parameters())
    candidate_params = sum(parameter.numel() for parameter in candidate.parameters())
    baseline_flops = get_flops(baseline, args.imgsz) or get_flops_with_torch_profiler(baseline, args.imgsz)
    candidate_flops = get_flops(candidate, args.imgsz) or get_flops_with_torch_profiler(candidate, args.imgsz)

    latency = paired_latency(fused_baseline, fused_candidate, sample)

    onnx_path = None
    if args.onnx:
        import onnx

        args.onnx.parent.mkdir(parents=True, exist_ok=True)
        export_model = copy.deepcopy(fused_candidate).eval()
        export_model.model[-1].export = True
        torch.onnx.export(
            export_model,
            sample,
            args.onnx,
            opset_version=17,
            input_names=["images"],
            output_names=["output"],
        )
        onnx.checker.check_model(onnx.load(args.onnx))
        onnx_path = str(args.onnx)

    report = {
        "model": str(CANDIDATE),
        "data": str(args.data.resolve()),
        "images": list(batch["im_file"]),
        "objects": int(len(batch["cls"])),
        "classes": sorted({int(value) for value in batch["cls"].flatten()}),
        "transfer": transfer,
        "initial_output_max_abs_error": float((candidate_output - baseline_output).abs().max().cpu()),
        "batch_isolation_max_abs_error": batch_isolation_error,
        "loss": float(loss.detach().sum().cpu()),
        "loss_components": [float(value) for value in components.detach().flatten().cpu()],
        "gamma_gradient_step0": float(gamma_grad_step0.detach().cpu()),
        "branch_gradient_l1_after_gate_probe": branch_gradients,
        "parameters": {
            "b0": baseline_params,
            "r1": candidate_params,
            "delta": candidate_params / baseline_params - 1,
        },
        "gflops": {
            "b0": baseline_flops,
            "r1": candidate_flops,
            "delta": candidate_flops / baseline_flops - 1,
        },
        "paired_fused_latency_ms": latency,
        "onnx": onnx_path,
        "ok": True,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
