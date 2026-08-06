"""Unified static verification for Japan4 S1, G1, and GS1."""

from __future__ import annotations

import argparse
import copy
import json
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ultralytics import YOLO  # noqa: E402
from ultralytics.cfg import get_cfg  # noqa: E402
from ultralytics.models.yolo.detect.train import DetectionTrainer  # noqa: E402
from ultralytics.nn.modules.head import StripAwareResidual  # noqa: E402
from ultralytics.nn.tasks import DetectionModel  # noqa: E402
from ultralytics.utils.torch_utils import get_flops, get_flops_with_torch_profiler  # noqa: E402

CANDIDATES = {
    "s1": ROOT / "ultralytics/cfg/models/26/yolo26n-japan4-s1-strip-regression.yaml",
    "g1": ROOT / "ultralytics/cfg/models/26/yolo26n-japan4-g1-region-guidance.yaml",
    "gs1": ROOT / "ultralytics/cfg/models/26/yolo26n-japan4-gs1-region-strip.yaml",
}
BASELINE = ROOT / "ultralytics/cfg/models/26/yolo26.yaml"


def prediction(output):
    """Return the deployed detection tensor."""
    return output[0] if isinstance(output, tuple) else output


def latency_ms(model: torch.nn.Module, sample: torch.Tensor) -> float | None:
    """Return synchronized median CUDA batch-1 latency."""
    if sample.device.type != "cuda":
        return None
    with torch.inference_mode():
        for _ in range(20):
            model(sample)
        torch.cuda.synchronize(sample.device)
        values = []
        for _ in range(100):
            start = time.perf_counter()
            model(sample)
            torch.cuda.synchronize(sample.device)
            values.append((time.perf_counter() - start) * 1000)
    return statistics.median(values)


def build_baseline(weights: Path, device: torch.device) -> DetectionModel:
    """Build the actual nc=4 B0 and transfer official pretrained tensors."""
    checkpoint = YOLO(str(weights), task="detect", verbose=False).model.float()
    model = DetectionModel(str(BASELINE), nc=4, ch=3, verbose=False).float()
    model.load(checkpoint, verbose=False)
    model.args = get_cfg()
    model.args.epochs = 30
    return model.to(device)


def build_candidate(name: str, baseline: DetectionModel, device: torch.device) -> tuple[DetectionModel, dict]:
    """Verify shared B0 transfer and Trainer reconstruction."""
    model = DetectionModel(str(CANDIDATES[name]), nc=4, ch=3, verbose=False).float()
    source, target = baseline.state_dict(), model.state_dict()
    missing = [key for key, value in source.items() if key not in target or target[key].shape != value.shape]
    if missing:
        raise AssertionError(f"{name} cannot inherit B0 tensors: {missing[:10]}")
    model.load(baseline, verbose=False)
    changed = [key for key, value in source.items() if not torch.equal(value.cpu(), model.state_dict()[key].cpu())]
    if changed:
        raise AssertionError(f"{name} changed inherited tensors: {changed[:10]}")

    trainer = object.__new__(DetectionTrainer)
    trainer.data = {"nc": 4, "channels": 3}
    rebuilt = DetectionTrainer.get_model(trainer, cfg=str(CANDIDATES[name]), weights=model, verbose=False).float()
    rebuilt_state = rebuilt.state_dict()
    rebuild_changed = [key for key, value in model.state_dict().items() if not torch.equal(value, rebuilt_state[key])]
    if rebuild_changed:
        raise AssertionError(f"Trainer reconstruction changed {name}: {rebuild_changed[:10]}")
    rebuilt.args = get_cfg()
    rebuilt.args.epochs = 30
    audit = {
        "b0_state_items": len(source),
        "candidate_state_items": len(target),
        "shared_missing_or_mismatched": len(missing),
        "shared_changed_after_load": len(changed),
        "trainer_rebuild_items": len(rebuilt_state),
        "trainer_rebuild_changed": len(rebuild_changed),
    }
    return rebuilt.to(device), audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", action="append", choices=tuple(CANDIDATES), dest="candidates")
    parser.add_argument("--weights", type=Path, default=ROOT / "yolo26n.pt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--imgsz", type=int, default=128)
    parser.add_argument("--onnx-dir", type=Path)
    parser.add_argument("--report", type=Path, default=ROOT / "reports/japan4_candidate_static_verification.json")
    args = parser.parse_args()
    names = args.candidates or list(CANDIDATES)
    if not args.weights.is_file() or args.imgsz < 64 or args.imgsz % 32:
        raise ValueError("Valid official weights and imgsz divisible by 32 are required")

    torch.manual_seed(42)
    device = torch.device(args.device)
    baseline = build_baseline(args.weights, device).eval()
    sample = torch.randn(1, 3, args.imgsz, args.imgsz, device=device)
    with torch.inference_mode():
        baseline_output = prediction(baseline(sample))
    fused_baseline = copy.deepcopy(baseline).eval().fuse(verbose=False)
    baseline_params = sum(parameter.numel() for parameter in baseline.parameters())
    baseline_flops = get_flops(baseline, args.imgsz) or get_flops_with_torch_profiler(baseline, args.imgsz)
    baseline_latency = latency_ms(fused_baseline, sample)

    results = {}
    for name in names:
        model, transfer = build_candidate(name, baseline, device)
        model.eval()
        with torch.inference_mode():
            initial_output = prediction(model(sample))
        initial_error = float((initial_output - baseline_output).abs().max().detach().cpu())
        assert initial_error == 0.0

        fused = copy.deepcopy(model).eval().fuse(verbose=False)
        with torch.inference_mode():
            fused_output = prediction(fused(sample))
            fused_baseline_output = prediction(fused_baseline(sample))
        torch.testing.assert_close(fused_output, fused_baseline_output, atol=0, rtol=0)
        if name in {"g1", "gs1"}:
            assert fused.model[-1].region_heads is None

        model.train()
        training_sample = sample.repeat(2, 1, 1, 1)
        batch = {
            "batch_idx": torch.tensor([0.0, 0.0, 1.0], device=device),
            "cls": torch.tensor([[0.0], [1.0], [3.0]], device=device),
            "bboxes": torch.tensor(
                [[0.3, 0.4, 0.25, 0.06], [0.7, 0.6, 0.08, 0.3], [0.5, 0.5, 0.2, 0.2]], device=device
            ),
        }
        criterion = model.init_criterion()
        loss, loss_items = criterion(model(training_sample), batch)
        assert torch.isfinite(loss).all() and torch.isfinite(loss_items).all()
        loss.sum().backward()
        gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
        assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)

        head = model.model[-1]
        region_gradient = None
        if name in {"g1", "gs1"}:
            region_gradient = sum(
                float(parameter.grad.detach().abs().sum().cpu())
                for parameter in head.region_heads.parameters()
                if parameter.grad is not None
            )
            assert region_gradient > 0
        active_strip_gamma = None
        if name in {"s1", "gs1"}:
            adapters = [
                *(module for module in head.strip_cv2 if isinstance(module, StripAwareResidual)),
                *(module for module in head.one2one_strip_cv2 if isinstance(module, StripAwareResidual)),
            ]
            active_strip_gamma = sum(
                module.gamma.grad is not None and module.gamma.grad.detach().abs().item() > 0 for module in adapters
            )
            assert active_strip_gamma > 0

        params = sum(parameter.numel() for parameter in model.parameters())
        flops = get_flops(model, args.imgsz) or get_flops_with_torch_profiler(model, args.imgsz)
        candidate_latency = latency_ms(fused, sample)
        latency_delta = None if baseline_latency is None else candidate_latency / baseline_latency - 1

        onnx_path = None
        if args.onnx_dir:
            import onnx

            args.onnx_dir.mkdir(parents=True, exist_ok=True)
            onnx_path = args.onnx_dir / f"yolo26n-japan4-{name}.onnx"
            export_model = copy.deepcopy(fused).eval()
            export_model.model[-1].export = True
            torch.onnx.export(
                export_model, sample, onnx_path, opset_version=17, input_names=["images"], output_names=["output"]
            )
            onnx.checker.check_model(onnx.load(onnx_path))

        results[name] = {
            "model_yaml": str(CANDIDATES[name]),
            "head": type(head).__name__,
            "transfer": transfer,
            "initial_output_max_abs_error": initial_error,
            "loss_items": loss_items.detach().cpu().tolist(),
            "finite_gradient_tensors": len(gradients),
            "region_head_gradient_l1": region_gradient,
            "active_strip_gamma_count": active_strip_gamma,
            "parameters": {"b0": baseline_params, "candidate": params, "delta": params / baseline_params - 1},
            "gflops": {"b0": baseline_flops, "candidate": flops, "delta": flops / baseline_flops - 1},
            "fused_latency_ms": {
                "b0": baseline_latency,
                "candidate": candidate_latency,
                "delta": latency_delta,
            },
            "onnx": str(onnx_path) if onnx_path else None,
        }

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
