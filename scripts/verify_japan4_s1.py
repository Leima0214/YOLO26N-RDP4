"""Verify S1 transfer, identity initialization, gradients, fuse, cost, and optional ONNX export."""

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
from ultralytics.nn.modules.head import Detect, StripAwareResidual, StripDetect  # noqa: E402
from ultralytics.nn.tasks import DetectionModel  # noqa: E402
from ultralytics.utils.torch_utils import get_flops, get_flops_with_torch_profiler  # noqa: E402


def prediction(output):
    """Return the deployed prediction tensor from an eval-mode detector output."""
    return output[0] if isinstance(output, tuple) else output


def latency_ms(model: torch.nn.Module, sample: torch.Tensor) -> float | None:
    """Measure synchronized batch-1 median latency on CUDA."""
    if sample.device.type != "cuda":
        return None
    with torch.inference_mode():
        for _ in range(20):
            model(sample)
        torch.cuda.synchronize(sample.device)
        timings = []
        for _ in range(100):
            start = time.perf_counter()
            model(sample)
            torch.cuda.synchronize(sample.device)
            timings.append((time.perf_counter() - start) * 1000)
    return statistics.median(timings)


def load_models(weights: Path, baseline_yaml: Path, s1_yaml: Path, device: torch.device):
    """Build the actual nc=4 training topology and transfer official B0 weights."""
    checkpoint = YOLO(str(weights), task="detect", verbose=False).model.float()
    baseline = DetectionModel(str(baseline_yaml), nc=4, ch=3, verbose=False).float()
    baseline.load(checkpoint, verbose=False)
    candidate = DetectionModel(str(s1_yaml), nc=4, ch=3, verbose=False).float()

    source_state, target_state = baseline.state_dict(), candidate.state_dict()
    missing = [name for name, value in source_state.items() if name not in target_state or target_state[name].shape != value.shape]
    if missing:
        raise AssertionError(f"S1 cannot inherit B0 tensors: {missing[:10]}")
    candidate.load(baseline, verbose=False)
    changed = [name for name, value in source_state.items() if not torch.equal(value, candidate.state_dict()[name])]
    if changed:
        raise AssertionError(f"S1 changed inherited B0 tensors: {changed[:10]}")

    trainer = object.__new__(DetectionTrainer)
    trainer.data = {"nc": 4, "channels": 3}
    rebuilt = DetectionTrainer.get_model(trainer, cfg=str(s1_yaml), weights=candidate, verbose=False).float()
    rebuilt_state = rebuilt.state_dict()
    rebuild_changed = [name for name, value in candidate.state_dict().items() if not torch.equal(value, rebuilt_state[name])]
    if rebuild_changed:
        raise AssertionError(f"Trainer reconstruction changed S1 tensors: {rebuild_changed[:10]}")
    return baseline.to(device), rebuilt.to(device), len(source_state)


def strip_modules(head: StripDetect) -> list[StripAwareResidual]:
    """Return the four trainable P3/P4 O2M/O2O strip adapters."""
    return [
        *(module for module in head.strip_cv2 if isinstance(module, StripAwareResidual)),
        *(module for module in head.one2one_strip_cv2 if isinstance(module, StripAwareResidual)),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, default=ROOT / "yolo26n.pt")
    parser.add_argument("--baseline", type=Path, default=ROOT / "ultralytics/cfg/models/26/yolo26.yaml")
    parser.add_argument(
        "--model",
        type=Path,
        default=ROOT / "ultralytics/cfg/models/26/yolo26n-japan4-s1-strip-regression.yaml",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--imgsz", type=int, default=128)
    parser.add_argument("--onnx", type=Path)
    parser.add_argument("--report", type=Path, default=ROOT / "reports/japan4_s1_static_verification.json")
    args = parser.parse_args()
    assert args.weights.is_file() and args.baseline.is_file() and args.model.is_file()
    assert args.imgsz >= 64 and args.imgsz % 32 == 0

    torch.manual_seed(42)
    device = torch.device(args.device)
    baseline, candidate, inherited_items = load_models(args.weights, args.baseline, args.model, device)
    baseline.args = candidate.args = get_cfg()
    baseline.args.epochs = candidate.args.epochs = 30
    baseline_head, head = baseline.model[-1], candidate.model[-1]
    assert isinstance(baseline_head, Detect) and isinstance(head, StripDetect)
    assert head.reg_max == baseline_head.reg_max == 1
    assert candidate.stride.tolist() == baseline.stride.tolist() == [8.0, 16.0, 32.0]
    assert isinstance(head.strip_cv2[2], torch.nn.Identity)
    assert isinstance(head.one2one_strip_cv2[2], torch.nn.Identity)

    adapters = strip_modules(head)
    assert len(adapters) == 4
    assert [module.square.in_channels for module in adapters] == [64, 128, 64, 128]
    assert [module.horizontal.kernel_size for module in adapters] == [(1, 7), (1, 5), (1, 7), (1, 5)]
    for module in adapters:
        torch.testing.assert_close(module.branch_weights(), torch.full((3,), 1 / 3, device=device))
        assert module.gamma.item() == 0.0

    sample = torch.randn(1, 3, args.imgsz, args.imgsz, device=device)
    baseline.eval()
    candidate.eval()
    with torch.inference_mode():
        baseline_output = prediction(baseline(sample))
        candidate_output = prediction(candidate(sample))
    torch.testing.assert_close(candidate_output, baseline_output, atol=0, rtol=0)

    candidate.train()
    batch = {
        "batch_idx": torch.zeros(1, device=device),
        "cls": torch.zeros(1, 1, device=device),
        "bboxes": torch.tensor([[0.5, 0.5, 0.35, 0.2]], device=device),
    }
    criterion = candidate.init_criterion()
    losses, loss_items = criterion(candidate(sample), batch)
    assert torch.isfinite(losses).all() and torch.isfinite(loss_items).all()
    losses.sum().backward()
    detection_gamma_active = sum(
        bool(module.gamma.grad is not None and torch.isfinite(module.gamma.grad).item() and module.gamma.grad.abs().item() > 0)
        for module in adapters
    )
    assert detection_gamma_active > 0

    candidate.zero_grad(set_to_none=True)
    probe_inputs = [
        torch.randn(2, module.square.in_channels, 16, 16, device=device) for module in adapters
    ]
    probe_loss = sum(module(value).square().mean() for module, value in zip(adapters, probe_inputs))
    probe_loss.backward()
    gamma_gradients = [float(module.gamma.grad.detach().cpu()) for module in adapters]
    assert all(torch.isfinite(module.gamma.grad) and module.gamma.grad.abs() > 0 for module in adapters)

    with torch.no_grad():
        for module in adapters:
            module.gamma.add_(-0.01 * module.gamma.grad.sign())
    candidate.zero_grad(set_to_none=True)
    second_probe_loss = sum(module(value).square().mean() for module, value in zip(adapters, probe_inputs))
    second_probe_loss.backward()
    branch_parameters = [
        parameter
        for module in adapters
        for parameter in (module.square.weight, module.horizontal.weight, module.vertical.weight, module.gate_logits)
    ]
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0
        for parameter in branch_parameters
    )

    candidate.eval()
    with torch.inference_mode():
        unfused_output = prediction(candidate(sample))
    fused = copy.deepcopy(candidate).eval().fuse(verbose=False)
    with torch.inference_mode():
        fused_output = prediction(fused(sample))
    torch.testing.assert_close(fused_output, unfused_output, atol=1e-4, rtol=1e-4)

    baseline_params = sum(parameter.numel() for parameter in baseline.parameters())
    candidate_params = sum(parameter.numel() for parameter in candidate.parameters())
    parameter_delta = candidate_params / baseline_params - 1
    assert parameter_delta <= 0.05
    baseline_flops = get_flops(baseline, args.imgsz) or get_flops_with_torch_profiler(baseline, args.imgsz)
    candidate_flops = get_flops(candidate, args.imgsz) or get_flops_with_torch_profiler(candidate, args.imgsz)
    flops_delta = candidate_flops / baseline_flops - 1
    assert flops_delta <= 0.05

    baseline.eval()
    candidate.eval()
    baseline_latency = latency_ms(baseline, sample)
    candidate_latency = latency_ms(candidate, sample)
    latency_delta = None if baseline_latency is None else candidate_latency / baseline_latency - 1
    if latency_delta is not None:
        assert latency_delta <= 0.05

    if args.onnx:
        import onnx

        args.onnx.parent.mkdir(parents=True, exist_ok=True)
        export_model = copy.deepcopy(candidate).eval()
        export_model.model[-1].export = True
        torch.onnx.export(export_model, sample, args.onnx, opset_version=17, input_names=["images"], output_names=["output"])
        assert args.onnx.is_file() and args.onnx.stat().st_size > 0
        onnx.checker.check_model(onnx.load(args.onnx))

    report = {
        "model": str(args.model),
        "inherited_b0_state_items": inherited_items,
        "initial_output_max_abs_error": float((candidate_output - baseline_output).abs().max()),
        "loss_items": loss_items.detach().cpu().tolist(),
        "detection_loss_active_gamma_count": detection_gamma_active,
        "gamma_gradients_step0": gamma_gradients,
        "parameters": {"b0": baseline_params, "s1": candidate_params, "delta": parameter_delta},
        "gflops": {"b0": baseline_flops, "s1": candidate_flops, "delta": flops_delta},
        "latency_ms": {"b0": baseline_latency, "s1": candidate_latency, "delta": latency_delta},
        "onnx": str(args.onnx) if args.onnx else None,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
