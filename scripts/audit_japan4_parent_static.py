"""Unified RTX GPU static audit for proposed Japan4 architecture parents."""

from __future__ import annotations

import argparse
import copy
import json
import statistics
import sys
import time
import traceback
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ultralytics.nn.modules.head import Detect  # noqa: E402
from ultralytics.nn.tasks import DetectionModel, torch_safe_load  # noqa: E402
from ultralytics.cfg import get_cfg  # noqa: E402
from ultralytics.utils.torch_utils import get_flops, get_flops_with_torch_profiler, intersect_dicts  # noqa: E402

MODELS = {
    "B0": ROOT / "ultralytics/cfg/models/26/yolo26.yaml",
    "StarNet": ROOT / "ultralytics/cfg/models/26/yolo26-StarNet.yaml",
    "MobileMamba": ROOT / "ultralytics/cfg/models/26/yolo26-MobileMamba-Backbone.yaml",
    "FFAFusion": ROOT / "ultralytics/cfg/models/26/yolo26-FFAFusion-Backbone.yaml",
    "EfficientViM": ROOT / "ultralytics/cfg/models/26/yolo26-EfficientViM-Backbone.yaml",
}


def tensors(value):
    """Yield tensors recursively from Ultralytics outputs."""
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from tensors(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from tensors(child)


def shape_tree(value):
    """Convert a nested output into JSON-safe shape metadata."""
    if isinstance(value, torch.Tensor):
        return list(value.shape)
    if isinstance(value, dict):
        return {key: shape_tree(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [shape_tree(child) for child in value]
    return type(value).__name__


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(round((len(ordered) - 1) * fraction), len(ordered) - 1)]


def transfer_audit(model: DetectionModel, weights: Path) -> tuple[dict, dict[str, torch.Tensor]]:
    """Measure the exact name-and-shape transfer used by the current implementation."""
    checkpoint, _ = torch_safe_load(weights)
    source_model = (checkpoint.get("ema") or checkpoint["model"]).float()
    source = source_model.state_dict()
    target = model.state_dict()
    matched = intersect_dicts(source, target)
    target_parameters = dict(model.named_parameters())
    source_parameters = dict(source_model.named_parameters())
    matched_target_parameters = sum(target_parameters[key].numel() for key in matched if key in target_parameters)
    matched_source_parameters = sum(source_parameters[key].numel() for key in matched if key in source_parameters)
    report = {
        "matched_state_items": len(matched),
        "target_state_items": len(target),
        "source_state_items": len(source),
        "matched_target_parameters": matched_target_parameters,
        "target_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "matched_target_parameter_fraction": matched_target_parameters / sum(
            parameter.numel() for parameter in model.parameters()
        ),
        "matched_source_parameters": matched_source_parameters,
        "source_parameters": sum(parameter.numel() for parameter in source_model.parameters()),
        "matched_source_parameter_fraction": matched_source_parameters / sum(
            parameter.numel() for parameter in source_model.parameters()
        ),
        "missing_or_mismatched_count": len(target) - len(matched),
        "unexpected_or_unused_count": len(source) - len(matched),
        "missing_or_mismatched_keys": [key for key in target if key not in matched],
        "unexpected_or_unused_keys": [key for key in source if key not in matched],
    }
    return report, matched


def load_matched(model: DetectionModel, matched: dict[str, torch.Tensor]) -> None:
    """Load and verify every matched official tensor exactly."""
    model.load_state_dict(matched, strict=False)
    state = model.state_dict()
    changed = [key for key, value in matched.items() if not torch.equal(state[key].cpu(), value.cpu())]
    if changed:
        raise AssertionError(f"Transferred tensors changed after load: {changed[:10]}")


def capture_detect_shapes(
    model: DetectionModel, image: torch.Tensor
) -> tuple[object, list[list[int]], dict[str, object]]:
    """Capture top-level outputs and the actual P3/P4/P5 tensors consumed by Detect."""
    head = model.model[-1]
    if not isinstance(head, Detect):
        raise TypeError(f"Final module is not Detect: {type(head).__name__}")
    captured: list[list[int]] = []
    top_level_outputs: dict[str, object] = {}

    def hook(_module, inputs):
        features = inputs[0]
        captured.extend(list(feature.shape) for feature in features)

    handle = head.register_forward_pre_hook(hook)
    output_handles = [
        module.register_forward_hook(
            lambda _module, _inputs, output, index=index: top_level_outputs.__setitem__(str(index), shape_tree(output))
        )
        for index, module in enumerate(model.model)
    ]
    try:
        output = model(image)
    finally:
        handle.remove()
        for output_handle in output_handles:
            output_handle.remove()
    return output, captured, top_level_outputs


def amp_detection_step(model: DetectionModel, device: torch.device, imgsz: int) -> dict:
    """Run real YOLO end-to-end detection loss and backward under CUDA AMP."""
    model.train().zero_grad(set_to_none=True)
    image = torch.rand(1, 3, imgsz, imgsz, device=device)
    batch = {
        "batch_idx": torch.tensor([0.0, 0.0], device=device),
        "cls": torch.tensor([[0.0], [2.0]], device=device),
        "bboxes": torch.tensor([[0.35, 0.45, 0.28, 0.07], [0.68, 0.62, 0.18, 0.22]], device=device),
    }
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    criterion = model.init_criterion()
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
        predictions = model(image)
        loss, loss_items = criterion(predictions, batch)
    if not all(torch.isfinite(tensor).all() for tensor in tensors(predictions)):
        raise FloatingPointError("Non-finite prediction under AMP")
    if not torch.isfinite(loss).all() or not torch.isfinite(loss_items).all():
        raise FloatingPointError("Non-finite detection loss under AMP")
    loss.sum().backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    if not gradients or not all(torch.isfinite(gradient).all() for gradient in gradients):
        raise FloatingPointError("Missing or non-finite gradients under AMP")
    layer_gradients = []
    for index, module in enumerate(model.model):
        parameters = list(module.parameters())
        with_grad = [parameter.grad for parameter in parameters if parameter.grad is not None]
        layer_gradients.append(
            {
                "index": index,
                "module": type(module).__name__,
                "parameter_tensors": len(parameters),
                "gradient_tensors": len(with_grad),
                "nonzero_gradient_tensors": sum(bool(gradient.detach().abs().sum() > 0) for gradient in with_grad),
                "all_finite": all(torch.isfinite(gradient).all() for gradient in with_grad),
            }
        )
    torch.cuda.synchronize(device)
    return {
        "loss": float(loss.detach().sum().cpu()),
        "loss_items": loss_items.detach().float().cpu().tolist(),
        "finite_gradient_tensors": len(gradients),
        "nonzero_gradient_tensors": sum(bool(gradient.detach().abs().sum() > 0) for gradient in gradients),
        "top_level_layer_gradients": layer_gradients,
        "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20,
        "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20,
    }


@torch.inference_mode()
def latency(model: DetectionModel, device: torch.device, imgsz: int, warmup: int, runs: int) -> dict:
    """Measure synchronized fused PyTorch FP32 batch-1 latency."""
    model = copy.deepcopy(model).to(device).float().eval().fuse(verbose=False)
    image = torch.zeros(1, 3, imgsz, imgsz, device=device)
    for _ in range(warmup):
        model(image)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    samples = []
    for _ in range(runs):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        model(image)
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return {
        "fused": True,
        "dtype": "float32",
        "batch": 1,
        "median_ms": statistics.median(samples),
        "p10_ms": percentile(samples, 0.10),
        "p90_ms": percentile(samples, 0.90),
        "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20,
    }


def export_onnx(model: DetectionModel, imgsz: int, path: Path) -> dict:
    """Export a fused CPU model and run ONNX structural validation."""
    import onnx

    export_model = copy.deepcopy(model).cpu().float().eval().fuse(verbose=False)
    export_model.model[-1].export = True
    image = torch.zeros(1, 3, imgsz, imgsz)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        export_model,
        image,
        path,
        opset_version=17,
        input_names=["images"],
        output_names=["output"],
    )
    onnx.checker.check_model(onnx.load(path))
    return {"passed": True, "path": str(path.resolve()), "bytes": path.stat().st_size}


def audit(name: str, args: argparse.Namespace) -> dict:
    yaml_path = MODELS[name]
    result = {
        "name": name,
        "yaml": str(yaml_path.resolve()),
        "yaml_exists": yaml_path.is_file(),
        "device": str(args.device),
        "imgsz": args.imgsz,
        "errors": {},
    }
    if not yaml_path.is_file():
        result["errors"]["yaml"] = "missing"
        return result

    torch.manual_seed(42)
    device = torch.device(args.device)
    started = time.time()
    try:
        model = DetectionModel(str(yaml_path), ch=3, nc=4, verbose=False).float()
        model.args = get_cfg()
        model.args.epochs = 30
        result["build"] = True
        result["module_count"] = len(model.model)
        result["top_level_modules"] = [type(module).__name__ for module in model.model]
        result["parameters"] = sum(parameter.numel() for parameter in model.parameters())
        transfer, matched = transfer_audit(model, args.weights)
        result["pretrained_transfer"] = transfer
        load_matched(model, matched)
        result["pretrained_load_verified"] = True

        sample = torch.zeros(1, 3, args.imgsz, args.imgsz, device=device)
        model = model.to(device).eval()
        with torch.inference_mode():
            output, detect_shapes, top_level_outputs = capture_detect_shapes(model, sample)
        result["forward"] = True
        result["forward_finite"] = all(torch.isfinite(tensor).all() for tensor in tensors(output))
        result["output_shapes"] = shape_tree(output)
        result["detect_input_shapes_p3_p4_p5"] = detect_shapes
        result["top_level_output_shapes"] = top_level_outputs
        if not result["forward_finite"]:
            raise FloatingPointError("Non-finite inference output")

        result["gflops"] = get_flops(model, args.imgsz) or get_flops_with_torch_profiler(model, args.imgsz)
        result["amp_detection_step"] = amp_detection_step(model, device, args.imgsz)
        result["amp"] = True
        result["backward"] = True
        model.zero_grad(set_to_none=True)
        result["latency"] = latency(model, device, args.imgsz, args.warmup, args.runs)
        try:
            result["onnx"] = export_onnx(model, args.imgsz, args.onnx_dir / f"{name}.onnx")
        except Exception as error:  # Preserve all other audit evidence when one exporter fails.
            result["onnx"] = {"passed": False, "error": f"{type(error).__name__}: {error}"}
            result["errors"]["onnx"] = traceback.format_exc()
    except Exception as error:
        result["errors"]["runtime"] = traceback.format_exc()
        result["runtime_error"] = f"{type(error).__name__}: {error}"
    result["elapsed_seconds"] = time.time() - started
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=tuple(MODELS), required=True)
    parser.add_argument("--weights", type=Path, default=ROOT / "yolo26n.pt")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--onnx-dir", type=Path, default=ROOT / "reports/japan4_parent_static/onnx")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if not args.weights.is_file() or not torch.cuda.is_available() or args.imgsz != 640:
        raise RuntimeError("Official weights, CUDA, and the frozen 640 input are required")
    result = audit(args.model, args)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    if result["errors"].get("runtime"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
