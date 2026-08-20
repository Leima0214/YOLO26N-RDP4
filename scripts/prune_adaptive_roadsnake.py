#!/usr/bin/env python3
"""Physically prune SA-RS/MG-SA-RS to a native YOLO26 Detect checkpoint."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import YOLO  # noqa: E402
from ultralytics.nn.modules.head import Detect  # noqa: E402
from ultralytics.nn.roadsnake_adaptive import (  # noqa: E402
    MetricGuidedScaleAdaptiveRoadSnakeDetect,
    ScaleAdaptiveRoadSnakeDetect,
)
from ultralytics.nn.tasks import DetectionModel  # noqa: E402
from ultralytics.utils.torch_utils import get_flops, get_flops_with_torch_profiler  # noqa: E402

BASELINE = ROOT / "ultralytics/cfg/models/26/yolo26.yaml"
HEAD_TYPES = {
    "sa": ScaleAdaptiveRoadSnakeDetect,
    "mgsa": MetricGuidedScaleAdaptiveRoadSnakeDetect,
}
BANNED_KEY_TOKENS = (
    "road_snake",
    "scale_head",
    "scale_base",
    "scale_metric",
    "sobel",
    "metric_",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=tuple(HEAD_TYPES), required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=100)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensors(value: Any) -> list[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, dict):
        result: list[torch.Tensor] = []
        for key in sorted(value):
            result.extend(tensors(value[key]))
        return result
    if isinstance(value, (tuple, list)):
        result = []
        for item in value:
            result.extend(tensors(item))
        return result
    return []


def output_error(reference: Any, candidate: Any) -> dict[str, float | int]:
    left_tensors, right_tensors = tensors(reference), tensors(candidate)
    if len(left_tensors) != len(right_tensors):
        raise AssertionError(
            f"Output tensor count differs: {len(left_tensors)} != {len(right_tensors)}"
        )
    maximum = 0.0
    elements = 0
    for left, right in zip(left_tensors, right_tensors):
        if left.shape != right.shape:
            raise AssertionError(f"Output shape differs: {left.shape} != {right.shape}")
        delta = (left.float() - right.float()).abs()
        maximum = max(maximum, float(delta.max()) if delta.numel() else 0.0)
        elements += delta.numel()
    return {"max_abs": maximum, "elements": elements}


def adaptive_head(
    model: torch.nn.Module, variant: str | None = None
) -> ScaleAdaptiveRoadSnakeDetect:
    candidate = model.model[-1]
    if variant is not None and type(candidate) is not HEAD_TYPES[variant]:
        raise TypeError(
            f"Expected {HEAD_TYPES[variant].__name__}, got {type(candidate).__name__}"
        )
    if not isinstance(candidate, ScaleAdaptiveRoadSnakeDetect):
        raise TypeError(f"Expected adaptive RoadSnake Detect, got {type(candidate).__name__}")
    return candidate


def prune_model(model: torch.nn.Module, variant: str | None = None) -> tuple[float, list[str]]:
    """Delete every adaptive adapter module and restore the native Detect class."""
    head = adaptive_head(model, variant)
    gamma = float(head.road_snake.gamma.detach().float().cpu())
    removed = sorted(
        f"model.{len(model.model) - 1}.road_snake.{key}"
        for key in head.road_snake.state_dict()
    )
    del head.road_snake
    head.__class__ = Detect
    yaml_data = getattr(model, "yaml", None)
    if isinstance(yaml_data, dict) and isinstance(yaml_data.get("head"), list):
        final = list(yaml_data["head"][-1])
        final[2] = "Detect"
        final[3] = [int(head.nc)]
        yaml_data["head"][-1] = final
    return gamma, removed


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def latency_ms(
    model: torch.nn.Module, image: torch.Tensor, warmup: int, repeats: int
) -> float:
    for _ in range(warmup):
        model(image)
    synchronize(image.device)
    started = time.perf_counter()
    for _ in range(repeats):
        model(image)
    synchronize(image.device)
    return (time.perf_counter() - started) * 1000.0 / repeats


def main() -> None:
    args = parse_args()
    args.source = args.source.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    args.report = args.report.expanduser().resolve()
    args.onnx = args.onnx.expanduser().resolve()
    if not args.source.is_file():
        raise FileNotFoundError(args.source)
    for target in (args.output, args.report, args.onnx):
        if target.exists():
            raise FileExistsError(f"Refusing to overwrite {target}")
        target.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(args.source, map_location="cpu", weights_only=False)
    selected_key = "ema" if checkpoint.get("ema") is not None else "model"
    if checkpoint.get(selected_key) is None:
        raise KeyError("Checkpoint contains neither a model nor an EMA model")
    trained = copy.deepcopy(checkpoint[selected_key]).float().eval()
    learned_gamma = float(adaptive_head(trained, args.variant).road_snake.gamma.detach())

    device = torch.device(f"cuda:{args.device}" if str(args.device).isdigit() else args.device)
    trained.to(device).eval()
    generator = torch.Generator(device="cpu").manual_seed(20260820)
    image = torch.rand((1, 3, args.imgsz, args.imgsz), generator=generator).to(device)
    with torch.inference_mode():
        full_output = trained(image)
    if not tensors(full_output) or not all(torch.isfinite(x).all() for x in tensors(full_output)):
        raise AssertionError("Full adaptive checkpoint produced non-finite output")

    gamma_zero = copy.deepcopy(trained)
    with torch.no_grad():
        adaptive_head(gamma_zero, args.variant).road_snake.gamma.zero_()
    pruned = copy.deepcopy(gamma_zero)
    pruned_gamma, removed_keys = prune_model(pruned, args.variant)
    gamma_zero.eval()
    pruned.eval()
    with torch.inference_mode():
        reference_output = gamma_zero(image)
        pruned_output = pruned(image)
    pre_save_error = output_error(reference_output, pruned_output)
    if pre_save_error["max_abs"] != 0.0:
        raise AssertionError(f"Gamma0 and physically pruned outputs differ: {pre_save_error}")

    gamma_zero_latency = latency_ms(gamma_zero, image, args.warmup, args.repeats)
    pruned_latency = latency_ms(pruned, image, args.warmup, args.repeats)
    peak_vram = None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        with torch.inference_mode():
            pruned(image)
        synchronize(device)
        peak_vram = int(torch.cuda.max_memory_allocated(device))

    for key in ("model", "ema"):
        if checkpoint.get(key) is None:
            continue
        serial_model = copy.deepcopy(checkpoint[key]).float().eval()
        prune_model(serial_model, args.variant)
        checkpoint[key] = serial_model.half()
    torch.save(checkpoint, args.output)

    reloaded = YOLO(str(args.output)).model.float().to(device).eval()
    if type(reloaded.model[-1]) is not Detect:
        raise AssertionError(f"Reloaded head is not native Detect: {type(reloaded.model[-1]).__name__}")
    residual_keys = [
        key
        for key in reloaded.state_dict()
        if any(token in key for token in BANNED_KEY_TOKENS)
    ]
    if residual_keys:
        raise AssertionError(f"Adaptive keys remain after pruning: {residual_keys[:10]}")
    with torch.inference_mode():
        reloaded_output = reloaded(image)
    reload_error = output_error(reference_output, reloaded_output)
    if reload_error["max_abs"] != 0.0:
        raise AssertionError(f"Reloaded pruned checkpoint is not bit-exact: {reload_error}")

    native = DetectionModel(str(BASELINE), nc=int(reloaded.model[-1].nc), ch=3, verbose=False)
    native_params = sum(parameter.numel() for parameter in native.parameters())
    pruned_params = sum(parameter.numel() for parameter in reloaded.parameters())
    if pruned_params != native_params:
        raise AssertionError(f"Pruned params {pruned_params} != native B0 {native_params}")
    native_flops = float(get_flops(native, args.imgsz) or get_flops_with_torch_profiler(native, args.imgsz))
    pruned_flops = float(get_flops(reloaded, args.imgsz) or get_flops_with_torch_profiler(reloaded, args.imgsz))
    if native_flops != pruned_flops:
        raise AssertionError(f"Pruned GFLOPs {pruned_flops} != native B0 {native_flops}")

    import onnx

    export_model = copy.deepcopy(reloaded).eval()
    export_model.model[-1].export = True
    torch.onnx.export(
        export_model,
        image,
        args.onnx,
        opset_version=17,
        input_names=["images"],
        output_names=["output"],
    )
    onnx.checker.check_model(onnx.load(args.onnx))

    report = {
        "variant": args.variant,
        "source": str(args.source),
        "source_sha256": sha256(args.source),
        "output": str(args.output),
        "output_sha256": sha256(args.output),
        "onnx": str(args.onnx),
        "onnx_sha256": sha256(args.onnx),
        "checkpoint_model_key": selected_key,
        "learned_gamma": learned_gamma,
        "gamma_zero_reference": pruned_gamma,
        "removed_state_items": len(removed_keys),
        "removed_state_keys": removed_keys,
        "residual_adaptive_keys": residual_keys,
        "head_class_after_reload": type(reloaded.model[-1]).__name__,
        "equivalence_before_save": pre_save_error,
        "equivalence_after_reload": reload_error,
        "parameters": {"native_b0": native_params, "pruned": pruned_params},
        "gflops_640": {"native_b0": native_flops, "pruned": pruned_flops},
        "latency_ms": {
            "gamma_zero_unpruned": gamma_zero_latency,
            "pruned_native": pruned_latency,
            "relative_delta": pruned_latency / gamma_zero_latency - 1.0,
        },
        "peak_vram_bytes": peak_vram,
        "test_sealed": True,
        "ok": True,
    }
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
