#!/usr/bin/env python3
"""Prune a trained RoadSnake adapter and emit a native YOLO26 Detect checkpoint.

RoadSnakeDetect subclasses the stock Detect head and adds only one residual
adapter plus a forward override.  T1 changes the trained head back to the
stock Detect class and removes that adapter.  All inherited O2M/O2O, box and
classification weights are retained byte-for-byte.
"""

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
from ultralytics.nn.roadsnake import RoadSnakeDetect, RoadSnakeO2MDetect  # noqa: E402
from ultralytics.utils.torch_utils import get_flops  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, required=True, help="Trained RoadSnake best.pt"
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Native-Detect T1 checkpoint"
    )
    parser.add_argument("--report", type=Path, required=True, help="JSON audit report")
    parser.add_argument(
        "--device",
        default="cpu",
        help="cpu or CUDA device used for equivalence/latency",
    )
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
        output: list[torch.Tensor] = []
        for key in sorted(value):
            output.extend(tensors(value[key]))
        return output
    if isinstance(value, (list, tuple)):
        output = []
        for item in value:
            output.extend(tensors(item))
        return output
    return []


def output_error(reference: Any, candidate: Any) -> dict[str, float | int]:
    reference_tensors = tensors(reference)
    candidate_tensors = tensors(candidate)
    if len(reference_tensors) != len(candidate_tensors):
        raise AssertionError(
            f"Output tensor count differs: {len(reference_tensors)} != {len(candidate_tensors)}"
        )
    max_abs = 0.0
    max_rel = 0.0
    elements = 0
    for left, right in zip(reference_tensors, candidate_tensors):
        if left.shape != right.shape:
            raise AssertionError(
                f"Output shape differs: {tuple(left.shape)} != {tuple(right.shape)}"
            )
        delta = (left.float() - right.float()).abs()
        max_abs = max(max_abs, float(delta.max()) if delta.numel() else 0.0)
        denominator = left.float().abs().clamp_min(1e-12)
        relative = delta / denominator
        max_rel = max(max_rel, float(relative.max()) if relative.numel() else 0.0)
        elements += delta.numel()
    return {"max_abs": max_abs, "max_rel": max_rel, "elements": elements}


def head(model: torch.nn.Module) -> RoadSnakeDetect | RoadSnakeO2MDetect:
    candidate = model.model[-1]
    if not isinstance(candidate, (RoadSnakeDetect, RoadSnakeO2MDetect)):
        raise TypeError(
            f"Expected RoadSnakeDetect/RoadSnakeO2MDetect, got {type(candidate).__name__}"
        )
    if not isinstance(candidate, Detect):
        raise TypeError("RoadSnakeDetect is no longer a Detect subclass")
    return candidate


def prune_model(model: torch.nn.Module) -> tuple[float, list[str]]:
    road_head = head(model)
    gamma = float(road_head.road_snake.gamma.detach().float().cpu())
    removed_keys = sorted(
        f"model.{len(model.model) - 1}.road_snake.{key}"
        for key in road_head.road_snake.state_dict()
    )

    # The subclass adds no storage layout beyond regular Python/nn.Module state.
    # Reclassing preserves every inherited Detect module and runtime attribute.
    del road_head.road_snake
    road_head.__class__ = Detect

    yaml = getattr(model, "yaml", None)
    if isinstance(yaml, dict) and isinstance(yaml.get("head"), list) and yaml["head"]:
        final = list(yaml["head"][-1])
        final[2] = "Detect"
        final[3] = [int(getattr(road_head, "nc", 80))]
        yaml["head"][-1] = final
    return gamma, removed_keys


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
    args.source = args.source.resolve()
    args.output = args.output.resolve()
    args.report = args.report.resolve()
    if args.source == args.output:
        raise ValueError("--output must not overwrite --source")
    if not args.source.is_file():
        raise FileNotFoundError(args.source)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(args.source, map_location="cpu", weights_only=False)
    selected_key = "ema" if checkpoint.get("ema") is not None else "model"
    if checkpoint.get(selected_key) is None:
        raise KeyError("Checkpoint contains neither a model nor an EMA model")

    trained = copy.deepcopy(checkpoint[selected_key]).float().eval()
    source_head = head(trained)
    learned_gamma = float(source_head.road_snake.gamma.detach().float().cpu())

    gamma_zero = copy.deepcopy(trained)
    with torch.no_grad():
        head(gamma_zero).road_snake.gamma.zero_()
    pruned = copy.deepcopy(gamma_zero)
    pruned_gamma, removed_keys = prune_model(pruned)

    device = torch.device("cuda:" + str(args.device) if str(args.device).isdigit() else str(args.device))
    gamma_zero.to(device).eval()
    pruned.to(device).eval()
    generator = torch.Generator(device="cpu").manual_seed(20260818)
    image = torch.rand((1, 3, args.imgsz, args.imgsz), generator=generator).to(device)
    with torch.inference_mode():
        reference_output = gamma_zero(image)
        pruned_output = pruned(image)
    pre_save_error = output_error(reference_output, pruned_output)
    if pre_save_error["max_abs"] != 0.0:
        raise AssertionError(
            f"Pruned model is not bit-exact before save: {pre_save_error}"
        )

    gamma_zero_latency = latency_ms(gamma_zero, image, args.warmup, args.repeats)
    pruned_latency = latency_ms(pruned, image, args.warmup, args.repeats)
    peak_vram = None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        with torch.inference_mode():
            pruned(image)
        synchronize(device)
        peak_vram = int(torch.cuda.max_memory_allocated(device))

    # Update all serialized inference candidates. Stripped Ultralytics checkpoints
    # normally contain only model; non-stripped checkpoints may also contain EMA.
    for key in ("model", "ema"):
        if checkpoint.get(key) is None:
            continue
        serial_model = copy.deepcopy(checkpoint[key]).float().eval()
        prune_model(serial_model)
        checkpoint[key] = serial_model.half()
    torch.save(checkpoint, args.output)

    reloaded = YOLO(str(args.output)).model.float().to(device).eval()
    if not isinstance(reloaded.model[-1], Detect) or isinstance(
        reloaded.model[-1], (RoadSnakeDetect, RoadSnakeO2MDetect)
    ):
        raise AssertionError(
            f"Reloaded head is not native Detect: {type(reloaded.model[-1]).__name__}"
        )
    if any("road_snake" in key for key in reloaded.state_dict()):
        raise AssertionError("Reloaded state still contains RoadSnake parameters")
    with torch.inference_mode():
        reload_output = reloaded(image)
    reload_error = output_error(reference_output, reload_output)
    if reload_error["max_abs"] != 0.0:
        raise AssertionError(f"Reloaded T1 model is not bit-exact: {reload_error}")

    report = {
        "source": str(args.source),
        "source_sha256": sha256(args.source),
        "output": str(args.output),
        "output_sha256": sha256(args.output),
        "checkpoint_model_key": selected_key,
        "learned_gamma": learned_gamma,
        "pruned_reference_gamma": pruned_gamma,
        "removed_state_items": len(removed_keys),
        "removed_state_keys": removed_keys,
        "head_class_after_reload": type(reloaded.model[-1]).__name__,
        "road_snake_keys_after_reload": sum(
            "road_snake" in key for key in reloaded.state_dict()
        ),
        "equivalence_before_save": pre_save_error,
        "equivalence_after_reload": reload_error,
        "parameters": {
            "gamma_zero_unpruned": sum(
                parameter.numel() for parameter in gamma_zero.parameters()
            ),
            "t1_pruned": sum(parameter.numel() for parameter in reloaded.parameters()),
        },
        "gflops_640": float(get_flops(reloaded, imgsz=args.imgsz)),
        "batch1_latency_ms": {
            "gamma_zero_unpruned": gamma_zero_latency,
            "t1_pruned": pruned_latency,
            "relative_delta": pruned_latency / gamma_zero_latency - 1.0,
        },
        "peak_vram_bytes": peak_vram,
        "imgsz": args.imgsz,
        "device": str(device),
        "ok": True,
    }
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
