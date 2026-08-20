#!/usr/bin/env python3
"""Unified static, real-batch, AMP, pruning, and deploy audit for SA-RS/MG-SA-RS."""

from __future__ import annotations

import argparse
import copy
import json
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.prune_adaptive_roadsnake import output_error, prune_model, tensors  # noqa: E402
from ultralytics import YOLO  # noqa: E402
from ultralytics.cfg import DEFAULT_CFG, get_cfg  # noqa: E402
from ultralytics.data.build import build_dataloader, build_yolo_dataset  # noqa: E402
from ultralytics.data.utils import check_det_dataset  # noqa: E402
from ultralytics.models.yolo.detect.train import DetectionTrainer  # noqa: E402
from ultralytics.nn.modules.head import Detect  # noqa: E402
from ultralytics.nn.roadsnake import RoadSnakeAdapter  # noqa: E402
from ultralytics.nn.roadsnake_adaptive import (  # noqa: E402
    MetricGuidedScaleAdaptiveRoadSnakeAdapter,
    MetricGuidedScaleAdaptiveRoadSnakeDetect,
    ScaleAdaptiveRoadSnakeAdapter,
    ScaleAdaptiveRoadSnakeDetect,
)
from ultralytics.nn.tasks import DetectionModel  # noqa: E402
from ultralytics.utils.torch_utils import get_flops, get_flops_with_torch_profiler  # noqa: E402

BASELINE = ROOT / "ultralytics/cfg/models/26/yolo26.yaml"
R1_YAML = ROOT / "ultralytics/cfg/models/26/yolo26n-japan4-roadsnake-r1.yaml"
VARIANTS = {
    "sa": {
        "yaml": ROOT / "ultralytics/cfg/models/26/yolo26n-japan4-sa-roadsnake.yaml",
        "head": ScaleAdaptiveRoadSnakeDetect,
        "adapter": ScaleAdaptiveRoadSnakeAdapter,
    },
    "mgsa": {
        "yaml": ROOT / "ultralytics/cfg/models/26/yolo26n-japan4-mg-sa-roadsnake.yaml",
        "head": MetricGuidedScaleAdaptiveRoadSnakeDetect,
        "adapter": MetricGuidedScaleAdaptiveRoadSnakeAdapter,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=tuple(VARIANTS), required=True)
    parser.add_argument("--weights", type=Path, default=ROOT / "yolo26n.pt")
    parser.add_argument("--data", type=Path, default=ROOT / "configs/japan4_clean_v3_remote.yaml")
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--report-dir", type=Path, default=ROOT / "reports/adaptive_roadsnake_static")
    parser.add_argument("--onnx", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--latency-warmup", type=int, default=20)
    parser.add_argument("--latency-repeats", type=int, default=100)
    return parser.parse_args()


def device_from(value: str) -> torch.device:
    return torch.device(f"cuda:{value}" if str(value).isdigit() else value)


def prediction(output: Any) -> torch.Tensor:
    return output[0] if isinstance(output, tuple) else output


def finite_tensors(value: Any) -> bool:
    values = tensors(value)
    return bool(values) and all(torch.isfinite(tensor).all() for tensor in values)


def grad_l1(parameter: torch.Tensor | None, name: str) -> float:
    if parameter is None or parameter.grad is None:
        raise AssertionError(f"Missing gradient: {name}")
    gradient = parameter.grad
    if not torch.isfinite(gradient).all() or gradient.abs().sum() == 0:
        raise AssertionError(f"Non-finite or zero gradient: {name}")
    return float(gradient.detach().float().abs().sum().cpu())


def make_real_batch(data_yaml: Path, imgsz: int, batch_size: int) -> tuple[dict, dict]:
    """Load only a deterministic labeled Train batch; Test is never resolved."""
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


def build_models(variant: str, weights: Path, device: torch.device) -> tuple[DetectionModel, DetectionModel, dict]:
    """Load every legal B0 tensor, then prove a Trainer rebuild preserves candidate state exactly."""
    spec = VARIANTS[variant]
    official = YOLO(str(weights), task="detect", verbose=False).model.float()
    baseline = DetectionModel(str(BASELINE), nc=4, ch=3, verbose=False).float()
    baseline.load(official, verbose=False)
    candidate = DetectionModel(str(spec["yaml"]), nc=4, ch=3, verbose=False).float()

    source, target = baseline.state_dict(), candidate.state_dict()
    missing = [key for key, value in source.items() if key not in target or target[key].shape != value.shape]
    if missing:
        raise AssertionError(f"Candidate cannot inherit all B0 tensors: {missing[:10]}")
    candidate.load(baseline, verbose=False)
    changed = [key for key, value in source.items() if not torch.equal(value, candidate.state_dict()[key])]
    if changed:
        raise AssertionError(f"Inherited B0 tensors changed: {changed[:10]}")

    trainer = object.__new__(DetectionTrainer)
    trainer.data = {"nc": 4, "channels": 3}
    rebuilt = DetectionTrainer.get_model(trainer, cfg=str(spec["yaml"]), weights=candidate, verbose=False).float()
    rebuilt_state = rebuilt.state_dict()
    rebuild_changed = [
        key for key, value in candidate.state_dict().items() if key not in rebuilt_state or not torch.equal(value, rebuilt_state[key])
    ]
    if rebuild_changed:
        raise AssertionError(f"Trainer reconstruction changed candidate state: {rebuild_changed[:10]}")

    for model in (baseline, rebuilt):
        model.args = get_cfg(DEFAULT_CFG, {"box": 7.5, "cls": 0.5, "dfl": 1.5, "epochs": 30})
        model.criterion = None
    audit = {
        "b0_state_items": len(source),
        "candidate_state_items": len(target),
        "b0_inherited_items": len(source),
        "b0_inherited_ratio": 1.0,
        "shared_pretrained_tensors": len(source),
        "transferred_tensors": len(source),
        "b0_inherited_values": sum(value.numel() for value in source.values()),
        "candidate_state_values": sum(value.numel() for value in target.values()),
        "new_state_values": sum(target[key].numel() for key in set(target) - set(source)),
        "new_state_items": sorted(set(target) - set(source)),
        "missing_new_tensors": sorted(set(target) - set(source)),
        "unexpected_tensors": [],
        "trainer_rebuild_changed": rebuild_changed,
    }
    return baseline.to(device), rebuilt.to(device), audit


@torch.inference_mode()
def paired_latency(
    baseline: torch.nn.Module,
    candidate: torch.nn.Module,
    sample: torch.Tensor,
    warmup: int,
    repeats: int,
) -> dict[str, float] | None:
    if sample.device.type != "cuda":
        return None
    for _ in range(warmup):
        baseline(sample)
        candidate(sample)
    torch.cuda.synchronize(sample.device)
    timings: tuple[list[float], list[float]] = ([], [])
    for index in range(repeats):
        order = ((baseline, timings[0]), (candidate, timings[1]))
        if index % 2:
            order = tuple(reversed(order))
        for model, values in order:
            torch.cuda.synchronize(sample.device)
            started = time.perf_counter_ns()
            model(sample)
            torch.cuda.synchronize(sample.device)
            values.append((time.perf_counter_ns() - started) / 1e6)
    base_ms, candidate_ms = map(statistics.median, timings)
    return {"b0_ms": base_ms, "candidate_ms": candidate_ms, "delta": candidate_ms / base_ms - 1.0}


def geometry_audit(adapter: ScaleAdaptiveRoadSnakeAdapter, device: torch.device) -> dict[str, Any]:
    """Prove scale=1 reproduces R1 and scale changes only the longitudinal axis."""
    generator = torch.Generator(device="cpu").manual_seed(20260820)
    feature = torch.randn((2, adapter.hidden, 12, 12), generator=generator).to(device)
    offset = torch.randn((2, adapter.kernel_size, 12, 12), generator=generator).tanh().to(device)
    r1 = RoadSnakeAdapter(
        channels=adapter.channels,
        kernel_size=adapter.kernel_size,
        expansion=adapter.hidden / adapter.channels,
        max_offset=adapter.max_offset,
    ).to(device)
    r1.load_state_dict(adapter.state_dict(), strict=False)
    with torch.inference_mode():
        r1_h = r1._sample_curve(feature, offset, horizontal=True)
        r1_v = r1._sample_curve(feature, offset, horizontal=False)
        ones = feature.new_ones((feature.shape[0], 1, feature.shape[2], feature.shape[3]))
        sa_h, h_x_1, h_y_1, curve_h_1 = adapter._sample_curve_with_scale(feature, offset, ones, horizontal=True)
        sa_v, v_x_1, v_y_1, curve_v_1 = adapter._sample_curve_with_scale(feature, offset, ones, horizontal=False)
    torch.testing.assert_close(sa_h, r1_h, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(sa_v, r1_v, atol=1e-6, rtol=1e-6)

    spans = {}
    coordinates = {}
    for scale_value in (0.5, 1.0, 2.0):
        scale = torch.full_like(ones, scale_value)
        h_x, h_y, h_curve = adapter.sampling_coordinates(feature, offset, scale, horizontal=True)
        v_x, v_y, v_curve = adapter.sampling_coordinates(feature, offset, scale, horizontal=False)
        torch.testing.assert_close(h_y - h_y_1, torch.zeros_like(h_y), atol=0, rtol=0)
        torch.testing.assert_close(v_x - v_x_1, torch.zeros_like(v_x), atol=0, rtol=0)
        torch.testing.assert_close(h_curve, curve_h_1, atol=0, rtol=0)
        torch.testing.assert_close(v_curve, curve_v_1, atol=0, rtol=0)
        span_h = float((h_x[:, -1] - h_x[:, 0]).mean().cpu())
        span_v = float((v_y[:, -1] - v_y[:, 0]).mean().cpu())
        expected = float((adapter.kernel_size - 1) * scale_value)
        if span_h != expected or span_v != expected:
            raise AssertionError(f"Unexpected longitudinal span at scale={scale_value}: {span_h}, {span_v}")
        spans[str(scale_value)] = {"feature_pixels": expected, "image_pixels_stride16": expected * 16.0}
        coordinates[str(scale_value)] = {
            "horizontal_grid_x_mean": float(h_x.mean().cpu()),
            "horizontal_grid_y_mean": float(h_y.mean().cpu()),
            "vertical_grid_x_mean": float(v_x.mean().cpu()),
            "vertical_grid_y_mean": float(v_y.mean().cpu()),
        }
    return {
        "scale1_r1_horizontal_max_abs": float((sa_h - r1_h).abs().max().cpu()),
        "scale1_r1_vertical_max_abs": float((sa_v - r1_v).abs().max().cpu()),
        "forced_scale_spans": spans,
        "coordinates": coordinates,
        "orthogonal_curve_unchanged": True,
    }


def export_onnx(model: torch.nn.Module, sample: torch.Tensor, path: Path) -> dict[str, Any]:
    import onnx

    path.parent.mkdir(parents=True, exist_ok=True)
    export_model = copy.deepcopy(model).eval()
    export_model.model[-1].export = True
    torch.onnx.export(
        export_model,
        sample,
        path,
        opset_version=17,
        input_names=["images"],
        output_names=["output"],
    )
    onnx.checker.check_model(onnx.load(path))
    return {"path": str(path), "bytes": path.stat().st_size, "checker": "PASS"}


def markdown_report(report: dict[str, Any]) -> str:
    return "\n".join(
        (
            f"# {report['variant'].upper()}-RS static audit",
            "",
            f"- Verdict: **{report['verdict']}**",
            f"- Test sealed: `{report['test_sealed']}`",
            f"- Initial B0 equivalence max abs: `{report['initial_output_max_abs_error']}`",
            f"- Trainer rebuild changed tensors: `{len(report['transfer']['trainer_rebuild_changed'])}`",
            f"- Params: `{report['parameters']}`",
            f"- GFLOPs: `{report['gflops_640']}`",
            f"- CUDA AMP: `{report['amp']}`",
            f"- Physical pruning: `{report['physical_pruning']}`",
            f"- Pruned ONNX: `{report['onnx']['pruned_native']}`",
            "",
            "No formal training was started.",
        )
    )


def main() -> None:
    args = parse_args()
    if args.imgsz < 64 or args.imgsz % 32:
        raise ValueError("imgsz must be >=64 and divisible by 32")
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    device = device_from(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA audit requested but CUDA is unavailable")

    report_dir = args.report_dir.expanduser().resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    spec = VARIANTS[args.variant]
    baseline, candidate, transfer = build_models(args.variant, args.weights.expanduser().resolve(), device)
    if type(baseline.model[-1]) is not Detect or type(candidate.model[-1]) is not spec["head"]:
        raise AssertionError("Unexpected baseline/candidate head class")
    adapter = candidate.model[-1].road_snake
    if type(adapter) is not spec["adapter"]:
        raise AssertionError(f"Unexpected adapter class: {type(adapter).__name__}")
    if float(adapter.gamma.detach()) != 0.0:
        raise AssertionError("Adaptive candidate must initialize gamma at zero")
    if not torch.count_nonzero(adapter.scale_head.weight) == 0 or not torch.count_nonzero(adapter.scale_head.bias) == 0:
        raise AssertionError("SA scale head must be zero initialized")
    if args.variant == "mgsa":
        if not torch.count_nonzero(adapter.scale_metric.weight) == 0 or not torch.count_nonzero(adapter.scale_metric.bias) == 0:
            raise AssertionError("MG scale metric head must be zero initialized")

    sample = torch.randn((1, 3, args.imgsz, args.imgsz), device=device)
    baseline.eval()
    candidate.eval()
    adapter.set_diagnostics(True)
    with torch.inference_mode():
        b0_raw_output = baseline(sample)
        candidate_raw_output = candidate(sample)
        b0_output = prediction(b0_raw_output)
        candidate_output = prediction(candidate_raw_output)
        initial_diagnostics = adapter.diagnostics()
    torch.testing.assert_close(candidate_output, b0_output, atol=0, rtol=0)
    initial_raw_error = output_error(b0_raw_output, candidate_raw_output)
    if initial_raw_error["max_abs"] != 0.0:
        raise AssertionError(f"Raw O2M/O2O step-0 output differs from B0: {initial_raw_error}")
    for name in ("scale_h", "scale_v"):
        torch.testing.assert_close(initial_diagnostics[name], torch.ones_like(initial_diagnostics[name]), atol=0, rtol=0)

    fused_baseline = copy.deepcopy(baseline).eval().fuse(verbose=False)
    fused_candidate = copy.deepcopy(candidate).eval().fuse(verbose=False)
    with torch.inference_mode():
        fused_error = output_error(fused_baseline(sample), fused_candidate(sample))
    if fused_error["max_abs"] != 0.0:
        raise AssertionError(f"Fused step-0 output differs from B0: {fused_error}")

    geometry = geometry_audit(adapter, device)
    feature = torch.randn((2, adapter.channels, args.imgsz // 16, args.imgsz // 16), device=device)
    with torch.inference_mode():
        adapter.gamma.fill_(0.05)
        isolated = adapter(feature[:1]).clone()
        batched = adapter(feature)[:1].clone()
        adapter.gamma.zero_()
    batch_isolation_error = float((isolated - batched).abs().max().cpu())
    torch.testing.assert_close(isolated, batched, atol=1e-6, rtol=1e-6)

    mg_audit: dict[str, Any] | None = None
    if isinstance(adapter, MetricGuidedScaleAdaptiveRoadSnakeAdapter):
        reduced = adapter.reduce(feature)
        cues, _ = adapter.metric_cues(reduced, detach=True)
        if cues.requires_grad or cues.shape[1] != 3 or not torch.isfinite(cues).all():
            raise AssertionError("MG metric cues are not detached, finite [B,3,H,W] tensors")
        with torch.inference_mode():
            sa_logits = adapter.scale_head(reduced)
            mg_logits, _ = adapter._scale_logits(reduced)
        torch.testing.assert_close(mg_logits, sa_logits, atol=0, rtol=0)
        with torch.no_grad():
            adapter.scale_metric.weight.fill_(0.01)
            changed_scale_h, changed_scale_v, _ = adapter.predict_scales(reduced)
            adapter.scale_metric.weight.zero_()
        spatial_std = float(torch.stack((changed_scale_h.std(), changed_scale_v.std())).mean().cpu())
        if spatial_std == 0.0:
            raise AssertionError("Forced non-zero metric weights did not change scale spatially")
        mg_audit = {
            "cue_shape": list(cues.shape),
            "cues_detached": not cues.requires_grad,
            "cues_finite": bool(torch.isfinite(cues).all()),
            "zero_metric_equals_sa_max_abs": float((mg_logits - sa_logits).abs().max().cpu()),
            "forced_metric_scale_spatial_std": spatial_std,
        }

    data, batch = make_real_batch(args.data.expanduser().resolve(), args.imgsz, args.batch)
    if not len(batch["cls"]):
        raise AssertionError("Selected Train batch has no labeled objects")
    batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
    batch["img"] = batch["img"].float().div(255.0)
    captured_features: list[list[torch.Tensor]] = []

    def retain_detect_inputs(_module: torch.nn.Module, inputs: tuple[Any, ...]) -> None:
        levels = inputs[0]
        p4 = levels[1]
        p4.retain_grad()
        captured_features.append(levels)

    hook = candidate.model[-1].register_forward_pre_hook(retain_detect_inputs)
    candidate.train().zero_grad(set_to_none=True)
    with torch.no_grad():
        adapter.gamma.fill_(0.05)
    loss, components = candidate.loss(batch)
    if not torch.isfinite(loss).all() or not torch.isfinite(components).all():
        raise AssertionError("Non-finite real detection loss")
    loss.sum().backward()
    hook.remove()
    gradients = {
        "gamma": grad_l1(adapter.gamma, "gamma"),
        "scale_head": grad_l1(adapter.scale_head.weight, "scale_head"),
        "offset": grad_l1(adapter.offset.weight, "offset"),
        "residual_fuse": grad_l1(adapter.fuse.conv.weight, "residual_fuse"),
    }
    if args.variant == "mgsa":
        gradients["scale_metric"] = grad_l1(adapter.scale_metric.weight, "scale_metric")
    if not captured_features or captured_features[-1][1].grad is None or not torch.isfinite(captured_features[-1][1].grad).all():
        raise AssertionError("Missing/non-finite shared P4 feature gradient")
    gradients["shared_p4"] = float(captured_features[-1][1].grad.detach().float().abs().sum().cpu())
    if gradients["shared_p4"] == 0.0:
        raise AssertionError("Shared P4 feature gradient is zero")

    amp_report: dict[str, Any]
    if device.type == "cuda":
        candidate.zero_grad(set_to_none=True)
        candidate.criterion = None
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            amp_loss, amp_components = candidate.loss(batch)
        if not torch.isfinite(amp_loss).all() or not torch.isfinite(amp_components).all():
            raise AssertionError("Non-finite AMP loss")
        amp_loss.sum().backward()
        if not all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in candidate.parameters()):
            raise AssertionError("Non-finite AMP gradients")
        amp_report = {"status": "PASS", "loss": float(amp_loss.detach().sum().cpu())}
    else:
        amp_report = {"status": "SKIPPED_NO_CUDA"}

    candidate.eval()
    with torch.no_grad():
        adapter.gamma.zero_()
    gamma_zero = copy.deepcopy(candidate).eval()
    pruned = copy.deepcopy(gamma_zero).eval()
    _, removed_keys = prune_model(pruned, args.variant)
    if type(pruned.model[-1]) is not Detect:
        raise AssertionError("Physical pruning did not restore native Detect")
    with torch.inference_mode():
        pruning_error = output_error(gamma_zero(sample), pruned(sample))
    if pruning_error["max_abs"] != 0.0:
        raise AssertionError(f"Gamma0 vs native-pruned output differs: {pruning_error}")

    baseline_params = sum(parameter.numel() for parameter in baseline.parameters())
    candidate_params = sum(parameter.numel() for parameter in candidate.parameters())
    pruned_params = sum(parameter.numel() for parameter in pruned.parameters())
    if pruned_params != baseline_params:
        raise AssertionError(f"Pruned params {pruned_params} != B0 {baseline_params}")
    baseline_flops = float(get_flops(baseline, args.imgsz) or get_flops_with_torch_profiler(baseline, args.imgsz))
    candidate_flops = float(get_flops(candidate, args.imgsz) or get_flops_with_torch_profiler(candidate, args.imgsz))
    pruned_flops = float(get_flops(pruned, args.imgsz) or get_flops_with_torch_profiler(pruned, args.imgsz))
    if pruned_flops != baseline_flops:
        raise AssertionError(f"Pruned GFLOPs {pruned_flops} != B0 {baseline_flops}")

    latency = paired_latency(
        fused_baseline,
        fused_candidate,
        sample,
        args.latency_warmup,
        args.latency_repeats,
    )
    peak_vram = None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        with torch.inference_mode():
            candidate(sample)
        torch.cuda.synchronize(device)
        peak_vram = int(torch.cuda.max_memory_allocated(device))

    onnx_report: dict[str, Any] = {"full_adaptive": "NOT_REQUESTED", "pruned_native": "NOT_REQUESTED"}
    if args.onnx:
        full_path = report_dir / f"{args.variant}_rs_full_adaptive.onnx"
        pruned_path = report_dir / f"{args.variant}_rs_pruned_native.onnx"
        try:
            onnx_report["full_adaptive"] = export_onnx(candidate, sample, full_path)
        except Exception as error:  # Full grid_sample export is diagnostic-only.
            onnx_report["full_adaptive"] = {"status": "NON_BLOCKING_FAIL", "error": repr(error)}
        onnx_report["pruned_native"] = export_onnx(pruned, sample, pruned_path)

    with torch.no_grad():
        adapter.gamma.zero_()
    report = {
        "variant": args.variant,
        "model": str(spec["yaml"]),
        "weights": str(args.weights.expanduser().resolve()),
        "data": str(args.data.expanduser().resolve()),
        "test_sealed": True,
        "train_batch_files": list(batch["im_file"]),
        "train_batch_objects": int(len(batch["cls"])),
        "train_batch_classes": sorted({int(value) for value in batch["cls"].flatten()}),
        "transfer": transfer,
        "strides": candidate.stride.tolist(),
        "detect_input_shapes": [list(feature.shape) for feature in captured_features[-1]],
        "initial_output_max_abs_error": float((candidate_output - b0_output).abs().max().cpu()),
        "initial_raw_o2m_o2o_equivalence": initial_raw_error,
        "fused_initial_equivalence": fused_error,
        "initial_scale": {
            "scale_h_max_abs_from_one": float((initial_diagnostics["scale_h"] - 1).abs().max().cpu()),
            "scale_v_max_abs_from_one": float((initial_diagnostics["scale_v"] - 1).abs().max().cpu()),
        },
        "geometry": geometry,
        "batch_isolation_max_abs_error": batch_isolation_error,
        "mg_special_case": mg_audit,
        "real_detection_loss": float(loss.detach().sum().cpu()),
        "loss_components": [float(value) for value in components.detach().flatten().cpu()],
        "gradient_l1": gradients,
        "amp": amp_report,
        "parameters": {
            "b0": baseline_params,
            "candidate": candidate_params,
            "delta": candidate_params / baseline_params - 1.0,
            "pruned_native": pruned_params,
        },
        "gflops_640": {
            "b0": baseline_flops,
            "candidate": candidate_flops,
            "delta": candidate_flops / baseline_flops - 1.0,
            "pruned_native": pruned_flops,
        },
        "paired_fused_latency_ms": latency,
        "peak_vram_bytes": peak_vram,
        "physical_pruning": {
            "head_class": type(pruned.model[-1]).__name__,
            "removed_state_items": len(removed_keys),
            "equivalence": pruning_error,
        },
        "onnx": onnx_report,
        "criterion": type(candidate.init_criterion()).__name__,
        "verdict": "READY_FOR_SA_RS_1E" if args.variant == "sa" else "STATIC_PASS_BUT_MG_LOCKED",
        "formal_training_started": False,
        "ok": True,
    }
    json_path = report_dir / f"{args.variant}_rs_static_audit.json"
    md_path = report_dir / f"{args.variant}_rs_static_audit.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(markdown_report(report), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
