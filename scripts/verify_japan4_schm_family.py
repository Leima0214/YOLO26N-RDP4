#!/usr/bin/env python3
"""Static and real-train-batch audit for SCHM and RS-SCHM."""

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
from ultralytics.models.yolo.detect.schm_train import SCHMDetectionTrainer  # noqa: E402
from ultralytics.nn.modules.head import Detect  # noqa: E402
from ultralytics.nn.roadsnake import RoadSnakeO2MDetect  # noqa: E402
from ultralytics.nn.tasks import DetectionModel  # noqa: E402
from ultralytics.utils.schm_loss import SCHME2ELoss  # noqa: E402
from ultralytics.utils.torch_utils import get_flops, get_flops_with_torch_profiler  # noqa: E402


BASELINE = ROOT / "ultralytics/cfg/models/26/yolo26.yaml"
MODELS = {
    "schm": ROOT / "ultralytics/cfg/models/26/yolo26n-japan4-schm.yaml",
    "rs-schm": ROOT / "ultralytics/cfg/models/26/yolo26n-japan4-rs-schm.yaml",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=tuple(MODELS), required=True)
    parser.add_argument("--weights", type=Path, default=ROOT / "yolo26n.pt")
    parser.add_argument(
        "--data", type=Path, default=ROOT / "configs/japan4_clean_v3_remote.yaml"
    )
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--onnx", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def deployed(output):
    return output[0] if isinstance(output, tuple) else output


def raw_one2one(output):
    if not isinstance(output, tuple) or not isinstance(output[1], dict):
        raise TypeError(
            "Expected eval output tuple containing raw end-to-end predictions"
        )
    return output[1]["one2one"]


def real_batch(data_yaml: Path, imgsz: int, batch_size: int):
    data = check_det_dataset(str(data_yaml.resolve()))
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
            "box": 7.5,
            "cls": 0.5,
            "dfl": 1.5,
            "epochs": 100,
        },
    )
    dataset = build_yolo_dataset(
        cfg, data["train"], batch_size, data, mode="train", rect=False, stride=32
    )
    loader = build_dataloader(
        dataset, batch_size, 0, shuffle=False, rank=-1, pin_memory=False
    )
    return cfg, next(iter(loader))


def build(variant: str, weights: Path, device: torch.device):
    checkpoint = YOLO(
        str(weights.resolve()), task="detect", verbose=False
    ).model.float()
    baseline = DetectionModel(str(BASELINE), nc=4, ch=3, verbose=False).float()
    baseline.load(checkpoint, verbose=False)
    candidate = DetectionModel(str(MODELS[variant]), nc=4, ch=3, verbose=False).float()

    source = baseline.state_dict()
    target = candidate.state_dict()
    missing = [
        key
        for key, value in source.items()
        if key not in target or target[key].shape != value.shape
    ]
    if missing:
        raise AssertionError(f"Candidate cannot inherit B0 tensors: {missing[:10]}")
    candidate.load(baseline, verbose=False)
    changed = [
        key
        for key, value in source.items()
        if not torch.equal(value, candidate.state_dict()[key])
    ]
    if changed:
        raise AssertionError(f"Candidate changed inherited B0 tensors: {changed[:10]}")

    trainer = object.__new__(SCHMDetectionTrainer)
    trainer.data = {"nc": 4, "channels": 3}
    rebuilt = SCHMDetectionTrainer.get_model(
        trainer, cfg=str(MODELS[variant]), weights=candidate, verbose=False
    ).float()
    rebuilt_state = rebuilt.state_dict()
    rebuild_changed = [
        key
        for key, value in candidate.state_dict().items()
        if not torch.equal(value, rebuilt_state[key])
    ]
    if rebuild_changed:
        raise AssertionError(
            f"Trainer reconstruction changed initialized tensors: {rebuild_changed[:10]}"
        )
    audit = {
        "b0_state_items": len(source),
        "candidate_state_items": len(target),
        "shared_missing_or_mismatched": len(missing),
        "shared_changed_after_load": len(changed),
        "trainer_rebuild_changed": len(rebuild_changed),
        "new_state_items": sorted(set(target) - set(source)),
        "b0_parameter_inheritance": 1.0,
    }
    return baseline.to(device), rebuilt.to(device), audit


def mapping_audit(shapes: list[tuple[int, int]]) -> dict:
    total = sum(height * width for height, width in shapes)
    rng = random.Random(20260818)
    indices = rng.sample(range(total), min(100, total))
    samples = []
    for index in indices:
        level, y, x = SCHME2ELoss.global_to_level_yx(index, shapes)
        roundtrip = SCHME2ELoss.level_yx_to_global(level, y, x, shapes)
        if roundtrip != index:
            raise AssertionError((index, level, y, x, roundtrip))
        samples.append(
            {
                "global_index": index,
                "scale_id": level,
                "y": y,
                "x": x,
                "roundtrip": roundtrip,
            }
        )
    return {
        "flatten_order": "P3 row-major y,x then P4 row-major y,x then P5 row-major y,x",
        "shapes": shapes,
        "offsets": [0, shapes[0][0] * shapes[0][1], sum(h * w for h, w in shapes[:2])],
        "random_roundtrip_checks": len(samples),
        "all_passed": True,
        "samples": samples,
    }


def finite_gradients(model: torch.nn.Module) -> tuple[int, int]:
    present = finite = 0
    for parameter in model.parameters():
        if parameter.grad is not None:
            present += 1
            finite += int(torch.isfinite(parameter.grad).all())
    return present, finite


def main() -> None:
    args = parse_args()
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    device = torch.device(
        f"cuda:{args.device}" if args.device.isdigit() else args.device
    )
    baseline, candidate, transfer = build(args.variant, args.weights, device)
    head = candidate.model[-1]
    if args.variant == "schm" and type(head) is not Detect:
        raise AssertionError(
            f"SCHM must retain native Detect, got {type(head).__name__}"
        )
    if args.variant == "rs-schm" and not isinstance(head, RoadSnakeO2MDetect):
        raise AssertionError(type(head).__name__)
    if args.variant == "rs-schm" and float(head.road_snake.gamma) != 0.0:
        raise AssertionError("RS-SCHM gamma must initialize at zero")
    if (
        candidate.stride.tolist() != baseline.stride.tolist()
        or candidate.stride.tolist() != [8.0, 16.0, 32.0]
    ):
        raise AssertionError("Detect strides changed")

    sample = torch.randn(2, 3, args.imgsz, args.imgsz, device=device)
    baseline.eval()
    candidate.eval()
    with torch.inference_mode():
        baseline_output = deployed(baseline(sample))
        candidate_output = deployed(candidate(sample))
    max_error = float((candidate_output - baseline_output).abs().max().cpu())
    torch.testing.assert_close(candidate_output, baseline_output, atol=0, rtol=0)
    # Keep the batch shape fixed and replace only the other image. Comparing batch=1 against batch=2 would conflate
    # sample isolation with legitimate cuDNN kernel/rounding changes caused by a different batch shape.
    alternate_sample = sample.clone()
    alternate_sample[1] = torch.randn_like(alternate_sample[1])
    with torch.inference_mode():
        isolated_raw = raw_one2one(candidate(sample))
        batched_raw = raw_one2one(candidate(alternate_sample))
    raw_errors = {
        key: float((isolated_raw[key][:1] - batched_raw[key][:1]).abs().max().cpu())
        for key in ("boxes", "scores")
    }
    for key in raw_errors:
        torch.testing.assert_close(
            isolated_raw[key][:1], batched_raw[key][:1], atol=0, rtol=0
        )
    batch_isolation_error = max(raw_errors.values())

    cfg, batch = real_batch(args.data, args.imgsz, args.batch)
    batch = {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }
    batch["img"] = batch["img"].float().div(255.0)
    candidate.args = cfg
    candidate.criterion = None
    candidate.train()
    candidate.zero_grad(set_to_none=True)
    predictions = candidate(batch["img"])
    total, components = candidate.loss(batch, predictions)
    criterion = candidate.criterion
    if not isinstance(criterion, SCHME2ELoss):
        raise AssertionError(type(criterion).__name__)
    if not torch.isfinite(total).all() or not torch.isfinite(components).all():
        raise AssertionError("Non-finite SCHM-family loss")
    if criterion.last_batch_stats["illegal_harvest_count"] != 0:
        raise AssertionError("Illegal O2M candidate harvested")
    if criterion.last_batch_stats["conflict_count"] != 0:
        raise AssertionError(
            "Unexpected candidate uniqueness conflict after native O2M conflict resolution"
        )

    # The auxiliary graph must update O2O localization but be disconnected from O2M predictions used for selection.
    grad_o2m = torch.autograd.grad(
        criterion.last_schm_raw,
        predictions["one2many"]["boxes"],
        retain_graph=True,
        allow_unused=True,
    )[0]
    grad_o2o = torch.autograd.grad(
        criterion.last_schm_raw,
        predictions["one2one"]["boxes"],
        retain_graph=True,
        allow_unused=True,
    )[0]
    if grad_o2m is not None and float(grad_o2m.abs().max()) != 0.0:
        raise AssertionError("SCHM selection leaked gradient into O2M predictions")
    if criterion.last_batch_stats["harvest_gt_count"] and (
        grad_o2o is None
        or not torch.isfinite(grad_o2o).all()
        or float(grad_o2o.abs().sum()) == 0.0
    ):
        raise AssertionError("SCHM failed to update harvested O2O localization")
    total.sum().backward()
    grad_present, grad_finite = finite_gradients(candidate)
    if grad_present == 0 or grad_present != grad_finite:
        raise AssertionError("Missing/non-finite full-loss gradients")

    rs_audit = None
    if args.variant == "rs-schm":
        gamma_grad = head.road_snake.gamma.grad
        if (
            gamma_grad is None
            or not torch.isfinite(gamma_grad)
            or float(gamma_grad.abs()) == 0.0
        ):
            raise AssertionError(
                "Native O2M loss did not give RoadSnake gamma a finite non-zero gradient"
            )
        captured = {}

        def capture_native(_, inputs):
            captured["features"] = [feature.detach().clone() for feature in inputs[0]]

        hook = head.register_forward_pre_hook(capture_native)
        with torch.no_grad():
            head.road_snake.gamma.fill_(0.05)
            routed = candidate(batch["img"])
            head.road_snake.gamma.zero_()
        hook.remove()
        native_p4 = captured["features"][1]
        o2o_native_error = float(
            (routed["one2one"]["feats"][1] - native_p4).abs().max().cpu()
        )
        o2m_effect = float(
            (routed["one2many"]["feats"][1] - native_p4).abs().mean().cpu()
        )
        if o2o_native_error != 0.0 or o2m_effect == 0.0:
            raise AssertionError("RS-SCHM did not isolate RoadSnake to O2M P4")
        rs_audit = {
            "gamma_gradient_from_native_o2m": float(gamma_grad.detach().cpu()),
            "o2o_native_p4_max_abs_error_at_gamma_0_05": o2o_native_error,
            "o2m_roadsnake_p4_mean_abs_effect_at_gamma_0_05": o2m_effect,
        }

    # AMP forward/loss/backward is independent from the full-precision graph above.
    candidate.zero_grad(set_to_none=True)
    candidate.criterion = None
    with torch.autocast(
        device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"
    ):
        amp_predictions = candidate(batch["img"])
        amp_total, amp_components = candidate.loss(batch, amp_predictions)
    if not torch.isfinite(amp_total).all() or not torch.isfinite(amp_components).all():
        raise AssertionError("AMP produced non-finite loss")
    amp_total.sum().backward()
    amp_present, amp_finite = finite_gradients(candidate)
    if amp_present == 0 or amp_present != amp_finite:
        raise AssertionError("AMP produced missing/non-finite gradients")

    shapes = [
        (int(x.shape[-2]), int(x.shape[-1]))
        for x in amp_predictions["one2one"]["feats"]
    ]
    mapping = mapping_audit(shapes)
    params = sum(parameter.numel() for parameter in candidate.parameters())
    b0_params = sum(parameter.numel() for parameter in baseline.parameters())
    gflops = get_flops(candidate.eval(), args.imgsz) or get_flops_with_torch_profiler(
        candidate.eval(), args.imgsz
    )
    b0_gflops = get_flops(baseline.eval(), args.imgsz) or get_flops_with_torch_profiler(
        baseline.eval(), args.imgsz
    )

    onnx_path = None
    if args.onnx:
        import onnx

        args.onnx.parent.mkdir(parents=True, exist_ok=True)
        # The criterion intentionally retains last-batch tensors for gradient-isolation auditing; it is training-only
        # and must not be copied into the deployment graph.
        candidate.criterion = None
        candidate.zero_grad(set_to_none=True)
        export_model = copy.deepcopy(candidate).eval().fuse(verbose=False)
        export_model.model[-1].export = True
        torch.onnx.export(
            export_model,
            sample[:1],
            args.onnx,
            opset_version=17,
            input_names=["images"],
            output_names=["output0"],
        )
        onnx.checker.check_model(onnx.load(args.onnx))
        onnx_path = str(args.onnx.resolve())

    same_gain_ratio = criterion.last_batch_stats["same_index_gain_count"] / max(
        criterion.last_batch_stats["positive_gain_count"], 1
    )
    # A high same-index ratio is reported as specified; it does not silently relax the v1 new-index-only rule.
    static_go = (
        criterion.last_batch_stats["harvest_gt_count"] > 0
        and criterion.last_batch_stats["illegal_harvest_count"] == 0
        and same_gain_ratio < 0.8
    )
    report = {
        "variant": args.variant,
        "model": str(MODELS[args.variant]),
        "data_split": "train-only static batch",
        "test_sealed": True,
        "transfer": transfer,
        "step0_inference_max_abs_error_vs_b0": max_error,
        "batch_isolation_max_abs_error": batch_isolation_error,
        "loss_components": [float(value) for value in components.detach().cpu()],
        "amp_loss_components": [
            float(value) for value in amp_components.detach().cpu()
        ],
        "full_precision_gradients": {"present": grad_present, "finite": grad_finite},
        "amp_gradients": {"present": amp_present, "finite": amp_finite},
        "schm_gradient_to_o2m_prediction": 0.0
        if grad_o2m is None
        else float(grad_o2m.abs().max()),
        "schm_gradient_to_o2o_prediction_l1": 0.0
        if grad_o2o is None
        else float(grad_o2o.abs().sum()),
        "harvest_initial": criterion.last_batch_stats,
        "same_index_positive_gain_ratio": same_gain_ratio,
        "index_mapping": mapping,
        "rs_schm_routing": rs_audit,
        "parameters": {
            "b0": b0_params,
            "candidate": params,
            "delta": params / b0_params - 1,
        },
        "gflops": {
            "b0": b0_gflops,
            "candidate": gflops,
            "delta": gflops / b0_gflops - 1,
        },
        "onnx": onnx_path,
        "static_go": static_go,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not static_go:
        raise SystemExit(
            "STATIC NO-GO: no legal new-index harvest or same-index gains dominate"
        )


if __name__ == "__main__":
    main()
