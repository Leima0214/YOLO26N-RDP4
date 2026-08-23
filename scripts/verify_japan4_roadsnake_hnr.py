"""Static, real-batch, AMP, gradient-scope, and gain-calibration audit for RoadSnake-HNR."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ultralytics import YOLO  # noqa: E402
from ultralytics.cfg import DEFAULT_CFG, get_cfg  # noqa: E402
from ultralytics.data.build import build_dataloader, build_yolo_dataset  # noqa: E402
from ultralytics.data.utils import check_det_dataset  # noqa: E402
from ultralytics.models.yolo.detect.train import DetectionTrainer  # noqa: E402
from ultralytics.nn.roadsnake import RoadSnakeDetect, RoadSnakeHNRDetect  # noqa: E402
from ultralytics.nn.tasks import DetectionModel  # noqa: E402
from ultralytics.utils.roadsnake_hnr_loss import RoadSnakeHNRE2ELoss  # noqa: E402
from ultralytics.utils.torch_utils import autocast, init_seeds  # noqa: E402

R1 = ROOT / "ultralytics/cfg/models/26/yolo26n-japan4-roadsnake-r1.yaml"
HNR = ROOT / "ultralytics/cfg/models/26/yolo26n-japan4-roadsnake-hnr.yaml"


def make_loader(data_yaml: Path, imgsz: int, batch_size: int):
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
    return build_dataloader(dataset, batch_size, 0, shuffle=False, rank=-1, pin_memory=False)


def prepare_batch(batch: dict, device: torch.device) -> dict:
    batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
    batch["img"] = batch["img"].float().div(255.0)
    return batch


def build_model(cfg: Path, checkpoint: DetectionModel, device: torch.device) -> DetectionModel:
    # Reset before each construction so the incompatible 80-class -> 4-class
    # output tensors receive the same fair initialization in R1 and HNR.
    init_seeds(42, deterministic=True)
    model = DetectionModel(str(cfg), nc=4, ch=3, verbose=False).float()
    model.load(checkpoint, verbose=False)
    model.args = get_cfg(DEFAULT_CFG, {"box": 7.5, "cls": 0.5, "dfl": 1.5, "epochs": 30})
    model.criterion = None
    return model.to(device)


def gradient_norm(loss: torch.Tensor, parameters: list[torch.nn.Parameter], retain_graph: bool) -> float:
    gradients = torch.autograd.grad(loss, parameters, retain_graph=retain_graph, allow_unused=True)
    total = sum(float(gradient.detach().float().square().sum().cpu()) for gradient in gradients if gradient is not None)
    return math.sqrt(total)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=ROOT / "yolo26n.pt")
    parser.add_argument("--data", type=Path, default=ROOT / "configs/japan4_clean_v3_remote.yaml")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--calibration-batches", type=int, default=8)
    parser.add_argument("--target-gradient-ratio", type=float, default=0.03)
    parser.add_argument("--report", type=Path, default=ROOT / "reports/japan4_roadsnake_hnr_preflight.json")
    args = parser.parse_args()
    for path in (args.weights, args.data, R1, HNR):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not 0 < args.target_gradient_ratio < 0.1:
        raise ValueError("target-gradient-ratio must be in (0, 0.1)")

    init_seeds(42, deterministic=True)
    device = torch.device(args.device)
    checkpoint = YOLO(str(args.weights), task="detect", verbose=False).model.float()
    r1 = build_model(R1, checkpoint, device)
    hnr = build_model(HNR, checkpoint, device)
    if not isinstance(r1.model[-1], RoadSnakeDetect) or not isinstance(hnr.model[-1], RoadSnakeHNRDetect):
        raise AssertionError("unexpected RoadSnake head type")

    r1_state, hnr_state = r1.state_dict(), hnr.state_dict()
    if r1_state.keys() != hnr_state.keys():
        raise AssertionError("HNR changed the RoadSnake-R1 state-dict contract")
    changed = [name for name in r1_state if not torch.equal(r1_state[name], hnr_state[name])]
    if changed:
        raise AssertionError(f"HNR changed inherited tensors: {changed[:8]}")
    if sum(parameter.numel() for parameter in r1.parameters()) != sum(parameter.numel() for parameter in hnr.parameters()):
        raise AssertionError("HNR must add zero model parameters")

    sample = torch.randn(1, 3, args.imgsz, args.imgsz, device=device)
    r1.eval()
    hnr.eval()
    with torch.inference_mode():
        r1_output = r1(sample)
        hnr_output = hnr(sample)
    r1_prediction = r1_output[0] if isinstance(r1_output, tuple) else r1_output
    hnr_prediction = hnr_output[0] if isinstance(hnr_output, tuple) else hnr_output
    torch.testing.assert_close(hnr_prediction, r1_prediction, atol=0, rtol=0)

    trainer = object.__new__(DetectionTrainer)
    trainer.data = {"nc": 4, "channels": 3}
    rebuilt = DetectionTrainer.get_model(trainer, cfg=str(HNR), weights=hnr, verbose=False).float().to(device)
    rebuild_changed = [
        name for name, value in hnr.state_dict().items() if not torch.equal(value, rebuilt.state_dict()[name])
    ]
    if rebuild_changed:
        raise AssertionError(f"Trainer reconstruction changed HNR tensors: {rebuild_changed[:8]}")
    rebuilt.args = hnr.args
    rebuilt.criterion = None

    loader = make_loader(args.data.resolve(), args.imgsz, args.batch)
    ratios = []
    batch_stats = []
    amp_components = None
    scope_batch = None
    configured_gain = rebuilt.model[-1].hnr_loss_gain
    if configured_gain <= 0:
        raise AssertionError(f"HNR calibration requires a positive configured gain, got {configured_gain}")
    for batch_index, raw_batch in enumerate(loader):
        if len(ratios) >= args.calibration_batches or batch_index >= args.calibration_batches * 8:
            break
        batch = prepare_batch(raw_batch, device)
        rebuilt.train().zero_grad(set_to_none=True)
        with autocast(enabled=device.type == "cuda"):
            predictions = rebuilt(batch["img"])
            losses, components = rebuilt.loss(batch, predictions)
        criterion = rebuilt.criterion
        if not isinstance(criterion, RoadSnakeHNRE2ELoss):
            raise AssertionError(f"unexpected criterion: {type(criterion).__name__}")
        if not torch.isfinite(losses).all() or not torch.isfinite(components).all():
            raise AssertionError("non-finite HNR loss")
        if criterion.last_stats["ranking_pairs"] <= 0:
            batch_stats.append({"batch": batch_index, **criterion.last_stats, "skipped": "no_small_positive_pairs"})
            continue

        parameters = [parameter for parameter in rebuilt.model[-1].one2one_cv3[0].parameters() if parameter.requires_grad]
        detection_norm = gradient_norm(losses[:3].sum(), parameters, retain_graph=True)
        hnr_norm = gradient_norm(losses[3], parameters, retain_graph=False)
        if not math.isfinite(detection_norm) or not math.isfinite(hnr_norm) or detection_norm <= 0 or hnr_norm <= 0:
            raise AssertionError(f"invalid gradient norms: detection={detection_norm}, hnr={hnr_norm}")
        # losses[3] already contains the YAML gain. Divide it out so the
        # recommendation remains an absolute gain and the audit is repeatable
        # both before and after the calibrated value is locked.
        ratio = (hnr_norm / configured_gain) / detection_norm
        ratios.append(ratio)
        batch_stats.append({"batch": batch_index, **criterion.last_stats, "raw_gradient_ratio": ratio})
        amp_components = [float(value) for value in components.detach().cpu()]
        scope_batch = batch

    if len(ratios) != args.calibration_batches:
        raise AssertionError(f"expected {args.calibration_batches} calibration batches, got {len(ratios)}")
    median_ratio = statistics.median(ratios)
    recommended_gain = args.target_gradient_ratio / median_ratio
    scaled_ratios = [ratio * recommended_gain for ratio in ratios]
    if not 0 < recommended_gain < 1:
        raise AssertionError(f"calibrated gain is implausible: {recommended_gain}")
    if statistics.median(scaled_ratios) > 0.05001 or max(scaled_ratios) > 0.10:
        raise AssertionError(f"calibrated gradient ratios exceed the safety envelope: {scaled_ratios}")

    # Prove raw HNR cannot update box heads, RoadSnake, backbone, or non-P3 O2O classification heads.
    if scope_batch is None:
        raise AssertionError("no HNR-active batch was retained for the gradient-scope audit")
    batch = scope_batch
    rebuilt.train().zero_grad(set_to_none=True)
    predictions = rebuilt(batch["img"])
    _ = rebuilt.loss(batch, predictions)
    raw_hnr = rebuilt.criterion.last_raw_hnr
    named_parameters = [(name, parameter) for name, parameter in rebuilt.named_parameters() if parameter.requires_grad]
    gradients = torch.autograd.grad(raw_hnr, [parameter for _, parameter in named_parameters], allow_unused=True)
    active = [name for (name, _), gradient in zip(named_parameters, gradients) if gradient is not None and gradient.abs().sum() > 0]
    illegal = [name for name in active if ".one2one_cv3.0." not in name]
    if illegal or not active:
        raise AssertionError(f"HNR gradient scope violation: active={active[:12]} illegal={illegal[:12]}")

    report = {
        "model": str(HNR),
        "weights": str(args.weights.resolve()),
        "data": str(args.data.resolve()),
        "device": str(device),
        "state_items_identical_to_r1": len(hnr_state),
        "trainer_rebuild_changed": len(rebuild_changed),
        "initial_output_max_abs_error": float((hnr_prediction - r1_prediction).abs().max().cpu()),
        "parameters": sum(parameter.numel() for parameter in hnr.parameters()),
        "configured_hnr_loss_gain": configured_gain,
        "amp_components_configured_gain": amp_components,
        "calibration_batches": batch_stats,
        "raw_gradient_ratio_median": median_ratio,
        "target_gradient_ratio": args.target_gradient_ratio,
        "recommended_hnr_loss_gain": recommended_gain,
        "scaled_gradient_ratios": scaled_ratios,
        "hnr_gradient_parameters": active,
        "verdict": "STATIC_GO_REPLACE_GAIN_BEFORE_TRAINING",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
