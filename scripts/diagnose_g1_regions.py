"""Visualize G1 maps and calibrate its fixed loss weight from one real train batch."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ultralytics import YOLO  # noqa: E402
from ultralytics.cfg import get_cfg  # noqa: E402
from ultralytics.data import build_dataloader, build_yolo_dataset  # noqa: E402
from ultralytics.data.utils import check_det_dataset  # noqa: E402
from ultralytics.utils.loss import E2ELoss  # noqa: E402
from ultralytics.utils.region_loss import gaussian_region_targets  # noqa: E402

MODELS = {
    "g1": ROOT / "ultralytics/cfg/models/26/yolo26n-japan4-g1-region-guidance.yaml",
    "gs1": ROOT / "ultralytics/cfg/models/26/yolo26n-japan4-gs1-region-strip.yaml",
}


def grad_norm(features: list[torch.Tensor]) -> float:
    """Joint L2 norm across retained P3/P4 feature gradients."""
    return sum(float(feature.grad.detach().float().square().sum().cpu()) for feature in features) ** 0.5


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", choices=tuple(MODELS), default="g1")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--weights", type=Path, default=ROOT / "yolo26n.pt")
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    device = torch.device(f"cuda:{args.device}" if args.device.isdigit() else args.device)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    cfg = get_cfg(
        overrides={
            "imgsz": args.imgsz,
            "batch": args.batch,
            "workers": args.workers,
            "data": str(args.data),
            "seed": 42,
            "deterministic": True,
        }
    )
    cfg.epochs = 30
    data = check_det_dataset(str(args.data.resolve()))
    dataset = build_yolo_dataset(cfg, data["train"], args.batch, data, mode="train", stride=32)
    loader = build_dataloader(dataset, args.batch, args.workers, shuffle=True, rank=-1)
    batch = next(iter(loader))
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.to(device, non_blocking=True)
    batch["img"] = batch["img"].float() / 255

    model = YOLO(str(MODELS[args.candidate]), task="detect").load(str(args.weights)).model.to(device)
    model.args = cfg
    model.train()
    head = model.model[-1]
    shared_features = []

    def retain_shared_features(_, inputs):
        shared_features.extend(inputs[0][:2])
        for feature in shared_features:
            feature.retain_grad()

    hook = head.register_forward_pre_hook(retain_shared_features)
    predictions = model(batch["img"])
    hook.remove()
    detection = E2ELoss(model)
    detection_loss, _ = detection(
        {key: predictions[key] for key in ("one2many", "one2one")}, batch
    )
    detection_loss.sum().backward(retain_graph=True)
    detection_gradient = grad_norm(shared_features)

    model.zero_grad(set_to_none=True)
    for feature in shared_features:
        feature.grad = None
    criterion = model.init_criterion()
    p3_loss, p4_loss = criterion.region_loss(predictions["region_logits"], batch)
    weighted_region = (criterion.lambda_p3 * p3_loss + criterion.lambda_p4 * p4_loss) * batch["img"].shape[0]
    weighted_region.backward()
    region_gradient = grad_norm(shared_features)
    ratio = region_gradient / max(detection_gradient, 1e-12)
    # The smallest accepted weight targets the lower edge (5%), independent of validation AP.
    factor = 0.05 / max(ratio, 1e-12)
    recommended = {
        "region_lambda_p3": criterion.lambda_p3 * factor,
        "region_lambda_p4": criterion.lambda_p4 * factor,
    }

    logits = predictions["region_logits"]
    targets = [
        gaussian_region_targets(
            batch["batch_idx"], batch["bboxes"], logit.shape[0], logit.shape[-2], logit.shape[-1], criterion.sigma_divisor
        )
        for logit in logits
    ]
    figure, axes = plt.subplots(2, 2, figsize=(10, 8))
    for row, (logit, target, level) in enumerate(zip(logits, targets, ("P3", "P4"))):
        axes[row, 0].imshow(target[0, 0].detach().cpu(), cmap="magma", vmin=0, vmax=1)
        axes[row, 0].set_title(f"{level} GT Gaussian")
        axes[row, 1].imshow(logit[0, 0].sigmoid().detach().cpu(), cmap="magma", vmin=0, vmax=1)
        axes[row, 1].set_title(f"{level} predicted region")
        for axis in axes[row]:
            axis.axis("off")
    figure.tight_layout()
    figure.savefig(args.output / "g1_region_maps.png", dpi=160)
    plt.close(figure)

    response = {}
    for level, logit, target in zip(("P3", "P4"), logits, targets):
        probability = logit.sigmoid().detach()
        foreground = target > 0.1
        image_has_gt = torch.bincount(batch["batch_idx"].long(), minlength=logit.shape[0]) > 0
        response[level] = {
            "target_mean": float(target.mean().cpu()),
            "prediction_mean": float(probability.mean().cpu()),
            "foreground_prediction_mean": float(probability[foreground].mean().cpu()) if foreground.any() else None,
            "background_prediction_mean": float(probability[~foreground].mean().cpu()),
            "empty_gt_prediction_mean": (
                float(probability[~image_has_gt].mean().cpu()) if (~image_has_gt).any() else None
            ),
        }

    report = {
        "candidate": args.candidate,
        "weights": str(args.weights.resolve()),
        "seed": 42,
        "batch_images": int(batch["img"].shape[0]),
        "batch_targets": int(batch["bboxes"].shape[0]),
        "detection_loss_sum": float(detection_loss.detach().sum()),
        "unscaled_region_loss": {"p3": float(p3_loss.detach()), "p4": float(p4_loss.detach())},
        "shared_feature_gradient_l2": {"detection": detection_gradient, "weighted_region": region_gradient},
        "region_to_detection_gradient_ratio": ratio,
        "accepted_range": [0.05, 0.15],
        "current_lambda": {"p3": criterion.lambda_p3, "p4": criterion.lambda_p4},
        "recommended_lambda_without_val_ap": recommended,
        "response": response,
    }
    (args.output / "g1_gradient_calibration.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
