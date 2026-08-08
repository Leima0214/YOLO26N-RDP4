"""Training-only Gaussian region supervision for Japan4 G1 and GS1."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from ultralytics.utils.loss import E2ELoss
from ultralytics.utils.torch_utils import autocast


def gaussian_region_targets(
    batch_idx: torch.Tensor,
    boxes_xywh: torch.Tensor,
    batch_size: int,
    height: int,
    width: int,
    sigma_divisor: float = 6.0,
) -> torch.Tensor:
    """Rasterize normalized xywh boxes as max-composed anisotropic Gaussian maps."""
    if sigma_divisor <= 0:
        raise ValueError(f"sigma_divisor must be positive, got {sigma_divisor}")
    device = boxes_xywh.device
    targets = torch.zeros((batch_size, 1, height, width), device=device, dtype=torch.float32)
    if boxes_xywh.numel() == 0:
        return targets

    grid_y = torch.arange(height, device=device, dtype=torch.float32).view(height, 1) + 0.5
    grid_x = torch.arange(width, device=device, dtype=torch.float32).view(1, width) + 0.5
    boxes = boxes_xywh.float()
    for image_index, box in zip(batch_idx.long(), boxes):
        cx, cy = box[0] * width, box[1] * height
        box_width, box_height = box[2] * width, box[3] * height
        sigma_x = (box_width / sigma_divisor).clamp_min(1.0)
        sigma_y = (box_height / sigma_divisor).clamp_min(1.0)
        gaussian = torch.exp(-0.5 * (((grid_x - cx) / sigma_x).square() + ((grid_y - cy) / sigma_y).square()))
        targets[image_index, 0] = torch.maximum(targets[image_index, 0], gaussian)
    return targets


class RegionGuidedE2ELoss(E2ELoss):
    """Standard E2E detection loss plus one fixed soft-BCE G1 objective."""

    def __init__(self, model):
        super().__init__(model)
        yaml = model.yaml
        self.lambda_p3 = float(yaml.get("region_lambda_p3", 0.05))
        self.lambda_p4 = float(yaml.get("region_lambda_p4", 0.05))
        self.sigma_divisor = float(yaml.get("region_sigma_divisor", 6.0))
        self.loss_type = yaml.get("region_loss_type", "soft_bce")
        if self.loss_type != "soft_bce":
            raise ValueError(f"Unsupported region_loss_type: {self.loss_type}")
        if min(self.lambda_p3, self.lambda_p4) < 0:
            raise ValueError("Region loss weights must be non-negative")
        self.last_region_losses = (0.0, 0.0)

    def region_loss(self, logits: list[torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, ...]:
        """Compute low-background-weight soft BCE at P3 and P4."""
        losses = []
        for logit in logits:
            target = gaussian_region_targets(
                batch["batch_idx"],
                batch["bboxes"],
                logit.shape[0],
                logit.shape[-2],
                logit.shape[-1],
                self.sigma_divisor,
            )
            with autocast(enabled=False):
                per_pixel = F.binary_cross_entropy_with_logits(logit.float(), target, reduction="none")
                # GT borders and possible unlabeled damage carry only weak negative supervision.
                weight = 0.1 + 0.9 * target
                losses.append((per_pixel * weight).mean())
        return tuple(losses)

    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Add G1 only when training logits are present; validation remains detection-only."""
        predictions = preds[1] if isinstance(preds, tuple) else preds
        detection_preds = {key: predictions[key] for key in ("one2many", "one2one")}
        detection_loss, detection_items = super().__call__(detection_preds, batch)
        logits = predictions.get("region_logits")
        if logits is None:
            weighted_p3 = weighted_p4 = detection_loss.sum() * 0.0
            self.last_region_losses = (0.0, 0.0)
        else:
            p3, p4 = self.region_loss(logits, batch)
            weighted_p3, weighted_p4 = self.lambda_p3 * p3, self.lambda_p4 * p4
            self.last_region_losses = (float(p3.detach()), float(p4.detach()))
        batch_size = detection_preds["one2one"]["boxes"].shape[0]
        return (
            torch.cat(
                (detection_loss, (weighted_p3 * batch_size).reshape(1), (weighted_p4 * batch_size).reshape(1))
            ),
            torch.cat((detection_items, weighted_p3.detach().reshape(1), weighted_p4.detach().reshape(1))),
        )
