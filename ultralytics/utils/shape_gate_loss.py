"""Positive-only shape supervision for the Japan4 S2 strip gates."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F

from ultralytics.utils.loss import E2ELoss
from ultralytics.utils.torch_utils import autocast


def shape_gate_targets(boxes_xyxy: torch.Tensor) -> torch.Tensor:
    """Return continuous horizontal and vertical targets from assigned box shape."""
    width, height = (boxes_xyxy[..., 2:] - boxes_xyxy[..., :2]).clamp_min(1e-6).unbind(-1)
    aspect = torch.log(width / height)
    magnitude = torch.sigmoid((aspect.abs() - math.log(1.5)) / 0.25)
    return torch.stack(
        (magnitude * torch.sigmoid(aspect / 0.25), magnitude * torch.sigmoid(-aspect / 0.25)), dim=-1
    )


class ShapeGateE2ELoss(E2ELoss):
    """Native E2E detection plus matched-positive gate BCE for both O2M and O2O."""

    def __init__(self, model):
        super().__init__(model)
        self.gate_gain = float(model.yaml.get("gate_loss_gain", 1.0))
        if self.gate_gain < 0:
            raise ValueError("gate_loss_gain must be non-negative")

    @staticmethod
    def gate_loss(preds: dict[str, torch.Tensor], assigned: tuple) -> torch.Tensor:
        """Supervise only assigned P3/P4 positives; P5 and unmatched anchors are ignored."""
        logits_by_scale = preds.get("gate_logits")
        if not logits_by_scale:
            return preds["boxes"].sum() * 0.0
        logits = torch.cat([value.flatten(2) for value in logits_by_scale], dim=2).permute(0, 2, 1)
        fg_mask, _, target_bboxes, _, _ = assigned
        positive = fg_mask[:, : logits.shape[1]]
        if not positive.any():
            return logits.sum() * 0.0
        targets = shape_gate_targets(target_bboxes[:, : logits.shape[1]][positive]).detach()
        with autocast(enabled=False):
            return F.binary_cross_entropy_with_logits(logits[positive].float(), targets.float())

    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Reuse the two native assigners and append one schedule-matched gate loss."""
        predictions = self.one2many.parse_output(preds)
        one2many, one2one = predictions["one2many"], predictions["one2one"]
        assigned_many, loss_many, _ = self.one2many.get_assigned_targets_and_loss(one2many, batch)
        assigned_one, loss_one, detached_one = self.one2one.get_assigned_targets_and_loss(one2one, batch)
        batch_size = one2one["boxes"].shape[0]
        gate = self.gate_loss(one2many, assigned_many) * self.o2m + self.gate_loss(one2one, assigned_one) * self.o2o
        detection = loss_many * batch_size * self.o2m + loss_one * batch_size * self.o2o
        return (
            torch.cat((detection, (gate * batch_size * self.gate_gain).reshape(1))),
            torch.cat((detached_one, (gate.detach() * self.gate_gain).reshape(1))),
        )
