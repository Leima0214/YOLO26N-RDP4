"""Gradient-balanced P3 region guidance for RoadSnake-GBRG."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from ultralytics.utils.loss import E2ELoss
from ultralytics.utils.torch_utils import autocast


def gbrg_region_targets(
    batch_idx: torch.Tensor,
    boxes_xywh: torch.Tensor,
    batch_size: int,
    height: int,
    width: int,
    sigma_divisor: float = 6.0,
    ignore_dilation: float = 1.25,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build Gaussian targets, in-box supervision, and conservative clear background."""
    if sigma_divisor <= 0:
        raise ValueError(f"sigma_divisor must be positive, got {sigma_divisor}")
    if ignore_dilation < 1:
        raise ValueError(f"ignore_dilation must be >= 1, got {ignore_dilation}")

    device = boxes_xywh.device
    targets = torch.zeros((batch_size, 1, height, width), device=device, dtype=torch.float32)
    inside = torch.zeros_like(targets, dtype=torch.bool)
    protected = torch.zeros_like(targets, dtype=torch.bool)
    if boxes_xywh.numel() == 0:
        return targets, inside, ~protected

    grid_y = torch.arange(height, device=device, dtype=torch.float32).view(height, 1) + 0.5
    grid_x = torch.arange(width, device=device, dtype=torch.float32).view(1, width) + 0.5
    for image_index, box in zip(batch_idx.long(), boxes_xywh.float()):
        cx, cy = box[0] * width, box[1] * height
        box_width = (box[2] * width).clamp_min(1.0)
        box_height = (box[3] * height).clamp_min(1.0)
        sigma_x = (box_width / sigma_divisor).clamp_min(1.0)
        sigma_y = (box_height / sigma_divisor).clamp_min(1.0)
        gaussian = torch.exp(-0.5 * (((grid_x - cx) / sigma_x).square() + ((grid_y - cy) / sigma_y).square()))

        in_box = (grid_x >= cx - box_width / 2) & (grid_x <= cx + box_width / 2)
        in_box = in_box & (grid_y >= cy - box_height / 2) & (grid_y <= cy + box_height / 2)
        dilated_width, dilated_height = box_width * ignore_dilation, box_height * ignore_dilation
        in_protected = (grid_x >= cx - dilated_width / 2) & (grid_x <= cx + dilated_width / 2)
        in_protected = in_protected & (grid_y >= cy - dilated_height / 2) & (grid_y <= cy + dilated_height / 2)

        targets[image_index, 0] = torch.maximum(targets[image_index, 0], gaussian * in_box)
        inside[image_index, 0] |= in_box
        protected[image_index, 0] |= in_protected
    return targets, inside, ~protected


class RoadSnakeGBRGE2ELoss(E2ELoss):
    """Standard E2E loss plus P3 region guidance held at a fixed gradient ratio."""

    def __init__(self, model):
        super().__init__(model)
        yaml = model.yaml
        self.target_ratio = float(yaml.get("gbrg_target_gradient_ratio", 0.03))
        self.max_ratio = float(yaml.get("gbrg_max_gradient_ratio", 0.08))
        self.ema_beta = float(yaml.get("gbrg_lambda_ema", 0.9))
        self.lambda_min = float(yaml.get("gbrg_lambda_min", 0.1))
        self.lambda_max = float(yaml.get("gbrg_lambda_max", 100.0))
        self.sigma_divisor = float(yaml.get("gbrg_sigma_divisor", 6.0))
        self.ignore_dilation = float(yaml.get("gbrg_ignore_dilation", 1.25))
        self.background_ratio = float(yaml.get("gbrg_background_ratio", 3.0))
        if not 0 < self.target_ratio <= self.max_ratio < 1:
            raise ValueError("GBRG gradient ratios must satisfy 0 < target <= max < 1")
        if not 0 <= self.ema_beta < 1:
            raise ValueError("gbrg_lambda_ema must be in [0, 1)")
        if not 0 < self.lambda_min <= self.lambda_max:
            raise ValueError("invalid GBRG lambda bounds")
        if self.background_ratio < 0:
            raise ValueError("gbrg_background_ratio must be non-negative")

        self.current_lambda: float | None = None
        self.last_raw_gradient_ratio = 0.0
        self.last_weighted_gradient_ratio = 0.0
        self.last_region_loss = 0.0
        self.last_positive_pixels = 0
        self.last_background_effective_pixels = 0.0

    def region_loss(self, logits: torch.Tensor, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute class-agnostic soft BCE with an ignored border and capped clear background."""
        target, inside, clear_background = gbrg_region_targets(
            batch["batch_idx"],
            batch["bboxes"],
            logits.shape[0],
            logits.shape[-2],
            logits.shape[-1],
            self.sigma_divisor,
            self.ignore_dilation,
        )
        with autocast(enabled=False):
            per_pixel = F.binary_cross_entropy_with_logits(logits.float(), target, reduction="none")
            losses = []
            positive_total = 0
            background_effective_total = 0.0
            for image_index in range(logits.shape[0]):
                positive = inside[image_index]
                positive_count = int(positive.sum())
                if positive_count == 0:
                    continue
                clear = clear_background[image_index]
                clear_count = int(clear.sum())
                positive_weight = positive.float() * (0.1 + 0.9 * target[image_index])
                if clear_count and self.background_ratio:
                    background_weight = min(0.1, self.background_ratio * positive_count / clear_count)
                else:
                    background_weight = 0.0
                weight = positive_weight + clear.float() * background_weight
                losses.append((per_pixel[image_index] * weight).sum() / weight.sum().clamp_min(1.0))
                positive_total += positive_count
                background_effective_total += clear_count * background_weight

            self.last_positive_pixels = positive_total
            self.last_background_effective_pixels = background_effective_total
            return torch.stack(losses).mean() if losses else logits.float().sum() * 0.0

    def _balanced_lambda(
        self,
        detection_loss: torch.Tensor,
        region_loss: torch.Tensor,
        p3_feature: torch.Tensor,
    ) -> float:
        """Update a detached EMA controller from raw gradient norms at shared P3."""
        det_grad = torch.autograd.grad(
            detection_loss.sum(), p3_feature, retain_graph=True, create_graph=False, allow_unused=False
        )[0]
        region_grad = torch.autograd.grad(
            region_loss, p3_feature, retain_graph=True, create_graph=False, allow_unused=False
        )[0]
        det_norm = det_grad.detach().float().norm()
        region_norm = region_grad.detach().float().norm()
        if not torch.isfinite(det_norm) or not torch.isfinite(region_norm):
            raise FloatingPointError("non-finite GBRG gradient norm")
        raw_ratio = float((region_norm / det_norm.clamp_min(1e-12)).cpu())
        desired = min(self.lambda_max, max(self.lambda_min, self.target_ratio / max(raw_ratio, 1e-12)))
        if self.current_lambda is None:
            self.current_lambda = desired
        else:
            self.current_lambda = self.ema_beta * self.current_lambda + (1 - self.ema_beta) * desired
            self.current_lambda = min(self.lambda_max, max(self.lambda_min, self.current_lambda))
        # The EMA smooths batch noise, while this detached ceiling prevents one
        # outlier batch from violating the pre-registered 8% safety envelope.
        safe_max = self.max_ratio / max(raw_ratio, 1e-12)
        self.current_lambda = min(self.current_lambda, safe_max)
        self.last_raw_gradient_ratio = raw_ratio
        self.last_weighted_gradient_ratio = raw_ratio * self.current_lambda
        return self.current_lambda

    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Add GBRG only during training; validation remains the unmodified R1 detector."""
        predictions = preds[1] if isinstance(preds, tuple) else preds
        detection_preds = {key: predictions[key] for key in ("one2many", "one2one")}
        detection_loss, detection_items = super().__call__(detection_preds, batch)
        logits = predictions.get("gbrg_region_logits")
        p3_feature = predictions.get("gbrg_p3_feature")
        if logits is None or p3_feature is None or not torch.is_grad_enabled():
            weighted = detection_loss.sum() * 0.0
            self.last_region_loss = 0.0
        else:
            raw_region = self.region_loss(logits, batch)
            batch_size = detection_preds["one2one"]["boxes"].shape[0]
            scaled_region = raw_region * batch_size
            if self.last_positive_pixels:
                balance = self._balanced_lambda(detection_loss, scaled_region, p3_feature)
                weighted = scaled_region * balance
            else:
                weighted = scaled_region * 0.0
            self.last_region_loss = float(raw_region.detach())
        return (
            torch.cat((detection_loss, weighted.reshape(1))),
            torch.cat((detection_items, (weighted.detach() / max(logits.shape[0] if logits is not None else 1, 1)).reshape(1))),
        )
