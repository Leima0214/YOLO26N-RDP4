"""Training-only P3 hard-negative ranking loss for RoadSnake-HNR."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from ultralytics.utils.loss import E2ELoss
from ultralytics.utils.metrics import box_iou
from ultralytics.utils.tal import make_anchors


class RoadSnakeHNRE2ELoss(E2ELoss):
    """Add a conservative P3 O2O positive-vs-clear-background ranking regularizer.

    Candidate geometry and selection are detached. Ambiguous unmatched anchors are
    ignored: only predictions with very low IoU to every annotation and centers
    outside a dilated union of all GT boxes can become negatives.
    """

    def __init__(self, model) -> None:
        super().__init__(model)
        head = model.model[-1]
        self.gain = head.hnr_loss_gain
        self.iou_threshold = head.hnr_iou_threshold
        self.gt_dilation = head.hnr_gt_dilation
        self.negatives_per_positive = head.hnr_negatives_per_positive
        self.margin = head.hnr_margin
        self.small_area = head.hnr_small_area
        self.last_stats = {
            "small_positives": 0,
            "clear_negative_pool": 0,
            "ranking_pairs": 0,
            "images_with_pairs": 0,
            "raw_loss": 0.0,
        }
        self.last_raw_hnr = None

    def _preprocess_targets(
        self,
        preds: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return padded labels, pixel-space boxes, and valid-GT mask."""
        batch_size = preds["scores"].shape[0]
        imgsz = torch.tensor(preds["feats"][0].shape[2:], device=self.one2one.device, dtype=dtype)
        imgsz *= self.one2one.stride[0]
        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.one2one.preprocess(
            targets.to(self.one2one.device),
            batch_size,
            scale_tensor=imgsz[[1, 0, 1, 0]],
        )
        gt_labels, gt_bboxes = targets.split((1, 4), dim=2)
        return gt_labels, gt_bboxes, gt_bboxes.sum(dim=2).gt(0)

    @staticmethod
    def _dilate_xyxy(boxes: torch.Tensor, factor: float) -> torch.Tensor:
        center = (boxes[:, :2] + boxes[:, 2:]) * 0.5
        half_size = (boxes[:, 2:] - boxes[:, :2]) * (0.5 * factor)
        return torch.cat((center - half_size, center + half_size), dim=1)

    def hard_negative_ranking_loss(
        self,
        preds: dict[str, torch.Tensor],
        assigned: tuple,
        batch: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Rank each matched P3-small positive above a few same-class clear negatives."""
        fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor = assigned
        pred_scores = preds["scores"].permute(0, 2, 1).contiguous()
        p3_count = preds["feats"][0].shape[-2] * preds["feats"][0].shape[-1]
        if p3_count <= 0:
            return pred_scores.sum() * 0.0

        target_area = (
            (target_bboxes[..., 2] - target_bboxes[..., 0]).clamp_min(0)
            * (target_bboxes[..., 3] - target_bboxes[..., 1]).clamp_min(0)
        )
        small_positive_mask = fg_mask[:, :p3_count] & (target_area[:, :p3_count] < self.small_area)

        selections: list[tuple[int, int, int, torch.Tensor]] = []
        clear_pool_count = 0
        images_with_pairs = 0
        with torch.no_grad():
            gt_labels, gt_bboxes, valid_gt = self._preprocess_targets(preds, batch, pred_scores.dtype)
            pred_distri = preds["boxes"].permute(0, 2, 1).contiguous()
            pred_bboxes = self.one2one.bbox_decode(anchor_points, pred_distri).detach() * stride_tensor
            p3_centers = anchor_points[:p3_count] * stride_tensor[:p3_count]

            for image_index in range(pred_scores.shape[0]):
                valid = valid_gt[image_index]
                if not valid.any():
                    continue
                image_gt = gt_bboxes[image_index, valid]
                pair_iou = box_iou(pred_bboxes[image_index, :p3_count], image_gt)
                max_iou = pair_iou.amax(dim=1)
                dilated = self._dilate_xyxy(image_gt, self.gt_dilation)
                centers = p3_centers
                inside = (
                    (centers[:, None, 0] >= dilated[None, :, 0])
                    & (centers[:, None, 0] <= dilated[None, :, 2])
                    & (centers[:, None, 1] >= dilated[None, :, 1])
                    & (centers[:, None, 1] <= dilated[None, :, 3])
                ).any(dim=1)
                clear_negative = (
                    (max_iou < self.iou_threshold)
                    & ~inside
                    & ~fg_mask[image_index, :p3_count]
                )
                negative_indices = clear_negative.nonzero(as_tuple=False).squeeze(1)
                clear_pool_count += int(negative_indices.numel())
                positive_indices = small_positive_mask[image_index].nonzero(as_tuple=False).squeeze(1)
                if not negative_indices.numel() or not positive_indices.numel():
                    continue

                image_has_pairs = False
                for positive_index in positive_indices.tolist():
                    gt_index = int(target_gt_idx[image_index, positive_index].item())
                    class_index = int(gt_labels[image_index, gt_index, 0].item())
                    negative_logits = pred_scores[image_index, negative_indices, class_index].detach()
                    count = min(self.negatives_per_positive, negative_logits.numel())
                    chosen = negative_indices[negative_logits.topk(count, largest=True).indices]
                    selections.append((image_index, positive_index, class_index, chosen))
                    image_has_pairs = True
                images_with_pairs += int(image_has_pairs)

        terms = []
        pair_count = 0
        for image_index, positive_index, class_index, negative_indices in selections:
            positive = pred_scores[image_index, positive_index, class_index]
            negatives = pred_scores[image_index, negative_indices, class_index]
            terms.append(F.softplus(negatives.float() - positive.float() + self.margin))
            pair_count += int(negative_indices.numel())
        raw = torch.cat(terms).mean() if terms else pred_scores.sum() * 0.0
        self.last_raw_hnr = raw
        self.last_stats = {
            "small_positives": int(small_positive_mask.sum().item()),
            "clear_negative_pool": clear_pool_count,
            "ranking_pairs": pair_count,
            "images_with_pairs": images_with_pairs,
            "raw_loss": float(raw.detach().cpu()),
        }
        return raw

    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Return standard E2E detection losses plus the separately logged HNR term."""
        preds = self.one2many.parse_output(preds)
        one2many, one2one = preds["one2many"], preds["one2one"]
        loss_one2many = self.one2many.loss(one2many, batch)
        assigned, one2one_loss, one2one_detached = self.one2one.get_assigned_targets_and_loss(one2one, batch)
        batch_size = one2one["boxes"].shape[0]
        raw_hnr = self.hard_negative_ranking_loss(one2one, assigned, batch)
        detection = loss_one2many[0] * self.o2m + one2one_loss * batch_size * self.o2o
        hnr = raw_hnr * batch_size * self.gain
        total = torch.cat((detection, hnr.reshape(1)))
        detached = torch.cat((one2one_detached, (raw_hnr.detach() * self.gain).reshape(1)))
        return total, detached
