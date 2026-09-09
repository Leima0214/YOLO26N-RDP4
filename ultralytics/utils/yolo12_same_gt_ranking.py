"""Training-only same-GT residual ranking for the native YOLO12 detector.

The auxiliary objective uses only native TaskAlignedAssigner positives.  For
two positives assigned to the same ground truth, it asks the positive with the
higher detached IoU to receive the higher logit for that ground-truth class.
It does not add parameters or alter Detect, decoding, validation, or NMS.
"""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy

import torch
import torch.nn.functional as F

from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.tal import make_anchors


def _aligned_iou_xyxy(box1: torch.Tensor, box2: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """Return aligned plain IoU for equally sized xyxy tensors."""
    inter_wh = (torch.minimum(box1[:, 2:], box2[:, 2:]) - torch.maximum(box1[:, :2], box2[:, :2])).clamp_(min=0)
    inter = inter_wh[:, 0] * inter_wh[:, 1]
    area1 = (box1[:, 2] - box1[:, 0]).clamp_(min=0) * (box1[:, 3] - box1[:, 1]).clamp_(min=0)
    area2 = (box2[:, 2] - box2[:, 0]).clamp_(min=0) * (box2[:, 3] - box2[:, 1]).clamp_(min=0)
    return inter / (area1 + area2 - inter + eps)


class TALPositiveSameGTRankingLoss(v8DetectionLoss):
    """Native YOLO loss plus a bounded pairwise ordering objective on TAL positives."""

    def __init__(
        self,
        model,
        tal_topk: int = 10,
        tal_topk2: int | None = None,
        rank_weight: float = 0.10,
        min_iou_gap: float = 0.05,
        temperature: float = 1.0,
    ) -> None:
        super().__init__(model, tal_topk=tal_topk, tal_topk2=tal_topk2)
        if rank_weight <= 0 or min_iou_gap <= 0 or temperature <= 0:
            raise ValueError("rank_weight, min_iou_gap, and temperature must be positive")
        self.rank_weight = float(rank_weight)
        self.min_iou_gap = float(min_iou_gap)
        self.temperature = float(temperature)
        self._epoch_stats = defaultdict(float)

    def get_assigned_targets_and_loss(self, preds, batch):
        assigned, loss, _ = super().get_assigned_targets_and_loss(preds, batch)
        fg_mask, target_gt_idx, target_bboxes, _, _ = assigned
        pred_distri = preds["boxes"].permute(0, 2, 1).contiguous()
        pred_logits = preds["scores"].permute(0, 2, 1).contiguous()
        anchor_points, stride_tensor = make_anchors(preds["feats"], self.stride, 0.5)
        pred_bboxes_px = self.bbox_decode(anchor_points, pred_distri) * stride_tensor

        batch_size = pred_logits.shape[0]
        dtype = pred_logits.dtype
        imgsz = torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]
        raw_targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        padded_targets = self.preprocess(
            raw_targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]]
        )
        gt_labels = padded_targets[..., 0].long()

        pair_losses = []
        pair_weights = []
        pair_correct = pred_logits.new_zeros(())
        pair_count = 0
        represented_gt = 0
        with torch.no_grad():
            detached_iou = [
                _aligned_iou_xyxy(pred_bboxes_px[b][fg_mask[b]].detach(), target_bboxes[b][fg_mask[b]])
                for b in range(batch_size)
            ]

        for b in range(batch_size):
            pos_idx = fg_mask[b].nonzero(as_tuple=False).squeeze(1)
            if pos_idx.numel() < 2:
                continue
            pos_gt = target_gt_idx[b, pos_idx].long()
            pos_iou = detached_iou[b]
            for gt_idx in pos_gt.unique():
                local = (pos_gt == gt_idx).nonzero(as_tuple=False).squeeze(1)
                if local.numel() < 2:
                    continue
                quality_delta = pos_iou[local][:, None] - pos_iou[local][None, :]
                higher, lower = (quality_delta > self.min_iou_gap).nonzero(as_tuple=True)
                if higher.numel() == 0:
                    continue
                represented_gt += 1
                cls_idx = gt_labels[b, gt_idx]
                logits = pred_logits[b, pos_idx[local], cls_idx]
                logit_delta = (logits[higher] - logits[lower]) / self.temperature
                weights = quality_delta[higher, lower]
                pair_losses.append(F.softplus(-logit_delta) * weights)
                pair_weights.append(weights)
                pair_correct = pair_correct + (logit_delta.detach() > 0).sum()
                pair_count += int(higher.numel())

        if pair_losses:
            weighted_loss = torch.cat(pair_losses).sum()
            weight_sum = torch.cat(pair_weights).sum().clamp_min(1e-9)
            rank_raw = weighted_loss / weight_sum
            rank_scaled = rank_raw * self.rank_weight * self.hyp.cls
            loss[1] = loss[1] + rank_scaled
            accuracy = pair_correct / pair_count
        else:
            rank_raw = pred_logits.sum() * 0.0
            rank_scaled = rank_raw
            accuracy = pred_logits.new_zeros(())

        self._epoch_stats["batches"] += 1
        self._epoch_stats["pairs"] += pair_count
        self._epoch_stats["represented_gt"] += represented_gt
        self._epoch_stats["rank_raw_sum"] += float(rank_raw.detach())
        self._epoch_stats["rank_scaled_sum"] += float(rank_scaled.detach())
        self._epoch_stats["pair_correct"] += float(pair_correct.detach())
        return assigned, loss, loss.detach()

    def pop_epoch_stats(self) -> dict[str, float]:
        """Return and reset accumulated training diagnostics."""
        stats = dict(self._epoch_stats)
        self._epoch_stats.clear()
        batches = max(stats.get("batches", 0.0), 1.0)
        pairs = max(stats.get("pairs", 0.0), 1.0)
        return {
            **stats,
            "rank_raw_mean": stats.get("rank_raw_sum", 0.0) / batches,
            "rank_scaled_mean": stats.get("rank_scaled_sum", 0.0) / batches,
            "pair_accuracy": stats.get("pair_correct", 0.0) / pairs,
        }


class YOLO12SameGTRankingModel(DetectionModel):
    """Native DetectionModel whose only change is the training criterion."""

    def init_criterion(self):
        spec = deepcopy(getattr(self, "same_gt_ranking_spec", {}))
        return TALPositiveSameGTRankingLoss(self, **spec)

