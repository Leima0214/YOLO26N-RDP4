"""Selective Candidate Harvest Matching for YOLO26 end-to-end detection.

O2M is used only as a legal candidate-index proposer.  The harvested index is
applied to the O2O prediction and the target is always the real ground truth.
No O2M logits, boxes, features, or scores are transferred as supervision.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import torch
import torch.nn.functional as F

from ultralytics.utils.loss import E2ELoss
from ultralytics.utils.metrics import bbox_iou
from ultralytics.utils.tal import bbox2dist

__all__ = ("SCHME2ELoss",)


class SCHME2ELoss(E2ELoss):
    """Add a detached, new-index-only O2M candidate harvest loss to native YOLO26 E2E loss."""

    def __init__(self, model):
        super().__init__(model)
        self.model = model
        self.lambda_schm = float(model.yaml.get("lambda_schm", 1.0))
        if self.lambda_schm < 0:
            raise ValueError(
                f"lambda_schm must be non-negative, got {self.lambda_schm}"
            )
        self.last_schm_raw = None
        self.last_native_o2o_box = None
        self.last_batch_stats: dict[str, Any] = {}
        self.last_epoch_stats: dict[str, float] = {}
        self._measure_gradient = True
        self._epoch_scalars: defaultdict[str, float] = defaultdict(float)
        self._epoch_values: defaultdict[str, list[float]] = defaultdict(list)
        self._epoch_groups: defaultdict[str, defaultdict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )

    @staticmethod
    def _level_layout(preds: dict[str, torch.Tensor]) -> tuple[list[int], list[int]]:
        sizes = [
            int(feature.shape[-2] * feature.shape[-1]) for feature in preds["feats"]
        ]
        offsets = [0]
        for size in sizes:
            offsets.append(offsets[-1] + size)
        return sizes, offsets

    @staticmethod
    def global_to_level_yx(
        index: int, shapes: list[tuple[int, int]]
    ) -> tuple[int, int, int]:
        """Map Detect's explicit P3/P4/P5 flatten order to (level, y, x)."""
        offset = 0
        for level, (height, width) in enumerate(shapes):
            size = height * width
            if index < offset + size:
                local = index - offset
                return level, local // width, local % width
            offset += size
        raise IndexError(f"global candidate index {index} outside {offset} positions")

    @staticmethod
    def level_yx_to_global(
        level: int, y: int, x: int, shapes: list[tuple[int, int]]
    ) -> int:
        """Inverse of :meth:`global_to_level_yx`."""
        if not 0 <= level < len(shapes):
            raise IndexError(level)
        height, width = shapes[level]
        if not (0 <= y < height and 0 <= x < width):
            raise IndexError((level, y, x))
        return sum(h * w for h, w in shapes[:level]) + y * width + x

    def _preprocess_gt(
        self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]
    ):
        pred_scores = preds["scores"].permute(0, 2, 1).contiguous()
        batch_size = pred_scores.shape[0]
        dtype = pred_scores.dtype
        imgsz = torch.tensor(
            preds["feats"][0].shape[2:], device=self.one2one.device, dtype=dtype
        )
        imgsz *= self.one2one.stride[0]
        targets = torch.cat(
            (batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]),
            1,
        )
        targets = self.one2one.preprocess(
            targets.to(self.one2one.device),
            batch_size,
            scale_tensor=imgsz[[1, 0, 1, 0]],
        )
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2).gt(0)
        return gt_labels, gt_bboxes, mask_gt, imgsz

    @staticmethod
    def _size_group(box: torch.Tensor) -> str:
        area = float(
            ((box[2] - box[0]).clamp_min(0) * (box[3] - box[1]).clamp_min(0)).item()
        )
        if area < 32**2:
            return "small"
        if area < 96**2:
            return "medium"
        return "large"

    @staticmethod
    def _gradient_l2(loss: torch.Tensor, parameters: list[torch.nn.Parameter]) -> float:
        if not loss.requires_grad:
            return 0.0
        gradients = torch.autograd.grad(
            loss, parameters, retain_graph=True, allow_unused=True
        )
        squared = sum(
            float(gradient.detach().float().square().sum().cpu())
            for gradient in gradients
            if gradient is not None
        )
        return squared**0.5

    def _native_localization(
        self,
        one2one: dict[str, torch.Tensor],
        anchor_points: torch.Tensor,
        stride_tensor: torch.Tensor,
        imgsz: torch.Tensor,
        selected_batch: torch.Tensor,
        selected_anchor: torch.Tensor,
        selected_gt_boxes: torch.Tensor,
        weights: torch.Tensor,
        n_gt: int,
    ) -> torch.Tensor:
        """Reuse YOLO26's CIoU plus reg_max=1 normalized-L1 localization primitives with N_GT normalization."""
        pred_distri = one2one["boxes"].permute(0, 2, 1).contiguous()
        pred_bboxes = self.one2one.bbox_decode(anchor_points, pred_distri)
        pred_distri = pred_distri[selected_batch, selected_anchor]
        pred_bboxes = pred_bboxes[selected_batch, selected_anchor]
        selected_stride = stride_tensor[selected_anchor]
        target_bboxes = selected_gt_boxes / selected_stride

        ciou = bbox_iou(pred_bboxes, target_bboxes, xywh=False, CIoU=True).squeeze(-1)
        loss_iou = ((1.0 - ciou) * weights).sum() / max(n_gt, 1)

        # YOLO26n uses reg_max=1. Keep the native normalized L1 localization primitive exactly.
        if self.one2one.use_dfl:
            raise NotImplementedError("SCHM v1 is locked to YOLO26 reg_max=1")
        target_ltrb = bbox2dist(anchor_points[selected_anchor], target_bboxes)
        target_ltrb = target_ltrb * selected_stride
        target_ltrb[..., 0::2] /= imgsz[1]
        target_ltrb[..., 1::2] /= imgsz[0]
        normalized_pred = pred_distri * selected_stride
        normalized_pred[..., 0::2] /= imgsz[1]
        normalized_pred[..., 1::2] /= imgsz[0]
        loss_l1 = F.l1_loss(normalized_pred, target_ltrb, reduction="none").mean(-1)
        loss_l1 = (loss_l1 * weights).sum() / max(n_gt, 1)
        return loss_iou * self.one2one.hyp.box + loss_l1 * self.one2one.hyp.dfl

    def _register_group(
        self,
        name: str,
        delta: float | None = None,
        weight: float | None = None,
        harvest=False,
    ):
        group = self._epoch_groups[name]
        group["gt"] += 1
        if delta is not None:
            group["delta_sum"] += delta
            group["delta_n"] += 1
        if weight is not None:
            group["weight_sum"] += weight
            group["weight_n"] += 1
        if harvest:
            group["harvest"] += 1

    def _accumulate(self, stats: dict[str, Any]) -> None:
        if not (self.model.training and torch.is_grad_enabled()):
            return
        for key in (
            "total_gt",
            "comparable_gt",
            "harvest_gt_count",
            "new_index_harvest_count",
            "same_index_count",
            "positive_gain_count",
            "same_index_gain_count",
            "conflict_count",
            "missing_o2m_count",
            "missing_o2o_count",
            "illegal_harvest_count",
        ):
            self._epoch_scalars[key] += float(stats[key])
        self._epoch_scalars["batch_count"] += 1
        self._epoch_scalars["schm_loss_sum"] += float(stats["schm_loss"])
        self._epoch_scalars["native_o2o_box_loss_sum"] += float(
            stats["native_o2o_box_loss"]
        )
        if stats.get("gradient_ratio") is not None:
            self._epoch_values["gradient_ratio"].append(float(stats["gradient_ratio"]))
        self._epoch_values["delta_iou"].extend(
            float(value) for value in stats["delta_values"]
        )
        self._epoch_values["weight"].extend(
            float(value) for value in stats["weight_values"]
        )
        for group_name, group_stats in stats["groups"].items():
            target = self._epoch_groups[group_name]
            for key, value in group_stats.items():
                target[key] += float(value)

    def _finalize_epoch(self) -> dict[str, float]:
        s = self._epoch_scalars
        total_gt = max(s["total_gt"], 1.0)
        comparable = max(s["comparable_gt"], 1.0)
        harvest = s["harvest_gt_count"]
        deltas = torch.tensor(self._epoch_values["delta_iou"], dtype=torch.float32)
        weights = torch.tensor(self._epoch_values["weight"], dtype=torch.float32)
        batches = max(s["batch_count"], 1.0)
        report = {
            "harvest_gt_count": harvest,
            "harvest_ratio": harvest / total_gt,
            "new_index_harvest_count": s["new_index_harvest_count"],
            "new_index_harvest_ratio": s["new_index_harvest_count"] / total_gt,
            "same_index_ratio": s["same_index_count"] / comparable,
            "same_index_positive_gain_ratio": s["same_index_gain_count"]
            / max(s["positive_gain_count"], 1.0),
            "mean_delta_iou": float(deltas.mean()) if deltas.numel() else 0.0,
            "median_delta_iou": float(deltas.median()) if deltas.numel() else 0.0,
            "p90_delta_iou": float(torch.quantile(deltas, 0.9))
            if deltas.numel()
            else 0.0,
            "mean_weight": float(weights.mean()) if weights.numel() else 0.0,
            "schm_loss": s["schm_loss_sum"] / batches,
            "native_o2o_box_loss": s["native_o2o_box_loss_sum"] / batches,
            "conflict_count": s["conflict_count"],
            "conflict_ratio": s["conflict_count"]
            / max(harvest + s["conflict_count"], 1.0),
            "missing_o2m_rate": s["missing_o2m_count"] / total_gt,
            "missing_o2o_rate": s["missing_o2o_count"] / total_gt,
            "illegal_harvest_count": s["illegal_harvest_count"],
            "gradient_ratio": (
                sum(self._epoch_values["gradient_ratio"])
                / len(self._epoch_values["gradient_ratio"])
                if self._epoch_values["gradient_ratio"]
                else 0.0
            ),
        }
        report["schm/native_box_loss_ratio"] = report["schm_loss"] / max(
            report["native_o2o_box_loss"], 1e-12
        )
        for name, group in sorted(self._epoch_groups.items()):
            report[f"{name}/harvest_ratio"] = group["harvest"] / max(group["gt"], 1.0)
            report[f"{name}/mean_delta"] = group["delta_sum"] / max(
                group["delta_n"], 1.0
            )
            report[f"{name}/mean_weight"] = group["weight_sum"] / max(
                group["weight_n"], 1.0
            )
        return report

    def update(self) -> None:
        """Finalize train-only SCHM statistics, reset them, and retain native O2M/O2O decay behavior."""
        super().update()
        self.last_epoch_stats = self._finalize_epoch()
        self._epoch_scalars = defaultdict(float)
        self._epoch_values = defaultdict(list)
        self._epoch_groups = defaultdict(lambda: defaultdict(float))
        self._measure_gradient = True

    def __call__(
        self, preds: Any, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        preds = self.one2many.parse_output(preds)
        one2many, one2one = preds["one2many"], preds["one2one"]
        assigned_m, loss_m, detached_m = self.one2many.get_assigned_targets_and_loss(
            one2many, batch
        )
        assigned_o, loss_o, detached_o = self.one2one.get_assigned_targets_and_loss(
            one2one, batch
        )
        batch_size = one2one["boxes"].shape[0]
        native = loss_m * batch_size * self.o2m + loss_o * batch_size * self.o2o
        native_detached = detached_m * self.o2m + detached_o * self.o2o
        self.last_native_o2o_box = loss_o[0] * batch_size * self.o2o

        fg_m, gt_idx_m, _, anchors_m, strides_m = assigned_m
        fg_o, gt_idx_o, _, anchors_o, strides_o = assigned_o
        shapes_m = [(int(x.shape[-2]), int(x.shape[-1])) for x in one2many["feats"]]
        shapes_o = [(int(x.shape[-2]), int(x.shape[-1])) for x in one2one["feats"]]
        if (
            shapes_m != shapes_o
            or not torch.equal(anchors_m, anchors_o)
            or not torch.equal(strides_m, strides_o)
        ):
            raise AssertionError(
                "O2M/O2O flatten order or scale coordinate mapping is not identical"
            )

        gt_labels, gt_bboxes, mask_gt, imgsz = self._preprocess_gt(one2one, batch)
        n_gt = int(mask_gt.sum().item())
        pred_m = one2many["boxes"].permute(0, 2, 1).contiguous()
        pred_o = one2one["boxes"].permute(0, 2, 1).contiguous()
        with torch.no_grad():
            decoded_m = (
                self.one2many.bbox_decode(anchors_m, pred_m.detach()) * strides_m
            )
            decoded_o = self.one2one.bbox_decode(anchors_o, pred_o.detach()) * strides_o
            selected: list[
                tuple[float, int, int, int, float, float, int, int, str, str]
            ] = []
            comparable = same_index = positive_gain = same_index_gain = missing_m = (
                missing_o
            ) = illegal = 0
            groups: defaultdict[str, defaultdict[str, float]] = defaultdict(
                lambda: defaultdict(float)
            )
            delta_values: list[float] = []
            weight_values: list[float] = []

            for image in range(batch_size):
                for gt_index in (
                    mask_gt[image].nonzero(as_tuple=False).flatten().tolist()
                ):
                    label = int(gt_labels[image, gt_index, 0].item())
                    class_group = (
                        f"class_D{[0, 10, 20, 40][label]:02d}"
                        if 0 <= label < 4
                        else f"class_{label}"
                    )
                    size_group = f"size_{self._size_group(gt_bboxes[image, gt_index])}"
                    for group_name in (class_group, size_group):
                        groups[group_name]["gt"] += 1

                    candidates_m = (
                        (fg_m[image] & (gt_idx_m[image] == gt_index))
                        .nonzero(as_tuple=False)
                        .flatten()
                    )
                    candidates_o = (
                        (fg_o[image] & (gt_idx_o[image] == gt_index))
                        .nonzero(as_tuple=False)
                        .flatten()
                    )
                    if candidates_m.numel() == 0:
                        missing_m += 1
                    if candidates_o.numel() == 0:
                        missing_o += 1
                    if candidates_m.numel() == 0 or candidates_o.numel() == 0:
                        continue
                    comparable += 1

                    gt_box = gt_bboxes[image, gt_index].unsqueeze(0)
                    q_m_all = bbox_iou(
                        decoded_m[image, candidates_m], gt_box, xywh=False
                    ).squeeze(-1)
                    best_local = int(q_m_all.argmax().item())
                    index_m = int(candidates_m[best_local].item())
                    q_m = float(q_m_all[best_local].clamp(0, 1).item())
                    # Native topk2=1 should leave one O2O reference. A defensive max-IoU choice preserves semantics if
                    # a future assigner revision returns more while remaining entirely within legal positives.
                    q_o_all = bbox_iou(
                        decoded_o[image, candidates_o], gt_box, xywh=False
                    ).squeeze(-1)
                    best_o = int(q_o_all.argmax().item())
                    index_o = int(candidates_o[best_o].item())
                    q_o = float(q_o_all[best_o].clamp(0, 1).item())
                    delta = q_m - q_o
                    weight = max(delta, 0.0)
                    delta_values.append(delta)
                    weight_values.append(weight)
                    if index_m == index_o:
                        same_index += 1
                    if weight > 0:
                        positive_gain += 1
                        if index_m == index_o:
                            same_index_gain += 1
                    level, _, _ = self.global_to_level_yx(index_m, shapes_m)
                    level_group = f"level_P{level + 3}"
                    for group_name in (class_group, size_group, level_group):
                        if group_name == level_group:
                            groups[group_name]["gt"] += 1
                        groups[group_name]["delta_sum"] += delta
                        groups[group_name]["delta_n"] += 1
                        groups[group_name]["weight_sum"] += weight
                        groups[group_name]["weight_n"] += 1

                    legal = bool(
                        fg_m[image, index_m] and gt_idx_m[image, index_m] == gt_index
                    )
                    if not legal:
                        illegal += 1
                        continue
                    if index_m != index_o and weight > 0:
                        selected.append(
                            (
                                delta,
                                image,
                                index_m,
                                gt_index,
                                q_m,
                                q_o,
                                label,
                                level,
                                class_group,
                                size_group,
                            )
                        )

            # Defensive one-candidate uniqueness resolution per image. Native post-conflict O2M positives normally
            # make this conflict count exactly zero, but the invariant remains explicit and audited.
            selected.sort(key=lambda item: item[0], reverse=True)
            claimed: set[tuple[int, int]] = set()
            unique = []
            conflicts = 0
            for item in selected:
                key = (item[1], item[2])
                if key in claimed:
                    conflicts += 1
                    continue
                claimed.add(key)
                unique.append(item)
            for delta, _, _, _, _, _, _, level, class_group, size_group in unique:
                for group_name in (class_group, size_group, f"level_P{level + 3}"):
                    groups[group_name]["harvest"] += 1

        if unique:
            selected_batch = torch.tensor(
                [item[1] for item in unique], device=pred_o.device, dtype=torch.long
            )
            selected_anchor = torch.tensor(
                [item[2] for item in unique], device=pred_o.device, dtype=torch.long
            )
            selected_gt = torch.stack(
                [gt_bboxes[item[1], item[3]] for item in unique]
            ).to(pred_o.dtype)
            weights = torch.tensor(
                [item[0] for item in unique], device=pred_o.device, dtype=pred_o.dtype
            ).detach()
            schm_raw = self._native_localization(
                one2one,
                anchors_o,
                strides_o,
                imgsz,
                selected_batch,
                selected_anchor,
                selected_gt,
                weights,
                n_gt,
            )
        else:
            schm_raw = one2one["boxes"].sum() * 0.0
        self.last_schm_raw = schm_raw
        scaled_schm = schm_raw * self.lambda_schm

        gradient_ratio = None
        if self._measure_gradient and self.model.training and torch.is_grad_enabled():
            parameters = [
                parameter
                for name, parameter in self.model.named_parameters()
                if "one2one_cv2" in name and parameter.requires_grad
            ]
            native_norm = self._gradient_l2(self.last_native_o2o_box, parameters)
            schm_norm = self._gradient_l2(scaled_schm, parameters)
            gradient_ratio = schm_norm / max(native_norm, 1e-12)
            self._measure_gradient = False

        batch_stats = {
            "total_gt": n_gt,
            "comparable_gt": comparable,
            "harvest_gt_count": len(unique),
            "new_index_harvest_count": len(unique),
            "same_index_count": same_index,
            "positive_gain_count": positive_gain,
            "same_index_gain_count": same_index_gain,
            "conflict_count": conflicts,
            "missing_o2m_count": missing_m,
            "missing_o2o_count": missing_o,
            "illegal_harvest_count": illegal,
            "delta_values": delta_values,
            "weight_values": weight_values,
            "schm_loss": float(scaled_schm.detach().cpu()),
            "native_o2o_box_loss": float(self.last_native_o2o_box.detach().cpu()),
            "gradient_ratio": gradient_ratio,
            "groups": {name: dict(values) for name, values in groups.items()},
            "level_shapes": shapes_o,
        }
        self.last_batch_stats = batch_stats
        self._accumulate(batch_stats)

        total = torch.cat((native, scaled_schm.reshape(1)))
        detached = torch.cat((native_detached, scaled_schm.detach().reshape(1)))
        return total, detached
