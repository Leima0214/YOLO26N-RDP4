"""Trainer integration for the SCHM experiment family."""

from __future__ import annotations

import json
from pathlib import Path

from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.utils.torch_utils import unwrap_model

__all__ = ("SCHMDetectionTrainer",)


class SCHMDetectionTrainer(DetectionTrainer):
    """Expose SCHM as a fourth loss item and persist train-only mechanism statistics per epoch."""

    loss_names = ("box_loss", "cls_loss", "dfl_loss", "schm_loss")

    _scalar_stat_names = (
        "harvest_gt_count",
        "harvest_ratio",
        "new_index_harvest_count",
        "new_index_harvest_ratio",
        "same_index_ratio",
        "same_index_positive_gain_ratio",
        "mean_delta_iou",
        "median_delta_iou",
        "p90_delta_iou",
        "mean_weight",
        "schm_loss",
        "native_o2o_box_loss",
        "schm/native_box_loss_ratio",
        "gradient_ratio",
        "conflict_count",
        "conflict_ratio",
        "missing_o2m_rate",
        "missing_o2o_rate",
        "illegal_harvest_count",
    )
    _group_names = (
        "class_D00",
        "class_D10",
        "class_D20",
        "class_D40",
        "size_small",
        "size_medium",
        "size_large",
        "level_P3",
        "level_P4",
        "level_P5",
    )

    def get_validator(self):
        validator = super().get_validator()
        self.loss_names = ("box_loss", "cls_loss", "dfl_loss", "schm_loss")
        return validator

    def _ordered_schm_stats(self) -> dict[str, float]:
        model = unwrap_model(self.model)
        criterion = getattr(model, "criterion", None)
        raw = getattr(criterion, "last_epoch_stats", {}) or {}
        ordered = {name: float(raw.get(name, 0.0)) for name in self._scalar_stat_names}
        for group in self._group_names:
            for metric in ("harvest_ratio", "mean_delta", "mean_weight"):
                key = f"{group}/{metric}"
                ordered[key] = float(raw.get(key, 0.0))
        return ordered

    def save_metrics(self, metrics):
        stats = self._ordered_schm_stats()
        merged = dict(metrics)
        merged.update({f"schm/{name}": value for name, value in stats.items()})
        super().save_metrics(merged)

        path = Path(self.save_dir) / "schm_epoch_stats.jsonl"
        record = {"epoch": int(self.epoch + 1), **stats}
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
