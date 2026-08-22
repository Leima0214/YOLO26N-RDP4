"""Val-only P3/P4/P5 attribution audit for YOLO26 end-to-end O2O predictions.

The script reproduces the native end-to-end top-k path while retaining each
selected anchor's feature-level identity. It reports isolated-level COCO
metrics and the level source of true positives in the combined prediction set.
No training or Test split access is performed.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import contextlib
import csv
import io
import json
from pathlib import Path
import statistics
import sys
from typing import Any

import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ultralytics import YOLO  # noqa: E402
from ultralytics.data.utils import check_det_dataset  # noqa: E402
from ultralytics.models.yolo.detect.val import DetectionValidator  # noqa: E402

LEVELS = ("P3", "P4", "P5")
AREAS = ("small", "medium", "large")


def parse_checkpoints(values: list[str]) -> dict[str, Path]:
    checkpoints: dict[str, Path] = {}
    for value in values:
        name, separator, raw_path = value.partition("=")
        path = Path(raw_path).expanduser().resolve()
        if not separator or not name or name in checkpoints or not path.is_file():
            raise ValueError(f"Expected unique NAME=CHECKPOINT, got {value}")
        checkpoints[name] = path
    return checkpoints


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def coco_evaluate(ground_truth: COCO, predictions: list[dict[str, Any]], category_ids: list[int]) -> tuple[COCO, COCOeval]:
    detections = ground_truth.loadRes(predictions)
    evaluator = COCOeval(ground_truth, detections, "bbox")
    evaluator.params.imgIds = sorted(ground_truth.imgs)
    evaluator.params.catIds = category_ids
    with contextlib.redirect_stdout(io.StringIO()):
        evaluator.evaluate()
        evaluator.accumulate()
    return detections, evaluator


def mean_valid(values: np.ndarray) -> float | None:
    values = values[values > -1]
    return float(values.mean()) if values.size else None


def metric(evaluator: COCOeval, kind: str, *, iou: float | None = None, area: str = "all", category_id: int | None = None) -> float | None:
    area_index = list(evaluator.params.areaRngLbl).index(area)
    category_slice: slice | list[int] = slice(None)
    if category_id is not None:
        category_slice = [list(evaluator.params.catIds).index(category_id)]
    if kind == "precision":
        values = evaluator.eval["precision"][:, :, category_slice, area_index, -1]
        if iou is not None:
            iou_index = int(np.argmin(np.abs(evaluator.params.iouThrs - iou)))
            values = values[iou_index : iou_index + 1]
    elif kind == "recall":
        values = evaluator.eval["recall"][:, category_slice, area_index, -1]
        if iou is not None:
            iou_index = int(np.argmin(np.abs(evaluator.params.iouThrs - iou)))
            values = values[iou_index : iou_index + 1]
    else:
        raise ValueError(kind)
    return mean_valid(values)


def xywh_iou(left: list[float], right: list[float]) -> float:
    lx1, ly1, lw, lh = left
    rx1, ry1, rw, rh = right
    lx2, ly2, rx2, ry2 = lx1 + lw, ly1 + lh, rx1 + rw, ry1 + rh
    intersection = max(0.0, min(lx2, rx2) - max(lx1, rx1)) * max(0.0, min(ly2, ry2) - max(ly1, ry1))
    union = lw * lh + rw * rh - intersection
    return intersection / union if union > 0 else 0.0


def area_label(area: float) -> str:
    if area < 32**2:
        return "small"
    if area < 96**2:
        return "medium"
    return "large"


def find_detect_head(predictor_model: Any):
    root = getattr(predictor_model, "model", predictor_model)
    candidates = [
        module
        for module in root.modules()
        if hasattr(module, "get_topk_index") and hasattr(module, "one2one") and hasattr(module, "nl")
    ]
    if not candidates:
        raise RuntimeError("Could not locate the end-to-end Detect head inside predictor model")
    return candidates[-1]


class LevelDetectionValidator(DetectionValidator):
    """Run the native validation pipeline while retaining O2O feature-level origin."""

    def init_metrics(self, model: Any) -> None:
        super().init_metrics(model)
        self._level_batches: list[list[torch.Tensor]] = []
        head = find_detect_head(model)

        def retain_level_indices(module, _inputs, output) -> None:
            if not isinstance(output, tuple) or len(output) < 2 or not isinstance(output[1], dict):
                raise RuntimeError(f"Unexpected Detect inference output type: {type(output)}")
            selected, raw = output[0], output[1]
            one2one = raw.get("one2one")
            if one2one is None:
                raise RuntimeError("Checkpoint did not expose O2O raw predictions")
            scores = one2one["scores"].sigmoid().permute(0, 2, 1).contiguous()
            _, _, anchor_indices = module.get_topk_index(scores, module.max_det)
            anchor_indices = anchor_indices.squeeze(-1)
            sizes = [feature.shape[-2] * feature.shape[-1] for feature in one2one["feats"]]
            if len(sizes) != 3:
                raise RuntimeError(f"Expected three O2O levels, got {sizes}")
            boundaries = torch.tensor(np.cumsum(sizes[:-1]), device=anchor_indices.device)
            levels = torch.bucketize(anchor_indices, boundaries)
            self._level_batches.append(
                [levels[index][selected[index, :, 4] > self.args.conf].detach().cpu() for index in range(selected.shape[0])]
            )

        self._level_hook = head.register_forward_hook(retain_level_indices)

    def postprocess(self, preds: torch.Tensor) -> list[dict[str, torch.Tensor]]:
        processed = super().postprocess(preds)
        if not self._level_batches:
            raise RuntimeError("Detect hook did not retain O2O level indices")
        levels = self._level_batches.pop(0)
        if len(processed) != len(levels):
            raise AssertionError(f"Batch result/level count mismatch: {len(processed)} != {len(levels)}")
        for prediction, image_levels in zip(processed, levels):
            if len(prediction["conf"]) != len(image_levels):
                raise AssertionError(
                    f"Detection/level count mismatch: {len(prediction['conf'])} != {len(image_levels)}"
                )
            prediction["level"] = image_levels.to(prediction["conf"].device)
        return processed

    def pred_to_json(self, predn: dict[str, torch.Tensor], pbatch: dict[str, Any]) -> None:
        start = len(self.jdict)
        super().pred_to_json(predn, pbatch)
        added = self.jdict[start:]
        if len(added) != len(predn["level"]):
            raise AssertionError(f"Serialized detection/level mismatch: {len(added)} != {len(predn['level'])}")
        for row, level in zip(added, predn["level"].tolist()):
            row["level"] = LEVELS[int(level)]


def predict_with_levels(
    checkpoint: Path,
    model_name: str,
    data_path: Path,
    validation_root: Path,
    image_id_by_stem: dict[str, int],
    category_id_by_index: dict[int, int],
    *,
    imgsz: int,
    batch: int,
    device: str,
    conf: float,
    max_det: int,
) -> list[dict[str, Any]]:
    model = YOLO(str(checkpoint), task="detect")
    model.val(
        validator=LevelDetectionValidator,
        data=str(data_path.resolve()),
        split="val",
        imgsz=imgsz,
        batch=batch,
        workers=8,
        device=device,
        conf=conf,
        iou=0.7,
        max_det=max_det,
        rect=True,
        save_json=True,
        plots=False,
        project=str(validation_root),
        name=model_name,
        exist_ok=True,
        verbose=False,
    )
    raw_predictions = json.loads((validation_root / model_name / "predictions.json").read_text(encoding="utf-8"))
    predictions: list[dict[str, Any]] = []
    for prediction in raw_predictions:
        stem = Path(prediction["file_name"]).stem
        class_index = int(prediction["category_id"]) - 1
        predictions.append(
            {
                "image_id": image_id_by_stem[stem],
                "category_id": category_id_by_index[class_index],
                "bbox": prediction["bbox"],
                "score": prediction["score"],
                "level": prediction["level"],
            }
        )
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return predictions


def metric_row(evaluator: COCOeval, *, model: str, level: str, area: str = "all", category_id: int | None = None, category: str = "all", prediction_count: int) -> dict[str, Any]:
    return {
        "model": model,
        "level": level,
        "category": category,
        "area": area,
        "predictions": prediction_count,
        "AP50_95": metric(evaluator, "precision", area=area, category_id=category_id),
        "AP50": metric(evaluator, "precision", iou=0.50, area=area, category_id=category_id),
        "AP75": metric(evaluator, "precision", iou=0.75, area=area, category_id=category_id),
        "AR100": metric(evaluator, "recall", area=area, category_id=category_id),
        "Recall50": metric(evaluator, "recall", iou=0.50, area=area, category_id=category_id),
        "Recall75": metric(evaluator, "recall", iou=0.75, area=area, category_id=category_id),
    }


def tp_contribution_rows(
    model: str,
    ground_truth: COCO,
    detections: COCO,
    evaluator: COCOeval,
    category_names: dict[int, str],
) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    all_range = tuple(evaluator.params.areaRng[list(evaluator.params.areaRngLbl).index("all")])
    for evaluated_image in evaluator.evalImgs:
        if evaluated_image is None or tuple(evaluated_image["aRng"]) != all_range:
            continue
        for threshold in (0.50, 0.75):
            threshold_index = int(np.argmin(np.abs(evaluator.params.iouThrs - threshold)))
            for detection_index, ground_truth_id in enumerate(evaluated_image["dtMatches"][threshold_index]):
                if ground_truth_id <= 0 or evaluated_image["dtIgnore"][threshold_index, detection_index]:
                    continue
                detection = detections.anns[int(evaluated_image["dtIds"][detection_index])]
                target = ground_truth.anns[int(ground_truth_id)]
                matches.append(
                    {
                        "threshold": threshold,
                        "level": detection["level"],
                        "category": category_names[target["category_id"]],
                        "area": area_label(float(target["area"])),
                        "iou": xywh_iou(detection["bbox"], target["bbox"]),
                    }
                )

    rows: list[dict[str, Any]] = []
    for threshold in (0.50, 0.75):
        threshold_matches = [row for row in matches if row["threshold"] == threshold]
        for dimension, groups in (
            ("overall", ("all",)),
            ("category", tuple(category_names.values())),
            ("area", AREAS),
        ):
            for group in groups:
                selected = threshold_matches if dimension == "overall" else [row for row in threshold_matches if row[dimension] == group]
                total = len(selected)
                for level in LEVELS:
                    values = [row["iou"] for row in selected if row["level"] == level]
                    rows.append(
                        {
                            "model": model,
                            "iou_threshold": threshold,
                            "group_type": dimension,
                            "group": group,
                            "level": level,
                            "tp_count": len(values),
                            "tp_share": len(values) / total if total else 0.0,
                            "matched_iou_mean": statistics.fmean(values) if values else None,
                            "matched_iou_median": statistics.median(values) if values else None,
                        }
                    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--device", default="0")
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--max-det", type=int, default=300)
    args = parser.parse_args()
    checkpoints = parse_checkpoints(args.checkpoint)
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)

    data = check_det_dataset(str(args.data.resolve()))
    coco_path = Path(data["path"]) / "annotations" / "instances_val.json"
    ground_truth = COCO(str(coco_path))
    category_names = {category_id: category["name"] for category_id, category in ground_truth.cats.items()}
    category_ids = sorted(category_names)
    category_id_by_index = {index: category_id for index, category_id in enumerate(category_ids)}
    image_id_by_stem = {Path(image["file_name"]).stem: image_id for image_id, image in ground_truth.imgs.items()}
    overall_rows: list[dict[str, Any]] = []
    level_rows: list[dict[str, Any]] = []
    class_rows: list[dict[str, Any]] = []
    contribution_rows: list[dict[str, Any]] = []
    artifacts: dict[str, str] = {}

    for model_name, checkpoint in checkpoints.items():
        print(f"LEVEL_AUDIT_START model={model_name} checkpoint={checkpoint}", flush=True)
        predictions = predict_with_levels(
            checkpoint, model_name, args.data, args.output / "validator_predictions", image_id_by_stem, category_id_by_index,
            imgsz=args.imgsz, batch=args.batch, device=args.device, conf=args.conf, max_det=args.max_det,
        )
        prediction_path = args.output / f"{model_name}_predictions_with_levels.json"
        prediction_path.write_text(json.dumps(predictions), encoding="utf-8")
        artifacts[model_name] = str(prediction_path)

        detections, combined_eval = coco_evaluate(ground_truth, predictions, category_ids)
        overall_rows.append(
            metric_row(
                combined_eval, model=model_name, level="combined", prediction_count=len(predictions)
            )
        )
        contribution_rows.extend(
            tp_contribution_rows(model_name, ground_truth, detections, combined_eval, category_names)
        )

        for level in LEVELS:
            level_predictions = [prediction for prediction in predictions if prediction["level"] == level]
            _, level_eval = coco_evaluate(ground_truth, level_predictions, category_ids)
            level_rows.append(
                metric_row(
                    level_eval, model=model_name, level=level, prediction_count=len(level_predictions)
                )
            )
            for area in AREAS:
                level_rows.append(
                    metric_row(
                        level_eval, model=model_name, level=level, area=area,
                        prediction_count=len(level_predictions),
                    )
                )
            for category_id in category_ids:
                class_rows.append(
                    metric_row(
                        level_eval, model=model_name, level=level, category_id=category_id,
                        category=category_names[category_id], prediction_count=len(level_predictions),
                    )
                )
        print(f"LEVEL_AUDIT_DONE model={model_name} predictions={len(predictions)}", flush=True)

    write_csv(args.output / "overall_metrics.csv", overall_rows)
    write_csv(args.output / "level_metrics.csv", level_rows)
    write_csv(args.output / "level_class_metrics.csv", class_rows)
    write_csv(args.output / "tp_level_contributions.csv", contribution_rows)
    report = {
        "protocol": {
            "split": "val_only",
            "data": str(args.data.resolve()),
            "imgsz": args.imgsz,
            "batch": args.batch,
            "conf": args.conf,
            "max_det_native": args.max_det,
            "max_det_coco": 100,
            "models": {name: str(path) for name, path in checkpoints.items()},
        },
        "overall": overall_rows,
        "level_metrics": level_rows,
        "level_class_metrics": class_rows,
        "tp_contributions": contribution_rows,
        "prediction_artifacts": artifacts,
    }
    (args.output / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"overall": overall_rows, "output": str(args.output)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
