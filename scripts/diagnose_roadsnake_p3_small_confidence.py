"""Decompose P3 small-object confidence errors from saved Val-only O2O predictions.

This analysis consumes predictions produced by ``diagnose_roadsnake_o2o_levels.py``.
It never runs training or reads Test. P3-small candidates are P3 detections whose
predicted box is COCO-small, plus P3 detections overlapping a small GT by IoU>=0.1.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import statistics
from typing import Any

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

SMALL_AREA = 32**2
LEVELS = ("P3", "P4", "P5")
ERROR_TYPES = (
    "tp_small",
    "cross_level_competition",
    "duplicate_p3",
    "class_confusion",
    "localization",
    "background",
    "other_overlap",
    "ignored_non_small_match",
)


def parse_named_paths(values: list[str]) -> dict[str, Path]:
    output: dict[str, Path] = {}
    for value in values:
        name, separator, raw_path = value.partition("=")
        path = Path(raw_path).expanduser().resolve()
        if not separator or not name or name in output or not path.is_file():
            raise ValueError(f"Expected unique NAME=FILE, got {value}")
        output[name] = path
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def iou_xywh(left: list[float], right: list[float]) -> float:
    lx, ly, lw, lh = left
    rx, ry, rw, rh = right
    intersection = max(0.0, min(lx + lw, rx + rw) - max(lx, rx)) * max(
        0.0, min(ly + lh, ry + rh) - max(ly, ry)
    )
    union = lw * lh + rw * rh - intersection
    return intersection / union if union > 0 else 0.0


def truncate_coco_maxdet(predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for prediction in predictions:
        grouped[(prediction["image_id"], prediction["category_id"])].append(prediction)
    return [
        prediction
        for values in grouped.values()
        for prediction in sorted(values, key=lambda row: row["score"], reverse=True)[:100]
    ]


def greedy_combined_matches(
    predictions: list[dict[str, Any]], ground_truth: COCO
) -> tuple[dict[int, int], dict[int, dict[str, Any]]]:
    """Approximate the non-crowd COCO IoU=.50 assignment for attribution."""
    by_group: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for prediction in predictions:
        by_group[(prediction["image_id"], prediction["category_id"])].append(prediction)
    targets: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for target in ground_truth.anns.values():
        if not target.get("iscrowd", 0):
            targets[(target["image_id"], target["category_id"])].append(target)

    match_by_prediction: dict[int, int] = {}
    winner_by_target: dict[int, dict[str, Any]] = {}
    for key, values in by_group.items():
        unmatched = {target["id"]: target for target in targets.get(key, [])}
        for prediction in sorted(values, key=lambda row: row["score"], reverse=True):
            if not unmatched:
                break
            target_id, overlap = max(
                ((target_id, iou_xywh(prediction["bbox"], target["bbox"])) for target_id, target in unmatched.items()),
                key=lambda item: item[1],
            )
            if overlap >= 0.50:
                match_by_prediction[prediction["_id"]] = target_id
                winner_by_target[target_id] = prediction
                del unmatched[target_id]
    return match_by_prediction, winner_by_target


def candidate_error(
    prediction: dict[str, Any],
    targets: list[dict[str, Any]],
    small_targets: list[dict[str, Any]],
    match_by_prediction: dict[int, int],
    winner_by_target: dict[int, dict[str, Any]],
) -> str:
    matched_id = match_by_prediction.get(prediction["_id"])
    if matched_id is not None:
        matched = next(target for target in targets if target["id"] == matched_id)
        return "tp_small" if matched["area"] < SMALL_AREA else "ignored_non_small_match"

    same_class_small = [target for target in small_targets if target["category_id"] == prediction["category_id"]]
    same_overlaps = [(target, iou_xywh(prediction["bbox"], target["bbox"])) for target in same_class_small]
    if same_overlaps:
        target, overlap = max(same_overlaps, key=lambda item: item[1])
        if overlap >= 0.50:
            winner = winner_by_target.get(target["id"])
            return "cross_level_competition" if winner is not None and winner["level"] != "P3" else "duplicate_p3"

    all_small_overlaps = [(target, iou_xywh(prediction["bbox"], target["bbox"])) for target in small_targets]
    if all_small_overlaps and max(overlap for _, overlap in all_small_overlaps) >= 0.50:
        return "class_confusion"
    if same_overlaps and max(overlap for _, overlap in same_overlaps) >= 0.10:
        return "localization"
    all_overlaps = [iou_xywh(prediction["bbox"], target["bbox"]) for target in targets]
    if not all_overlaps or max(all_overlaps) < 0.10:
        return "background"
    return "other_overlap"


def precision_recall_rows(model: str, ground_truth: COCO, predictions: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, float]]:
    p3 = [{key: value for key, value in row.items() if not key.startswith("_")} for row in predictions if row["level"] == "P3"]
    detections = ground_truth.loadRes(p3)
    evaluator = COCOeval(ground_truth, detections, "bbox")
    evaluator.params.imgIds = sorted(ground_truth.imgs)
    evaluator.params.catIds = sorted(ground_truth.cats)
    evaluator.evaluate()
    evaluator.accumulate()
    area_index = list(evaluator.params.areaRngLbl).index("small")
    iou_index = int(np.argmin(np.abs(evaluator.params.iouThrs - 0.50)))
    precision = evaluator.eval["precision"][iou_index, :, :, area_index, -1]
    rows = []
    for recall_index, recall in enumerate(evaluator.params.recThrs):
        values = precision[recall_index]
        values = values[values > -1]
        rows.append({"model": model, "recall": float(recall), "precision": float(values.mean()) if values.size else None})
    valid = evaluator.eval["precision"][:, :, :, area_index, -1]
    valid = valid[valid > -1]
    ap_small = float(valid.mean()) if valid.size else float("nan")
    recall_values = evaluator.eval["recall"][iou_index, :, area_index, -1]
    recall_values = recall_values[recall_values > -1]
    recall50 = float(recall_values.mean()) if recall_values.size else float("nan")
    return rows, {"AP_small": ap_small, "P3_small_recall50": recall50}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction", action="append", required=True, metavar="NAME=FILE")
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prediction_paths = parse_named_paths(args.prediction)
    args.output = args.output.resolve()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)

    ground_truth = COCO(str(args.annotations.resolve()))
    targets_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for target in ground_truth.anns.values():
        if not target.get("iscrowd", 0):
            targets_by_image[target["image_id"]].append(target)
    category_names = {category_id: category["name"] for category_id, category in ground_truth.cats.items()}

    decile_rows: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []
    class_rows: list[dict[str, Any]] = []
    pr_rows: list[dict[str, Any]] = []
    gt_score_rows: list[dict[str, Any]] = []
    winner_rows: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    gt_scores_by_model: dict[str, dict[int, float]] = {}

    for model, prediction_path in prediction_paths.items():
        raw = json.loads(prediction_path.read_text(encoding="utf-8"))
        predictions = truncate_coco_maxdet([{**row, "_id": index} for index, row in enumerate(raw)])
        match_by_prediction, winner_by_target = greedy_combined_matches(predictions, ground_truth)
        candidates: list[dict[str, Any]] = []
        for prediction in predictions:
            if prediction["level"] != "P3":
                continue
            targets = targets_by_image[prediction["image_id"]]
            small_targets = [target for target in targets if target["area"] < SMALL_AREA]
            predicted_area = prediction["bbox"][2] * prediction["bbox"][3]
            max_small_iou = max((iou_xywh(prediction["bbox"], target["bbox"]) for target in small_targets), default=0.0)
            if predicted_area >= SMALL_AREA and max_small_iou < 0.10:
                continue
            candidates.append(
                {
                    **prediction,
                    "error": candidate_error(
                        prediction, targets, small_targets, match_by_prediction, winner_by_target
                    ),
                }
            )

        candidates.sort(key=lambda row: row["score"])
        for rank, candidate in enumerate(candidates):
            candidate["decile"] = min(10, rank * 10 // max(1, len(candidates)) + 1)
        for decile in range(1, 11):
            selected = [row for row in candidates if row["decile"] == decile]
            counts = Counter(row["error"] for row in selected)
            evaluated = len(selected) - counts["ignored_non_small_match"]
            decile_rows.append(
                {
                    "model": model,
                    "decile": decile,
                    "count": len(selected),
                    "score_min": min((row["score"] for row in selected), default=None),
                    "score_mean": statistics.fmean(row["score"] for row in selected) if selected else None,
                    "score_max": max((row["score"] for row in selected), default=None),
                    "precision_small": counts["tp_small"] / evaluated if evaluated else None,
                    **{error: counts[error] for error in ERROR_TYPES},
                }
            )

        counts = Counter(row["error"] for row in candidates)
        high = [row for row in candidates if row["decile"] == 10]
        high_counts = Counter(row["error"] for row in high)
        for scope, scope_counts, total in (("all_deciles", counts, len(candidates)), ("top_decile", high_counts, len(high))):
            for error in ERROR_TYPES:
                error_rows.append(
                    {
                        "model": model,
                        "scope": scope,
                        "error": error,
                        "count": scope_counts[error],
                        "share": scope_counts[error] / total if total else 0.0,
                    }
                )
        for category_id, category_name in category_names.items():
            selected = [row for row in candidates if row["category_id"] == category_id]
            selected_counts = Counter(row["error"] for row in selected)
            evaluated = len(selected) - selected_counts["ignored_non_small_match"]
            class_rows.append(
                {
                    "model": model,
                    "category": category_name,
                    "candidates": len(selected),
                    "precision_small": selected_counts["tp_small"] / evaluated if evaluated else None,
                    **{error: selected_counts[error] for error in ERROR_TYPES},
                }
            )

        small_targets = [target for target in ground_truth.anns.values() if target["area"] < SMALL_AREA]
        p3_by_image_category: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
        for prediction in predictions:
            if prediction["level"] == "P3":
                p3_by_image_category[(prediction["image_id"], prediction["category_id"])].append(prediction)
        gt_scores: dict[int, float] = {}
        for target in small_targets:
            eligible = [
                prediction
                for prediction in p3_by_image_category[(target["image_id"], target["category_id"])]
                if iou_xywh(prediction["bbox"], target["bbox"]) >= 0.50
            ]
            if eligible:
                best = max(eligible, key=lambda row: row["score"])
                gt_scores[target["id"]] = best["score"]
                gt_score_rows.append(
                    {
                        "model": model,
                        "gt_id": target["id"],
                        "category": category_names[target["category_id"]],
                        "best_p3_score": best["score"],
                    }
                )
            winner = winner_by_target.get(target["id"])
            winner_rows.append(
                {
                    "model": model,
                    "gt_id": target["id"],
                    "category": category_names[target["category_id"]],
                    "matched": winner is not None,
                    "winning_level": winner["level"] if winner is not None else "unmatched",
                    "winning_score": winner["score"] if winner is not None else None,
                }
            )
        gt_scores_by_model[model] = gt_scores
        model_pr_rows, pr_summary = precision_recall_rows(model, ground_truth, predictions)
        pr_rows.extend(model_pr_rows)
        tp_scores = [row["score"] for row in candidates if row["error"] == "tp_small"]
        summaries[model] = {
            "p3_small_candidates": len(candidates),
            "error_counts": dict(counts),
            "top_decile_error_counts": dict(high_counts),
            "tp_small_score_mean": statistics.fmean(tp_scores) if tp_scores else None,
            "tp_small_score_median": statistics.median(tp_scores) if tp_scores else None,
            "small_gt_with_p3_iou50": len(gt_scores),
            "small_gt_total": len(small_targets),
            **pr_summary,
        }

    paired_rows: list[dict[str, Any]] = []
    model_names = list(prediction_paths)
    reference = model_names[0]
    for model in model_names[1:]:
        common = sorted(set(gt_scores_by_model[reference]) & set(gt_scores_by_model[model]))
        deltas = [gt_scores_by_model[model][target_id] - gt_scores_by_model[reference][target_id] for target_id in common]
        paired_rows.append(
            {
                "reference": reference,
                "model": model,
                "common_small_gt": len(common),
                "mean_score_delta": statistics.fmean(deltas) if deltas else None,
                "median_score_delta": statistics.median(deltas) if deltas else None,
                "fraction_score_decreased": sum(delta < 0 for delta in deltas) / len(deltas) if deltas else None,
            }
        )

    write_csv(args.output / "confidence_deciles.csv", decile_rows)
    write_csv(args.output / "error_composition.csv", error_rows)
    write_csv(args.output / "per_class_errors.csv", class_rows)
    write_csv(args.output / "p3_small_pr_curve.csv", pr_rows)
    write_csv(args.output / "small_gt_best_p3_scores.csv", gt_score_rows)
    write_csv(args.output / "small_gt_winning_levels.csv", winner_rows)
    write_csv(args.output / "paired_gt_score_deltas.csv", paired_rows)
    report = {
        "protocol": {
            "split": "val_only",
            "annotations": str(args.annotations.resolve()),
            "small_area": SMALL_AREA,
            "candidate_scope": "P3 and (predicted_area<1024 or max_small_gt_iou>=0.1)",
            "matching": "score-ordered class-aware IoU>=0.5, COCO maxDet100 per image/category",
            "prediction_files": {name: str(path) for name, path in prediction_paths.items()},
        },
        "models": summaries,
        "paired_gt_score_deltas": paired_rows,
    }
    (args.output / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
