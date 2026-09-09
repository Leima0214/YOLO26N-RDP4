"""Training-free diagnostic route for the official YOLO12n SVRDD7 baseline.

The script reads only the frozen validation split and an official non-end2end
Detect checkpoint. It preserves the canonical evaluator thresholds and writes
all artifacts to a new, independent directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT / "scripts"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from rs_mid_bootstrap import guard_optional_visualization_imports  # noqa: E402

guard_optional_visualization_imports()

from rs_mid_o2m import NAMES, ValCOCOEvaluator, write_json  # noqa: E402
from ultralytics import __version__ as ultralytics_version  # noqa: E402
from ultralytics.nn.modules.head import Detect  # noqa: E402
from ultralytics.nn.tasks import load_checkpoint  # noqa: E402
from ultralytics.utils.metrics import box_iou  # noqa: E402
from ultralytics.utils.nms import non_max_suppression  # noqa: E402
from ultralytics.utils.ops import scale_boxes, xywh2xyxy  # noqa: E402
from ultralytics.utils.torch_utils import get_flops, select_device  # noqa: E402


ASPECT_BUCKETS = ("[1,2)", "[2,4)", "[4,8)", "[8,16)", "[16,inf)")
SCORE_BUCKETS = ((0.25, 0.40), (0.40, 0.60), (0.60, 0.80), (0.80, 1.000001))
IOU_SCORE_BUCKETS = ((0.0, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.000001))
LEVELS = ("P3", "P4", "P5")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--canonical-metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--score-floor", type=float, default=0.001)
    parser.add_argument("--error-conf", type=float, default=0.25)
    parser.add_argument("--nms-iou", type=float, default=0.70)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--max-nms", type=int, default=30000)
    parser.add_argument("--case-limit", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--max-images", type=int, default=0)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def git_value(*args: str) -> str:
    result = subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else f"unavailable: {result.stderr.strip()}"


def aspect_bucket(ratio: float) -> str:
    if ratio < 2:
        return ASPECT_BUCKETS[0]
    if ratio < 4:
        return ASPECT_BUCKETS[1]
    if ratio < 8:
        return ASPECT_BUCKETS[2]
    if ratio < 16:
        return ASPECT_BUCKETS[3]
    return ASPECT_BUCKETS[4]


def size_bucket(area: float) -> str:
    if area < 32**2:
        return "small"
    if area < 96**2:
        return "medium"
    return "large"


def quartile_bucket(area: float, cuts: np.ndarray) -> str:
    return ("Q1", "Q2", "Q3", "Q4")[int(np.searchsorted(cuts, area, side="right"))]


def safe_mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def safe_median(values: list[float]) -> float | None:
    return float(np.median(values)) if values else None


def correlation(x: list[float], y: list[float], rank: bool = False) -> float | None:
    if len(x) < 3:
        return None
    xa, ya = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if rank:
        def ranks(v: np.ndarray) -> np.ndarray:
            order = np.argsort(v, kind="mergesort")
            out = np.empty(len(v), dtype=float)
            i = 0
            while i < len(v):
                j = i + 1
                while j < len(v) and v[order[j]] == v[order[i]]:
                    j += 1
                out[order[i:j]] = (i + j - 1) / 2
                i = j
            return out
        xa, ya = ranks(xa), ranks(ya)
    if np.ptp(xa) == 0 or np.ptp(ya) == 0:
        return None
    value = float(np.corrcoef(xa, ya)[0, 1])
    return value if math.isfinite(value) else None


def greedy_matches(pred: torch.Tensor, gt_boxes: torch.Tensor, gt_cls: torch.Tensor, threshold: float) -> dict[int, int]:
    """Map GT index to prediction index using score-ordered, class-correct matching."""
    matched: dict[int, int] = {}
    used: set[int] = set()
    if not len(pred) or not len(gt_boxes):
        return matched
    ious = box_iou(pred[:, :4], gt_boxes)
    for pi in pred[:, 4].argsort(descending=True).tolist():
        same = torch.where(gt_cls == int(pred[pi, 5]))[0]
        same = torch.tensor([int(x) for x in same.tolist() if int(x) not in used], device=pred.device)
        if not len(same):
            continue
        values = ious[pi, same]
        local = int(values.argmax())
        if float(values[local]) >= threshold:
            gi = int(same[local])
            matched[gi] = pi
            used.add(gi)
    return matched


def group_summary(rows: list[dict[str, Any]], dimension: str, values: list[str]) -> list[dict[str, Any]]:
    output = []
    for value in values:
        items = rows if value == "all" else [row for row in rows if str(row[dimension]) == value]
        if not items:
            continue
        output.append({
            "dimension": dimension,
            "value": value,
            "n_gt": len(items),
            "recall_50": safe_mean([row["hit50"] for row in items]),
            "recall_75": safe_mean([row["hit75"] for row in items]),
            "mean_best_raw_iou": safe_mean([row["best_raw_iou"] for row in items]),
            "median_best_raw_iou": safe_median([row["best_raw_iou"] for row in items]),
            "good_candidate_rate": safe_mean([row["raw_good50"] for row in items]),
            "high_quality_candidate_rate": safe_mean([row["raw_good75"] for row in items]),
            "final_tp_rate": safe_mean([row["hit50"] for row in items]),
            "final_miss_rate": 1.0 - float(safe_mean([row["hit50"] for row in items]) or 0.0),
            "mean_best_candidate_score": safe_mean([row["best_raw_correct_score"] for row in items]),
            "confidence_survival": safe_mean([row["confidence_survival50"] for row in items]),
            "nms_survival": safe_mean([row["nms_survival50"] for row in items if row["prefilter_good50"]]),
        })
    return output


def prediction_error_rows(final: torch.Tensor, gt_boxes: torch.Tensor, gt_cls: torch.Tensor, error_conf: float) -> list[dict[str, Any]]:
    selected = final[final[:, 4] >= error_conf]
    if not len(selected):
        return []
    ious = box_iou(selected[:, :4], gt_boxes) if len(gt_boxes) else torch.zeros((len(selected), 0), device=selected.device)
    used: set[int] = set()
    rows = []
    for pi in selected[:, 4].argsort(descending=True).tolist():
        cls = int(selected[pi, 5])
        same = torch.where(gt_cls == cls)[0]
        same_values = ious[pi, same] if len(same) else torch.zeros(0, device=selected.device)
        best_same = float(same_values.max()) if len(same_values) else 0.0
        best_same_gt = int(same[int(same_values.argmax())]) if len(same_values) else -1
        best_any = float(ious[pi].max()) if len(gt_boxes) else 0.0
        best_any_gt = int(ious[pi].argmax()) if len(gt_boxes) else -1
        if best_same >= 0.5 and best_same_gt not in used:
            kind = "TP"
            used.add(best_same_gt)
        elif best_same >= 0.5:
            kind = "duplicate_prediction"
        elif best_any >= 0.5 and best_any_gt >= 0 and int(gt_cls[best_any_gt]) != cls:
            kind = "classification_error"
        elif best_same >= 0.1:
            kind = "localization_error"
        else:
            kind = "background_false_positive"
        rows.append({"prediction_index": pi, "class": NAMES[cls], "score": float(selected[pi, 4]),
                     "kind": kind, "best_same_iou": best_same, "best_any_iou": best_any})
    return rows


def render_case(image_path: Path, gt: list[dict[str, Any]], preds: list[dict[str, Any]], title: str, output: Path) -> bool:
    image = cv2.imread(str(image_path))
    if image is None:
        return False
    for item in gt:
        x1, y1, x2, y2 = [round(float(x)) for x in item["box"]]
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 200, 0), 2)
        cv2.putText(image, f"GT {item['class']}", (x1, max(18, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 200, 0), 1)
    for item in preds:
        x1, y1, x2, y2 = [round(float(x)) for x in item["box"]]
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        label = f"{item['class']} {item['score']:.2f} IoU={item.get('iou', 0):.2f}"
        cv2.putText(image, label, (x1, min(image.shape[0] - 5, max(18, y1 + 16))), cv2.FONT_HERSHEY_SIMPLEX, .42, (0, 0, 255), 1)
    cv2.putText(image, title, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, .62, (255, 180, 0), 2)
    output.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(output), image))


def markdown_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]], percent: set[str] | None = None) -> str:
    percent = percent or set()
    lines = ["| " + " | ".join(label for _, label in columns) + " |", "|" + "|".join("---:" for _ in columns) + "|"]
    for row in rows:
        cells = []
        for key, _ in columns:
            value = row.get(key)
            if value is None or value == "":
                cells.append("—")
            elif isinstance(value, float):
                cells.append(f"{100 * value:.2f}" if key in percent else f"{value:.3f}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    for key in ("weights", "run_dir", "data", "canonical_metrics", "output"):
        setattr(args, key, getattr(args, key).expanduser().resolve())
    for path in (args.weights, args.run_dir, args.data, args.canonical_metrics):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)

    evaluator = ValCOCOEvaluator(args.data, args.imgsz, args.batch, args.workers, native_o2m=True)
    test = evaluator.data.get("test")
    if not test:
        raise RuntimeError("Dataset YAML has no explicit sealed Test path")
    all_areas = np.asarray([float(ann.get("area") or ann["bbox"][2] * ann["bbox"][3]) for ann in evaluator.gt.anns.values()])
    cuts = np.quantile(all_areas, [0.25, 0.50, 0.75])

    model, _ = load_checkpoint(str(args.weights), device=select_device(args.device, verbose=False))
    model.float().eval()
    head = model.model[-1]
    if type(head) is not Detect or head.end2end or head.nc != len(NAMES):
        raise RuntimeError("Expected an unfused official seven-class YOLO11/YOLO12 Detect checkpoint")
    if list(model.names.values()) if isinstance(model.names, dict) else list(model.names) != list(NAMES):
        names = list(model.names.values()) if isinstance(model.names, dict) else list(model.names)
        if names != list(NAMES):
            raise RuntimeError(f"Checkpoint class order mismatch: {names}")

    gt_rows: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []
    candidate_pairs: list[dict[str, Any]] = []
    image_records: dict[str, dict[str, Any]] = {}
    seen = 0
    device = next(model.parameters()).device
    with torch.inference_mode():
        for batch in evaluator.loader:
            images = batch["img"].to(device, non_blocking=True).float() / 255.0
            output = model(images)
            raw = output[1] if isinstance(output, tuple) else None
            if not isinstance(raw, dict) or not {"boxes", "scores", "feats"}.issubset(raw):
                raise RuntimeError("Official raw prediction dictionary is unavailable")
            decoded = head._inference(raw).float()
            raw_xyxy = xywh2xyxy(decoded[:, :4].permute(0, 2, 1))
            raw_scores = decoded[:, 4:].permute(0, 2, 1)
            lengths = [int(feat.shape[-2] * feat.shape[-1]) for feat in raw["feats"]]
            if len(lengths) != 3:
                raise RuntimeError(f"Expected P3/P4/P5, got {lengths}")
            level_by_anchor = torch.cat([torch.full((n,), i, device=device, dtype=torch.long) for i, n in enumerate(lengths)])
            nms_input = decoded.clone()
            nms_out, kept_anchor = non_max_suppression(
                nms_input, args.score_floor, args.nms_iou, nc=head.nc, multi_label=True,
                max_det=args.max_det, max_nms=args.max_nms, return_idxs=True,
            )
            h, w = images.shape[-2:]
            for bi in range(images.shape[0]):
                if args.max_images and seen >= args.max_images:
                    break
                stem = Path(batch["im_file"][bi]).stem
                image_id = evaluator.ids_by_stem[stem]
                image_path = Path(batch["im_file"][bi]).resolve()
                mask = batch["batch_idx"].view(-1).long() == bi
                classes = batch["cls"].view(-1).long()[mask].to(device)
                boxes = xywh2xyxy(batch["bboxes"][mask].float().to(device))
                boxes *= torch.tensor((w, h, w, h), device=device)
                boxes_orig = boxes.clone()
                if len(boxes_orig):
                    scale_boxes((h, w), boxes_orig, batch["ori_shape"][bi], batch["ratio_pad"][bi])
                final = nms_out[bi]
                anchors = kept_anchor[bi].long()
                operating = final[final[:, 4] >= args.error_conf]
                final_orig = final[:, :4].clone()
                if len(final_orig):
                    scale_boxes((h, w), final_orig, batch["ori_shape"][bi], batch["ratio_pad"][bi])
                match50 = greedy_matches(final, boxes, classes, 0.50)
                match75 = greedy_matches(final, boxes, classes, 0.75)
                match50_operating = greedy_matches(operating, boxes, classes, 0.50)
                iou_raw = box_iou(raw_xyxy[bi], boxes) if len(boxes) else torch.zeros((len(raw_xyxy[bi]), 0), device=device)
                iou_final = box_iou(final[:, :4], boxes) if len(final) and len(boxes) else torch.zeros((len(final), len(boxes)), device=device)
                # Assign each raw anchor-class candidate to at most one same-class
                # GT. This prevents a candidate for another instance of the same
                # class from corrupting the current GT's score-IoU ranking.
                assigned_anchor_by_gt: dict[int, torch.Tensor] = {}
                for cls_value in classes.unique().tolist():
                    class_gts = torch.where(classes == int(cls_value))[0]
                    class_ious = iou_raw[:, class_gts]
                    nearest_iou, nearest_local = class_ious.max(dim=1)
                    for local_index, gt_index in enumerate(class_gts.tolist()):
                        assigned_anchor_by_gt[int(gt_index)] = (nearest_local == local_index) & (nearest_iou >= 0.10)
                predictions_visual = []
                for pi, pred in enumerate(final):
                    predictions_visual.append({"box": final_orig[pi].cpu().tolist(), "class": NAMES[int(pred[5])],
                                               "score": float(pred[4]), "raw_anchor": int(anchors[pi])})
                gt_visual = []
                for gi, (box, cls_tensor) in enumerate(zip(boxes_orig, classes)):
                    cls = int(cls_tensor)
                    bw, bh = float(box[2] - box[0]), float(box[3] - box[1])
                    area = max(0.0, bw * bh)
                    ratio = max(bw / max(bh, 1e-9), bh / max(bw, 1e-9))
                    ious = iou_raw[:, gi]
                    best_idx = int(ious.argmax()) if len(ious) else -1
                    best_iou = float(ious[best_idx]) if best_idx >= 0 else 0.0
                    correct_scores = raw_scores[bi, :, cls]
                    good50 = ious >= 0.50
                    good75 = ious >= 0.75
                    best_score = float(correct_scores[best_idx]) if best_idx >= 0 else 0.0
                    best_pred_cls = int(raw_scores[bi, best_idx].argmax()) if best_idx >= 0 else -1
                    high50_score = float(correct_scores[good50].max()) if bool(good50.any()) else 0.0
                    high75_score = float(correct_scores[good75].max()) if bool(good75.any()) else 0.0
                    assigned_indices = torch.where(assigned_anchor_by_gt.get(gi, torch.zeros_like(ious, dtype=torch.bool)))[0]
                    if len(assigned_indices):
                        assigned_scores = correct_scores[assigned_indices]
                        top_score_idx = int(assigned_indices[int(assigned_scores.argmax())])
                        top_score_iou = float(ious[top_score_idx])
                        top_score = float(correct_scores[top_score_idx])
                        top_score_level = LEVELS[int(level_by_anchor[top_score_idx])]
                    else:
                        top_score_idx, top_score_iou, top_score, top_score_level = -1, 0.0, 0.0, "none"
                    same_final = torch.where(final[:, 5].long() == cls)[0]
                    same_final_ious = iou_final[same_final, gi] if len(same_final) else torch.zeros(0, device=device)
                    final_best_iou = float(same_final_ious.max()) if len(same_final_ious) else 0.0
                    final_best_local = int(same_final_ious.argmax()) if len(same_final_ious) else -1
                    final_best_pi = int(same_final[final_best_local]) if final_best_local >= 0 else -1
                    final_best_score = float(final[final_best_pi, 4]) if final_best_pi >= 0 else 0.0
                    final_best_level = LEVELS[int(level_by_anchor[anchors[final_best_pi]])] if final_best_pi >= 0 else "none"
                    pre50 = bool(good50.any() and high50_score > args.score_floor)
                    final50 = final_best_iou >= 0.50
                    any_final_iou, any_final_cls = 0.0, -1
                    if len(final):
                        any_pi = int(iou_final[:, gi].argmax())
                        any_final_iou = float(iou_final[any_pi, gi])
                        any_final_cls = int(final[any_pi, 5])
                    hit50, hit75 = int(gi in match50), int(gi in match75)
                    if hit50:
                        death = "final_tp"
                    elif best_iou < 0.50:
                        death = "raw_geometry_failure"
                    elif best_iou >= 0.75 and high75_score <= args.score_floor:
                        death = "confidence_filter_loss"
                    elif best_iou >= 0.75 and high75_score < args.error_conf:
                        death = "high_iou_low_score"
                    elif pre50 and not final50:
                        death = "nms_or_maxdet_loss"
                    elif any_final_iou >= 0.50 and any_final_cls != cls and not final50:
                        death = "classification_mismatch"
                    else:
                        death = "final_localization_or_assignment"
                    row = {
                        "image": stem, "image_id": image_id, "gt_index": gi, "class": NAMES[cls],
                        "box_original": json.dumps(box.cpu().tolist()), "area": area, "coco_size": size_bucket(area),
                        "area_quartile": quartile_bucket(area, cuts), "aspect_ratio": ratio,
                        "aspect_bucket": aspect_bucket(ratio), "best_raw_iou": best_iou,
                        "best_raw_correct_score": best_score, "best_raw_predicted_class": NAMES[best_pred_cls],
                        "best_raw_level": LEVELS[int(level_by_anchor[best_idx])], "raw_good50": int(best_iou >= 0.50),
                        "raw_good75": int(best_iou >= 0.75), "prefilter_good50": int(pre50),
                        "prefilter_good75": int(bool(good75.any() and high75_score > args.score_floor)),
                        "max_score_good50": high50_score, "max_score_good75": high75_score,
                        "top_score_iou": top_score_iou, "top_score": top_score,
                        "top_score_level": top_score_level,
                        "final_best_correct_iou": final_best_iou, "final_best_correct_score": final_best_score,
                        "final_best_level": final_best_level, "hit50": hit50, "hit75": hit75,
                        "hit50_operating": int(gi in match50_operating),
                        "confidence_survival50": int(high50_score > args.error_conf),
                        "nms_survival50": int(final50), "death_reason": death,
                    }
                    gt_rows.append(row)
                    gt_visual.append({"box": box.cpu().tolist(), "class": NAMES[cls], "gt_index": gi, "death_reason": death})
                    if len(assigned_indices):
                        order = correct_scores[assigned_indices].topk(min(100, len(assigned_indices))).indices
                        top = assigned_indices[order]
                    else:
                        top = assigned_indices
                    for rank, anchor in enumerate(top.tolist(), 1):
                        candidate_pairs.append({"image": stem, "gt_index": gi, "class": NAMES[cls], "rank": rank,
                                                "score": float(correct_scores[anchor]), "iou": float(ious[anchor]),
                                                "level": LEVELS[int(level_by_anchor[anchor])]})
                per_pred = prediction_error_rows(final, boxes, classes, args.error_conf)
                for row in per_pred:
                    pi = row.pop("prediction_index")
                    row.update({"image": stem, "image_id": image_id, "box_original": json.dumps(final_orig[pi].cpu().tolist()),
                                "source_level": LEVELS[int(level_by_anchor[anchors[pi]])]})
                    error_rows.append(row)
                image_records[stem] = {"path": str(image_path), "gt": gt_visual, "pred": predictions_visual}
                seen += 1
            if args.max_images and seen >= args.max_images:
                break

    expected = min(args.max_images, len(evaluator.gt.imgs)) if args.max_images else len(evaluator.gt.imgs)
    if seen != expected:
        raise RuntimeError(f"Incomplete validation diagnosis: {seen}/{expected}")

    # Add the missed-GT part of the mutually exclusive error decomposition.
    misses = sum(1 for row in gt_rows if not row["hit50_operating"])
    error_counts = Counter(row["kind"] for row in error_rows)
    error_counts["missed_gt"] = misses
    error_total = sum(error_counts.values())
    error_summary = [{"scope": "overall", "error": key, "count": value, "fraction_of_decomposed_events": value / error_total}
                     for key, value in sorted(error_counts.items())]
    for class_name in NAMES:
        class_pred = [row for row in error_rows if row["class"] == class_name]
        counts = Counter(row["kind"] for row in class_pred)
        counts["missed_gt"] = sum(row["class"] == class_name and not row["hit50_operating"] for row in gt_rows)
        total = sum(counts.values())
        error_summary.extend({"scope": class_name, "error": key, "count": value,
                              "fraction_of_decomposed_events": value / total if total else 0.0}
                             for key, value in sorted(counts.items()))

    metrics_payload = json.loads(args.canonical_metrics.read_text(encoding="utf-8"))
    metrics = metrics_payload["metrics"]["current"] if "metrics" in metrics_payload else metrics_payload["current"]
    iou_rows = []
    for scope in ("overall", *NAMES):
        value = metrics["overall"] if scope == "overall" else metrics["per_class"][scope]
        items = gt_rows if scope == "overall" else [row for row in gt_rows if row["class"] == scope]
        iou_rows.append({"class": scope, "AP50": value["AP50"], "AP75": value["AP75"],
                         "AP50_minus_AP75": value["AP50"] - value["AP75"],
                         "Recall50": safe_mean([row["hit50"] for row in items]),
                         "Recall75": safe_mean([row["hit75"] for row in items])})
    iou_rows.sort(key=lambda row: row["AP50_minus_AP75"], reverse=True)

    aspect_distribution = []
    for scope in ("overall", *NAMES):
        source = gt_rows if scope == "overall" else [row for row in gt_rows if row["class"] == scope]
        total = len(source)
        for bucket in ASPECT_BUCKETS:
            items = [row for row in source if row["aspect_bucket"] == bucket]
            aspect_distribution.append({"class": scope, "aspect_bucket": bucket, "n_gt": len(items),
                                        "fraction": len(items) / total if total else 0.0})
    aspect_performance = group_summary(gt_rows, "aspect_bucket", ["all", *ASPECT_BUCKETS])
    for class_name in NAMES:
        for row in group_summary([x for x in gt_rows if x["class"] == class_name], "aspect_bucket", list(ASPECT_BUCKETS)):
            row["class"] = class_name
            aspect_performance.append(row)
    scale_rows = group_summary(gt_rows, "coco_size", ["all", "small", "medium", "large"])
    scale_rows += group_summary(gt_rows, "area_quartile", ["Q1", "Q2", "Q3", "Q4"])

    death_summary = []
    for dimension, values in (("overall", ["all"]), ("class", list(NAMES)),
                              ("aspect_bucket", list(ASPECT_BUCKETS)),
                              ("coco_size", ["small", "medium", "large"]),
                              ("area_quartile", ["Q1", "Q2", "Q3", "Q4"])):
        for value in values:
            items = gt_rows if value == "all" else [row for row in gt_rows if str(row.get(dimension)) == value]
            counts = Counter(row["death_reason"] for row in items)
            death_summary.extend({"dimension": dimension, "value": value, "reason": reason, "count": count,
                                  "fraction": count / len(items) if items else 0.0}
                                 for reason, count in sorted(counts.items()))

    alignment_rows = []
    for scope in ("overall", *NAMES):
        pairs = candidate_pairs if scope == "overall" else [row for row in candidate_pairs if row["class"] == scope]
        items = gt_rows if scope == "overall" else [row for row in gt_rows if row["class"] == scope]
        scores, ious = [row["score"] for row in pairs], [row["iou"] for row in pairs]
        alignment_rows.append({"scope": scope, "kind": "summary", "n_candidates": len(pairs),
                               "pearson": correlation(scores, ious), "spearman": correlation(scores, ious, True),
                               "mean_top_score_iou": safe_mean([row["top_score_iou"] for row in items]),
                               "mean_best_iou": safe_mean([row["best_raw_iou"] for row in items]),
                               "top_score_not_best_quality_rate": safe_mean([int(row["top_score_iou"] + 1e-6 < row["best_raw_iou"])
                                                                               for row in items]),
                               "mean_best_minus_top_score_iou": safe_mean([row["best_raw_iou"] - row["top_score_iou"] for row in items]),
                               "quality_gap_gt005_rate": safe_mean([int(row["best_raw_iou"] - row["top_score_iou"] > 0.05) for row in items]),
                               "quality_gap_gt010_rate": safe_mean([int(row["best_raw_iou"] - row["top_score_iou"] > 0.10) for row in items]),
                               "high_iou_but_low_score_rate": safe_mean([int(row["raw_good75"] and row["max_score_good75"] < args.error_conf)
                                                                          for row in items])})
    for low, high in IOU_SCORE_BUCKETS:
        pairs = [row for row in candidate_pairs if low <= row["iou"] < high]
        alignment_rows.append({"scope": "overall", "kind": f"iou_[{low:.1f},{min(high, 1):.1f}{']' if high > 1 else ')'}",
                               "n_candidates": len(pairs), "mean_score": safe_mean([row["score"] for row in pairs])})

    background_rows = []
    background = [row for row in error_rows if row["kind"] == "background_false_positive"]
    fps = [row for row in error_rows if row["kind"] != "TP"]
    for scope in ("overall", *NAMES):
        scoped_bg = background if scope == "overall" else [row for row in background if row["class"] == scope]
        scoped_fp = fps if scope == "overall" else [row for row in fps if row["class"] == scope]
        for low, high in SCORE_BUCKETS:
            items = [row for row in scoped_bg if low <= row["score"] < high]
            background_rows.append({"class": scope, "score_bucket": f"[{low:.2f},{min(high, 1):.2f}{']' if high > 1 else ')'}",
                                    "background_fp": len(items), "all_fp": len(scoped_fp),
                                    "background_fraction_of_fp": len(scoped_bg) / len(scoped_fp) if scoped_fp else 0.0})

    fpn_rows = []
    for dimension, values in (("overall", ["all"]), ("class", list(NAMES)), ("coco_size", ["small", "medium", "large"]),
                              ("aspect_bucket", list(ASPECT_BUCKETS))):
        for value in values:
            items = gt_rows if value == "all" else [row for row in gt_rows if str(row.get(dimension)) == value]
            for level in LEVELS:
                fpn_rows.append({"dimension": dimension, "value": value, "level": level, "n_gt": len(items),
                                 "best_raw_level_count": sum(row["best_raw_level"] == level for row in items),
                                 "best_raw_level_fraction": safe_mean([int(row["best_raw_level"] == level) for row in items]),
                                 "top_score_level_fraction": safe_mean([int(row["top_score_level"] == level) for row in items]),
                                 "final_best_level_fraction": safe_mean([int(row["final_best_level"] == level) for row in items])})

    # Deterministic galleries. Automated background labels remain review candidates.
    failure_index = []
    gallery = args.output / "figures" / "failure_cases"
    categories = {
        "class_miss": lambda r: not r["hit50"],
        "high_aspect_miss": lambda r: r["aspect_ratio"] >= 8 and not r["hit50"],
        "small_miss": lambda r: r["coco_size"] == "small" and not r["hit50"],
        "high_iou_low_score": lambda r: r["death_reason"] == "high_iou_low_score",
        "nms_competition": lambda r: r["death_reason"] == "nms_or_maxdet_loss",
        "localization_error": lambda r: r["death_reason"] in {"raw_geometry_failure", "final_localization_or_assignment"},
    }
    for category, predicate in categories.items():
        candidates = sorted([row for row in gt_rows if predicate(row)], key=lambda r: (-r["aspect_ratio"], r["image"], r["gt_index"]))
        for index, row in enumerate(candidates[: args.case_limit], 1):
            record = image_records[row["image"]]
            preds = []
            for pred in record["pred"]:
                pred = dict(pred)
                gt_box = torch.tensor(json.loads(row["box_original"]), dtype=torch.float32)[None]
                pred["iou"] = float(box_iou(torch.tensor(pred["box"])[None], gt_box)[0, 0])
                preds.append(pred)
            path = gallery / category / f"{index:02d}_{row['image']}_gt{row['gt_index']}.jpg"
            if render_case(Path(record["path"]), record["gt"], preds, category, path):
                failure_index.append({"category": category, "class": row["class"], "image": row["image"],
                                      "gt_index": row["gt_index"], "path": str(path.relative_to(args.output))})
    bg_gallery = args.output / "figures" / "background_fp"
    for class_name in NAMES:
        candidates = sorted([row for row in background if row["class"] == class_name], key=lambda r: (-r["score"], r["image"]))
        for index, row in enumerate(candidates[: args.case_limit], 1):
            record = image_records[row["image"]]
            pred = {"box": json.loads(row["box_original"]), "class": row["class"], "score": row["score"], "iou": row["best_any_iou"]}
            path = bg_gallery / class_name / f"{index:02d}_{row['image']}.jpg"
            if render_case(Path(record["path"]), record["gt"], [pred], "background FP - manual review", path):
                failure_index.append({"category": "background_fp", "class": class_name, "image": row["image"], "path": str(path.relative_to(args.output))})

    # Baseline freeze manifest from source artifacts, not manual transcription.
    train_rows = read_csv(args.run_dir / "results.csv")
    best_train = max(train_rows, key=lambda row: float(row["metrics/mAP50-95(B)"]))
    train_args = yaml.safe_load((args.run_dir / "args.yaml").read_text(encoding="utf-8"))
    canonical_overall = metrics["overall"]
    try:
        flops = float(get_flops(model, imgsz=args.imgsz))
    except Exception:
        flops = None
    manifest = {
        "git": {"commit": git_value("rev-parse", "HEAD"), "branch": git_value("branch", "--show-current"),
                "status_porcelain": git_value("status", "--porcelain")},
        "artifacts": {"best_pt": str(args.weights), "best_sha256": sha256(args.weights),
                      "last_pt": str(args.run_dir / "weights" / "last.pt"),
                      "last_sha256": sha256(args.run_dir / "weights" / "last.pt"),
                      "results_csv": str(args.run_dir / "results.csv"), "args_yaml": str(args.run_dir / "args.yaml"),
                      "data_yaml": str(args.data)},
        "training": {key: train_args.get(key) for key in ("epochs", "imgsz", "batch", "seed", "optimizer", "lr0", "lrf",
                                                                  "momentum", "weight_decay", "warmup_epochs", "mosaic", "mixup",
                                                                  "copy_paste", "close_mosaic")},
        "software": {"ultralytics": ultralytics_version, "python": sys.version, "torch": torch.__version__},
        "model": {"parameters": sum(p.numel() for p in model.parameters()), "GFLOPs": flops,
                  "detect_type": type(head).__name__, "end2end": head.end2end, "strides": head.stride.cpu().tolist()},
        "selection": {"best_epoch": int(float(best_train["epoch"])),
                      "native_AP50": float(best_train["metrics/mAP50(B)"]),
                      "native_AP50_95": float(best_train["metrics/mAP50-95(B)"])},
        "canonical_val": canonical_overall,
        "protocol": {"split": "val", "images": seen, "imgsz": args.imgsz, "batch": args.batch,
                     "score_floor": args.score_floor, "error_conf": args.error_conf, "nms_iou": args.nms_iou,
                     "max_det": args.max_det, "test_accessed": False, "class_names": NAMES,
                     "class_note": "D00/D10/D20/D40 are absent from SVRDD7; no mapping was invented."},
    }

    write_json(args.output / "b0_metrics.json", metrics)
    write_json(args.output / "manifest.json", manifest)
    write_json(args.output / "image_records.json", image_records)
    write_csv(args.output / "per_gt_diagnostics.csv", gt_rows)
    write_csv(args.output / "raw_candidate_score_iou.csv", candidate_pairs)
    write_csv(args.output / "error_decomposition.csv", error_summary)
    write_csv(args.output / "iou_threshold_analysis.csv", iou_rows)
    write_csv(args.output / "aspect_ratio_distribution.csv", aspect_distribution)
    write_csv(args.output / "aspect_ratio_performance.csv", aspect_performance)
    write_csv(args.output / "scale_analysis.csv", scale_rows)
    write_csv(args.output / "candidate_survival.csv", death_summary)
    write_csv(args.output / "score_iou_alignment.csv", alignment_rows)
    write_csv(args.output / "background_fp.csv", background_rows)
    write_csv(args.output / "fpn_level_analysis.csv", fpn_rows)
    write_csv(args.output / "prediction_errors_raw.csv", error_rows)
    write_csv(args.output / "failure_case_index.csv", failure_index)

    per_class = []
    for name in NAMES:
        value = metrics["per_class"][name]
        items = [row for row in gt_rows if row["class"] == name]
        per_class.append({"class": name, "AP50_95": value["AP"], "AP50": value["AP50"], "AP75": value["AP75"],
                          "Recall50": safe_mean([row["hit50"] for row in items]), "Recall75": safe_mean([row["hit75"] for row in items]),
                          "AR100": value["AR100"]})
    write_csv(args.output / "per_class_metrics.csv", per_class)

    # Evidence-led route decisions.
    overall_align = next(row for row in alignment_rows if row["scope"] == "overall" and row["kind"] == "summary")
    aspect_overall = {row["value"]: row for row in aspect_performance if row.get("class") is None}
    compact = aspect_overall.get("[1,2)", {})
    extreme = aspect_overall.get("[8,16)", {})
    extreme2 = aspect_overall.get("[16,inf)", {})
    extreme_items = [x for x in (extreme, extreme2) if x]
    extreme_r75 = safe_mean([x["recall_75"] for x in extreme_items if x["recall_75"] is not None])
    extreme_iou = safe_mean([x["mean_best_raw_iou"] for x in extreme_items if x["mean_best_raw_iou"] is not None])
    aspect_drop = (compact.get("recall_75") is not None and extreme_r75 is not None and compact["recall_75"] - extreme_r75 >= 0.08
                   and compact.get("mean_best_raw_iou") is not None and extreme_iou is not None
                   and compact["mean_best_raw_iou"] - extreme_iou >= 0.05)
    bg_fraction = len(background) / len(fps) if fps else 0.0
    small = next((row for row in scale_rows if row["dimension"] == "coco_size" and row["value"] == "small"), None)
    medium = next((row for row in scale_rows if row["dimension"] == "coco_size" and row["value"] == "medium"), None)
    p2_go = bool(small and medium and small["good_candidate_rate"] + 0.15 < medium["good_candidate_rate"] and small["recall_50"] + 0.15 < medium["recall_50"])
    alignment_go = bool(overall_align["spearman"] is not None and overall_align["spearman"] < 0.35
                        and overall_align["quality_gap_gt010_rate"] > 0.20
                        and overall_align["high_iou_but_low_score_rate"] > 0.05)
    routes = [
        {"route": "A Strip Localization Head", "decision": "GO" if aspect_drop else "NO-GO",
         "evidence": f"compact R75={compact.get('recall_75')}; extreme R75={extreme_r75}; compact raw IoU={compact.get('mean_best_raw_iou')}; extreme raw IoU={extreme_iou}"},
        {"route": "B Task / Quality-Aligned Head", "decision": "GO" if alignment_go else "NO-GO",
         "evidence": f"Spearman={overall_align['spearman']}; IoU-gap>0.10={overall_align['quality_gap_gt010_rate']}; high-IoU low-score={overall_align['high_iou_but_low_score_rate']}"},
        {"route": "C SET / Background Spectral Suppression", "decision": "REVIEW" if bg_fraction >= 0.35 else "NO-GO",
         "evidence": f"automatic background FP fraction={bg_fraction:.3f}; visual categories require human review"},
        {"route": "D GFB / PKI-lite / Multi-kernel", "decision": "LOW PRIORITY",
         "evidence": "scale heterogeneity alone cannot identify receptive-field causality"},
        {"route": "E P2 / Tiny Object Layer", "decision": "GO" if p2_go else "NO-GO",
         "evidence": f"small candidate={small.get('good_candidate_rate') if small else None}, small R50={small.get('recall_50') if small else None}, medium candidate={medium.get('good_candidate_rate') if medium else None}"},
        {"route": "F DSConv / DCNv4 / Geometry-aware backbone", "decision": "LOW PRIORITY" if aspect_drop else "NO-GO",
         "evidence": "box-only diagnosis cannot prove curved morphology; prefer the simpler supported geometry test first"},
        {"route": "G Class-balanced / Hard-sample Loss", "decision": "AUXILIARY ONLY",
         "evidence": "per-class gaps are descriptive; localization and frequency explanations must be separated"},
    ]
    write_csv(args.output / "route_decision.csv", routes)

    pct = {"AP50_95", "AP50", "AP75", "Recall50", "Recall75", "AR100"}
    per_class_table = markdown_table(per_class, [("class", "Class"), ("AP50_95", "AP"), ("AP50", "AP50"),
                                                        ("AP75", "AP75"), ("Recall50", "R@.50"), ("Recall75", "R@.75")], pct)
    iou_table = markdown_table(iou_rows, [("class", "Class"), ("AP50", "AP50"), ("AP75", "AP75"),
                                                   ("AP50_minus_AP75", "Gap"), ("Recall50", "R@.50"), ("Recall75", "R@.75")],
                               {"AP50", "AP75", "AP50_minus_AP75", "Recall50", "Recall75"})
    error_table = markdown_table([x for x in error_summary if x["scope"] == "overall"],
                                 [("error", "Error"), ("count", "Count"), ("fraction_of_decomposed_events", "Share")],
                                 {"fraction_of_decomposed_events"})
    route_table = markdown_table(routes, [("route", "Route"), ("decision", "Decision"), ("evidence", "Evidence")])

    report = f"""# YOLO12n B0 Diagnostic Report

## 1. Baseline

FACT: official YOLO12n, SVRDD7 Val only, {seen} images, `best.pt`, native Detect/DFL/loss, no model or training change. Test was not accessed. Canonical AP={100*canonical_overall['AP']:.3f}, AP50={100*canonical_overall['AP50']:.3f}, AP75={100*canonical_overall['AP75']:.3f}.

`D00/D10/D20/D40` are not SVRDD7 category names. The actual classes are `{', '.join(NAMES)}`; no cross-dataset label mapping was assumed.

## 2. Per-class performance

{per_class_table}

## 3. Error decomposition

At confidence >= {args.error_conf:.2f}; automatic IoU categories are mutually exclusive. Background cases remain candidates for human review.

{error_table}

## 4. IoU / localization analysis

{iou_table}

## 5. Aspect-ratio analysis

FACT: compact R@.75={compact.get('recall_75')}; extreme R@.75={extreme_r75}; compact mean raw-best IoU={compact.get('mean_best_raw_iou')}; extreme={extreme_iou}.

INFERENCE: {'The joint raw-IoU and strict-recall drop supports an aspect-ratio localization bottleneck.' if aspect_drop else 'The predeclared joint evidence threshold for a high-aspect-ratio localization bottleneck was not met.'}

## 6. Scale analysis

FACT: small good-candidate rate={small.get('good_candidate_rate') if small else None}, small R@.50={small.get('recall_50') if small else None}; medium good-candidate rate={medium.get('good_candidate_rate') if medium else None}, medium R@.50={medium.get('recall_50') if medium else None}.

INFERENCE: {'Small-object raw generation is sufficiently worse to permit a P2 experiment.' if p2_go else 'The strict P2 GO condition was not met.'}

## 7. Candidate survival

FACT: all raw-to-final death reasons, by class/aspect/scale, are in `candidate_survival.csv`. The canonical score floor is {args.score_floor}; diagnostic operating confidence is {args.error_conf}.

## 8. Score-IoU alignment

FACT: top-100 correct-class candidates per GT give Pearson={overall_align['pearson']}, Spearman={overall_align['spearman']}; mean best-IoU minus top-score-IoU gap={overall_align['mean_best_minus_top_score_iou']:.3f}, and {100*overall_align['quality_gap_gt010_rate']:.2f}% of GTs have a gap over 0.10. High-IoU/low-score rate={100*overall_align['high_iou_but_low_score_rate']:.2f}%.

INFERENCE: {'The predeclared evidence supports a quality-alignment experiment.' if alignment_go else 'The full quality-alignment GO condition was not met.'}

## 9. Background FP analysis

FACT: automatically classified background FP are {100*bg_fraction:.2f}% of FP events at confidence >= {args.error_conf}. These include possible annotation ambiguity and are not a measured missing-label rate.

HYPOTHESIS: spectral/background suppression needs manual review of the exported cases before structural authorization.

## 10. FPN-level analysis

FACT: raw anchor provenance is reliable because the native P3/P4/P5 tensors and NMS kept-anchor indices were captured directly. Results are in `fpn_level_analysis.csv`.

## 11. Failure cases

Deterministically selected cases are indexed by `failure_case_index.csv`; background cases are separated under `figures/background_fp/` for manual semantic review.

## 12. Main bottlenecks

FACT: the quantitative bottlenecks are reported above without importing YOLO26/RoadSnake values.

INFERENCE: route priority follows only conditions met by this YOLO12n checkpoint.

## 13. Alternative explanations / limitations

- Box annotations cannot prove that a miss is caused by curvature or texture semantics.
- Automatic background-FP labels may contain real but unannotated damage.
- Raw candidate analysis uses decoded native anchors and correct-class scores; it is not a training-assignment audit.
- This is one seed and one validation split; route evidence authorizes an experiment, not a final causal claim.

## 14. Route decision

{route_table}

## 15. Top-3 next experiments

Candidates are ranked in `route_decision.md` from routes that clear or approach their evidence gates. No training was started.
"""
    (args.output / "diagnostic_report.md").write_text(report, encoding="utf-8")
    (args.output / "README.md").write_text("# YOLO12n B0 diagnosis\n\nRe-run with `scripts/diagnose_yolo12n_b0_route.py`. Test is sealed.\n", encoding="utf-8")
    (args.output / "b0_manifest.md").write_text("# B0 manifest\n\n```json\n" + json.dumps(manifest, indent=2, ensure_ascii=False) + "\n```\n", encoding="utf-8")
    (args.output / "per_class_metrics.md").write_text("# Per-class metrics\n\n" + per_class_table + "\n", encoding="utf-8")
    (args.output / "error_decomposition.md").write_text("# Error decomposition\n\n" + error_table + "\n", encoding="utf-8")
    (args.output / "iou_threshold_analysis.md").write_text("# IoU threshold analysis\n\n" + iou_table + "\n", encoding="utf-8")
    (args.output / "aspect_ratio_analysis.md").write_text("# Aspect-ratio analysis\n\nSee `aspect_ratio_distribution.csv` and `aspect_ratio_performance.csv`.\n", encoding="utf-8")
    (args.output / "scale_analysis.md").write_text("# Scale analysis\n\nSee `scale_analysis.csv`.\n", encoding="utf-8")
    (args.output / "candidate_survival_summary.md").write_text("# Candidate survival\n\nSee `candidate_survival.csv`.\n", encoding="utf-8")
    (args.output / "score_iou_alignment.md").write_text("# Score-IoU alignment\n\nSee section 8 of `diagnostic_report.md`.\n", encoding="utf-8")
    (args.output / "background_fp_analysis.md").write_text("# Background FP analysis\n\nAutomated candidates require human semantic review. See `background_fp.csv`.\n", encoding="utf-8")
    (args.output / "fpn_level_analysis.md").write_text("# FPN-level analysis\n\nDirect P3/P4/P5 raw provenance is in `fpn_level_analysis.csv`.\n", encoding="utf-8")
    (args.output / "failure_case_index.md").write_text("# Failure case index\n\n" + markdown_table(failure_index, [("category", "Category"), ("class", "Class"), ("image", "Image"), ("path", "Path")]) + "\n", encoding="utf-8")

    top = [row for row in routes if row["decision"] == "GO"][:3]
    route_text = "# Route decision\n\n" + route_table + "\n\n## Evidence-ranked next experiments\n"
    for index, row in enumerate(top, 1):
        route_text += f"\n### Top {index}: {row['route']}\n\n- Evidence: {row['evidence']}\n- Target: the measured failure mode named by this route.\n- Priority: {row['decision']}; simpler matched single-change experiment first.\n- Main risk: diagnostic association may not be causal.\n- First experiment: one insertion/change against the frozen YOLO12n B0 protocol.\n- Component: route-specific Backbone/Neck/Head/Loss only.\n- Inference cost: measure rather than assume.\n- Ablation: B0 vs capacity-matched control vs proposed mechanism where applicable.\n- Paper role: candidate only until matched multi-seed evidence.\n"
    if not top:
        route_text += "\nNo route cleared its GO gate.\n"
    elif len(top) < 3:
        route_text += f"\nOnly {len(top)} route(s) cleared a GO gate; weaker routes are not promoted merely to fill a Top-3 list.\n"
    (args.output / "route_decision.md").write_text(route_text, encoding="utf-8")
    write_json(args.output / "diagnostic_summary.json", {"seen_images": seen, "gt": len(gt_rows), "routes": routes,
                                                          "alignment": overall_align, "background_fp_fraction": bg_fraction,
                                                          "aspect_drop_gate": aspect_drop, "p2_gate": p2_go,
                                                          "test_accessed": False})
    print(json.dumps({"output": str(args.output), "images": seen, "gt": len(gt_rows), "routes": routes}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
