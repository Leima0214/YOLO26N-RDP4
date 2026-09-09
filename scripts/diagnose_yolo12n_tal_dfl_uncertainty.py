"""Audit native TAL positives and DFL uncertainty for the frozen YOLO12n SVRDD7 baseline.

This is a Val-only, training-free diagnostic. It reconstructs the exact native
TaskAlignedAssigner used by v8DetectionLoss, records every assigned positive,
and tests whether DFL distribution statistics carry localization-quality
information beyond the native class score, FPN level, and target scale.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT / "scripts"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from rs_mid_bootstrap import guard_optional_visualization_imports  # noqa: E402

guard_optional_visualization_imports()

from rs_mid_o2m import NAMES, ValCOCOEvaluator, write_json  # noqa: E402
from ultralytics.nn.modules.head import Detect  # noqa: E402
from ultralytics.nn.tasks import load_checkpoint  # noqa: E402
from ultralytics.utils.ops import scale_boxes, xywh2xyxy, xyxy2xywh  # noqa: E402
from ultralytics.utils.tal import TaskAlignedAssigner, dist2bbox, make_anchors  # noqa: E402
from ultralytics.utils.torch_utils import select_device  # noqa: E402


LEVELS = ("P3", "P4", "P5")
SIDES = ("left", "top", "right", "bottom")
IOU_BINS = ((0.0, 0.50), (0.50, 0.60), (0.60, 0.70), (0.70, 0.80), (0.80, 0.90), (0.90, 1.000001))
CERTAINTY_KEYS = ("dfl_peak_mean", "dfl_margin_mean", "dfl_neg_entropy_mean", "dfl_neg_variance_mean")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--pair-iou-gap", type=float, default=0.05)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_value(*args: str) -> str:
    result = subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else f"unavailable: {result.stderr.strip()}"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def size_bucket(area: float) -> str:
    if area < 32**2:
        return "small"
    if area < 96**2:
        return "medium"
    return "large"


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def correlation(x: list[float] | np.ndarray, y: list[float] | np.ndarray, *, spearman: bool) -> float | None:
    xa, ya = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    valid = np.isfinite(xa) & np.isfinite(ya)
    xa, ya = xa[valid], ya[valid]
    if len(xa) < 3 or np.ptp(xa) == 0 or np.ptp(ya) == 0:
        return None
    if spearman:
        xa, ya = rankdata(xa), rankdata(ya)
    value = float(np.corrcoef(xa, ya)[0, 1])
    return value if math.isfinite(value) else None


def partial_pearson(rows: list[dict[str, Any]], key: str) -> float | None:
    """Partial correlation with IoU after controlling score, FPN level, and log GT area."""
    if len(rows) < 8:
        return None
    y = np.asarray([row["iou"] for row in rows], dtype=float)
    x = np.asarray([row[key] for row in rows], dtype=float)
    controls = np.asarray(
        [
            [1.0, row["pred_score"], float(row["level"] == "P4"), float(row["level"] == "P5"), row["log_gt_area"]]
            for row in rows
        ],
        dtype=float,
    )
    y_residual = y - controls @ np.linalg.lstsq(controls, y, rcond=None)[0]
    x_residual = x - controls @ np.linalg.lstsq(controls, x, rcond=None)[0]
    return correlation(x_residual, y_residual, spearman=False)


def aligned_iou(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    top_left = torch.maximum(first[:, :2], second[:, :2])
    bottom_right = torch.minimum(first[:, 2:], second[:, 2:])
    intersection = (bottom_right - top_left).clamp_min(0).prod(1)
    area_first = (first[:, 2:] - first[:, :2]).clamp_min(0).prod(1)
    area_second = (second[:, 2:] - second[:, :2]).clamp_min(0).prod(1)
    return intersection / (area_first + area_second - intersection).clamp_min(1e-9)


def preprocess_targets(batch: dict[str, Any], batch_size: int, scale: torch.Tensor, device: torch.device) -> torch.Tensor:
    targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1).to(device)
    if not len(targets):
        return torch.zeros(batch_size, 0, 5, device=device)
    image_indices = targets[:, 0]
    _, counts = image_indices.unique(return_counts=True)
    output = torch.zeros(batch_size, int(counts.max()), 5, device=device)
    for image_index in range(batch_size):
        selected = image_indices == image_index
        if int(selected.sum()):
            output[image_index, : int(selected.sum())] = targets[selected, 1:]
    output[..., 1:5] = xywh2xyxy(output[..., 1:5] * scale)
    return output


def pairwise_accuracy(items: list[dict[str, Any]], key: str, gap: float) -> tuple[float, int]:
    correct, count = 0.0, 0
    for left in range(len(items)):
        for right in range(left + 1, len(items)):
            delta_iou = items[left]["iou"] - items[right]["iou"]
            if abs(delta_iou) < gap:
                continue
            delta_score = items[left][key] - items[right][key]
            if delta_score == 0:
                correct += 0.5
            elif delta_score * delta_iou > 0:
                correct += 1.0
            count += 1
    return (correct / count if count else float("nan")), count


def candidate_summary(rows: list[dict[str, Any]], dimension: str, value: str) -> dict[str, Any]:
    selected = rows if value == "all" else [row for row in rows if row[dimension] == value]
    output: dict[str, Any] = {
        "dimension": dimension,
        "value": value,
        "positive_candidates": len(selected),
        "unique_gt": len({(row["image_id"], row["gt_index"]) for row in selected}),
    }
    if not selected:
        return output
    iou = [row["iou"] for row in selected]
    output.update(
        {
            "mean_iou": float(np.mean(iou)),
            "median_iou": float(np.median(iou)),
            "iou50_rate": float(np.mean(np.asarray(iou) >= 0.50)),
            "iou75_rate": float(np.mean(np.asarray(iou) >= 0.75)),
            "mean_pred_score": float(np.mean([row["pred_score"] for row in selected])),
            "mean_target_score": float(np.mean([row["target_score"] for row in selected])),
            "pred_score_iou_pearson": correlation([row["pred_score"] for row in selected], iou, spearman=False),
            "pred_score_iou_spearman": correlation([row["pred_score"] for row in selected], iou, spearman=True),
            "target_score_iou_spearman": correlation([row["target_score"] for row in selected], iou, spearman=True),
        }
    )
    for key in CERTAINTY_KEYS:
        output[f"{key}_iou_pearson"] = correlation([row[key] for row in selected], iou, spearman=False)
        output[f"{key}_iou_spearman"] = correlation([row[key] for row in selected], iou, spearman=True)
        output[f"{key}_partial_pearson"] = partial_pearson(selected, key)
    return output


def within_gt_rows(candidates: list[dict[str, Any]], pair_gap: float) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        grouped[(row["image_id"], row["gt_index"])].append(row)
    output = []
    rank_keys = ("pred_score", "target_score", *CERTAINTY_KEYS)
    for (image_id, gt_index), items in grouped.items():
        ious = np.asarray([row["iou"] for row in items], dtype=float)
        base = items[0]
        row: dict[str, Any] = {
            "image_id": image_id,
            "image": base["image"],
            "gt_index": gt_index,
            "class": base["class"],
            "size": base["size"],
            "positive_candidates": len(items),
            "levels": "/".join(level for level in LEVELS if any(item["level"] == level for item in items)),
            "best_iou": float(ious.max()),
            "top_score_iou": float(items[int(np.argmax([item["pred_score"] for item in items]))]["iou"]),
            "best_minus_top_score_iou": float(ious.max() - items[int(np.argmax([item["pred_score"] for item in items]))]["iou"]),
            "quality_gap_gt_010": int(ious.max() - items[int(np.argmax([item["pred_score"] for item in items]))]["iou"] > 0.10),
            "high_iou_low_score": int(bool(np.any(ious >= 0.75)) and max((item["pred_score"] for item in items if item["iou"] >= 0.75), default=1.0) < 0.25),
        }
        for key in rank_keys:
            values = [item[key] for item in items]
            row[f"{key}_iou_spearman"] = correlation(values, ious, spearman=True)
            accuracy, pairs = pairwise_accuracy(items, key, pair_gap)
            row[f"{key}_pairwise_accuracy"] = accuracy if math.isfinite(accuracy) else None
            row[f"{key}_pair_count"] = pairs
            row[f"top_{key}_iou"] = float(items[int(np.argmax(values))]["iou"])
        output.append(row)
    return output


def weighted_pairwise(rows: list[dict[str, Any]], key: str) -> float | None:
    numerator = denominator = 0.0
    for row in rows:
        accuracy, count = row.get(f"{key}_pairwise_accuracy"), row.get(f"{key}_pair_count", 0)
        if accuracy is not None and count:
            numerator += float(accuracy) * int(count)
            denominator += int(count)
    return numerator / denominator if denominator else None


def within_summary(rows: list[dict[str, Any]], dimension: str, value: str) -> dict[str, Any]:
    selected = rows if value == "all" else [row for row in rows if row[dimension] == value]
    output: dict[str, Any] = {"dimension": dimension, "value": value, "gt": len(selected)}
    if not selected:
        return output
    output.update(
        {
            "mean_positive_candidates": float(np.mean([row["positive_candidates"] for row in selected])),
            "mean_best_iou": float(np.mean([row["best_iou"] for row in selected])),
            "mean_top_score_iou": float(np.mean([row["top_score_iou"] for row in selected])),
            "mean_best_minus_top_score_iou": float(np.mean([row["best_minus_top_score_iou"] for row in selected])),
            "quality_gap_gt_010_rate": float(np.mean([row["quality_gap_gt_010"] for row in selected])),
            "high_iou_low_score_rate": float(np.mean([row["high_iou_low_score"] for row in selected])),
        }
    )
    for key in ("pred_score", "target_score", *CERTAINTY_KEYS):
        correlations = [row[f"{key}_iou_spearman"] for row in selected if row[f"{key}_iou_spearman"] is not None]
        output[f"{key}_mean_within_gt_spearman"] = float(np.mean(correlations)) if correlations else None
        output[f"{key}_weighted_pairwise_accuracy"] = weighted_pairwise(selected, key)
        output[f"mean_top_{key}_iou"] = float(np.mean([row[f"top_{key}_iou"] for row in selected]))
    return output


def markdown_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    lines = ["| " + " | ".join(label for _, label in columns) + " |", "|" + "|".join("---:" for _ in columns) + "|"]
    for row in rows:
        cells = []
        for key, _ in columns:
            value = row.get(key)
            if value is None:
                cells.append("—")
            elif isinstance(value, float):
                cells.append(f"{value:.4f}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    for key in ("weights", "data", "output"):
        setattr(args, key, getattr(args, key).expanduser().resolve())
    if not args.weights.is_file() or not args.data.is_file():
        raise FileNotFoundError(f"Missing weights/data: {args.weights}, {args.data}")
    actual_sha = sha256(args.weights)
    if actual_sha.lower() != args.expected_sha256.lower():
        raise RuntimeError(f"Checkpoint SHA256 mismatch: {actual_sha}")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)

    evaluator = ValCOCOEvaluator(args.data, args.imgsz, args.batch, args.workers, max_images=args.max_images, native_o2m=True)
    if not evaluator.data.get("test"):
        raise RuntimeError("Dataset YAML must define a sealed Test split")
    device = select_device(args.device, verbose=False)
    model, _ = load_checkpoint(str(args.weights), device=device)
    model.float().eval()
    head = model.model[-1]
    if type(head) is not Detect or head.end2end or head.nc != len(NAMES) or head.reg_max <= 1:
        raise RuntimeError("Expected official non-end2end YOLO12 Detect with DFL")
    if tuple(float(value) for value in head.stride.tolist()) != (8.0, 16.0, 32.0):
        raise RuntimeError(f"Unexpected Detect strides: {head.stride.tolist()}")
    names = list(model.names.values()) if isinstance(model.names, dict) else list(model.names)
    if names != list(NAMES):
        raise RuntimeError(f"Class order mismatch: {names}")

    assigner = TaskAlignedAssigner(
        topk=10,
        num_classes=head.nc,
        alpha=0.5,
        beta=6.0,
        stride=head.stride.tolist(),
    )
    bins = torch.arange(head.reg_max, device=device, dtype=torch.float32)
    rows: list[dict[str, Any]] = []
    seen_ids: set[int] = set()
    gt_seen: set[tuple[int, int]] = set()
    decode_max_abs = 0.0

    with torch.inference_mode():
        for batch_index, batch in enumerate(evaluator.loader):
            images = batch["img"].to(device, non_blocking=True).float() / 255.0
            output = model(images)
            raw = output[1] if isinstance(output, tuple) else output
            if not isinstance(raw, dict) or not {"boxes", "scores", "feats"}.issubset(raw):
                raise RuntimeError("Official raw prediction dictionary is unavailable")
            pred_dist = raw["boxes"].permute(0, 2, 1).contiguous().float()
            pred_logits = raw["scores"].permute(0, 2, 1).contiguous().float()
            pred_scores = pred_logits.sigmoid()
            anchor_points, stride_tensor = make_anchors(raw["feats"], head.stride, 0.5)
            probabilities = pred_dist.view(images.shape[0], -1, 4, head.reg_max).softmax(-1)
            expected = (probabilities * bins).sum(-1)
            pred_boxes_grid = dist2bbox(expected, anchor_points, xywh=False)
            pred_boxes_px = pred_boxes_grid * stride_tensor

            decoded = head._inference(raw).float()[:, :4].permute(0, 2, 1)
            decode_diff = float((decoded - xyxy2xywh(pred_boxes_px)).abs().max())
            decode_max_abs = max(decode_max_abs, decode_diff)
            if decode_diff > 0.05:
                raise RuntimeError(f"DFL reconstruction disagrees with native decode: {decode_diff}")

            image_scale = torch.tensor((images.shape[3], images.shape[2], images.shape[3], images.shape[2]), device=device)
            targets = preprocess_targets(batch, images.shape[0], image_scale, device)
            gt_labels, gt_boxes = targets.split((1, 4), 2)
            mask_gt = gt_boxes.sum(2, keepdim=True).gt(0)
            _, target_boxes, target_scores, fg_mask, target_gt_idx = assigner(
                pred_scores.detach(),
                pred_boxes_px.detach().type(gt_boxes.dtype),
                anchor_points * stride_tensor,
                gt_labels,
                gt_boxes,
                mask_gt,
            )

            level_lengths = [int(feature.shape[-2] * feature.shape[-1]) for feature in raw["feats"]]
            level_index = torch.cat(
                [torch.full((length,), index, dtype=torch.long, device=device) for index, length in enumerate(level_lengths)]
            )
            entropy = -(probabilities.clamp_min(1e-12) * probabilities.clamp_min(1e-12).log()).sum(-1) / math.log(head.reg_max)
            variance = (probabilities * (bins - expected.unsqueeze(-1)).square()).sum(-1) / max((head.reg_max - 1) ** 2, 1)
            top2 = probabilities.topk(2, dim=-1).values
            peak = top2[..., 0]
            margin = top2[..., 0] - top2[..., 1]

            for image_index in range(images.shape[0]):
                stem = Path(batch["im_file"][image_index]).stem
                image_id = evaluator.ids_by_stem[stem]
                if image_id in seen_ids:
                    raise RuntimeError(f"Duplicate validation image: {stem}")
                seen_ids.add(image_id)
                valid_gt = int(mask_gt[image_index].sum())
                original_boxes = gt_boxes[image_index, :valid_gt].clone()
                if valid_gt:
                    scale_boxes(
                        tuple(images.shape[-2:]),
                        original_boxes,
                        batch["ori_shape"][image_index],
                        batch["ratio_pad"][image_index],
                    )
                positive_indices = torch.where(fg_mask[image_index])[0]
                if not len(positive_indices):
                    continue
                matched_gt_indices = target_gt_idx[image_index, positive_indices].long()
                matched_boxes = target_boxes[image_index, positive_indices]
                candidate_boxes = pred_boxes_px[image_index, positive_indices]
                candidate_iou = aligned_iou(candidate_boxes, matched_boxes)
                for local_index, anchor_index_tensor in enumerate(positive_indices):
                    anchor_index = int(anchor_index_tensor)
                    gt_index = int(matched_gt_indices[local_index])
                    if gt_index >= valid_gt:
                        raise RuntimeError(f"Assigned padded GT index {gt_index} >= {valid_gt}")
                    cls = int(gt_labels[image_index, gt_index, 0])
                    gt_seen.add((image_id, gt_index))
                    box = original_boxes[gt_index]
                    width = max(float(box[2] - box[0]), 1e-9)
                    height = max(float(box[3] - box[1]), 1e-9)
                    area = width * height
                    level = LEVELS[int(level_index[anchor_index])]
                    stride = float(stride_tensor[anchor_index])
                    anchor_px = anchor_points[anchor_index] * stride_tensor[anchor_index]
                    target_box = matched_boxes[local_index]
                    target_dist = torch.stack(
                        (
                            anchor_px[0] - target_box[0],
                            anchor_px[1] - target_box[1],
                            target_box[2] - anchor_px[0],
                            target_box[3] - anchor_px[1],
                        )
                    ) / stride
                    side_error = (expected[image_index, anchor_index] - target_dist).abs()
                    p_entropy = entropy[image_index, anchor_index]
                    p_variance = variance[image_index, anchor_index]
                    p_peak = peak[image_index, anchor_index]
                    p_margin = margin[image_index, anchor_index]
                    row: dict[str, Any] = {
                        "batch": batch_index,
                        "image_id": image_id,
                        "image": stem,
                        "gt_index": gt_index,
                        "class_id": cls,
                        "class": NAMES[cls],
                        "size": size_bucket(area),
                        "gt_area": area,
                        "log_gt_area": math.log(max(area, 1e-9)),
                        "level": level,
                        "stride": stride,
                        "anchor_index": anchor_index,
                        "anchor_x": float(anchor_px[0]),
                        "anchor_y": float(anchor_px[1]),
                        "iou": float(candidate_iou[local_index]),
                        "pred_score": float(pred_scores[image_index, anchor_index, cls]),
                        "target_score": float(target_scores[image_index, anchor_index, cls]),
                        "dfl_entropy_mean": float(p_entropy.mean()),
                        "dfl_variance_mean": float(p_variance.mean()),
                        "dfl_peak_mean": float(p_peak.mean()),
                        "dfl_margin_mean": float(p_margin.mean()),
                        "dfl_neg_entropy_mean": -float(p_entropy.mean()),
                        "dfl_neg_variance_mean": -float(p_variance.mean()),
                    }
                    for side_index, side in enumerate(SIDES):
                        normalizer = width if side in ("left", "right") else height
                        row[f"{side}_entropy"] = float(p_entropy[side_index])
                        row[f"{side}_variance"] = float(p_variance[side_index])
                        row[f"{side}_peak"] = float(p_peak[side_index])
                        row[f"{side}_margin"] = float(p_margin[side_index])
                        row[f"{side}_abs_error_px"] = float(side_error[side_index] * stride)
                        row[f"{side}_normalized_error"] = float(side_error[side_index] * stride / normalizer)
                    rows.append(row)
            if len(seen_ids) % 100 == 0:
                print(f"TAL_DFL_AUDIT images={len(seen_ids)} positives={len(rows)}", flush=True)
            if args.max_images and len(seen_ids) >= args.max_images:
                break

    expected_images = min(args.max_images, len(evaluator.gt.imgs)) if args.max_images else len(evaluator.gt.imgs)
    if len(seen_ids) != expected_images:
        raise RuntimeError(f"Validation coverage mismatch: {len(seen_ids)} != {expected_images}")
    if not rows:
        raise RuntimeError("No TAL positives were collected")

    candidate_summaries = [candidate_summary(rows, "overall", "all")]
    candidate_summaries += [candidate_summary(rows, "level", value) for value in LEVELS]
    candidate_summaries += [candidate_summary(rows, "size", value) for value in ("small", "medium", "large")]
    candidate_summaries += [candidate_summary(rows, "class", value) for value in NAMES]

    gt_rows = within_gt_rows(rows, args.pair_iou_gap)
    gt_summaries = [within_summary(gt_rows, "overall", "all")]
    gt_summaries += [within_summary(gt_rows, "size", value) for value in ("small", "medium", "large")]
    gt_summaries += [within_summary(gt_rows, "class", value) for value in NAMES]

    iou_bin_rows = []
    for low, high in IOU_BINS:
        selected = [row for row in rows if low <= row["iou"] < high]
        iou_bin_rows.append(
            {
                "iou_bin": f"[{low:.1f},{high if high <= 1 else 1.0:.1f}{')' if high <= 1 else ']'}",
                "n": len(selected),
                "mean_peak": float(np.mean([row["dfl_peak_mean"] for row in selected])) if selected else None,
                "mean_margin": float(np.mean([row["dfl_margin_mean"] for row in selected])) if selected else None,
                "mean_entropy": float(np.mean([row["dfl_entropy_mean"] for row in selected])) if selected else None,
                "mean_variance": float(np.mean([row["dfl_variance_mean"] for row in selected])) if selected else None,
                "mean_pred_score": float(np.mean([row["pred_score"] for row in selected])) if selected else None,
                "mean_target_score": float(np.mean([row["target_score"] for row in selected])) if selected else None,
            }
        )

    side_rows = []
    for side in SIDES:
        errors = [row[f"{side}_normalized_error"] for row in rows]
        side_rows.append(
            {
                "side": side,
                "mean_normalized_error": float(np.mean(errors)),
                "median_normalized_error": float(np.median(errors)),
                "entropy_error_spearman": correlation([row[f"{side}_entropy"] for row in rows], errors, spearman=True),
                "variance_error_spearman": correlation([row[f"{side}_variance"] for row in rows], errors, spearman=True),
                "peak_neg_error_spearman": correlation([row[f"{side}_peak"] for row in rows], [-x for x in errors], spearman=True),
                "margin_neg_error_spearman": correlation([row[f"{side}_margin"] for row in rows], [-x for x in errors], spearman=True),
            }
        )

    overall = candidate_summaries[0]
    within = gt_summaries[0]
    feature_evidence = []
    level_summaries = {row["value"]: row for row in candidate_summaries if row["dimension"] == "level"}
    for key in CERTAINTY_KEYS:
        level_positive = sum(
            1
            for level in LEVELS
            if (level_summaries[level].get(f"{key}_iou_spearman") or 0.0) >= 0.10
        )
        feature_evidence.append(
            {
                "feature": key,
                "pooled_spearman": overall.get(f"{key}_iou_spearman"),
                "partial_pearson": overall.get(f"{key}_partial_pearson"),
                "within_gt_pairwise_accuracy": within.get(f"{key}_weighted_pairwise_accuracy"),
                "levels_spearman_ge_010": level_positive,
            }
        )
    best_feature = max(
        feature_evidence,
        key=lambda row: (
            row["partial_pearson"] if row["partial_pearson"] is not None else -1.0,
            row["pooled_spearman"] if row["pooled_spearman"] is not None else -1.0,
        ),
    )
    dfl_checks = {
        "pooled_spearman_ge_015": (best_feature["pooled_spearman"] or 0.0) >= 0.15,
        "partial_pearson_ge_005": (best_feature["partial_pearson"] or 0.0) >= 0.05,
        "within_gt_pairwise_ge_053": (best_feature["within_gt_pairwise_accuracy"] or 0.0) >= 0.53,
        "two_levels_ge_010": best_feature["levels_spearman_ge_010"] >= 2,
    }
    passed = sum(dfl_checks.values())
    dfl_decision = "GO" if passed == 4 else "REVIEW" if passed >= 2 else "NO-GO"
    native_pairwise = within.get("pred_score_weighted_pairwise_accuracy") or 0.0
    target_pairwise = within.get("target_score_weighted_pairwise_accuracy") or 0.0
    rank_checks = {
        "native_pairwise_below_070": native_pairwise < 0.70,
        "target_exceeds_native_by_005": target_pairwise - native_pairwise >= 0.05,
        "mean_quality_gap_ge_003": (within.get("mean_best_minus_top_score_iou") or 0.0) >= 0.03,
    }
    rank_decision = "GO" if all(rank_checks.values()) else "REVIEW" if sum(rank_checks.values()) >= 2 else "NO-GO"

    write_csv(args.output / "tal_positive_candidates.csv", rows)
    write_csv(args.output / "candidate_group_summary.csv", candidate_summaries)
    write_csv(args.output / "within_gt_ranking.csv", gt_rows)
    write_csv(args.output / "within_gt_summary.csv", gt_summaries)
    write_csv(args.output / "dfl_iou_bins.csv", iou_bin_rows)
    write_csv(args.output / "dfl_side_uncertainty.csv", side_rows)
    write_csv(args.output / "dfl_feature_evidence.csv", feature_evidence)

    manifest = {
        "protocol": {
            "split": "val",
            "test_accessed": False,
            "training_performed": False,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "workers": args.workers,
            "native_tal": {"topk": 10, "alpha": 0.5, "beta": 6.0},
            "pair_iou_gap": args.pair_iou_gap,
        },
        "weights": str(args.weights),
        "weights_sha256": actual_sha,
        "data": str(args.data),
        "data_sha256": sha256(args.data),
        "git_commit": git_value("rev-parse", "HEAD"),
        "git_status_porcelain": git_value("status", "--porcelain"),
        "model": {"head": type(head).__name__, "reg_max": head.reg_max, "strides": head.stride.tolist(), "classes": names},
        "coverage": {"images": len(seen_ids), "gt_with_positive": len(gt_seen), "tal_positives": len(rows)},
        "decode_reconstruction_max_abs": decode_max_abs,
    }
    decisions = {
        "dfl_quality": {"decision": dfl_decision, "best_feature": best_feature, "checks": dfl_checks},
        "same_gt_ranking": {"decision": rank_decision, "checks": rank_checks, "native_pairwise": native_pairwise, "target_pairwise": target_pairwise},
    }
    write_json(args.output / "manifest.json", manifest)
    write_json(args.output / "decision.json", decisions)

    level_table = markdown_table(
        [row for row in candidate_summaries if row["dimension"] in ("overall", "level")],
        [
            ("value", "Scope"),
            ("positive_candidates", "Pos"),
            ("unique_gt", "GT"),
            ("mean_iou", "Mean IoU"),
            ("pred_score_iou_spearman", "Score rho"),
            ("target_score_iou_spearman", "TAL target rho"),
            ("dfl_peak_mean_iou_spearman", "Peak rho"),
            ("dfl_margin_mean_iou_spearman", "Margin rho"),
            ("dfl_neg_entropy_mean_iou_spearman", "-Entropy rho"),
            ("dfl_neg_variance_mean_iou_spearman", "-Variance rho"),
        ],
    )
    size_table = markdown_table(
        [row for row in candidate_summaries if row["dimension"] == "size"],
        [("value", "Size"), ("positive_candidates", "Pos"), ("unique_gt", "GT"), ("mean_iou", "Mean IoU"),
         ("iou75_rate", "IoU>=.75"), ("pred_score_iou_spearman", "Score rho"),
         ("dfl_peak_mean_iou_spearman", "Peak rho"), ("dfl_neg_entropy_mean_iou_spearman", "-Entropy rho")],
    )
    within_table = markdown_table(
        [within],
        [("gt", "GT"), ("mean_positive_candidates", "Pos/GT"), ("mean_best_iou", "Best IoU"),
         ("mean_top_score_iou", "Top-score IoU"), ("mean_best_minus_top_score_iou", "Gap"),
         ("pred_score_weighted_pairwise_accuracy", "Score pair acc"),
         ("target_score_weighted_pairwise_accuracy", "TAL target pair acc"),
         ("dfl_peak_mean_weighted_pairwise_accuracy", "Peak pair acc"),
         ("dfl_neg_entropy_mean_weighted_pairwise_accuracy", "-Entropy pair acc")],
    )
    feature_table = markdown_table(
        feature_evidence,
        [("feature", "DFL certainty"), ("pooled_spearman", "Pooled rho"), ("partial_pearson", "Partial r"),
         ("within_gt_pairwise_accuracy", "Within-GT pair acc"), ("levels_spearman_ge_010", "Levels rho>=.10")],
    )
    bin_table = markdown_table(
        iou_bin_rows,
        [("iou_bin", "IoU bin"), ("n", "N"), ("mean_peak", "Peak"), ("mean_margin", "Margin"),
         ("mean_entropy", "Entropy"), ("mean_variance", "Variance"), ("mean_pred_score", "Pred score"),
         ("mean_target_score", "TAL target")],
    )

    report = f"""# YOLO12n native TAL-positive and DFL uncertainty diagnosis

## Protocol

- Frozen checkpoint: `{args.weights}`
- SHA256: `{actual_sha}`
- SVRDD7 Val only: {len(seen_ids)} images
- Model path: official non-end2end `Detect`, native TAL topk=10, alpha=0.5, beta=6.0, DFL reg_max={head.reg_max}
- Image size: {args.imgsz}; batch: {args.batch}; no Voting
- Test accessed: **false**; detector training performed: **false**
- Native decode reconstruction max absolute difference: `{decode_max_abs:.8f}` pixels

## Coverage

- GT with at least one native TAL positive: **{len(gt_seen)}**
- Native TAL positive candidates: **{len(rows)}**
- Mean positives per represented GT: **{len(rows) / max(len(gt_seen), 1):.3f}**

## Candidate-level quality

{level_table}

{size_table}

`rho` is Spearman correlation with the candidate's plain IoU to its assigned GT. Entropy and variance are negated in the certainty columns, so a positive value has the expected direction.

## Within-GT ordering

{within_table}

Pairwise accuracy uses only positive pairs whose IoU differs by at least {args.pair_iou_gap:.2f}. A value of 0.5 is chance ordering.

## Incremental DFL evidence

{feature_table}

`Partial r` controls native class score, P3/P4/P5 level, and log GT area. This is the key check against a misleading pooled scale correlation.

## Monotonicity by IoU bin

{bin_table}

## Decisions

### DFL-guided quality estimation: **{dfl_decision}**

- Best diagnostic feature: `{best_feature['feature']}`
- Checks: `{json.dumps(dfl_checks, ensure_ascii=False)}`
- Interpretation: a GO requires pooled, conditional, within-GT, and cross-level evidence together. REVIEW means some useful signal exists but a DFL-only quality branch is not yet justified.

### Same-GT residual ranking supervision: **{rank_decision}**

- Native score pairwise accuracy: `{native_pairwise:.4f}`
- Native TAL target pairwise accuracy: `{target_pairwise:.4f}`
- Checks: `{json.dumps(rank_checks, ensure_ascii=False)}`
- Interpretation: the TAL target already encodes quality. A GO means the learned score fails to realize a materially stronger target ordering, leaving a specific residual-ranking hypothesis.

## Limits

- This audit evaluates actual positives selected by the frozen model's native TAL. It does not claim that a new quality module will recover the GT-IoU oracle.
- Correlation is mechanism evidence, not a performance result.
- DFL statistics may identify localization certainty but cannot by themselves prove foreground semantics or suppress every background false positive.
- One checkpoint and one Val split authorize or reject development; they do not establish a final paper claim.
"""
    (args.output / "DIAGNOSIS.md").write_text(report, encoding="utf-8")
    print(json.dumps({"output": str(args.output), "coverage": manifest["coverage"], "decisions": decisions}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
