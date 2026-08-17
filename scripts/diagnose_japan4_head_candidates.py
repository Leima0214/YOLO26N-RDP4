"""Compare YOLO26 O2O/O2M candidate generation and score ranking without training."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.data.build import build_dataloader, build_yolo_dataset
from ultralytics.data.utils import check_det_dataset
from ultralytics.nn.modules.head import Detect
from ultralytics.utils import DEFAULT_CFG
from ultralytics.utils.metrics import box_iou
from ultralytics.utils.nms import non_max_suppression
from ultralytics.utils.ops import xywh2xyxy, xyxy2xywh
from ultralytics.utils.torch_utils import select_device


CLASS_NAMES = ("D00", "D10", "D20", "D40")
BRANCHES = ("o2o", "o2m")
METRIC_COLUMNS = (
    "global_max_iou",
    "oracle_assigned_iou",
    "correct_top1_iou",
    "correct_top10_max_iou",
    "correct_top100_max_iou",
    "correct_top100_gap",
    "correct_spearman",
    "correct_best_rank",
    "correct_duplicate_iou50",
    "correct_top1_is_top100_best",
    "any_top1_iou",
    "any_top10_max_iou",
    "any_top100_max_iou",
    "any_top100_gap",
    "any_spearman",
    "any_best_rank",
    "any_duplicate_iou50",
    "any_top1_is_top100_best",
    "any_top1_class_correct",
    "final_correct_max_iou",
    "final_any_max_iou",
    "final_correct_score_at_max_iou",
    "final_correct_duplicate_iou50",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        metavar="NAME=CHECKPOINT",
        help="Repeat for each checkpoint, e.g. --model B0=/path/best.pt",
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--score-floor", type=float, default=0.001)
    parser.add_argument("--max-images", type=int, default=0, help="0 uses the complete validation set")
    parser.add_argument("--skip-val-sweep", action="store_true")
    return parser.parse_args()


def parse_models(values: list[str]) -> dict[str, Path]:
    models: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected NAME=CHECKPOINT, got {value!r}")
        name, raw_path = value.split("=", 1)
        if not name or name in models:
            raise ValueError(f"Invalid or duplicate model name: {name!r}")
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        models[name] = path
    return models


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def make_loader(data_yaml: Path, args: argparse.Namespace):
    data = check_det_dataset(str(data_yaml))
    cfg = get_cfg(
        DEFAULT_CFG,
        {
            "mode": "val",
            "task": "detect",
            "imgsz": args.imgsz,
            "batch": args.batch,
            "workers": args.workers,
            "rect": True,
            "cache": False,
        },
    )
    dataset = build_yolo_dataset(cfg, data["val"], args.batch, data, mode="val", rect=True, stride=32)
    loader = build_dataloader(dataset, args.batch, args.workers, shuffle=False, rank=-1, pin_memory=False)
    return data, loader


def spearman(scores: torch.Tensor, ious: torch.Tensor) -> float | None:
    score_values = scores.detach().float().cpu().numpy()
    iou_values = ious.detach().float().cpu().numpy()
    if len(score_values) < 3 or np.ptp(score_values) == 0 or np.ptp(iou_values) == 0:
        return None
    score_ranks = rankdata(score_values, method="average")
    iou_ranks = rankdata(iou_values, method="average")
    value = float(np.corrcoef(score_ranks, iou_ranks)[0, 1])
    return value if np.isfinite(value) else None


def bucket_size(area: float) -> str:
    if area < 32**2:
        return "small"
    if area < 96**2:
        return "medium"
    return "large"


def bucket_aspect(width: float, height: float) -> tuple[float, str]:
    ratio = max(width / max(height, 1e-9), height / max(width, 1e-9))
    if ratio < 2:
        return ratio, "compact_lt2"
    if ratio < 5:
        return ratio, "elongated_2to5"
    return ratio, "very_elongated_ge5"


def candidate_view(
    scores: torch.Tensor,
    ious: torch.Tensor,
    indices: torch.Tensor,
    topk: int,
    score_floor: float,
) -> dict[str, Any]:
    if not scores.numel():
        return {
            "top1_iou": 0.0,
            "top10_max_iou": 0.0,
            "top100_max_iou": 0.0,
            "top100_gap": 0.0,
            "spearman": None,
            "best_rank": None,
            "duplicate_iou50": 0,
            "top1_index": None,
            "top100_best_index": None,
            "top1_is_top100_best": 0,
        }
    k100 = min(topk, scores.numel())
    k10 = min(10, k100)
    order = scores.topk(k100).indices
    top_scores, top_ious = scores[order], ious[order]
    global_best = int(ious.argmax())
    best_score = scores[global_best]
    best_rank = int((scores > best_score).sum().item()) + 1
    top_best_position = int(top_ious.argmax())
    top1_iou = float(top_ious[0])
    top100_iou = float(top_ious[top_best_position])
    return {
        "top1_iou": top1_iou,
        "top10_max_iou": float(top_ious[:k10].max()),
        "top100_max_iou": top100_iou,
        "top100_gap": top100_iou - top1_iou,
        "spearman": spearman(top_scores, top_ious),
        "best_rank": best_rank,
        "duplicate_iou50": int(((scores >= score_floor) & (ious >= 0.5)).sum().item()),
        "top1_index": int(indices[order[0]]),
        "top100_best_index": int(indices[order[top_best_position]]),
        "top1_is_top100_best": int(top_best_position == 0),
    }


def level_for(index: int | None, ends: np.ndarray) -> str:
    if index is None:
        return "none"
    return f"P{int(np.searchsorted(ends, index, side='right')) + 3}"


def owned_candidate_indices(
    iou_matrix: torch.Tensor,
    peer_gt_indices: torch.Tensor,
    gt_index: int,
    candidate_indices: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return overlapping candidates whose nearest peer GT is this GT."""
    if candidate_indices is None:
        candidate_indices = torch.arange(iou_matrix.shape[1], device=iou_matrix.device)
    if not len(peer_gt_indices) or not len(candidate_indices):
        return candidate_indices[:0]
    peer_ious = iou_matrix[peer_gt_indices][:, candidate_indices]
    best_ious, owners = peer_ious.max(0)
    return candidate_indices[(peer_gt_indices[owners] == gt_index) & (best_ious > 0)]


def oracle_assignments(iou_matrix: torch.Tensor, classes: torch.Tensor, scores: torch.Tensor, topk: int) -> torch.Tensor:
    """Assign unique top-scored same-class candidates to GTs for an oracle reranking upper bound."""
    assigned = torch.zeros(len(classes), device=iou_matrix.device)
    for class_index in classes.unique().tolist():
        gt_indices = torch.where(classes == class_index)[0]
        candidate_indices = scores[:, int(class_index)].topk(min(topk, scores.shape[0])).indices
        matrix = iou_matrix[gt_indices][:, candidate_indices].detach().float().cpu().numpy()
        gt_rows, candidate_columns = linear_sum_assignment(-matrix)
        assigned[gt_indices[torch.as_tensor(gt_rows, device=gt_indices.device)]] = torch.as_tensor(
            matrix[gt_rows, candidate_columns], device=assigned.device
        )
    return assigned


def final_stats(
    final_predictions: torch.Tensor,
    gt_boxes: torch.Tensor,
    classes: torch.Tensor,
) -> list[dict[str, Any]]:
    if final_predictions.numel():
        final_ious = box_iou(gt_boxes, final_predictions[:, :4])
        final_classes = final_predictions[:, 5].long()
    else:
        final_ious = torch.zeros((len(gt_boxes), 0), device=gt_boxes.device)
        final_classes = torch.zeros(0, dtype=torch.long, device=gt_boxes.device)
    output = []
    for gt_index, gt_class in enumerate(classes):
        all_gt_indices = torch.arange(len(classes), device=classes.device)
        any_indices = owned_candidate_indices(final_ious, all_gt_indices, gt_index)
        any_iou = float(final_ious[gt_index, any_indices].max()) if len(any_indices) else 0.0
        peer_gt_indices = torch.where(classes == gt_class)[0]
        correct_candidates = torch.where(final_classes == gt_class)[0]
        correct_indices = owned_candidate_indices(final_ious, peer_gt_indices, gt_index, correct_candidates)
        if len(correct_indices):
            correct_ious = final_ious[gt_index, correct_indices]
            correct_predictions = final_predictions[correct_indices]
            best = int(correct_ious.argmax())
            correct_iou = float(correct_ious[best])
            correct_score = float(correct_predictions[best, 4])
            duplicates = int((correct_ious >= 0.5).sum())
        else:
            correct_iou = correct_score = 0.0
            duplicates = 0
        output.append(
            {
                "final_correct_max_iou": correct_iou,
                "final_any_max_iou": any_iou,
                "final_correct_score_at_max_iou": correct_score,
                "final_correct_duplicate_iou50": duplicates,
            }
        )
    return output


def inspect_checkpoint(
    label: str,
    checkpoint: Path,
    loader,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    wrapped = YOLO(str(checkpoint))
    device = select_device(args.device, verbose=False)
    net = wrapped.model.to(device).float().eval()
    head = net.model[-1]
    if not isinstance(head, Detect) or not head.end2end or head.nc != len(CLASS_NAMES):
        raise RuntimeError(f"{label}: expected four-class end-to-end Detect head")

    rows: list[dict[str, Any]] = []
    seen_images = 0
    amp = device.type == "cuda"
    with torch.inference_mode():
        for batch in loader:
            images = batch["img"].to(device, non_blocking=True).float() / 255
            with torch.autocast(device_type=device.type, enabled=amp):
                output = net(images)
            raw = output[1] if isinstance(output, tuple) else output
            if not isinstance(raw, dict) or not {"one2one", "one2many"}.issubset(raw):
                raise RuntimeError(f"{label}: checkpoint did not expose both head branches")

            height, width = images.shape[-2:]
            batch_indices = batch["batch_idx"].view(-1).long()
            gt_classes = batch["cls"].view(-1).long()
            gt_xywh = batch["bboxes"].float()
            for branch, raw_branch in (("o2o", raw["one2one"]), ("o2m", raw["one2many"])):
                decoded = head._inference(raw_branch).float()
                if branch == "o2o":
                    final_outputs = [
                        prediction[prediction[:, 4] >= args.score_floor]
                        for prediction in head.postprocess(decoded.transpose(1, 2))
                    ]
                else:
                    nms_input = torch.cat(
                        (xyxy2xywh(decoded[:, :4].transpose(1, 2)).transpose(1, 2), decoded[:, 4:]), dim=1
                    )
                    final_outputs = non_max_suppression(
                        nms_input,
                        conf_thres=args.score_floor,
                        iou_thres=0.7,
                        nc=head.nc,
                        max_det=300,
                    )
                lengths = [feature.shape[2] * feature.shape[3] for feature in raw_branch["feats"]]
                ends = np.cumsum(lengths)
                for image_index in range(images.shape[0]):
                    absolute_index = seen_images + image_index
                    if args.max_images and absolute_index >= args.max_images:
                        continue
                    mask = batch_indices == image_index
                    if not mask.any():
                        continue
                    boxes = decoded[image_index, :4].transpose(0, 1)
                    scores = decoded[image_index, 4:].transpose(0, 1)
                    gt_boxes = xywh2xyxy(gt_xywh[mask]).to(device)
                    gt_boxes *= torch.tensor((width, height, width, height), device=device)
                    classes = gt_classes[mask].to(device)
                    iou_matrix = box_iou(gt_boxes, boxes)
                    oracle_ious = oracle_assignments(iou_matrix, classes, scores, args.topk)
                    final_rows = final_stats(final_outputs[image_index], gt_boxes, classes)
                    image_name = Path(batch["im_file"][image_index]).name

                    for gt_index, (gt_box, gt_class) in enumerate(zip(gt_boxes, classes)):
                        ious = iou_matrix[gt_index]
                        class_index = int(gt_class)
                        peer_gt_indices = torch.where(classes == gt_class)[0]
                        correct_indices = owned_candidate_indices(iou_matrix, peer_gt_indices, gt_index)
                        correct_scores = scores[correct_indices, class_index]
                        any_scores, any_classes = scores.max(1)
                        all_gt_indices = torch.arange(len(classes), device=device)
                        any_indices = owned_candidate_indices(iou_matrix, all_gt_indices, gt_index)
                        correct = candidate_view(
                            correct_scores, ious[correct_indices], correct_indices, args.topk, args.score_floor
                        )
                        any_view = candidate_view(
                            any_scores[any_indices], ious[any_indices], any_indices, args.topk, args.score_floor
                        )
                        box_width = float(gt_box[2] - gt_box[0])
                        box_height = float(gt_box[3] - gt_box[1])
                        area = box_width * box_height
                        aspect, aspect_bucket = bucket_aspect(box_width, box_height)
                        rows.append(
                            {
                                "model": label,
                                "branch": branch,
                                "image": image_name,
                                "gt_index": gt_index,
                                "class": CLASS_NAMES[class_index],
                                "input_area": area,
                                "size": bucket_size(area),
                                "aspect_ratio": aspect,
                                "aspect": aspect_bucket,
                                "global_max_iou": float(ious.max()),
                                "oracle_assigned_iou": float(oracle_ious[gt_index]),
                                **{f"correct_{key}": value for key, value in correct.items() if not key.endswith("index")},
                                **{f"any_{key}": value for key, value in any_view.items() if not key.endswith("index")},
                                **final_rows[gt_index],
                                "any_top1_class": (
                                    CLASS_NAMES[int(any_classes[any_view["top1_index"]])]
                                    if any_view["top1_index"] is not None
                                    else "none"
                                ),
                                "any_top1_class_correct": int(
                                    any_view["top1_index"] is not None
                                    and any_classes[any_view["top1_index"]] == gt_class
                                ),
                                "correct_top1_level": level_for(correct["top1_index"], ends),
                                "correct_top100_best_level": level_for(correct["top100_best_index"], ends),
                                "any_top1_level": level_for(any_view["top1_index"], ends),
                                "any_top100_best_level": level_for(any_view["top100_best_index"], ends),
                            }
                        )
            seen_images += images.shape[0]
            if args.max_images and seen_images >= args.max_images:
                break
    del net, wrapped
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows


def mean(values: list[Any]) -> float | None:
    valid = [float(value) for value in values if value is not None and np.isfinite(float(value))]
    return float(np.mean(valid)) if valid else None


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        dimensions = (
            ("all", "all"),
            ("class", row["class"]),
            ("size", row["size"]),
            ("aspect", row["aspect"]),
        )
        for dimension, value in dimensions:
            groups[(row["model"], row["branch"], dimension, value)].append(row)

    output = []
    for (model, branch, dimension, value), items in sorted(groups.items()):
        summary: dict[str, Any] = {
            "model": model,
            "branch": branch,
            "dimension": dimension,
            "value": value,
            "n_gt": len(items),
        }
        for column in METRIC_COLUMNS:
            summary[f"mean_{column}"] = mean([item[column] for item in items])
        thresholds = np.arange(0.5, 1.0, 0.05)
        per_class_upper = []
        for class_name in sorted({item["class"] for item in items}):
            class_items = [item for item in items if item["class"] == class_name]
            per_class_upper.append(
                np.mean(
                    [mean([float(item["oracle_assigned_iou"] >= threshold) for item in class_items]) for threshold in thresholds]
                )
            )
        summary["oracle_rerank_ap50_95_upper"] = float(np.mean(per_class_upper))
        for prefix in (
            "global",
            "correct_top10",
            "correct_top100",
            "any_top10",
            "any_top100",
            "final_correct",
            "final_any",
        ):
            key = "global_max_iou" if prefix == "global" else f"{prefix}_max_iou"
            for threshold in (0.5, 0.75):
                summary[f"recall_{prefix}_iou{str(threshold).replace('.', '')}"] = mean(
                    [float(item[key] >= threshold) for item in items]
                )
        for level_key in ("correct_top1_level", "correct_top100_best_level", "any_top1_level", "any_top100_best_level"):
            counts = Counter(item[level_key] for item in items)
            for level in ("P3", "P4", "P5"):
                summary[f"{level_key}_{level}_fraction"] = counts[level] / len(items)
        output.append(summary)
    return output


def branch_gaps(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    paired: dict[tuple[str, str, int, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        key = (row["model"], row["image"], int(row["gt_index"]), row["class"])
        paired[key][row["branch"]] = row

    gaps = []
    for (model, image, gt_index, class_name), branches in paired.items():
        if set(branches) != set(BRANCHES):
            raise RuntimeError(f"Missing paired branch for {(model, image, gt_index, class_name)}")
        o2o, o2m = branches["o2o"], branches["o2m"]
        gaps.append(
            {
                "model": model,
                "image": image,
                "gt_index": gt_index,
                "class": class_name,
                "size": o2o["size"],
                "aspect": o2o["aspect"],
                "o2m_candidate_top100_iou": o2m["correct_top100_max_iou"],
                "o2m_final_iou": o2m["final_correct_max_iou"],
                "o2o_candidate_top100_iou": o2o["correct_top100_max_iou"],
                "o2o_final_iou": o2o["final_correct_max_iou"],
                "o2m_candidate_to_o2o_final_loss": o2m["correct_top100_max_iou"] - o2o["final_correct_max_iou"],
                "o2m_final_minus_o2o_final": o2m["final_correct_max_iou"] - o2o["final_correct_max_iou"],
                "o2m_candidate_minus_o2o_candidate": o2m["correct_top100_max_iou"] - o2o["correct_top100_max_iou"],
                "o2m_oracle_minus_o2o_oracle": o2m["oracle_assigned_iou"] - o2o["oracle_assigned_iou"],
            }
        )

    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in gaps:
        for dimension, value in (("all", "all"), ("class", row["class"]), ("size", row["size"]), ("aspect", row["aspect"])):
            grouped[(row["model"], dimension, value)].append(row)
    summary = []
    numeric = tuple(gaps[0].keys())[6:]
    for (model, dimension, value), items in sorted(grouped.items()):
        summary.append(
            {
                "model": model,
                "dimension": dimension,
                "value": value,
                "n_gt": len(items),
                **{f"mean_{column}": mean([item[column] for item in items]) for column in numeric},
            }
        )
    return gaps, summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"No rows generated for {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def validation_sweep(models: dict[str, Path], data_yaml: Path, output: Path, args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = []
    for label, checkpoint in models.items():
        settings = [("o2o", 0.7), ("o2m", 0.5), ("o2m", 0.6), ("o2m", 0.7)]
        for branch, nms_iou in settings:
            model = YOLO(str(checkpoint))
            head = model.model.model[-1]
            if branch == "o2m":
                del head.one2one_cv2
                del head.one2one_cv3
            metrics = model.val(
                data=str(data_yaml),
                imgsz=args.imgsz,
                batch=args.batch,
                device=args.device,
                workers=args.workers,
                conf=args.score_floor,
                iou=nms_iou,
                max_det=300,
                rect=True,
                plots=False,
                verbose=False,
                project=str(output / "validation"),
                name=f"{label}_{branch}_iou{nms_iou:g}",
                exist_ok=True,
            )
            rows.append(
                {
                    "model": label,
                    "branch": branch,
                    "nms_iou": nms_iou,
                    "precision": float(metrics.box.mp),
                    "recall": float(metrics.box.mr),
                    "map50": float(metrics.box.map50),
                    "map50_95": float(metrics.box.map),
                    **{f"{name}_map50_95": float(value) for name, value in zip(CLASS_NAMES, metrics.box.maps)},
                }
            )
    return rows


def main() -> None:
    args = parse_args()
    models = parse_models(args.model)
    args.data = args.data.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    if not args.data.is_file():
        raise FileNotFoundError(args.data)
    if args.topk < 10 or not 0 <= args.score_floor <= 1:
        raise ValueError("--topk must be >=10 and --score-floor must be in [0,1]")
    args.output.mkdir(parents=True, exist_ok=True)
    data, loader = make_loader(args.data, args)

    rows = []
    for label, checkpoint in models.items():
        print(f"DIAGNOSING {label}: {checkpoint}", flush=True)
        rows.extend(inspect_checkpoint(label, checkpoint, loader, args))
    summaries = aggregate(rows)
    gaps, gap_summaries = branch_gaps(rows)
    write_csv(args.output / "per_gt_candidates.csv", rows)
    write_csv(args.output / "candidate_summary.csv", summaries)
    write_csv(args.output / "o2m_to_o2o_per_gt.csv", gaps)
    write_csv(args.output / "o2m_to_o2o_summary.csv", gap_summaries)

    sweep = [] if args.skip_val_sweep or args.max_images else validation_sweep(models, args.data, args.output, args)
    if sweep:
        write_csv(args.output / "validation_sweep.csv", sweep)

    report = {
        "protocol": {
            "data": str(args.data),
            "resolved_val": data["val"],
            "imgsz": args.imgsz,
            "batch": args.batch,
            "topk": args.topk,
            "score_floor": args.score_floor,
            "size_definition": "model-input pixels: small < 32^2, medium < 96^2, large otherwise",
            "branches": list(BRANCHES),
            "note": (
                "Candidates are uniquely owned by the overlapping nearest peer GT before local ranking metrics. "
                "Oracle AP uses GT assignment and perfect reranking as a macro class upper bound, never deployable AP."
            ),
        },
        "weights": {label: {"path": str(path), "sha256": sha256(path)} for label, path in models.items()},
        "counts": {
            "per_gt_rows": len(rows),
            "summary_rows": len(summaries),
            "paired_branch_rows": len(gaps),
            "paired_branch_summary_rows": len(gap_summaries),
        },
        "artifacts": {
            "per_gt": "per_gt_candidates.csv",
            "candidate_summary": "candidate_summary.csv",
            "o2m_to_o2o_per_gt": "o2m_to_o2o_per_gt.csv",
            "o2m_to_o2o_summary": "o2m_to_o2o_summary.csv",
            "validation_sweep": "validation_sweep.csv" if sweep else None,
        },
    }
    (args.output / "summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
