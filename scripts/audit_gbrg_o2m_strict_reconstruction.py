"""Canonical O2M representative-reconstruction audit for frozen R1 and GBRG.

This stage-1 audit is deliberately training-free.  It reproduces the locked
paper evaluator (batch=32, rectangular FP32 inference, conf=0.001, NMS=0.70),
recovers exact NMS suppression groups, and changes coordinates only.  Scores,
classes, output counts, and global order are invariant for every variant.

The audit separates three different ceilings:
1. same-class opportunistic candidate oracle (the earlier broad ceiling),
2. assignment-preserving candidate oracle, and
3. assignment-preserving original-versus-vote oracle.

Only the third ceiling can authorize a later conservative gate experiment.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from pycocotools.coco import COCO

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from audit_gbrg_o2m_csrg_candidates import make_loader, metric_row, paired_gt  # noqa: E402
from audit_gbrg_o2m_region_signal import groups_anchored_to_reference  # noqa: E402
from ultralytics import YOLO  # noqa: E402
from ultralytics.data.utils import check_det_dataset  # noqa: E402
from ultralytics.nn.modules.head import Detect  # noqa: E402
from ultralytics.utils.metrics import box_iou  # noqa: E402
from ultralytics.utils.nms import non_max_suppression  # noqa: E402
from ultralytics.utils.ops import scale_boxes, xywh2xyxy, xyxy2xywh  # noqa: E402
from ultralytics.utils.torch_utils import select_device  # noqa: E402


EXPECTED_KEYS = {
    "AP": "AP50_95",
    "AP50": "AP50",
    "AP75": "AP75",
    "AP_small": "AP_small",
    "AP_medium": "AP_medium",
    "AP_large": "AP_large",
    "AR100": "AR100",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r1", type=Path, required=True)
    parser.add_argument("--gbrg", type=Path, required=True)
    parser.add_argument("--canonical-summary", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--score-floor", type=float, default=0.001)
    parser.add_argument("--nms-iou", type=float, default=0.70)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--max-nms", type=int, default=30000)
    parser.add_argument("--assignment-min-iou", type=float, default=0.10)
    parser.add_argument("--ambiguous-gt-iou", type=float, default=0.50)
    parser.add_argument("--protocol-tolerance", type=float, default=1e-6)
    parser.add_argument("--oracle-no-go", type=float, default=0.003)
    parser.add_argument("--oracle-go", type=float, default=0.005)
    parser.add_argument("--max-images", type=int, default=0)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def current_nms(decoded: torch.Tensor, nc: int, args: argparse.Namespace) -> torch.Tensor:
    nms_input = torch.cat(
        (xyxy2xywh(decoded[:4].T).T.unsqueeze(0), decoded[4:].unsqueeze(0)),
        dim=1,
    )
    return non_max_suppression(
        nms_input,
        conf_thres=args.score_floor,
        iou_thres=args.nms_iou,
        nc=nc,
        multi_label=True,
        max_det=args.max_det,
        max_nms=args.max_nms,
    )[0]


def raw_level_indices(feats: list[torch.Tensor], device: torch.device) -> torch.Tensor:
    values = []
    for level, feat in enumerate(feats):
        values.append(torch.full((feat.shape[-2] * feat.shape[-1],), level, dtype=torch.long, device=device))
    return torch.cat(values)


def expanded_candidates(
    decoded: torch.Tensor,
    level_by_raw: torch.Tensor,
    nc: int,
    args: argparse.Namespace,
) -> torch.Tensor:
    """Return pre-NMS multi-label candidates plus raw anchor and pyramid-level provenance."""
    boxes = xywh2xyxy(xyxy2xywh(decoded[:4].T))
    class_scores = decoded[4 : 4 + nc].T
    candidate = class_scores.amax(1) > args.score_floor
    raw_candidates = torch.where(candidate)[0]
    boxes = boxes[candidate]
    class_scores = class_scores[candidate]
    row_index, class_index = torch.where(class_scores > args.score_floor)
    raw_index = raw_candidates[row_index]
    expanded = torch.cat(
        (
            boxes[row_index],
            class_scores[row_index, class_index, None],
            class_index[:, None].float(),
            raw_index[:, None].float(),
            level_by_raw[raw_index, None].float(),
        ),
        dim=1,
    )
    if len(expanded) > args.max_nms:
        order = expanded[:, 4].argsort(descending=True)[: args.max_nms]
        expanded = expanded[order]
    return expanded


def one_to_one_gt_assignment(
    current: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_classes: torch.Tensor,
    min_iou: float,
) -> tuple[list[int], list[float]]:
    """Greedily assign output boxes to unique same-class GTs by descending IoU."""
    assignment = [-1] * len(current)
    assigned_iou = [0.0] * len(current)
    for class_index in sorted(set(int(x) for x in current[:, 5].tolist())):
        output_indices = torch.where(current[:, 5].long() == class_index)[0]
        gt_indices = torch.where(gt_classes.long() == class_index)[0]
        if not len(output_indices) or not len(gt_indices):
            continue
        ious = box_iou(current[output_indices, :4].float(), gt_boxes[gt_indices].float())
        flat = ious.flatten().argsort(descending=True)
        used_outputs: set[int] = set()
        used_gts: set[int] = set()
        for flat_index in flat.tolist():
            local_output = flat_index // len(gt_indices)
            local_gt = flat_index % len(gt_indices)
            value = float(ious[local_output, local_gt])
            if value < min_iou:
                break
            output_index = int(output_indices[local_output])
            gt_index = int(gt_indices[local_gt])
            if output_index in used_outputs or gt_index in used_gts:
                continue
            assignment[output_index] = gt_index
            assigned_iou[output_index] = value
            used_outputs.add(output_index)
            used_gts.add(gt_index)
    return assignment, assigned_iou


def weighted_medoid(boxes: torch.Tensor, weights: torch.Tensor) -> int:
    if len(boxes) <= 1:
        return 0
    return int((box_iou(boxes, boxes) @ weights).argmax())


def normalized_cluster_features(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    levels: torch.Tensor,
    root_level: int,
) -> dict[str, Any]:
    weights = scores.clamp_min(0)
    weights = weights / weights.sum().clamp_min(1e-12)
    root = boxes[0]
    root_w = float((root[2] - root[0]).clamp_min(1.0))
    root_h = float((root[3] - root[1]).clamp_min(1.0))
    cx = (boxes[:, 0] + boxes[:, 2]) / 2
    cy = (boxes[:, 1] + boxes[:, 3]) / 2
    width = (boxes[:, 2] - boxes[:, 0]).clamp_min(1.0)
    height = (boxes[:, 3] - boxes[:, 1]).clamp_min(1.0)
    pairwise = box_iou(boxes, boxes)
    off_diag = pairwise[~torch.eye(len(boxes), dtype=torch.bool, device=boxes.device)] if len(boxes) > 1 else None
    entropy = float(-(weights * weights.clamp_min(1e-12).log()).sum())
    entropy_norm = entropy / math.log(len(weights)) if len(weights) > 1 else 0.0
    score_sorted = scores.sort(descending=True).values
    medoid_index = weighted_medoid(boxes, weights)
    vote = (boxes * weights[:, None]).sum(0)
    vote_cx = float((vote[0] + vote[2]) / 2)
    vote_cy = float((vote[1] + vote[3]) / 2)
    root_cx = float((root[0] + root[2]) / 2)
    root_cy = float((root[1] + root[3]) / 2)
    level_counts = Counter(int(x) for x in levels.tolist())
    return {
        "members": len(boxes),
        "effective_members": float(1.0 / weights.square().sum().clamp_min(1e-12)),
        "top1_score_fraction": float(weights[0]),
        "score_margin_1_2": float(score_sorted[0] - score_sorted[1]) if len(scores) > 1 else float(scores[0]),
        "score_entropy_norm": entropy_norm,
        "pairwise_iou_mean": float(off_diag.mean()) if off_diag is not None and len(off_diag) else 1.0,
        "pairwise_iou_min": float(off_diag.min()) if off_diag is not None and len(off_diag) else 1.0,
        "var_cx_norm": float(cx.var(unbiased=False) / (root_w**2)),
        "var_cy_norm": float(cy.var(unbiased=False) / (root_h**2)),
        "var_w_norm": float(width.var(unbiased=False) / (root_w**2)),
        "var_h_norm": float(height.var(unbiased=False) / (root_h**2)),
        "var_log_aspect": float((width / height).log().var(unbiased=False)),
        "winner_medoid_iou": float(box_iou(root[None], boxes[medoid_index : medoid_index + 1])[0, 0]),
        "vote_shift_norm": math.sqrt(((vote_cx - root_cx) / root_w) ** 2 + ((vote_cy - root_cy) / root_h) ** 2),
        "winner_level": root_level,
        "unique_levels": len(level_counts),
        "cross_level": len(level_counts) > 1,
        "level0_members": level_counts.get(0, 0),
        "level1_members": level_counts.get(1, 0),
        "level2_members": level_counts.get(2, 0),
    }


def original_size_bucket(
    gt_box: torch.Tensor,
    model_shape: tuple[int, int],
    original_shape: tuple[int, int],
    ratio_pad: Any,
) -> str:
    scaled = gt_box[None].clone()
    scale_boxes(model_shape, scaled, original_shape, ratio_pad)
    area = float((scaled[0, 2] - scaled[0, 0]).clamp_min(0) * (scaled[0, 3] - scaled[0, 1]).clamp_min(0))
    return "small" if area < 32**2 else "medium" if area < 96**2 else "large"


def canonical_coco_rows(
    predictions: torch.Tensor,
    image_id: int,
    category_ids: dict[int, int],
    model_shape: tuple[int, int],
    original_shape: tuple[int, int],
    ratio_pad: Any,
) -> list[dict[str, Any]]:
    if not len(predictions):
        return []
    scaled = predictions[:, :4].clone()
    scale_boxes(model_shape, scaled, original_shape, ratio_pad)
    xywh = xyxy2xywh(scaled)
    xywh[:, :2] -= xywh[:, 2:] / 2
    return [
        {
            "image_id": image_id,
            "category_id": category_ids[int(prediction[5])],
            "bbox": [round(float(value), 3) for value in box],
            "score": round(float(prediction[4]), 5),
        }
        for prediction, box in zip(predictions, xywh)
    ]


def build_variants(
    model_name: str,
    image_name: str,
    current: torch.Tensor,
    expanded: torch.Tensor,
    groups: list[torch.Tensor],
    gt_boxes: torch.Tensor,
    gt_classes: torch.Tensor,
    model_shape: tuple[int, int],
    original_shape: tuple[int, int],
    ratio_pad: Any,
    names: list[str],
    args: argparse.Namespace,
) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]]]:
    variants = {
        "current": current.clone(),
        "wide_candidate_oracle": current.clone(),
        "strict_candidate_oracle": current.clone(),
        "score_box_vote": current.clone(),
        "medoid": current.clone(),
        "strict_original_vote_oracle": current.clone(),
    }
    assignments, assigned_ious = one_to_one_gt_assignment(
        current, gt_boxes, gt_classes, args.assignment_min_iou
    )
    rows: list[dict[str, Any]] = []
    for output_index, members in enumerate(groups):
        boxes = expanded[members, :4].float()
        scores = expanded[members, 4].float()
        levels = expanded[members, 7].long()
        weights = scores.clamp_min(0)
        weights = weights / weights.sum().clamp_min(1e-12)
        vote = (boxes * weights[:, None]).sum(0)
        medoid_index = weighted_medoid(boxes, weights)
        variants["score_box_vote"][output_index, :4] = vote
        variants["medoid"][output_index, :4] = boxes[medoid_index]

        class_index = int(current[output_index, 5])
        same_gt_indices = torch.where(gt_classes.long() == class_index)[0]
        wide_iou = assigned_ious[output_index]
        wide_member = 0
        ambiguous_gt_count = 0
        if len(same_gt_indices):
            member_gt_iou = box_iou(boxes, gt_boxes[same_gt_indices].float())
            max_by_gt = member_gt_iou.max(0).values
            ambiguous_gt_count = int((max_by_gt >= args.ambiguous_gt_iou).sum())
            flat_index = int(member_gt_iou.flatten().argmax())
            candidate_member = flat_index // member_gt_iou.shape[1]
            candidate_iou = float(member_gt_iou.flatten()[flat_index])
            root_best = float(box_iou(boxes[0:1], gt_boxes[same_gt_indices].float()).max())
            if candidate_iou > root_best:
                variants["wide_candidate_oracle"][output_index, :4] = boxes[candidate_member]
                wide_member = candidate_member
                wide_iou = candidate_iou

        gt_index = assignments[output_index]
        strict_member = 0
        strict_iou = assigned_ious[output_index]
        vote_iou = 0.0
        size = "unmatched"
        if gt_index >= 0:
            target = gt_boxes[gt_index : gt_index + 1].float()
            member_iou = box_iou(boxes, target)[:, 0]
            strict_member = int(member_iou.argmax())
            strict_iou = float(member_iou[strict_member])
            vote_iou = float(box_iou(vote[None], target)[0, 0])
            variants["strict_candidate_oracle"][output_index, :4] = boxes[strict_member]
            if vote_iou > assigned_ious[output_index]:
                variants["strict_original_vote_oracle"][output_index, :4] = vote
            size = original_size_bucket(gt_boxes[gt_index], model_shape, original_shape, ratio_pad)

        row = {
            "model": model_name,
            "image": image_name,
            "output_index": output_index,
            "class": names[class_index],
            "score": float(current[output_index, 4]),
            "assigned_gt": gt_index,
            "size": size,
            "original_iou": assigned_ious[output_index],
            "vote_iou": vote_iou,
            "vote_delta_iou": vote_iou - assigned_ious[output_index] if gt_index >= 0 else None,
            "strict_candidate_iou": strict_iou,
            "strict_candidate_delta_iou": strict_iou - assigned_ious[output_index] if gt_index >= 0 else None,
            "wide_candidate_iou": wide_iou,
            "wide_candidate_member": wide_member,
            "strict_candidate_member": strict_member,
            "ambiguous_gt_count": ambiguous_gt_count,
            "vote_better": bool(gt_index >= 0 and vote_iou > assigned_ious[output_index]),
            "vote_harm_75": bool(gt_index >= 0 and assigned_ious[output_index] >= 0.75 and vote_iou < 0.75),
            "vote_help_75": bool(gt_index >= 0 and assigned_ious[output_index] < 0.75 and vote_iou >= 0.75),
        }
        row.update(normalized_cluster_features(boxes, scores, levels, int(levels[0])))
        rows.append(row)

    for name, variant in variants.items():
        if len(variant) != len(current):
            raise RuntimeError(f"{name} changed output count")
        if len(variant):
            if not torch.equal(variant[:, 5].long(), current[:, 5].long()):
                raise RuntimeError(f"{name} changed classes")
            if not torch.equal(variant[:, 4], current[:, 4]):
                raise RuntimeError(f"{name} changed scores/order")
    return variants, rows


def expected_rows(summary_path: Path) -> dict[str, dict[str, Any]]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return {row["model"]: row for row in summary["metrics"]["main"]}


def parity_report(
    model_name: str,
    current_metric: dict[str, Any],
    expected: dict[str, Any],
) -> dict[str, Any]:
    differences = {
        key: float(current_metric[key]) - float(expected[expected_key]) for key, expected_key in EXPECTED_KEYS.items()
    }
    return {
        "model": model_name,
        "expected_model": f"{model_name}_O2M",
        "differences": differences,
        "max_abs_error": max(abs(value) for value in differences.values()),
    }


def aggregate_group_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    keys = sorted({(row["model"], row["class"], row["size"]) for row in rows})
    keys += sorted({(row["model"], "all", "all") for row in rows})
    for model, class_name, size in keys:
        selected = [
            row
            for row in rows
            if row["model"] == model
            and (class_name == "all" or row["class"] == class_name)
            and (size == "all" or row["size"] == size)
        ]
        matched = [row for row in selected if row["assigned_gt"] >= 0]
        output.append(
            {
                "model": model,
                "class": class_name,
                "size": size,
                "groups": len(selected),
                "matched_groups": len(matched),
                "multi_member_groups": sum(row["members"] > 1 for row in selected),
                "ambiguous_groups": sum(row["ambiguous_gt_count"] > 1 for row in selected),
                "vote_better_rate": sum(row["vote_better"] for row in matched) / len(matched) if matched else None,
                "vote_harm_75": sum(row["vote_harm_75"] for row in matched),
                "vote_help_75": sum(row["vote_help_75"] for row in matched),
                "vote_delta_iou_mean": (
                    sum(float(row["vote_delta_iou"]) for row in matched) / len(matched) if matched else None
                ),
                "strict_candidate_delta_iou_mean": (
                    sum(float(row["strict_candidate_delta_iou"]) for row in matched) / len(matched) if matched else None
                ),
            }
        )
    return output


def main() -> None:
    args = parse_args()
    for name in ("r1", "gbrg", "canonical_summary", "data", "output"):
        value = getattr(args, name)
        setattr(args, name, value.expanduser().resolve())
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {args.output}")
    if args.batch != 32:
        raise ValueError("Canonical stage-1 protocol requires batch=32")
    args.output.mkdir(parents=True)

    data_info = check_det_dataset(str(args.data))
    coco_path = Path(data_info["path"]) / "annotations" / "instances_val.json"
    coco_gt = COCO(str(coco_path))
    image_id_by_stem = {Path(image["file_name"]).stem: image_id for image_id, image in coco_gt.imgs.items()}
    category_id_by_name = {category["name"]: category_id for category_id, category in coco_gt.cats.items()}
    names = [data_info["names"][index] for index in range(len(data_info["names"]))]
    category_ids = {index: category_id_by_name[name] for index, name in enumerate(names)}
    canonical = expected_rows(args.canonical_summary)
    device = select_device(args.device, verbose=False)

    model_paths = {"R1": args.r1, "GBRG": args.gbrg}
    checkpoint_before = {name: sha256(path) for name, path in model_paths.items()}
    prediction_rows: dict[str, dict[str, list[dict[str, Any]]]] = {}
    group_rows: list[dict[str, Any]] = []
    reconstruction_rows: list[dict[str, Any]] = []

    for model_name, checkpoint in model_paths.items():
        _, loader = make_loader(args.data, args)
        wrapped = YOLO(str(checkpoint))
        net = wrapped.model.to(device).float().eval()
        head = net.model[-1]
        if not isinstance(head, Detect) or not head.end2end or head.nc != len(names):
            raise RuntimeError(f"{model_name} is not a compatible four-class end-to-end checkpoint")
        if isinstance(net.args, dict):
            net.args = SimpleNamespace(**net.args)
        prediction_rows[model_name] = {}
        seen_images = 0

        with torch.inference_mode():
            for batch in loader:
                images = batch["img"].to(device, non_blocking=True).float() / 255.0
                device_batch = {
                    key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
                    for key, value in batch.items()
                }
                device_batch["img"] = images
                output = net(images)  # Locked canonical FP32 inference: no autocast.
                raw = output[1] if isinstance(output, tuple) else output
                one2many = raw["one2many"]
                decoded = head._inference(one2many).float()
                level_by_raw = raw_level_indices(one2many["feats"], device)
                model_shape = tuple(images.shape[-2:])

                for image_index in range(images.shape[0]):
                    absolute_index = seen_images + image_index
                    if args.max_images and absolute_index >= args.max_images:
                        continue
                    image_name = Path(batch["im_file"][image_index]).name
                    image_id = image_id_by_stem[Path(image_name).stem]
                    gt_boxes, gt_classes = paired_gt(
                        device_batch, image_index, model_shape[1], model_shape[0]
                    )
                    expanded = expanded_candidates(decoded[image_index], level_by_raw, head.nc, args)
                    reference = current_nms(decoded[image_index], head.nc, args)
                    roots, groups, check = groups_anchored_to_reference(
                        expanded[:, :6], reference, args.nms_iou
                    )
                    # Restore provenance columns after group recovery; indices are unchanged.
                    current = expanded[roots, :6].clone()
                    check.update(
                        {
                            "model": model_name,
                            "image": image_name,
                            "expanded_candidates": len(expanded),
                            "nms_outputs": len(current),
                        }
                    )
                    reconstruction_rows.append(check)
                    variants, image_group_rows = build_variants(
                        model_name,
                        image_name,
                        current,
                        expanded,
                        groups,
                        gt_boxes,
                        gt_classes,
                        model_shape,
                        batch["ori_shape"][image_index],
                        batch["ratio_pad"][image_index],
                        names,
                        args,
                    )
                    group_rows.extend(image_group_rows)
                    for case, variant in variants.items():
                        prediction_rows[model_name].setdefault(case, []).extend(
                            canonical_coco_rows(
                                variant,
                                image_id,
                                category_ids,
                                model_shape,
                                batch["ori_shape"][image_index],
                                batch["ratio_pad"][image_index],
                            )
                        )

                seen_images += images.shape[0]
                print(
                    f"STRICT_STAGE1_PROGRESS model={model_name} "
                    f"images={min(seen_images, len(loader.dataset))}/{len(loader.dataset)}",
                    flush=True,
                )
                if args.max_images and seen_images >= args.max_images:
                    break
        del net, wrapped
        if device.type == "cuda":
            torch.cuda.empty_cache()

    metrics: list[dict[str, Any]] = []
    metric_lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for model_name, cases in prediction_rows.items():
        model_dir = args.output / model_name.lower()
        model_dir.mkdir()
        for case, rows in cases.items():
            path = model_dir / f"predictions_{case}.json"
            path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
            metric = metric_row(case, coco_gt, rows, category_id_by_name)
            metric = {"model": model_name, **metric}
            metrics.append(metric)
            metric_lookup[(model_name, case)] = metric

    parity = [
        parity_report(model_name, metric_lookup[(model_name, "current")], canonical[f"{model_name}_O2M"])
        for model_name in model_paths
    ]
    protocol_max_error = max(row["max_abs_error"] for row in parity)
    group_summary = aggregate_group_summary(group_rows)

    current = metric_lookup[("GBRG", "current")]
    vote = metric_lookup[("GBRG", "score_box_vote")]
    strict_ov = metric_lookup[("GBRG", "strict_original_vote_oracle")]
    best_realized_ap = max(float(current["AP"]), float(vote["AP"]))
    strict_ov_headroom = float(strict_ov["AP"]) - best_realized_ap
    d40_restored = float(strict_ov["D40_AP75"]) >= float(current["D40_AP75"])
    if protocol_max_error > args.protocol_tolerance:
        decision = "PROTOCOL_MISMATCH_STOP"
        stage2_allowed = False
    elif strict_ov_headroom < args.oracle_no_go or not d40_restored:
        decision = "NO_GO_CONSERVATIVE_GATE"
        stage2_allowed = False
    elif strict_ov_headroom < args.oracle_go:
        decision = "BORDERLINE_DO_NOT_AUTO_ADVANCE"
        stage2_allowed = False
    else:
        decision = "GO_STAGE2_PREDICTABILITY_AUDIT"
        stage2_allowed = True

    r1_vote_gain = float(metric_lookup[("R1", "score_box_vote")]["AP"]) - float(
        metric_lookup[("R1", "current")]["AP"]
    )
    gbrg_vote_gain = float(vote["AP"]) - float(current["AP"])
    r1_oracle_headroom = float(metric_lookup[("R1", "strict_original_vote_oracle")]["AP"]) - max(
        float(metric_lookup[("R1", "current")]["AP"]),
        float(metric_lookup[("R1", "score_box_vote")]["AP"]),
    )

    checkpoint_after = {name: sha256(path) for name, path in model_paths.items()}
    summary = {
        "protocol": {
            "split": "Val only; Test untouched",
            "data": str(args.data),
            "imgsz": args.imgsz,
            "batch": args.batch,
            "workers": args.workers,
            "rect": True,
            "precision": "FP32; autocast disabled",
            "score_floor": args.score_floor,
            "nms_iou": args.nms_iou,
            "max_det": args.max_det,
            "max_nms": args.max_nms,
            "coco_max_dets": 100,
            "json_rounding": "bbox=3 decimals; score=5 decimals",
            "assignment": f"same-class greedy one-to-one by IoU, minimum {args.assignment_min_iou}",
            "invariant": "scores, classes, count, and global order unchanged; coordinates only",
            "canonical_summary": str(args.canonical_summary),
        },
        "checkpoints": {
            name: {
                "path": str(path),
                "sha256_before": checkpoint_before[name],
                "sha256_after": checkpoint_after[name],
                "unchanged": checkpoint_before[name] == checkpoint_after[name],
            }
            for name, path in model_paths.items()
        },
        "nms_reconstruction": {
            "images": len(reconstruction_rows),
            "max_box_error": max(row["max_box_error"] for row in reconstruction_rows),
            "max_score_error": max(row["max_score_error"] for row in reconstruction_rows),
            "all_class_equal": all(row["class_equal"] for row in reconstruction_rows),
        },
        "canonical_parity": parity,
        "metrics": metrics,
        "interaction": {
            "R1_vote_gain_AP": r1_vote_gain,
            "GBRG_vote_gain_AP": gbrg_vote_gain,
            "vote_gain_difference_GBRG_minus_R1": gbrg_vote_gain - r1_vote_gain,
            "R1_strict_original_vote_headroom_over_best_realized": r1_oracle_headroom,
            "GBRG_strict_original_vote_headroom_over_best_realized": strict_ov_headroom,
            "strict_headroom_difference_GBRG_minus_R1": strict_ov_headroom - r1_oracle_headroom,
        },
        "gate": {
            "protocol_max_abs_error": protocol_max_error,
            "protocol_tolerance": args.protocol_tolerance,
            "GBRG_best_realized_AP": best_realized_ap,
            "GBRG_strict_original_vote_oracle_AP": strict_ov["AP"],
            "GBRG_strict_original_vote_headroom": strict_ov_headroom,
            "GBRG_current_D40_AP75": current["D40_AP75"],
            "GBRG_strict_original_vote_D40_AP75": strict_ov["D40_AP75"],
            "D40_restored": d40_restored,
            "no_go_below": args.oracle_no_go,
            "go_at_least": args.oracle_go,
            "decision": decision,
            "stage2_allowed": stage2_allowed,
        },
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    write_csv(args.output / "metrics.csv", metrics)
    write_csv(args.output / "canonical_parity.csv", parity)
    write_csv(args.output / "nms_reconstruction.csv", reconstruction_rows)
    write_csv(args.output / "groups.csv", group_rows)
    write_csv(args.output / "group_summary.csv", group_summary)
    print(json.dumps(summary["gate"], indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
