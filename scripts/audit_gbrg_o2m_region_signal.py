"""Stage-2 Train-to-Val audit of GBRG Gaussian-region representative signals.

This is a frozen-model audit.  It never changes detector weights or global
detection scores.  Region quality is used only inside exact greedy-NMS
suppression groups, and the primary SP-GRCS rule is fixed before Val metrics.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from pycocotools.coco import COCO

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from audit_gbrg_o2m_csrg_candidates import coco_rows, metric_row, paired_gt  # noqa: E402
from audit_gbrg_o2m_representative_oracle import (  # noqa: E402
    current_nms,
    expanded_candidates,
)
from ultralytics import YOLO  # noqa: E402
from ultralytics.cfg import get_cfg  # noqa: E402
from ultralytics.data.build import build_dataloader, build_yolo_dataset  # noqa: E402
from ultralytics.data.utils import check_det_dataset  # noqa: E402
from ultralytics.nn.roadsnake import RoadSnakeGBRGDetect  # noqa: E402
from ultralytics.utils import DEFAULT_CFG  # noqa: E402
from ultralytics.utils.metrics import box_iou  # noqa: E402
from ultralytics.utils.torch_utils import select_device  # noqa: E402


METHODS = ("score", "medoid", "region", "region_medoid_borda")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--score-floor", type=float, default=0.001)
    parser.add_argument("--nms-iou", type=float, default=0.70)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--max-nms", type=int, default=30000)
    parser.add_argument("--quality-gap", type=float, default=0.05)
    parser.add_argument("--match-iou", type=float, default=0.50)
    parser.add_argument("--max-train-images", type=int, default=0)
    parser.add_argument("--max-val-images", type=int, default=0)
    parser.add_argument("--pair-delta-gate", type=float, default=0.05)
    parser.add_argument("--ap-delta-gate", type=float, default=0.002)
    parser.add_argument("--ap75-tolerance", type=float, default=0.001)
    parser.add_argument("--aps-tolerance", type=float, default=0.001)
    parser.add_argument("--ar-tolerance", type=float, default=0.002)
    parser.add_argument("--d40-ap75-tolerance", type=float, default=0.002)
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


def make_split_loader(data_yaml: Path, split: str, args: argparse.Namespace):
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
    dataset = build_yolo_dataset(cfg, data[split], args.batch, data, mode="val", rect=True, stride=32)
    loader = build_dataloader(dataset, args.batch, args.workers, shuffle=False, rank=-1, pin_memory=False)
    return data, loader


def target_size(box: torch.Tensor) -> str:
    area = float((box[2] - box[0]).clamp_min(0) * (box[3] - box[1]).clamp_min(0))
    return "small" if area < 32**2 else "medium" if area < 96**2 else "large"


def centered_cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = a.float() - a.float().mean()
    b = b.float() - b.float().mean()
    denominator = a.square().sum().sqrt() * b.square().sum().sqrt()
    return (a * b).sum() / denominator.clamp_min(1e-12)


def region_consistency_scores(
    region_probability: torch.Tensor,
    boxes: torch.Tensor,
    model_height: int,
    model_width: int,
) -> torch.Tensor:
    """Compare frozen P3 region evidence with each candidate's GBRG Gaussian template."""
    if len(boxes) == 1:
        return boxes.new_zeros((1,), dtype=torch.float32)
    map_height, map_width = region_probability.shape[-2:]
    scale = boxes.new_tensor((map_width / model_width, map_height / model_height, map_width / model_width, map_height / model_height))
    mapped = boxes.float() * scale
    x0 = max(0, int(torch.floor(mapped[:, 0].min()).item()) - 1)
    y0 = max(0, int(torch.floor(mapped[:, 1].min()).item()) - 1)
    x1 = min(map_width, int(torch.ceil(mapped[:, 2].max()).item()) + 1)
    y1 = min(map_height, int(torch.ceil(mapped[:, 3].max()).item()) + 1)
    if x1 <= x0 or y1 <= y0:
        return boxes.new_zeros((len(boxes),), dtype=torch.float32)

    evidence = region_probability[y0:y1, x0:x1].float()
    grid_y = torch.arange(y0, y1, device=boxes.device, dtype=torch.float32).view(-1, 1) + 0.5
    grid_x = torch.arange(x0, x1, device=boxes.device, dtype=torch.float32).view(1, -1) + 0.5
    scores = []
    for box in mapped:
        cx = (box[0] + box[2]) / 2
        cy = (box[1] + box[3]) / 2
        width = (box[2] - box[0]).clamp_min(1.0)
        height = (box[3] - box[1]).clamp_min(1.0)
        sigma_x = (width / 6.0).clamp_min(1.0)
        sigma_y = (height / 6.0).clamp_min(1.0)
        gaussian = torch.exp(-0.5 * (((grid_x - cx) / sigma_x).square() + ((grid_y - cy) / sigma_y).square()))
        inside = (grid_x >= box[0]) & (grid_x <= box[2]) & (grid_y >= box[1]) & (grid_y <= box[3])
        scores.append(centered_cosine(evidence, gaussian * inside))
    return torch.stack(scores)


def rank01(values: torch.Tensor) -> torch.Tensor:
    if len(values) <= 1:
        return torch.zeros_like(values, dtype=torch.float32)
    order = values.argsort()
    ranks = torch.empty_like(values, dtype=torch.float32)
    ranks[order] = torch.arange(len(values), device=values.device, dtype=torch.float32)
    return ranks / (len(values) - 1)


def method_scores(
    expanded: torch.Tensor,
    members: torch.Tensor,
    region_probability: torch.Tensor,
    model_height: int,
    model_width: int,
) -> dict[str, torch.Tensor]:
    boxes = expanded[members, :4].float()
    score = expanded[members, 4].float()
    weights = score.clamp_min(0)
    weights = weights / weights.sum().clamp_min(1e-12)
    medoid = box_iou(boxes, boxes) @ weights if len(boxes) > 1 else score.new_zeros((1,))
    region = region_consistency_scores(region_probability, boxes, model_height, model_width)
    borda = rank01(region) + rank01(medoid)
    return {"score": score, "medoid": medoid, "region": region, "region_medoid_borda": borda}


def groups_anchored_to_reference(
    expanded: torch.Tensor,
    reference: torch.Tensor,
    iou_threshold: float,
) -> tuple[list[int], list[torch.Tensor], dict[str, Any]]:
    """Map exact NMS outputs back to candidates and recover their direct suppression groups.

    Anchoring to the actual torchvision/Ultralytics output avoids changing the
    winner when candidates have exactly tied scores with backend-specific order.
    """
    if not len(reference):
        return [], [], {"rows": 0, "max_box_error": 0.0, "max_score_error": 0.0, "class_equal": True}
    used = torch.zeros(len(expanded), dtype=torch.bool, device=expanded.device)
    roots: list[int] = []
    for detection in reference:
        same_class = expanded[:, 5].long() == int(detection[5])
        same_score = torch.isclose(expanded[:, 4], detection[4], atol=1e-7, rtol=0)
        candidates = torch.where(same_class & same_score & ~used)[0]
        if not len(candidates):
            raise RuntimeError("Could not map an exact NMS output to the expanded candidate set")
        errors = (expanded[candidates, :4] - detection[:4]).abs().amax(1)
        best_local = int(errors.argmin())
        if float(errors[best_local]) > 1e-4:
            raise RuntimeError(f"NMS output-to-candidate mapping error: {float(errors[best_local]):.6g}")
        root = int(candidates[best_local])
        roots.append(root)
        used[root] = True

    root_mask = torch.zeros(len(expanded), dtype=torch.bool, device=expanded.device)
    root_mask[torch.tensor(roots, device=expanded.device)] = True
    assigned = root_mask.clone()
    groups: list[torch.Tensor] = []
    for root in roots:
        eligible = ~assigned & ~root_mask & (expanded[:, 5] == expanded[root, 5])
        indices = torch.where(eligible)[0]
        if len(indices):
            overlaps = box_iou(expanded[root : root + 1, :4].float(), expanded[indices, :4].float())[0]
            suppressed = indices[overlaps > iou_threshold]
        else:
            suppressed = indices
        group = torch.cat((torch.tensor([root], device=expanded.device), suppressed))
        groups.append(group)
        assigned[suppressed] = True

    reconstructed = expanded[roots]
    box_error = float((reconstructed[:, :4] - reference[:, :4]).abs().max())
    score_error = float((reconstructed[:, 4] - reference[:, 4]).abs().max())
    class_equal = bool(torch.equal(reconstructed[:, 5].long(), reference[:, 5].long()))
    if box_error > 1e-4 or score_error > 1e-7 or not class_equal:
        raise RuntimeError("Reference-anchored NMS mapping changed the actual output")
    return roots, groups, {
        "rows": len(reference),
        "max_box_error": box_error,
        "max_score_error": score_error,
        "class_equal": class_equal,
    }


def add_signal_stat(
    accumulator: dict[tuple[str, str, str, str], dict[str, float]],
    split: str,
    method: str,
    class_name: str,
    size: str,
    quality: torch.Tensor,
    signal: torch.Tensor,
    quality_gap: float,
) -> tuple[float, int, float]:
    selected = int(signal.argmax())
    best_quality = float(quality.max())
    selected_quality = float(quality[selected])
    better, worse = torch.where((quality[:, None] - quality[None, :]) >= quality_gap)
    if len(better):
        differences = signal[better] - signal[worse]
        correct = float((differences > 0).sum()) + 0.5 * float((differences == 0).sum())
        pair_count = len(better)
    else:
        correct = 0.0
        pair_count = 0
    top1 = float(selected_quality >= best_quality - 1e-6)
    for group_class, group_size in ((class_name, size), ("all", "all")):
        row = accumulator[(split, method, group_class, group_size)]
        row["groups"] += 1
        row["pairs"] += pair_count
        row["pair_correct"] += correct
        row["top1_correct"] += top1
        row["selected_iou_sum"] += selected_quality
        row["oracle_iou_sum"] += best_quality
    return selected_quality, pair_count, top1


def summarize_signal(accumulator: dict[tuple[str, str, str, str], dict[str, float]]) -> list[dict[str, Any]]:
    rows = []
    for (split, method, class_name, size), values in sorted(accumulator.items()):
        groups = values["groups"]
        pairs = values["pairs"]
        rows.append(
            {
                "split": split,
                "method": method,
                "class": class_name,
                "size": size,
                "groups": int(groups),
                "pairs": int(pairs),
                "pair_accuracy": values["pair_correct"] / pairs if pairs else None,
                "top1_accuracy": values["top1_correct"] / groups if groups else None,
                "selected_iou_mean": values["selected_iou_sum"] / groups if groups else None,
                "oracle_iou_mean": values["oracle_iou_sum"] / groups if groups else None,
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    args.checkpoint = args.checkpoint.expanduser().resolve()
    args.data = args.data.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {args.output}")
    args.output.mkdir(parents=True)

    data_info = check_det_dataset(str(args.data))
    names = [data_info["names"][index] for index in range(len(data_info["names"]))]
    device = select_device(args.device, verbose=False)
    wrapped = YOLO(str(args.checkpoint))
    net = wrapped.model.to(device).float().eval()
    head = net.model[-1]
    if not isinstance(head, RoadSnakeGBRGDetect) or head.gbrg_region_head is None or not head.end2end:
        raise RuntimeError("Checkpoint must contain an unfused end-to-end RoadSnakeGBRGDetect head")
    if isinstance(net.args, dict):
        net.args = SimpleNamespace(**net.args)
    checkpoint_before = sha256(args.checkpoint)

    captured: dict[str, torch.Tensor] = {}

    def capture_detect_input(_module, inputs) -> None:
        features = inputs[0]
        captured["p3"] = features[0]

    hook = head.register_forward_pre_hook(capture_detect_input)
    accumulator: dict[tuple[str, str, str, str], dict[str, float]] = defaultdict(
        lambda: {
            "groups": 0.0,
            "pairs": 0.0,
            "pair_correct": 0.0,
            "top1_correct": 0.0,
            "selected_iou_sum": 0.0,
            "oracle_iou_sum": 0.0,
        }
    )
    group_rows: list[dict[str, Any]] = []
    val_predictions: dict[str, list[dict[str, Any]]] = {
        "current": [],
        "medoid_rep": [],
        "region_rep": [],
        "region_medoid_borda": [],
    }
    reconstruction: list[dict[str, Any]] = []
    amp = device.type == "cuda"

    for split in ("train", "val"):
        _, loader = make_split_loader(args.data, split, args)
        max_images = args.max_train_images if split == "train" else args.max_val_images
        if split == "val":
            coco_path = Path(data_info["path"]) / "annotations" / "instances_val.json"
            coco_gt = COCO(str(coco_path))
            image_id_by_stem = {Path(image["file_name"]).stem: image_id for image_id, image in coco_gt.imgs.items()}
            category_id_by_name = {category["name"]: category_id for category_id, category in coco_gt.cats.items()}
            category_ids = {index: category_id_by_name[name] for index, name in enumerate(names)}
        seen_images = 0

        with torch.inference_mode():
            for batch in loader:
                images = batch["img"].to(device, non_blocking=True).float() / 255.0
                device_batch = {
                    key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
                    for key, value in batch.items()
                }
                device_batch["img"] = images
                captured.clear()
                with torch.autocast(device_type=device.type, enabled=amp):
                    output = net(images)
                p3 = captured.pop("p3")
                region_probability = head.gbrg_region_head(p3.float()).sigmoid()
                raw = output[1] if isinstance(output, tuple) else output
                decoded = head._inference(raw["one2many"]).float()
                height, width = images.shape[-2:]

                for image_index in range(images.shape[0]):
                    absolute_index = seen_images + image_index
                    if max_images and absolute_index >= max_images:
                        continue
                    image_name = Path(batch["im_file"][image_index]).name
                    gt_boxes, gt_classes = paired_gt(device_batch, image_index, width, height)
                    expanded = expanded_candidates(decoded[image_index], head.nc, args)
                    reference = current_nms(decoded[image_index], head.nc, args)
                    selected_roots, selected_groups, check = groups_anchored_to_reference(
                        expanded, reference, args.nms_iou
                    )
                    current = reference.clone()
                    check.update({"split": split, "image": image_name})
                    reconstruction.append(check)
                    variants = {
                        "current": current.clone(),
                        "medoid_rep": current.clone(),
                        "region_rep": current.clone(),
                        "region_medoid_borda": current.clone(),
                    }

                    for output_index, (root, members) in enumerate(zip(selected_roots, selected_groups)):
                        class_index = int(expanded[root, 5])
                        same_gt_indices = torch.where(gt_classes == class_index)[0]
                        matched = False
                        quality = None
                        size = None
                        if len(same_gt_indices):
                            member_gt_iou = box_iou(expanded[members, :4].float(), gt_boxes[same_gt_indices].float())
                            flat = int(member_gt_iou.argmax())
                            member_index = flat // member_gt_iou.shape[1]
                            target_local = flat % member_gt_iou.shape[1]
                            best_iou = float(member_gt_iou[member_index, target_local])
                            if best_iou >= args.match_iou:
                                matched = True
                                quality = member_gt_iou[:, target_local]
                                target_box = gt_boxes[same_gt_indices[target_local]]
                                size = target_size(target_box)
                        # Train is used only for matched-group signal statistics.  Val must
                        # still transform every group to measure the deployable full pipeline.
                        if split == "train" and not matched:
                            continue

                        signals = method_scores(
                            expanded,
                            members,
                            region_probability[image_index, 0],
                            height,
                            width,
                        )
                        selected_by_method = {method: int(values.argmax()) for method, values in signals.items()}
                        variants["medoid_rep"][output_index, :4] = expanded[members[selected_by_method["medoid"]], :4]
                        variants["region_rep"][output_index, :4] = expanded[members[selected_by_method["region"]], :4]
                        variants["region_medoid_borda"][output_index, :4] = expanded[
                            members[selected_by_method["region_medoid_borda"]], :4
                        ]

                        if matched:
                            selected_quality = {}
                            pair_counts = {}
                            for method in METHODS:
                                selected_iou, pair_count, _ = add_signal_stat(
                                    accumulator,
                                    split,
                                    method,
                                    names[class_index],
                                    size,
                                    quality,
                                    signals[method],
                                    args.quality_gap,
                                )
                                selected_quality[method] = selected_iou
                                pair_counts[method] = pair_count
                            group_rows.append(
                                {
                                    "split": split,
                                    "image": image_name,
                                    "output_index": output_index,
                                    "class": names[class_index],
                                    "size": size,
                                    "members": len(members),
                                    "oracle_iou": float(quality.max()),
                                    **{f"selected_iou_{method}": selected_quality[method] for method in METHODS},
                                    **{f"selected_index_{method}": selected_by_method[method] for method in METHODS},
                                    "pair_count": pair_counts["score"],
                                }
                            )

                    for name, variant in variants.items():
                        if len(variant):
                            if not torch.equal(variant[:, 4], current[:, 4]):
                                raise RuntimeError(f"{name} changed global scores/order")
                            if not torch.equal(variant[:, 5].long(), current[:, 5].long()):
                                raise RuntimeError(f"{name} changed classes")
                        if split == "val":
                            image_id = image_id_by_stem[Path(image_name).stem]
                            common = (
                                image_id,
                                category_ids,
                                (height, width),
                                batch["ori_shape"][image_index],
                                batch["ratio_pad"][image_index],
                            )
                            val_predictions[name].extend(coco_rows(variant, *common))

                seen_images += images.shape[0]
                print(
                    f"STAGE2_PROGRESS split={split} images={min(seen_images, len(loader.dataset))}/{len(loader.dataset)}",
                    flush=True,
                )
                if max_images and seen_images >= max_images:
                    break

    hook.remove()
    signal_summary = summarize_signal(accumulator)
    metrics = []
    prediction_manifest = {}
    for name, rows in val_predictions.items():
        path = args.output / f"predictions_{name}.json"
        path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
        prediction_manifest[name] = {"path": str(path), "rows": len(rows)}
        metrics.append(metric_row(name, coco_gt, rows, category_id_by_name))

    signal_lookup = {
        (row["split"], row["method"], row["class"], row["size"]): row for row in signal_summary
    }
    train_score = signal_lookup[("train", "score", "all", "all")]
    train_region = signal_lookup[("train", "region", "all", "all")]
    val_score = signal_lookup[("val", "score", "all", "all")]
    val_region = signal_lookup[("val", "region", "all", "all")]
    metric_lookup = {row["case"]: row for row in metrics}
    current_metric = metric_lookup["current"]
    region_metric = metric_lookup["region_rep"]
    pair_delta_train = train_region["pair_accuracy"] - train_score["pair_accuracy"]
    pair_delta_val = val_region["pair_accuracy"] - val_score["pair_accuracy"]
    ap_delta = region_metric["AP"] - current_metric["AP"]
    checks = {
        "train_pair_delta_at_least_gate": pair_delta_train >= args.pair_delta_gate,
        "val_pair_delta_positive": pair_delta_val > 0,
        "full_val_AP_delta_at_least_gate": ap_delta >= args.ap_delta_gate,
        "AP75_preserved": region_metric["AP75"] >= current_metric["AP75"] - args.ap75_tolerance,
        "AP_small_preserved": region_metric["AP_small"] >= current_metric["AP_small"] - args.aps_tolerance,
        "AR100_preserved": region_metric["AR100"] >= current_metric["AR100"] - args.ar_tolerance,
        "D40_AP75_preserved": region_metric["D40_AP75"] >= current_metric["D40_AP75"] - args.d40_ap75_tolerance,
    }
    passed = all(checks.values())
    decision = "GO_IMPLEMENT_SP_GRCS" if passed else "NO_GO_SP_GRCS_REGION_SIGNAL"

    write_csv(args.output / "group_signal.csv", group_rows)
    write_csv(args.output / "signal_summary.csv", signal_summary)
    write_csv(args.output / "metrics.csv", metrics)
    write_csv(args.output / "nms_reconstruction.csv", reconstruction)
    checkpoint_after = sha256(args.checkpoint)
    summary = {
        "protocol": {
            "train_view": "Train membership/labels with deterministic val-style preprocessing; no fitting",
            "val_view": "Val full COCO; Test untouched",
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256_before": checkpoint_before,
            "checkpoint_sha256_after": checkpoint_after,
            "checkpoint_unchanged": checkpoint_before == checkpoint_after,
            "score_floor": args.score_floor,
            "nms_iou": args.nms_iou,
            "max_det": args.max_det,
            "quality_gap": args.quality_gap,
            "match_iou": args.match_iou,
            "primary_rule": "within each exact NMS suppression group choose max Gaussian-template NCC; preserve score/class/count/order",
            "controls": ["original classification score", "score-weighted IoU medoid", "equal-rank region+medoid Borda"],
        },
        "nms_reconstruction": {
            "images": len(reconstruction),
            "max_box_error": max((row["max_box_error"] for row in reconstruction), default=0.0),
            "max_score_error": max((row["max_score_error"] for row in reconstruction), default=0.0),
            "all_class_equal": all(row["class_equal"] for row in reconstruction),
        },
        "signal_summary": signal_summary,
        "metrics": metrics,
        "predictions": prediction_manifest,
        "gate": {
            "train_score_pair_accuracy": train_score["pair_accuracy"],
            "train_region_pair_accuracy": train_region["pair_accuracy"],
            "train_pair_accuracy_delta": pair_delta_train,
            "val_score_pair_accuracy": val_score["pair_accuracy"],
            "val_region_pair_accuracy": val_region["pair_accuracy"],
            "val_pair_accuracy_delta": pair_delta_val,
            "current_AP": current_metric["AP"],
            "region_rep_AP": region_metric["AP"],
            "region_rep_delta_AP": ap_delta,
            "checks": checks,
            "decision": decision,
        },
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary["gate"], indent=2), flush=True)


if __name__ == "__main__":
    main()
