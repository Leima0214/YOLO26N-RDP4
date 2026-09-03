"""Stage-1 score-preserving O2M representative-selection audit for frozen GBRG.

The audit reconstructs the exact class-aware greedy-NMS suppression groups.
Every evaluated variant keeps the original output count, class labels, scores,
and score order.  Only box coordinates may change.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
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

from audit_gbrg_o2m_csrg_candidates import (  # noqa: E402
    coco_rows,
    make_loader,
    metric_row,
    paired_gt,
)
from ultralytics import YOLO  # noqa: E402
from ultralytics.data.utils import check_det_dataset  # noqa: E402
from ultralytics.nn.modules.head import Detect  # noqa: E402
from ultralytics.utils.metrics import box_iou  # noqa: E402
from ultralytics.utils.nms import non_max_suppression  # noqa: E402
from ultralytics.utils.ops import xywh2xyxy, xyxy2xywh  # noqa: E402
from ultralytics.utils.torch_utils import select_device  # noqa: E402


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
    parser.add_argument("--oracle-min-iou", type=float, default=0.50)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--no-go-gate", type=float, default=0.003)
    parser.add_argument("--go-gate", type=float, default=0.005)
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


def expanded_candidates(decoded: torch.Tensor, nc: int, args: argparse.Namespace) -> torch.Tensor:
    """Reproduce the multi-label candidate matrix immediately before NMS."""
    boxes = xywh2xyxy(xyxy2xywh(decoded[:4].T))
    class_scores = decoded[4 : 4 + nc].T
    candidate = class_scores.amax(1) > args.score_floor
    boxes, class_scores = boxes[candidate], class_scores[candidate]
    raw_index, class_index = torch.where(class_scores > args.score_floor)
    expanded = torch.cat(
        (
            boxes[raw_index],
            class_scores[raw_index, class_index, None],
            class_index[:, None].float(),
        ),
        dim=1,
    )
    if len(expanded) > args.max_nms:
        order = expanded[:, 4].argsort(descending=True)[: args.max_nms]
        expanded = expanded[order]
    return expanded


def greedy_groups(expanded: torch.Tensor, iou_threshold: float) -> tuple[list[int], list[torch.Tensor]]:
    """Return greedy-NMS roots and each root's directly suppressed candidates."""
    if not len(expanded):
        return [], []
    order = expanded[:, 4].argsort(descending=True)
    roots: list[int] = []
    groups: list[torch.Tensor] = []
    while order.numel():
        root = order[0]
        roots.append(int(root))
        if order.numel() == 1:
            groups.append(root.view(1))
            break
        rest = order[1:]
        same_class = expanded[rest, 5] == expanded[root, 5]
        overlaps = box_iou(expanded[root : root + 1, :4].float(), expanded[rest, :4].float())[0]
        suppressed = same_class & (overlaps > iou_threshold)
        groups.append(torch.cat((root.view(1), rest[suppressed])))
        order = rest[~suppressed]
    return roots, groups


def verify_current(reconstructed: torch.Tensor, reference: torch.Tensor) -> dict[str, Any]:
    if reconstructed.shape != reference.shape:
        raise RuntimeError(f"NMS reconstruction shape mismatch: {reconstructed.shape} != {reference.shape}")
    if not len(reference):
        return {"rows": 0, "max_box_error": 0.0, "max_score_error": 0.0, "class_equal": True}
    box_error = float((reconstructed[:, :4] - reference[:, :4]).abs().max())
    score_error = float((reconstructed[:, 4] - reference[:, 4]).abs().max())
    class_equal = bool(torch.equal(reconstructed[:, 5].long(), reference[:, 5].long()))
    if box_error > 1e-4 or score_error > 1e-6 or not class_equal:
        raise RuntimeError(
            f"NMS reconstruction mismatch: box={box_error:.3g}, score={score_error:.3g}, class={class_equal}"
        )
    return {
        "rows": len(reference),
        "max_box_error": box_error,
        "max_score_error": score_error,
        "class_equal": class_equal,
    }


def representative_variants(
    expanded: torch.Tensor,
    roots: list[int],
    groups: list[torch.Tensor],
    gt_boxes: torch.Tensor,
    gt_classes: torch.Tensor,
    args: argparse.Namespace,
    image_name: str,
    names: list[str],
) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]]]:
    selected_roots = roots[: args.max_det]
    selected_groups = groups[: args.max_det]
    current = expanded[selected_roots].clone()
    oracle = current.clone()
    medoid = current.clone()
    box_vote = current.clone()
    rows: list[dict[str, Any]] = []

    for output_index, (root, members) in enumerate(zip(selected_roots, selected_groups)):
        boxes = expanded[members, :4].float()
        scores = expanded[members, 4].float()
        class_index = int(expanded[root, 5])
        same_gt = gt_boxes[gt_classes == class_index].float()
        current_best = float(box_iou(expanded[root : root + 1, :4].float(), same_gt).max()) if len(same_gt) else 0.0

        oracle_member = 0
        oracle_best = current_best
        if len(same_gt):
            member_gt_iou = box_iou(boxes, same_gt)
            flat_index = int(member_gt_iou.argmax())
            candidate_member = flat_index // member_gt_iou.shape[1]
            candidate_best = float(member_gt_iou.flatten()[flat_index])
            if candidate_best >= args.oracle_min_iou:
                oracle_member = candidate_member
                oracle_best = candidate_best
                oracle[output_index, :4] = boxes[oracle_member]

        if len(members) > 1:
            weights = scores.clamp_min(0)
            weights = weights / weights.sum().clamp_min(1e-12)
            pairwise = box_iou(boxes, boxes)
            medoid_member = int((pairwise @ weights).argmax())
            medoid[output_index, :4] = boxes[medoid_member]
            box_vote[output_index, :4] = (boxes * weights[:, None]).sum(0)
        else:
            medoid_member = 0

        rows.append(
            {
                "image": image_name,
                "output_index": output_index,
                "class": names[class_index],
                "score": float(expanded[root, 4]),
                "members": len(members),
                "current_best_same_class_gt_iou": current_best,
                "oracle_best_same_class_gt_iou": oracle_best,
                "oracle_changed": oracle_member != 0,
                "oracle_member_index": oracle_member,
                "medoid_changed": medoid_member != 0,
                "medoid_member_index": medoid_member,
            }
        )

    variants = {"current": current, "oracle_rep": oracle, "medoid_rep": medoid, "score_box_vote": box_vote}
    for name, variant in variants.items():
        if len(variant) != len(current):
            raise RuntimeError(f"{name} changed output count")
        if len(variant):
            if not torch.equal(variant[:, 5].long(), current[:, 5].long()):
                raise RuntimeError(f"{name} changed classes")
            if not torch.equal(variant[:, 4], current[:, 4]):
                raise RuntimeError(f"{name} changed scores or order")
    return variants, rows


def main() -> None:
    args = parse_args()
    args.checkpoint = args.checkpoint.expanduser().resolve()
    args.data = args.data.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {args.output}")
    args.output.mkdir(parents=True)

    data_info = check_det_dataset(str(args.data))
    coco_path = Path(data_info["path"]) / "annotations" / "instances_val.json"
    coco_gt = COCO(str(coco_path))
    image_id_by_stem = {Path(image["file_name"]).stem: image_id for image_id, image in coco_gt.imgs.items()}
    category_id_by_name = {category["name"]: category_id for category_id, category in coco_gt.cats.items()}
    names = [data_info["names"][index] for index in range(len(data_info["names"]))]
    category_ids = {index: category_id_by_name[name] for index, name in enumerate(names)}

    device = select_device(args.device, verbose=False)
    _, loader = make_loader(args.data, args)
    wrapped = YOLO(str(args.checkpoint))
    net = wrapped.model.to(device).float().eval()
    head = net.model[-1]
    if not isinstance(head, Detect) or not head.end2end or head.nc != len(names):
        raise RuntimeError("Checkpoint is not a compatible four-class end-to-end detector")
    if isinstance(net.args, dict):
        net.args = SimpleNamespace(**net.args)

    checkpoint_before = sha256(args.checkpoint)
    prediction_rows: dict[str, list[dict[str, Any]]] = {
        "current": [],
        "oracle_rep": [],
        "medoid_rep": [],
        "score_box_vote": [],
    }
    group_rows: list[dict[str, Any]] = []
    reconstruction_rows: list[dict[str, Any]] = []
    seen_images = 0
    amp = device.type == "cuda"

    with torch.inference_mode():
        for batch in loader:
            images = batch["img"].to(device, non_blocking=True).float() / 255.0
            device_batch = {
                key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
                for key, value in batch.items()
            }
            device_batch["img"] = images
            with torch.autocast(device_type=device.type, enabled=amp):
                output = net(images)
            raw = output[1] if isinstance(output, tuple) else output
            decoded = head._inference(raw["one2many"]).float()
            height, width = images.shape[-2:]

            for image_index in range(images.shape[0]):
                absolute_index = seen_images + image_index
                if args.max_images and absolute_index >= args.max_images:
                    continue
                image_name = Path(batch["im_file"][image_index]).name
                image_id = image_id_by_stem[Path(image_name).stem]
                gt_boxes, gt_classes = paired_gt(device_batch, image_index, width, height)
                expanded = expanded_candidates(decoded[image_index], head.nc, args)
                roots, groups = greedy_groups(expanded, args.nms_iou)
                reconstructed = expanded[roots[: args.max_det]].clone()
                reference = current_nms(decoded[image_index], head.nc, args)
                check = verify_current(reconstructed, reference)
                check.update({"image": image_name, "expanded_candidates": len(expanded), "nms_roots": len(roots)})
                reconstruction_rows.append(check)
                variants, image_group_rows = representative_variants(
                    expanded,
                    roots,
                    groups,
                    gt_boxes,
                    gt_classes,
                    args,
                    image_name,
                    names,
                )
                group_rows.extend(image_group_rows)
                common = (
                    image_id,
                    category_ids,
                    (height, width),
                    batch["ori_shape"][image_index],
                    batch["ratio_pad"][image_index],
                )
                for name, variant in variants.items():
                    prediction_rows[name].extend(coco_rows(variant, *common))

            seen_images += images.shape[0]
            print(f"STAGE1_PROGRESS images={min(seen_images, len(loader.dataset))}/{len(loader.dataset)}", flush=True)
            if args.max_images and seen_images >= args.max_images:
                break

    metrics = []
    prediction_manifest = {}
    for name, rows in prediction_rows.items():
        prediction_path = args.output / f"predictions_{name}.json"
        prediction_path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
        prediction_manifest[name] = {"path": str(prediction_path), "rows": len(rows)}
        metrics.append(metric_row(name, coco_gt, rows, category_id_by_name))
    metric_by_name = {row["case"]: row for row in metrics}
    current_ap = metric_by_name["current"]["AP"]
    oracle_delta = metric_by_name["oracle_rep"]["AP"] - current_ap
    if oracle_delta < args.no_go_gate:
        decision = "NO_GO_REPRESENTATIVE_SELECTION"
        stage2_allowed = False
    elif oracle_delta < args.go_gate:
        decision = "BORDERLINE_DO_NOT_AUTO_ADVANCE"
        stage2_allowed = False
    else:
        decision = "GO_STAGE2_REGION_SIGNAL_AUDIT"
        stage2_allowed = True

    write_csv(args.output / "groups.csv", group_rows)
    write_csv(args.output / "nms_reconstruction.csv", reconstruction_rows)
    write_csv(args.output / "metrics.csv", metrics)
    checkpoint_after = sha256(args.checkpoint)
    summary = {
        "protocol": {
            "split": "Val only; Test untouched",
            "data": str(args.data),
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256_before": checkpoint_before,
            "checkpoint_sha256_after": checkpoint_after,
            "checkpoint_unchanged": checkpoint_before == checkpoint_after,
            "imgsz": args.imgsz,
            "score_floor": args.score_floor,
            "nms_iou": args.nms_iou,
            "max_det": args.max_det,
            "oracle_min_iou": args.oracle_min_iou,
            "invariant": "identical output count/classes/scores/order; coordinates only may change",
        },
        "nms_reconstruction": {
            "images": len(reconstruction_rows),
            "max_box_error": max((row["max_box_error"] for row in reconstruction_rows), default=0.0),
            "max_score_error": max((row["max_score_error"] for row in reconstruction_rows), default=0.0),
            "all_class_equal": all(row["class_equal"] for row in reconstruction_rows),
        },
        "group_statistics": {
            "groups": len(group_rows),
            "groups_with_multiple_members": sum(row["members"] > 1 for row in group_rows),
            "oracle_changed_groups": sum(row["oracle_changed"] for row in group_rows),
            "medoid_changed_groups": sum(row["medoid_changed"] for row in group_rows),
        },
        "metrics": metrics,
        "predictions": prediction_manifest,
        "gate": {
            "current_AP": current_ap,
            "oracle_representative_AP": metric_by_name["oracle_rep"]["AP"],
            "oracle_delta_AP": oracle_delta,
            "no_go_below": args.no_go_gate,
            "go_at_least": args.go_gate,
            "decision": decision,
            "stage2_allowed": stage2_allowed,
        },
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary["gate"], indent=2), flush=True)


if __name__ == "__main__":
    main()
