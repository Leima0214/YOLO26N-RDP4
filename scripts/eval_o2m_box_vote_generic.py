#!/usr/bin/env python3
"""Class- and split-generic O2M + Box-Voting COCO evaluator for end-to-end YOLO26."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

def protocol_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(score_floor=args.score_floor, nms_iou=args.nms_iou, max_det=args.max_det,
                           max_nms=args.max_nms, assignment_min_iou=0.10, ambiguous_gt_iou=0.50)


def make_loader(data: dict[str, Any], split: str, args: argparse.Namespace):
    from ultralytics.cfg import get_cfg
    from ultralytics.data.build import build_dataloader, build_yolo_dataset
    from ultralytics.utils import DEFAULT_CFG

    cfg = get_cfg(DEFAULT_CFG, {"mode": "val", "task": "detect", "imgsz": args.imgsz,
                                "batch": args.batch, "workers": args.workers, "rect": True, "cache": False})
    dataset = build_yolo_dataset(cfg, data[split], args.batch, data, mode="val", rect=True, stride=32)
    return build_dataloader(dataset, args.batch, args.workers, shuffle=False, rank=-1, pin_memory=False)


def summarize(evaluator: COCOeval) -> dict[str, float]:
    def ap(area: str = "all", iou: float | None = None) -> float:
        area_index = list(evaluator.params.areaRngLbl).index(area)
        values = evaluator.eval["precision"]
        if iou is not None:
            index = int(np.argmin(np.abs(evaluator.params.iouThrs - iou)))
            values = values[index:index + 1]
        values = values[:, :, :, area_index, -1]
        values = values[values > -1]
        return float(values.mean()) if values.size else float("nan")

    def ar(area: str = "all") -> float:
        area_index = list(evaluator.params.areaRngLbl).index(area)
        values = evaluator.eval["recall"][:, :, area_index, -1]
        values = values[values > -1]
        return float(values.mean()) if values.size else float("nan")
    return {"AP": ap(), "AP50": ap(iou=0.50), "AP75": ap(iou=0.75),
            "AP_small": ap("small"), "AP_medium": ap("medium"), "AP_large": ap("large"), "AR100": ar()}


def coco_metrics(gt: COCO, predictions: list[dict[str, Any]], names_by_category: dict[int, str]) -> dict[str, Any]:
    def evaluate(rows: list[dict[str, Any]], category_ids: list[int]) -> dict[str, float]:
        if rows:
            detections = gt.loadRes(rows)
        else:
            detections = COCO()
            detections.dataset = {"images": list(gt.dataset["images"]),
                                  "categories": list(gt.dataset["categories"]), "annotations": []}
            detections.createIndex()
        evaluator = COCOeval(gt, detections, "bbox")
        evaluator.params.imgIds = sorted(gt.imgs)
        evaluator.params.catIds = category_ids
        evaluator.params.maxDets = [100]
        evaluator.evaluate()
        evaluator.accumulate()
        return summarize(evaluator)
    overall = evaluate(predictions, sorted(names_by_category))
    per_class = {}
    for category_id, name in names_by_category.items():
        rows = [row for row in predictions if row["category_id"] == category_id]
        per_class[name] = evaluate(rows, [category_id])
    return {"overall": overall, "per_class": per_class}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--score-floor", type=float, default=0.001)
    parser.add_argument("--nms-iou", type=float, default=0.70)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--max-nms", type=int, default=30000)
    parser.add_argument("--max-images", type=int, default=0)
    args = parser.parse_args()
    from audit_gbrg_o2m_csrg_candidates import paired_gt
    from audit_gbrg_o2m_strict_reconstruction import (
        build_variants, canonical_coco_rows, current_nms, expanded_candidates, raw_level_indices,
    )
    from audit_gbrg_o2m_region_signal import groups_anchored_to_reference
    from ultralytics.data.utils import check_det_dataset
    from ultralytics.nn.modules.head import Detect
    from ultralytics.utils.torch_utils import select_device

    for key in ("checkpoint", "data", "output"):
        setattr(args, key, getattr(args, key).expanduser().resolve())
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {args.output}")
    data = check_det_dataset(str(args.data))
    names = [data["names"][index] for index in range(len(data["names"]))]
    coco_path = Path(data["path"]) / "annotations" / f"instances_{args.split}.json"
    if not coco_path.is_file():
        raise FileNotFoundError(f"Missing {coco_path}; run scripts/prepare_svrdd7.py first")
    coco_gt = COCO(str(coco_path))
    category_by_name = {category["name"]: category_id for category_id, category in coco_gt.cats.items()}
    missing = [name for name in names if name not in category_by_name]
    if missing:
        raise ValueError(f"COCO categories do not match data YAML; missing {missing}")
    category_ids = {index: category_by_name[name] for index, name in enumerate(names)}
    image_id_by_stem: dict[str, int] = {}
    for image_id, image in coco_gt.imgs.items():
        stem = Path(image["file_name"]).stem
        if stem in image_id_by_stem:
            raise ValueError(f"COCO annotation has duplicate image stem: {stem}")
        image_id_by_stem[stem] = image_id

    from ultralytics import YOLO
    net = YOLO(str(args.checkpoint)).model
    device = select_device(args.device, verbose=False)
    net = net.to(device).float().eval()
    head = net.model[-1]
    if not isinstance(head, Detect) or not head.end2end or int(head.nc) != len(names):
        raise RuntimeError(f"Incompatible checkpoint head={type(head).__name__} end2end={getattr(head, 'end2end', None)} "
                           f"nc={getattr(head, 'nc', None)} expected_nc={len(names)}")
    if isinstance(net.args, dict):
        net.args = SimpleNamespace(**net.args)
    loader = make_loader(data, args.split, args)
    rows: dict[str, list[dict[str, Any]]] = {"current": [], "score_box_vote": []}
    seen = 0
    p_args = protocol_args(args)
    with torch.inference_mode():
        for batch in loader:
            images = batch["img"].to(device, non_blocking=True).float() / 255.0
            device_batch = {key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
                            for key, value in batch.items()}
            device_batch["img"] = images
            output = net(images)
            raw = output[1] if isinstance(output, tuple) else output
            one2many = raw["one2many"]
            decoded = head._inference(one2many).float()
            level_by_raw = raw_level_indices(one2many["feats"], device)
            model_shape = tuple(images.shape[-2:])
            for image_index in range(images.shape[0]):
                if args.max_images and seen >= args.max_images:
                    break
                image_name = Path(batch["im_file"][image_index]).name
                stem = Path(image_name).stem
                if stem not in image_id_by_stem:
                    raise KeyError(f"Image {image_name} absent from {coco_path}")
                gt_boxes, gt_classes = paired_gt(device_batch, image_index, model_shape[1], model_shape[0])
                expanded = expanded_candidates(decoded[image_index], level_by_raw, head.nc, p_args)
                reference = current_nms(decoded[image_index], head.nc, p_args)
                roots, groups, _ = groups_anchored_to_reference(expanded[:, :6], reference, p_args.nms_iou)
                current = expanded[roots, :6].clone()
                variants, _ = build_variants("eval", image_name, current, expanded, groups, gt_boxes, gt_classes,
                                             model_shape, batch["ori_shape"][image_index], batch["ratio_pad"][image_index],
                                             names, p_args)
                for case in rows:
                    rows[case].extend(canonical_coco_rows(variants[case], image_id_by_stem[stem], category_ids,
                                                          model_shape, batch["ori_shape"][image_index],
                                                          batch["ratio_pad"][image_index]))
                seen += 1
                if seen % 100 == 0:
                    print(f"EVAL_PROGRESS split={args.split} images={seen}", flush=True)
            if args.max_images and seen >= args.max_images:
                break
    names_by_category = {category_by_name[name]: name for name in names}
    metrics = {case: coco_metrics(coco_gt, case_rows, names_by_category) for case, case_rows in rows.items()}
    args.output.mkdir(parents=True)
    payload = {"checkpoint": str(args.checkpoint), "data": str(args.data), "split": args.split,
               "images": seen, "classes": names, "protocol": vars(p_args), "metrics": metrics}
    (args.output / "metrics.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("EVAL " + json.dumps(metrics, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
