"""Training-free R1/GBRG box-score cross-swap + score-localization audit.

Paper O2M protocol: batch=32, rect=True, conf=0.001, NMS IoU=0.7, max_det=300,
COCO maxDets=100, FP32, Val only, Test sealed.  No training, no parameter edits.

Four counterfactual groups built BEFORE NMS at identical anchors:
  A: R1 box    + R1 score   (reproduces R1-O2M)
  B: GBRG box  + GBRG score (reproduces GBRG-O2M)
  C: GBRG box  + R1 score   (score-preservation counterfactual)
  D: R1 box    + GBRG score (box-preservation counterfactual)

Also saves the full pre-NMS cache (boxes/scores/level/image/anchor order) and a
per-anchor score-localization table (IoU bands / class / level / size / winner).
"""
from __future__ import annotations

import argparse
import csv
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

from audit_gbrg_o2m_csrg_candidates import make_loader, paired_gt  # noqa: E402
from audit_gbrg_o2m_strict_reconstruction import (  # noqa: E402
    canonical_coco_rows,
    current_nms,
    expanded_candidates,
    raw_level_indices,
)
from audit_gbrg_o2m_region_signal import groups_anchored_to_reference  # noqa: E402
from ultralytics import YOLO  # noqa: E402
from ultralytics.data.utils import check_det_dataset  # noqa: E402
from ultralytics.nn.modules.head import Detect  # noqa: E402
from ultralytics.utils.torch_utils import select_device  # noqa: E402

NAMES = ("D00", "D10", "D20", "D40")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r1", type=Path, required=True)
    parser.add_argument("--gbrg", type=Path, required=True)
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
    parser.add_argument("--max-images", type=int, default=0)
    return parser.parse_args()


def flat_ratio_pad(ratio_pad: Any) -> list[list[float]]:
    out = []
    for part in ratio_pad:
        if torch.is_tensor(part):
            out.append([float(v) for v in part.flatten().tolist()])
        else:
            out.append([float(v) for v in part])
    return out


def rebuild_ratio_pad(stored: list[list[float]]) -> Any:
    return tuple(tuple(part) for part in stored)


def coco_metrics(gt: COCO, predictions: list[dict[str, Any]]) -> dict[str, float]:
    detections = gt.loadRes(predictions)
    evaluator = COCOeval(gt, detections, "bbox")
    evaluator.params.imgIds = sorted(gt.imgs)
    evaluator.params.catIds = sorted(gt.cats)
    evaluator.params.maxDets = [100]
    evaluator.evaluate()
    evaluator.accumulate()

    def ap_at(area: str = "all", iou: float | None = None) -> float:
        area_index = list(evaluator.params.areaRngLbl).index(area)
        precision = evaluator.eval["precision"]
        if iou is not None:
            iou_index = int(np.argmin(np.abs(evaluator.params.iouThrs - iou)))
            precision = precision[iou_index : iou_index + 1]
        values = precision[:, :, :, area_index, -1]
        values = values[values > -1]
        return float(values.mean()) if values.size else float("nan")

    def ar_at(area: str = "all") -> float:
        area_index = list(evaluator.params.areaRngLbl).index(area)
        values = evaluator.eval["recall"][:, :, area_index, -1]
        values = values[values > -1]
        return float(values.mean()) if values.size else float("nan")

    out = {
        "AP": ap_at(), "AP50": ap_at(iou=0.50), "AP75": ap_at(iou=0.75),
        "AP_small": ap_at("small"), "AP_medium": ap_at("medium"), "AP_large": ap_at("large"),
        "AR100": ar_at(),
    }
    for class_id, name in {1: "D00", 2: "D10", 3: "D20", 4: "D40"}.items():
        subset = [p for p in predictions if p["category_id"] == class_id]
        if not subset:
            out[f"{name}_AP"] = out[f"{name}_AP50"] = out[f"{name}_AP75"] = float("nan")
            out[f"{name}_AP75_small"] = out[f"{name}_AP75_medium"] = out[f"{name}_AR100"] = float("nan")
            continue
        class_eval = COCOeval(gt, gt.loadRes(subset), "bbox")
        class_eval.params.imgIds = sorted(gt.imgs)
        class_eval.params.catIds = [class_id]
        class_eval.params.maxDets = [100]
        class_eval.evaluate()
        class_eval.accumulate()

        def class_ap(area: str = "all", iou: float | None = None) -> float:
            area_index = list(class_eval.params.areaRngLbl).index(area)
            precision = class_eval.eval["precision"]
            if iou is not None:
                iou_index = int(np.argmin(np.abs(class_eval.params.iouThrs - iou)))
                precision = precision[iou_index : iou_index + 1]
            values = precision[:, :, :, area_index, -1]
            values = values[values > -1]
            return float(values.mean()) if values.size else float("nan")

        def class_ar(area: str = "all") -> float:
            area_index = list(class_eval.params.areaRngLbl).index(area)
            values = class_eval.eval["recall"][:, :, area_index, -1]
            values = values[values > -1]
            return float(values.mean()) if values.size else float("nan")

        out[f"{name}_AP"] = class_ap()
        out[f"{name}_AP50"] = class_ap(iou=0.50)
        out[f"{name}_AP75"] = class_ap(iou=0.75)
        out[f"{name}_AP75_small"] = class_ap("small", iou=0.75)
        out[f"{name}_AP75_medium"] = class_ap("medium", iou=0.75)
        out[f"{name}_AR100"] = class_ar()
    return out


def iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between two xyxy arrays."""
    a_x1, a_y1, a_x2, a_y2 = boxes_a[:, 0], boxes_a[:, 1], boxes_a[:, 2], boxes_a[:, 3]
    b_x1, b_y1, b_x2, b_y2 = boxes_b[:, 0], boxes_b[:, 1], boxes_b[:, 2], boxes_b[:, 3]
    inter_w = np.maximum(0, np.minimum(a_x2[:, None], b_x2) - np.maximum(a_x1[:, None], b_x1))
    inter_h = np.maximum(0, np.minimum(a_y2[:, None], b_y2) - np.maximum(a_y1[:, None], b_y1))
    inter = inter_w * inter_h
    area_a = (a_x2 - a_x1) * (a_y2 - a_y1)
    area_b = (b_x2 - b_x1) * (b_y2 - b_y1)
    return inter / np.maximum(area_a[:, None] + area_b - inter, 1e-9)


def size_of(box: np.ndarray, shape: tuple[int, int]) -> str:
    area = float((box[2] - box[0]) * (box[3] - box[1]))
    if area < 32 * 32:
        return "small"
    if area < 96 * 96:
        return "medium"
    return "large"


def main() -> None:
    args = parse_args()
    for name in ("r1", "gbrg", "data", "output"):
        value = getattr(args, name)
        setattr(args, name, value.expanduser().resolve())
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

    # ---- pass 1: pre-NMS cache for both models ----
    cache: dict[str, dict[str, Any]] = {}
    for model_name, checkpoint in {"R1": args.r1, "GBRG": args.gbrg}.items():
        _, loader = make_loader(args.data, args)
        wrapped = YOLO(str(checkpoint))
        net = wrapped.model.to(device).float().eval()
        head = net.model[-1]
        if not isinstance(head, Detect) or not head.end2end or head.nc != len(names):
            raise RuntimeError(f"{model_name} is not a compatible four-class end-to-end checkpoint")
        if isinstance(net.args, dict):
            net.args = SimpleNamespace(**net.args)
        boxes_list, scores_list, level_list, image_list = [], [], [], []
        meta: list[dict[str, Any]] = []
        with torch.inference_mode():
            for batch in loader:
                images = batch["img"].to(device, non_blocking=True).float() / 255.0
                output = net(images)
                raw = output[1] if isinstance(output, tuple) else output
                one2many = raw["one2many"]
                decoded = head._inference(one2many).float()
                level_by_raw = raw_level_indices(one2many["feats"], device)
                model_shape = tuple(images.shape[-2:])
                device_batch = {
                    key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
                    for key, value in batch.items()
                }
                device_batch["img"] = images
                for image_index in range(images.shape[0]):
                    if args.max_images and len(meta) >= args.max_images:
                        break
                    image_name = Path(batch["im_file"][image_index]).name
                    image_id = image_id_by_stem[Path(image_name).stem]
                    dec = decoded[image_index].cpu()
                    gt_boxes, gt_classes = paired_gt(
                        device_batch, image_index, model_shape[1], model_shape[0]
                    )
                    meta.append({
                        "image": image_name, "image_id": image_id,
                        "model_shape": [int(model_shape[0]), int(model_shape[1])],
                        "ori_shape": [int(v) for v in batch["ori_shape"][image_index]],
                        "ratio_pad": flat_ratio_pad(batch["ratio_pad"][image_index]),
                        "gt_boxes": [[float(v) for v in box.tolist()] for box in gt_boxes],
                        "gt_classes": [int(v) for v in gt_classes.tolist()],
                        "n_anchors": int(dec.shape[1]),
                    })
                    boxes_list.append(dec[:4].T.numpy())
                    scores_list.append(dec[4:].T.numpy())
                    level_list.append(level_by_raw.cpu().numpy())
                    image_list.append(np.full(dec.shape[1], image_id, dtype=np.int64))
                print(f"CROSSSWAP_CACHE model={model_name} images={len(meta)}", flush=True)
                if args.max_images and len(meta) >= args.max_images:
                    break
        cache[model_name] = {
            "boxes": np.concatenate(boxes_list), "scores": np.concatenate(scores_list),
            "level": np.concatenate(level_list), "image": np.concatenate(image_list), "meta": meta,
        }
        cache_dir = args.output / "pre_nms_cache"
        cache_dir.mkdir(exist_ok=True)
        np.savez_compressed(
            cache_dir / f"{model_name}.npz",
            boxes=cache[model_name]["boxes"], scores=cache[model_name]["scores"],
            level=cache[model_name]["level"], image=cache[model_name]["image"],
        )
        (cache_dir / f"{model_name}.meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
        del wrapped, net
        torch.cuda.empty_cache()

    assert [m["image"] for m in cache["R1"]["meta"]] == [m["image"] for m in cache["GBRG"]["meta"]]
    r1_meta, gbrg_meta = cache["R1"]["meta"], cache["GBRG"]["meta"]
    offsets = np.cumsum([0] + [m["n_anchors"] for m in r1_meta])
    g_offsets = np.cumsum([0] + [m["n_anchors"] for m in gbrg_meta])

    cases = {
        "A_R1box_R1score": ("R1", "R1"),
        "B_GBRGbox_GBRGscore": ("GBRG", "GBRG"),
        "C_GBRGbox_R1score": ("GBRG", "R1"),
        "D_R1box_GBRGscore": ("R1", "GBRG"),
    }
    predictions: dict[str, list[dict[str, Any]]] = {case: [] for case in cases}
    winner_anchors: dict[str, dict[int, list[int]]] = {case: {} for case in cases}
    winner_level_fraction: dict[str, dict[str, float]] = {case: {"P3": 0, "P4": 0, "P5": 0} for case in cases}
    level_names = ("P3", "P4", "P5")

    for meta_index, meta_row in enumerate(r1_meta):
        image_id = meta_row["image_id"]
        r1_slice = slice(int(offsets[meta_index]), int(offsets[meta_index + 1]))
        g_slice = slice(int(g_offsets[meta_index]), int(g_offsets[meta_index + 1]))
        boxes_r = torch.from_numpy(cache["R1"]["boxes"][r1_slice]).float().to(device)
        scores_r = torch.from_numpy(cache["R1"]["scores"][r1_slice]).float().to(device)
        boxes_g = torch.from_numpy(cache["GBRG"]["boxes"][g_slice]).float().to(device)
        scores_g = torch.from_numpy(cache["GBRG"]["scores"][g_slice]).float().to(device)
        level = torch.from_numpy(cache["R1"]["level"][r1_slice]).long().to(device)
        model_shape = (meta_row["model_shape"][0], meta_row["model_shape"][1])
        ori_shape = (meta_row["ori_shape"][0], meta_row["ori_shape"][1])
        ratio_pad = rebuild_ratio_pad(meta_row["ratio_pad"])

        originals = {"R1": (boxes_r, scores_r), "GBRG": (boxes_g, scores_g)}
        for case, (box_name, score_name) in cases.items():
            boxes, scores = originals[box_name][0], originals[score_name][1]
            decoded = torch.cat((boxes.T.contiguous(), scores.T.contiguous()), dim=0)
            expanded = expanded_candidates(decoded, level, len(names), args)
            reference = current_nms(decoded, len(names), args)
            roots, _, _ = groups_anchored_to_reference(expanded[:, :6], reference, args.nms_iou)
            anchors = [int(expanded[root, 6]) for root in roots]
            winner_anchors[case][image_id] = anchors
            for root in roots:
                lvl = int(expanded[root, 7])
                winner_level_fraction[case][level_names[lvl]] += 1
            rows = canonical_coco_rows(reference, image_id, category_ids, model_shape, ori_shape, ratio_pad)
            predictions[case].extend(rows)
        print(f"CROSSSWAP images={meta_index + 1}/{len(r1_meta)}", flush=True)

    for case in cases:
        total = sum(winner_level_fraction[case].values())
        winner_level_fraction[case] = {k: (v / total if total else 0.0) for k, v in winner_level_fraction[case].items()}

    metrics = {}
    for case in cases:
        metrics[case] = coco_metrics(coco_gt, predictions[case])
        metrics[case]["case"] = case
    (args.output / "cross_swap_metrics.json").write_text(
        json.dumps({"cases": cases, "metrics": metrics, "winner_level_fraction": winner_level_fraction},
                   indent=2, ensure_ascii=False) + "\n"
    )
    print("CROSSSWAP_METRICS " + json.dumps(metrics, indent=2), flush=True)

    # ---- score localization (vectorized per image) ----
    loc_rows: list[dict[str, Any]] = []
    for meta_index, meta_row in enumerate(r1_meta):
        r1_slice = slice(int(offsets[meta_index]), int(offsets[meta_index + 1]))
        g_slice = slice(int(g_offsets[meta_index]), int(g_offsets[meta_index + 1]))
        boxes_r = cache["R1"]["boxes"][r1_slice]
        scores_r = cache["R1"]["scores"][r1_slice]
        boxes_g = cache["GBRG"]["boxes"][g_slice]
        scores_g = cache["GBRG"]["scores"][g_slice]
        level = cache["R1"]["level"][r1_slice]
        gt_boxes = np.asarray(meta_row["gt_boxes"], dtype=np.float64)
        gt_classes = np.asarray(meta_row["gt_classes"], dtype=np.int64)
        if not len(gt_boxes):
            continue
        max_r = scores_r.max(1)
        max_g = scores_g.max(1)
        mask = (max_r >= args.score_floor) | (max_g >= args.score_floor)
        idx = np.where(mask)[0]
        if not len(idx):
            continue
        sr, sg = max_r[idx], max_g[idx]
        cr, cg = scores_r[idx].argmax(1), scores_g[idx].argmax(1)
        use_class = np.where(sr >= sg, cr, cg)
        iou_r = np.zeros(len(idx)); iou_g = np.zeros(len(idx))
        best_size = np.full(len(idx), "bg", dtype=object)
        for class_id in range(len(names)):
            cls_mask = use_class == class_id
            if not cls_mask.any():
                continue
            gt_sel = gt_boxes[gt_classes == class_id]
            if not len(gt_sel):
                continue
            ious_r = iou_matrix(boxes_r[idx[cls_mask]], gt_sel)
            ious_g = iou_matrix(boxes_g[idx[cls_mask]], gt_sel)
            best_r = ious_r.max(1); best_g = ious_g.max(1)
            iou_r[cls_mask] = best_r
            iou_g[cls_mask] = best_g
            best_gt = gt_sel[ious_r.argmax(1)]
            scale = (meta_row["ori_shape"][0] * meta_row["ori_shape"][1]) / (
                meta_row["model_shape"][0] * meta_row["model_shape"][1]
            )
            for k, pos in enumerate(np.where(cls_mask)[0]):
                box = best_gt[k]
                area_orig = float((box[2] - box[0]) * (box[3] - box[1]) * scale)
                best_size[pos] = "small" if area_orig < 32 * 32 else "medium" if area_orig < 96 * 96 else "large"
        for k in range(len(idx)):
            loc_rows.append({
                "image_id": int(meta_row["image_id"]), "anchor": int(idx[k]),
                "level": int(level[idx[k]]), "class": NAMES[int(use_class[k])],
                "size": str(best_size[k]),
                "score_r": float(sr[k]), "score_g": float(sg[k]),
                "iou_r": float(iou_r[k]), "iou_g": float(iou_g[k]),
                "class_r": int(cr[k]), "class_g": int(cg[k]),
            })
    with open(args.output / "score_localization.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(loc_rows[0].keys()))
        writer.writeheader()
        writer.writerows(loc_rows)

    # ---- winner-change statistics ----
    winner_change_rows = []
    for image_id in winner_anchors["A_R1box_R1score"]:
        a = set(winner_anchors["A_R1box_R1score"][image_id])
        b = set(winner_anchors["B_GBRGbox_GBRGscore"][image_id])
        c = set(winner_anchors["C_GBRGbox_R1score"][image_id])
        winner_change_rows.append({
            "image_id": image_id, "n_A": len(a), "n_B": len(b), "n_C": len(c),
            "B_not_in_A": len(b - a), "A_not_in_B": len(a - b),
            "C_not_in_B": len(c - b), "B_not_in_C": len(b - c),
        })
    with open(args.output / "winner_change.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(winner_change_rows[0].keys()))
        writer.writeheader()
        writer.writerows(winner_change_rows)

    summary = {
        "metrics": metrics,
        "winner_level_fraction": winner_level_fraction,
        "winner_change": {
            "B_not_in_A": sum(r["B_not_in_A"] for r in winner_change_rows),
            "A_not_in_B": sum(r["A_not_in_B"] for r in winner_change_rows),
            "C_not_in_B": sum(r["C_not_in_B"] for r in winner_change_rows),
            "B_not_in_C": sum(r["B_not_in_C"] for r in winner_change_rows),
            "n_A_total": sum(r["n_A"] for r in winner_change_rows),
            "n_B_total": sum(r["n_B"] for r in winner_change_rows),
        },
        "localization_rows": len(loc_rows),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print("SUMMARY " + json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
