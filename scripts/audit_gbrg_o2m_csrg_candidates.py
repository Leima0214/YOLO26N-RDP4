"""Val-only O2M candidate, NMS-survival, and fixed-box oracle audit for CS-GBRG."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import importlib.machinery
import io
import json
import sys
import types
from collections import Counter, defaultdict
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

_pandas_placeholder = types.ModuleType("pandas")
_pandas_placeholder.__spec__ = importlib.machinery.ModuleSpec("pandas", loader=None)
sys.modules.setdefault("pandas", _pandas_placeholder)

from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.data.build import build_dataloader, build_yolo_dataset
from ultralytics.data.utils import check_det_dataset
from ultralytics.nn.modules.head import Detect
from ultralytics.utils import DEFAULT_CFG
from ultralytics.utils.metrics import box_iou
from ultralytics.utils.nms import non_max_suppression
from ultralytics.utils.ops import scale_boxes, xywh2xyxy, xyxy2xywh
from ultralytics.utils.torch_utils import select_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", required=True, help="NAME=CHECKPOINT; use R1 and GBRG")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--score-floor", type=float, default=0.001)
    parser.add_argument("--nms-iou", type=float, default=0.70)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--oracle-delta-gate", type=float, default=0.003)
    parser.add_argument("--expected-gbrg-ap", type=float, default=0.25986853598949455)
    parser.add_argument("--reproduction-tolerance", type=float, default=0.0005)
    parser.add_argument("--max-images", type=int, default=0, help="0 audits the complete Val split")
    return parser.parse_args()


def parse_models(values: list[str]) -> dict[str, Path]:
    models: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected NAME=CHECKPOINT, got {value!r}")
        name, raw = value.split("=", 1)
        path = Path(raw).expanduser().resolve()
        if not name or name in models or not path.is_file():
            raise ValueError(f"Invalid model entry {value!r}")
        models[name] = path
    if "GBRG" not in models or "R1" not in models:
        raise ValueError("The audit requires R1 and GBRG model entries")
    return models


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


def coco_eval(coco_gt: COCO, predictions: list[dict[str, Any]], category_ids: list[int]) -> COCOeval:
    coco_dt = coco_gt.loadRes(predictions)
    evaluator = COCOeval(coco_gt, coco_dt, "bbox")
    evaluator.params.imgIds = sorted(coco_gt.imgs)
    evaluator.params.catIds = category_ids
    with contextlib.redirect_stdout(io.StringIO()):
        evaluator.evaluate()
        evaluator.accumulate()
    return evaluator


def mean_valid(values: np.ndarray) -> float | None:
    values = values[values > -1]
    return None if values.size == 0 else float(values.mean())


def ap(evaluator: COCOeval, area: str = "all", iou: float | None = None) -> float | None:
    area_index = list(evaluator.params.areaRngLbl).index(area)
    precision = evaluator.eval["precision"]
    if iou is not None:
        index = int(np.argmin(np.abs(evaluator.params.iouThrs - iou)))
        precision = precision[index : index + 1]
    return mean_valid(precision[:, :, :, area_index, -1])


def ar100(evaluator: COCOeval, area: str = "all") -> float | None:
    area_index = list(evaluator.params.areaRngLbl).index(area)
    return mean_valid(evaluator.eval["recall"][:, :, area_index, -1])


def metric_row(case: str, coco_gt: COCO, predictions: list[dict[str, Any]], category_ids: dict[str, int]) -> dict[str, Any]:
    all_eval = coco_eval(coco_gt, predictions, list(category_ids.values()))
    d00_eval = coco_eval(coco_gt, predictions, [category_ids["D00"]])
    d40_eval = coco_eval(coco_gt, predictions, [category_ids["D40"]])
    return {
        "case": case,
        "AP": ap(all_eval),
        "AP50": ap(all_eval, iou=0.50),
        "AP75": ap(all_eval, iou=0.75),
        "AP_small": ap(all_eval, "small"),
        "AP_medium": ap(all_eval, "medium"),
        "AP_large": ap(all_eval, "large"),
        "AR100": ar100(all_eval),
        "D00_AP": ap(d00_eval),
        "D00_AP50": ap(d00_eval, iou=0.50),
        "D00_AP75": ap(d00_eval, iou=0.75),
        "D00_AR100": ar100(d00_eval),
        "D40_AP": ap(d40_eval),
        "D40_AP50": ap(d40_eval, iou=0.50),
        "D40_AP75": ap(d40_eval, iou=0.75),
        "D40_AR100": ar100(d40_eval),
    }


def paired_gt(batch: dict[str, Any], image_index: int, width: int, height: int) -> tuple[torch.Tensor, torch.Tensor]:
    mask = batch["batch_idx"].view(-1).long() == image_index
    classes = batch["cls"].view(-1).long()[mask]
    boxes = xywh2xyxy(batch["bboxes"].float()[mask])
    boxes *= torch.tensor((width, height, width, height), device=boxes.device)
    return boxes, classes


def final_o2m(decoded: torch.Tensor, nc: int, args: argparse.Namespace) -> list[torch.Tensor]:
    nms_input = torch.cat((xyxy2xywh(decoded[:, :4].transpose(1, 2)).transpose(1, 2), decoded[:, 4:]), dim=1)
    return non_max_suppression(
        nms_input,
        conf_thres=args.score_floor,
        iou_thres=args.nms_iou,
        nc=nc,
        multi_label=True,
        max_det=args.max_det,
    )


def oracle_score_matrix(raw_boxes: torch.Tensor, gt_boxes: torch.Tensor, gt_classes: torch.Tensor, nc: int) -> torch.Tensor:
    scores = torch.zeros((len(raw_boxes), nc), device=raw_boxes.device, dtype=torch.float32)
    for class_index in range(nc):
        same = gt_boxes[gt_classes == class_index]
        if len(same):
            scores[:, class_index] = box_iou(raw_boxes.float(), same.float()).amax(1)
    return scores


def oracle_nms(raw_boxes: torch.Tensor, oracle_scores: torch.Tensor, nc: int, args: argparse.Namespace) -> torch.Tensor:
    decoded = torch.cat((raw_boxes.T.unsqueeze(0), oracle_scores.T.unsqueeze(0)), dim=1)
    nms_input = torch.cat((xyxy2xywh(decoded[:, :4].transpose(1, 2)).transpose(1, 2), decoded[:, 4:]), dim=1)
    return non_max_suppression(
        nms_input,
        conf_thres=1e-6,
        iou_thres=args.nms_iou,
        nc=nc,
        multi_label=True,
        max_det=args.max_det,
    )[0]


def coco_rows(
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
    rows = []
    for prediction, box in zip(predictions, scaled):
        rows.append(
            {
                "image_id": image_id,
                "category_id": category_ids[int(prediction[5])],
                "bbox": [float(box[0]), float(box[1]), float(box[2] - box[0]), float(box[3] - box[1])],
                "score": float(prediction[4]),
            }
        )
    return rows


def size_bucket(box: torch.Tensor) -> str:
    area = float((box[2] - box[0]).clamp_min(0) * (box[3] - box[1]).clamp_min(0))
    return "small" if area < 32**2 else "medium" if area < 96**2 else "large"


def candidate_rows_for_image(
    model_name: str,
    image_name: str,
    raw_boxes: torch.Tensor,
    raw_scores: torch.Tensor,
    final: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_classes: torch.Tensor,
    names: list[str],
    score_floor: float,
    nms_iou: float,
) -> list[dict[str, Any]]:
    rows = []
    for gt_index, (gt_box, gt_class) in enumerate(zip(gt_boxes, gt_classes)):
        class_index = int(gt_class)
        class_scores = raw_scores[:, class_index]
        eligible = class_scores >= score_floor
        eligible_indices = torch.where(eligible)[0]
        if len(eligible_indices):
            eligible_boxes = raw_boxes[eligible_indices]
            ious = box_iou(eligible_boxes.float(), gt_box[None].float())[:, 0]
            best_local = int(ious.argmax())
            best_source = int(eligible_indices[best_local])
            best_iou = float(ious[best_local])
            best_score = float(class_scores[best_source])
            rank = int((class_scores[eligible] > best_score).sum()) + 1
            count50 = int((ious >= 0.50).sum())
            count75 = int((ious >= 0.75).sum())
            same_final = final[final[:, 5].long() == class_index] if len(final) else final
            if len(same_final):
                final_ious = box_iou(same_final[:, :4].float(), gt_box[None].float())[:, 0]
                post_best_iou = float(final_ious.max())
                raw_to_final = box_iou(raw_boxes[best_source : best_source + 1].float(), same_final[:, :4].float())[0]
                survives = bool((raw_to_final >= 0.9999).any())
                if survives:
                    reason = "survived"
                elif bool((raw_to_final >= nms_iou).any()):
                    suppressor = same_final[int(raw_to_final.argmax())]
                    suppressor_gt_iou = float(box_iou(suppressor[None, :4].float(), gt_box[None].float())[0, 0])
                    reason = "same_gt_nms" if suppressor_gt_iou >= 0.50 else "neighbor_or_background_nms"
                else:
                    reason = "topk_or_other_filter"
            else:
                post_best_iou = 0.0
                survives = False
                reason = "no_same_class_post_nms"
        else:
            best_source = -1
            best_iou = best_score = post_best_iou = 0.0
            rank = count50 = count75 = 0
            survives = False
            reason = "below_score_floor"
        rows.append(
            {
                "model": model_name,
                "image": image_name,
                "gt_index": gt_index,
                "class": names[class_index],
                "size": size_bucket(gt_box),
                "raw_best_source": best_source,
                "raw_best_iou": best_iou,
                "raw_best_score": best_score,
                "raw_best_score_rank": rank,
                "raw_candidates_iou50": count50,
                "raw_candidates_iou75": count75,
                "post_nms_best_iou": post_best_iou,
                "raw_best_survived": survives,
                "filter_reason": reason,
            }
        )
    return rows


def summarize_candidates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["model"], row["class"], row["size"])].append(row)
        grouped[(row["model"], row["class"], "all")].append(row)
        grouped[(row["model"], "all", "all")].append(row)
    output = []
    for (model, class_name, size), items in sorted(grouped.items()):
        reasons = Counter(item["filter_reason"] for item in items)
        output.append(
            {
                "model": model,
                "class": class_name,
                "size": size,
                "gt": len(items),
                "raw_recall50": float(np.mean([item["raw_best_iou"] >= 0.50 for item in items])),
                "raw_recall75": float(np.mean([item["raw_best_iou"] >= 0.75 for item in items])),
                "post_nms_recall50": float(np.mean([item["post_nms_best_iou"] >= 0.50 for item in items])),
                "post_nms_recall75": float(np.mean([item["post_nms_best_iou"] >= 0.75 for item in items])),
                "raw_best_iou_mean": float(np.mean([item["raw_best_iou"] for item in items])),
                "post_nms_best_iou_mean": float(np.mean([item["post_nms_best_iou"] for item in items])),
                "raw_best_survival_fraction": float(np.mean([item["raw_best_survived"] for item in items])),
                "raw_best_score_rank_median": float(np.median([item["raw_best_score_rank"] for item in items])),
                **{f"reason_{reason}": count for reason, count in sorted(reasons.items())},
            }
        )
    return output


def main() -> None:
    args = parse_args()
    models = parse_models(args.model)
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
    names = [data_info["names"][i] for i in range(len(data_info["names"]))]
    category_ids = {index: category_id_by_name[name] for index, name in enumerate(names)}
    device = select_device(args.device, verbose=False)
    all_candidate_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    prediction_manifest: dict[str, dict[str, str]] = {}
    model_audit: list[dict[str, Any]] = []
    amp = device.type == "cuda"

    for model_name, checkpoint in models.items():
        _, loader = make_loader(args.data, args)
        wrapped = YOLO(str(checkpoint))
        net = wrapped.model.to(device).float().eval()
        head = net.model[-1]
        if not isinstance(head, Detect) or not head.end2end or head.nc != len(names):
            raise RuntimeError(f"{model_name} is not a compatible four-class end-to-end detector")
        if isinstance(net.args, dict):
            net.args = SimpleNamespace(**net.args)
        state_before = sha256(checkpoint)
        predictions = {"current": [], "post_nms_score_oracle": [], "pre_nms_oracle_nms": []}
        seen_images = 0
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
                if not isinstance(raw, dict) or "one2many" not in raw:
                    raise RuntimeError(f"{model_name} did not expose one2many predictions")
                decoded = head._inference(raw["one2many"]).float()
                finals = final_o2m(decoded, head.nc, args)
                height, width = images.shape[-2:]
                for image_index in range(images.shape[0]):
                    absolute_index = seen_images + image_index
                    if args.max_images and absolute_index >= args.max_images:
                        continue
                    image_name = Path(batch["im_file"][image_index]).name
                    image_id = image_id_by_stem[Path(image_name).stem]
                    gt_boxes, gt_classes = paired_gt(device_batch, image_index, width, height)
                    raw_boxes = decoded[image_index, :4].T
                    raw_scores = decoded[image_index, 4:].T
                    final = finals[image_index]
                    all_candidate_rows.extend(
                        candidate_rows_for_image(
                            model_name,
                            image_name,
                            raw_boxes,
                            raw_scores,
                            final,
                            gt_boxes,
                            gt_classes,
                            names,
                            args.score_floor,
                            args.nms_iou,
                        )
                    )
                    oracle_scores = oracle_score_matrix(raw_boxes, gt_boxes, gt_classes, head.nc)
                    post_oracle = final.clone()
                    if len(post_oracle):
                        for prediction_index, prediction in enumerate(post_oracle):
                            same_gt = gt_boxes[gt_classes == int(prediction[5])]
                            post_oracle[prediction_index, 4] = (
                                box_iou(prediction[None, :4].float(), same_gt.float()).max() if len(same_gt) else 0.0
                            )
                    pre_oracle = oracle_nms(raw_boxes, oracle_scores, head.nc, args)
                    common = (image_id, category_ids, (height, width), batch["ori_shape"][image_index], batch["ratio_pad"][image_index])
                    predictions["current"].extend(coco_rows(final, *common))
                    predictions["post_nms_score_oracle"].extend(coco_rows(post_oracle, *common))
                    predictions["pre_nms_oracle_nms"].extend(coco_rows(pre_oracle, *common))
                seen_images += images.shape[0]
                if args.max_images and seen_images >= args.max_images:
                    break
        for variant, rows in predictions.items():
            path = args.output / f"predictions_{model_name}_{variant}.json"
            path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
            prediction_manifest[f"{model_name}_{variant}"] = {"path": str(path), "rows": len(rows)}
            metric_rows.append(metric_row(f"{model_name}_{variant}", coco_gt, rows, category_id_by_name))
        model_audit.append(
            {
                "model": model_name,
                "checkpoint": str(checkpoint),
                "sha256_before": state_before,
                "sha256_after": sha256(checkpoint),
                "checkpoint_unchanged": state_before == sha256(checkpoint),
                "images": min(seen_images, args.max_images) if args.max_images else seen_images,
                "head": type(head).__name__,
            }
        )
        del net, wrapped
        if device.type == "cuda":
            torch.cuda.empty_cache()

    candidate_summary = summarize_candidates(all_candidate_rows)
    metric_by_case = {row["case"]: row for row in metric_rows}
    gbrg_current = metric_by_case["GBRG_current"]
    gbrg_post = metric_by_case["GBRG_post_nms_score_oracle"]
    gbrg_pre = metric_by_case["GBRG_pre_nms_oracle_nms"]
    current_reproduced = (
        args.max_images == 0 and abs(gbrg_current["AP"] - args.expected_gbrg_ap) <= args.reproduction_tolerance
    ) or args.max_images > 0
    post_delta = gbrg_post["AP"] - gbrg_current["AP"]
    verdict = {
        "current_GBRG_AP": gbrg_current["AP"],
        "expected_GBRG_AP": args.expected_gbrg_ap,
        "current_metric_reproduced": current_reproduced,
        "post_nms_fixed_box_score_oracle_AP": gbrg_post["AP"],
        "post_nms_fixed_box_score_oracle_delta_AP": post_delta,
        "pre_nms_fixed_box_oracle_plus_same_NMS_AP": gbrg_pre["AP"],
        "pre_nms_fixed_box_oracle_plus_same_NMS_delta_AP": gbrg_pre["AP"] - gbrg_current["AP"],
        "criterion_4_oracle_delta_at_least_gate": current_reproduced and post_delta >= args.oracle_delta_gate,
        "oracle_delta_gate": args.oracle_delta_gate,
    }
    write_csv(args.output / "per_gt_candidates.csv", all_candidate_rows)
    write_csv(args.output / "candidate_summary.csv", candidate_summary)
    write_csv(args.output / "metrics.csv", metric_rows)
    payload = {
        "protocol": {
            "split": "Val only; Test untouched",
            "data": str(args.data),
            "coco_gt": str(coco_path),
            "imgsz": args.imgsz,
            "batch": args.batch,
            "workers": args.workers,
            "score_floor": args.score_floor,
            "nms_iou": args.nms_iou,
            "max_det": args.max_det,
            "oracle_definition": {
                "post_nms_score_oracle": "identical final boxes/classes; score replaced by best same-class GT IoU",
                "pre_nms_oracle_nms": "identical raw boxes/classes; score replaced by best same-class GT IoU; same NMS reapplied",
                "boundary": "GT-aware non-deployable ceiling; no box coordinate is changed",
            },
        },
        "models": model_audit,
        "metrics": metric_rows,
        "candidate_summary": candidate_summary,
        "predictions": prediction_manifest,
        "verdict": verdict,
    }
    (args.output / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(verdict, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
