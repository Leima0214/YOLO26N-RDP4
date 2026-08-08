"""Val-only native and COCO paper metrics for Japan4 candidate checkpoints."""

from __future__ import annotations

import argparse
import contextlib
import copy
import csv
import io
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ultralytics import YOLO  # noqa: E402
from ultralytics.data.utils import check_det_dataset  # noqa: E402
from ultralytics.utils.torch_utils import get_flops, get_flops_with_torch_profiler  # noqa: E402

NAMES = ("D00", "D10", "D20", "D40")


def parse_checkpoints(values: list[str]) -> dict[str, Path]:
    checkpoints = {}
    for value in values:
        name, separator, path = value.partition("=")
        resolved = Path(path).expanduser().resolve()
        if not separator or not name or name in checkpoints or not resolved.is_file():
            raise ValueError(f"Expected unique NAME=CHECKPOINT, got {value}")
        checkpoints[name] = resolved
    return checkpoints


def remap_predictions(path: Path, ground_truth: COCO) -> list[dict]:
    image_ids = {Path(image["file_name"]).stem: image_id for image_id, image in ground_truth.imgs.items()}
    category_ids = {category["name"]: category_id for category_id, category in ground_truth.cats.items()}
    output = []
    for prediction in json.loads(path.read_text(encoding="utf-8")):
        stem = Path(prediction["file_name"]).stem
        class_index = int(prediction["category_id"]) - 1
        if stem not in image_ids or not 0 <= class_index < len(NAMES):
            raise ValueError(f"Cannot map prediction: {prediction}")
        output.append(
            {
                "image_id": image_ids[stem],
                "category_id": category_ids[NAMES[class_index]],
                "bbox": prediction["bbox"],
                "score": prediction["score"],
            }
        )
    return output


def evaluate(ground_truth: COCO, predictions: list[dict], category_ids: list[int]) -> COCOeval:
    detections = ground_truth.loadRes(predictions)
    evaluator = COCOeval(ground_truth, detections, "bbox")
    evaluator.params.imgIds = sorted(ground_truth.imgs)
    evaluator.params.catIds = category_ids
    with contextlib.redirect_stdout(io.StringIO()):
        evaluator.evaluate()
        evaluator.accumulate()
    return evaluator


def mean_valid(values: np.ndarray) -> float | None:
    values = values[values > -1]
    return float(values.mean()) if values.size else None


def ap(evaluator: COCOeval, area: str = "all", iou: float | None = None) -> float | None:
    area_index = list(evaluator.params.areaRngLbl).index(area)
    precision = evaluator.eval["precision"]
    if iou is not None:
        iou_index = int(np.argmin(np.abs(evaluator.params.iouThrs - iou)))
        precision = precision[iou_index : iou_index + 1]
    return mean_valid(precision[:, :, :, area_index, -1])


def ar100(evaluator: COCOeval, area: str = "all") -> float | None:
    area_index = list(evaluator.params.areaRngLbl).index(area)
    return mean_valid(evaluator.eval["recall"][:, :, area_index, -1])


def latency_ms(model: torch.nn.Module, device: torch.device, imgsz: int) -> float | None:
    if device.type != "cuda":
        return None
    sample = torch.zeros(1, 3, imgsz, imgsz, device=device)
    deployed = copy.deepcopy(model).to(device).eval().fuse(verbose=False)
    with torch.inference_mode():
        for _ in range(20):
            deployed(sample)
        torch.cuda.synchronize(device)
        values = []
        for _ in range(100):
            start = time.perf_counter()
            deployed(sample)
            torch.cuda.synchronize(device)
            values.append((time.perf_counter() - start) * 1000)
    return statistics.median(values)


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()
    checkpoints = parse_checkpoints(args.checkpoint)
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)

    data = check_det_dataset(str(args.data.resolve()))
    coco_path = Path(data["path"]) / "annotations" / "instances_val.json"
    if not coco_path.is_file():
        raise FileNotFoundError(coco_path)
    ground_truth = COCO(str(coco_path))
    categories = {category["name"]: category_id for category_id, category in ground_truth.cats.items()}
    if set(categories) != set(NAMES):
        raise ValueError(f"Unexpected categories: {categories}")
    device = torch.device(f"cuda:{args.device}" if args.device.isdigit() else args.device)
    main_rows, class_rows = [], []

    for name, checkpoint in checkpoints.items():
        model = YOLO(str(checkpoint), task="detect")
        metrics = model.val(
            data=str(args.data.resolve()),
            split="val",
            imgsz=args.imgsz,
            batch=args.batch,
            workers=args.workers,
            device=args.device,
            conf=0.001,
            iou=0.7,
            max_det=300,
            rect=True,
            save_json=True,
            plots=False,
            project=str(args.output / "predictions"),
            name=name,
            exist_ok=True,
            verbose=False,
        )
        prediction_path = args.output / "predictions" / name / "predictions.json"
        predictions = remap_predictions(prediction_path, ground_truth)
        all_eval = evaluate(ground_truth, predictions, list(categories.values()))
        native = metrics.results_dict
        parameters = sum(parameter.numel() for parameter in model.model.parameters())
        gflops = get_flops(model.model, args.imgsz) or get_flops_with_torch_profiler(model.model, args.imgsz)
        main_rows.append(
            {
                "model": name,
                "P": native["metrics/precision(B)"],
                "R": native["metrics/recall(B)"],
                "AP50": ap(all_eval, iou=0.50),
                "AP50_95": ap(all_eval),
                "AP75": ap(all_eval, iou=0.75),
                "AP_small": ap(all_eval, "small"),
                "AP_medium": ap(all_eval, "medium"),
                "AP_large": ap(all_eval, "large"),
                "AR100": ar100(all_eval),
                "params": parameters,
                "GFLOPs": gflops,
                "pytorch_batch1_latency_ms": latency_ms(model.model, device, args.imgsz),
                "checkpoint_MB": checkpoint.stat().st_size / 2**20,
            }
        )
        for class_index, class_name in metrics.names.items():
            class_eval = evaluate(ground_truth, predictions, [categories[class_name]])
            class_rows.append(
                {
                    "model": name,
                    "class": class_name,
                    "P": float(metrics.box.p[class_index]),
                    "R": float(metrics.box.r[class_index]),
                    "AP50": ap(class_eval, iou=0.50),
                    "AP50_95": ap(class_eval),
                    "AP75": ap(class_eval, iou=0.75),
                    "AR100": ar100(class_eval),
                }
            )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    write_csv(args.output / "main_metrics.csv", main_rows)
    write_csv(args.output / "per_class_metrics.csv", class_rows)
    report = {
        "protocol": {
            "split": "val_only",
            "data": str(args.data.resolve()),
            "imgsz": args.imgsz,
            "batch": args.batch,
            "conf": 0.001,
            "iou": 0.7,
            "validator_max_det": 300,
            "coco_max_det": 100,
        },
        "main": main_rows,
        "per_class": class_rows,
    }
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
