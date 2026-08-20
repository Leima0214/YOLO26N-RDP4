"""Evaluate Japan4 checkpoints with one fixed native/COCO paper-table protocol."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from pycocotools.coco import COCO

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
for path in (SCRIPT_DIR, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from diagnose_japan4_c3_dysample import ap, ar100, coco_eval, remap_predictions, sha256
from ultralytics import YOLO
from ultralytics.data.utils import check_det_dataset
from ultralytics.utils.torch_utils import get_flops, get_flops_with_torch_profiler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", choices=("val", "test"), default=("val", "test"))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=300)
    return parser.parse_args()


def parse_checkpoints(values: list[str]) -> dict[str, Path]:
    checkpoints = {}
    for value in values:
        name, separator, raw_path = value.partition("=")
        if not separator or not name or not raw_path or name in checkpoints:
            raise ValueError(f"Expected unique NAME=PATH checkpoint, got {value!r}")
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        checkpoints[name] = path
    return checkpoints


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def coco_row(evaluator, prefix: str = "coco_") -> dict[str, float | None]:
    return {
        f"{prefix}AP50_95": ap(evaluator),
        f"{prefix}AP50": ap(evaluator, iou=0.50),
        f"{prefix}AP75": ap(evaluator, iou=0.75),
        f"{prefix}AP_small": ap(evaluator, area="small"),
        f"{prefix}AP_medium": ap(evaluator, area="medium"),
        f"{prefix}AP_large": ap(evaluator, area="large"),
        f"{prefix}AR100": ar100(evaluator),
        f"{prefix}AR_small": ar100(evaluator, area="small"),
        f"{prefix}AR_medium": ar100(evaluator, area="medium"),
        f"{prefix}AR_large": ar100(evaluator, area="large"),
    }


def size_counts(coco_gt: COCO) -> dict[str, int]:
    counts = Counter()
    for annotation in coco_gt.anns.values():
        area = annotation.get("area", annotation["bbox"][2] * annotation["bbox"][3])
        counts["small" if area < 32**2 else "medium" if area < 96**2 else "large"] += 1
    return dict(counts)


def main() -> None:
    args = parse_args()
    checkpoints = parse_checkpoints(args.checkpoint)
    args.data = args.data.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    if not args.data.is_file():
        raise FileNotFoundError(args.data)
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    prediction_root = args.output / "predictions"
    data_info = check_det_dataset(str(args.data))

    paper_rows: list[dict[str, Any]] = []
    class_rows: list[dict[str, Any]] = []
    datasets: dict[str, Any] = {}
    for split in args.splits:
        coco_path = Path(data_info["path"]) / "annotations" / f"instances_{split}.json"
        if not coco_path.is_file():
            raise FileNotFoundError(coco_path)
        coco_gt = COCO(str(coco_path))
        category_ids = {category["name"]: category_id for category_id, category in coco_gt.cats.items()}
        datasets[split] = {
            "coco_gt": str(coco_path),
            "images": len(coco_gt.imgs),
            "instances": len(coco_gt.anns),
            "size_counts": size_counts(coco_gt),
        }

        for name, checkpoint in checkpoints.items():
            model = YOLO(str(checkpoint))
            params = sum(parameter.numel() for parameter in model.model.parameters())
            gflops = get_flops(model.model, args.imgsz) or get_flops_with_torch_profiler(model.model, args.imgsz)
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            metrics = model.val(
                data=str(args.data),
                split=split,
                imgsz=args.imgsz,
                batch=args.batch,
                device=args.device,
                workers=args.workers,
                conf=args.conf,
                iou=args.iou,
                max_det=args.max_det,
                rect=True,
                save_json=True,
                plots=False,
                project=str(prediction_root),
                name=f"{name}_{split}",
                exist_ok=True,
                verbose=False,
            )
            prediction_path = prediction_root / f"{name}_{split}" / "predictions.json"
            predictions = remap_predictions(prediction_path, coco_gt)
            all_eval = coco_eval(coco_gt, predictions, list(category_ids.values()))
            native = metrics.results_dict
            speed = metrics.speed
            paper_rows.append(
                {
                    "split": split,
                    "model": name,
                    "native_P": native["metrics/precision(B)"],
                    "native_R": native["metrics/recall(B)"],
                    "native_mAP50": native["metrics/mAP50(B)"],
                    "native_mAP75": metrics.box.map75,
                    "native_mAP50_95": native["metrics/mAP50-95(B)"],
                    **coco_row(all_eval),
                    "params": params,
                    "GFLOPs": gflops,
                    "preprocess_ms": speed["preprocess"],
                    "inference_ms": speed["inference"],
                    "postprocess_ms": speed["postprocess"],
                    "peak_VRAM_GB": torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else None,
                }
            )
            for class_index, class_name in metrics.names.items():
                class_eval = coco_eval(coco_gt, predictions, [category_ids[class_name]])
                class_rows.append(
                    {
                        "split": split,
                        "model": name,
                        "class": class_name,
                        "native_P": metrics.box.p[class_index],
                        "native_R": metrics.box.r[class_index],
                        "native_AP50": metrics.box.ap50[class_index],
                        "native_AP50_95": metrics.box.maps[class_index],
                        **coco_row(class_eval),
                    }
                )
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    write_csv(args.output / "paper_main_metrics.csv", paper_rows)
    write_csv(args.output / "per_class_metrics.csv", class_rows)
    summary = {
        "protocol": {
            "data": str(args.data),
            "splits": list(args.splits),
            "imgsz": args.imgsz,
            "batch": args.batch,
            "conf": args.conf,
            "iou": args.iou,
            "validator_max_det": args.max_det,
            "coco_max_det": 100,
            "size_definition": "COCO area: small < 32^2, medium < 96^2, large otherwise",
            "note": "native_* and coco_* metrics use different evaluators and must not be mixed.",
        },
        "datasets": datasets,
        "weights": {name: {"path": str(path), "sha256": sha256(path)} for name, path in checkpoints.items()},
        "artifacts": ["paper_main_metrics.csv", "per_class_metrics.csv", "predictions/"],
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"summary": summary, "paper_main_metrics": paper_rows}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
