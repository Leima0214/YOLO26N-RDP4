"""Audit Japan4 train-GT box spans on stride-8/16/32 feature maps at imgsz=640."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

CLASS_NAMES = {0: "D00", 1: "D10", 2: "D20", 3: "D40"}
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
QUANTILES = (0.25, 0.50, 0.75, 0.90)


def summarize(records: list[dict], stride: int) -> dict:
    output = {"n": len(records)}
    for metric in ("width", "height", "long_side", "short_side"):
        values = np.asarray([record[metric] / stride for record in records], dtype=np.float64)
        output[f"{metric}_cells_p25_p50_p75_p90"] = np.quantile(values, QUANTILES).tolist()
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--split", default="train", choices=("train",))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    image_dir, label_dir = args.root / f"images/{args.split}", args.root / f"labels/{args.split}"
    if not image_dir.is_dir() or not label_dir.is_dir():
        raise FileNotFoundError((image_dir, label_dir))

    records = {class_id: [] for class_id in CLASS_NAMES}
    missing_images = invalid_labels = 0
    label_files = sorted(label_dir.glob("*.txt"))
    for label_path in label_files:
        image_path = next(
            (image_dir / f"{label_path.stem}{suffix}" for suffix in IMAGE_SUFFIXES if (image_dir / f"{label_path.stem}{suffix}").is_file()),
            None,
        )
        if image_path is None:
            missing_images += 1
            continue
        with Image.open(image_path) as image:
            original_width, original_height = image.size
        scale = min(args.imgsz / original_width, args.imgsz / original_height)
        for line in label_path.read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if not fields:
                continue
            class_id, normalized_width, normalized_height = int(float(fields[0])), float(fields[3]), float(fields[4])
            if class_id not in records or normalized_width <= 0 or normalized_height <= 0:
                invalid_labels += 1
                continue
            width = normalized_width * original_width * scale
            height = normalized_height * original_height * scale
            records[class_id].append(
                {"width": width, "height": height, "long_side": max(width, height), "short_side": min(width, height)}
            )

    all_records = [record for class_records in records.values() for record in class_records]
    groups = {CLASS_NAMES[class_id]: values for class_id, values in records.items()}
    groups["all"] = all_records
    report = {
        "root": str(args.root.resolve()),
        "split": args.split,
        "test_read": False,
        "imgsz": args.imgsz,
        "geometry": "original normalized YOLO box scaled by standard aspect-preserving letterbox ratio to imgsz; padding does not change width/height",
        "augmentation": "disabled; this is the nominal pre-augmentation cell span",
        "label_files": len(label_files),
        "instances": len(all_records),
        "missing_images": missing_images,
        "invalid_labels": invalid_labels,
        "strides": {
            str(stride): {name: summarize(values, stride) for name, values in groups.items()}
            for stride in (8, 16, 32)
        },
    }
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
