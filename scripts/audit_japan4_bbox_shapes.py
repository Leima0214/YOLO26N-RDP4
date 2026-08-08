"""Audit train-split bounding-box aspect ratios using original image dimensions."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image


CLASS_NAMES = {0: "D00", 1: "D10", 2: "D20", 3: "D40"}
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def summarize(name: str, ratios: list[float]) -> dict:
    values = np.asarray(ratios, dtype=np.float64)
    logs = np.log(values)
    lower, upper = 1 / 1.5, 1.5
    magnitude = 1 / (1 + np.exp(-(np.abs(logs) - math.log(1.5)) / 0.25))
    horizontal = magnitude / (1 + np.exp(-logs / 0.25))
    vertical = magnitude / (1 + np.exp(logs / 0.25))
    return {
        "class": name,
        "n": int(values.size),
        "ratio_p25_p50_p75": np.quantile(values, (0.25, 0.5, 0.75)).tolist(),
        "log_ratio_p25_p50_p75": np.quantile(logs, (0.25, 0.5, 0.75)).tolist(),
        "horizontal_ratio_gt_1_5": float(np.mean(values > upper)),
        "vertical_ratio_lt_1_over_1_5": float(np.mean(values < lower)),
        "near_square_ratio": float(np.mean((values >= lower) & (values <= upper))),
        "target_h_mean": float(horizontal.mean()),
        "target_v_mean": float(vertical.mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    label_dir, image_dir = args.root / "labels/train", args.root / "images/train"
    ratios = {class_id: [] for class_id in CLASS_NAMES}
    missing_images = invalid_labels = 0
    for label_path in sorted(label_dir.glob("*.txt")):
        image_path = next(
            (image_dir / f"{label_path.stem}{suffix}" for suffix in IMAGE_SUFFIXES if (image_dir / f"{label_path.stem}{suffix}").exists()),
            None,
        )
        if image_path is None:
            missing_images += 1
            continue
        with Image.open(image_path) as image:
            image_width, image_height = image.size
        for line in label_path.read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if not fields:
                continue
            class_id, width, height = int(float(fields[0])), float(fields[3]), float(fields[4])
            if class_id not in ratios or width <= 0 or height <= 0:
                invalid_labels += 1
                continue
            ratios[class_id].append(width * image_width / (height * image_height))

    rows = [summarize(CLASS_NAMES[class_id], values) for class_id, values in ratios.items()]
    rows.append(summarize("all", [ratio for values in ratios.values() for ratio in values]))
    report = {
        "root": str(args.root),
        "split": "train",
        "label_files": len(list(label_dir.glob("*.txt"))),
        "missing_images": missing_images,
        "invalid_labels": invalid_labels,
        "near_square_definition": "1/1.5 <= pixel_width/pixel_height <= 1.5",
        "rows": rows,
    }
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
