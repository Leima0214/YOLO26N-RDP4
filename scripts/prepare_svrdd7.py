#!/usr/bin/env python3
"""Convert SVRDD YOLO labels into COCO annotations without changing the dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

NAMES = ("LC", "TC", "AC", "P", "MC", "LP", "TP")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/SVRDD/SVRRDD_YOLO_READY"))
    parser.add_argument("--split", choices=("all", "train", "val", "test"), default="all")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def convert(root: Path, split: str, overwrite: bool) -> Path:
    image_dir, label_dir = root / "images" / split, root / "labels" / split
    if not image_dir.is_dir() or not label_dir.is_dir():
        raise FileNotFoundError(f"Missing {image_dir} or {label_dir}")
    output = root / "annotations" / f"instances_{split}.json"
    if output.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {output}; pass --overwrite explicitly")
    images = sorted(path for path in image_dir.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)
    stems: dict[str, Path] = {}
    for path in images:
        if path.stem in stems:
            raise ValueError(f"Duplicate image stem in {split}: {path.stem}: {stems[path.stem]}, {path}")
        stems[path.stem] = path
    coco_images, annotations = [], []
    annotation_id = 1
    for image_id, image_path in enumerate(images, 1):
        with Image.open(image_path) as image:
            width, height = image.size
        coco_images.append({"id": image_id, "file_name": str(image_path.relative_to(root)).replace("\\", "/"),
                            "width": width, "height": height})
        label_path = label_dir / image_path.relative_to(image_dir).with_suffix(".txt")
        if not label_path.is_file():
            raise FileNotFoundError(f"Missing label for {image_path}: {label_path}")
        for line_no, line in enumerate(label_path.read_text(encoding="utf-8-sig").splitlines(), 1):
            if not line.strip():
                continue
            fields = line.split()
            if len(fields) != 5:
                raise ValueError(f"{label_path}:{line_no}: expected 5 fields")
            class_value, x, y, w, h = map(float, fields)
            class_id = int(class_value)
            if class_value != class_id or not 0 <= class_id < len(NAMES):
                raise ValueError(f"{label_path}:{line_no}: invalid class {class_value}")
            if w <= 0 or h <= 0 or not all(0 <= value <= 1 for value in (x, y, w, h)):
                raise ValueError(f"{label_path}:{line_no}: invalid normalized box")
            box_w, box_h = w * width, h * height
            x_min, y_min = (x - w / 2) * width, (y - h / 2) * height
            if x_min < -1e-4 or y_min < -1e-4 or x_min + box_w > width + 1e-4 or y_min + box_h > height + 1e-4:
                raise ValueError(f"{label_path}:{line_no}: box outside image")
            annotations.append({"id": annotation_id, "image_id": image_id, "category_id": class_id + 1,
                                "bbox": [x_min, y_min, box_w, box_h], "area": box_w * box_h, "iscrowd": 0})
            annotation_id += 1
    document = {"info": {"description": f"SVRDD {split} converted from frozen YOLO labels"},
                "licenses": [], "images": coco_images, "annotations": annotations,
                "categories": [{"id": i + 1, "name": name, "supercategory": "road_damage"} for i, name in enumerate(NAMES)]}
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(document, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(output)
    return output


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    splits = ("train", "val", "test") if args.split == "all" else (args.split,)
    for split in splits:
        output = convert(root, split, args.overwrite)
        print(f"WROTE {output}")


if __name__ == "__main__":
    main()
