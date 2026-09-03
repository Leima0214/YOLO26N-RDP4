#!/usr/bin/env python3
"""Audit the SVRDD YOLO dataset before any training or evaluation is allowed."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any

from PIL import Image

NAMES = ("LC", "TC", "AC", "P", "MC", "LP", "TP")
EXPECTED_IMAGES = {"train": 6000, "val": 1000, "test": 1000}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/SVRDD/SVRRDD_YOLO_READY"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--strict-counts", action="store_true", help="Treat expected 6000/1000/1000 mismatch as an error")
    return parser.parse_args()


def quantiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {key: None for key in ("min", "p25", "median", "p75", "max")}
    ordered = sorted(values)
    pick = lambda q: ordered[round((len(ordered) - 1) * q)]
    return {"min": ordered[0], "p25": pick(0.25), "median": median(ordered), "p75": pick(0.75), "max": ordered[-1]}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_split(root: Path, split: str) -> tuple[dict[str, Any], list[str], list[str]]:
    image_dir, label_dir = root / "images" / split, root / "labels" / split
    errors: list[str] = []
    warnings: list[str] = []
    if not image_dir.is_dir() or not label_dir.is_dir():
        return {}, [f"missing images/labels directory for {split}"], warnings
    images = sorted(path for path in image_dir.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)
    labels = sorted(label_dir.rglob("*.txt"))
    image_by_stem: dict[str, list[Path]] = defaultdict(list)
    label_by_stem: dict[str, list[Path]] = defaultdict(list)
    for path in images:
        image_by_stem[path.stem].append(path)
    for path in labels:
        label_by_stem[path.stem].append(path)
    duplicate_image_stems = {stem: [str(p) for p in paths] for stem, paths in image_by_stem.items() if len(paths) > 1}
    duplicate_label_stems = {stem: [str(p) for p in paths] for stem, paths in label_by_stem.items() if len(paths) > 1}
    if duplicate_image_stems:
        errors.append(f"{split}: duplicate image stems ({len(duplicate_image_stems)})")
    if duplicate_label_stems:
        errors.append(f"{split}: duplicate label stems ({len(duplicate_label_stems)})")
    missing_labels = sorted(set(image_by_stem) - set(label_by_stem))
    orphan_labels = sorted(set(label_by_stem) - set(image_by_stem))
    if missing_labels:
        errors.append(f"{split}: images without label files ({len(missing_labels)})")
    if orphan_labels:
        errors.append(f"{split}: labels without images ({len(orphan_labels)})")

    class_counts = Counter()
    size_counts = Counter()
    dimensions = Counter()
    image_aspects: list[float] = []
    object_aspects: list[float] = []
    empty_labels = 0
    corrupt_images: list[str] = []
    malformed: list[str] = []
    duplicate_boxes = 0
    content_hashes: dict[str, list[str]] = defaultdict(list)
    boxes_total = 0
    for stem, paths in image_by_stem.items():
        if len(paths) != 1:
            continue
        image_path = paths[0]
        try:
            with Image.open(image_path) as image:
                image.verify()
            with Image.open(image_path) as image:
                width, height = image.size
            if width <= 0 or height <= 0:
                raise ValueError(f"invalid dimensions {width}x{height}")
            dimensions[f"{width}x{height}"] += 1
            image_aspects.append(width / height)
            content_hashes[sha256(image_path)].append(str(image_path))
        except Exception as exc:  # noqa: BLE001
            corrupt_images.append(f"{image_path}: {exc}")
            continue
        label_paths = label_by_stem.get(stem, [])
        if len(label_paths) != 1:
            continue
        lines = [line.strip() for line in label_paths[0].read_text(encoding="utf-8-sig").splitlines() if line.strip()]
        if not lines:
            empty_labels += 1
            continue
        seen_boxes: set[tuple[int, float, float, float, float]] = set()
        for line_no, line in enumerate(lines, 1):
            fields = line.split()
            try:
                if len(fields) != 5:
                    raise ValueError(f"expected 5 fields, got {len(fields)}")
                raw_class, *raw_box = fields
                class_float = float(raw_class)
                class_id = int(class_float)
                if class_float != class_id or not 0 <= class_id < len(NAMES):
                    raise ValueError(f"class id out of range: {raw_class}")
                x, y, w, h = map(float, raw_box)
                if not all(math.isfinite(value) for value in (x, y, w, h)):
                    raise ValueError("non-finite coordinate")
                if w <= 0 or h <= 0:
                    raise ValueError("non-positive width/height")
                if not all(0.0 <= value <= 1.0 for value in (x, y, w, h)):
                    raise ValueError("coordinate outside normalized [0,1]")
                if x - w / 2 < -1e-6 or y - h / 2 < -1e-6 or x + w / 2 > 1 + 1e-6 or y + h / 2 > 1 + 1e-6:
                    raise ValueError("box extends outside image")
                key = (class_id, x, y, w, h)
                duplicate_boxes += key in seen_boxes
                seen_boxes.add(key)
                class_counts[class_id] += 1
                boxes_total += 1
                area = w * width * h * height
                size_counts["small" if area < 32**2 else "medium" if area < 96**2 else "large"] += 1
                object_aspects.append((w * width) / (h * height))
            except ValueError as exc:
                malformed.append(f"{label_paths[0]}:{line_no}: {exc}")
    duplicate_content = [paths for paths in content_hashes.values() if len(paths) > 1]
    if corrupt_images:
        errors.append(f"{split}: corrupt images ({len(corrupt_images)})")
    if malformed:
        errors.append(f"{split}: malformed label rows ({len(malformed)})")
    if duplicate_content:
        warnings.append(f"{split}: duplicate image content groups ({len(duplicate_content)})")
    if duplicate_boxes:
        warnings.append(f"{split}: exact duplicate label rows ({duplicate_boxes})")
    return {
        "images": len(images), "labels": len(labels), "empty_labels": empty_labels, "instances": boxes_total,
        "class_instances": {NAMES[i]: class_counts[i] for i in range(len(NAMES))},
        "size_instances": dict(size_counts), "image_dimensions": dict(dimensions),
        "image_aspect_ratio": quantiles(image_aspects), "object_aspect_ratio": quantiles(object_aspects),
        "missing_label_stems": missing_labels, "orphan_label_stems": orphan_labels,
        "duplicate_image_stems": duplicate_image_stems, "duplicate_label_stems": duplicate_label_stems,
        "duplicate_image_content_groups": duplicate_content, "duplicate_box_rows": duplicate_boxes,
        "corrupt_images": corrupt_images, "malformed_labels": malformed,
        "stems": sorted(image_by_stem),
    }, errors, warnings


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing report: {output}")
    reports: dict[str, Any] = {}
    errors: list[str] = []
    warnings: list[str] = []
    for split in ("train", "val", "test"):
        report, split_errors, split_warnings = audit_split(root, split)
        reports[split] = report
        errors.extend(split_errors)
        warnings.extend(split_warnings)
        actual = report.get("images")
        if actual is not None and actual != EXPECTED_IMAGES[split]:
            message = f"{split}: expected {EXPECTED_IMAGES[split]} images, found {actual}"
            (errors if args.strict_counts else warnings).append(message)
    overlap: dict[str, list[str]] = {}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        shared = sorted(set(reports.get(left, {}).get("stems", [])) & set(reports.get(right, {}).get("stems", [])))
        overlap[f"{left}-{right}"] = shared
        if shared:
            errors.append(f"stem leakage {left}-{right}: {len(shared)}")
    for report in reports.values():
        report.pop("stems", None)
    result = {"root": str(root), "classes": list(NAMES), "expected_images": EXPECTED_IMAGES,
              "splits": reports, "cross_split_stem_overlap": overlap,
              "errors": errors, "warnings": warnings, "passed": not errors}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "passed": not errors, "errors": errors, "warnings": warnings}, indent=2))
    if errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
