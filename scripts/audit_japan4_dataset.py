"""Read-only integrity audit for the frozen Japan4-cleanV3 train/val protocol."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import yaml
from PIL import Image

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def image_files(path: Path) -> list[Path]:
    return sorted(file for file in path.rglob("*") if file.suffix.lower() in IMAGE_SUFFIXES)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.data.resolve().read_text(encoding="utf-8"))
    names = config.get("names", {})
    names = list(names.values()) if isinstance(names, dict) else list(names)
    if int(config.get("nc", -1)) != 4 or names != ["D00", "D10", "D20", "D40"]:
        raise AssertionError("Data YAML is not the frozen four-class Japan4 protocol")
    root = Path(config["path"])
    if not root.is_absolute():
        root = (args.data.resolve().parent / root).resolve()

    report = {"data_yaml": str(args.data.resolve()), "root": str(root), "splits": {}}
    split_stems = {}
    failures = []
    for split in ("train", "val"):
        image_dir = root / config[split]
        label_dir = root / "labels" / split
        if not image_dir.is_dir() or not label_dir.is_dir():
            raise FileNotFoundError(f"Missing {split} images or labels: {image_dir}, {label_dir}")
        images = image_files(image_dir)
        stems = {image.stem for image in images}
        split_stems[split] = stems
        orphan_labels = sorted(label.stem for label in label_dir.glob("*.txt") if label.stem not in stems)
        class_counts = [0, 0, 0, 0]
        invalid_labels = []
        empty_labels = 0
        corrupt_images = []
        for image in images:
            try:
                with Image.open(image) as handle:
                    handle.verify()
            except Exception as error:
                corrupt_images.append({"file": str(image), "error": str(error)})
            label = label_dir / f"{image.stem}.txt"
            if not label.is_file() or not label.read_text(encoding="utf-8").strip():
                empty_labels += 1
                continue
            for line_number, line in enumerate(label.read_text(encoding="utf-8").splitlines(), start=1):
                fields = line.split()
                try:
                    class_id = int(fields[0])
                    box = [float(value) for value in fields[1:]]
                    valid = (
                        len(fields) == 5
                        and 0 <= class_id < 4
                        and 0 <= box[0] <= 1
                        and 0 <= box[1] <= 1
                        and 0 < box[2] <= 1
                        and 0 < box[3] <= 1
                        and box[0] - box[2] / 2 >= -1e-6
                        and box[1] - box[3] / 2 >= -1e-6
                        and box[0] + box[2] / 2 <= 1 + 1e-6
                        and box[1] + box[3] / 2 <= 1 + 1e-6
                    )
                except (ValueError, IndexError):
                    valid = False
                    class_id = -1
                if not valid:
                    invalid_labels.append(f"{label}:{line_number}:{line}")
                else:
                    class_counts[class_id] += 1
        report["splits"][split] = {
            "images": len(images),
            "empty_or_missing_labels": empty_labels,
            "class_counts": dict(zip(("D00", "D10", "D20", "D40"), class_counts)),
            "orphan_labels": orphan_labels,
            "invalid_labels": invalid_labels,
            "corrupt_images": corrupt_images,
        }
        failures.extend(orphan_labels + invalid_labels + [item["file"] for item in corrupt_images])

    overlap = sorted(split_stems["train"] & split_stems["val"])
    report["train_val_filename_overlap"] = overlap
    failures.extend(overlap)

    assignment_path = root / "audit" / "split_assignment.csv"
    split_report_path = root / "audit" / "split_report.json"
    if not assignment_path.is_file() or not split_report_path.is_file():
        raise FileNotFoundError("Frozen split audit assets are missing")
    group_splits = defaultdict(set)
    with assignment_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            group_splits[row["group_id"]].add(row["v3_split"])
    cross_split_groups = sorted(group for group, splits in group_splits.items() if len(splits) > 1)
    frozen_report = json.loads(split_report_path.read_text(encoding="utf-8"))
    report["group_safe"] = {
        "groups": len(group_splits),
        "cross_split_groups_recomputed": len(cross_split_groups),
        "cross_split_groups_manifest": frozen_report.get("cross_split_group_count"),
        "seed": frozen_report.get("seed"),
        "ratios": frozen_report.get("ratios"),
    }
    if cross_split_groups or frozen_report.get("cross_split_group_count") != 0:
        failures.extend(cross_split_groups or ["split_report cross_split_group_count != 0"])

    report["passed"] = not failures
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if failures:
        raise SystemExit(f"Japan4-cleanV3 audit failed with {len(failures)} issue(s)")


if __name__ == "__main__":
    main()
