#!/usr/bin/env python3
"""Val-only wrapper around the frozen Japan4 paper-metrics evaluator."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
EVALUATOR = ROOT / "scripts/evaluate_japan4_paper_metrics.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="0")
    return parser.parse_args()


def _number(value: str) -> Any:
    if value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return value


def main() -> None:
    args = parse_args()
    command = [
        sys.executable,
        "-u",
        str(EVALUATOR),
        "--data",
        str(args.data.expanduser().resolve()),
        "--output",
        str(args.output.expanduser().resolve()),
        "--splits",
        "val",
        "--imgsz",
        str(args.imgsz),
        "--batch",
        str(args.batch),
        "--workers",
        str(args.workers),
        "--device",
        str(args.device),
        "--conf",
        "0.001",
        "--iou",
        "0.7",
        "--max-det",
        "300",
    ]
    for checkpoint in args.checkpoint:
        command.extend(("--checkpoint", checkpoint))
    print("TEST_SEALED=PASS (wrapper hard-codes --splits val)")
    subprocess.run(command, cwd=ROOT, check=True)

    output = args.output.expanduser().resolve()
    source = output / "paper_main_metrics.csv"
    class_source = output / "per_class_metrics.csv"
    if not source.is_file() or not class_source.is_file():
        raise FileNotFoundError("Frozen evaluator did not produce expected metric tables")
    with source.open(encoding="utf-8", newline="") as file:
        main_rows = [{key: _number(value) for key, value in row.items()} for row in csv.DictReader(file)]
    with class_source.open(encoding="utf-8", newline="") as file:
        class_rows = [{key: _number(value) for key, value in row.items()} for row in csv.DictReader(file)]
    if not main_rows or any(row.get("split") != "val" for row in main_rows):
        raise AssertionError("Adaptive evaluator emitted a non-Val row")
    (output / "main_metrics.csv").write_bytes(source.read_bytes())
    metrics = {
        "test_sealed": True,
        "evaluator": str(EVALUATOR),
        "paper_main_metrics": main_rows,
        "per_class_metrics": class_rows,
        "artifacts": [
            "main_metrics.csv",
            "paper_main_metrics.csv",
            "per_class_metrics.csv",
            "summary.json",
            "predictions/",
        ],
    }
    (output / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
