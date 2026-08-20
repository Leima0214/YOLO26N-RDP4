#!/usr/bin/env python3
"""Read-only Train/Val scale diagnostics for SA-RS and MG-SA-RS checkpoints."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import YOLO  # noqa: E402
from ultralytics.cfg import DEFAULT_CFG, get_cfg  # noqa: E402
from ultralytics.data.build import build_dataloader, build_yolo_dataset  # noqa: E402
from ultralytics.data.utils import check_det_dataset  # noqa: E402
from ultralytics.nn.roadsnake_adaptive import ScaleAdaptiveRoadSnakeDetect  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--max-batches", type=int, default=0, help="0 means all batches")
    return parser.parse_args()


def _rank(values: np.ndarray) -> np.ndarray:
    """Return average ranks with deterministic tie handling."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def spearman(left: list[float], right: list[float]) -> float | None:
    if len(left) < 2:
        return None
    x, y = _rank(np.asarray(left)), _rank(np.asarray(right))
    if x.std() == 0 or y.std() == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def scale_stats(values: list[float] | np.ndarray) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=np.float64)
    if not array.size:
        return {"count": 0}
    quantiles = np.quantile(array, (0.10, 0.25, 0.50, 0.75, 0.90))
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "p10": float(quantiles[0]),
        "p25": float(quantiles[1]),
        "p50": float(quantiles[2]),
        "p75": float(quantiles[3]),
        "p90": float(quantiles[4]),
        "min": float(array.min()),
        "max": float(array.max()),
        "near_min_ratio": float((array < 0.45).mean()),
        "near_max_ratio": float((array > 2.40).mean()),
    }


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.expanduser().resolve()
    data_yaml = args.data.expanduser().resolve()
    output = args.output.expanduser().resolve()
    for path in (checkpoint, data_yaml):
        if not path.is_file():
            raise FileNotFoundError(path)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")

    device = torch.device(f"cuda:{args.device}" if str(args.device).isdigit() else args.device)
    model = YOLO(str(checkpoint)).model.float().to(device).eval()
    head = model.model[-1]
    if not isinstance(head, ScaleAdaptiveRoadSnakeDetect):
        raise TypeError(f"Expected adaptive RoadSnake head, got {type(head).__name__}")
    adapter = head.road_snake
    adapter.set_diagnostics(True)

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
            "mosaic": 0.0,
            "mixup": 0.0,
            "copy_paste": 0.0,
        },
    )
    dataset = build_yolo_dataset(
        cfg,
        data[args.split],
        args.batch,
        data,
        mode="val",
        rect=True,
        stride=32,
    )
    loader = build_dataloader(
        dataset,
        args.batch,
        args.workers,
        shuffle=False,
        rank=-1,
        pin_memory=device.type == "cuda",
    )

    records: list[dict[str, float | int | str]] = []
    dense_scale_h: list[np.ndarray] = []
    dense_scale_v: list[np.ndarray] = []
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            if args.max_batches and batch_index >= args.max_batches:
                break
            image = batch["img"].to(device).float().div(255.0)
            model(image)
            diagnostics = adapter.diagnostics(clone=False)
            scale_h, scale_v = diagnostics["scale_h"], diagnostics["scale_v"]
            dense_scale_h.append(scale_h.detach().float().cpu().numpy().reshape(-1))
            dense_scale_v.append(scale_v.detach().float().cpu().numpy().reshape(-1))
            p4_h, p4_w = scale_h.shape[-2:]
            image_h, image_w = image.shape[-2:]
            for target_index, image_index_tensor in enumerate(batch["batch_idx"]):
                image_index = int(image_index_tensor.item())
                box = batch["bboxes"][target_index]
                cx, cy, width, height = (float(value) for value in box)
                x = min(max(int(cx * p4_w), 0), p4_w - 1)
                y = min(max(int(cy * p4_h), 0), p4_h - 1)
                width_px, height_px = width * image_w, height * image_h
                area = width_px * height_px
                class_index = int(batch["cls"][target_index].item())
                records.append(
                    {
                        "class": f"D{[0, 10, 20, 40][class_index]:02d}",
                        "size": "small" if area < 32**2 else "medium" if area < 96**2 else "large",
                        "width": width_px,
                        "height": height_px,
                        "area": area,
                        "scale_h": float(scale_h[image_index, 0, y, x].cpu()),
                        "scale_v": float(scale_v[image_index, 0, y, x].cpu()),
                    }
                )

    groups: dict[str, list[dict[str, float | int | str]]] = defaultdict(list)
    for record in records:
        groups["all"].append(record)
        groups[f"class_{record['class']}"].append(record)
        groups[f"size_{record['size']}"].append(record)

    summaries = {}
    correlations = {}
    for name, rows in sorted(groups.items()):
        scale_h = [float(row["scale_h"]) for row in rows]
        scale_v = [float(row["scale_v"]) for row in rows]
        summaries[name] = {
            "scale_h": scale_stats(scale_h),
            "scale_v": scale_stats(scale_v),
            "scale_mean": scale_stats([(h + v) / 2 for h, v in zip(scale_h, scale_v)]),
        }
        correlations[name] = {
            "gt_width_vs_scale_h": spearman(
                [float(row["width"]) for row in rows], scale_h
            ),
            "gt_height_vs_scale_v": spearman(
                [float(row["height"]) for row in rows], scale_v
            ),
            "gt_area_vs_mean_scale": spearman(
                [float(row["area"]) for row in rows],
                [(h + v) / 2 for h, v in zip(scale_h, scale_v)],
            ),
        }
    all_rows = groups["all"]
    dense_h = np.concatenate(dense_scale_h).astype(np.float64, copy=False)
    dense_v = np.concatenate(dense_scale_v).astype(np.float64, copy=False)
    report = {
        "checkpoint": str(checkpoint),
        "data": str(data_yaml),
        "split": args.split,
        "test_sealed": True,
        "objects": len(records),
        "size_definition": "box area after letterbox to model input: small<32^2, medium<96^2",
        "dense_all_locations": {
            "scale_h": scale_stats(dense_h),
            "scale_v": scale_stats(dense_v),
            "scale_mean": scale_stats((dense_h + dense_v) / 2.0),
        },
        "groups": summaries,
        "spearman": {
            "gt_width_vs_scale_h": spearman(
                [float(row["width"]) for row in all_rows],
                [float(row["scale_h"]) for row in all_rows],
            ),
            "gt_height_vs_scale_v": spearman(
                [float(row["height"]) for row in all_rows],
                [float(row["scale_v"]) for row in all_rows],
            ),
            "gt_area_vs_mean_scale": spearman(
                [float(row["area"]) for row in all_rows],
                [
                    (float(row["scale_h"]) + float(row["scale_v"])) / 2
                    for row in all_rows
                ],
            ),
        },
        "spearman_by_group": correlations,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
