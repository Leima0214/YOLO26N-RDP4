#!/usr/bin/env python3
"""Calibrate one frozen lambda_schm from fixed real Japan4 train batches without reading Val/Test."""

from __future__ import annotations

import argparse
import json
import random
import sys
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


MODELS = {
    "schm": ROOT / "ultralytics/cfg/models/26/yolo26n-japan4-schm.yaml",
    "rs-schm": ROOT / "ultralytics/cfg/models/26/yolo26n-japan4-rs-schm.yaml",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=tuple(MODELS), required=True)
    parser.add_argument("--weights", type=Path, default=ROOT / "yolo26n.pt")
    parser.add_argument(
        "--data", type=Path, default=ROOT / "configs/japan4_clean_v3_remote.yaml"
    )
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--batches", type=int, default=10)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 8 <= args.batches <= 12:
        raise ValueError("Calibration is locked to 8-12 fixed train batches")
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    device = torch.device(
        f"cuda:{args.device}" if args.device.isdigit() else args.device
    )

    data = check_det_dataset(str(args.data.resolve()))
    cfg = get_cfg(
        DEFAULT_CFG,
        {
            "mode": "train",
            "task": "detect",
            "imgsz": args.imgsz,
            "batch": args.batch,
            "workers": 0,
            "rect": False,
            "cache": False,
            "mosaic": 1.0,
            "mixup": 0.0,
            "copy_paste": 0.0,
            "close_mosaic": 10,
            "seed": 42,
            "deterministic": True,
            "box": 7.5,
            "cls": 0.5,
            "dfl": 1.5,
            "epochs": 100,
        },
    )
    dataset = build_yolo_dataset(
        cfg, data["train"], args.batch, data, mode="train", rect=False, stride=32
    )
    loader = build_dataloader(
        dataset, args.batch, 0, shuffle=False, rank=-1, pin_memory=False
    )

    detector = YOLO(str(MODELS[args.variant]), task="detect", verbose=False)
    detector.load(str(args.weights.resolve()))
    model = detector.model.to(device)
    model.args = cfg
    model.criterion = None
    model.train()
    # Keep the checkpoint's BN statistics fixed; calibration measures only loss-gradient ratios.
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()

    rows = []
    for index, batch in enumerate(loader):
        if index >= args.batches:
            break
        batch = {
            key: value.to(device, non_blocking=False)
            if isinstance(value, torch.Tensor)
            else value
            for key, value in batch.items()
        }
        batch["img"] = batch["img"].float().div(255.0)
        model.zero_grad(set_to_none=True)
        predictions = model(batch["img"])
        if model.criterion is not None:
            model.criterion.lambda_schm = 1.0
            model.criterion._measure_gradient = True
        loss, components = model.loss(batch, predictions)
        criterion = model.criterion
        ratio = criterion.last_batch_stats.get("gradient_ratio")
        if ratio is None or not np.isfinite(ratio):
            raise AssertionError(
                f"Missing/non-finite gradient ratio at calibration batch {index}"
            )
        rows.append(
            {
                "batch": index,
                "objects": int(batch["cls"].numel()),
                "gradient_ratio_lambda1": float(ratio),
                "harvest_gt_count": criterion.last_batch_stats["harvest_gt_count"],
                "harvest_ratio": criterion.last_batch_stats["harvest_gt_count"]
                / max(criterion.last_batch_stats["total_gt"], 1),
                "same_index_ratio": criterion.last_batch_stats["same_index_count"]
                / max(criterion.last_batch_stats["comparable_gt"], 1),
                "illegal_harvest_count": criterion.last_batch_stats[
                    "illegal_harvest_count"
                ],
                "loss_components_lambda1": [
                    float(value) for value in components.detach().cpu()
                ],
            }
        )

    if len(rows) != args.batches:
        raise AssertionError(
            f"Expected {args.batches} calibration batches, got {len(rows)}"
        )
    ratios = torch.tensor(
        [row["gradient_ratio_lambda1"] for row in rows], dtype=torch.float64
    )
    median = float(ratios.median())
    p90 = float(torch.quantile(ratios, 0.9))
    if median <= 0.10:
        calibrated = 1.0  # Never amplify a naturally conservative auxiliary gradient.
    else:
        calibrated = min(0.075 / median, 0.095 / max(p90, 1e-12), 1.0)
    calibrated = round(float(calibrated), 6)
    report = {
        "variant": args.variant,
        "data_split": "train-only",
        "test_sealed": True,
        "workers": 0,
        "seed": 42,
        "batch_size": args.batch,
        "batches": args.batches,
        "lambda1_gradient_ratio": {
            "median": median,
            "p90": p90,
            "values": ratios.tolist(),
        },
        "recommended_lambda_schm": calibrated,
        "projected_gradient_ratio": {
            "median": median * calibrated,
            "p90": p90 * calibrated,
        },
        "rows": rows,
    }
    if any(row["illegal_harvest_count"] for row in rows):
        raise AssertionError("illegal_harvest_count must remain zero")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
