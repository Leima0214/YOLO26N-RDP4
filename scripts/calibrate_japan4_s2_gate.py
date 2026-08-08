"""Calibrate the single S2 gate-loss gain on fixed train batches without Val access."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ultralytics import YOLO  # noqa: E402
from ultralytics.cfg import get_cfg  # noqa: E402
from ultralytics.data import build_dataloader, build_yolo_dataset  # noqa: E402
from ultralytics.data.utils import check_det_dataset  # noqa: E402
from ultralytics.utils.loss import E2ELoss  # noqa: E402


def joint_l2(features: list[torch.Tensor]) -> float:
    return float(torch.sqrt(sum(feature.grad.detach().float().square().sum() for feature in features)).cpu())


def clear_gradients(model: torch.nn.Module, features: list[torch.Tensor]) -> None:
    model.zero_grad(set_to_none=True)
    for feature in features:
        feature.grad = None


def choose_gain(raw_ratios: np.ndarray) -> tuple[float, dict]:
    """Choose one gain satisfying the prespecified median/P75/P90 gradient gates."""
    median, p75, p90 = np.quantile(raw_ratios, (0.5, 0.75, 0.9))
    lower = 0.03 / median
    upper = min(0.05 / median, 0.08 / p75, 0.10 / p90)
    if lower > upper:
        raise RuntimeError(f"No lambda satisfies the fixed gradient gates: lower={lower}, upper={upper}")
    gain = min(max(0.04 / median, lower), upper)
    scaled = raw_ratios * gain
    audit = {
        "median": float(np.median(scaled)),
        "p75": float(np.quantile(scaled, 0.75)),
        "p90": float(np.quantile(scaled, 0.9)),
        "max": float(scaled.max()),
        "passes": bool(0.03 <= np.median(scaled) <= 0.05 and np.quantile(scaled, 0.75) < 0.08 and np.quantile(scaled, 0.9) < 0.10),
    }
    return float(gain), audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=ROOT / "yolo26n.pt")
    parser.add_argument("--model", type=Path, default=ROOT / "ultralytics/cfg/models/26/yolo26n-japan4-s2-shape-strip.yaml")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-batches", type=int, default=12, choices=range(8, 13))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0, choices=(0,))
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cfg = get_cfg(
        overrides={
            "data": str(args.data), "imgsz": args.imgsz, "batch": args.batch, "workers": 0,
            "seed": args.seed, "deterministic": True, "epochs": 5, "mosaic": 1.0,
            "mixup": 0.0, "copy_paste": 0.0, "close_mosaic": 10,
        }
    )
    data = check_det_dataset(str(args.data.resolve()))
    dataset = build_yolo_dataset(cfg, data["train"], args.batch, data, mode="train", stride=32)
    loader = build_dataloader(dataset, args.batch, 0, shuffle=False, rank=-1)
    device = torch.device(f"cuda:{args.device}" if args.device.isdigit() else args.device)
    model = YOLO(str(args.model), task="detect", verbose=False).load(str(args.weights)).model.to(device)
    model.args = cfg
    model.train()
    gate_criterion = model.init_criterion()
    detection_criterion = E2ELoss(model)
    head = model.model[-1]
    records = []

    for batch_number, batch in enumerate(loader):
        if batch_number >= args.num_batches:
            break
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(device, non_blocking=False)
        batch["img"] = batch["img"].float() / 255
        shared = []

        def retain(_, inputs):
            shared.extend(inputs[0][:2])
            for feature in shared:
                feature.retain_grad()

        hook = head.register_forward_pre_hook(retain)
        predictions = model(batch["img"])
        hook.remove()
        detection, _ = detection_criterion(predictions, batch)
        detection.sum().backward(retain_graph=True)
        detection_norm = joint_l2(shared)
        clear_gradients(model, shared)

        many, one = predictions["one2many"], predictions["one2one"]
        assigned_many = gate_criterion.one2many.get_assigned_targets_and_loss(many, batch)[0]
        assigned_one = gate_criterion.one2one.get_assigned_targets_and_loss(one, batch)[0]
        gate = gate_criterion.gate_loss(many, assigned_many) * gate_criterion.o2m
        gate = gate + gate_criterion.gate_loss(one, assigned_one) * gate_criterion.o2o
        (gate * batch["img"].shape[0]).backward()
        gate_norm = joint_l2(shared)
        clear_gradients(model, shared)
        ratio = gate_norm / max(detection_norm, 1e-12)
        records.append({"batch": batch_number, "detection_grad_l2": detection_norm, "raw_gate_grad_l2": gate_norm, "raw_ratio": ratio})
        print(f"batch {batch_number}: det={detection_norm:.6g} gate={gate_norm:.6g} raw_ratio={ratio:.4%}", flush=True)

    raw = np.asarray([record["raw_ratio"] for record in records])
    gain, scaled = choose_gain(raw)
    report = {
        "model": str(args.model.resolve()), "weights": str(args.weights.resolve()), "split": "train",
        "workers": 0, "num_batches": len(records), "seed": args.seed, "selection_used_val": False,
        "raw_ratio": {"median": float(np.median(raw)), "p75": float(np.quantile(raw, 0.75)), "p90": float(np.quantile(raw, 0.9)), "max": float(raw.max())},
        "selected_gate_loss_gain": gain, "scaled_ratio": scaled, "batches": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
