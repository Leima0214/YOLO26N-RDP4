"""Measure S2 gate specialization on matched Val positives without reading Test."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ultralytics import YOLO  # noqa: E402
from ultralytics.cfg import get_cfg  # noqa: E402
from ultralytics.data import build_dataloader, build_yolo_dataset  # noqa: E402
from ultralytics.data.utils import check_det_dataset  # noqa: E402
from ultralytics.nn.modules.head import ShapeSupervisedStripResidual  # noqa: E402


def summarize(aspects: np.ndarray, gates: np.ndarray) -> dict:
    difference = gates[:, 0] - gates[:, 1]
    horizontal, vertical = aspects > math.log(1.5), aspects < -math.log(1.5)
    rho = float(spearmanr(difference, aspects).statistic) if len(aspects) > 1 else float("nan")
    return {
        "positive_count": int(len(aspects)),
        "horizontal_count": int(horizontal.sum()),
        "vertical_count": int(vertical.sum()),
        "gate_h_mean": float(gates[:, 0].mean()),
        "gate_v_mean": float(gates[:, 1].mean()),
        "gate_h_std": float(gates[:, 0].std()),
        "gate_v_std": float(gates[:, 1].std()),
        "difference_std": float(difference.std()),
        "horizontal_mean_gh_minus_gv": float(difference[horizontal].mean()) if horizontal.any() else None,
        "vertical_mean_gv_minus_gh": float((-difference[vertical]).mean()) if vertical.any() else None,
        "spearman_difference_vs_log_aspect": rho,
        "gate_fraction_lt_0_05": float((gates < 0.05).mean()),
        "gate_fraction_gt_0_95": float((gates > 0.95).mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0, choices=(0,))
    parser.add_argument("--max-batches", type=int, default=0, help="0 uses the full Val split")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.device}" if args.device.isdigit() else args.device)
    cfg = get_cfg(overrides={"imgsz": args.imgsz, "batch": args.batch, "workers": 0, "data": str(args.data)})
    data = check_det_dataset(str(args.data.resolve()))
    dataset = build_yolo_dataset(cfg, data["val"], args.batch, data, mode="val", rect=True, stride=32)
    loader = build_dataloader(dataset, args.batch, 0, shuffle=False, rank=-1)
    model = YOLO(str(args.weights), task="detect").model.to(device).eval()
    model.args = cfg
    criterion = model.init_criterion()
    head = model.model[-1]
    head.training = True  # raw predictions; child BN modules remain in eval mode
    collected = {"one2many": {"aspect": [], "gates": []}, "one2one": {"aspect": [], "gates": []}}

    with torch.inference_mode():
        for batch_number, batch in enumerate(loader):
            if args.max_batches and batch_number >= args.max_batches:
                break
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = value.to(device, non_blocking=False)
            batch["img"] = batch["img"].float() / 255
            predictions = model(batch["img"])
            for name, branch_criterion in (("one2many", criterion.one2many), ("one2one", criterion.one2one)):
                branch = predictions[name]
                assigned = branch_criterion.get_assigned_targets_and_loss(branch, batch)[0]
                fg_mask, _, target_bboxes, _, _ = assigned
                logits = torch.cat([value.flatten(2) for value in branch["gate_logits"]], dim=2).permute(0, 2, 1)
                positive = fg_mask[:, : logits.shape[1]]
                if not positive.any():
                    continue
                boxes = target_bboxes[:, : logits.shape[1]][positive]
                width, height = (boxes[:, 2:] - boxes[:, :2]).clamp_min(1e-6).unbind(1)
                collected[name]["aspect"].append(torch.log(width / height).cpu())
                collected[name]["gates"].append(logits[positive].sigmoid().cpu())

    branch_stats = {}
    for name, values in collected.items():
        aspects = torch.cat(values["aspect"]).numpy()
        gates = torch.cat(values["gates"]).numpy()
        branch_stats[name] = summarize(aspects, gates)

    adapters = {
        name: {"gamma_h": float(module.gamma_h.detach().cpu()), "gamma_v": float(module.gamma_v.detach().cpu())}
        for name, module in head.named_modules()
        if isinstance(module, ShapeSupervisedStripResidual)
    }
    gates_pass = all(
        stats["horizontal_mean_gh_minus_gv"] is not None
        and stats["horizontal_mean_gh_minus_gv"] >= 0.05
        and stats["vertical_mean_gv_minus_gh"] is not None
        and stats["vertical_mean_gv_minus_gh"] >= 0.05
        and stats["spearman_difference_vs_log_aspect"] >= 0.20
        and stats["difference_std"] >= 0.02
        and stats["gate_fraction_lt_0_05"] < 0.95
        and stats["gate_fraction_gt_0_95"] < 0.95
        for stats in branch_stats.values()
    )
    gamma_pass = any(abs(value) > 1e-4 for pair in adapters.values() for value in pair.values())
    report = {
        "weights": str(args.weights.resolve()), "split": "val", "test_read": False,
        "branches": branch_stats, "adapters": adapters,
        "mechanism_gate": {"direction_specialization": gates_pass, "gamma_nonzero": gamma_pass, "go": gates_pass and gamma_pass},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
