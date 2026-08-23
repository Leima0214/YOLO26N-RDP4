"""Static, synthetic-batch, AMP, and initialization audit for RoadSnake-GBRG."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ultralytics import YOLO  # noqa: E402
from ultralytics.cfg import DEFAULT_CFG, get_cfg  # noqa: E402
from ultralytics.models.yolo.detect.train import DetectionTrainer  # noqa: E402
from ultralytics.nn.roadsnake import RoadSnakeDetect, RoadSnakeGBRGDetect  # noqa: E402
from ultralytics.utils.roadsnake_gbrg_loss import RoadSnakeGBRGE2ELoss, gbrg_region_targets  # noqa: E402
from ultralytics.utils.torch_utils import init_seeds  # noqa: E402

R1 = ROOT / "ultralytics/cfg/models/26/yolo26n-japan4-roadsnake-r1.yaml"
GBRG = ROOT / "ultralytics/cfg/models/26/yolo26n-japan4-roadsnake-gbrg.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=ROOT / "yolo26n.pt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--imgsz", type=int, default=160)
    parser.add_argument("--report", type=Path, default=ROOT / "reports/japan4_roadsnake_gbrg_static.json")
    return parser.parse_args()


def tensors(value: Any) -> list[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, dict):
        return sum((tensors(v) for v in value.values()), [])
    if isinstance(value, (tuple, list)):
        return sum((tensors(v) for v in value), [])
    return []


def build(path: Path, weights: Path, device: torch.device) -> YOLO:
    init_seeds(42, deterministic=True)
    model = YOLO(str(path), task="detect")
    model.load(str(weights))
    model.model.to(device)
    return model


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    for path in (args.weights, R1, GBRG):
        if not path.is_file():
            raise FileNotFoundError(path)

    r1 = build(R1, args.weights, device)
    gbrg = build(GBRG, args.weights, device)
    r1_head, gbrg_head = r1.model.model[-1], gbrg.model.model[-1]
    if not isinstance(r1_head, RoadSnakeDetect) or not isinstance(gbrg_head, RoadSnakeGBRGDetect):
        raise AssertionError("unexpected Detect head types")

    r1_state, gbrg_state = r1.model.state_dict(), gbrg.model.state_dict()
    shared = {name: value for name, value in r1_state.items() if name in gbrg_state and value.shape == gbrg_state[name].shape}
    changed = [name for name, value in shared.items() if not torch.equal(value, gbrg_state[name])]
    if changed:
        raise AssertionError(f"GBRG changed shared R1 initialization: {changed[:8]}")
    new_keys = sorted(set(gbrg_state) - set(r1_state))
    if new_keys != ["model.23.gbrg_region_head.bias", "model.23.gbrg_region_head.weight"]:
        raise AssertionError(f"unexpected GBRG-only tensors: {new_keys}")

    trainer = object.__new__(DetectionTrainer)
    trainer.data = {"nc": 4, "channels": 3}
    rebuilt = DetectionTrainer.get_model(trainer, cfg=str(GBRG), weights=gbrg.model, verbose=False).to(device)
    rebuilt_state = rebuilt.state_dict()
    rebuilt_compatible = {
        name: value for name, value in gbrg_state.items() if name in rebuilt_state and value.shape == rebuilt_state[name].shape
    }
    rebuilt_changed = [
        name for name, value in rebuilt_compatible.items() if not torch.equal(value, rebuilt_state[name])
    ]
    if rebuilt_changed:
        raise AssertionError(f"Trainer reconstruction changed GBRG tensors: {rebuilt_changed[:8]}")

    x = torch.rand(2, 3, args.imgsz, args.imgsz, device=device)
    r1.model.eval()
    gbrg.model.eval()
    with torch.no_grad():
        r1_output = tensors(r1.model(x))
        gbrg_output = tensors(gbrg.model(x))
    if len(r1_output) != len(gbrg_output):
        raise AssertionError("eval output structures differ")
    initial_max_error = max(float((a - b).abs().max()) for a, b in zip(r1_output, gbrg_output))
    if initial_max_error != 0.0:
        raise AssertionError(f"step-0 detection mismatch: {initial_max_error}")

    target, inside, clear = gbrg_region_targets(
        torch.tensor([0, 1], device=device),
        torch.tensor([[0.35, 0.45, 0.20, 0.10], [0.65, 0.55, 0.12, 0.25]], device=device),
        2,
        args.imgsz // 8,
        args.imgsz // 8,
    )
    if not target.isfinite().all() or not inside.any() or not clear.any() or (inside & clear).any():
        raise AssertionError("invalid GBRG region masks")

    gbrg.model.train()
    gbrg.model.zero_grad(set_to_none=True)
    gbrg.model.args = get_cfg(DEFAULT_CFG, {"box": 7.5, "cls": 0.5, "dfl": 1.5, "epochs": 30})
    gbrg.model.criterion = None
    batch = {
        "batch_idx": torch.tensor([0, 1], device=device),
        "cls": torch.tensor([[0.0], [3.0]], device=device),
        "bboxes": torch.tensor([[0.35, 0.45, 0.20, 0.10], [0.65, 0.55, 0.12, 0.25]], device=device),
    }
    criterion = gbrg.model.init_criterion()
    if not isinstance(criterion, RoadSnakeGBRGE2ELoss):
        raise AssertionError(f"unexpected criterion: {type(criterion).__name__}")
    amp_enabled = device.type == "cuda"
    with torch.autocast(device_type=device.type, enabled=amp_enabled):
        predictions = gbrg.model(x)
        loss, items = criterion(predictions, batch)
        total = loss.sum()
    total.backward()
    gradients = {
        name: float(parameter.grad.detach().float().norm())
        for name, parameter in gbrg_head.named_parameters()
        if parameter.grad is not None
    }
    if not gradients or not all(torch.isfinite(torch.tensor(value)) for value in gradients.values()):
        raise AssertionError(f"missing or non-finite head gradients: {gradients}")
    required_gradients = ("gbrg_region_head.weight", "gbrg_region_head.bias", "road_snake.gamma")
    missing = {name: gradients.get(name) for name in required_gradients if gradients.get(name, 0.0) <= 0}
    if missing:
        raise AssertionError(f"required GBRG/R1 gradients are inactive: {missing}")
    if not 0 < criterion.last_weighted_gradient_ratio < 0.08:
        raise AssertionError(f"gradient controller missed its safety band: {criterion.last_weighted_gradient_ratio}")

    report = {
        "device": str(device),
        "shared_r1_state_items": len(shared),
        "new_state_items": new_keys,
        "trainer_reconstruction_preserved_items": len(rebuilt_compatible),
        "initial_detection_max_abs_error": initial_max_error,
        "loss": [float(value) for value in loss.detach().cpu()],
        "loss_items": [float(value) for value in items.detach().cpu()],
        "raw_gradient_ratio": criterion.last_raw_gradient_ratio,
        "lambda": criterion.current_lambda,
        "weighted_gradient_ratio": criterion.last_weighted_gradient_ratio,
        "positive_pixels": criterion.last_positive_pixels,
        "background_effective_pixels": criterion.last_background_effective_pixels,
        "head_gradient_norms": gradients,
        "passed": True,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
