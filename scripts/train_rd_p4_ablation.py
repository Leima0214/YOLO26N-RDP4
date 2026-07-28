"""Run the matched Japan7 B0/R1 screen for the clean YOLO26n RD-P4 candidate."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) in sys.path:
    sys.path.remove(str(ROOT))
sys.path.insert(0, str(ROOT))

from ultralytics import YOLO
from ultralytics.nn.rd_adapter import RDP4Stage

BASELINE_MODEL = ROOT / "ultralytics/cfg/models/26/yolo26.yaml"
RD_MODEL = ROOT / "ultralytics/cfg/models/26/yolo26n-rd-p4.yaml"
PROJECT = ROOT / "runs/paper1_rd_p4"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("B0", "R1"), required=True)
    parser.add_argument("--data", default="configs/japan7_remote.yaml")
    parser.add_argument("--weights", default="yolo26n.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--name", default=None)
    return parser.parse_args()


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def build_model(stage: str, weights: Path) -> YOLO:
    yaml = BASELINE_MODEL if stage == "B0" else RD_MODEL
    assert yaml.is_file(), yaml
    assert weights.is_file(), weights
    model = YOLO(str(yaml), task="detect")
    model.load(str(weights))
    if stage == "R1":
        rd_stages = [module for module in model.model.modules() if isinstance(module, RDP4Stage)]
        assert len(rd_stages) == 1 and rd_stages[0].rd.gamma.item() == 0.0, "RD-P4 construction failed"
        assert any(name.startswith("model.6.rd.") for name in model.model.state_dict()), "RD-P4 state is missing"
    return model


def main() -> None:
    args = parse_args()
    data, weights = resolve(args.data), resolve(args.weights)
    assert data.is_file(), data
    assert args.epochs > 0 and args.batch > 0, (args.epochs, args.batch)
    model = build_model(args.stage, weights)
    run_name = args.name or f"rdp4_{args.stage.lower()}_japan7_{args.epochs}e_seed{args.seed}"
    model.train(
        data=str(data),
        project=str(PROJECT),
        name=run_name,
        epochs=args.epochs,
        imgsz=640,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        seed=args.seed,
        deterministic=True,
        optimizer="auto",
        cos_lr=False,
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        mosaic=1.0,
        mixup=0.0,
        copy_paste=0.0,
        close_mosaic=10,
        amp=True,
        conf=0.001,
        iou=0.7,
        max_det=300,
    )


if __name__ == "__main__":
    main()
