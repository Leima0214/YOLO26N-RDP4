"""Fresh-start Japan4-cleanV3 30E training entry for RoadSnake-R1."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ultralytics import YOLO  # noqa: E402


MODEL = ROOT / "ultralytics/cfg/models/26/yolo26n-japan4-roadsnake-r1.yaml"
WEIGHTS = ROOT / "yolo26n.pt"
DATA = ROOT / "configs/japan4_clean_v3_remote.yaml"
RUN_NAME = "yolo26n-japan4-roadsnake-r1_cleanv3_30e_seed42_20260817"


def main() -> None:
    for path in (MODEL, WEIGHTS, DATA):
        if not path.is_file():
            raise FileNotFoundError(path)

    model = YOLO(str(MODEL), task="detect")
    model.load(str(WEIGHTS))
    model.train(
        data=str(DATA),
        project=str(ROOT / "runs/paper1_japan4_clean"),
        name=RUN_NAME,
        epochs=30,
        imgsz=640,
        batch=32,
        device="0",
        workers=8,
        seed=42,
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
        save_period=5,
        resume=False,
        exist_ok=False,
    )


if __name__ == "__main__":
    main()
