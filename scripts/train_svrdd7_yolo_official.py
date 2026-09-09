"""Run an official Ultralytics YOLO11/YOLO12 checkpoint on SVRDD7 unchanged."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for directory in (ROOT, SCRIPTS):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

NAMES = ["LC", "TC", "AC", "P", "MC", "LP", "TP"]
KNOWN_SHA256 = {
    "yolo11n.pt": "0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="yolo11n.pt",
                        help="Official local checkpoint, e.g. yolo11n.pt or yolo12n.pt")
    parser.add_argument("--name", required=True)
    parser.add_argument("--data", type=Path, default=ROOT / "configs/svrdd7_rs_mid_remote.yaml")
    parser.add_argument("--project", type=Path, default=ROOT / "runs/paper1_svrdd7_official")
    parser.add_argument("--epochs", type=int, choices=(30, 100), default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="0")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if not args.name or any(token in args.name for token in ("/", "\\")):
        parser.error("--name must be a single output directory name")
    if args.seed < 0 or "," in args.device or args.device not in {"cpu", "0"}:
        parser.error("Use a nonnegative seed and one device (0 or cpu)")
    return args


def validate_data(path: Path) -> None:
    import yaml

    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    names = config.get("names", [])
    if isinstance(names, dict):
        names = [names[key] for key in sorted(names)]
    if config.get("nc") != 7 or names != NAMES or not config.get("train") or not config.get("val"):
        raise ValueError("Expected the original seven-class SVRDD7 train/val configuration")
    if not config.get("test"):
        raise ValueError("The SVRDD7 YAML must retain the sealed Test path")


def resolved(args: argparse.Namespace, data: Path, weights: Path) -> dict:
    return {
        "model_family": Path(args.model).stem,
        "model_checkpoint": str(weights),
        "model_checkpoint_sha256": sha256(weights),
        "official_checkpoint_unchanged": True,
        "training_entry": "ultralytics.YOLO.train",
        "data": str(data),
        "data_sha256": sha256(data),
        "protocol": {
            "epochs": args.epochs,
            "imgsz": 640,
            "batch": 32,
            "workers": 8,
            "seed": args.seed,
            "deterministic": True,
            "optimizer": "auto",
            "cos_lr": False,
            "lr0": 0.01,
            "lrf": 0.01,
            "momentum": 0.937,
            "weight_decay": 0.0005,
            "warmup_epochs": 3.0,
            "mosaic": 1.0,
            "mixup": 0.0,
            "copy_paste": 0.0,
            "close_mosaic": 10,
            "amp": True,
            "val": True,
            "split": "val",
            "conf": 0.001,
            "iou": 0.7,
            "max_det": 300,
            "save_period": 5,
        },
        "head_mode": "official native one-to-many Detect",
        "post_training_evaluation": "manual; no Test evaluation",
        "test_sealed": True,
    }


def main(argv=None) -> None:
    args = parse_args(argv)
    data = args.data.expanduser().resolve()
    weights = (ROOT / args.model).resolve() if not Path(args.model).is_absolute() else Path(args.model).resolve()
    project = args.project.expanduser().resolve()
    if args.dry_run:
        print(json.dumps({"data": str(data), "weights": str(weights), "project": str(project),
                          "epochs": args.epochs, "seed": args.seed}, indent=2))
        return
    if not data.is_file() or not weights.is_file():
        raise FileNotFoundError(f"Missing data or official checkpoint: {data}, {weights}")
    validate_data(data)
    expected = KNOWN_SHA256.get(weights.name)
    digest = sha256(weights)
    if expected and digest != expected:
        raise ValueError(f"Official checkpoint SHA256 mismatch for {weights.name}")
    run_dir = project / args.name
    metadata_dir = ROOT / "runtime_meta" / args.name
    if run_dir.exists() or metadata_dir.exists():
        raise FileExistsError(f"Fresh output required; refusing {run_dir} or {metadata_dir}")
    metadata_dir.mkdir(parents=True)
    metadata = resolved(args, data, weights)
    metadata.update({"command": sys.argv, "python": sys.executable, "pid": os.getpid()})
    (metadata_dir / "resolved_run.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    (metadata_dir / "data_snapshot.yaml").write_text(data.read_text(encoding="utf-8"), encoding="utf-8")
    train_rc = 1
    try:
        # The guard only handles the remote optional pandas/seaborn import
        # problem; it does not change the model, checkpoint, or trainer.
        from rs_mid_bootstrap import guard_optional_visualization_imports
        guard_optional_visualization_imports()
        from ultralytics import YOLO

        model = YOLO(str(weights), task="detect")
        head = model.model.model[-1]
        if type(head).__name__ != "Detect" or getattr(head, "end2end", False):
            raise ValueError("The official checkpoint must remain the native one-to-many Detect")
        metadata["checkpoint_head"] = {
            "type": type(head).__name__,
            "end2end": bool(getattr(head, "end2end", False)),
            "nc_before_data_override": int(head.nc),
        }
        (metadata_dir / "resolved_run.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        model.train(
            data=str(data), project=str(project), name=args.name,
            epochs=args.epochs, imgsz=640, batch=32, device=args.device, workers=8,
            seed=args.seed, deterministic=True, optimizer="auto", cos_lr=False,
            lr0=0.01, lrf=0.01, momentum=0.937, weight_decay=0.0005,
            warmup_epochs=3.0, mosaic=1.0, mixup=0.0, copy_paste=0.0,
            close_mosaic=10, amp=True, val=True, split="val", conf=0.001,
            iou=0.7, max_det=300, save_period=5,
        )
        train_rc = 0
        print(json.dumps({"official_training_completed": True, "run": str(run_dir),
                          "best": str(run_dir / "weights/best.pt")}, ensure_ascii=False), flush=True)
    finally:
        (metadata_dir / "train.exit_code.txt").write_text(f"{train_rc}\n", encoding="utf-8")
        (metadata_dir / "pipeline.exit_code.txt").write_text(f"{train_rc}\n", encoding="utf-8")


if __name__ == "__main__":
    main()
