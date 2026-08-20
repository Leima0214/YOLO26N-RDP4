#!/usr/bin/env python3
"""Shared, protocol-locked training entry for adaptive RoadSnake variants."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Literal

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import YOLO  # noqa: E402
from ultralytics.data.utils import check_det_dataset  # noqa: E402

Variant = Literal["sa", "mgsa"]
RECIPE = ROOT / "configs/experiments/japan4_cleanv3_adaptive_roadsnake_matched.yaml"
MODELS = {
    "sa": ROOT / "ultralytics/cfg/models/26/yolo26n-japan4-sa-roadsnake.yaml",
    "mgsa": ROOT / "ultralytics/cfg/models/26/yolo26n-japan4-mg-sa-roadsnake.yaml",
}
MGSA_UNLOCK_AP = 0.2499


def _resolve(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    return candidate.resolve() if candidate.is_absolute() else (ROOT / candidate).resolve()


def _load_recipe() -> dict[str, Any]:
    if not RECIPE.is_file():
        raise FileNotFoundError(RECIPE)
    recipe = yaml.safe_load(RECIPE.read_text(encoding="utf-8"))
    if not isinstance(recipe, dict):
        raise TypeError(f"Expected mapping in {RECIPE}")
    return recipe


def _sa_pruned_ap(metrics_path: Path) -> float:
    """Read the frozen Val COCO AP from the val-only adaptive evaluator output."""
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    rows = payload.get("paper_main_metrics") or payload.get("main_metrics") or []
    if not isinstance(rows, list):
        raise ValueError("SA metrics JSON has no list-valued main metrics")
    candidates = [
        row
        for row in rows
        if isinstance(row, dict)
        and row.get("split") == "val"
        and "pruned" in str(row.get("model", "")).lower()
    ]
    if len(candidates) != 1:
        raise ValueError(
            "MG-SA-RS unlock requires exactly one Val row whose model name contains 'pruned'"
        )
    value = candidates[0].get("coco_AP50_95")
    if value is None:
        raise ValueError("SA pruned Val row lacks coco_AP50_95")
    return float(value)


def _parser(variant: Variant) -> argparse.ArgumentParser:
    recipe = _load_recipe()
    parser = argparse.ArgumentParser(
        description=f"Fresh matched Japan4-cleanV3 training for {variant.upper()}-RS"
    )
    parser.add_argument("--data", default=str(recipe["data"]))
    parser.add_argument("--epochs", type=int, choices=(1, 30, 100), required=True)
    parser.add_argument("--imgsz", type=int, default=int(recipe["imgsz"]))
    parser.add_argument("--batch", type=int, default=int(recipe["batch"]))
    parser.add_argument("--device", default=str(recipe["device"]))
    parser.add_argument("--workers", type=int, default=int(recipe["workers"]))
    parser.add_argument("--seed", type=int, default=int(recipe["seed"]))
    parser.add_argument("--project", default=str(recipe["project"]))
    parser.add_argument("--name", required=True)
    parser.add_argument(
        "--amp", action=argparse.BooleanOptionalAction, default=bool(recipe["amp"])
    )
    parser.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=False
    )
    if variant == "mgsa":
        parser.add_argument(
            "--sa-pruned-metrics",
            type=Path,
            required=True,
            help="Val-only SA-RS pruned metrics.json proving coco AP >= 0.2499",
        )
    return parser


def run(variant: Variant) -> None:
    """Run one explicitly requested fresh experiment under the shared recipe."""
    recipe = _load_recipe()
    args = _parser(variant).parse_args()
    if args.resume:
        raise ValueError("Adaptive RoadSnake formal experiments must be fresh starts; --resume is forbidden")
    if args.imgsz != 640 or args.batch != 32 or args.seed != 42:
        raise ValueError("Formal protocol is frozen to imgsz=640, batch=32, seed=42")
    if not args.amp:
        raise ValueError("Formal protocol requires AMP; --no-amp is static-debug only and not accepted here")

    if variant == "mgsa":
        metrics_path = args.sa_pruned_metrics.expanduser().resolve()
        if not metrics_path.is_file():
            raise FileNotFoundError(metrics_path)
        sa_ap = _sa_pruned_ap(metrics_path)
        if sa_ap < MGSA_UNLOCK_AP:
            raise PermissionError(
                f"MG-SA-RS remains LOCKED: SA-RS pruned Val AP {sa_ap:.5f} < {MGSA_UNLOCK_AP:.4f}"
            )
        print(f"MG-SA-RS UNLOCKED by SA-RS pruned Val AP={sa_ap:.5f}")

    model_path = MODELS[variant]
    weights_path = _resolve(recipe["weights"])
    data_path = _resolve(args.data)
    project_path = _resolve(args.project)
    for path in (model_path, weights_path, data_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    data = check_det_dataset(str(data_path))
    for split in ("train", "val"):
        raw_paths = data[split]
        paths = [Path(p) for p in raw_paths] if isinstance(raw_paths, (list, tuple)) else [Path(raw_paths)]
        if not all(Path(path).exists() for path in paths):
            raise FileNotFoundError(f"Resolved {split} split is missing: {paths}")
    print("TEST_SEALED=PASS (training entry resolves and uses Train+Val only)")
    print(
        json.dumps(
            {
                "variant": variant,
                "model": str(model_path),
                "weights": str(weights_path),
                "data": str(data_path),
                "epochs": args.epochs,
                "imgsz": args.imgsz,
                "batch": args.batch,
                "workers": args.workers,
                "device": args.device,
                "seed": args.seed,
                "amp": args.amp,
                "fresh_start": True,
            },
            indent=2,
        )
    )

    model = YOLO(str(model_path), task="detect")
    model.load(str(weights_path))
    model.train(
        data=str(data_path),
        project=str(project_path),
        name=args.name,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        seed=args.seed,
        deterministic=bool(recipe["deterministic"]),
        optimizer=str(recipe["optimizer"]),
        cos_lr=bool(recipe["cos_lr"]),
        lr0=float(recipe["lr0"]),
        lrf=float(recipe["lrf"]),
        momentum=float(recipe["momentum"]),
        weight_decay=float(recipe["weight_decay"]),
        warmup_epochs=float(recipe["warmup_epochs"]),
        mosaic=float(recipe["mosaic"]),
        mixup=float(recipe["mixup"]),
        copy_paste=float(recipe["copy_paste"]),
        close_mosaic=int(recipe["close_mosaic"]),
        amp=args.amp,
        conf=float(recipe["conf"]),
        iou=float(recipe["iou"]),
        max_det=int(recipe["max_det"]),
        save_period=int(recipe["save_period"]),
        resume=False,
        exist_ok=bool(recipe["exist_ok"]),
    )
