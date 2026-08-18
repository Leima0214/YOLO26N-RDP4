#!/usr/bin/env python3
"""Fresh-start matched Japan4-cleanV3 runner for SCHM and RS-SCHM."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import YOLO  # noqa: E402
from ultralytics.models.yolo.detect.schm_train import SCHMDetectionTrainer  # noqa: E402


CONFIGS = {
    "schm": ROOT / "configs/experiments/japan4_cleanv3_schm_100e.yaml",
    "rs-schm": ROOT / "configs/experiments/japan4_cleanv3_rs_schm_100e.yaml",
}
TRAIN_KEYS = {
    "epochs",
    "imgsz",
    "batch",
    "workers",
    "device",
    "seed",
    "deterministic",
    "optimizer",
    "cos_lr",
    "lr0",
    "lrf",
    "momentum",
    "weight_decay",
    "warmup_epochs",
    "mosaic",
    "mixup",
    "copy_paste",
    "close_mosaic",
    "amp",
    "conf",
    "iou",
    "max_det",
    "save_period",
    "resume",
    "exist_ok",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True, choices=tuple(CONFIGS))
    parser.add_argument("--config", type=Path)
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Training smoke only; formal scripts never use this",
    )
    parser.add_argument(
        "--smoke", action="store_true", help="Run a fresh 1E engineering smoke only"
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_repo_path(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def load_config(path: Path, variant: str) -> dict:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if config.get("variant") != variant:
        raise ValueError(
            f"Config variant {config.get('variant')!r} does not match --variant {variant!r}"
        )
    if (
        config.get("epochs") != 100
        or config.get("resume") is not False
        or config.get("exist_ok") is not False
    ):
        raise ValueError(
            "Formal SCHM-family runs must be fresh-start 100E and refuse overwrite"
        )
    if config.get("lambda_schm", 0) <= 0:
        raise ValueError("lambda_schm must be frozen to a positive calibrated value")
    return config


def verify_protocol(config: dict) -> tuple[Path, Path, Path, Path]:
    model_path = resolve_repo_path(config["model"])
    weights_path = resolve_repo_path(config["weights"])
    data_path = resolve_repo_path(config["data"])
    project_path = resolve_repo_path(config["project"])
    for path in (model_path, weights_path, data_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    data_cfg = yaml.safe_load(data_path.read_text(encoding="utf-8"))
    if "train" not in data_cfg or "val" not in data_cfg:
        raise ValueError("Japan4 data YAML must define train and val")
    root = Path(data_cfg["path"])
    for key in ("train", "val"):
        split = root / data_cfg[key]
        if not split.is_dir():
            raise FileNotFoundError(split)
    # Deliberately do not stat, glob, load, or evaluate the test path.
    print(
        "TEST_SEALED=PASS (runner resolves train+val only; evaluator receives --splits val)"
    )

    model_cfg = yaml.safe_load(model_path.read_text(encoding="utf-8"))
    if float(model_cfg.get("lambda_schm", -1)) != float(config["lambda_schm"]):
        raise ValueError(
            "Experiment and model YAML lambda_schm differ; calibration must be frozen in both"
        )
    return model_path, weights_path, data_path, project_path


def unique_name(project: Path, base: str) -> str:
    if not (project / base).exists():
        return base
    return f"{base}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def best_epoch(csv_path: Path) -> tuple[int | None, float | None]:
    if not csv_path.is_file():
        return None, None
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    keys = [key for key in rows[0] if "mAP50-95" in key] if rows else []
    if not keys:
        return None, None
    key = keys[0]
    best = max(rows, key=lambda row: float(row[key]))
    return int(float(best["epoch"])), float(best[key])


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def unified_val(
    checkpoint: Path, data: Path, output: Path, config: dict, name: str
) -> None:
    run(
        [
            sys.executable,
            "-u",
            str(ROOT / "scripts/evaluate_japan4_paper_metrics.py"),
            "--checkpoint",
            f"{name}={checkpoint}",
            "--data",
            str(data),
            "--output",
            str(output),
            "--splits",
            "val",
            "--imgsz",
            str(config["imgsz"]),
            "--batch",
            str(config["batch"]),
            "--workers",
            str(config["workers"]),
            "--device",
            str(config["device"]),
            "--conf",
            str(config["conf"]),
            "--iou",
            str(config["iou"]),
            "--max-det",
            str(config["max_det"]),
        ]
    )


def main() -> None:
    args = parse_args()
    config_path = (args.config or CONFIGS[args.variant]).resolve()
    config = load_config(config_path, args.variant)
    model_path, weights_path, data_path, project_path = verify_protocol(config)
    if args.smoke:
        project_path = ROOT / "runs/smoke_schm_family"
        run_name = unique_name(project_path, f"{config['name']}_SMOKE1E")
    else:
        run_name = unique_name(project_path, config["name"])
    print(
        json.dumps(
            {
                "git": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
                ).strip(),
                "config_path": str(config_path),
                "config": config,
                "resolved_run_name": run_name,
            },
            indent=2,
        )
    )

    model = YOLO(str(model_path), task="detect")
    model.load(str(weights_path))
    train_args = {key: config[key] for key in TRAIN_KEYS}
    if args.smoke:
        train_args.update({"epochs": 1, "save_period": -1})
    model.train(
        trainer=SCHMDetectionTrainer,
        data=str(data_path),
        project=str(project_path),
        name=run_name,
        **train_args,
    )
    save_dir = Path(model.trainer.save_dir).resolve()
    best = save_dir / "weights/best.pt"
    if not best.is_file():
        raise FileNotFoundError(best)
    epoch, fitness = best_epoch(save_dir / "results.csv")
    summary = {
        "variant": args.variant,
        "run": str(save_dir),
        "best": str(best),
        "best_sha256": sha256(best),
        "best_epoch": epoch,
        "best_map50_95": fitness,
        "lambda_schm": config["lambda_schm"],
        "test_sealed": True,
    }

    if not args.skip_eval:
        unified_val(best, data_path, save_dir / "unified_val", config, args.variant)

    if args.variant == "rs-schm":
        pruned = save_dir / "weights/best_pruned_native.pt"
        prune_report = save_dir / "best_pruned_native_audit.json"
        run(
            [
                sys.executable,
                "-u",
                str(ROOT / "scripts/prune_roadsnake_t1.py"),
                "--source",
                str(best),
                "--output",
                str(pruned),
                "--report",
                str(prune_report),
                "--device",
                str(config["device"]),
                "--imgsz",
                str(config["imgsz"]),
            ]
        )
        summary.update(
            {"best_pruned_native": str(pruned), "best_pruned_sha256": sha256(pruned)}
        )
        if not args.skip_eval:
            unified_val(
                pruned,
                data_path,
                save_dir / "unified_val_pruned",
                config,
                "rs-schm-pruned",
            )
        export = YOLO(str(pruned), task="detect").export(
            format="onnx",
            imgsz=config["imgsz"],
            batch=1,
            device=config["device"],
            opset=17,
            simplify=False,
        )
        summary["pruned_onnx"] = str(export)

    (save_dir / "schm_experiment_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
