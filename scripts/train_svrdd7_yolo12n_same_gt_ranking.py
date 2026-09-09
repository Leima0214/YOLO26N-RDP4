"""Train the matched YOLO12n 30E TAL-positive same-GT residual-ranking experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT / "scripts"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from rs_mid_bootstrap import guard_optional_visualization_imports

OPTIONAL_IMPORT_GUARD_USED = guard_optional_visualization_imports()

import torch
import yaml

from ultralytics import YOLO
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.utils.yolo12_same_gt_ranking import (
    TALPositiveSameGTRankingLoss,
    YOLO12SameGTRankingModel,
)

YOLO12N_SHA256 = "419ff3dca37d69bacc93a50fa0c186a1c6f9fe62fae0f108b0872829689e9ca6"
NAMES = ["LC", "TC", "AC", "P", "MC", "LP", "TP"]


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class YOLO12SameGTRankingTrainer(DetectionTrainer):
    """Build the native YOLO12 graph with the isolated ranking criterion."""

    ranking_spec = {"rank_weight": 0.10, "min_iou_gap": 0.05, "temperature": 1.0}

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = YOLO12SameGTRankingModel(
            deepcopy(cfg), nc=self.data["nc"], ch=self.data["channels"], verbose=verbose
        )
        if weights is not None:
            model.load(weights)
        model.same_gt_ranking_spec = deepcopy(self.ranking_spec)
        return model


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="YOLO12N_TAL_SAMEGT_RANK_30E_seed42")
    parser.add_argument("--weights", type=Path, default=ROOT / "yolo12n.pt")
    parser.add_argument("--data", type=Path, default=ROOT / "configs/svrdd7_rs_mid_remote.yaml")
    parser.add_argument("--project", type=Path, default=ROOT / "runs/paper1_svrdd7_yolo12_ranking")
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.name):
        parser.error("--name must be one simple directory name")
    if args.smoke and not args.name.startswith("SMOKE_"):
        parser.error("Smoke names must start with SMOKE_")
    return args


def validate_data(path: Path) -> None:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    names = cfg.get("names", [])
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]
    if cfg.get("nc") != 7 or names != NAMES or not cfg.get("train") or not cfg.get("val"):
        raise ValueError("Expected the frozen seven-class SVRDD Train/Val configuration")


def main(argv=None):
    args = parse_args(argv)
    weights = args.weights.expanduser().resolve()
    data = args.data.expanduser().resolve()
    project = args.project.expanduser().resolve()
    run_dir = project / args.name
    meta_dir = ROOT / "runtime_meta" / args.name
    for path in (weights, data):
        if not path.is_file():
            raise FileNotFoundError(path)
    if file_hash(weights) != YOLO12N_SHA256:
        raise ValueError("The experiment requires the frozen official generic yolo12n.pt")
    validate_data(data)
    if run_dir.exists() or meta_dir.exists():
        raise FileExistsError(f"Fresh outputs required; refusing to reuse {run_dir} or {meta_dir}")

    protocol = {
        "epochs": 1 if args.smoke else 30,
        "batch": 2 if args.smoke else 32,
        "imgsz": 128 if args.smoke else 640,
        "workers": 0 if args.smoke else 8,
        "fraction": 4 / 6000 if args.smoke else 1.0,
        "mosaic": 0.0 if args.smoke else 1.0,
        "close_mosaic": 0 if args.smoke else 10,
        "plots": False if args.smoke else True,
        "save_period": 1 if args.smoke else 5,
    }
    source_paths = [
        Path(__file__),
        ROOT / "ultralytics/utils/yolo12_same_gt_ranking.py",
        ROOT / "ultralytics/utils/loss.py",
        ROOT / "ultralytics/utils/tal.py",
        ROOT / "ultralytics/nn/tasks.py",
        ROOT / "ultralytics/nn/modules/head.py",
        ROOT / "ultralytics/cfg/models/12/yolo12.yaml",
    ]
    source_hashes = {str(p.relative_to(ROOT)): file_hash(p) for p in source_paths}
    meta_dir.mkdir(parents=True)
    for path in source_paths:
        target = meta_dir / "source" / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    shutil.copy2(data, meta_dir / "data_snapshot.yaml")

    resolved = {
        "experiment": "YOLO12n + TAL-positive same-GT residual ranking",
        "baseline_run": "/root/YOLO26N-RDP4/runs/paper1_svrdd7_official/YOLO12N_OFFICIAL_SVRDD7_30E_seed42",
        "baseline_unified_no_voting_AP": 0.3833842063296054,
        "ranking": YOLO12SameGTRankingTrainer.ranking_spec,
        "training": {**protocol, "seed": args.seed, "optimizer": "auto", "amp": True},
        "inference_changes": False,
        "additional_parameters": 0,
        "test_sealed": True,
        "weights": str(weights),
        "weights_sha256": file_hash(weights),
        "data": str(data),
        "data_sha256": file_hash(data),
        "source_sha256": source_hashes,
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=False
        ).stdout.strip(),
        "command": sys.argv,
        "optional_import_guard_used": OPTIONAL_IMPORT_GUARD_USED,
    }
    (meta_dir / "resolved_run.json").write_text(json.dumps(resolved, indent=2) + "\n", encoding="utf-8")

    model = YOLO(str(weights))

    def audit_setup(trainer):
        criterion = trainer.model.init_criterion()
        if not isinstance(criterion, TALPositiveSameGTRankingLoss):
            raise AssertionError(f"Unexpected criterion {type(criterion).__name__}")
        trainer.model.criterion = criterion
        head = trainer.model.model[-1]
        if head.__class__.__name__ != "Detect" or head.nc != 7 or getattr(head, "end2end", False):
            raise AssertionError("Official YOLO12 non-end2end Detect was not preserved")
        expected_images = 4 if args.smoke else 6000
        if len(trainer.train_loader.dataset) != expected_images:
            raise AssertionError(f"Expected {expected_images} training images")
        audit = {
            "passed": True,
            "model_class": type(trainer.model).__name__,
            "head_class": type(head).__name__,
            "criterion": type(criterion).__name__,
            "parameters": sum(p.numel() for p in trainer.model.parameters()),
            "ranking": trainer.ranking_spec,
            "inference_changes": False,
            "additional_parameters": 0,
        }
        (Path(trainer.save_dir) / "initialization_audit.json").write_text(
            json.dumps(audit, indent=2) + "\n", encoding="utf-8"
        )

    def record_ranking(trainer):
        criterion = getattr(trainer.model, "criterion", None)
        if not isinstance(criterion, TALPositiveSameGTRankingLoss):
            raise AssertionError("Ranking criterion disappeared during training")
        payload = {"epoch": int(trainer.epoch + 1), **criterion.pop_epoch_stats()}
        path = Path(trainer.save_dir) / "same_gt_ranking_stats.jsonl"
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload) + "\n")
        print("SAME_GT_RANKING " + json.dumps(payload), flush=True)

    model.add_callback("on_pretrain_routine_end", audit_setup)
    model.add_callback("on_train_epoch_end", record_ranking)
    exit_code = 1
    try:
        model.train(
            trainer=YOLO12SameGTRankingTrainer,
            data=str(data),
            epochs=protocol["epochs"],
            batch=protocol["batch"],
            imgsz=protocol["imgsz"],
            workers=protocol["workers"],
            fraction=protocol["fraction"],
            mosaic=protocol["mosaic"],
            close_mosaic=protocol["close_mosaic"],
            plots=protocol["plots"],
            save_period=protocol["save_period"],
            project=str(project),
            name=args.name,
            device=args.device,
            seed=args.seed,
            deterministic=True,
            optimizer="auto",
            patience=1000000000,
            amp=True,
            val=True,
            split="val",
            conf=0.001,
            iou=0.7,
            max_det=300,
            pretrained=True,
            exist_ok=False,
        )
        if not (run_dir / "weights/best.pt").is_file() or not (run_dir / "weights/last.pt").is_file():
            raise FileNotFoundError("Training ended without best.pt and last.pt")
        if len((run_dir / "same_gt_ranking_stats.jsonl").read_text().splitlines()) != protocol["epochs"]:
            raise RuntimeError("Ranking diagnostics do not cover the full epoch budget")
        if {str(p.relative_to(ROOT)): file_hash(p) for p in source_paths} != source_hashes:
            raise RuntimeError("Experiment source changed during training")
        exit_code = 0
    finally:
        (meta_dir / "train.exit_code.txt").write_text(f"{exit_code}\n", encoding="utf-8")
        if run_dir.exists():
            shutil.copytree(meta_dir, run_dir / "runtime_meta", dirs_exist_ok=True)


if __name__ == "__main__":
    main()
