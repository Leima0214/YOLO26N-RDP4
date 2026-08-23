"""Fresh-start Japan4-cleanV3 30E RoadSnake-HNR signal experiment."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ultralytics import YOLO  # noqa: E402
from ultralytics.nn.roadsnake import RoadSnakeHNRDetect  # noqa: E402
from ultralytics.utils.torch_utils import init_seeds  # noqa: E402

DATA = ROOT / "configs/japan4_clean_v3_remote.yaml"
WEIGHTS = ROOT / "yolo26n.pt"
MODEL = ROOT / "ultralytics/cfg/models/26/yolo26n-japan4-roadsnake-hnr.yaml"
PROJECT = ROOT / "runs/paper1_japan4_clean"
RUN_NAME = "yolo26n-japan4-roadsnake-hnr_cleanv3_30e_seed42_20260823"
SEED = 42


def tensor_map_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def main() -> None:
    for path in (DATA, WEIGHTS, MODEL):
        if not path.is_file():
            raise FileNotFoundError(path)
    run_dir = PROJECT / RUN_NAME
    if run_dir.exists():
        raise FileExistsError(f"fresh-start run directory already exists: {run_dir}")

    os.environ["PYTHONHASHSEED"] = str(SEED)
    init_seeds(SEED, deterministic=True)
    model = YOLO(str(MODEL), task="detect")
    model.load(str(WEIGHTS))
    head = model.model.model[-1]
    if not isinstance(head, RoadSnakeHNRDetect):
        raise AssertionError(f"unexpected head: {type(head).__name__}")
    if not 0 < head.hnr_loss_gain < 1:
        raise AssertionError(
            f"formal training requires the calibrated HNR gain, got {head.hnr_loss_gain}; "
            "run verify_japan4_roadsnake_hnr.py and replace the YAML placeholder first"
        )
    initial_state = {name: tensor.detach().cpu().clone() for name, tensor in model.model.state_dict().items()}
    initial_hash = tensor_map_sha256(initial_state)

    def verify_trainer_reconstruction(trainer) -> None:
        rebuilt_state = trainer.model.state_dict()
        changed = [
            name
            for name, value in initial_state.items()
            if name not in rebuilt_state
            or rebuilt_state[name].shape != value.shape
            or not torch.equal(rebuilt_state[name].detach().cpu(), value)
        ]
        if changed:
            raise AssertionError(f"Trainer reconstruction changed HNR initialization: {changed[:8]}")
        rebuilt_head = trainer.model.model[-1]
        if not isinstance(rebuilt_head, RoadSnakeHNRDetect):
            raise AssertionError(f"Trainer rebuilt the wrong head: {type(rebuilt_head).__name__}")
        audit = {
            "seed": SEED,
            "model": str(MODEL),
            "weights": str(WEIGHTS),
            "state_sha256_before_trainer": initial_hash,
            "state_sha256_after_trainer": tensor_map_sha256(
                {name: tensor.detach().cpu() for name, tensor in rebuilt_state.items()}
            ),
            "state_items_preserved": len(initial_state),
            "gamma_initial": float(rebuilt_head.road_snake.gamma.detach().cpu()),
            "hnr_loss_gain": rebuilt_head.hnr_loss_gain,
            "hnr_iou_threshold": rebuilt_head.hnr_iou_threshold,
            "hnr_gt_dilation": rebuilt_head.hnr_gt_dilation,
            "hnr_negatives_per_positive": rebuilt_head.hnr_negatives_per_positive,
            "hnr_margin": rebuilt_head.hnr_margin,
            "hnr_small_area": rebuilt_head.hnr_small_area,
            "passed": True,
        }
        audit_path = Path(trainer.save_dir) / "initialization_audit.json"
        audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
        print("HNR_INIT_AUDIT " + json.dumps(audit, sort_keys=True), flush=True)

    model.add_callback("on_pretrain_routine_end", verify_trainer_reconstruction)
    print(
        "HNR_START "
        + json.dumps(
            {"run": RUN_NAME, "seed": SEED, "gain": head.hnr_loss_gain, "state_sha256": initial_hash},
            sort_keys=True,
        ),
        flush=True,
    )
    model.train(
        data=str(DATA),
        project=str(PROJECT),
        name=RUN_NAME,
        epochs=30,
        patience=1_000_000_000,
        imgsz=640,
        batch=32,
        device="0",
        workers=8,
        seed=SEED,
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
        val=True,
        split="val",
        conf=0.001,
        iou=0.7,
        max_det=300,
        save_period=5,
        resume=False,
        exist_ok=False,
    )
    print(f"HNR_DONE run={RUN_NAME}", flush=True)


if __name__ == "__main__":
    main()
