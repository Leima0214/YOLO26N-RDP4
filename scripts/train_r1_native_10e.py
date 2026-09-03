"""R1 Native 10E Restart: start from frozen R1 best.pt, re-init optimizer.

Mirrors gbrg_o2m_control_from_best_10e exactly: resume=False, fresh optimizer/
LR/epoch schedule, 10 epochs, native losses, no VAGR, no GBRG.  Val only,
Test sealed.  Needed for the fairness patch I_restart = (GBRG R10 - GBRG) -
(R1 R10 - R1).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ultralytics import YOLO  # noqa: E402
from ultralytics.utils.torch_utils import init_seeds  # noqa: E402

DATA = ROOT / "configs/japan4_clean_v3_remote.yaml"
PARENT = ROOT / "runs/paper1_japan4_clean/yolo26n-japan4-roadsnake-r1_cleanv3_100e_seed42_20260818/weights/best.pt"
PROJECT = ROOT / "runs/paper1_japan4_clean"
SEED = 42


def state_hash(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    args = parser.parse_args()
    run_name = "r1_o2m_native_from_best_10e_seed42_20260902"
    run_dir = PROJECT / run_name
    if run_dir.exists():
        raise FileExistsError(run_dir)
    for path in (DATA, PARENT):
        if not path.is_file():
            raise FileNotFoundError(path)

    os.environ["PYTHONHASHSEED"] = str(SEED)
    init_seeds(SEED, deterministic=True)
    model = YOLO(str(PARENT))
    initial_hash = state_hash(model.model)

    def verify_start(trainer) -> None:
        rebuilt_hash = state_hash(trainer.model)
        audit = {
            "variant": "native_r1_10e",
            "parent": str(PARENT),
            "initial_state_sha256": initial_hash,
            "rebuilt_state_sha256": rebuilt_hash,
            "state_equal": initial_hash == rebuilt_hash,
            "head": type(trainer.model.model[-1]).__name__,
        }
        if not audit["state_equal"]:
            raise AssertionError(audit)
        (Path(trainer.save_dir) / "r1_restart_initialization_audit.json").write_text(
            json.dumps(audit, indent=2), encoding="utf-8"
        )
        print("R1_RESTART_INIT " + json.dumps(audit, sort_keys=True), flush=True)

    model.add_callback("on_pretrain_routine_end", verify_start)
    print("R1_RESTART_START " + json.dumps({"run": run_name, "parent_hash": initial_hash}, sort_keys=True),
          flush=True)
    model.train(
        data=str(DATA), project=str(PROJECT), name=run_name,
        epochs=10, patience=1_000_000_000, imgsz=640, batch=32,
        device="0", workers=8, seed=SEED, deterministic=True,
        optimizer="auto", cos_lr=False, lr0=0.01, lrf=0.01,
        momentum=0.937, weight_decay=0.0005, warmup_epochs=3.0,
        mosaic=1.0, mixup=0.0, copy_paste=0.0, close_mosaic=10,
        amp=True, val=True, split="val", conf=0.001, iou=0.7,
        max_det=300, save_period=1, resume=False, exist_ok=False,
    )
    print(f"R1_RESTART_DONE run={run_name}", flush=True)


if __name__ == "__main__":
    main()
