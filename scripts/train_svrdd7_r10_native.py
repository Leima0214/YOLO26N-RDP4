#!/usr/bin/env python3
"""Fresh native 10E restart from an SVRDD7 best.pt (no optimizer/scheduler resume)."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

NAMES = ["LC", "TC", "AC", "P", "MC", "LP", "TP"]
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
    parser.add_argument("--parent", type=Path, required=True, help="SVRDD7 best.pt used only as model weights")
    parser.add_argument("--data", type=Path, default=ROOT / "configs/svrdd7_remote.yaml")
    parser.add_argument("--project", type=Path, default=ROOT / "runs/paper1_svrdd7")
    parser.add_argument("--name", required=True)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import init_seeds

    parent, data, project = args.parent.expanduser().resolve(), args.data.expanduser().resolve(), args.project.expanduser().resolve()
    if parent.name != "best.pt" or not parent.is_file():
        raise ValueError("--parent must be an existing best.pt")
    if not data.is_file():
        raise FileNotFoundError(data)
    spec = yaml.safe_load(data.read_text(encoding="utf-8"))
    names = spec.get("names", {})
    names = [names[index] for index in sorted(names)] if isinstance(names, dict) else list(names)
    if spec.get("nc") != 7 or names != NAMES:
        raise ValueError(f"Expected SVRDD7 data, got nc={spec.get('nc')} names={names}")
    if (project / args.name).exists():
        raise FileExistsError(project / args.name)

    os.environ["PYTHONHASHSEED"] = str(SEED)
    init_seeds(SEED, deterministic=True)
    model = YOLO(str(parent), task="detect")
    if int(model.model.model[-1].nc) != len(NAMES):
        raise AssertionError(f"Parent head.nc={model.model.model[-1].nc}, expected 7")
    initial_hash = state_hash(model.model)

    def verify_start(trainer) -> None:
        rebuilt_hash = state_hash(trainer.model)
        audit = {"parent": str(parent), "head": type(trainer.model.model[-1]).__name__,
                 "head_nc": int(trainer.model.model[-1].nc), "initial_state_sha256": initial_hash,
                 "rebuilt_state_sha256": rebuilt_hash, "state_equal": initial_hash == rebuilt_hash,
                 "epochs": 10, "resume": False, "passed": initial_hash == rebuilt_hash}
        if audit["head_nc"] != 7 or not audit["state_equal"]:
            raise AssertionError(audit)
        (Path(trainer.save_dir) / "r10_initialization_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
        print("SVRDD7_R10_INIT " + json.dumps(audit, sort_keys=True), flush=True)

    model.add_callback("on_pretrain_routine_end", verify_start)
    model.train(
        data=str(data), project=str(project), name=args.name, epochs=10,
        patience=1_000_000_000, imgsz=640, batch=32, device=args.device, workers=8,
        seed=SEED, deterministic=True, optimizer="auto", cos_lr=False, lr0=0.01,
        lrf=0.01, momentum=0.937, weight_decay=0.0005, warmup_epochs=3.0,
        mosaic=1.0, mixup=0.0, copy_paste=0.0, close_mosaic=10, amp=True,
        val=True, split="val", conf=0.001, iou=0.7, max_det=300,
        save_period=1, resume=False, exist_ok=False,
    )


if __name__ == "__main__":
    main()
