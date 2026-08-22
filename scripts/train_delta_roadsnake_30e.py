"""Fresh-start Japan4-cleanV3 30E training for reference-subtracted DeltaRoadSnake-P4."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import YOLO  # noqa: E402
from ultralytics.nn.roadsnake import DeltaRoadSnakeDetect  # noqa: E402
from ultralytics.utils.torch_utils import init_seeds  # noqa: E402

DATA = ROOT / "configs/japan4_clean_v3_remote.yaml"
WEIGHTS = ROOT / "yolo26n.pt"
MODEL = ROOT / "ultralytics/cfg/models/26/yolo26n-japan4-delta-roadsnake.yaml"
PROJECT = ROOT / "runs/paper1_japan4_clean"
RUN_NAME = "yolo26n-japan4-delta-roadsnake_cleanv3_30e_seed42_20260822"
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
        raise FileExistsError(f"Fresh-start run directory already exists: {run_dir}")

    os.environ["PYTHONHASHSEED"] = str(SEED)
    init_seeds(SEED, deterministic=True)
    model = YOLO(str(MODEL), task="detect")
    model.load(str(WEIGHTS))
    if not isinstance(model.model.model[-1], DeltaRoadSnakeDetect):
        raise TypeError(type(model.model.model[-1]).__name__)

    initial_state = {name: tensor.detach().cpu().clone() for name, tensor in model.model.state_dict().items()}
    adapter_state = {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.model.model[-1].road_snake.state_dict().items()
    }
    expected_adapter_hash = tensor_map_sha256(adapter_state)

    def verify_trainer_reconstruction(trainer) -> None:
        trained_state = trainer.model.state_dict()
        compatible = {
            name: value
            for name, value in initial_state.items()
            if name in trained_state and trained_state[name].shape == value.shape
        }
        changed = [
            name
            for name, value in compatible.items()
            if not torch.equal(trained_state[name].detach().cpu(), value)
        ]
        if changed:
            raise AssertionError(f"Trainer reconstruction changed tensors: {changed[:8]}")
        adapter = trainer.model.model[-1].road_snake
        actual_hash = tensor_map_sha256(
            {name: tensor.detach().cpu() for name, tensor in adapter.state_dict().items()}
        )
        if actual_hash != expected_adapter_hash:
            raise AssertionError(
                f"Trainer reconstruction changed DeltaRoadSnake: {expected_adapter_hash} != {actual_hash}"
            )
        audit = {
            "seed": SEED,
            "model": str(MODEL),
            "weights": str(WEIGHTS),
            "compatible_state_items_preserved": len(compatible),
            "adapter_state_items": len(adapter_state),
            "adapter_sha256_before_trainer": expected_adapter_hash,
            "adapter_sha256_after_trainer": actual_hash,
            "offset_weight_nonzero_initial": int(torch.count_nonzero(adapter.offset.weight)),
            "offset_bias_nonzero_initial": int(torch.count_nonzero(adapter.offset.bias)),
            "passed": True,
        }
        (Path(trainer.save_dir) / "initialization_audit.json").write_text(
            json.dumps(audit, indent=2), encoding="utf-8"
        )
        print("DELTA_RS_INIT_AUDIT " + json.dumps(audit, sort_keys=True), flush=True)

    model.add_callback("on_pretrain_routine_end", verify_trainer_reconstruction)
    print(
        f"DELTA_RS_START seed={SEED} adapter_sha256={expected_adapter_hash} run={RUN_NAME}",
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
    print(f"DELTA_RS_DONE seed={SEED} run={RUN_NAME}", flush=True)


if __name__ == "__main__":
    main()
