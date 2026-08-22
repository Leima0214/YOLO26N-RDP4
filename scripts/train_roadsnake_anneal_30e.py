"""Fresh-start 30E RoadSnake-Anneal signal experiment on Japan4-cleanV3.

The sole experimental variable relative to RoadSnake-R1 is a fixed schedule:
18 epochs full RoadSnake, 6 epochs frozen cosine withdrawal, and 6 epochs with
the adapter completely bypassed.  A separate best_native.pt is selected only
from the final native phase.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
from ultralytics import YOLO  # noqa: E402
from ultralytics.utils.torch_utils import init_seeds, unwrap_model  # noqa: E402


DATA = ROOT / "configs/japan4_clean_v3_remote.yaml"
WEIGHTS = ROOT / "yolo26n.pt"
MODEL = ROOT / "ultralytics/cfg/models/26/yolo26n-japan4-roadsnake-r1.yaml"
PROJECT = ROOT / "runs/paper1_japan4_clean"
RUN_NAME = "yolo26n-japan4-roadsnake-anneal_cleanv3_30e_seed42_20260822"
SEED = 42
EPOCHS = 30
FULL_END = 18
ANNEAL_END = 24


def tensor_map_sha256(state: dict[str, torch.Tensor]) -> str:
    """Hash a named tensor mapping deterministically on CPU."""
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def adapter_from(model):
    """Return the RoadSnake adapter from a raw, wrapped, or EMA detection model."""
    return unwrap_model(model).model[-1].road_snake


def schedule(epoch: int) -> tuple[str, float, bool]:
    """Return phase, residual scale, and adapter-freeze state for a zero-based epoch."""
    if epoch < FULL_END:
        return "full", 1.0, False
    if epoch < ANNEAL_END:
        progress = (epoch - FULL_END) / (ANNEAL_END - FULL_END)
        return "withdraw", 0.5 * (1.0 + math.cos(math.pi * progress)), True
    return "native", 0.0, True


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
    initial_state = {name: value.detach().cpu().clone() for name, value in model.model.state_dict().items()}
    initial_adapter = {
        name: value.detach().cpu().clone() for name, value in adapter_from(model.model).state_dict().items()
    }
    initial_adapter_hash = tensor_map_sha256(initial_adapter)
    native_best = {"fitness": float("-inf"), "epoch": None}

    def verify_reconstruction(trainer) -> None:
        trained = trainer.model.state_dict()
        compatible = {
            name: value for name, value in initial_state.items() if name in trained and trained[name].shape == value.shape
        }
        changed = [
            name for name, value in compatible.items() if not torch.equal(trained[name].detach().cpu(), value)
        ]
        if changed:
            raise AssertionError(f"Trainer reconstruction changed {len(changed)} tensors: {changed[:8]}")
        actual_hash = tensor_map_sha256(
            {name: value.detach().cpu() for name, value in adapter_from(trainer.model).state_dict().items()}
        )
        if actual_hash != initial_adapter_hash:
            raise AssertionError(
                f"Trainer reconstruction changed RoadSnake: expected={initial_adapter_hash} actual={actual_hash}"
            )
        audit = {
            "seed": SEED,
            "epochs": EPOCHS,
            "schedule": {"full": [1, 18], "withdraw": [19, 24], "native": [25, 30]},
            "compatible_state_items_preserved": len(compatible),
            "adapter_sha256_before_trainer": initial_adapter_hash,
            "adapter_sha256_after_trainer": actual_hash,
            "passed": True,
        }
        (Path(trainer.save_dir) / "initialization_audit.json").write_text(
            json.dumps(audit, indent=2), encoding="utf-8"
        )
        print("ANNEAL_INIT_AUDIT " + json.dumps(audit, sort_keys=True), flush=True)

    def apply_epoch_schedule(trainer) -> None:
        phase, scale, frozen = schedule(trainer.epoch)
        train_adapter = adapter_from(trainer.model)
        train_adapter.set_anneal_scale(scale)
        for parameter in train_adapter.parameters():
            parameter.requires_grad_(not frozen)
        # Validation and serialized checkpoints are based on EMA, so its runtime
        # path must use the exact same scale as the live training model.
        if trainer.ema is not None:
            adapter_from(trainer.ema.ema).set_anneal_scale(scale)
        print(
            f"ANNEAL_PHASE epoch={trainer.epoch + 1}/{EPOCHS} phase={phase} "
            f"scale={scale:.8f} adapter_frozen={int(frozen)}",
            flush=True,
        )

    def save_native_candidate(trainer) -> None:
        phase, scale, _ = schedule(trainer.epoch)
        if phase != "native" or scale != 0.0 or trainer.fitness is None:
            return
        fitness = float(trainer.fitness)
        if fitness <= native_best["fitness"]:
            return
        source = Path(trainer.last)
        if not source.is_file():
            raise FileNotFoundError(f"Expected epoch checkpoint before native selection: {source}")
        destination = Path(trainer.wdir) / "best_native.pt"
        shutil.copy2(source, destination)
        native_best.update(fitness=fitness, epoch=trainer.epoch + 1)
        record = {
            "best_native_epoch": native_best["epoch"],
            "best_native_fitness": native_best["fitness"],
            "source": str(source),
            "checkpoint": str(destination),
            "road_snake_scale": 0.0,
        }
        (Path(trainer.save_dir) / "best_native.json").write_text(
            json.dumps(record, indent=2), encoding="utf-8"
        )
        print("ANNEAL_NATIVE_BEST " + json.dumps(record, sort_keys=True), flush=True)

    model.add_callback("on_pretrain_routine_end", verify_reconstruction)
    model.add_callback("on_train_epoch_start", apply_epoch_schedule)
    model.add_callback("on_fit_epoch_end", save_native_candidate)
    print(f"ANNEAL_START seed={SEED} run={RUN_NAME} adapter_sha256={initial_adapter_hash}", flush=True)
    model.train(
        data=str(DATA),
        project=str(PROJECT),
        name=RUN_NAME,
        epochs=EPOCHS,
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
    print("ANNEAL_DONE " + json.dumps(native_best, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
