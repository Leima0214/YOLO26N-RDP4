"""Fresh-start Japan4-cleanV3 training entry for RoadSnake-DP-v1."""

from __future__ import annotations

import argparse
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
from ultralytics.nn.roadsnake import RoadSnakeDualPathDetect  # noqa: E402
from ultralytics.utils.torch_utils import init_seeds  # noqa: E402

MODEL = ROOT / "ultralytics/cfg/models/26/yolo26n-japan4-roadsnake-dp-v1.yaml"
DATA = ROOT / "configs/japan4_clean_v3_remote.yaml"
WEIGHTS = ROOT / "yolo26n.pt"
PROJECT = ROOT / "runs/paper1_japan4_clean"
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, choices=(1, 30, 100), required=True)
    parser.add_argument("--name")
    parser.add_argument("--device", default="0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in (MODEL, DATA, WEIGHTS):
        if not path.is_file():
            raise FileNotFoundError(path)

    suffix = "smoke" if args.epochs == 1 else "formal"
    run_name = args.name or f"yolo26n-japan4-roadsnake-dp-v1_{suffix}_{args.epochs}e_seed42_20260822"
    run_dir = PROJECT / run_name
    if run_dir.exists():
        raise FileExistsError(f"Fresh-start run directory already exists: {run_dir}")

    os.environ["PYTHONHASHSEED"] = str(SEED)
    init_seeds(SEED, deterministic=True)
    model = YOLO(str(MODEL), task="detect")
    model.load(str(WEIGHTS))
    head = model.model.model[-1]
    if not isinstance(head, RoadSnakeDualPathDetect):
        raise TypeError(type(head).__name__)
    if float(head.road_snake.gamma) != 0.0:
        raise AssertionError("RoadSnake-DP-v1 must start with gamma=0")

    initial_state = {name: tensor.detach().cpu().clone() for name, tensor in model.model.state_dict().items()}
    adapter_state = {
        name: tensor.detach().cpu().clone() for name, tensor in head.road_snake.state_dict().items()
    }
    expected_adapter_hash = tensor_map_sha256(adapter_state)

    def verify_trainer_reconstruction(trainer) -> None:
        rebuilt = trainer.model
        rebuilt_head = rebuilt.model[-1]
        if not isinstance(rebuilt_head, RoadSnakeDualPathDetect):
            raise TypeError(type(rebuilt_head).__name__)
        rebuilt_state = rebuilt.state_dict()
        compatible = {
            name: value
            for name, value in initial_state.items()
            if name in rebuilt_state and rebuilt_state[name].shape == value.shape
        }
        changed = [
            name
            for name, value in compatible.items()
            if not torch.equal(rebuilt_state[name].detach().cpu(), value)
        ]
        if changed:
            raise AssertionError(f"Trainer reconstruction changed tensors: {changed[:8]}")
        actual_adapter_hash = tensor_map_sha256(
            {name: tensor.detach().cpu() for name, tensor in rebuilt_head.road_snake.state_dict().items()}
        )
        if actual_adapter_hash != expected_adapter_hash:
            raise AssertionError(
                "Trainer reconstruction changed RoadSnake-DP adapter: "
                f"{expected_adapter_hash} != {actual_adapter_hash}"
            )
        audit = {
            "seed": SEED,
            "epochs": args.epochs,
            "model": str(MODEL),
            "weights": str(WEIGHTS),
            "compatible_state_items_preserved": len(compatible),
            "adapter_state_items": len(adapter_state),
            "adapter_sha256_before_trainer": expected_adapter_hash,
            "adapter_sha256_after_trainer": actual_adapter_hash,
            "gamma_initial": float(rebuilt_head.road_snake.gamma),
            "passed": True,
        }
        output = Path(trainer.save_dir) / "initialization_audit.json"
        output.write_text(json.dumps(audit, indent=2), encoding="utf-8")
        print("ROAD_SNAKE_DP_INIT_AUDIT " + json.dumps(audit, sort_keys=True), flush=True)

    model.add_callback("on_pretrain_routine_end", verify_trainer_reconstruction)
    print(
        f"ROAD_SNAKE_DP_START epochs={args.epochs} seed={SEED} "
        f"adapter_sha256={expected_adapter_hash} run={run_name}",
        flush=True,
    )
    model.train(
        data=str(DATA),
        project=str(PROJECT),
        name=run_name,
        epochs=args.epochs,
        patience=1_000_000_000,
        imgsz=640,
        batch=32,
        device=args.device,
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
        save_period=5 if args.epochs > 1 else 1,
        resume=False,
        exist_ok=False,
    )
    print(f"ROAD_SNAKE_DP_DONE epochs={args.epochs} seed={SEED} run={run_name}", flush=True)


if __name__ == "__main__":
    main()

