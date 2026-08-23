"""Fresh-start Japan4-cleanV3 30E RoadSnake-GBRG signal experiment."""

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
from ultralytics.nn.roadsnake import RoadSnakeGBRGDetect  # noqa: E402
from ultralytics.utils.roadsnake_gbrg_loss import RoadSnakeGBRGE2ELoss  # noqa: E402
from ultralytics.utils.torch_utils import init_seeds  # noqa: E402

DATA = ROOT / "configs/japan4_clean_v3_remote.yaml"
WEIGHTS = ROOT / "yolo26n.pt"
MODEL = ROOT / "ultralytics/cfg/models/26/yolo26n-japan4-roadsnake-gbrg.yaml"
PROJECT = ROOT / "runs/paper1_japan4_clean"
RUN_NAME = "yolo26n-japan4-roadsnake-gbrg_cleanv3_30e_seed42_20260823"
EPOCHS = 30
GBRG_ANNEAL_START_EPOCH = None
GBRG_ANNEAL_END_EPOCH = None
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
    if not isinstance(head, RoadSnakeGBRGDetect):
        raise AssertionError(f"unexpected head: {type(head).__name__}")
    initial_state = {name: tensor.detach().cpu().clone() for name, tensor in model.model.state_dict().items()}
    adapter_hash = tensor_map_sha256({name: tensor for name, tensor in head.road_snake.state_dict().items()})
    region_hash = tensor_map_sha256({name: tensor for name, tensor in head.gbrg_region_head.state_dict().items()})

    def verify_trainer_reconstruction(trainer) -> None:
        rebuilt_state = trainer.model.state_dict()
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
            raise AssertionError(f"Trainer reconstruction changed GBRG tensors: {changed[:8]}")
        rebuilt_head = trainer.model.model[-1]
        if not isinstance(rebuilt_head, RoadSnakeGBRGDetect):
            raise AssertionError(f"Trainer rebuilt the wrong head: {type(rebuilt_head).__name__}")
        audit = {
            "seed": SEED,
            "model": str(MODEL),
            "weights": str(WEIGHTS),
            "compatible_state_items_preserved": len(compatible),
            "incompatible_or_new_state_items": len(rebuilt_state) - len(compatible),
            "adapter_sha256_before": adapter_hash,
            "adapter_sha256_after": tensor_map_sha256(rebuilt_head.road_snake.state_dict()),
            "region_sha256_before": region_hash,
            "region_sha256_after": tensor_map_sha256(rebuilt_head.gbrg_region_head.state_dict()),
            "gamma_initial": float(rebuilt_head.road_snake.gamma.detach().cpu()),
            "passed": True,
        }
        if audit["adapter_sha256_before"] != audit["adapter_sha256_after"]:
            raise AssertionError("Trainer changed the seeded RoadSnake adapter")
        if audit["region_sha256_before"] != audit["region_sha256_after"]:
            raise AssertionError("Trainer changed the seeded GBRG head")
        (Path(trainer.save_dir) / "initialization_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
        print("GBRG_INIT_AUDIT " + json.dumps(audit, sort_keys=True), flush=True)

    def log_controller(trainer) -> None:
        criterion = getattr(trainer.model, "criterion", None)
        if not isinstance(criterion, RoadSnakeGBRGE2ELoss):
            return
        stats = {
            "epoch": int(trainer.epoch + 1),
            "lambda": criterion.current_lambda,
            "anneal_scale": criterion.anneal_scale,
            "raw_gradient_ratio": criterion.last_raw_gradient_ratio,
            "pre_anneal_weighted_gradient_ratio": criterion.last_pre_anneal_weighted_gradient_ratio,
            "weighted_gradient_ratio": criterion.last_weighted_gradient_ratio,
            "region_loss": criterion.last_region_loss,
            "positive_pixels": criterion.last_positive_pixels,
            "background_effective_pixels": criterion.last_background_effective_pixels,
        }
        with (Path(trainer.save_dir) / "gbrg_controller.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(stats, sort_keys=True) + "\n")
        print("GBRG_EPOCH " + json.dumps(stats, sort_keys=True), flush=True)

    def update_anneal_scale(trainer) -> None:
        if GBRG_ANNEAL_START_EPOCH is None or GBRG_ANNEAL_END_EPOCH is None:
            return
        criterion = getattr(trainer.model, "criterion", None)
        if not isinstance(criterion, RoadSnakeGBRGE2ELoss):
            return
        if criterion.anneal_start_epoch is None:
            criterion.configure_anneal(GBRG_ANNEAL_START_EPOCH, GBRG_ANNEAL_END_EPOCH)
        criterion.set_epoch(int(trainer.epoch + 1))

    model.add_callback("on_pretrain_routine_end", verify_trainer_reconstruction)
    model.add_callback("on_train_epoch_start", update_anneal_scale)
    model.add_callback("on_train_epoch_end", log_controller)
    print(
        "GBRG_START "
        + json.dumps(
            {
                "run": RUN_NAME,
                "seed": SEED,
                "adapter_sha256": adapter_hash,
                "region_sha256": region_hash,
                "anneal_start_epoch": GBRG_ANNEAL_START_EPOCH,
                "anneal_end_epoch": GBRG_ANNEAL_END_EPOCH,
            },
            sort_keys=True,
        ),
        flush=True,
    )
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
    print(f"GBRG_DONE run={RUN_NAME}", flush=True)


if __name__ == "__main__":
    main()
