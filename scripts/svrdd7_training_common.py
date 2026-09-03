"""Shared frozen SVRDD7 training protocol for baseline, RoadSnake-R1, and GBRG."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

NAMES = ["LC", "TC", "AC", "P", "MC", "LP", "TP"]
MODELS = {
    "baseline": ROOT / "ultralytics/cfg/models/26/yolo26.yaml",
    "roadsnake_r1": ROOT / "ultralytics/cfg/models/26/yolo26n-japan4-roadsnake-r1.yaml",
    "roadsnake_gbrg": ROOT / "ultralytics/cfg/models/26/yolo26n-japan4-roadsnake-gbrg.yaml",
}
PROTOCOL: dict[str, Any] = {
    "epochs": 100, "patience": 1_000_000_000, "imgsz": 640, "batch": 32,
    "workers": 8, "seed": 42, "deterministic": True, "optimizer": "auto",
    "cos_lr": False, "lr0": 0.01, "lrf": 0.01, "momentum": 0.937,
    "weight_decay": 0.0005, "warmup_epochs": 3.0, "mosaic": 1.0,
    "mixup": 0.0, "copy_paste": 0.0, "close_mosaic": 10, "amp": True,
    "val": True, "split": "val", "conf": 0.001, "iou": 0.7,
    "max_det": 300, "save_period": 5, "resume": False, "exist_ok": False,
}


def tensor_map_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def command_output(command: list[str]) -> str:
    return subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False).stdout.strip()


def validate_data(path: Path) -> None:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    names = data.get("names", {})
    names = [names[index] for index in sorted(names)] if isinstance(names, dict) else list(names)
    if data.get("nc") != 7 or names != NAMES:
        raise ValueError(f"Expected SVRDD7 classes {NAMES}, got nc={data.get('nc')} names={names}")
    if not all(data.get(split) for split in ("train", "val", "test")):
        raise ValueError("SVRDD7 YAML must define train, val, and sealed test paths")


def parser_for(variant: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=f"Fresh 100E SVRDD7 {variant} training")
    parser.add_argument("--data", type=Path, default=ROOT / "configs/svrdd7_remote.yaml")
    parser.add_argument("--weights", type=Path, default=ROOT / "yolo26n.pt")
    parser.add_argument("--project", type=Path, default=ROOT / "runs/paper1_svrdd7")
    parser.add_argument("--name", required=True)
    parser.add_argument("--device", default="0")
    return parser


def run_training(variant: str) -> None:
    args = parser_for(variant).parse_args()
    from ultralytics import YOLO
    from ultralytics.nn.roadsnake import RoadSnakeDetect, RoadSnakeGBRGDetect
    from ultralytics.utils.roadsnake_gbrg_loss import RoadSnakeGBRGE2ELoss
    from ultralytics.utils.torch_utils import init_seeds

    data = args.data.expanduser().resolve()
    weights = args.weights.expanduser().resolve()
    project = args.project.expanduser().resolve()
    model_yaml = MODELS[variant].resolve()
    for path in (data, weights, model_yaml):
        if not path.is_file():
            raise FileNotFoundError(path)
    validate_data(data)
    run_dir = project / args.name
    metadata_dir = ROOT / "runtime_meta" / args.name
    if run_dir.exists() or metadata_dir.exists():
        raise FileExistsError(f"Fresh-start output already exists: {run_dir} or {metadata_dir}")

    os.environ["PYTHONHASHSEED"] = str(PROTOCOL["seed"])
    init_seeds(PROTOCOL["seed"], deterministic=True)
    model = YOLO(str(model_yaml), task="detect")
    model.load(str(weights))
    initial_state = {name: value.detach().cpu().clone() for name, value in model.model.state_dict().items()}
    head = model.model.model[-1]
    module_hashes: dict[str, str] = {}
    if variant in {"roadsnake_r1", "roadsnake_gbrg"}:
        if not isinstance(head, RoadSnakeDetect):
            raise AssertionError(f"Expected RoadSnake head, got {type(head).__name__}")
        module_hashes["road_snake"] = tensor_map_sha256(head.road_snake.state_dict())
    if variant == "roadsnake_gbrg":
        if not isinstance(head, RoadSnakeGBRGDetect):
            raise AssertionError(f"Expected RoadSnakeGBRGDetect, got {type(head).__name__}")
        module_hashes["gbrg_region_head"] = tensor_map_sha256(head.gbrg_region_head.state_dict())

    metadata_dir.mkdir(parents=True)
    metadata = {
        "variant": variant, "classes": NAMES, "data": str(data), "weights": str(weights),
        "model_yaml": str(model_yaml), "project": str(project), "name": args.name,
        "device": args.device, "protocol": PROTOCOL, "command": shlex.join([sys.executable, *sys.argv]),
        "git": {"commit": command_output(["git", "rev-parse", "HEAD"]),
                "branch": command_output(["git", "branch", "--show-current"]),
                "status": command_output(["git", "status", "--short"])},
        "environment": {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
                        "cuda": torch.version.cuda, "cuda_available": torch.cuda.is_available()},
        "initial_module_hashes": module_hashes,
    }
    (metadata_dir / "resolved_run.json").write_text(json.dumps(metadata, indent=2, default=str) + "\n", encoding="utf-8")
    shutil.copy2(data, metadata_dir / "data_snapshot.yaml")
    shutil.copy2(model_yaml, metadata_dir / "model_snapshot.yaml")

    def verify_reconstruction(trainer) -> None:
        rebuilt = trainer.model
        rebuilt_head = rebuilt.model[-1]
        if int(rebuilt_head.nc) != len(NAMES):
            raise AssertionError(f"Trainer head.nc={rebuilt_head.nc}, expected {len(NAMES)}")
        compatible = {name: value for name, value in initial_state.items()
                      if name in rebuilt.state_dict() and rebuilt.state_dict()[name].shape == value.shape}
        changed = [name for name, value in compatible.items()
                   if not torch.equal(rebuilt.state_dict()[name].detach().cpu(), value)]
        if changed:
            raise AssertionError(f"Trainer reconstruction changed compatible tensors: {changed[:8]}")
        audit = {"variant": variant, "head": type(rebuilt_head).__name__, "head_nc": int(rebuilt_head.nc),
                 "compatible_state_items_preserved": len(compatible), "module_hashes_before": module_hashes,
                 "passed": True}
        if "road_snake" in module_hashes:
            audit["road_snake_after"] = tensor_map_sha256(rebuilt_head.road_snake.state_dict())
            if audit["road_snake_after"] != module_hashes["road_snake"]:
                raise AssertionError("Trainer changed seeded RoadSnake adapter")
        if "gbrg_region_head" in module_hashes:
            audit["gbrg_region_head_after"] = tensor_map_sha256(rebuilt_head.gbrg_region_head.state_dict())
            if audit["gbrg_region_head_after"] != module_hashes["gbrg_region_head"]:
                raise AssertionError("Trainer changed seeded GBRG head")
        (Path(trainer.save_dir) / "initialization_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
        print("SVRDD7_INIT_AUDIT " + json.dumps(audit, sort_keys=True), flush=True)

    def log_gbrg(trainer) -> None:
        criterion = getattr(trainer.model, "criterion", None)
        if not isinstance(criterion, RoadSnakeGBRGE2ELoss):
            return
        stats = {"epoch": int(trainer.epoch + 1), "lambda": criterion.current_lambda,
                 "raw_gradient_ratio": criterion.last_raw_gradient_ratio,
                 "weighted_gradient_ratio": criterion.last_weighted_gradient_ratio,
                 "region_loss": criterion.last_region_loss, "positive_pixels": criterion.last_positive_pixels,
                 "background_effective_pixels": criterion.last_background_effective_pixels}
        with (Path(trainer.save_dir) / "gbrg_controller.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(stats, sort_keys=True) + "\n")

    model.add_callback("on_pretrain_routine_end", verify_reconstruction)
    model.add_callback("on_train_epoch_end", log_gbrg)
    exit_code = 1
    save_dir: Path | None = None
    try:
        model.train(data=str(data), project=str(project), name=args.name, device=args.device, **PROTOCOL)
        save_dir = Path(model.trainer.save_dir)
        exit_code = 0
    finally:
        (metadata_dir / "exit_code.txt").write_text(f"{exit_code}\n", encoding="utf-8")
        if save_dir is not None:
            shutil.copytree(metadata_dir, save_dir / "runtime_meta", dirs_exist_ok=True)
