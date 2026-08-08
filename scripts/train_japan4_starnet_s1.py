"""Fresh protocol-locked 30E training for official StarNet-S1-YOLO26."""

from __future__ import annotations

import argparse
import json
import platform
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ultralytics import YOLO  # noqa: E402
from ultralytics.nn.tasks import DetectionModel  # noqa: E402
from ultralytics.utils.parent_semantic_init import (  # noqa: E402
    initialize_official_starnet_s1_yolo26,
    state_sha256,
)

MODEL_YAML = ROOT / "ultralytics/cfg/models/26/yolo26-StarNet-S1-Official.yaml"
PROTOCOL = {
    "epochs": 30,
    "imgsz": 640,
    "batch": 32,
    "workers": 8,
    "device": "0",
    "seed": 42,
    "deterministic": True,
    "optimizer": "auto",
    "amp": True,
    "val": True,
    "patience": 1_000_000_000,
    "cos_lr": False,
    "lr0": 0.01,
    "lrf": 0.01,
    "momentum": 0.937,
    "weight_decay": 0.0005,
    "warmup_epochs": 3.0,
    "mosaic": 1.0,
    "mixup": 0.0,
    "copy_paste": 0.0,
    "close_mosaic": 10,
    "conf": 0.001,
    "iou": 0.7,
    "max_det": 300,
    "save_period": 5,
}


def command_output(command: list[str]) -> str:
    return subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False).stdout.strip()


def validate_data(path: Path) -> None:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    names = data.get("names", {})
    names = list(names.values()) if isinstance(names, dict) else list(names)
    if data.get("nc") != 4 or names != ["D00", "D10", "D20", "D40"]:
        raise ValueError(f"Expected Japan4-cleanV3, got nc={data.get('nc')} names={names}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--star-checkpoint", type=Path, required=True)
    parser.add_argument("--yolo-weights", type=Path, default=ROOT / "yolo26n.pt")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--project", type=Path, default=ROOT / "runs/paper1_japan4_clean")
    args = parser.parse_args()
    star_checkpoint = args.star_checkpoint.expanduser().resolve()
    yolo_weights = args.yolo_weights.expanduser().resolve()
    data = args.data.expanduser().resolve()
    project = args.project.expanduser().resolve()
    if not all(path.is_file() for path in (MODEL_YAML, star_checkpoint, yolo_weights, data)):
        raise FileNotFoundError((MODEL_YAML, star_checkpoint, yolo_weights, data))
    validate_data(data)
    metadata_dir = ROOT / "runtime_meta" / args.name
    if metadata_dir.exists() or (project / args.name).exists():
        raise FileExistsError(f"Fresh-start protection: run already exists: {args.name}")

    torch.manual_seed(PROTOCOL["seed"])
    initialized = DetectionModel(str(MODEL_YAML), ch=3, nc=4, verbose=False).float()
    initialization = initialize_official_starnet_s1_yolo26(initialized, star_checkpoint, yolo_weights)
    expected = {name: value.detach().cpu().clone() for name, value in initialized.state_dict().items()}
    expected_hash = state_sha256(expected)

    wrapper = YOLO(str(MODEL_YAML), task="detect")
    wrapper.model = initialized
    wrapper.ckpt = {"semantic_initialization": True}
    wrapper.overrides["model"] = str(MODEL_YAML)
    metadata_dir.mkdir(parents=True)

    def verify_trainer_rebuild(trainer) -> None:
        actual = trainer.model.state_dict()
        mismatched = [name for name, value in expected.items() if not torch.equal(value, actual[name].detach().cpu())]
        actual_hash = state_sha256(actual)
        rebuild = {
            "passed": not mismatched and actual_hash == expected_hash,
            "before_sha256": expected_hash,
            "after_sha256": actual_hash,
            "mismatched": mismatched,
            "official_backbone_hash_preserved": initialization["official_backbone_state_sha256"]
            == state_sha256(actual, [name for name in actual if name.startswith(("model.0.stem.", "model.0.stages."))]),
            "adapter_hash_preserved": initialization["adapter_state_sha256"]
            == state_sha256(actual, [name for name in actual if name.startswith("model.0.adapters.")]),
        }
        (metadata_dir / "trainer_rebuild_audit.json").write_text(json.dumps(rebuild, indent=2), encoding="utf-8")
        if not rebuild["passed"] or not rebuild["official_backbone_hash_preserved"] or not rebuild["adapter_hash_preserved"]:
            raise AssertionError(f"Trainer reconstruction invalidated initialization: {rebuild}")
        print(f"TRAINER_REBUILD_INITIALIZATION_VERIFIED sha256={actual_hash}")

    wrapper.add_callback("on_pretrain_routine_end", verify_trainer_rebuild)
    metadata = {
        "initialization": initialization,
        "protocol": PROTOCOL,
        "model_yaml": str(MODEL_YAML.resolve()),
        "data": str(data),
        "command": shlex.join([sys.executable, *sys.argv]),
        "git": {
            "commit": command_output(["git", "rev-parse", "HEAD"]),
            "branch": command_output(["git", "branch", "--show-current"]),
            "status": command_output(["git", "status", "--short"]),
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "test_split_read": False,
        "fresh_start": True,
    }
    (metadata_dir / "initialization_audit.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    shutil.copy2(MODEL_YAML, metadata_dir / "model_snapshot.yaml")
    shutil.copy2(data, metadata_dir / "data_snapshot.yaml")

    exit_code = 1
    save_dir = None
    try:
        wrapper.train(
            data=str(data),
            project=str(project),
            name=args.name,
            exist_ok=False,
            resume=False,
            pretrained=False,
            **PROTOCOL,
        )
        save_dir = Path(wrapper.trainer.save_dir)
        exit_code = 0
    finally:
        (metadata_dir / "exit_code.txt").write_text(f"{exit_code}\n", encoding="utf-8")
        if save_dir is not None:
            shutil.copytree(metadata_dir, save_dir / "runtime_meta", dirs_exist_ok=True)


if __name__ == "__main__":
    main()
