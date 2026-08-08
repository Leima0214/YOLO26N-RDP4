"""Protocol-locked fresh 30E training for Japan4 RoadAdaptiveScaleFusion-N1."""

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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ultralytics import YOLO  # noqa: E402
from ultralytics.nn.tasks import DetectionModel  # noqa: E402
from ultralytics.nn.yolo26_cvpr_improvements import RoadAdaptiveScaleFusionN1  # noqa: E402
from verify_japan4_road_adaptive_scale_fusion_n1 import (  # noqa: E402
    N1_YAML,
    component_coverage,
    load_and_verify,
    n1_semantic_transfer,
    state_sha256,
)

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
    text = path.read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    names = data.get("names", {})
    names = list(names.values()) if isinstance(names, dict) else list(names)
    if data.get("nc") != 4 or names != ["D00", "D10", "D20", "D40"]:
        raise ValueError(f"Expected Japan4-cleanV3, got nc={data.get('nc')} names={names}")
    if "test:" in text.lower():
        print("DATA_YAML_CONTAINS_TEST_ENTRY_BUT_RUN_IS_VAL_ONLY")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yolo-weights", type=Path, default=ROOT / "yolo26n.pt")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--project", type=Path, default=ROOT / "runs/paper1_japan4_clean")
    args = parser.parse_args()
    weights = args.yolo_weights.expanduser().resolve()
    data = args.data.expanduser().resolve()
    project = args.project.expanduser().resolve()
    if not all(path.is_file() for path in (N1_YAML, weights, data)):
        raise FileNotFoundError((N1_YAML, weights, data))
    validate_data(data)

    metadata_dir = ROOT / "runtime_meta" / args.name
    run_dir = project / args.name
    if metadata_dir.exists() or run_dir.exists():
        raise FileExistsError(f"Fresh-start protection: run already exists: {args.name}")

    torch.manual_seed(PROTOCOL["seed"])
    initialized = DetectionModel(str(N1_YAML), ch=3, nc=4, verbose=False).float()
    transfer, matched = n1_semantic_transfer(initialized, weights)
    if transfer["adapter_loaded_items"]:
        raise AssertionError(f"New N1 adapter unexpectedly inherited B0 tensors: {transfer['adapter_loaded_items']}")
    load_and_verify(initialized, matched)
    inherited = component_coverage(initialized, matched)
    required_full = ("backbone_0_10", "neck_11_22", "detect_regression")
    failed = {name: inherited[name] for name in required_full if inherited[name]["coverage"] != 1.0}
    if failed:
        raise AssertionError(f"Required semantic inheritance incomplete: {failed}")
    adapter = initialized.model[23]
    if not isinstance(adapter, RoadAdaptiveScaleFusionN1):
        raise AssertionError(f"Invalid N1 topology: {type(adapter)}")
    if adapter.gamma.item() != torch.tensor(0.01).item():
        raise AssertionError(f"N1 gamma changed before training: {adapter.gamma.item()}")

    expected = {name: value.detach().cpu().clone() for name, value in initialized.state_dict().items()}
    expected_hash = state_sha256(expected)
    wrapper = YOLO(str(N1_YAML), task="detect")
    wrapper.model = initialized
    wrapper.ckpt = {"semantic_initialization": True}
    wrapper.overrides["model"] = str(N1_YAML)
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
            "matched_pretrained_items_preserved": all(torch.equal(actual[name].detach().cpu(), expected[name]) for name in matched),
        }
        (metadata_dir / "trainer_rebuild_audit.json").write_text(json.dumps(rebuild, indent=2), encoding="utf-8")
        if not rebuild["passed"] or not rebuild["matched_pretrained_items_preserved"]:
            raise AssertionError(f"Trainer reconstruction invalidated N1 initialization: {rebuild}")
        print(f"TRAINER_REBUILD_INITIALIZATION_VERIFIED sha256={actual_hash}")

    wrapper.add_callback("on_pretrain_routine_end", verify_trainer_rebuild)
    metadata = {
        "candidate": "RoadAdaptiveScaleFusion-N1 = unchanged pretrained B0 plus one gamma=0.01 P4 spatial scale adapter",
        "transfer": transfer,
        "component_coverage": inherited,
        "protocol": PROTOCOL,
        "model_yaml": str(N1_YAML.resolve()),
        "data": str(data),
        "command": shlex.join([sys.executable, *sys.argv]),
        "git": {"commit": command_output(["git", "rev-parse", "HEAD"]), "branch": command_output(["git", "branch", "--show-current"]),
                "status": command_output(["git", "status", "--short"])},
        "environment": {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__, "cuda": torch.version.cuda,
                        "cudnn": torch.backends.cudnn.version(), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},
        "test_split_read": False,
        "fresh_start": True,
    }
    (metadata_dir / "initialization_audit.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    shutil.copy2(N1_YAML, metadata_dir / "model_snapshot.yaml")
    shutil.copy2(data, metadata_dir / "data_snapshot.yaml")

    exit_code, save_dir = 1, None
    try:
        wrapper.train(data=str(data), project=str(project), name=args.name, exist_ok=False, resume=False, pretrained=False, **PROTOCOL)
        save_dir = Path(wrapper.trainer.save_dir)
        exit_code = 0
    finally:
        (metadata_dir / "exit_code.txt").write_text(f"{exit_code}\n", encoding="utf-8")
        if save_dir is None and getattr(wrapper, "trainer", None) is not None:
            save_dir = Path(wrapper.trainer.save_dir)
        if save_dir is not None and save_dir.exists():
            shutil.copytree(metadata_dir, save_dir / "runtime_meta", dirs_exist_ok=True)


if __name__ == "__main__":
    main()
