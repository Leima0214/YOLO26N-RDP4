"""Single protocol-locked training entry for Japan4 candidates."""

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

MODELS = {
    "b0": ROOT / "ultralytics/cfg/models/26/yolo26.yaml",
    "s1": ROOT / "ultralytics/cfg/models/26/yolo26n-japan4-s1-strip-regression.yaml",
    "s2": ROOT / "ultralytics/cfg/models/26/yolo26n-japan4-s2-shape-strip.yaml",
    "g1": ROOT / "ultralytics/cfg/models/26/yolo26n-japan4-g1-region-guidance.yaml",
    "gs1": ROOT / "ultralytics/cfg/models/26/yolo26n-japan4-gs1-region-strip.yaml",
}
TRAINING_PROTOCOL = {
    "optimizer": "auto",
    "amp": True,
    "deterministic": True,
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
    """Run a read-only metadata command without making training depend on it."""
    return subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False).stdout.strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True, choices=tuple(MODELS))
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--epochs", type=int, required=True, choices=(1, 5, 30, 100))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--name", required=True)
    parser.add_argument("--project", type=Path, default=ROOT / "runs/paper1_japan4_clean")
    parser.add_argument("--weights", type=Path, default=ROOT / "yolo26n.pt")
    parser.add_argument("--resume-from", type=Path)
    return parser.parse_args()


def validate_resume(args: argparse.Namespace, model_yaml: Path) -> Path:
    """Allow only last.pt from the same candidate and original total-epoch schedule."""
    checkpoint = args.resume_from.expanduser().resolve()
    if checkpoint.name != "last.pt" or not checkpoint.is_file():
        raise ValueError("--resume-from must point to an existing last.pt")
    run_args_path = checkpoint.parents[1] / "args.yaml"
    if not run_args_path.is_file():
        raise FileNotFoundError(run_args_path)
    run_args = yaml.safe_load(run_args_path.read_text(encoding="utf-8"))
    if int(run_args.get("epochs", -1)) != args.epochs:
        raise ValueError("Resume total epochs must equal the original run; 30E cannot be extended into formal 100E")
    if Path(str(run_args.get("model", ""))).name != model_yaml.name:
        raise ValueError(f"Resume checkpoint is not from candidate {args.candidate}")
    expected = {
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "seed": args.seed,
        "name": args.name,
    }
    mismatched = {key: (run_args.get(key), value) for key, value in expected.items() if run_args.get(key) != value}
    if mismatched:
        raise ValueError(f"Resume protocol differs from the original run: {mismatched}")
    return checkpoint


def validate_data_yaml(path: Path) -> None:
    """Reject accidental use of a non-Japan4 protocol before training starts."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    names = data.get("names", {})
    names = list(names.values()) if isinstance(names, dict) else list(names)
    if data.get("nc") != 4 or names != ["D00", "D10", "D20", "D40"]:
        raise ValueError(f"Expected frozen Japan4-cleanV3 classes, got nc={data.get('nc')} names={names}")


def save_metadata(args: argparse.Namespace, model_yaml: Path, data_yaml: Path, metadata_dir: Path) -> None:
    """Snapshot reproducibility metadata before GPU work starts."""
    metadata_dir.mkdir(parents=True, exist_ok=True)
    resolved = {
        **vars(args),
        "data": str(data_yaml),
        "project": str(args.project),
        "weights": str(args.weights),
        "resume_from": str(args.resume_from) if args.resume_from else None,
        "model_yaml": str(model_yaml),
        **TRAINING_PROTOCOL,
    }
    (metadata_dir / "resolved_arguments.json").write_text(json.dumps(resolved, indent=2, default=str), encoding="utf-8")
    (metadata_dir / "command.txt").write_text(shlex.join([sys.executable, *sys.argv]) + "\n", encoding="utf-8")
    shutil.copy2(data_yaml, metadata_dir / "data_snapshot.yaml")
    shutil.copy2(model_yaml, metadata_dir / "model_snapshot.yaml")
    git = {
        "commit": command_output(["git", "rev-parse", "HEAD"]),
        "branch": command_output(["git", "branch", "--show-current"]),
        "status": command_output(["git", "status", "--short"]),
    }
    (metadata_dir / "git.json").write_text(json.dumps(git, indent=2), encoding="utf-8")
    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    (metadata_dir / "environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")
    (metadata_dir / "pip_freeze.txt").write_text(
        command_output([sys.executable, "-m", "pip", "freeze"]) + "\n", encoding="utf-8"
    )
    (metadata_dir / "nvidia_smi.txt").write_text(command_output(["nvidia-smi"]) + "\n", encoding="utf-8")
    print(json.dumps({"resolved_arguments": resolved, "git": git, "environment": environment}, indent=2, default=str))


def main() -> None:
    args = parse_args()
    model_yaml = MODELS[args.candidate].resolve()
    data_yaml = args.data.expanduser().resolve()
    args.project = args.project.expanduser().resolve()
    args.weights = args.weights.expanduser().resolve()
    if not model_yaml.is_file() or not data_yaml.is_file():
        raise FileNotFoundError(f"Missing model or data YAML: {model_yaml}, {data_yaml}")
    validate_data_yaml(data_yaml)
    if args.epochs == 5 and args.candidate != "s2":
        raise ValueError("The 5E mechanism gate is reserved for S2")
    if args.imgsz != 640 or args.batch != 32 or args.workers != 8 or args.seed != 42:
        raise ValueError("Japan4 formal protocol is fixed to imgsz=640, batch=32, workers=8, seed=42")

    resume = validate_resume(args, model_yaml) if args.resume_from else None
    if not resume and not args.weights.is_file():
        raise FileNotFoundError(args.weights)
    metadata_dir = ROOT / "runtime_meta" / args.name
    if metadata_dir.exists() and not resume:
        raise FileExistsError(f"Refusing to overwrite existing metadata: {metadata_dir}")
    save_metadata(args, model_yaml, data_yaml, metadata_dir)

    exit_code = 1
    save_dir = None
    try:
        if resume:
            model = YOLO(str(resume), task="detect")
            model.train(resume=str(resume))
        else:
            model = YOLO(str(model_yaml), task="detect").load(str(args.weights))
            model.train(
                data=str(data_yaml),
                epochs=args.epochs,
                imgsz=args.imgsz,
                batch=args.batch,
                workers=args.workers,
                device=args.device,
                seed=args.seed,
                **TRAINING_PROTOCOL,
                project=str(args.project),
                name=args.name,
                exist_ok=False,
            )
        exit_code = 0
        save_dir = Path(model.trainer.save_dir)
    finally:
        (metadata_dir / "exit_code.txt").write_text(f"{exit_code}\n", encoding="utf-8")
        if save_dir is not None:
            shutil.copytree(metadata_dir, save_dir / "runtime_meta", dirs_exist_ok=True)


if __name__ == "__main__":
    main()
