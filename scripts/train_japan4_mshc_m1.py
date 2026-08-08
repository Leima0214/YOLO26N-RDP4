"""Protocol-locked fresh 30E training for Japan4 M1 (P3 MSHC + P4 MSHC)."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import shlex
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from audit_japan4_parent_static import load_matched, transfer_audit  # noqa: E402
from ultralytics import YOLO  # noqa: E402
from ultralytics.nn.tasks import DetectionModel  # noqa: E402
from ultralytics.nn.yolo26_cvpr_improvements.modules import MSHCBlock  # noqa: E402

MODEL_YAML = ROOT / "ultralytics/cfg/models/26/yolo26-MSHC.yaml"
MONITOR_EPOCHS = {1, 5, 10, 20, 30}
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


def state_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def validate_data(path: Path) -> None:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    names = data.get("names", {})
    names = list(names.values()) if isinstance(names, dict) else list(names)
    if data.get("nc") != 4 or names != ["D00", "D10", "D20", "D40"]:
        raise ValueError(f"Expected Japan4-cleanV3, got nc={data.get('nc')} names={names}")
    rendered = path.read_text(encoding="utf-8").lower()
    if "test:" in rendered:
        # A test entry may exist in the data YAML, but this run never invokes it.
        print("DATA_YAML_CONTAINS_TEST_ENTRY_BUT_RUN_IS_VAL_ONLY")


class MSHCPassiveMonitor:
    """Observe one normal training batch at selected epochs without adding model passes or losses."""

    def __init__(self, metadata_dir: Path):
        self.metadata_dir = metadata_dir
        self.active = False
        self.current_epoch = 0
        self.activations: dict[str, list[dict[str, float]]] = defaultdict(list)
        self.grad_sq: dict[str, float] = defaultdict(float)
        self.handles = []
        self.records: list[dict] = []
        self.block_parameter_counts: dict[str, int] = {}
        self.trainer = None
        self.amp_scale_at_backward = 1.0

    @staticmethod
    def _tensor_stats(output: torch.Tensor) -> dict[str, float]:
        value = output.detach().float()
        return {
            "l2_norm": float(torch.linalg.vector_norm(value).item()),
            "rms": float(torch.sqrt(torch.mean(value.square())).item()),
            "mean": float(value.mean().item()),
            "std": float(value.std(unbiased=False).item()),
        }

    def install(self, trainer) -> None:
        self.trainer = trainer
        blocks = {"P3_layer4": trainer.model.model[4], "P4_layer6": trainer.model.model[6]}
        if not all(isinstance(block, MSHCBlock) for block in blocks.values()):
            raise TypeError({name: type(block).__name__ for name, block in blocks.items()})

        for block_name, block in blocks.items():
            modules = {
                "square_3x3": block.square[0],
                "square_5x5": block.square[1],
                "square_7x7": block.square[2],
                "horizontal_1x7": block.horizontal,
                "vertical_7x1": block.vertical,
                "gate": block.gate,
            }
            for branch_name, module in modules.items():
                key = f"{block_name}.{branch_name}"

                def activation_hook(_module, _inputs, output, capture_key=key):
                    if self.active:
                        self.activations[capture_key].append(self._tensor_stats(output))

                self.handles.append(module.register_forward_hook(activation_hook))

            parameters = list(block.named_parameters())
            self.block_parameter_counts[block_name] = sum(parameter.numel() for _, parameter in parameters)
            for parameter_name, parameter in parameters:
                group = parameter_name.split(".", 1)[0]
                keys = (f"{block_name}.all", f"{block_name}.{group}")

                def gradient_hook(gradient, capture_keys=keys):
                    if self.active:
                        self.amp_scale_at_backward = (
                            float(self.trainer.scaler.get_scale()) if getattr(self.trainer, "amp", False) else 1.0
                        )
                        norm_sq = float(gradient.detach().float().square().sum().item())
                        for capture_key in capture_keys:
                            self.grad_sq[capture_key] += norm_sq
                    return gradient

                self.handles.append(parameter.register_hook(gradient_hook))

    def epoch_start(self, trainer) -> None:
        epoch = int(trainer.epoch) + 1
        self.current_epoch = epoch
        self.active = epoch in MONITOR_EPOCHS
        if self.active:
            self.activations.clear()
            self.grad_sq.clear()
            self.amp_scale_at_backward = 1.0

    def batch_end(self, trainer) -> None:
        if not self.active:
            return
        amp_scale = self.amp_scale_at_backward
        activation_summary = {}
        for key, values in sorted(self.activations.items()):
            activation_summary[key] = {
                metric: float(sum(item[metric] for item in values) / len(values))
                for metric in ("l2_norm", "rms", "mean", "std")
            }
        gradient_summary = {
            key: math.sqrt(value) / amp_scale for key, value in sorted(self.grad_sq.items())
        }
        record = {
            "epoch": self.current_epoch,
            "sample": "first normal augmented training batch",
            "extra_forward_or_backward": False,
            "amp_scale": amp_scale,
            "activations": activation_summary,
            "gradient_l2_norm_unscaled": gradient_summary,
            "mshc_parameter_counts": self.block_parameter_counts,
        }
        self.records.append(record)
        (self.metadata_dir / "mshc_monitor.json").write_text(json.dumps(self.records, indent=2), encoding="utf-8")
        with (self.metadata_dir / "mshc_monitor.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        print(f"MSHC_PASSIVE_MONITOR epoch={self.current_epoch} record={json.dumps(record, separators=(',', ':'))}")
        self.active = False

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yolo-weights", type=Path, default=ROOT / "yolo26n.pt")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--project", type=Path, default=ROOT / "runs/paper1_japan4_clean")
    parser.add_argument("--monitor", action="store_true", help="Enable optional passive MSHC diagnostics")
    args = parser.parse_args()
    yolo_weights = args.yolo_weights.expanduser().resolve()
    data = args.data.expanduser().resolve()
    project = args.project.expanduser().resolve()
    if not all(path.is_file() for path in (MODEL_YAML, yolo_weights, data)):
        raise FileNotFoundError((MODEL_YAML, yolo_weights, data))
    validate_data(data)
    metadata_dir = ROOT / "runtime_meta" / args.name
    if metadata_dir.exists() or (project / args.name).exists():
        raise FileExistsError(f"Fresh-start protection: run already exists: {args.name}")

    torch.manual_seed(PROTOCOL["seed"])
    initialized = DetectionModel(str(MODEL_YAML), ch=3, nc=4, verbose=False).float()
    transfer, matched = transfer_audit(initialized, yolo_weights)
    from verify_japan4_mshc import coverage, semantic_safety  # local import avoids audit side effects

    safety = semantic_safety(initialized, yolo_weights, matched)
    if not safety["passed"]:
        raise AssertionError(f"Unsafe YOLO26 transfer: {safety}")
    load_matched(initialized, matched)
    component_coverage = coverage(initialized, matched)
    expected = {name: value.detach().cpu().clone() for name, value in initialized.state_dict().items()}
    expected_hash = state_sha256(expected)

    wrapper = YOLO(str(MODEL_YAML), task="detect")
    wrapper.model = initialized
    wrapper.ckpt = {"semantic_initialization": True}
    wrapper.overrides["model"] = str(MODEL_YAML)
    metadata_dir.mkdir(parents=True)
    monitor = MSHCPassiveMonitor(metadata_dir)

    def verify_trainer_rebuild(trainer) -> None:
        actual = trainer.model.state_dict()
        mismatched = [name for name, value in expected.items() if not torch.equal(value, actual[name].detach().cpu())]
        actual_hash = state_sha256(actual)
        rebuild = {
            "passed": not mismatched and actual_hash == expected_hash,
            "before_sha256": expected_hash,
            "after_sha256": actual_hash,
            "mismatched": mismatched,
            "matched_pretrained_items_preserved": all(
                torch.equal(actual[name].detach().cpu(), expected[name]) for name in matched
            ),
        }
        (metadata_dir / "trainer_rebuild_audit.json").write_text(json.dumps(rebuild, indent=2), encoding="utf-8")
        if not rebuild["passed"] or not rebuild["matched_pretrained_items_preserved"]:
            raise AssertionError(f"Trainer reconstruction invalidated initialization: {rebuild}")
        if args.monitor:
            monitor.install(trainer)
        print(f"TRAINER_REBUILD_INITIALIZATION_VERIFIED sha256={actual_hash}")

    wrapper.add_callback("on_pretrain_routine_end", verify_trainer_rebuild)
    if args.monitor:
        wrapper.add_callback("on_train_epoch_start", monitor.epoch_start)
        wrapper.add_callback("on_train_batch_end", monitor.batch_end)
    metadata = {
        "candidate": "M1 = one MSHC at P3 plus one MSHC at P4",
        "transfer": transfer,
        "semantic_safety": safety,
        "component_coverage": component_coverage,
        "protocol": PROTOCOL,
        "monitor_epochs": sorted(MONITOR_EPOCHS) if args.monitor else [],
        "monitor_enabled": args.monitor,
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
        if args.monitor:
            monitor.close()
        (metadata_dir / "exit_code.txt").write_text(f"{exit_code}\n", encoding="utf-8")
        if save_dir is None and getattr(wrapper, "trainer", None) is not None:
            save_dir = Path(wrapper.trainer.save_dir)
        if save_dir is not None and save_dir.exists():
            shutil.copytree(metadata_dir, save_dir / "runtime_meta", dirs_exist_ok=True)


if __name__ == "__main__":
    main()
