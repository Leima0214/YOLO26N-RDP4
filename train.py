"""Matched Japan4-Clean training entry; switch MODEL to select an experiment."""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch

from ultralytics import YOLO
from ultralytics.nn.rd_adapter import RDP3Stage

ROOT = Path(__file__).resolve().parent

# Change MODEL only. RUN_NAME=None derives a distinct name from the selected YAML.
MODEL = "ultralytics/cfg/models/26/yolo26.yaml"  # B0
# MODEL = "ultralytics/cfg/models/26/yolo26n-japan4-som.yaml"
# MODEL = "ultralytics/cfg/models/26/yolo26n-japan4-maf.yaml"
# MODEL = "ultralytics/cfg/models/26/yolo26n-japan4-wtc.yaml"
# MODEL = "ultralytics/cfg/models/26/yolo26n-japan4-som-maf.yaml"
# MODEL = "ultralytics/cfg/models/26/yolo26n-japan4-som-wtc.yaml"
# MODEL = "ultralytics/cfg/models/26/yolo26n-japan4-maf-wtc.yaml"
# MODEL = "ultralytics/cfg/models/26/yolo26n-japan4-som-maf-wtc.yaml"
RUN_NAME = None
DATA = "configs/japan4_clean_remote.yaml"
WEIGHTS = "yolo26n.pt"
EPOCHS = 30
BATCH = 32
DEVICE = "0"
SEED = 42


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_rd_dictionary(path: Path, weights: Path) -> torch.Tensor:
    assert path.is_file(), path
    try:
        package = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        package = torch.load(path, map_location="cpu")
    assert isinstance(package, dict) and isinstance(package.get("centroids"), torch.Tensor)
    meta = package.get("meta", {})
    assert meta.get("layer") == 4 and meta.get("stride") == 8, meta
    assert meta.get("weights_sha256") == sha256(weights), "dictionary/checkpoint SHA256 mismatch"
    return package["centroids"]


def initialize_rd_if_requested(model: YOLO, weights: Path) -> None:
    dictionary = model.model.yaml.get("rd_dictionary")
    if not dictionary:
        return
    if "," in DEVICE:
        raise ValueError("R2 dictionary initialization requires one GPU so DDP cannot discard in-memory atoms")
    stage = model.model.model[4]
    assert isinstance(stage, RDP3Stage), type(stage)
    stage.rd.initialize_atoms(load_rd_dictionary(resolve(dictionary), weights))
    expected = {name: value.detach().cpu().clone() for name, value in stage.rd.state_dict().items()}

    def check(trainer) -> None:
        trained = trainer.model.model[4]
        assert isinstance(trained, RDP3Stage), type(trained)
        actual = trained.rd.state_dict()
        assert actual.keys() == expected.keys()
        for name, value in actual.items():
            assert torch.equal(value.detach().cpu(), expected[name]), (
                f"Trainer reconstruction changed initialized RD state: {name}"
            )

    model.add_callback("on_pretrain_routine_end", check)


def main() -> None:
    model_path, data_path, weights = resolve(MODEL), resolve(DATA), resolve(WEIGHTS)
    assert model_path.is_file() and data_path.is_file() and weights.is_file()
    model = YOLO(str(model_path), task="detect")
    model.load(str(weights))
    initialize_rd_if_requested(model, weights)
    model.train(
        data=str(data_path),
        project=str(ROOT / "runs/paper1_japan4_clean"),
        name=RUN_NAME or f"{model_path.stem}_japan4_clean_{EPOCHS}e_seed{SEED}",
        epochs=EPOCHS,
        imgsz=640,
        batch=BATCH,
        device=DEVICE,
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
        conf=0.001,
        iou=0.7,
        max_det=300,
    )


if __name__ == "__main__":
    main()
