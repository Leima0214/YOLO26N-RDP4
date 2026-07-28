"""Run matched Japan7 B0/R1/R2 screens for clean YOLO26n Retriever-Dictionary candidates."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) in sys.path:
    sys.path.remove(str(ROOT))
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from ultralytics import YOLO  # noqa: E402
from ultralytics.nn.rd_adapter import RDP3Stage  # noqa: E402

BASELINE_MODEL = ROOT / "ultralytics/cfg/models/26/yolo26.yaml"
RD_P3_MODEL = ROOT / "ultralytics/cfg/models/26/yolo26n-rd-p3-japan7.yaml"
PROJECT = ROOT / "runs/paper1_yolo_rd"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("B0", "R1", "R2"),
        required=True,
        help="B0=baseline, R1=P3 random dictionary, R2=P3 Japan7-initialized dictionary.",
    )
    parser.add_argument("--data", default="configs/japan7_remote.yaml")
    parser.add_argument("--weights", default="yolo26n.pt")
    parser.add_argument("--dictionary", default="reports/japan7_rd_p3_atoms.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--name", default=None)
    return parser.parse_args()


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_dictionary(path: Path, weights: Path) -> torch.Tensor:
    assert path.is_file(), path
    try:
        package = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        package = torch.load(path, map_location="cpu")
    assert isinstance(package, dict) and isinstance(
        package.get("centroids"), torch.Tensor
    ), path
    meta = package.get("meta", {})
    assert meta.get("layer") == 4 and meta.get("stride") == 8, meta
    assert meta.get("weights_sha256") == sha256(weights), (
        "dictionary/checkpoint SHA256 mismatch"
    )
    return package["centroids"]


def build_model(stage: str, weights: Path, dictionary: Path) -> YOLO:
    yaml = BASELINE_MODEL if stage == "B0" else RD_P3_MODEL
    assert yaml.is_file(), yaml
    assert weights.is_file(), weights
    model = YOLO(str(yaml), task="detect")
    model.load(str(weights))
    if stage != "B0":
        rd_stage = model.model.model[4]
        assert (
            isinstance(rd_stage, RDP3Stage)
            and abs(rd_stage.rd.gamma.item() - 1e-3) < 1e-8
        ), "RD construction failed"
        assert any(
            name.startswith("model.4.rd.") for name in model.model.state_dict()
        ), "RD state is missing"
        if stage == "R2":
            rd_stage.rd.initialize_atoms(load_dictionary(dictionary, weights))
            norms = rd_stage.rd.dictionary.normalized_weight().flatten(2).norm(dim=0)
            assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5, rtol=0), (
                "dictionary atoms are not normalized"
            )
    return model


def guard_r2_transfer(model: YOLO) -> None:
    """Fail before epoch 1 if Trainer reconstruction drops the initialized RD core."""
    rd = model.model.model[4].rd
    expected = {
        "coefficient": rd.coefficient.conv.weight.detach().cpu().clone(),
        "exchange": rd.exchange.conv.weight.detach().cpu().clone(),
        "dictionary": rd.dictionary.weight.detach().cpu().clone(),
        "gamma": rd.gamma.detach().cpu().clone(),
    }

    def check(trainer) -> None:
        trained = trainer.model.model[4]
        assert isinstance(trained, RDP3Stage), type(trained)
        actual = {
            "coefficient": trained.rd.coefficient.conv.weight,
            "exchange": trained.rd.exchange.conv.weight,
            "dictionary": trained.rd.dictionary.weight,
            "gamma": trained.rd.gamma,
        }
        for name, value in actual.items():
            assert torch.equal(value.detach().cpu(), expected[name]), (
                f"Trainer reconstruction changed initialized RD parameter: {name}"
            )

    model.add_callback("on_pretrain_routine_end", check)


def main() -> None:
    args = parse_args()
    data, weights, dictionary = (
        resolve(args.data),
        resolve(args.weights),
        resolve(args.dictionary),
    )
    assert data.is_file(), data
    assert args.epochs > 0 and args.batch > 0, (args.epochs, args.batch)
    if args.stage == "R2" and "," in args.device:
        raise ValueError(
            "R2 dictionary initialization requires one GPU so DDP cannot discard in-memory atoms"
        )
    model = build_model(args.stage, weights, dictionary)
    if args.stage == "R2":
        guard_r2_transfer(model)
    run_name = (
        args.name
        or f"yolord_{args.stage.lower()}_japan7_{args.epochs}e_seed{args.seed}"
    )
    model.train(
        data=str(data),
        project=str(PROJECT),
        name=run_name,
        epochs=args.epochs,
        imgsz=640,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        seed=args.seed,
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
