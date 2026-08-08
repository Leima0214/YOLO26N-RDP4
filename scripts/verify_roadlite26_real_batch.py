"""Verify RoadLite inherited checkpoints on one real labeled detection batch."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ultralytics import YOLO
from ultralytics.cfg import DEFAULT_CFG, get_cfg
from ultralytics.data.build import build_dataloader, build_yolo_dataset
from ultralytics.data.utils import check_det_dataset
from ultralytics.models.yolo.detect.train import DetectionTrainer


DEFAULT_WEIGHTS = (
    ROOT / "weights/roadlite26_inherited/roadlite26-a1-b0inherit-nc4.pt",
    ROOT / "weights/roadlite26_inherited/roadlite26-a2-b0inherit-nc4.pt",
)


def make_batch(data_yaml: Path, image_size: int, batch_size: int) -> tuple[dict, dict]:
    data = check_det_dataset(str(data_yaml))
    cfg = get_cfg(
        DEFAULT_CFG,
        {
            "mode": "train",
            "task": "detect",
            "imgsz": image_size,
            "batch": batch_size,
            "workers": 0,
            "rect": False,
            "cache": False,
            "mosaic": 0.0,
            "mixup": 0.0,
            "copy_paste": 0.0,
        },
    )
    dataset = build_yolo_dataset(cfg, data["train"], batch_size, data, mode="train", rect=False, stride=32)
    loader = build_dataloader(dataset, batch_size, 0, shuffle=False, rank=-1, pin_memory=False)
    return data, next(iter(loader))


def rebuild(wrapper: YOLO, data: dict):
    trainer = object.__new__(DetectionTrainer)
    trainer.data = {"nc": data["nc"], "channels": data.get("channels", 3)}
    model = DetectionTrainer.get_model(trainer, cfg=wrapper.model.yaml, weights=wrapper.model, verbose=False)
    expected, actual = wrapper.model.float().state_dict(), model.state_dict()
    changed = [name for name, value in expected.items() if not torch.equal(value, actual[name])]
    if changed:
        raise AssertionError(f"Trainer reconstruction changed tensors: {changed[:10]}")
    return model, len(expected)


def verify(weights: Path, data: dict, source_batch: dict) -> dict:
    wrapper = YOLO(str(weights))
    model, exact_keys = rebuild(wrapper, data)
    model.args = get_cfg(DEFAULT_CFG, {"box": 7.5, "cls": 0.5, "dfl": 1.5})
    model.criterion = None
    model.float().train().zero_grad(set_to_none=True)
    batch = {
        key: value.clone() if isinstance(value, torch.Tensor) else value
        for key, value in source_batch.items()
    }
    batch["img"] = batch["img"].float() / 255.0
    loss, components = model.loss(batch)
    if not torch.isfinite(loss).all() or not torch.isfinite(components).all():
        raise AssertionError(f"non-finite real-batch loss: {weights}")
    loss.sum().backward()

    parameters = dict(model.named_parameters())
    probes = (
        "model.0.conv.weight",
        "model.13.cv1.conv.weight",
        "model.23.cv2.0.0.conv.weight",
        "model.23.cv3.0.2.weight",
        "model.23.one2one_cv2.0.0.conv.weight",
        "model.23.one2one_cv3.0.2.weight",
    )
    for name in probes:
        gradient = parameters[name].grad
        if gradient is None or not torch.isfinite(gradient).all():
            raise AssertionError(f"missing/non-finite real-batch gradient: {name}")
    required_nonzero = {
        "backbone": ("model.0.",),
        "neck": ("model.13.",),
        "o2m_box": ("model.23.cv2.",),
        "o2m_cls": ("model.23.cv3.",),
        "o2o_box": ("model.23.one2one_cv2.",),
        "o2o_cls": ("model.23.one2one_cv3.",),
    }
    gradient_sums = {}
    for family, prefixes in required_nonzero.items():
        total = sum(
            parameter.grad.detach().abs().sum().item()
            for name, parameter in parameters.items()
            if name.startswith(prefixes) and parameter.grad is not None
        )
        if not total:
            raise AssertionError(f"zero real-batch gradient for whole branch family: {family}")
        gradient_sums[family] = total
    gradients = [parameter.grad for parameter in parameters.values() if parameter.grad is not None]
    if not gradients or not all(torch.isfinite(gradient).all() for gradient in gradients):
        raise AssertionError("one or more model gradients are non-finite")
    return {
        "weights": str(weights.resolve()),
        "trainer_rebuild_exact": f"{exact_keys}/{exact_keys}",
        "loss": float(loss.detach().sum()),
        "loss_components": [float(value) for value in components.detach().flatten()],
        "parameters_with_grad": len(gradients),
        "gradient_abs_sum_by_family": gradient_sums,
        "gradient_probes": list(probes),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "configs/japan4_clean_v3_local.yaml")
    parser.add_argument("--weights", nargs="+", type=Path, default=list(DEFAULT_WEIGHTS))
    parser.add_argument("--imgsz", type=int, default=128)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--report", type=Path, default=ROOT / "reports/roadlite26_inheritance/real_batch_verification.json"
    )
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    data, batch = make_batch(args.data.resolve(), args.imgsz, args.batch)
    if not len(batch["cls"]):
        raise AssertionError("selected real batch contains no labeled objects")
    results = [verify(path.resolve(), data, batch) for path in args.weights]
    report = {
        "data": str(args.data.resolve()),
        "images": list(batch["im_file"]),
        "image_shape": list(batch["img"].shape),
        "objects": int(len(batch["cls"])),
        "classes": sorted({int(value) for value in batch["cls"].flatten()}),
        "models": results,
        "ok": True,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
