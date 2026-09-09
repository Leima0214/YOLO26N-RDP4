#!/usr/bin/env python3
"""Training-free P2-to-P3 early-detail survival audit for official YOLO12n.

The audit reads SVRDD7 Val only. It triangulates target/background contrast,
image-grouped frozen linear probes, and association with frozen detection
outcomes. It never trains or changes the detector and emits no visualizations.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT / "scripts"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from rs_mid_bootstrap import guard_optional_visualization_imports  # noqa: E402

guard_optional_visualization_imports()

from rs_mid_o2m import NAMES, ValCOCOEvaluator  # noqa: E402
from ultralytics.nn.modules.head import Detect  # noqa: E402
from ultralytics.nn.tasks import load_checkpoint  # noqa: E402
from ultralytics.utils.ops import xywh2xyxy  # noqa: E402
from ultralytics.utils.torch_utils import select_device  # noqa: E402


STAGES = ("P2", "P3", "P4")
STRIDES = {"P2": 4, "P3": 8, "P4": 16}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--outcomes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260909)
    parser.add_argument("--max-images", type=int, default=0)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_value(*args: str) -> str:
    result = subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else f"unavailable: {result.stderr.strip()}"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_outcomes(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    rows: dict[tuple[str, int], dict[str, Any]] = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for raw in csv.DictReader(handle):
            key = (raw["image"], int(raw["gt_index"]))
            best_raw_iou = float(raw["best_raw_iou"])
            if int(raw["hit50"]):
                outcome = "hit"
            elif raw["death_reason"] == "classification_mismatch":
                outcome = "classification_error"
            elif best_raw_iou < 0.10:
                outcome = "complete_miss"
            else:
                outcome = "localization_error"
            rows[key] = {
                **raw,
                "best_raw_iou": best_raw_iou,
                "top_score_iou": float(raw["top_score_iou"]),
                "outcome": outcome,
            }
    if len(rows) != 2612:
        raise RuntimeError(f"Expected 2612 frozen GT outcomes, got {len(rows)}")
    return rows


def discover_stage_layers(model: torch.nn.Module, images: torch.Tensor) -> tuple[dict[str, int], dict[int, torch.Tensor]]:
    """Find the last backbone tensor at strides 4/8/16 from real output shapes."""
    first_upsample = next(
        (i for i, module in enumerate(model.model) if isinstance(module, torch.nn.Upsample)),
        len(model.model) - 1,
    )
    captured: dict[int, torch.Tensor] = {}
    hooks = []

    def hook(index: int):
        def capture(_module, _inputs, output):
            if isinstance(output, torch.Tensor) and output.ndim == 4:
                captured[index] = output
        return capture

    for index, module in enumerate(model.model[:first_upsample]):
        hooks.append(module.register_forward_hook(hook(index)))
    try:
        with torch.inference_mode():
            model(images)
    finally:
        for item in hooks:
            item.remove()

    mapping: dict[str, int] = {}
    input_h, input_w = images.shape[-2:]
    for stage, stride in STRIDES.items():
        candidates = [
            index
            for index, feature in captured.items()
            if tuple(feature.shape[-2:]) == (input_h // stride, input_w // stride)
        ]
        if not candidates:
            raise RuntimeError(f"Could not discover {stage} stride={stride}; shapes={[(i, x.shape) for i, x in captured.items()]}")
        mapping[stage] = max(candidates)
    if len(set(mapping.values())) != len(mapping):
        raise RuntimeError(f"Stage discovery was not unique: {mapping}")
    return mapping, captured


def sample_feature(
    feature: torch.Tensor, box: torch.Tensor, image_hw: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Sample target interior and an expanded perimeter ring with bilinear interpolation."""
    channels, _, _ = feature.shape
    mean = feature.mean(dim=(1, 2), keepdim=True)
    std = feature.std(dim=(1, 2), keepdim=True, unbiased=False).clamp_min(1e-6)
    normalized = (feature - mean) / std

    x1, y1, x2, y2 = [float(value) for value in box]
    width, height = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
    center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
    inner_fractions = (0.2, 0.5, 0.8)
    inner = [(x1 + width * fx, y1 + height * fy) for fy in inner_fractions for fx in inner_fractions]

    ring_scale = 1.8
    rx1 = max(0.0, center_x - width * ring_scale / 2)
    ry1 = max(0.0, center_y - height * ring_scale / 2)
    image_h, image_w = image_hw
    rx2 = min(float(image_w) - 1e-3, center_x + width * ring_scale / 2)
    ry2 = min(float(image_h) - 1e-3, center_y + height * ring_scale / 2)
    edge_fractions = (0.1, 0.3, 0.5, 0.7, 0.9)
    ring = []
    for fraction in edge_fractions:
        px = rx1 + (rx2 - rx1) * fraction
        py = ry1 + (ry2 - ry1) * fraction
        ring.extend(((px, ry1), (px, ry2), (rx1, py), (rx2, py)))

    def interpolate(points: list[tuple[float, float]]) -> torch.Tensor:
        grid = torch.tensor(
            [[[[(2.0 * x / image_w - 1.0), (2.0 * y / image_h - 1.0)] for x, y in points]]],
            dtype=normalized.dtype,
            device=normalized.device,
        )
        values = F.grid_sample(
            normalized.unsqueeze(0), grid, mode="bilinear", padding_mode="border", align_corners=False
        )
        return values.reshape(channels, -1)

    target_values = interpolate(inner)
    ring_values = interpolate(ring)
    target_vector = target_values.mean(dim=1)
    ring_vector = ring_values.mean(dim=1)
    target_response = float(target_values.abs().mean())
    ring_response = float(ring_values.abs().mean())
    return (
        target_vector.float().cpu().numpy(),
        ring_vector.float().cpu().numpy(),
        target_response,
        ring_response,
    )


def stable_fold(image: str, folds: int) -> int:
    return int(hashlib.sha1(image.encode()).hexdigest()[:8], 16) % folds


def rank_auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    positive = labels == 1
    negative = labels == 0
    n_pos, n_neg = int(positive.sum()), int(negative.sum())
    if not n_pos or not n_neg:
        return None
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    start = 0
    while start < len(scores):
        stop = start + 1
        while stop < len(scores) and scores[order[stop]] == scores[order[start]]:
            stop += 1
        ranks[order[start:stop]] = (start + stop + 1) / 2
        start = stop
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def cross_validated_probe(items: list[dict[str, Any]], stage: str, folds: int) -> dict[str, Any]:
    if len(items) < max(20, folds * 2):
        return {"n_gt": len(items), "auc": None, "balanced_accuracy": None, "paired_accuracy": None, "margin": None}
    labels_all, scores_all, pair_correct, margins = [], [], [], []
    for fold in range(folds):
        train = [item for item in items if stable_fold(item["image"], folds) != fold]
        test = [item for item in items if stable_fold(item["image"], folds) == fold]
        if not train or not test:
            continue
        positives = np.stack([item[f"{stage}_target_vector"] for item in train])
        negatives = np.stack([item[f"{stage}_ring_vector"] for item in train])
        mean_pos, mean_neg = positives.mean(0), negatives.mean(0)
        pooled_var = 0.5 * (positives.var(0) + negatives.var(0))
        ridge = max(float(np.median(pooled_var)) * 0.1, 1e-3)
        weight = (mean_pos - mean_neg) / (pooled_var + ridge)
        center = 0.5 * (mean_pos + mean_neg)
        for item in test:
            positive_score = float((item[f"{stage}_target_vector"] - center) @ weight)
            negative_score = float((item[f"{stage}_ring_vector"] - center) @ weight)
            labels_all.extend((1, 0))
            scores_all.extend((positive_score, negative_score))
            pair_correct.append(float(positive_score > negative_score))
            margins.append(positive_score - negative_score)
    labels = np.asarray(labels_all, dtype=int)
    scores = np.asarray(scores_all, dtype=float)
    if not len(labels):
        return {"n_gt": len(items), "auc": None, "balanced_accuracy": None, "paired_accuracy": None, "margin": None}
    predictions = scores > 0
    tpr = float(predictions[labels == 1].mean())
    tnr = float((~predictions[labels == 0]).mean())
    return {
        "n_gt": len(items),
        "auc": rank_auc(labels, scores),
        "balanced_accuracy": (tpr + tnr) / 2,
        "paired_accuracy": float(np.mean(pair_correct)),
        "margin": float(np.mean(margins)),
    }


def mean_ci_by_image(
    items: list[dict[str, Any]], value: Callable[[dict[str, Any]], float], repeats: int, seed: int
) -> dict[str, Any]:
    by_image: dict[str, list[float]] = defaultdict(list)
    for item in items:
        by_image[item["image"]].append(float(value(item)))
    if not by_image:
        return {"n_gt": 0, "n_images": 0, "mean": None, "ci95_low": None, "ci95_high": None}
    images = sorted(by_image)
    sums = np.asarray([sum(by_image[image]) for image in images], dtype=float)
    counts = np.asarray([len(by_image[image]) for image in images], dtype=float)
    estimate = float(sums.sum() / counts.sum())
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(repeats):
        draw = rng.integers(0, len(images), len(images))
        samples.append(float(sums[draw].sum() / counts[draw].sum()))
    return {
        "n_gt": int(counts.sum()),
        "n_images": len(images),
        "mean": estimate,
        "ci95_low": float(np.quantile(samples, 0.025)),
        "ci95_high": float(np.quantile(samples, 0.975)),
    }


def difference_ci_by_image(
    items: list[dict[str, Any]],
    first: Callable[[dict[str, Any]], bool],
    second: Callable[[dict[str, Any]], bool],
    metric: str,
    repeats: int,
    seed: int,
) -> dict[str, Any]:
    images = sorted({item["image"] for item in items})
    if not images:
        return {"mean": None, "ci95_low": None, "ci95_high": None}
    stats = {}
    for image in images:
        image_items = [item for item in items if item["image"] == image]
        a = [float(item[metric]) for item in image_items if first(item)]
        b = [float(item[metric]) for item in image_items if second(item)]
        stats[image] = (sum(a), len(a), sum(b), len(b))
    array = np.asarray([stats[image] for image in images], dtype=float)
    if not array[:, 1].sum() or not array[:, 3].sum():
        return {"mean": None, "ci95_low": None, "ci95_high": None}
    estimate = float(array[:, 0].sum() / array[:, 1].sum() - array[:, 2].sum() / array[:, 3].sum())
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(repeats):
        draw = array[rng.integers(0, len(array), len(array))]
        if draw[:, 1].sum() and draw[:, 3].sum():
            samples.append(float(draw[:, 0].sum() / draw[:, 1].sum() - draw[:, 2].sum() / draw[:, 3].sum()))
    return {
        "mean": estimate,
        "ci95_low": float(np.quantile(samples, 0.025)),
        "ci95_high": float(np.quantile(samples, 0.975)),
    }


def group_filters() -> list[tuple[str, Callable[[dict[str, Any]], bool]]]:
    groups: list[tuple[str, Callable[[dict[str, Any]], bool]]] = [
        ("overall", lambda _item: True),
        ("small", lambda item: item["size"] == "small"),
        ("medium", lambda item: item["size"] == "medium"),
        ("large", lambda item: item["size"] == "large"),
        ("elongated_ar_ge_5", lambda item: item["aspect_ratio"] >= 5),
        ("small_hit", lambda item: item["size"] == "small" and item["outcome"] == "hit"),
        ("small_nonhit", lambda item: item["size"] == "small" and item["outcome"] != "hit"),
        ("small_localization_error", lambda item: item["size"] == "small" and item["outcome"] == "localization_error"),
        ("small_complete_miss", lambda item: item["size"] == "small" and item["outcome"] == "complete_miss"),
        ("elongated_hit", lambda item: item["aspect_ratio"] >= 5 and item["outcome"] == "hit"),
        ("elongated_nonhit", lambda item: item["aspect_ratio"] >= 5 and item["outcome"] != "hit"),
    ]
    for name in NAMES:
        groups.append((f"class:{name}", lambda item, name=name: item["class"] == name))
    return groups


def make_report(payload: dict[str, Any]) -> str:
    lines = [
        "# YOLO12n P2-to-P3 early-detail survival diagnosis",
        "",
        f"Decision: **{payload['decision']['route']}**",
        "",
        "This is a training-free SVRDD7 Val audit. Test was not read and no visualization was generated.",
        "",
        "## Discovered stages",
        "",
        "| Stage | Layer | Module | Stride | Shape | Channels |",
        "|---|---:|---|---:|---|---:|",
    ]
    for stage in STAGES:
        item = payload["stages"][stage]
        lines.append(
            f"| {stage} | {item['layer']} | {item['module']} | {item['stride']} | {item['shape']} | {item['channels']} |"
        )
    lines.extend([
        "",
        "## Small-object evidence",
        "",
        "| Group | Stage | n GT | Contrast mean | 95% CI | Probe AUC | Paired accuracy |",
        "|---|---|---:|---:|---|---:|---:|",
    ])
    contrast = {(row["group"], row["stage"]): row for row in payload["contrast"]}
    probes = {(row["group"], row["stage"]): row for row in payload["probes"]}
    for group in ("small", "small_hit", "small_nonhit", "small_localization_error", "small_complete_miss", "elongated_ar_ge_5"):
        for stage in STAGES:
            c, p = contrast[(group, stage)], probes[(group, stage)]
            auc = "NA" if p["auc"] is None else f"{p['auc']:.4f}"
            pair = "NA" if p["paired_accuracy"] is None else f"{p['paired_accuracy']:.4f}"
            mean = "NA" if c["mean"] is None else f"{c['mean']:.5f}"
            interval = "NA" if c["ci95_low"] is None else f"[{c['ci95_low']:.5f}, {c['ci95_high']:.5f}]"
            lines.append(f"| {group} | {stage} | {c['n_gt']} | {mean} | {interval} | {auc} | {pair} |")
    lines.extend(["", "## P2-to-P3 paired change", "", "| Group | n GT | Delta contrast | 95% CI |", "|---|---:|---:|---|"])
    for row in payload["stage_deltas"]:
        if row["transition"] == "P2_to_P3" and row["group"] in {
            "small", "small_hit", "small_nonhit", "small_localization_error", "small_complete_miss", "elongated_ar_ge_5"
        }:
            mean = "NA" if row["mean"] is None else f"{row['mean']:.5f}"
            interval = "NA" if row["ci95_low"] is None else f"[{row['ci95_low']:.5f}, {row['ci95_high']:.5f}]"
            lines.append(f"| {row['group']} | {row['n_gt']} | {mean} | {interval} |")
    lines.extend([
        "",
        "## Adjudication",
        "",
        f"- {payload['decision']['reason']}",
        f"- Next action: {payload['decision']['next_action']}",
        "- Current same-GT residual ranking remains rejected and is not part of this route.",
        "",
        "## Limits",
        "",
        "Feature contrast and a frozen linear probe are representation proxies, not causal proof that a trainable injection will improve AP. Stage receptive fields differ. Any proposed module still requires a fresh matched training comparison.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    for key in ("weights", "data", "outcomes", "output"):
        setattr(args, key, getattr(args, key).expanduser().resolve())
    for path in (args.weights, args.data, args.outcomes):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    if args.imgsz != 640:
        raise ValueError("This frozen diagnosis requires imgsz=640")
    if args.folds < 3 or args.bootstrap < 100:
        raise ValueError("At least 3 folds and 100 bootstrap repetitions are required")
    outcomes = read_outcomes(args.outcomes)

    evaluator = ValCOCOEvaluator(args.data, args.imgsz, args.batch, args.workers, native_o2m=True)
    if len(evaluator.gt.imgs) != 1000:
        raise RuntimeError(f"Expected SVRDD7 Val 1000, got {len(evaluator.gt.imgs)}")
    model, _ = load_checkpoint(str(args.weights), device=select_device(args.device, verbose=False))
    model.float().eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    head = model.model[-1]
    if type(head) is not Detect or head.end2end or head.nc != len(NAMES):
        raise RuntimeError("Expected official non-end2end seven-class YOLO12 Detect")
    device = next(model.parameters()).device

    stage_mapping: dict[str, int] | None = None
    stage_shapes: dict[str, list[int]] = {}
    features: dict[str, torch.Tensor] = {}
    persistent_hooks = []
    records: list[dict[str, Any]] = []
    seen = 0

    for batch in evaluator.loader:
        images = batch["img"].to(device, non_blocking=True).float() / 255.0
        if stage_mapping is None:
            stage_mapping, discovery = discover_stage_layers(model, images[:1])
            for stage, layer in stage_mapping.items():
                tensor = discovery[layer]
                stage_shapes[stage] = list(tensor.shape)

                def capture(_module, _inputs, output, stage=stage):
                    features[stage] = output

                persistent_hooks.append(model.model[layer].register_forward_hook(capture))
        features.clear()
        with torch.inference_mode():
            model(images)
        if set(features) != set(STAGES):
            raise RuntimeError(f"Missing stage features: {set(STAGES) - set(features)}")
        h, w = images.shape[-2:]
        boxes_all = xywh2xyxy(batch["bboxes"].float().to(device))
        boxes_all *= torch.tensor((w, h, w, h), device=device)
        for batch_index in range(images.shape[0]):
            if args.max_images and seen >= args.max_images:
                break
            stem = Path(batch["im_file"][batch_index]).stem
            mask = batch["batch_idx"].view(-1).long() == batch_index
            boxes = boxes_all[mask]
            classes = batch["cls"].view(-1).long()[mask]
            for gt_index, (box, cls_tensor) in enumerate(zip(boxes, classes)):
                key = (stem, gt_index)
                if key not in outcomes:
                    raise RuntimeError(f"Frozen outcome mapping missing {key}")
                frozen = outcomes[key]
                class_name = NAMES[int(cls_tensor)]
                if class_name != frozen["class"]:
                    raise RuntimeError(f"GT order mismatch at {key}: {class_name} != {frozen['class']}")
                # Scale and aspect groups must remain in original-image coordinates.
                # The current `box` is letterboxed and is used only for feature sampling.
                size = frozen["coco_size"]
                ratio = float(frozen["aspect_ratio"])
                item: dict[str, Any] = {
                    "image": stem,
                    "gt_index": gt_index,
                    "class": class_name,
                    "size": size,
                    "aspect_ratio": ratio,
                    "elongated": int(ratio >= 5),
                    "outcome": frozen["outcome"],
                    "best_raw_iou": frozen["best_raw_iou"],
                    "top_score_iou": frozen["top_score_iou"],
                }
                for stage in STAGES:
                    target, ring, target_response, ring_response = sample_feature(
                        features[stage][batch_index], box, (h, w)
                    )
                    item[f"{stage}_target_vector"] = target
                    item[f"{stage}_ring_vector"] = ring
                    item[f"{stage}_target_response"] = target_response
                    item[f"{stage}_ring_response"] = ring_response
                    item[f"{stage}_contrast"] = target_response - ring_response
                records.append(item)
            seen += 1
        if args.max_images and seen >= args.max_images:
            break
    for hook in persistent_hooks:
        hook.remove()

    expected_images = min(args.max_images, 1000) if args.max_images else 1000
    if seen != expected_images:
        raise RuntimeError(f"Incomplete Val audit: {seen}/{expected_images}")
    if not args.max_images and len(records) != len(outcomes):
        raise RuntimeError(f"Incomplete GT audit: {len(records)}/{len(outcomes)}")

    filters = group_filters()
    contrast_rows, probe_rows, delta_rows = [], [], []
    for group_index, (group, predicate) in enumerate(filters):
        items = [item for item in records if predicate(item)]
        for stage_index, stage in enumerate(STAGES):
            summary = mean_ci_by_image(
                items,
                lambda item, stage=stage: item[f"{stage}_contrast"],
                args.bootstrap,
                args.seed + group_index * 31 + stage_index,
            )
            contrast_rows.append({"group": group, "stage": stage, **summary})
            probe_rows.append({"group": group, "stage": stage, **cross_validated_probe(items, stage, args.folds)})
        for transition, start, stop in (("P2_to_P3", "P2", "P3"), ("P3_to_P4", "P3", "P4")):
            summary = mean_ci_by_image(
                items,
                lambda item, start=start, stop=stop: item[f"{stop}_contrast"] - item[f"{start}_contrast"],
                args.bootstrap,
                args.seed + group_index * 37 + (0 if start == "P2" else 1),
            )
            delta_rows.append({"group": group, "transition": transition, **summary})

    association_rows = []
    for group, predicate in filters:
        items = [item for item in records if predicate(item)]
        for stage in STAGES:
            association_rows.append({
                "group": group,
                "stage": stage,
                "hit_minus_nonhit_contrast": difference_ci_by_image(
                    items,
                    lambda item: item["outcome"] == "hit",
                    lambda item: item["outcome"] != "hit",
                    f"{stage}_contrast",
                    args.bootstrap,
                    args.seed + 101 + len(association_rows),
                ),
            })

    contrast_lookup = {(row["group"], row["stage"]): row for row in contrast_rows}
    probe_lookup = {(row["group"], row["stage"]): row for row in probe_rows}
    delta_lookup = {(row["group"], row["transition"]): row for row in delta_rows}
    gate_group = "small_complete_miss"
    if contrast_lookup[(gate_group, "P2")]["n_gt"] < 30:
        gate_group = "small_nonhit"
    p2_contrast = contrast_lookup[(gate_group, "P2")]
    p2_probe = probe_lookup[(gate_group, "P2")]
    p3_probe = probe_lookup[(gate_group, "P3")]
    p2_p3_delta = delta_lookup[(gate_group, "P2_to_P3")]
    p2_informative = (
        p2_contrast["ci95_low"] is not None
        and p2_contrast["ci95_low"] > 0
        and p2_probe["auc"] is not None
        and p2_probe["auc"] >= 0.60
    )
    p3_informative = p3_probe["auc"] is not None and p3_probe["auc"] >= 0.60
    p3_collapse = (
        p2_p3_delta["ci95_high"] is not None
        and p2_p3_delta["ci95_high"] < 0
        and p2_probe["auc"] is not None
        and p3_probe["auc"] is not None
        and p2_probe["auc"] - p3_probe["auc"] >= 0.05
    )
    if p2_informative and p3_collapse:
        decision = {
            "route": "GO_P2_TO_P3_DETAIL_INJECTION",
            "reason": f"{gate_group} retains target information at P2 and loses it consistently at P3.",
            "next_action": "Design one bounded semantic-gated P2-to-P3 residual detail injection and compare it with official B0.",
        }
    elif p2_informative and p3_informative:
        decision = {
            "route": "NO_GO_DETAIL_INJECTION_GO_P3_LOCALIZATION_AUDIT",
            "reason": f"{gate_group} remains linearly separable at both P2 and P3 without the predeclared collapse pattern.",
            "next_action": "Audit P3 regression, DFL distributions, and scale-conditioned localization before changing features.",
        }
    elif not p2_informative:
        decision = {
            "route": "NO_GO_DETAIL_INJECTION_REVIEW_EARLY_REPRESENTATION",
            "reason": f"{gate_group} does not meet the joint P2 contrast and probe evidence gate.",
            "next_action": "Study input or shallow geometry representation; do not inject an unproven P2 signal.",
        }
    else:
        decision = {
            "route": "REVIEW_INCONCLUSIVE",
            "reason": f"{gate_group} evidence is mixed across contrast and probe measures.",
            "next_action": "Inspect numeric subgroup stability without training or changing the model.",
        }
    decision.update({
        "gate_group": gate_group,
        "p2_informative": p2_informative,
        "p3_informative": p3_informative,
        "p3_collapse": p3_collapse,
        "thresholds": {"positive_contrast_ci": True, "probe_auc": 0.60, "probe_auc_drop": 0.05},
    })

    stage_payload = {}
    assert stage_mapping is not None
    for stage in STAGES:
        layer = stage_mapping[stage]
        shape = stage_shapes[stage]
        stage_payload[stage] = {
            "layer": layer,
            "module": type(model.model[layer]).__name__,
            "stride": STRIDES[stage],
            "shape": shape,
            "channels": shape[1],
        }
    serializable_records = []
    for item in records:
        serializable_records.append({key: value for key, value in item.items() if not key.endswith("_vector")})
    outcome_counts: dict[str, int] = defaultdict(int)
    for item in records:
        outcome_counts[item["outcome"]] += 1
    payload = {
        "scope": "Official YOLO12n B0 best.pt, SVRDD7 Val 1000, imgsz=640, training-free, Test sealed, no Voting",
        "stages": stage_payload,
        "sampling": {
            "feature_normalization": "per-image per-channel spatial z-score",
            "target": "3x3 continuous bilinear samples inside GT",
            "background": "20 continuous bilinear samples on 1.8x expanded perimeter ring",
            "contrast": "mean(abs(z_target)) - mean(abs(z_ring))",
            "probe": f"diagonal-LDA linear probe with {args.folds}-fold image-grouped cross-validation",
        },
        "outcome_definitions": {
            "hit": "Frozen score-ordered same-class match at IoU>=0.50.",
            "classification_error": "Frozen diagnosis found a wrong-class overlap at IoU>=0.50.",
            "localization_error": "Not hit/classification error; a raw candidate overlaps at IoU>=0.10.",
            "complete_miss": "Best raw candidate IoU<0.10.",
        },
        "images": seen,
        "gt": len(records),
        "outcomes": dict(outcome_counts),
        "contrast": contrast_rows,
        "probes": probe_rows,
        "stage_deltas": delta_rows,
        "associations": association_rows,
        "decision": decision,
        "evidence": {
            "weights": {"path": str(args.weights), "sha256": sha256(args.weights)},
            "data": {"path": str(args.data), "sha256": sha256(args.data)},
            "outcomes": {"path": str(args.outcomes), "sha256": sha256(args.outcomes)},
            "git_commit": git_value("rev-parse", "HEAD"),
            "git_branch": git_value("branch", "--show-current"),
            "bootstrap_repeats": args.bootstrap,
            "seed": args.seed,
            "max_images": args.max_images,
            "test_accessed": False,
            "visualizations_generated": False,
        },
    }
    write_csv(args.output / "per_gt_stage.csv", serializable_records)
    write_csv(args.output / "contrast_summary.csv", contrast_rows)
    write_csv(args.output / "probe_summary.csv", probe_rows)
    write_csv(args.output / "stage_delta_bootstrap.csv", delta_rows)
    (args.output / "diagnosis.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    (args.output / "DIAGNOSIS.md").write_text(make_report(payload), encoding="utf-8")
    print(json.dumps({"stages": stage_payload, "images": seen, "gt": len(records), "outcomes": dict(outcome_counts), "decision": decision}, indent=2))
    print(f"YOLO12_P2_P3_DIAGNOSIS_COMPLETE output={args.output}")


if __name__ == "__main__":
    main()
