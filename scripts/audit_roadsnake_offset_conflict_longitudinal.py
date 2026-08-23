"""Longitudinal, no-training audit of RoadSnake offset gradient conflict.

The audit reuses paired, genuinely augmented train batches for every checkpoint
within a seed. It never creates an optimizer, never calls backward(), freezes BN
running statistics, and verifies that model parameters are unchanged.

Primary question:
    Is the O2M classification-vs-localization conflict on RoadSnake offset
    parameters persistent across seeds and training stages, rather than a
    late-checkpoint or single-class artifact?
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.data.build import build_dataloader, build_yolo_dataset
from ultralytics.data.utils import check_det_dataset
from ultralytics.nn.roadsnake import RoadSnakeDetect
from ultralytics.utils import DEFAULT_CFG
from ultralytics.utils.loss import E2ELoss
from ultralytics.utils.tal import make_anchors


CHECKPOINT_NAMES = ("epoch5.pt", "epoch20.pt", "epoch50.pt", "best.pt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed42-run", type=Path, required=True)
    parser.add_argument("--seed43-run", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--batches", type=int, default=16)
    parser.add_argument("--audit-seed", type=int, default=20260823)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_hash(batch: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for key in ("img", "batch_idx", "cls", "bboxes"):
        value = batch[key].detach().cpu().contiguous()
        digest.update(key.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def model_hash(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def checkpoint_train_args(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    args = payload.get("train_args", {})
    if not isinstance(args, dict):
        args = vars(args)
    return dict(args)


def build_augmented_batches(
    checkpoint: Path,
    data_yaml: Path,
    imgsz: int,
    batch_size: int,
    count: int,
    seed: int,
) -> tuple[list[dict[str, torch.Tensor]], list[dict[str, Any]], dict[str, Any]]:
    """Materialize real train-mode augmentation once so all checkpoints see identical tensors."""
    seed_everything(seed)
    train_args = checkpoint_train_args(checkpoint)
    overrides = {
        **train_args,
        "mode": "train",
        "task": "detect",
        "data": str(data_yaml),
        "imgsz": imgsz,
        "batch": batch_size,
        "workers": 0,
        "rect": False,
        "cache": False,
        "seed": seed,
        "deterministic": True,
    }
    cfg = get_cfg(DEFAULT_CFG, overrides)
    data = check_det_dataset(str(data_yaml))
    dataset = build_yolo_dataset(cfg, data["train"], batch_size, data, mode="train", rect=False, stride=32)
    loader = build_dataloader(dataset, batch_size, 0, shuffle=False, rank=-1, pin_memory=False)
    cached: list[dict[str, torch.Tensor]] = []
    audit_rows: list[dict[str, Any]] = []
    for batch_index, batch in enumerate(loader):
        if batch_index >= count:
            break
        kept = {key: batch[key].detach().cpu().clone() for key in ("img", "batch_idx", "cls", "bboxes")}
        cached.append(kept)
        counts = torch.bincount(kept["cls"].view(-1).long(), minlength=len(data["names"]))
        audit_rows.append(
            {
                "seed": seed,
                "batch": batch_index,
                "hash": tensor_hash(kept),
                "images": int(kept["img"].shape[0]),
                "instances": int(kept["cls"].numel()),
                **{f"instances_{data['names'][i]}": int(counts[i]) for i in range(len(data["names"]))},
            }
        )
    if len(cached) != count:
        raise RuntimeError(f"Requested {count} augmented batches but materialized {len(cached)}")
    augmentation = {
        key: getattr(cfg, key, None)
        for key in (
            "mosaic",
            "mixup",
            "copy_paste",
            "degrees",
            "translate",
            "scale",
            "shear",
            "perspective",
            "fliplr",
            "flipud",
            "hsv_h",
            "hsv_s",
            "hsv_v",
        )
    }
    return cached, audit_rows, augmentation


def find_head(net: nn.Module) -> RoadSnakeDetect:
    head = net.model[-1]
    if not isinstance(head, RoadSnakeDetect):
        raise TypeError(f"Expected RoadSnakeDetect, got {type(head).__name__}")
    if not head.end2end:
        raise RuntimeError("Audit requires the YOLO26 end-to-end O2M/O2O head")
    return head


def prepare_model(checkpoint: Path, device: torch.device) -> tuple[YOLO, nn.Module, RoadSnakeDetect, E2ELoss]:
    wrapped = YOLO(str(checkpoint))
    net = wrapped.model.to(device)
    net.args = get_cfg(DEFAULT_CFG, net.args)
    for parameter in net.parameters():
        parameter.requires_grad_(True)
    net.train()
    for module in net.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()
    head = find_head(net)
    return wrapped, net, head, E2ELoss(net)


def flatten_gradients(grads: tuple[torch.Tensor | None, ...], params: list[nn.Parameter]) -> torch.Tensor:
    return torch.cat(
        [
            (torch.zeros_like(param) if grad is None else grad).detach().float().reshape(-1)
            for grad, param in zip(grads, params)
        ]
    )


def gradient_metrics(loc: torch.Tensor, cls: torch.Tensor) -> dict[str, Any]:
    loc_norm = float(loc.norm().item())
    cls_norm = float(cls.norm().item())
    cosine = None
    if loc_norm > 1e-12 and cls_norm > 1e-12:
        cosine = float(torch.dot(loc, cls).div(loc.norm() * cls.norm()).item())
    ratio = cls_norm / loc_norm if loc_norm > 1e-12 else None
    finite = bool(torch.isfinite(loc).all() and torch.isfinite(cls).all())
    return {
        "loc_norm": loc_norm,
        "cls_norm": cls_norm,
        "cls_to_loc_norm_ratio": ratio,
        "log10_cls_to_loc_ratio": math.log10(max(ratio, 1e-30)) if ratio is not None else None,
        "cosine": cosine,
        "conflict": int(cosine < 0) if cosine is not None else None,
        "finite": int(finite),
    }


def class_losses(
    loss_obj,
    preds: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
) -> dict[int, tuple[torch.Tensor, torch.Tensor, int]]:
    """Decompose the original O2M assignment by assigned GT class.

    Localization uses only positives assigned to the class. Classification uses
    the corresponding class logit over all anchors (including true background),
    preserving the score-suppression pressure relevant to that class.
    """
    pred_distri = preds["boxes"].permute(0, 2, 1).contiguous()
    pred_scores = preds["scores"].permute(0, 2, 1).contiguous()
    anchor_points, stride_tensor = make_anchors(preds["feats"], loss_obj.stride, 0.5)
    dtype = pred_scores.dtype
    batch_size = pred_scores.shape[0]
    imgsz = torch.tensor(preds["feats"][0].shape[2:], device=loss_obj.device, dtype=dtype) * loss_obj.stride[0]
    targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
    targets = loss_obj.preprocess(targets.to(loss_obj.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
    gt_labels, gt_bboxes = targets.split((1, 4), 2)
    mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
    pred_bboxes = loss_obj.bbox_decode(anchor_points, pred_distri)
    _, target_bboxes, target_scores, fg_mask, target_gt_idx = loss_obj.assigner(
        pred_scores.detach().sigmoid(),
        (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
        anchor_points * stride_tensor,
        gt_labels,
        gt_bboxes,
        mask_gt,
    )
    if not gt_labels.shape[1]:
        return {}
    assigned_class = gt_labels.squeeze(-1).gather(1, target_gt_idx.long()).long()
    results: dict[int, tuple[torch.Tensor, torch.Tensor, int]] = {}
    for class_id in range(loss_obj.nc):
        fg_class = fg_mask & assigned_class.eq(class_id)
        positives = int(fg_class.sum().item())
        if positives == 0:
            continue
        target_scores_class = torch.zeros_like(target_scores)
        target_scores_class[..., class_id] = target_scores[..., class_id]
        score_sum = target_scores_class.sum().clamp_min(1.0)
        cls = loss_obj.bce(pred_scores[..., class_id], target_scores[..., class_id].to(dtype)).sum() / score_sum
        box, dfl = loss_obj.bbox_loss(
            pred_distri,
            pred_bboxes,
            anchor_points,
            target_bboxes / stride_tensor,
            target_scores_class,
            score_sum,
            fg_class,
            imgsz,
            stride_tensor,
        )
        loc = box * loss_obj.hyp.box + dfl * loss_obj.hyp.dfl
        cls = cls * loss_obj.hyp.cls
        results[class_id] = (loc, cls, positives)
    return results


def best_epoch(run: Path) -> int | None:
    path = run / "results.csv"
    if not path.is_file():
        return None
    frame = pd.read_csv(path)
    frame.columns = [column.strip() for column in frame.columns]
    metric = "metrics/mAP50-95(B)"
    if metric not in frame:
        return None
    return int(frame.loc[frame[metric].idxmax(), "epoch"])


def summarize(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(row)
    summaries: list[dict[str, Any]] = []
    for values, selected in sorted(grouped.items(), key=lambda item: tuple(str(v) for v in item[0])):
        cosines = np.asarray([row["cosine"] for row in selected if row["cosine"] is not None], dtype=float)
        loc_norms = np.asarray([row["loc_norm"] for row in selected], dtype=float)
        cls_norms = np.asarray([row["cls_norm"] for row in selected], dtype=float)
        ratios = np.asarray(
            [row["cls_to_loc_norm_ratio"] for row in selected if row["cls_to_loc_norm_ratio"] is not None],
            dtype=float,
        )
        summary = {key: value for key, value in zip(keys, values)}
        summary.update(
            {
                "batches": len(selected),
                "valid_cosine_batches": int(cosines.size),
                "conflict_fraction": float(np.mean(cosines < 0)) if cosines.size else None,
                "cosine_p25": float(np.quantile(cosines, 0.25)) if cosines.size else None,
                "cosine_median": float(np.median(cosines)) if cosines.size else None,
                "cosine_p75": float(np.quantile(cosines, 0.75)) if cosines.size else None,
                "loc_norm_median": float(np.median(loc_norms)) if loc_norms.size else None,
                "cls_norm_median": float(np.median(cls_norms)) if cls_norms.size else None,
                "ratio_p25": float(np.quantile(ratios, 0.25)) if ratios.size else None,
                "ratio_median": float(np.median(ratios)) if ratios.size else None,
                "ratio_p75": float(np.quantile(ratios, 0.75)) if ratios.size else None,
                "all_finite": int(all(row["finite"] for row in selected)),
            }
        )
        summaries.append(summary)
    return summaries


def verdict(checkpoint_summary: list[dict[str, Any]], class_summary: list[dict[str, Any]]) -> dict[str, Any]:
    seed_results: dict[str, Any] = {}
    all_seed_go = True
    for seed in (42, 43):
        checkpoints = [row for row in checkpoint_summary if row["seed"] == seed and row["scope"] == "all"]
        conflict_checkpoints = sum(row["conflict_fraction"] > 0.5 for row in checkpoints)
        negative_median_checkpoints = sum(row["cosine_median"] < 0 for row in checkpoints)
        balanced_checkpoints = sum(0.01 <= row["ratio_median"] <= 100.0 for row in checkpoints)
        early = [row for row in checkpoints if row["checkpoint"] in ("epoch5", "epoch20")]
        early_support = any(row["conflict_fraction"] > 0.5 and row["cosine_median"] < 0 for row in early)
        classes = [row for row in class_summary if row["seed"] == seed and row["scope"] != "all"]
        supported_classes = [
            row["scope"]
            for row in classes
            if row["conflict_fraction"] > 0.5 and row["cosine_median"] < 0
        ]
        seed_go = (
            conflict_checkpoints >= 3
            and negative_median_checkpoints >= 3
            and balanced_checkpoints >= 3
            and early_support
            and len(supported_classes) >= 2
        )
        seed_results[str(seed)] = {
            "checkpoints_conflict_gt_50pct": conflict_checkpoints,
            "checkpoints_negative_median": negative_median_checkpoints,
            "checkpoints_ratio_within_0p01_to_100": balanced_checkpoints,
            "early_epoch5_or_epoch20_support": early_support,
            "supported_classes": supported_classes,
            "seed_go": seed_go,
        }
        all_seed_go &= seed_go
    return {
        "criteria": {
            "per_seed_majority": "at least 3/4 checkpoints with conflict_fraction > 0.5 and cosine median < 0",
            "not_late_only": "epoch5 or epoch20 supports conflict in each seed",
            "not_single_class": "at least two classes per seed have aggregate conflict_fraction > 0.5 and negative median",
            "norm_balance": "at least 3/4 checkpoint median cls/loc ratios are between 0.01 and 100",
        },
        "seeds": seed_results,
        "gcs_longitudinal_go": all_seed_go,
    }


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    runs = {42: args.seed42_run.resolve(), 43: args.seed43_run.resolve()}
    data_yaml = args.data.resolve()
    raw_rows: list[dict[str, Any]] = []
    batch_audit: list[dict[str, Any]] = []
    checkpoint_audit: list[dict[str, Any]] = []
    shared_reference = runs[42] / "weights" / "epoch5.pt"
    cached, batch_audit, augmentation = build_augmented_batches(
        shared_reference,
        data_yaml,
        args.imgsz,
        args.batch,
        args.batches,
        args.audit_seed,
    )

    for seed, run in runs.items():
        checkpoint_paths = {name.removesuffix(".pt"): run / "weights" / name for name in CHECKPOINT_NAMES}
        missing = [str(path) for path in checkpoint_paths.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing checkpoints: {missing}")
        for checkpoint_label, checkpoint in checkpoint_paths.items():
            seed_everything(seed)
            wrapped, net, head, criterion = prepare_model(checkpoint, device)
            before_hash = model_hash(net)
            offset_named = [
                (name, parameter)
                for name, parameter in head.road_snake.named_parameters()
                if name.startswith("offset.") and parameter.requires_grad
            ]
            if not offset_named:
                raise RuntimeError("No trainable RoadSnake offset parameters found")
            params = [parameter for _, parameter in offset_named]
            for batch_index, batch in enumerate(cached):
                net.zero_grad(set_to_none=True)
                images = batch["img"].to(device, non_blocking=False).float().div_(255.0)
                with torch.autocast(device_type=device.type, enabled=args.amp and device.type == "cuda"):
                    preds = net(images)
                    if not isinstance(preds, dict) or "one2many" not in preds:
                        raise RuntimeError("Expected raw end-to-end training predictions")
                    _, loss_parts, _ = criterion.one2many.get_assigned_targets_and_loss(preds["one2many"], batch)
                    pairs: list[tuple[str, torch.Tensor, torch.Tensor, int]] = [
                        ("all", loss_parts[0] + loss_parts[2], loss_parts[1], int(batch["cls"].numel()))
                    ]
                    for class_id, (loc, cls, positives) in class_losses(
                        criterion.one2many, preds["one2many"], batch
                    ).items():
                        pairs.append((str(net.names[class_id]), loc, cls, positives))
                for pair_index, (scope, loc_loss, cls_loss, positives) in enumerate(pairs):
                    loc_grads = torch.autograd.grad(loc_loss, params, retain_graph=True, allow_unused=True)
                    cls_grads = torch.autograd.grad(
                        cls_loss,
                        params,
                        retain_graph=pair_index < len(pairs) - 1,
                        allow_unused=True,
                    )
                    loc_vector = flatten_gradients(loc_grads, params)
                    cls_vector = flatten_gradients(cls_grads, params)
                    raw_rows.append(
                        {
                            "seed": seed,
                            "checkpoint": checkpoint_label,
                            "batch": batch_index,
                            "batch_hash": batch_audit[batch_index]["hash"],
                            "scope": scope,
                            "instances_or_assigned_positives": positives,
                            "box_loss": float(loss_parts[0].detach().item()),
                            "cls_loss": float(loss_parts[1].detach().item()),
                            "dfl_loss": float(loss_parts[2].detach().item()),
                            **gradient_metrics(loc_vector, cls_vector),
                        }
                    )
            after_hash = model_hash(net)
            checkpoint_audit.append(
                {
                    "seed": seed,
                    "checkpoint": checkpoint_label,
                    "path": str(checkpoint),
                    "sha256": sha256(checkpoint),
                    "best_training_epoch": best_epoch(run) if checkpoint_label == "best" else None,
                    "offset_parameter_tensors": len(params),
                    "offset_parameters": sum(parameter.numel() for parameter in params),
                    "model_hash_before": before_hash,
                    "model_hash_after": after_hash,
                    "model_unchanged": before_hash == after_hash,
                }
            )
            del wrapped, net, head, criterion
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    checkpoint_summary = summarize(raw_rows, ("seed", "checkpoint", "scope"))
    class_summary = summarize(raw_rows, ("seed", "scope"))
    decision = verdict(checkpoint_summary, class_summary)
    write_csv(args.output / "gradient_batches.csv", raw_rows)
    write_csv(args.output / "checkpoint_summary.csv", checkpoint_summary)
    write_csv(args.output / "class_summary.csv", class_summary)
    write_csv(args.output / "augmented_batches.csv", batch_audit)
    write_csv(args.output / "checkpoint_integrity.csv", checkpoint_audit)
    payload = {
        "audit": "RoadSnake offset longitudinal O2M classification-vs-localization gradient conflict",
        "no_training": True,
        "test_read": False,
        "amp_forward": args.amp,
        "bn_running_stats_frozen": True,
        "paired_augmented_batches": True,
        "batch_size": args.batch,
        "batches_per_checkpoint": args.batches,
        "audit_seed": args.audit_seed,
        "augmentation": augmentation,
        "checkpoints": checkpoint_audit,
        "checkpoint_summary": checkpoint_summary,
        "class_summary": class_summary,
        "verdict": decision,
    }
    (args.output / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(decision, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
