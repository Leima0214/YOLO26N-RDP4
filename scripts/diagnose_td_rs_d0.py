"""TD-RS D0: no-training gradient conflict and post-hoc head-routing audit.

This script never updates model parameters. It answers two bounded questions on a
frozen RoadSnake-R1 checkpoint:

1. Do O2M localization and classification losses ask the RoadSnake adapter (and
   its P4 input) to move in conflicting directions?
2. With the same frozen weights, what happens when RoadSnake/native P4 features
   are routed independently to the O2O box and classification heads?

The A/B/C/D routing sweep is diagnostic only because B/C are out-of-distribution
post-hoc combinations. It must not be reported as a trained detector ablation.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from types import MethodType
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from pycocotools.coco import COCO
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_japan4_paper_metrics import coco_eval, coco_row, remap_predictions
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.data.build import build_dataloader, build_yolo_dataset
from ultralytics.data.utils import check_det_dataset
from ultralytics.nn.roadsnake import RoadSnakeDetect
from ultralytics.utils import DEFAULT_CFG
from ultralytics.utils.loss import E2ELoss
from ultralytics.utils.metrics import box_iou
from ultralytics.utils.ops import xywh2xyxy
from ultralytics.utils.torch_utils import select_device


ROUTES = {
    "A_full": (True, True),
    "B_box_only": (True, False),
    "C_cls_only": (False, True),
    "D_native": (False, False),
}
GROUPS = ("all", "reduce", "local", "offset", "curve", "fuse", "p4_input")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--gradient-batch", type=int, default=8)
    parser.add_argument("--gradient-batches", type=int, default=12)
    parser.add_argument("--routing-batch", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--raw-batches", type=int, default=12)
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-coco", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
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


def make_loader(
    data_yaml: Path,
    split: str,
    batch: int,
    workers: int,
    imgsz: int,
    *,
    rect: bool,
):
    """Build a deterministic no-augmentation loader, even for the train split."""
    data = check_det_dataset(str(data_yaml))
    cfg = get_cfg(
        DEFAULT_CFG,
        {
            "mode": "val",
            "task": "detect",
            "imgsz": imgsz,
            "batch": batch,
            "workers": workers,
            "rect": rect,
            "cache": False,
            "seed": 42,
        },
    )
    dataset = build_yolo_dataset(cfg, data[split], batch, data, mode="val", rect=rect, stride=32)
    loader = build_dataloader(dataset, batch, workers, shuffle=False, rank=-1, pin_memory=False)
    return data, loader


def prepare_images(batch: dict[str, Any], device: torch.device) -> torch.Tensor:
    return batch["img"].to(device, non_blocking=False).float().div_(255.0)


def find_head(net: nn.Module) -> RoadSnakeDetect:
    head = net.model[-1]
    if not isinstance(head, RoadSnakeDetect):
        raise TypeError(f"Expected RoadSnakeDetect, got {type(head).__name__}")
    if not head.end2end:
        raise RuntimeError("D0 requires the YOLO26 end-to-end O2M/O2O head")
    return head


def set_gradient_mode(net: nn.Module) -> None:
    """Emit training dictionaries while keeping all BN running statistics frozen."""
    # Ultralytics may load inference checkpoints with parameters frozen. D0 needs
    # autograd vectors but never creates an optimizer or mutates parameter data.
    for parameter in net.parameters():
        parameter.requires_grad_(True)
    net.train()
    for module in net.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()


def parameter_group(name: str) -> str | None:
    if name.startswith("reduce."):
        return "reduce"
    if name.startswith("local."):
        return "local"
    if name.startswith("offset."):
        return "offset"
    if name.startswith(("horizontal_weight", "vertical_weight", "horizontal_norm.", "vertical_norm.")):
        return "curve"
    if name.startswith("fuse."):
        return "fuse"
    return None


def flatten_gradients(
    grads: tuple[torch.Tensor | None, ...], params: list[nn.Parameter]
) -> torch.Tensor:
    vectors = [
        (torch.zeros_like(param) if grad is None else grad).detach().float().reshape(-1)
        for grad, param in zip(grads, params)
    ]
    return torch.cat(vectors) if vectors else torch.empty(0)


def gradient_pair_metrics(loc: torch.Tensor, cls: torch.Tensor) -> dict[str, float | None]:
    loc_norm = float(loc.norm().item()) if loc.numel() else 0.0
    cls_norm = float(cls.norm().item()) if cls.numel() else 0.0
    cosine = None
    if loc_norm > 1e-12 and cls_norm > 1e-12:
        cosine = float(torch.dot(loc, cls).div(loc.norm() * cls.norm()).item())
    return {
        "loc_norm": loc_norm,
        "cls_norm": cls_norm,
        "cls_to_loc_norm_ratio": cls_norm / loc_norm if loc_norm > 1e-12 else None,
        "cosine": cosine,
        "conflict": int(cosine < 0) if cosine is not None else None,
    }


def run_gradient_audit(
    checkpoint: Path,
    data_yaml: Path,
    output: Path,
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    wrapped = YOLO(str(checkpoint))
    net = wrapped.model.to(device)
    # Standalone checkpoint loading preserves args as a plain dict, while the
    # loss expects the Trainer-style attribute namespace.
    net.args = get_cfg(DEFAULT_CFG, net.args)
    head = find_head(net)
    set_gradient_mode(net)
    criterion = E2ELoss(net)
    # D0 explicitly isolates the O2M task gradients; schedule weights are not applied.
    _, loader = make_loader(
        data_yaml,
        "train",
        args.gradient_batch,
        0,
        args.imgsz,
        rect=False,
    )

    named = [(name, param) for name, param in head.road_snake.named_parameters() if param.requires_grad]
    params = [param for _, param in named]
    subgroup_indices: dict[str, list[int]] = defaultdict(list)
    for index, (name, _) in enumerate(named):
        group = parameter_group(name)
        if group:
            subgroup_indices[group].append(index)

    captured: dict[str, torch.Tensor] = {}

    def capture_input(_module, inputs):
        p4 = inputs[0]
        p4.retain_grad()
        captured["p4"] = p4

    hook = head.road_snake.register_forward_pre_hook(capture_input)
    rows: list[dict[str, Any]] = []
    gamma_rows: list[dict[str, Any]] = []
    try:
        for batch_index, batch in enumerate(loader):
            if batch_index >= args.gradient_batches:
                break
            net.zero_grad(set_to_none=True)
            images = prepare_images(batch, device)
            preds = net(images)
            if not isinstance(preds, dict) or "one2many" not in preds:
                raise RuntimeError("Expected raw end-to-end training predictions")
            _, loss_parts, _ = criterion.one2many.get_assigned_targets_and_loss(preds["one2many"], batch)
            loc_loss = loss_parts[0] + loss_parts[2]
            cls_loss = loss_parts[1]
            p4 = captured["p4"]
            targets = params + [p4]
            loc_grads = torch.autograd.grad(loc_loss, targets, retain_graph=True, allow_unused=True)
            cls_grads = torch.autograd.grad(cls_loss, targets, retain_graph=False, allow_unused=True)

            loc_param_grads = loc_grads[:-1]
            cls_param_grads = cls_grads[:-1]
            all_loc = flatten_gradients(loc_param_grads, params)
            all_cls = flatten_gradients(cls_param_grads, params)
            row_base = {
                "batch": batch_index,
                "images": int(images.shape[0]),
                "instances": int(batch["cls"].numel()),
                "box_loss": float(loss_parts[0].detach().item()),
                "cls_loss": float(loss_parts[1].detach().item()),
                "dfl_loss": float(loss_parts[2].detach().item()),
            }
            rows.append({**row_base, "group": "all", **gradient_pair_metrics(all_loc, all_cls)})
            for group in ("reduce", "local", "offset", "curve", "fuse"):
                indices = subgroup_indices[group]
                group_params = [params[index] for index in indices]
                group_loc = flatten_gradients(tuple(loc_param_grads[index] for index in indices), group_params)
                group_cls = flatten_gradients(tuple(cls_param_grads[index] for index in indices), group_params)
                rows.append({**row_base, "group": group, **gradient_pair_metrics(group_loc, group_cls)})
            p4_loc = torch.zeros_like(p4) if loc_grads[-1] is None else loc_grads[-1]
            p4_cls = torch.zeros_like(p4) if cls_grads[-1] is None else cls_grads[-1]
            rows.append(
                {
                    **row_base,
                    "group": "p4_input",
                    **gradient_pair_metrics(p4_loc.detach().float().reshape(-1), p4_cls.detach().float().reshape(-1)),
                }
            )

            gamma_index = next(index for index, (name, _) in enumerate(named) if name == "gamma")
            gamma_loc = loc_param_grads[gamma_index]
            gamma_cls = cls_param_grads[gamma_index]
            loc_value = float(gamma_loc.item()) if gamma_loc is not None else 0.0
            cls_value = float(gamma_cls.item()) if gamma_cls is not None else 0.0
            gamma_rows.append(
                {
                    **row_base,
                    "gamma": float(head.road_snake.gamma.detach().item()),
                    "loc_derivative": loc_value,
                    "cls_derivative": cls_value,
                    "same_sign": int(loc_value * cls_value > 0),
                    "opposite_sign": int(loc_value * cls_value < 0),
                }
            )
    finally:
        hook.remove()
        del wrapped
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summaries: list[dict[str, Any]] = []
    for group in GROUPS:
        selected = [row for row in rows if row["group"] == group]
        cosines = np.asarray([row["cosine"] for row in selected if row["cosine"] is not None], dtype=float)
        loc_norms = np.asarray([row["loc_norm"] for row in selected], dtype=float)
        cls_norms = np.asarray([row["cls_norm"] for row in selected], dtype=float)
        summaries.append(
            {
                "group": group,
                "batches": len(selected),
                "valid_cosine_batches": int(cosines.size),
                "cosine_p25": float(np.quantile(cosines, 0.25)) if cosines.size else None,
                "cosine_median": float(np.median(cosines)) if cosines.size else None,
                "cosine_p75": float(np.quantile(cosines, 0.75)) if cosines.size else None,
                "conflict_fraction": float(np.mean(cosines < 0)) if cosines.size else None,
                "loc_norm_median": float(np.median(loc_norms)) if loc_norms.size else None,
                "cls_norm_median": float(np.median(cls_norms)) if cls_norms.size else None,
                "cls_to_loc_norm_ratio_median": (
                    float(np.median(cls_norms / np.maximum(loc_norms, 1e-12))) if loc_norms.size else None
                ),
            }
        )
    gamma_loc = np.asarray([row["loc_derivative"] for row in gamma_rows], dtype=float)
    gamma_cls = np.asarray([row["cls_derivative"] for row in gamma_rows], dtype=float)
    gamma_summary = {
        "batches": len(gamma_rows),
        "opposite_sign_fraction": float(np.mean(gamma_loc * gamma_cls < 0)) if gamma_rows else None,
        "loc_derivative_median": float(np.median(gamma_loc)) if gamma_rows else None,
        "cls_derivative_median": float(np.median(gamma_cls)) if gamma_rows else None,
        "note": "A scalar has no meaningful vector cosine; derivative signs are reported instead.",
    }
    write_csv(output / "gradient_batches.csv", rows)
    write_csv(output / "gradient_summary.csv", summaries)
    write_csv(output / "gamma_batches.csv", gamma_rows)
    return rows, summaries, gamma_summary


def routed_forward(self: RoadSnakeDetect, x: list[torch.Tensor]):
    """Post-hoc O2O route; route flags are installed on the head by the audit."""
    native = list(x)
    refined = list(native)
    refined[1] = self.road_snake(refined[1])
    box_features = refined if self._d0_box_rs else native
    cls_features = refined if self._d0_cls_rs else native
    one2one = {
        "boxes": torch.cat(
            [
                self.one2one_cv2[i](box_features[i]).view(box_features[0].shape[0], 4 * self.reg_max, -1)
                for i in range(self.nl)
            ],
            dim=-1,
        ),
        "scores": torch.cat(
            [
                self.one2one_cv3[i](cls_features[i]).view(cls_features[0].shape[0], self.nc, -1)
                for i in range(self.nl)
            ],
            dim=-1,
        ),
        "feats": box_features,
    }
    preds = {"one2many": one2one, "one2one": one2one}
    if self.training:
        return preds
    decoded = self._inference(one2one)
    result = self.postprocess(decoded.permute(0, 2, 1))
    return result if self.export else (result, preds)


def install_route(head: RoadSnakeDetect, box_rs: bool, cls_rs: bool) -> None:
    head._d0_box_rs = bool(box_rs)
    head._d0_cls_rs = bool(cls_rs)
    head.forward = MethodType(routed_forward, head)


@torch.no_grad()
def route_equivalence(checkpoint: Path, device: torch.device, imgsz: int) -> float:
    wrapped = YOLO(str(checkpoint))
    net = wrapped.model.to(device).eval()
    head = find_head(net)
    image = torch.rand(1, 3, imgsz, imgsz, device=device)
    original = net(image)[0]
    install_route(head, True, True)
    routed = net(image)[0]
    error = float((original - routed).abs().max().item())
    del wrapped
    return error


def run_coco_routes(
    checkpoint: Path,
    data_yaml: Path,
    output: Path,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    data_info = check_det_dataset(str(data_yaml))
    coco_path = Path(data_info["path"]) / "annotations" / "instances_val.json"
    if not coco_path.is_file():
        raise FileNotFoundError(coco_path)
    coco_gt = COCO(str(coco_path))
    category_ids = {category["name"]: category_id for category_id, category in coco_gt.cats.items()}
    main_rows: list[dict[str, Any]] = []
    class_rows: list[dict[str, Any]] = []
    prediction_root = output / "predictions"
    for route, (box_rs, cls_rs) in ROUTES.items():
        wrapped = YOLO(str(checkpoint))
        head = find_head(wrapped.model)
        install_route(head, box_rs, cls_rs)
        metrics = wrapped.val(
            data=str(data_yaml),
            split="val",
            imgsz=args.imgsz,
            batch=args.routing_batch,
            device=args.device,
            workers=args.workers,
            conf=args.conf,
            iou=args.iou,
            max_det=args.max_det,
            rect=True,
            save_json=True,
            plots=False,
            project=str(prediction_root),
            name=route,
            exist_ok=False,
            verbose=False,
        )
        prediction_path = Path(metrics.save_dir) / "predictions.json"
        predictions = remap_predictions(prediction_path, coco_gt)
        evaluator = coco_eval(coco_gt, predictions, list(category_ids.values()))
        native = metrics.results_dict
        main_rows.append(
            {
                "route": route,
                "box_uses_roadsnake": int(box_rs),
                "cls_uses_roadsnake": int(cls_rs),
                "native_P": native["metrics/precision(B)"],
                "native_R": native["metrics/recall(B)"],
                "native_AP50": native["metrics/mAP50(B)"],
                "native_AP75": metrics.box.map75,
                "native_AP50_95": native["metrics/mAP50-95(B)"],
                **coco_row(evaluator),
            }
        )
        for index, class_name in metrics.names.items():
            class_eval = coco_eval(coco_gt, predictions, [category_ids[class_name]])
            class_rows.append(
                {
                    "route": route,
                    "class": class_name,
                    "native_P": metrics.box.p[index],
                    "native_R": metrics.box.r[index],
                    "native_AP50": metrics.box.ap50[index],
                    "native_AP50_95": metrics.box.maps[index],
                    **coco_row(class_eval),
                }
            )
        del wrapped
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    write_csv(output / "routing_main_metrics.csv", main_rows)
    write_csv(output / "routing_per_class_metrics.csv", class_rows)
    return main_rows, class_rows


def spearman(scores: torch.Tensor, ious: torch.Tensor) -> float | None:
    if scores.numel() < 3:
        return None
    score_values = scores.detach().float().cpu().numpy()
    iou_values = ious.detach().float().cpu().numpy()
    if np.ptp(score_values) == 0 or np.ptp(iou_values) == 0:
        return None
    value = float(np.corrcoef(rankdata(score_values), rankdata(iou_values))[0, 1])
    return value if np.isfinite(value) else None


@torch.no_grad()
def run_raw_route_audit(
    checkpoint: Path,
    data_yaml: Path,
    output: Path,
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    wrapped = YOLO(str(checkpoint))
    net = wrapped.model.to(device).eval()
    head = find_head(net)
    _, loader = make_loader(data_yaml, "val", args.routing_batch, args.workers, args.imgsz, rect=True)
    captured: dict[str, list[torch.Tensor]] = {}

    def capture_inputs(_module, inputs):
        captured["native"] = list(inputs[0])

    hook = head.register_forward_pre_hook(capture_inputs)
    gt_rows: list[dict[str, Any]] = []
    overlap_rows: list[dict[str, Any]] = []
    try:
        for batch_index, batch in enumerate(loader):
            if batch_index >= args.raw_batches:
                break
            images = prepare_images(batch, device)
            _ = net(images)
            native = captured["native"]
            refined = list(native)
            refined[1] = head.road_snake(refined[1])
            route_raw: dict[str, dict[str, torch.Tensor]] = {}
            route_decoded: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
            for route, (box_rs, cls_rs) in ROUTES.items():
                box_features = refined if box_rs else native
                cls_features = refined if cls_rs else native
                raw = {
                    "boxes": torch.cat(
                        [head.one2one_cv2[i](box_features[i]).view(images.shape[0], 4 * head.reg_max, -1) for i in range(head.nl)],
                        dim=-1,
                    ),
                    "scores": torch.cat(
                        [head.one2one_cv3[i](cls_features[i]).view(images.shape[0], head.nc, -1) for i in range(head.nl)],
                        dim=-1,
                    ),
                    "feats": box_features,
                }
                decoded = head._inference(raw)
                # YOLO26 end-to-end Detect.decode_bboxes already emits xyxy.
                boxes = decoded[:, :4].permute(0, 2, 1)
                scores = decoded[:, 4:].permute(0, 2, 1)
                route_raw[route] = raw
                route_decoded[route] = (boxes, scores)

            batch_indices = batch["batch_idx"].view(-1).long()
            labels = batch["cls"].view(-1).long()
            normalized = batch["bboxes"]
            height, width = images.shape[2:]
            gt_xyxy = xywh2xyxy(normalized) * normalized.new_tensor([width, height, width, height])
            for image_index in range(images.shape[0]):
                mask = batch_indices == image_index
                image_gt = gt_xyxy[mask].to(device)
                image_labels = labels[mask].to(device)
                if not image_gt.numel():
                    continue
                base_scores = route_decoded["A_full"][1][image_index].amax(dim=1)
                base_top = set(base_scores.topk(min(args.topk, base_scores.numel())).indices.cpu().tolist())
                for route, (boxes_batch, scores_batch) in route_decoded.items():
                    boxes = boxes_batch[image_index]
                    scores = scores_batch[image_index]
                    top = set(scores.amax(dim=1).topk(min(args.topk, scores.shape[0])).indices.cpu().tolist())
                    overlap_rows.append(
                        {
                            "batch": batch_index,
                            "image_index": image_index,
                            "route": route,
                            "topk": args.topk,
                            "overlap_with_A": len(base_top & top) / max(len(base_top), 1),
                            "jaccard_with_A": len(base_top & top) / max(len(base_top | top), 1),
                        }
                    )
                    iou_matrix = box_iou(image_gt, boxes)
                    for gt_index in range(image_gt.shape[0]):
                        class_id = int(image_labels[gt_index].item())
                        same_class_gt = torch.where(image_labels == class_id)[0]
                        class_ious = iou_matrix[same_class_gt]
                        owner_local = class_ious.argmax(dim=0)
                        local_gt_index = int(torch.where(same_class_gt == gt_index)[0].item())
                        owned = owner_local == local_gt_index
                        candidate_scores = scores[:, class_id][owned]
                        candidate_ious = iou_matrix[gt_index][owned]
                        if not candidate_scores.numel():
                            continue
                        k = min(args.topk, candidate_scores.numel())
                        order = candidate_scores.topk(k).indices
                        selected_scores = candidate_scores[order]
                        selected_ious = candidate_ious[order]
                        gt_rows.append(
                            {
                                "batch": batch_index,
                                "image_index": image_index,
                                "gt_index": gt_index,
                                "class_id": class_id,
                                "route": route,
                                "owned_candidates": int(candidate_scores.numel()),
                                "top1_iou": float(selected_ious[0].item()),
                                "topk_max_iou": float(selected_ious.max().item()),
                                "topk_score_iou_spearman": spearman(selected_scores, selected_ious),
                            }
                        )
    finally:
        hook.remove()
        del wrapped
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary_rows: list[dict[str, Any]] = []
    for route in ROUTES:
        selected = [row for row in gt_rows if row["route"] == route]
        correlations = np.asarray(
            [row["topk_score_iou_spearman"] for row in selected if row["topk_score_iou_spearman"] is not None],
            dtype=float,
        )
        overlaps = [row for row in overlap_rows if row["route"] == route]
        summary_rows.append(
            {
                "route": route,
                "gt_count": len(selected),
                "valid_spearman_gt_count": int(correlations.size),
                "within_gt_spearman_mean": float(np.mean(correlations)) if correlations.size else None,
                "within_gt_spearman_median": float(np.median(correlations)) if correlations.size else None,
                "top1_iou_mean": float(np.mean([row["top1_iou"] for row in selected])) if selected else None,
                "topk_max_iou_mean": float(np.mean([row["topk_max_iou"] for row in selected])) if selected else None,
                "topk_overlap_with_A_mean": float(np.mean([row["overlap_with_A"] for row in overlaps])) if overlaps else None,
                "topk_jaccard_with_A_mean": float(np.mean([row["jaccard_with_A"] for row in overlaps])) if overlaps else None,
            }
        )
    write_csv(output / "routing_raw_per_gt.csv", gt_rows)
    write_csv(output / "routing_topk_overlap.csv", overlap_rows)
    write_csv(output / "routing_raw_summary.csv", summary_rows)
    return gt_rows, summary_rows


def decide(gradient_summary: list[dict[str, Any]], routing_rows: list[dict[str, Any]]) -> dict[str, Any]:
    all_row = next(row for row in gradient_summary if row["group"] == "all")
    median = all_row["cosine_median"]
    fraction = all_row["conflict_fraction"]
    strong_conflict = median is not None and fraction is not None and median < -0.05 and fraction >= 0.60
    route_map = {row["route"]: row for row in routing_rows}
    posthoc_support = None
    delta_box_only = None
    if "A_full" in route_map and "B_box_only" in route_map:
        delta_box_only = route_map["B_box_only"]["native_AP50_95"] - route_map["A_full"]["native_AP50_95"]
        posthoc_support = delta_box_only >= 0
    return {
        "strong_gradient_conflict": strong_conflict,
        "posthoc_box_only_not_worse": posthoc_support,
        "posthoc_box_only_delta_native_AP50_95": delta_box_only,
        "td_rs_d0_verdict": "GO_CONTROLLED_TRAINING_ABLATION" if strong_conflict else "NO_GO_OR_WEAK_EVIDENCE",
        "interpretation_boundary": (
            "The gradient gate is primary. Post-hoc B/C routes are distribution-shift diagnostics and cannot alone "
            "prove or disprove TD-RS. A GO authorizes only a fresh matched O2M-RS-clean vs TD-RS experiment."
        ),
    }


def main() -> None:
    args = parse_args()
    args.checkpoint = args.checkpoint.expanduser().resolve()
    args.data = args.data.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if not args.data.is_file():
        raise FileNotFoundError(args.data)
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = select_device(args.device)

    equivalence_error = route_equivalence(args.checkpoint, device, args.imgsz)
    if equivalence_error > 1e-6:
        raise AssertionError(f"A_full route does not reproduce R1: max_abs_error={equivalence_error}")

    _, gradient_summary, gamma_summary = run_gradient_audit(
        args.checkpoint, args.data, args.output, device, args
    )
    _, raw_summary = run_raw_route_audit(args.checkpoint, args.data, args.output, device, args)
    routing_rows: list[dict[str, Any]] = []
    if not args.skip_coco:
        routing_rows, _ = run_coco_routes(args.checkpoint, args.data, args.output, args)
    decision = decide(gradient_summary, routing_rows)
    summary = {
        "protocol": {
            "diagnostic_only": True,
            "training_performed": False,
            "test_read": False,
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": sha256(args.checkpoint),
            "data": str(args.data),
            "imgsz": args.imgsz,
            "gradient_split": "train",
            "gradient_preprocessing": "deterministic val-style letterbox; no augmentation; workers=0; shuffle=False",
            "gradient_batch": args.gradient_batch,
            "gradient_batches": args.gradient_batches,
            "gradient_tasks": "O2M localization=(box+dfl) versus O2M classification; schedule weights omitted",
            "routing_split": "val",
            "routing_routes": {
                key: {"box_uses_roadsnake": value[0], "cls_uses_roadsnake": value[1]}
                for key, value in ROUTES.items()
            },
            "posthoc_warning": "B/C are untrained mixed routes and are diagnostic, not formal model results.",
        },
        "preflight": {"A_full_output_max_abs_error": equivalence_error},
        "gradient_summary": gradient_summary,
        "gamma_summary": gamma_summary,
        "routing_main_metrics": routing_rows,
        "routing_raw_summary": raw_summary,
        "decision": decision,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
