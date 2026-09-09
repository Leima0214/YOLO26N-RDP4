"""Val-only canonical COCO O2M selection and GT-independent score Box Voting.

Shares the project's established NMS, coordinate serialization and vote-group
construction. Every epoch selects on FP32 O2M/no Voting, COCO maxDets=100.
No call to model.fuse(): that would discard the O2M head on end-to-end models.
"""

from __future__ import annotations

import contextlib
import io
import json
import math
import sys
from copy import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT / "scripts"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

NAMES = ["LC", "TC", "AC", "P", "MC", "LP", "TP"]
EVAL_PROTOCOL = dict(score_floor=0.001, nms_iou=0.7, max_det=300, max_nms=30000,
                     coco_max_dets=100, precision="float32", split="val", rect=True)


def write_json(path: Path, payload) -> None:
    def clean(value):
        if isinstance(value, dict):
            return {key: clean(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [clean(item) for item in value]
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(payload), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def score_box_vote(current: torch.Tensor, expanded: torch.Tensor, groups: list[torch.Tensor]) -> torch.Tensor:
    """Keep class/score/order; replace only coordinates by score-weighted group means."""
    voted = current.clone()
    for index, members in enumerate(groups):
        candidates = expanded[members]
        weights = candidates[:, 4].float().clamp_min(0)
        weights = weights / weights.sum().clamp_min(1e-12)
        voted[index, :4] = (candidates[:, :4].float() * weights[:, None]).sum(0)
    return voted


def coco_summary(gt, predictions, image_ids: list[int], category_names: dict[int, str]):
    """One COCO pass yields exactly the same per-category slices as separate passes."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    with contextlib.redirect_stdout(io.StringIO()):
        if predictions:
            # pycocotools augments each annotation dict in-place with area/id/
            # segmentation.  Evaluate shallow copies so saved canonical rows
            # remain the exact detector output contract.
            detections = gt.loadRes([dict(item) for item in predictions])
        else:
            detections = COCO()
            detections.dataset = {"images": list(gt.dataset["images"]),
                                  "categories": list(gt.dataset["categories"]), "annotations": []}
            detections.createIndex()
        evaluator = COCOeval(gt, detections, "bbox")
        evaluator.params.imgIds = sorted(image_ids)
        evaluator.params.catIds = sorted(category_names)
        evaluator.params.maxDets = [100]
        evaluator.evaluate()
        evaluator.accumulate()

    def summary(category=None):
        precision, recall = evaluator.eval["precision"], evaluator.eval["recall"]
        if category is not None:
            index = evaluator.params.catIds.index(category)
            precision, recall = precision[:, :, index:index + 1], recall[:, index:index + 1]

        def mean_valid(values):
            values = values[values > -1]
            return float(values.mean()) if values.size else float("nan")

        def ap(area="all", iou=None):
            values = precision
            if iou is not None:
                index = int(np.argmin(np.abs(evaluator.params.iouThrs - iou)))
                values = values[index:index + 1]
            return mean_valid(values[:, :, :, evaluator.params.areaRngLbl.index(area), -1])
        return {"AP": ap(), "AP50": ap(iou=0.5), "AP75": ap(iou=0.75),
                "AP_small": ap("small"), "AP_medium": ap("medium"), "AP_large": ap("large"),
                "AR100": mean_valid(recall[:, :, 0, -1])}

    return {"overall": summary(), "per_class": {name: summary(category) for category, name in category_names.items()}}


class ValCOCOEvaluator:
    """Use the canonical Val loader and evaluate the raw O2M tensors, never O2O."""

    def __init__(self, data: str | Path, imgsz=640, batch=32, workers=8, max_images=0, loader=None,
                 native_o2m=False):
        from pycocotools.coco import COCO
        from ultralytics.data.utils import check_det_dataset

        self.data_path = Path(data).resolve()
        self.data = check_det_dataset(str(self.data_path), autodownload=False)
        self.names = [self.data["names"][i] for i in range(len(self.data["names"]))]
        if self.names != NAMES:
            raise ValueError(f"Expected the original seven SVRDD classes, got {self.names}")
        test = self.data.get("test")
        test_paths = [Path(value).resolve() for value in (test if isinstance(test, list) else [test]) if value]
        for split in ("train", "val"):
            paths = self.data[split] if isinstance(self.data[split], list) else [self.data[split]]
            for value in paths:
                path = Path(value).resolve()
                if any(path == sealed or path.is_relative_to(sealed) or sealed.is_relative_to(path)
                       for sealed in test_paths):
                    raise ValueError(f"{split} path overlaps the sealed Test path")
        self.coco_path = Path(self.data["path"]) / "annotations/instances_val.json"
        with contextlib.redirect_stdout(io.StringIO()):
            self.gt = COCO(str(self.coco_path))
        self.gt.dataset.setdefault("info", {})
        by_name = {category["name"]: cid for cid, category in self.gt.cats.items()}
        if set(by_name) != set(NAMES) or len(by_name) != len(self.gt.cats):
            raise ValueError("Val COCO categories must match the seven dataset classes exactly")
        self.category_ids = {index: by_name[name] for index, name in enumerate(NAMES)}
        self.category_names = {by_name[name]: name for name in NAMES}
        self.ids_by_stem = {}
        for iid, item in self.gt.imgs.items():
            stem = Path(item["file_name"]).stem
            if stem in self.ids_by_stem:
                raise ValueError(f"Duplicate Val image stem: {stem}")
            self.ids_by_stem[stem] = iid
        self.imgsz, self.batch, self.workers, self.max_images = imgsz, batch, workers, max_images
        self.native_o2m = bool(native_o2m)
        self.loader = loader if loader is not None else self.make_loader()
        for filename in self.loader.dataset.im_files:
            if any(Path(filename).resolve().is_relative_to(sealed) for sealed in test_paths):
                raise ValueError("A validation image resolves into the sealed Test directory")
        self.last_rows = {}

    def make_loader(self):
        from ultralytics.cfg import get_cfg
        from ultralytics.data.build import build_dataloader, build_yolo_dataset
        from ultralytics.utils import DEFAULT_CFG
        cfg = get_cfg(DEFAULT_CFG, {"mode": "val", "task": "detect", "imgsz": self.imgsz,
                                   "batch": self.batch, "workers": self.workers, "rect": True, "cache": False})
        dataset = build_yolo_dataset(cfg, self.data["val"], self.batch, self.data, mode="val", rect=True, stride=32)
        return build_dataloader(dataset, self.batch, self.workers, shuffle=False, rank=-1, pin_memory=False)

    @torch.inference_mode()
    def evaluate(self, model, *, voting=False, compute_loss=False, output: Path | None = None):
        from audit_gbrg_o2m_strict_reconstruction import canonical_coco_rows, current_nms, expanded_candidates, raw_level_indices
        from audit_gbrg_o2m_region_signal import groups_anchored_to_reference
        from ultralytics.nn.modules.head import Detect
        from ultralytics.nn.roadsnake_mid import RoadSnakeMidFusion
        from ultralytics.nn.rs_detail_companion import FrequencyDetailReconstruction
        from ultralytics.utils.ops import xywh2xyxy

        if output is not None and output.exists():
            raise FileExistsError(f"Refusing to overwrite evaluation: {output}")
        head = model.model[-1]
        if type(head) is not Detect or head.nc != 7:
            raise ValueError("SVRDD7 O2M evaluation requires an intact seven-class Detect")
        if self.native_o2m and head.end2end:
            raise ValueError("Native YOLO11 O2M evaluation received an end-to-end dual Detect")
        if not self.native_o2m and not head.end2end:
            raise ValueError("RS-Mid/B0 evaluation requires an intact native seven-class dual Detect")
        if self.max_images and hasattr(self.loader, "reset"):
            self.loader.reset()  # repeat the same diagnostic subset, not the next loader batch
        parameter = next(model.parameters())
        dtype, training, device = parameter.dtype, model.training, parameter.device
        old_args = model.args
        if isinstance(model.args, dict):
            model.args = SimpleNamespace(**model.args)
        blocks = [m for m in model.modules() if isinstance(m, RoadSnakeMidFusion)]
        detail_blocks = [m for m in model.modules() if isinstance(m, FrequencyDetailReconstruction)]
        diagnostic_blocks = [*blocks, *detail_blocks]
        old_capture = [m.capture_diagnostics for m in diagnostic_blocks]
        for m in diagnostic_blocks:
            m.capture_diagnostics = True
        model.float().eval()
        protocol = SimpleNamespace(**EVAL_PROTOCOL)
        rows = {"current": []}
        if voting:
            rows["score_box_vote"] = []
        seen_ids, loss_sum, batches = [], torch.zeros(3, device=device), 0
        try:
            for batch in self.loader:
                images = batch["img"].to(device, non_blocking=True).float() / 255.0
                predictions = model(images)
                raw = predictions[1] if isinstance(predictions, tuple) else predictions
                if not isinstance(raw, dict):
                    raise RuntimeError("Raw O2M output is missing; a fused/O2O-only model is not supported")
                o2m_raw = raw if self.native_o2m else raw.get("one2many")
                if not isinstance(o2m_raw, dict) or not {"boxes", "scores", "feats"}.issubset(o2m_raw):
                    raise RuntimeError("Raw O2M output is missing; a fused/O2O-only model is not supported")
                if compute_loss:
                    device_batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value
                                    for key, value in batch.items()}
                    device_batch["img"] = images
                    _, items = model.loss(device_batch, raw)
                    loss_sum += items.detach()
                decoded = head._inference(o2m_raw).float()
                # Native YOLO11/YOLO12 Detect decodes boxes as xywh, while the
                # frozen RS O2M helpers consume xyxy.  End-to-end YOLO26 heads
                # already decode to xyxy, so only convert the native path.
                if self.native_o2m:
                    boxes_xyxy = xywh2xyxy(decoded[:, :4].permute(0, 2, 1)).permute(0, 2, 1)
                    decoded = torch.cat((boxes_xyxy, decoded[:, 4:]), dim=1)
                levels = raw_level_indices(o2m_raw["feats"], device) if voting else None
                shape = tuple(images.shape[-2:])
                for index in range(images.shape[0]):
                    if self.max_images and len(seen_ids) >= self.max_images:
                        break
                    stem = Path(batch["im_file"][index]).stem
                    image_id = self.ids_by_stem[stem]
                    if image_id in seen_ids:
                        raise RuntimeError(f"Duplicate validation image: {stem}")
                    seen_ids.append(image_id)
                    current = current_nms(decoded[index], head.nc, protocol)
                    to_rows = lambda values: canonical_coco_rows(
                        values, image_id, self.category_ids, shape, batch["ori_shape"][index], batch["ratio_pad"][index])
                    current_rows = to_rows(current)
                    rows["current"].extend(current_rows)
                    if voting:
                        expanded = expanded_candidates(decoded[index], levels, head.nc, protocol)
                        roots, groups, _ = groups_anchored_to_reference(expanded[:, :6], current, protocol.nms_iou)
                        anchored = expanded[roots, :6].clone()
                        if to_rows(anchored) != current_rows:
                            raise RuntimeError("NMS reference and canonical anchored roots disagree")
                        rows["score_box_vote"].extend(to_rows(score_box_vote(anchored, expanded, groups)))
                batches += 1
                if len(seen_ids) % 100 == 0:
                    print(f"RS_MID_O2M_VAL images={len(seen_ids)} voting={voting}", flush=True)
                if self.max_images and len(seen_ids) >= self.max_images:
                    break
            expected = min(self.max_images, len(self.gt.imgs)) if self.max_images else len(self.gt.imgs)
            if len(seen_ids) != expected or (not self.max_images and set(seen_ids) != set(self.gt.imgs)):
                raise RuntimeError(f"Incomplete Val evaluation: {len(seen_ids)} / {expected}")
            metrics = {case: coco_summary(self.gt, case_rows, seen_ids, self.category_names)
                       for case, case_rows in rows.items()}
            if not math.isfinite(metrics["current"]["overall"]["AP"]):
                raise RuntimeError("Undefined O2M AP cannot be used for checkpoint selection")
            diagnostics = [{key: value.detach().cpu().tolist() for key, value in block.last_diagnostics.items()}
                           for block in blocks]
            payload = {"split": "val", "images": len(seen_ids), "partial": bool(self.max_images),
                       "classes": NAMES, "data": str(self.data_path), "coco": str(self.coco_path),
                       "protocol": {**EVAL_PROTOCOL, "imgsz": self.imgsz, "batch": self.batch, "workers": self.workers},
                       "metrics": metrics, "rs_mid_last_batch_diagnostics": diagnostics,
                       "detail_last_batch_diagnostics": [
                           {key: value.detach().cpu().tolist() for key, value in block.last_diagnostics.items()}
                           for block in detail_blocks
                       ]}
            if compute_loss:
                payload["loss_items"] = (loss_sum / max(batches, 1)).cpu().tolist()
            self.last_rows = rows
            if output is not None:
                output.mkdir(parents=True)
                for case, values in rows.items():
                    write_json(output / f"predictions_{case}.json", values)
                write_json(output / "metrics.json", payload)
            return payload
        finally:
            model.to(dtype=dtype).train(training)
            model.args = old_args
            for block, capture in zip(diagnostic_blocks, old_capture):
                block.capture_diagnostics = capture


class O2MMetrics:
    keys = ["metrics/mAP50(B)", "metrics/mAP50-95(B)", "metrics/AP75(B)", "metrics/AR100(B)"]

    def __init__(self):
        self.results_dict = dict.fromkeys(self.keys, 0.0)
        self.speed = {}


class O2MSelectionValidator:
    """Trainer adapter whose fitness is canonical COCO O2M AP, never native O2O AP."""

    def __init__(self, args, save_dir: Path, loader=None, max_images=0, native_o2m=False):
        self.args = copy(args)
        self.save_dir = Path(save_dir)
        self.metrics = O2MMetrics()
        self.plots = {}
        self.speed = {}
        self.evaluator = ValCOCOEvaluator(args.data, args.imgsz, args.batch, args.workers, max_images, loader,
                                          native_o2m=native_o2m)
        self.last_payload = None

    def __call__(self, trainer=None, model=None):
        if trainer is not None:
            net = trainer.ema.ema if trainer.ema else trainer.model
        elif isinstance(model, (str, Path)):
            from ultralytics.nn.tasks import load_checkpoint
            from ultralytics.utils.torch_utils import select_device
            net, _ = load_checkpoint(str(model), device=select_device(self.args.device, verbose=False))
        else:
            net = model
        if net is None:
            raise ValueError("A trainer, model or checkpoint is required")
        payload = self.evaluator.evaluate(net, compute_loss=trainer is not None)
        self.last_payload = payload
        values = payload["metrics"]["current"]["overall"]
        stats = dict(zip(self.metrics.keys, (values["AP50"], values["AP"], values["AP75"], values["AR100"])))
        stats["fitness"] = values["AP"]
        self.metrics.results_dict = dict(stats)
        if trainer is not None:
            stats.update(trainer.label_loss_items(payload["loss_items"], prefix="val"))
            record = {"epoch": int(trainer.epoch + 1), "selection": "COCO_O2M_no_voting_AP",
                      "images": payload["images"], "partial": payload["partial"], "metrics": values,
                      "diagnostics": {"rs_mid": payload["rs_mid_last_batch_diagnostics"],
                                      "detail": payload["detail_last_batch_diagnostics"]}}
            with (self.save_dir / "o2m_selection.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record) + "\n")
        print(f"COCO O2M/no Voting: AP={values['AP']:.6f} AP50={values['AP50']:.6f} AP75={values['AP75']:.6f}", flush=True)
        return stats
