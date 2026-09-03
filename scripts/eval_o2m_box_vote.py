"""Shared O2M + Box-Voting evaluator for frozen checkpoints (paper protocol).

batch=32, rect=True, conf=0.001, NMS IoU=0.7, max_det=300, COCO maxDets=100,
FP32 (autocast disabled), Val only, Test sealed.  No training, no edits.

Exposes evaluate_model() so attribution scripts can evaluate in-memory hybrids.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from audit_gbrg_o2m_box_score_cross_swap import (  # noqa: E402
    coco_metrics,
    flat_ratio_pad,
    rebuild_ratio_pad,
)
from audit_gbrg_o2m_csrg_candidates import make_loader, paired_gt  # noqa: E402
from audit_gbrg_o2m_strict_reconstruction import (  # noqa: E402
    build_variants,
    canonical_coco_rows,
    current_nms,
    expanded_candidates,
    raw_level_indices,
)
from audit_gbrg_o2m_region_signal import groups_anchored_to_reference  # noqa: E402
from ultralytics.data.utils import check_det_dataset  # noqa: E402
from ultralytics.nn.modules.head import Detect  # noqa: E402
from ultralytics.utils.torch_utils import select_device  # noqa: E402

NAMES = ("D00", "D10", "D20", "D40")


def protocol_args(score_floor=0.001, nms_iou=0.70, max_det=300, max_nms=30000,
                  assignment_min_iou=0.10, ambiguous_gt_iou=0.50):
    return SimpleNamespace(
        score_floor=score_floor, nms_iou=nms_iou, max_det=max_det, max_nms=max_nms,
        assignment_min_iou=assignment_min_iou, ambiguous_gt_iou=ambiguous_gt_iou,
    )


def evaluate_model(
    net: torch.nn.Module,
    args: argparse.Namespace,
    data_yaml: Path,
    device: torch.device,
    progress: Callable[[str], None] | None = None,
    max_images: int = 0,
) -> dict[str, Any]:
    """Return coco metrics for 'current' and 'score_box_vote' variants."""
    data_info = check_det_dataset(str(data_yaml))
    from pycocotools.coco import COCO

    coco_path = Path(data_info["path"]) / "annotations" / "instances_val.json"
    coco_gt = COCO(str(coco_path))
    image_id_by_stem = {Path(image["file_name"]).stem: image_id for image_id, image in coco_gt.imgs.items()}
    category_id_by_name = {category["name"]: category_id for category_id, category in coco_gt.cats.items()}
    names = [data_info["names"][index] for index in range(len(data_info["names"]))]
    category_ids = {index: category_id_by_name[name] for index, name in enumerate(names)}

    _, loader = make_loader(data_yaml, args)
    net = net.to(device).float().eval()
    head = net.model[-1]
    if not isinstance(head, Detect) or not head.end2end or head.nc != len(names):
        raise RuntimeError(f"Not a compatible four-class end-to-end checkpoint: {type(head).__name__}")
    if isinstance(net.args, dict):
        net.args = SimpleNamespace(**net.args)

    rows: dict[str, list[dict[str, Any]]] = {"current": [], "score_box_vote": []}
    seen = 0
    with torch.inference_mode():
        for batch in loader:
            images = batch["img"].to(device, non_blocking=True).float() / 255.0
            device_batch = {
                key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
                for key, value in batch.items()
            }
            device_batch["img"] = images
            output = net(images)
            raw = output[1] if isinstance(output, tuple) else output
            one2many = raw["one2many"]
            decoded = head._inference(one2many).float()
            level_by_raw = raw_level_indices(one2many["feats"], device)
            model_shape = tuple(images.shape[-2:])
            p_args = protocol_args()
            for image_index in range(images.shape[0]):
                if max_images and seen >= max_images:
                    break
                image_name = Path(batch["im_file"][image_index]).name
                image_id = image_id_by_stem[Path(image_name).stem]
                gt_boxes, gt_classes = paired_gt(device_batch, image_index, model_shape[1], model_shape[0])
                expanded = expanded_candidates(decoded[image_index], level_by_raw, head.nc, p_args)
                reference = current_nms(decoded[image_index], head.nc, p_args)
                roots, groups, _ = groups_anchored_to_reference(expanded[:, :6], reference, p_args.nms_iou)
                current = expanded[roots, :6].clone()
                variants, _ = build_variants(
                    "eval", image_name, current, expanded, groups, gt_boxes, gt_classes,
                    model_shape, batch["ori_shape"][image_index], batch["ratio_pad"][image_index],
                    names, p_args,
                )
                for case in ("current", "score_box_vote"):
                    rows[case].extend(
                        canonical_coco_rows(
                            variants[case], image_id, category_ids, model_shape,
                            batch["ori_shape"][image_index], batch["ratio_pad"][image_index],
                        )
                    )
                seen += 1
                if progress is not None and seen % 100 == 0:
                    progress(f"images={seen}")
            if max_images and seen >= max_images:
                break

    metrics = {case: coco_metrics(coco_gt, rows[case]) for case in rows}
    return {"metrics": metrics, "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--score-floor", type=float, default=0.001)
    parser.add_argument("--nms-iou", type=float, default=0.70)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--max-nms", type=int, default=30000)
    parser.add_argument("--max-images", type=int, default=0)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {args.output}")
    args.output.mkdir(parents=True)

    from ultralytics import YOLO

    wrapped = YOLO(str(args.checkpoint.resolve()))
    net = wrapped.model
    device = select_device(args.device, verbose=False)
    result = evaluate_model(net, args, args.data.resolve(), device, max_images=args.max_images)
    (args.output / "metrics.json").write_text(
        json.dumps(result["metrics"], indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print("EVAL " + json.dumps(result["metrics"], indent=2), flush=True)


if __name__ == "__main__":
    main()
