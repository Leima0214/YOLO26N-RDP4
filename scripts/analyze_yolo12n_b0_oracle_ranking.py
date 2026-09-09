"""Evaluate a GT-dependent score-IoU oracle on saved YOLO12n B0 predictions.

This is a diagnostic upper bound only. It never changes boxes/classes, never
reads Test, and must reproduce the canonical baseline before reporting deltas.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT / "scripts"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from rs_mid_bootstrap import guard_optional_visualization_imports  # noqa: E402

guard_optional_visualization_imports()

from rs_mid_o2m import NAMES, coco_summary, write_json  # noqa: E402
from ultralytics.data.utils import check_det_dataset  # noqa: E402


def iou(box: list[float], gt: list[float]) -> float:
    x1, y1, x2, y2 = box
    gx, gy, gw, gh = gt
    gx2, gy2 = gx + gw, gy + gh
    iw, ih = max(0.0, min(x2, gx2) - max(x1, gx)), max(0.0, min(y2, gy2) - max(y1, gy))
    inter = iw * ih
    union = max(0.0, x2 - x1) * max(0.0, y2 - y1) + gw * gh - inter
    return inter / union if union > 0 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnosis", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    diagnosis, data, output = (x.expanduser().resolve() for x in (args.diagnosis, args.data, args.output))
    if output.exists():
        raise FileExistsError(output)
    records = json.loads((diagnosis / "image_records.json").read_text(encoding="utf-8"))
    canonical = json.loads((diagnosis / "b0_metrics.json").read_text(encoding="utf-8"))
    info = check_det_dataset(str(data), autodownload=False)
    test = info.get("test")
    if not test:
        raise RuntimeError("Test sealing cannot be verified")
    from pycocotools.coco import COCO

    gt = COCO(str(Path(info["path"]) / "annotations" / "instances_val.json"))
    gt.dataset.setdefault("info", {})
    ids_by_stem = {Path(item["file_name"]).stem: iid for iid, item in gt.imgs.items()}
    category_by_name = {item["name"]: cid for cid, item in gt.cats.items()}
    category_names = {cid: name for name, cid in category_by_name.items()}
    baseline, oracle = [], []
    for stem, record in records.items():
        image_id = ids_by_stem[stem]
        anns = gt.imgToAnns[image_id]
        for pred in record["pred"]:
            box = [float(x) for x in pred["box"]]
            name = pred["class"]
            xywh = [box[0], box[1], box[2] - box[0], box[3] - box[1]]
            row = {"image_id": image_id, "category_id": category_by_name[name], "bbox": xywh,
                   "score": float(pred["score"])}
            baseline.append(row)
            same = [ann for ann in anns if ann["category_id"] == category_by_name[name]]
            quality = max((iou(box, ann["bbox"]) for ann in same), default=0.0)
            oracle.append({**row, "score": quality})
    image_ids = sorted(ids_by_stem.values())
    baseline_metrics = coco_summary(gt, baseline, image_ids, category_names)
    oracle_metrics = coco_summary(gt, oracle, image_ids, category_names)
    expected = canonical["overall"]["AP"]
    if not math.isclose(baseline_metrics["overall"]["AP"], expected, abs_tol=1e-4):
        raise RuntimeError({"stored_prediction_AP": baseline_metrics["overall"]["AP"], "canonical_AP": expected})
    payload = {
        "protocol": {"split": "val", "test_accessed": False, "boxes_unchanged": True, "classes_unchanged": True,
                     "oracle": "score replaced by maximum same-class GT IoU; diagnostic upper bound only"},
        "baseline": baseline_metrics,
        "oracle_iou_score": oracle_metrics,
        "delta": {key: oracle_metrics["overall"][key] - baseline_metrics["overall"][key]
                  for key in baseline_metrics["overall"]},
    }
    write_json(output, payload)
    print(json.dumps(payload["delta"], indent=2))


if __name__ == "__main__":
    main()
