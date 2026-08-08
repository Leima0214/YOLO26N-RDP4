"""Audit legal YOLO26 semantic inheritance for full-backbone parent candidates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ultralytics.nn.tasks import DetectionModel  # noqa: E402
from ultralytics.utils.parent_semantic_init import initialize_parent  # noqa: E402

MODELS = {
    "starnet": ROOT / "ultralytics/cfg/models/26/yolo26-StarNet.yaml",
    "mobilemamba": ROOT / "ultralytics/cfg/models/26/yolo26-MobileMamba-Backbone.yaml",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=ROOT / "yolo26n.pt")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    weights = args.weights.expanduser().resolve()
    if not weights.is_file():
        raise FileNotFoundError(weights)

    report = {
        "policy": {
            "semantic_and_shape_required": True,
            "shape_only_loading_forbidden": True,
            "random_backbone_formal_30e_forbidden": True,
            "test_split_read": False,
        },
        "models": {},
    }
    for candidate, yaml_path in MODELS.items():
        torch.manual_seed(42)
        model = DetectionModel(str(yaml_path), ch=3, nc=4, verbose=False).float()
        audit = initialize_parent(model, candidate, weights)
        report["models"][candidate] = audit

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    compact = {}
    for name, audit in report["models"].items():
        compact[name] = {
            "backbone_pretrained": audit["backbone_pretrained"],
            "components": audit["components"],
            "loaded_parameter_fraction": audit["loaded_parameter_fraction"],
            "random_parameter_fraction": audit["random_parameter_fraction"],
            "formal_30e_eligible": audit["components"]["backbone"]["coverage"] > 0.0,
        }
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
