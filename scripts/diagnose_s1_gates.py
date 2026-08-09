"""Audit frozen S1 gates and branch responses by Japan4 class and COCO size."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ultralytics import YOLO  # noqa: E402
from ultralytics.cfg import get_cfg  # noqa: E402
from ultralytics.data import build_dataloader, build_yolo_dataset  # noqa: E402
from ultralytics.data.utils import check_det_dataset  # noqa: E402
from ultralytics.nn.modules.head import StripAwareResidual  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-batches", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    device = torch.device(f"cuda:{args.device}" if args.device.isdigit() else args.device)
    cfg = get_cfg(overrides={"imgsz": args.imgsz, "batch": args.batch, "workers": args.workers, "data": str(args.data)})
    data = check_det_dataset(str(args.data.resolve()))
    dataset = build_yolo_dataset(cfg, data["val"], args.batch, data, mode="val", rect=True, stride=32)
    loader = build_dataloader(dataset, args.batch, args.workers, shuffle=False, rank=-1)

    model = YOLO(str(args.weights), task="detect").model.to(device).eval()
    head = model.model[-1]
    named_adapters = {
        name: module for name, module in head.named_modules() if isinstance(module, StripAwareResidual)
    }
    if len(named_adapters) != 4:
        raise AssertionError(f"Expected four S1 adapters, found {list(named_adapters)}")

    captures = {}
    hooks = []
    for adapter_name, adapter in named_adapters.items():
        captures[adapter_name] = {}
        for branch_name in ("square", "horizontal", "vertical"):
            branch = getattr(adapter, branch_name)

            def capture(_, __, output, key=adapter_name, branch_key=branch_name):
                captures[key][branch_key] = output.detach().abs().mean(dim=(1, 2, 3)).cpu()

            hooks.append(branch.register_forward_hook(capture))

    grouped = {"class": defaultdict(lambda: defaultdict(list)), "size": defaultdict(lambda: defaultdict(list))}
    with torch.inference_mode():
        for batch_number, batch in enumerate(loader):
            if batch_number >= args.max_batches:
                break
            image = batch["img"].to(device).float() / 255
            model(image)
            image_classes = defaultdict(set)
            image_sizes = defaultdict(set)
            for index, class_id, box in zip(batch["batch_idx"].long(), batch["cls"].long().flatten(), batch["bboxes"]):
                image_classes[int(index)].add(data["names"][int(class_id)])
                area = float(box[2] * args.imgsz * box[3] * args.imgsz)
                image_sizes[int(index)].add("small" if area < 32**2 else "medium" if area < 96**2 else "large")
            for image_index in range(image.shape[0]):
                for adapter_name, branches in captures.items():
                    for branch_name, values in branches.items():
                        metric = f"{adapter_name}.{branch_name}"
                        for class_name in image_classes[image_index]:
                            grouped["class"][class_name][metric].append(float(values[image_index]))
                        for size_name in image_sizes[image_index]:
                            grouped["size"][size_name][metric].append(float(values[image_index]))

    for hook in hooks:
        hook.remove()

    report = {
        "weights": str(args.weights.resolve()),
        "gate_conditioning": "global parameters; weights do not vary by image, class, or size",
        "adapters": {
            name: {
                "gamma": float(module.gamma.detach().cpu()),
                "weights_square_horizontal_vertical": module.branch_weights().detach().cpu().tolist(),
            }
            for name, module in named_adapters.items()
        },
        "branch_response_mean": {
            group_type: {
                group: {metric: sum(values) / len(values) for metric, values in metrics.items() if values}
                for group, metrics in groups.items()
            }
            for group_type, groups in grouped.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
