"""Audit B0 and RoadLite-26 asymmetric channel candidates on one fixed protocol."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ultralytics.nn.tasks import DetectionModel, torch_safe_load
from ultralytics.utils.torch_utils import intersect_dicts

DEFAULT_MODELS = {
    "B0": ROOT / "ultralytics/cfg/models/26/yolo26.yaml",
    "A1": ROOT / "ultralytics/cfg/models/26/roadlite26-a1.yaml",
    "A2": ROOT / "ultralytics/cfg/models/26/roadlite26-a2.yaml",
    "A3": ROOT / "ultralytics/cfg/models/26/roadlite26-a3.yaml",
}
EXPECTED_CHANNELS = {
    "B0": [128, 128, 256, 64, 128, 256],
    "A1": [128, 96, 160, 64, 96, 160],
    "A2": [128, 80, 128, 64, 80, 128],
    "A3": [144, 96, 128, 72, 96, 128],
}
AUDIT_LAYERS = (4, 6, 10, 16, 19, 22)


def shape_of(value):
    if isinstance(value, torch.Tensor):
        return list(value.shape)
    if isinstance(value, (list, tuple)):
        return [shape_of(item) for item in value]
    if isinstance(value, dict):
        return {key: shape_of(item) for key, item in value.items()}
    return type(value).__name__


def tensor_bytes(value):
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, (list, tuple)):
        return sum(tensor_bytes(item) for item in value)
    if isinstance(value, dict):
        return sum(tensor_bytes(item) for item in value.values())
    return 0


def forward_layers(model, image, with_macs=True):
    rows, saved, current = [], [], image
    for layer in model.model:
        layer_input = (
            saved[layer.f]
            if isinstance(layer.f, int) and layer.f != -1
            else [current if index == -1 else saved[index] for index in layer.f]
            if isinstance(layer.f, list)
            else current
        )
        macs = [0]
        handles = []
        if with_macs:
            def count_ops(module, _inputs, output):
                if isinstance(module, torch.nn.Conv2d):
                    macs[0] += output.numel() * module.in_channels // module.groups * module.kernel_size[0] * module.kernel_size[1]
                elif isinstance(module, torch.nn.Linear):
                    macs[0] += output.numel() * module.in_features

            handles = [
                module.register_forward_hook(count_ops)
                for module in layer.modules()
                if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear))
            ]
        current = layer(layer_input)
        for handle in handles:
            handle.remove()
        saved.append(current if layer.i in model.save else None)
        rows.append(
            {
                "index": layer.i,
                "from": layer.f,
                "module": layer.type,
                "params": sum(parameter.numel() for parameter in layer.parameters()),
                "macs": macs[0],
                "output": shape_of(current),
                "activation_bytes": tensor_bytes(current),
            }
        )
    return rows


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[min(round((len(ordered) - 1) * fraction), len(ordered) - 1)]


@torch.inference_mode()
def cuda_latency(model, imgsz, batch, warmup, runs, half):
    dtype = torch.float16 if half else torch.float32
    image = torch.zeros(batch, 3, imgsz, imgsz, device="cuda", dtype=dtype)
    model = model.cuda().eval()
    if half:
        model.half()
    for _ in range(warmup):
        model(image)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    samples = []
    for _ in range(runs):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        model(image)
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return {
        "batch": batch,
        "dtype": str(dtype).removeprefix("torch."),
        "median_ms": statistics.median(samples),
        "p95_ms": percentile(samples, 0.95),
        "throughput_images_s": batch * 1000 / statistics.median(samples),
        "peak_memory_mib": torch.cuda.max_memory_allocated() / 2**20,
    }


def transfer_coverage(model, weights_path):
    if not weights_path.is_file():
        return None
    checkpoint, _ = torch_safe_load(weights_path)
    source = (checkpoint.get("ema") or checkpoint["model"]).float().state_dict()
    target = model.state_dict()
    matched = intersect_dicts(source, target)
    return {
        "matched_tensors": len(matched),
        "target_tensors": len(target),
        "tensor_fraction": len(matched) / len(target),
        "matched_parameters": sum(value.numel() for value in matched.values()),
        "target_parameters": sum(value.numel() for value in target.values()),
        "parameter_fraction": sum(value.numel() for value in matched.values()) / sum(value.numel() for value in target.values()),
    }


def audit(name, yaml_path, args):
    model = DetectionModel(str(yaml_path), ch=3, nc=args.nc, verbose=False).eval()
    image = torch.zeros(1, 3, args.imgsz, args.imgsz)
    rows = forward_layers(model, image)
    channels = [rows[index]["output"][1] for index in AUDIT_LAYERS]
    if name in EXPECTED_CHANNELS:
        assert channels == EXPECTED_CHANNELS[name], (name, channels, EXPECTED_CHANNELS[name])
    result = {
        "name": name,
        "yaml": str(yaml_path),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "gflops": 2 * sum(row["macs"] for row in rows) / 1e9,
        "channels_P3_P4_P5_backbone_detect": channels,
        "pretrained_transfer": transfer_coverage(model, args.weights),
        "layers": rows,
    }
    if args.cuda:
        result["pytorch_latency"] = [
            cuda_latency(model, args.imgsz, batch, args.warmup, args.runs, args.half)
            for batch in (1, 32)
        ]
    return result


def markdown(results):
    baseline = results[0]
    lines = [
        "# RoadLite-26 Stage-1 static audit",
        "",
        "Protocol: nc=4, input=640x640. GFLOPs = 2 x Conv/Linear MACs; B0 differs from the recorded THOP value by <0.1%.",
        "",
        "| Model | Params | Params vs B0 | GFLOPs | GFLOPs vs B0 | Backbone P3/P4/P5 | Detect P3/P4/P5 | Transfer params |",
        "|---|---:|---:|---:|---:|---|---|---:|",
    ]
    for result in results:
        transfer = result["pretrained_transfer"]
        lines.append(
            f"| {result['name']} | {result['parameters']:,} | "
            f"{result['parameters'] / baseline['parameters'] - 1:+.1%} | {result['gflops']:.4f} | "
            f"{result['gflops'] / baseline['gflops'] - 1:+.1%} | "
            f"{result['channels_P3_P4_P5_backbone_detect'][:3]} | "
            f"{result['channels_P3_P4_P5_backbone_detect'][3:]} | "
            f"{transfer['parameter_fraction']:.1%} |" if transfer else
            f"| {result['name']} | {result['parameters']:,} | "
            f"{result['parameters'] / baseline['parameters'] - 1:+.1%} | {result['gflops']:.4f} | "
            f"{result['gflops'] / baseline['gflops'] - 1:+.1%} | "
            f"{result['channels_P3_P4_P5_backbone_detect'][:3]} | "
            f"{result['channels_P3_P4_P5_backbone_detect'][3:]} | n/a |"
        )
    if "pytorch_latency" in baseline:
        lines.extend([
            "",
            "## PyTorch CUDA latency",
            "",
            "| Model | B1 median | B1 p95 | B32 throughput | B32 peak VRAM |",
            "|---|---:|---:|---:|---:|",
        ])
        for result in results:
            batch1, batch32 = result["pytorch_latency"]
            lines.append(
                f"| {result['name']} | {batch1['median_ms']:.3f} ms | {batch1['p95_ms']:.3f} ms | "
                f"{batch32['throughput_images_s']:.1f} img/s | {batch32['peak_memory_mib']:.1f} MiB |"
            )
    lines.extend([
        "",
        "## Complete channel mapping",
        "",
        "| i | module | B0 | A1 | A2 | A3 |",
        "|---:|---|---:|---:|---:|---:|",
    ])
    for index in range(23):
        outputs = [result["layers"][index]["output"] for result in results]
        channels = [output[1] if isinstance(output, list) and len(output) == 4 else "-" for output in outputs]
        lines.append(f"| {index} | {baseline['layers'][index]['module'].rsplit('.', 1)[-1]} | {channels[0]} | {channels[1]} | {channels[2]} | {channels[3]} |")
    lines.extend(["", "## B0 per-layer audit", "", "| i | from | module | params | MACs | activation MiB | output |", "|---:|---|---|---:|---:|---:|---|"])
    for row in baseline["layers"]:
        lines.append(
            f"| {row['index']} | {row['from']} | {row['module']} | {row['params']:,} | "
            f"{row['macs'] / 1e6:.3f}M | {row['activation_bytes'] / 2**20:.3f} | `{row['output']}` |"
        )
    return "\n".join(lines) + "\n"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--nc", type=int, default=4)
    parser.add_argument("--weights", type=Path, default=ROOT / "yolo26n.pt")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/roadlite26_stage1_audit.json")
    parser.add_argument("--cuda", action="store_true", help="also benchmark PyTorch batch 1 and 32")
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--runs", type=int, default=200)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.cuda and not torch.cuda.is_available():
        raise RuntimeError("--cuda requested but CUDA is unavailable")
    started = time.time()
    results = [audit(name, path, args) for name, path in DEFAULT_MODELS.items()]
    payload = {"protocol": vars(args) | {"weights": str(args.weights), "output": str(args.output)}, "seconds": time.time() - started, "models": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    args.output.with_suffix(".md").write_text(markdown(results), encoding="utf-8")
    print(markdown(results))


if __name__ == "__main__":
    main()
