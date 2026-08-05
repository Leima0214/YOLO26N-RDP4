"""Build and benchmark static TensorRT FP16 engines for RoadLite-26 candidates."""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ultralytics import YOLO  # noqa: E402
from ultralytics.nn.autobackend import AutoBackend  # noqa: E402


MODELS = {
    "B0": ROOT / "ultralytics/cfg/models/26/yolo26.yaml",
    "A1": ROOT / "ultralytics/cfg/models/26/roadlite26-a1.yaml",
    "A2": ROOT / "ultralytics/cfg/models/26/roadlite26-a2.yaml",
    "A3": ROOT / "ultralytics/cfg/models/26/roadlite26-a3.yaml",
}


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[min(round((len(ordered) - 1) * fraction), len(ordered) - 1)]


def timed_cuda(call, warmup, runs):
    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    values = []
    for _ in range(runs):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        call()
        end.record()
        end.synchronize()
        values.append(start.elapsed_time(end))
    return statistics.median(values), percentile(values, 0.95)


def engine_path(name, batch, args):
    output = args.output / "engines" / f"{name.lower()}_fp16_b{batch}.engine"
    if output.is_file():
        return output
    config_dir = args.output / "engine_configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    config = config_dir / f"{name.lower()}_fp16_b{batch}.yaml"
    config.write_text(MODELS[name].read_text(encoding="utf-8").replace("nc: 80", f"nc: {args.nc}", 1), encoding="utf-8")
    exported = Path(
        YOLO(str(config), task="detect").export(
            format="engine",
            imgsz=args.imgsz,
            batch=batch,
            half=True,
            dynamic=False,
            device=0,
            workspace=args.workspace,
            verbose=False,
        )
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(exported, output)
    onnx = exported.with_suffix(".onnx")
    if onnx.is_file():
        shutil.move(onnx, output.with_suffix(".onnx"))
    return output


def benchmark_engine(path, batch, args):
    backend = AutoBackend(str(path), device=torch.device("cuda:0"), fp16=True, verbose=False)
    image = torch.zeros(batch, 3, args.imgsz, args.imgsz, device="cuda", dtype=torch.float16)
    torch.cuda.reset_peak_memory_stats()
    median, p95 = timed_cuda(lambda: backend(image), args.warmup, args.runs)
    return {
        "engine": str(path),
        "engine_bytes": path.stat().st_size,
        "batch": batch,
        "median_ms": median,
        "p95_ms": p95,
        "throughput_images_s": batch * 1000 / median,
        "peak_memory_mib": torch.cuda.max_memory_allocated() / 2**20,
    }


def benchmark_e2e(path, args):
    model = YOLO(str(path), task="detect")
    image = np.zeros((args.imgsz, args.imgsz, 3), dtype=np.uint8)
    call = lambda: model.predict(image, imgsz=args.imgsz, device=0, verbose=False)
    for _ in range(min(args.warmup, 50)):
        call()
    values = []
    for _ in range(min(args.runs, 200)):
        torch.cuda.synchronize()
        start = time.perf_counter()
        call()
        torch.cuda.synchronize()
        values.append((time.perf_counter() - start) * 1000)
    return {"median_ms": statistics.median(values), "p95_ms": percentile(values, 0.95)}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--batches", nargs="+", type=int, default=[1, 32])
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--nc", type=int, default=4)
    parser.add_argument("--workspace", type=float, default=4)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--runs", type=int, default=1000)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/roadlite26_tensorrt")
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in args.models:
        model_rows = [benchmark_engine(engine_path(name, batch, args), batch, args) for batch in args.batches]
        rows.append({"name": name, "engines": model_rows, "e2e_batch1": benchmark_e2e(Path(model_rows[0]["engine"]), args)})
    payload = {
        "protocol": {"imgsz": args.imgsz, "nc": args.nc, "fp16": True, "static": True, "warmup": args.warmup, "runs": args.runs, "workspace_gib": args.workspace, "gpu": torch.cuda.get_device_name(0)},
        "models": rows,
    }
    output = args.output / "benchmark.json"
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
