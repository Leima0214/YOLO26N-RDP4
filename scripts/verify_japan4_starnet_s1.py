"""RTX static and Trainer-rebuild verification for official StarNet-S1-YOLO26."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from audit_japan4_parent_static import (  # noqa: E402
    amp_detection_step,
    capture_detect_shapes,
    export_onnx,
    latency,
    shape_tree,
    tensors,
)
from ultralytics.cfg import get_cfg  # noqa: E402
from ultralytics.models.yolo.detect.train import DetectionTrainer  # noqa: E402
from ultralytics.nn.tasks import DetectionModel  # noqa: E402
from ultralytics.utils.parent_semantic_init import (  # noqa: E402
    initialize_official_starnet_s1_yolo26,
    state_sha256,
)
from ultralytics.utils.torch_utils import get_flops, get_flops_with_torch_profiler  # noqa: E402

MODEL_YAML = ROOT / "ultralytics/cfg/models/26/yolo26-StarNet-S1-Official.yaml"


def group_names(state: dict[str, torch.Tensor]) -> dict[str, list[str]]:
    return {
        "official_backbone": [name for name in state if name.startswith(("model.0.stem.", "model.0.stages."))],
        "adapter": [name for name in state if name.startswith("model.0.adapters.")],
        "tail": [name for name in state if name.startswith(("model.4.", "model.5."))],
        "neck": [name for name in state if any(name.startswith(f"model.{i}.") for i in range(6, 18))],
        "detect_regression": [
            name for name in state if name.startswith(("model.18.cv2.", "model.18.one2one_cv2."))
        ],
        "detect_classification": [
            name for name in state if name.startswith(("model.18.cv3.", "model.18.one2one_cv3."))
        ],
    }


def hashes(state: dict[str, torch.Tensor]) -> dict[str, str]:
    return {name: state_sha256(state, names) for name, names in group_names(state).items()}


def trainer_rebuild_probe(model: DetectionModel, data: Path, report_dir: Path) -> dict:
    """Reproduce DetectionTrainer.get_model and require bit-exact initialized state."""
    before = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    overrides = {
        "model": str(MODEL_YAML),
        "data": str(data),
        "epochs": 30,
        "imgsz": 640,
        "batch": 32,
        "workers": 0,
        "device": "0",
        "seed": 42,
        "deterministic": True,
        "optimizer": "auto",
        "amp": True,
        "project": str(report_dir / "trainer_probe"),
        "name": "rebuild",
        "exist_ok": True,
        "verbose": False,
    }
    trainer = DetectionTrainer(overrides=overrides)
    rebuilt = trainer.get_model(weights=model, cfg=model.yaml, verbose=False).float()
    after = rebuilt.state_dict()
    mismatched = [name for name, value in before.items() if not torch.equal(value, after[name].cpu())]
    if mismatched:
        raise AssertionError(f"Trainer reconstruction changed initialized state: {mismatched[:10]}")
    return {
        "passed": True,
        "state_items": len(before),
        "mismatched": mismatched,
        "before_sha256": state_sha256(before),
        "after_sha256": state_sha256(after),
        "group_hashes_before": hashes(before),
        "group_hashes_after": hashes(after),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--star-checkpoint", type=Path, required=True)
    parser.add_argument("--yolo-weights", type=Path, default=ROOT / "yolo26n.pt")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--runs", type=int, default=100)
    args = parser.parse_args()
    args.report = args.report.expanduser().resolve()
    report_dir = args.report.parent
    report_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(42)
    model = DetectionModel(str(MODEL_YAML), ch=3, nc=4, verbose=False).float()
    model.args = get_cfg()
    initialization = initialize_official_starnet_s1_yolo26(model, args.star_checkpoint, args.yolo_weights)
    rebuild = trainer_rebuild_probe(copy.deepcopy(model), args.data.expanduser().resolve(), report_dir)

    device = torch.device(args.device)
    model = model.to(device).eval()
    sample = torch.zeros(1, 3, 640, 640, device=device)
    with torch.inference_mode():
        output, detect_shapes, top_level_shapes = capture_detect_shapes(model, sample)
    if not all(torch.isfinite(tensor).all() for tensor in tensors(output)):
        raise FloatingPointError("Non-finite forward output")
    expected_shapes = [[1, 64, 80, 80], [1, 128, 40, 40], [1, 256, 20, 20]]
    if detect_shapes != expected_shapes:
        raise AssertionError(f"Unexpected Detect inputs: {detect_shapes}")
    gflops = get_flops(model, 640) or get_flops_with_torch_profiler(model, 640)
    amp_step = amp_detection_step(model, device, 640)
    model.zero_grad(set_to_none=True)
    latency_result = latency(model, device, 640, args.warmup, args.runs)
    onnx_result = export_onnx(model, 640, args.onnx.expanduser().resolve())
    result = {
        "model": str(MODEL_YAML.resolve()),
        "initialization": initialization,
        "trainer_rebuild": rebuild,
        "forward": {
            "passed": True,
            "finite": True,
            "output_shapes": shape_tree(output),
            "detect_input_shapes_p3_p4_p5": detect_shapes,
            "top_level_shapes": top_level_shapes,
        },
        "detection_loss_backward_amp": amp_step,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "gflops": gflops,
        "latency": latency_result,
        "peak_vram_mib": amp_step["peak_allocated_mib"],
        "onnx": onnx_result,
        "test_split_read": False,
    }
    args.report.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

