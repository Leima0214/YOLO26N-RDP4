"""Val/Test-free static and semantic-initialization audit for the repository S2FracMix Neck."""

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
    load_matched,
    shape_tree,
    tensors,
    transfer_audit,
)
from ultralytics.cfg import get_cfg  # noqa: E402
from ultralytics.models.yolo.detect.train import DetectionTrainer  # noqa: E402
from ultralytics.nn.tasks import DetectionModel, torch_safe_load  # noqa: E402
from ultralytics.nn.yolo26_eccv2026 import S2FracMixBlock, S2FracMixC2f, S2FracMixFusion  # noqa: E402
from ultralytics.utils.torch_utils import get_flops, get_flops_with_torch_profiler  # noqa: E402
from verify_japan4_mshc_m2 import (  # noqa: E402
    B0_FEATURES,
    B0_YAML,
    capture_features,
    compare_features,
    feature_stats,
    load_and_verify,
    real_batch_probe,
    split_layer_key,
    state_sha256,
)

YAML_PATH = ROOT / "ultralytics/cfg/models/26/yolo26-ECCV2026-S2FracMixNeck.yaml"
FEATURES = B0_FEATURES
NAMES = {0: "D00", 1: "D10", 2: "D20", 3: "D40"}
UNCHANGED_LAYERS = set(range(0, 11)) | {17, 20, 23}
NEW_NECK_LAYERS = {12, 13, 15, 16, 18, 19, 21, 22}


def semantic_transfer(model: DetectionModel, weights: Path):
    checkpoint, _ = torch_safe_load(weights)
    source_model = (checkpoint.get("ema") or checkpoint["model"]).float()
    source, target = source_model.state_dict(), model.state_dict()
    parameters = dict(model.named_parameters())
    matched, mapping, mismatches, blocked = {}, {}, [], []
    for source_name, value in source.items():
        parsed = split_layer_key(source_name)
        if parsed is None:
            continue
        index, suffix = parsed
        target_name = f"model.{index}.{suffix}"
        if target_name not in target:
            continue
        same_shape = target[target_name].shape == value.shape
        same_type = type(source_model.model[index]) is type(model.model[index])
        if index in UNCHANGED_LAYERS and same_type and same_shape:
            matched[target_name] = value
            mapping[target_name] = source_name
        elif index in UNCHANGED_LAYERS:
            mismatches.append(
                {
                    "source": source_name,
                    "target": target_name,
                    "same_type": same_type,
                    "source_shape": list(value.shape),
                    "target_shape": list(target[target_name].shape),
                }
            )
        elif index in NEW_NECK_LAYERS and same_shape:
            blocked.append(
                {
                    "source": source_name,
                    "target": target_name,
                    "source_type": type(source_model.model[index]).__name__,
                    "target_type": type(model.model[index]).__name__,
                    "parameters": parameters[target_name].numel() if target_name in parameters else 0,
                    "reason": "same key/shape but different top-level module semantics",
                }
            )
    loaded_parameters = sum(parameters[name].numel() for name in matched if name in parameters)
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    return (
        {
            "rule": "inherit only unchanged B0 layers 0..10, neck downsample Conv 17/20, and shape-compatible Detect 23; block new Fusion/C2f even when keys and shapes coincide",
            "matched_state_items": len(matched),
            "target_state_items": len(target),
            "matched_target_parameters": loaded_parameters,
            "target_parameters": total_parameters,
            "matched_target_parameter_fraction": loaded_parameters / total_parameters,
            "expected_shape_mismatches": mismatches,
            "blocked_same_shape_items": blocked,
            "blocked_same_shape_parameter_total": sum(item["parameters"] for item in blocked),
        },
        matched,
        mapping,
    )


def component_coverage(model: DetectionModel, matched: dict[str, torch.Tensor]) -> dict:
    parameters = dict(model.named_parameters())
    groups = {
        "backbone_0_10": [name for name in parameters if (p := split_layer_key(name)) and p[0] <= 10],
        "new_fracmix_fusions": [name for name in parameters if (p := split_layer_key(name)) and p[0] in {12, 15, 18, 21}],
        "new_fracmix_c2f": [name for name in parameters if (p := split_layer_key(name)) and p[0] in {13, 16, 19, 22}],
        "retained_neck_downsample": [name for name in parameters if (p := split_layer_key(name)) and p[0] in {17, 20}],
        "detect_regression": [name for name in parameters if name.startswith(("model.23.cv2.", "model.23.one2one_cv2."))],
        "detect_classification": [name for name in parameters if name.startswith(("model.23.cv3.", "model.23.one2one_cv3."))],
    }
    result = {}
    for group, names in groups.items():
        total = sum(parameters[name].numel() for name in names)
        loaded = sum(parameters[name].numel() for name in names if name in matched)
        result[group] = {
            "loaded_parameters": loaded,
            "total_parameters": total,
            "coverage": loaded / total if total else 0.0,
            "random_parameter_names": [name for name in names if name not in matched],
        }
    return result


def trainer_rebuild(model: DetectionModel, data: Path, report_dir: Path) -> dict:
    before = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    trainer = DetectionTrainer(
        overrides={
            "model": str(YAML_PATH), "data": str(data), "epochs": 30, "imgsz": 640, "batch": 8,
            "workers": 0, "device": "0", "seed": 42, "deterministic": True, "optimizer": "auto",
            "amp": True, "project": str(report_dir / "trainer_probe"), "name": "s2fracmix_rebuild",
            "exist_ok": True, "plots": False, "verbose": False,
        }
    )
    rebuilt = trainer.get_model(weights=model, cfg=model.yaml, verbose=False).float()
    after = rebuilt.state_dict()
    mismatched = [name for name, value in before.items() if not torch.equal(value, after[name].cpu())]
    return {
        "passed": not mismatched,
        "before_sha256": state_sha256(before),
        "after_sha256": state_sha256(after),
        "mismatched": mismatched,
    }


def fixed_train_batch(model: DetectionModel, data: Path, device: torch.device, report_dir: Path):
    trainer = DetectionTrainer(
        overrides={
            "model": str(YAML_PATH), "data": str(data), "epochs": 30, "imgsz": 640, "batch": 8,
            "workers": 0, "device": "0", "seed": 42, "deterministic": True, "optimizer": "auto",
            "amp": True, "augment": False, "rect": True, "project": str(report_dir / "fixed_batch_probe"),
            "name": "loader", "exist_ok": True, "plots": False, "verbose": False,
        }
    )
    trainer.model = model.to(device)
    trainer.set_model_attributes()
    loader = trainer.get_dataloader(trainer.data["train"], batch_size=8, rank=-1, mode="val")
    raw = next(iter(loader))
    return trainer.preprocess_batch(raw), trainer.args, [str(path) for path in raw.get("im_file", [])]


def module_audit(model: DetectionModel) -> dict:
    result = {}
    for index, module in enumerate(model.model):
        if isinstance(module, S2FracMixFusion):
            blocks = [module.frac]
            result[str(index)] = {
                "type": type(module).__name__,
                "input_channels": [proj.conv.in_channels for proj in module.proj],
                "output_channels": module.c2,
                "reference_input": 0,
                "level_weights_init": module.level_logits.softmax(0).detach().cpu().tolist(),
                "frac_blocks": len(blocks),
                "scale_ratios": blocks[0].scale_ratios,
                "shape_kernels": [[3, 3], [1, 7], [7, 1]],
            }
        elif isinstance(module, S2FracMixC2f):
            result[str(index)] = {
                "type": type(module).__name__,
                "repeats": len(module.blocks),
                "scale_ratios": module.blocks[0].scale_ratios,
                "shape_kernels": [[3, 3], [1, 7], [7, 1]],
                "shortcut": module.shortcut,
            }
    return result


def build_b0(weights: Path):
    model = DetectionModel(B0_YAML, ch=3, nc=4, verbose=False).float()
    model.args = get_cfg()
    model.args.epochs = 30
    transfer, matched = transfer_audit(model, weights)
    load_matched(model, matched)
    return model, transfer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--runs", type=int, default=100)
    args = parser.parse_args()
    device, report_dir = torch.device(args.device), args.report.parent
    report_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(42)
    candidate = DetectionModel(str(YAML_PATH), ch=3, nc=4, verbose=False).float()
    candidate.args = get_cfg()
    candidate.args.epochs = 30
    transfer, matched, mapping = semantic_transfer(candidate, args.weights.resolve())
    load_and_verify(candidate, matched)
    coverage = component_coverage(candidate, matched)
    required = ("backbone_0_10", "retained_neck_downsample", "detect_regression")
    if any(coverage[name]["coverage"] != 1.0 for name in required):
        raise AssertionError({name: coverage[name] for name in required})
    if coverage["new_fracmix_fusions"]["coverage"] or coverage["new_fracmix_c2f"]["coverage"]:
        raise AssertionError("New FracMix neck received pretrained tensors")
    rebuild = trainer_rebuild(copy.deepcopy(candidate), args.data.resolve(), report_dir)
    if not rebuild["passed"]:
        raise AssertionError(rebuild)

    candidate = candidate.to(device).eval()
    sample = torch.zeros(1, 3, args.imgsz, args.imgsz, device=device)
    with torch.inference_mode():
        output, detect_shapes, top_shapes = capture_detect_shapes(candidate, sample)
    forward_finite = all(torch.isfinite(value).all() for value in tensors(output))
    amp = amp_detection_step(copy.deepcopy(candidate), device, args.imgsz)
    gflops = get_flops(candidate, args.imgsz) or get_flops_with_torch_profiler(candidate, args.imgsz)
    measured_latency = latency(candidate, device, args.imgsz, args.warmup, args.runs)
    try:
        onnx = export_onnx(candidate, args.imgsz, args.onnx.resolve())
    except Exception as error:
        onnx = {"passed": False, "error": f"{type(error).__name__}: {error}"}

    batch, trainer_args, paths = fixed_train_batch(candidate, args.data.resolve(), device, report_dir)
    b0, b0_transfer = build_b0(args.weights.resolve())
    b0 = b0.to(device)
    image = batch["img"]
    b0_features = capture_features(b0, image, B0_FEATURES)
    candidate_features = capture_features(candidate, image, FEATURES)
    losses = {
        "b0": real_batch_probe(copy.deepcopy(b0), copy.deepcopy(batch), trainer_args),
        "s2fracmix": real_batch_probe(
            copy.deepcopy(candidate), copy.deepcopy(batch), trainer_args, backward=True, mshc_indices=NEW_NECK_LAYERS
        ),
    }
    result = {
        "model": str(YAML_PATH.resolve()),
        "scope": {"split_read": "fixed non-augmented train batch only", "test_read": False, "training_started": False},
        "scientific_identity": {
            "repository_docstring": "detector-oriented adapters inspired by the papers, not drop-in copies of the full original training pipelines",
            "actual_form": "inference-time feature-space neck replacement",
            "paper_form": "training-time label-preserving within-image self-saliency/fractal mixup augmentation",
            "faithful_paper_implementation": False,
        },
        "structure": {
            "top_level_modules": [type(module).__name__ for module in candidate.model],
            "backbone": "B0 layers 0..10 unchanged",
            "fusion_replacements": [12, 15, 18, 21],
            "c3k2_replacements": [13, 16, 19, 22],
            "retained_neck_downsample": [17, 20],
            "detect": 23,
            "s2fracmix_blocks_total": 8,
            "module_audit": module_audit(candidate),
        },
        "parameters": sum(parameter.numel() for parameter in candidate.parameters()),
        "gflops": gflops,
        "pretrained_transfer": transfer,
        "semantic_target_to_source": mapping,
        "component_coverage": coverage,
        "b0_reference_transfer": b0_transfer,
        "trainer_rebuild": rebuild,
        "forward": {
            "passed": True, "finite": forward_finite, "output_shapes": shape_tree(output),
            "detect_input_shapes_p3_p4_p5": detect_shapes, "top_level_shapes": top_shapes,
        },
        "amp_detection_loss_backward": amp,
        "real_batch": {"images": paths, "losses": losses},
        "features": {
            "b0": {name: feature_stats(value) for name, value in b0_features.items()},
            "s2fracmix": {name: feature_stats(value) for name, value in candidate_features.items()},
            "s2fracmix_vs_b0": compare_features(candidate_features, b0_features),
        },
        "latency": measured_latency,
        "peak_vram_mib": amp["peak_allocated_mib"],
        "onnx": onnx,
        "engineering_pass": bool(
            forward_finite and rebuild["passed"] and losses["s2fracmix"]["finite"]
            and losses["s2fracmix"]["all_selected_gradients_finite_nonzero"] and onnx.get("passed")
        ),
    }
    args.report.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
