"""Val/Test-free static and initialization audit for Japan4 MSHC M2."""

from __future__ import annotations

import argparse
import copy
import hashlib
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
from ultralytics.nn.yolo26_cvpr_improvements.modules import MSHCBlock  # noqa: E402
from ultralytics.utils.torch_utils import get_flops, get_flops_with_torch_profiler  # noqa: E402

M2_YAML = ROOT / "ultralytics/cfg/models/26/yolo26-MSHC-M2.yaml"
M1_YAML = ROOT / "ultralytics/cfg/models/26/yolo26-MSHC.yaml"
B0_YAML = "yolo26n.yaml"
NAMES = {0: "D00", 1: "D10", 2: "D20", 3: "D40"}

B0_FEATURES = {"backbone_p3": 4, "backbone_p4": 6, "backbone_p5": 10, "detect_p3": 16, "detect_p4": 19, "detect_p5": 22}
M1_FEATURES = B0_FEATURES
M2_FEATURES = {"backbone_p3": 4, "backbone_p4": 7, "backbone_p5": 11, "detect_p3": 17, "detect_p4": 20, "detect_p5": 23}


def state_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def split_layer_key(name: str) -> tuple[int, str] | None:
    parts = name.split(".", 2)
    if len(parts) != 3 or parts[0] != "model" or not parts[1].isdigit():
        return None
    return int(parts[1]), parts[2]


def m2_semantic_transfer(model: DetectionModel, weights: Path) -> tuple[dict, dict[str, torch.Tensor], dict[str, str]]:
    """Map unchanged YOLO26 layers semantically across M2's inserted P4 block."""
    checkpoint, _ = torch_safe_load(weights)
    source_model = (checkpoint.get("ema") or checkpoint["model"]).float()
    source, target = source_model.state_dict(), model.state_dict()
    matched, target_to_source, unsafe, shape_mismatches = {}, {}, [], []
    for source_name, value in source.items():
        parsed = split_layer_key(source_name)
        if parsed is None:
            continue
        source_index, suffix = parsed
        if source_index <= 5:
            target_index = source_index
        elif source_index == 6:
            continue  # B0 P4 C3k2 is replaced by two new MSHC blocks.
        else:
            target_index = source_index + 1
        target_name = f"model.{target_index}.{suffix}"
        if target_name not in target:
            continue
        same_type = type(source_model.model[source_index]) is type(model.model[target_index])
        same_shape = target[target_name].shape == value.shape
        if same_type and same_shape:
            matched[target_name] = value
            target_to_source[target_name] = source_name
        elif not same_type:
            unsafe.append(
                {
                    "source": source_name,
                    "target": target_name,
                    "source_type": type(source_model.model[source_index]).__name__,
                    "target_type": type(model.model[target_index]).__name__,
                    "source_shape": list(value.shape),
                    "target_shape": list(target[target_name].shape),
                }
            )
        else:
            shape_mismatches.append(
                {"source": source_name, "target": target_name, "source_shape": list(value.shape), "target_shape": list(target[target_name].shape)}
            )
    parameters = dict(model.named_parameters())
    loaded_parameters = sum(parameters[name].numel() for name in matched if name in parameters)
    report = {
        "rule": "same semantic top-level module, same suffix, same shape; B0 layers 7..23 shift to M2 layers 8..24",
        "matched_state_items": len(matched),
        "target_state_items": len(target),
        "matched_target_parameters": loaded_parameters,
        "target_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "matched_target_parameter_fraction": loaded_parameters / sum(parameter.numel() for parameter in model.parameters()),
        "unsafe_semantic_candidates": unsafe,
        "expected_shape_mismatches": shape_mismatches,
        "mshc_loaded_items": [name for name in matched if split_layer_key(name)[0] in {6, 7}],
    }
    return report, matched, target_to_source


def load_and_verify(model: DetectionModel, matched: dict[str, torch.Tensor]) -> None:
    model.load_state_dict(matched, strict=False)
    actual = model.state_dict()
    changed = [name for name, value in matched.items() if not torch.equal(actual[name].cpu(), value.cpu())]
    if changed:
        raise AssertionError(f"Transferred tensors changed: {changed[:10]}")


def component_coverage(model: DetectionModel, matched: dict[str, torch.Tensor]) -> dict:
    parameters = dict(model.named_parameters())
    groups = {
        "p3_original_layer4": [name for name in parameters if name.startswith("model.4.")],
        "p4_mshc_layer6": [name for name in parameters if name.startswith("model.6.")],
        "p4_mshc_layer7": [name for name in parameters if name.startswith("model.7.")],
        "other_backbone": [name for name in parameters if (parsed := split_layer_key(name)) and parsed[0] in {0, 1, 2, 3, 5, 8, 9, 10, 11}],
        "neck": [name for name in parameters if (parsed := split_layer_key(name)) and parsed[0] in set(range(12, 24))],
        "detect_regression": [name for name in parameters if name.startswith(("model.24.cv2.", "model.24.one2one_cv2."))],
        "detect_classification": [name for name in parameters if name.startswith(("model.24.cv3.", "model.24.one2one_cv3."))],
    }
    output = {}
    for group, names in groups.items():
        total = sum(parameters[name].numel() for name in names)
        loaded = sum(parameters[name].numel() for name in names if name in matched)
        output[group] = {
            "loaded_parameters": loaded,
            "total_parameters": total,
            "coverage": loaded / total if total else 0.0,
            "random_parameter_names": [name for name in names if name not in matched],
        }
    return output


def trainer_rebuild(model: DetectionModel, data: Path, report_dir: Path) -> dict:
    before = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    trainer = DetectionTrainer(
        overrides={
            "model": str(M2_YAML), "data": str(data), "epochs": 30, "imgsz": 640, "batch": 8,
            "workers": 0, "device": "0", "seed": 42, "deterministic": True, "optimizer": "auto",
            "amp": True, "project": str(report_dir / "trainer_probe"), "name": "m2_rebuild",
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
            "model": str(M2_YAML), "data": str(data), "epochs": 30, "imgsz": 640, "batch": 8,
            "workers": 0, "device": "0", "seed": 42, "deterministic": True, "optimizer": "auto",
            "amp": True, "augment": False, "rect": True, "project": str(report_dir / "fixed_batch_probe"),
            "name": "loader", "exist_ok": True, "plots": False, "verbose": False,
        }
    )
    trainer.model = model.to(device)
    trainer.set_model_attributes()
    loader = trainer.get_dataloader(trainer.data["train"], batch_size=8, rank=-1, mode="val")
    raw = next(iter(loader))
    paths = [str(path) for path in raw.get("im_file", [])]
    return trainer.preprocess_batch(raw), trainer.args, paths


def capture_features(model: DetectionModel, image: torch.Tensor, indices: dict[str, int]) -> dict[str, torch.Tensor]:
    captured, handles = {}, []
    for name, index in indices.items():
        def hook(_module, _inputs, output, key=name):
            captured[key] = output.detach()
        handles.append(model.model[index].register_forward_hook(hook))
    model.eval()
    with torch.inference_mode():
        model(image)
    for handle in handles:
        handle.remove()
    return captured


def feature_stats(value: torch.Tensor) -> dict:
    value = value.detach().float()
    return {
        "shape": list(value.shape), "mean": value.mean().item(), "std": value.std(unbiased=False).item(),
        "abs_mean": value.abs().mean().item(), "l2": torch.linalg.vector_norm(value).item(),
        "zero_fraction": (value == 0).float().mean().item(),
    }


def compare_features(candidate: dict[str, torch.Tensor], b0: dict[str, torch.Tensor]) -> dict:
    output = {}
    for name in b0:
        reference, value = b0[name].float().flatten(), candidate[name].float().flatten()
        output[name] = {
            "cosine": torch.nn.functional.cosine_similarity(reference, value, dim=0).item(),
            "relative_l2": (torch.linalg.vector_norm(value - reference) / torch.linalg.vector_norm(reference).clamp_min(1e-12)).item(),
        }
    return output


def real_batch_probe(model: DetectionModel, batch: dict, trainer_args, backward: bool = False, mshc_indices: set[int] | None = None) -> dict:
    model.train().zero_grad(set_to_none=True)
    model.args, model.nc, model.names = trainer_args, 4, NAMES
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        total, items = model.loss(batch)
    if backward:
        total.sum().backward()
    gradients = {}
    if mshc_indices:
        for name, parameter in model.named_parameters():
            parsed = split_layer_key(name)
            if parsed and parsed[0] in mshc_indices:
                gradient = parameter.grad
                gradients[name] = {
                    "present": gradient is not None,
                    "finite": gradient is not None and bool(torch.isfinite(gradient).all()),
                    "nonzero": gradient is not None and bool(gradient.detach().abs().sum() > 0),
                }
    return {
        "total": total.detach().float().sum().item(), "box": items[0].item(), "cls": items[1].item(),
        "dfl": items[2].item(), "finite": bool(torch.isfinite(total).all() and torch.isfinite(items).all()),
        "backward": backward,
        "all_selected_gradients_finite_nonzero": all(item["present"] and item["finite"] and item["nonzero"] for item in gradients.values()) if gradients else None,
        "selected_gradient_tensors": len(gradients),
    }


def build_pretrained(yaml_path: str | Path, weights: Path, mode: str) -> tuple[DetectionModel, dict]:
    torch.manual_seed(42)
    model = DetectionModel(str(yaml_path), ch=3, nc=4, verbose=False).float()
    model.args = get_cfg()
    model.args.epochs = 30
    if mode == "m2":
        transfer, matched, mapping = m2_semantic_transfer(model, weights)
        if transfer["unsafe_semantic_candidates"] or transfer["mshc_loaded_items"]:
            raise AssertionError(transfer)
        load_and_verify(model, matched)
        transfer["target_to_source"] = mapping
    else:
        transfer, matched = transfer_audit(model, weights)
        load_matched(model, matched)
    return model, {"transfer": transfer, "matched": matched}


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

    b0, b0_init = build_pretrained(B0_YAML, args.weights, "direct")
    m1, m1_init = build_pretrained(M1_YAML, args.weights, "direct")
    m2, m2_init = build_pretrained(M2_YAML, args.weights, "m2")
    coverage = component_coverage(m2, m2_init["matched"])
    if coverage["p3_original_layer4"]["coverage"] != 1.0:
        raise AssertionError(f"P3 was not fully inherited: {coverage['p3_original_layer4']}")
    rebuild = trainer_rebuild(copy.deepcopy(m2), args.data.resolve(), report_dir)
    if not rebuild["passed"]:
        raise AssertionError(rebuild)

    m2 = m2.to(device).eval()
    sample = torch.zeros(1, 3, args.imgsz, args.imgsz, device=device)
    with torch.inference_mode():
        output, detect_shapes, top_shapes = capture_detect_shapes(m2, sample)
    forward_finite = all(torch.isfinite(value).all() for value in tensors(output))
    amp = amp_detection_step(copy.deepcopy(m2), device, args.imgsz)
    gflops = get_flops(m2, args.imgsz) or get_flops_with_torch_profiler(m2, args.imgsz)
    measured_latency = latency(m2, device, args.imgsz, args.warmup, args.runs)
    onnx = export_onnx(m2, args.imgsz, args.onnx.resolve())

    batch, trainer_args, paths = fixed_train_batch(m2, args.data.resolve(), device, report_dir)
    image = batch["img"]
    b0, m1 = b0.to(device), m1.to(device)
    b0_features = capture_features(b0, image, B0_FEATURES)
    m1_features = capture_features(m1, image, M1_FEATURES)
    m2_features = capture_features(m2, image, M2_FEATURES)
    losses = {
        "b0": real_batch_probe(copy.deepcopy(b0), copy.deepcopy(batch), trainer_args),
        "m1": real_batch_probe(copy.deepcopy(m1), copy.deepcopy(batch), trainer_args),
        "m2": real_batch_probe(copy.deepcopy(m2), copy.deepcopy(batch), trainer_args, backward=True, mshc_indices={6, 7}),
    }

    result = {
        "model": str(M2_YAML.resolve()),
        "scope": {"split_read": "fixed non-augmented train batch only", "test_read": False, "training_started": False},
        "structure": {
            "definition": "original YOLO26 P3 C3k2 plus two consecutive P4 MSHCBlock modules",
            "top_level_modules": [type(module).__name__ for module in m2.model],
            "p3_layer4": type(m2.model[4]).__name__,
            "p4_layers6_7": [type(m2.model[index]).__name__ for index in (6, 7)],
            "identity_residual_added": False,
            "detect_indices": [17, 20, 23],
        },
        "parameters": sum(parameter.numel() for parameter in m2.parameters()),
        "gflops": gflops,
        "pretrained_transfer": m2_init["transfer"],
        "component_coverage": coverage,
        "trainer_rebuild": rebuild,
        "forward": {"passed": True, "finite": forward_finite, "output_shapes": shape_tree(output), "detect_input_shapes_p3_p4_p5": detect_shapes, "top_level_shapes": top_shapes},
        "amp_detection_loss_backward": amp,
        "real_batch": {"images": paths, "losses": losses},
        "features": {
            "b0": {name: feature_stats(value) for name, value in b0_features.items()},
            "m1": {name: feature_stats(value) for name, value in m1_features.items()},
            "m2": {name: feature_stats(value) for name, value in m2_features.items()},
            "m1_vs_b0": compare_features(m1_features, b0_features),
            "m2_vs_b0": compare_features(m2_features, b0_features),
        },
        "latency": measured_latency,
        "peak_vram_mib": amp["peak_allocated_mib"],
        "onnx": onnx,
        "engineering_pass": forward_finite and rebuild["passed"] and losses["m2"]["finite"] and losses["m2"]["all_selected_gradients_finite_nonzero"],
    }
    args.report.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
