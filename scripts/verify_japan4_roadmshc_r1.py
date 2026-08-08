"""Val/Test-free static and initialization audit for Japan4 RoadMSHC-R1."""

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

from audit_japan4_parent_static import amp_detection_step, capture_detect_shapes, export_onnx, latency, shape_tree, tensors  # noqa: E402
from ultralytics.cfg import get_cfg  # noqa: E402
from ultralytics.models.yolo.detect.train import DetectionTrainer  # noqa: E402
from ultralytics.nn.tasks import DetectionModel, torch_safe_load  # noqa: E402
from ultralytics.nn.yolo26_cvpr_improvements.modules import RoadMSHCAdapter  # noqa: E402
from ultralytics.utils.torch_utils import get_flops, get_flops_with_torch_profiler  # noqa: E402
from verify_japan4_mshc_m2 import (  # noqa: E402
    B0_FEATURES,
    B0_YAML,
    M1_FEATURES,
    M1_YAML,
    NAMES,
    build_pretrained,
    capture_features,
    compare_features,
    feature_stats,
    load_and_verify,
    real_batch_probe,
    split_layer_key,
    state_sha256,
)

R1_YAML = ROOT / "ultralytics/cfg/models/26/yolo26-RoadMSHC-R1.yaml"
R1_FEATURES = {"backbone_p3": 4, "backbone_p4": 7, "backbone_p5": 11, "detect_p3": 17, "detect_p4": 20, "detect_p5": 23}


def r1_semantic_transfer(model: DetectionModel, weights: Path):
    checkpoint, _ = torch_safe_load(weights)
    source_model = (checkpoint.get("ema") or checkpoint["model"]).float()
    source, target = source_model.state_dict(), model.state_dict()
    matched, mapping, unsafe, shape_mismatches = {}, {}, [], []
    for source_name, value in source.items():
        parsed = split_layer_key(source_name)
        if parsed is None:
            continue
        source_index, suffix = parsed
        target_index = source_index if source_index <= 6 else source_index + 1
        target_name = f"model.{target_index}.{suffix}"
        if target_name not in target:
            continue
        same_type = type(source_model.model[source_index]) is type(model.model[target_index])
        same_shape = target[target_name].shape == value.shape
        if same_type and same_shape:
            matched[target_name] = value
            mapping[target_name] = source_name
        elif not same_type:
            unsafe.append({"source": source_name, "target": target_name, "source_type": type(source_model.model[source_index]).__name__, "target_type": type(model.model[target_index]).__name__})
        else:
            shape_mismatches.append({"source": source_name, "target": target_name, "source_shape": list(value.shape), "target_shape": list(target[target_name].shape)})
    parameters = dict(model.named_parameters())
    loaded_parameters = sum(parameters[name].numel() for name in matched if name in parameters)
    return {
        "rule": "B0 layers 0..6 retain indices; B0 layers 7..23 map semantically to R1 layers 8..24",
        "matched_state_items": len(matched),
        "target_state_items": len(target),
        "matched_target_parameters": loaded_parameters,
        "target_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "matched_target_parameter_fraction": loaded_parameters / sum(parameter.numel() for parameter in model.parameters()),
        "unsafe_semantic_candidates": unsafe,
        "expected_shape_mismatches": shape_mismatches,
        "adapter_loaded_items": [name for name in matched if split_layer_key(name)[0] == 7],
    }, matched, mapping


def coverage(model: DetectionModel, matched: dict[str, torch.Tensor]) -> dict:
    parameters = dict(model.named_parameters())
    groups = {
        "p3_original": [name for name in parameters if (parsed := split_layer_key(name)) and parsed[0] in {3, 4}],
        "p4_original": [name for name in parameters if (parsed := split_layer_key(name)) and parsed[0] in {5, 6}],
        "roadmshc_adapter": [name for name in parameters if name.startswith("model.7.")],
        "p5_sppf_c2psa": [name for name in parameters if (parsed := split_layer_key(name)) and parsed[0] in {8, 9, 10, 11}],
        "early_backbone": [name for name in parameters if (parsed := split_layer_key(name)) and parsed[0] in {0, 1, 2}],
        "neck": [name for name in parameters if (parsed := split_layer_key(name)) and parsed[0] in set(range(12, 24))],
        "detect_regression": [name for name in parameters if name.startswith(("model.24.cv2.", "model.24.one2one_cv2."))],
        "detect_classification": [name for name in parameters if name.startswith(("model.24.cv3.", "model.24.one2one_cv3."))],
    }
    result = {}
    for group, names in groups.items():
        total = sum(parameters[name].numel() for name in names)
        loaded = sum(parameters[name].numel() for name in names if name in matched)
        result[group] = {"loaded_parameters": loaded, "total_parameters": total, "coverage": loaded / total if total else 0.0, "random_parameter_names": [name for name in names if name not in matched]}
    return result


def trainer_rebuild(model: DetectionModel, data: Path, report_dir: Path) -> dict:
    before = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    trainer = DetectionTrainer(overrides={
        "model": str(R1_YAML), "data": str(data), "epochs": 30, "imgsz": 640, "batch": 8, "workers": 0,
        "device": "0", "seed": 42, "deterministic": True, "optimizer": "auto", "amp": True,
        "project": str(report_dir / "trainer_probe"), "name": "r1_rebuild", "exist_ok": True, "plots": False, "verbose": False,
    })
    rebuilt = trainer.get_model(weights=model, cfg=model.yaml, verbose=False).float()
    after = rebuilt.state_dict()
    mismatched = [name for name, value in before.items() if not torch.equal(value, after[name].cpu())]
    return {"passed": not mismatched, "before_sha256": state_sha256(before), "after_sha256": state_sha256(after), "mismatched": mismatched}


def fixed_train_batch(model: DetectionModel, data: Path, device: torch.device, report_dir: Path):
    trainer = DetectionTrainer(overrides={
        "model": str(R1_YAML), "data": str(data), "epochs": 30, "imgsz": 640, "batch": 8, "workers": 0,
        "device": "0", "seed": 42, "deterministic": True, "optimizer": "auto", "amp": True,
        "augment": False, "rect": True, "project": str(report_dir / "fixed_batch_probe"), "name": "loader",
        "exist_ok": True, "plots": False, "verbose": False,
    })
    trainer.model = model.to(device)
    trainer.set_model_attributes()
    loader = trainer.get_dataloader(trainer.data["train"], batch_size=8, rank=-1, mode="val")
    raw = next(iter(loader))
    return trainer.preprocess_batch(raw), trainer.args, [str(path) for path in raw.get("im_file", [])]


def adapter_real_batch_probe(model: DetectionModel, batch: dict, trainer_args) -> dict:
    model.train().zero_grad(set_to_none=True)
    model.args, model.nc, model.names = trainer_args, 4, NAMES
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        total, items = model.loss(batch)
    total.sum().backward()
    groups = {name: [] for name in ("gamma", "reduce", "square_3", "square_5", "square_7", "horizontal", "vertical", "fuse", "gate")}
    for name, parameter in model.model[7].named_parameters():
        if name == "gamma":
            group = "gamma"
        elif name.startswith("square.0"):
            group = "square_3"
        elif name.startswith("square.1"):
            group = "square_5"
        elif name.startswith("square.2"):
            group = "square_7"
        else:
            group = name.split(".", 1)[0]
        gradient = parameter.grad
        groups[group].append({"name": name, "present": gradient is not None, "finite": gradient is not None and bool(torch.isfinite(gradient).all()), "nonzero": gradient is not None and bool(gradient.detach().abs().sum() > 0)})
    group_pass = {group: bool(items_) and all(item["present"] and item["finite"] and item["nonzero"] for item in items_) for group, items_ in groups.items()}
    return {
        "total": total.detach().float().sum().item(), "box": items[0].item(), "cls": items[1].item(), "dfl": items[2].item(),
        "finite": bool(torch.isfinite(total).all() and torch.isfinite(items).all()), "gradient_group_pass": group_pass,
        "all_adapter_groups_finite_nonzero": all(group_pass.values()),
    }


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

    b0, _ = build_pretrained(B0_YAML, args.weights, "direct")
    m1, _ = build_pretrained(M1_YAML, args.weights, "direct")
    torch.manual_seed(42)
    r1 = DetectionModel(str(R1_YAML), ch=3, nc=4, verbose=False).float()
    r1.args = get_cfg()
    r1.args.epochs = 30
    transfer, matched, mapping = r1_semantic_transfer(r1, args.weights)
    if transfer["unsafe_semantic_candidates"] or transfer["adapter_loaded_items"]:
        raise AssertionError(transfer)
    load_and_verify(r1, matched)
    inherited = coverage(r1, matched)
    required = ("early_backbone", "p3_original", "p4_original", "p5_sppf_c2psa", "neck", "detect_regression")
    if any(inherited[name]["coverage"] != 1.0 for name in required):
        raise AssertionError({name: inherited[name] for name in required})
    adapter = r1.model[7]
    if not isinstance(adapter, RoadMSHCAdapter) or not isinstance(adapter.shortcut, torch.nn.Identity) or adapter.gamma.item() != torch.tensor(0.01).item():
        raise AssertionError({"adapter": type(adapter).__name__, "shortcut": type(adapter.shortcut).__name__, "gamma": adapter.gamma.item()})
    rebuild = trainer_rebuild(copy.deepcopy(r1), args.data.resolve(), report_dir)
    if not rebuild["passed"]:
        raise AssertionError(rebuild)

    r1 = r1.to(device).eval()
    sample = torch.zeros(1, 3, args.imgsz, args.imgsz, device=device)
    with torch.inference_mode():
        output, detect_shapes, top_shapes = capture_detect_shapes(r1, sample)
    forward_finite = all(torch.isfinite(value).all() for value in tensors(output))
    amp = amp_detection_step(copy.deepcopy(r1), device, args.imgsz)
    gflops = get_flops(r1, args.imgsz) or get_flops_with_torch_profiler(r1, args.imgsz)
    measured_latency = latency(r1, device, args.imgsz, args.warmup, args.runs)
    onnx = export_onnx(r1, args.imgsz, args.onnx.resolve())

    batch, trainer_args, paths = fixed_train_batch(r1, args.data.resolve(), device, report_dir)
    image = batch["img"]
    b0, m1 = b0.to(device), m1.to(device)
    b0_features = capture_features(b0, image, B0_FEATURES)
    m1_features = capture_features(m1, image, M1_FEATURES)
    r1_features = capture_features(r1, image, R1_FEATURES)
    losses = {
        "b0": real_batch_probe(copy.deepcopy(b0), copy.deepcopy(batch), trainer_args),
        "m1": real_batch_probe(copy.deepcopy(m1), copy.deepcopy(batch), trainer_args),
        "r1": adapter_real_batch_probe(copy.deepcopy(r1), copy.deepcopy(batch), trainer_args),
    }
    result = {
        "model": str(R1_YAML.resolve()),
        "scope": {"split_read": "fixed non-augmented train batch only", "test_read": False, "training_started": False},
        "structure": {"definition": "pretrained B0 P4 C3k2 followed by RoadMSHCAdapter", "shortcut": type(adapter.shortcut).__name__, "gamma_init": adapter.gamma.item(), "kernels": [[3, 3], [5, 5], [7, 7], [1, 7], [7, 1]], "detect_indices": [17, 20, 23]},
        "parameters": sum(parameter.numel() for parameter in r1.parameters()), "gflops": gflops,
        "pretrained_transfer": transfer, "semantic_target_to_source": mapping, "component_coverage": inherited,
        "trainer_rebuild": rebuild,
        "forward": {"passed": True, "finite": forward_finite, "output_shapes": shape_tree(output), "detect_input_shapes_p3_p4_p5": detect_shapes, "top_level_shapes": top_shapes},
        "amp_detection_loss_backward": amp,
        "real_batch": {"images": paths, "losses": losses},
        "features": {
            "b0": {name: feature_stats(value) for name, value in b0_features.items()},
            "m1": {name: feature_stats(value) for name, value in m1_features.items()},
            "r1": {name: feature_stats(value) for name, value in r1_features.items()},
            "m1_vs_b0": compare_features(m1_features, b0_features), "r1_vs_b0": compare_features(r1_features, b0_features),
        },
        "latency": measured_latency, "peak_vram_mib": amp["peak_allocated_mib"], "onnx": onnx,
        "engineering_pass": forward_finite and rebuild["passed"] and losses["r1"]["finite"] and losses["r1"]["all_adapter_groups_finite_nonzero"],
    }
    args.report.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
