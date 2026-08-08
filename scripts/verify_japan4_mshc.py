"""Static, semantic-transfer, and RTX audit for yolo26-MSHC on Japan4-cleanV3."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from collections import defaultdict
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

MODEL_YAML = ROOT / "ultralytics/cfg/models/26/yolo26-MSHC.yaml"


def state_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def layer_index(name: str) -> int | None:
    parts = name.split(".", 2)
    return int(parts[1]) if len(parts) == 3 and parts[0] == "model" and parts[1].isdigit() else None


def parameter_groups(model: DetectionModel) -> dict[str, list[str]]:
    names = list(dict(model.named_parameters()))
    return {
        "unchanged_backbone": [name for name in names if layer_index(name) in {0, 1, 2, 3, 5, 7, 8, 9, 10}],
        "mshc_p3_layer4": [name for name in names if layer_index(name) == 4],
        "mshc_p4_layer6": [name for name in names if layer_index(name) == 6],
        "neck": [name for name in names if layer_index(name) in set(range(11, 23))],
        "detect_regression": [name for name in names if name.startswith(("model.23.cv2.", "model.23.one2one_cv2."))],
        "detect_classification": [name for name in names if name.startswith(("model.23.cv3.", "model.23.one2one_cv3."))],
        "detect_other": [
            name
            for name in names
            if layer_index(name) == 23
            and not name.startswith(("model.23.cv2.", "model.23.one2one_cv2.", "model.23.cv3.", "model.23.one2one_cv3."))
        ],
    }


def coverage(model: DetectionModel, matched: dict[str, torch.Tensor]) -> dict:
    parameters = dict(model.named_parameters())
    loaded = {name for name in matched if name in parameters}
    result = {}
    for group, names in parameter_groups(model).items():
        total = sum(parameters[name].numel() for name in names)
        inherited = sum(parameters[name].numel() for name in names if name in loaded)
        result[group] = {
            "loaded_parameters": inherited,
            "total_parameters": total,
            "coverage": inherited / total if total else 0.0,
            "random_parameter_names": [name for name in names if name not in loaded],
        }
    return result


def semantic_safety(model: DetectionModel, weights: Path, matched: dict[str, torch.Tensor]) -> dict:
    package, _ = torch_safe_load(weights)
    source_model = (package.get("ema") or package["model"]).float()
    unsafe = []
    records = []
    for name, value in matched.items():
        index = layer_index(name)
        if index is None:
            continue
        source_type = type(source_model.model[index]).__name__
        target_type = type(model.model[index]).__name__
        safe = source_type == target_type and index not in {4, 6}
        records.append({"name": name, "layer": index, "source_type": source_type, "target_type": target_type, "shape": list(value.shape), "safe": safe})
        if not safe:
            unsafe.append(records[-1])
    mshc_loaded = [name for name in matched if layer_index(name) in {4, 6}]
    unchanged_type_mismatches = []
    for index in set(range(24)) - {4, 6}:
        source_type = type(source_model.model[index]).__name__
        target_type = type(model.model[index]).__name__
        if source_type != target_type:
            unchanged_type_mismatches.append({"layer": index, "source": source_type, "target": target_type})
    return {
        "rule": "same top-level layer index, same module type, same parameter name, same shape",
        "safe_matched_items": len(records) - len(unsafe),
        "unsafe_matched_items": unsafe,
        "mshc_parameters_accidentally_loaded": mshc_loaded,
        "unchanged_layer_type_mismatches": unchanged_type_mismatches,
        "passed": not unsafe and not mshc_loaded and not unchanged_type_mismatches,
    }


def trainer_rebuild(model: DetectionModel, data: Path, report_dir: Path) -> dict:
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
        "name": "mshc_rebuild",
        "exist_ok": True,
        "plots": False,
        "verbose": False,
    }
    trainer = DetectionTrainer(overrides=overrides)
    rebuilt = trainer.get_model(weights=model, cfg=model.yaml, verbose=False).float()
    after = rebuilt.state_dict()
    mismatched = [name for name, value in before.items() if not torch.equal(value, after[name].cpu())]
    return {
        "passed": not mismatched,
        "before_sha256": state_sha256(before),
        "after_sha256": state_sha256(after),
        "mismatched": mismatched,
        "transferred_state_items": len(after) - len(mismatched),
    }


def branch_audit(model: DetectionModel, device: torch.device, imgsz: int) -> dict:
    blocks = {"P3_layer4": model.model[4], "P4_layer6": model.model[6]}
    if not all(isinstance(block, MSHCBlock) for block in blocks.values()):
        raise TypeError({name: type(block).__name__ for name, block in blocks.items()})
    captured = defaultdict(list)
    handles = []
    for block_name, block in blocks.items():
        modules = {
            "proj": block.proj,
            "reduce": block.reduce,
            **{f"square_k{branch.conv.kernel_size[0]}": branch for branch in block.square},
            "horizontal_1x7": block.horizontal,
            "vertical_7x1": block.vertical,
            "fuse": block.fuse,
            "gate": block.gate,
        }
        for branch_name, module in modules.items():
            def hook(_module, _inputs, output, key=f"{block_name}.{branch_name}"):
                captured[key].append(list(output.shape))
            handles.append(module.register_forward_hook(hook))
    model.train().zero_grad(set_to_none=True)
    image = torch.rand(2, 3, imgsz, imgsz, device=device)
    batch = {
        "batch_idx": torch.tensor([0.0, 1.0], device=device),
        "cls": torch.tensor([[0.0], [2.0]], device=device),
        "bboxes": torch.tensor([[0.35, 0.45, 0.28, 0.07], [0.68, 0.62, 0.18, 0.22]], device=device),
    }
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        predictions = model(image)
        loss, loss_items = model.init_criterion()(predictions, batch)
    loss.sum().backward()
    for handle in handles:
        handle.remove()
    gradients = {}
    for name, parameter in model.named_parameters():
        if layer_index(name) in {4, 6}:
            gradient = parameter.grad
            gradients[name] = {
                "numel": parameter.numel(),
                "present": gradient is not None,
                "finite": gradient is not None and bool(torch.isfinite(gradient).all()),
                "nonzero": gradient is not None and bool(gradient.detach().abs().sum() > 0),
                "l2": None if gradient is None else torch.linalg.vector_norm(gradient.detach().float()).item(),
            }
    return {
        "branch_output_shapes": dict(captured),
        "loss": loss.detach().float().sum().item(),
        "loss_items": loss_items.detach().float().tolist(),
        "parameter_gradients": gradients,
        "all_mshc_parameters_finite_nonzero": all(item["present"] and item["finite"] and item["nonzero"] for item in gradients.values()),
        "block_details": {
            name: {
                "input_channels": block.proj.conv.in_channels,
                "output_channels": block.proj.conv.out_channels,
                "hidden_channels": block.reduce.conv.out_channels,
                "square_kernels": [list(branch.conv.kernel_size) for branch in block.square],
                "horizontal_kernel": list(block.horizontal.kernel_size),
                "vertical_kernel": list(block.vertical.kernel_size),
                "gate": "global-average-pool -> 1x1 full-channel conv -> sigmoid",
                "residual": "proj(x) + fuse(branches) * gate(fuse(branches))",
            }
            for name, block in blocks.items()
        },
    }


def feature_stats(value: torch.Tensor) -> dict:
    value = value.detach().float()
    return {
        "shape": list(value.shape),
        "mean": value.mean().item(),
        "std": value.std(unbiased=False).item(),
        "abs_mean": value.abs().mean().item(),
        "l2": torch.linalg.vector_norm(value).item(),
        "zero_fraction": (value == 0).float().mean().item(),
    }


def fixed_train_batch(model: DetectionModel, data: Path, device: torch.device, report_dir: Path) -> tuple[dict, object, list[str]]:
    trainer = DetectionTrainer(
        overrides={
            "model": str(MODEL_YAML),
            "data": str(data),
            "epochs": 30,
            "imgsz": 640,
            "batch": 8,
            "workers": 0,
            "device": "0",
            "seed": 42,
            "deterministic": True,
            "optimizer": "auto",
            "amp": True,
            "augment": False,
            "rect": True,
            "project": str(report_dir / "fixed_batch_probe"),
            "name": "loader",
            "exist_ok": True,
            "plots": False,
            "verbose": False,
        }
    )
    trainer.model = model.to(device)
    trainer.set_model_attributes()
    loader = trainer.get_dataloader(trainer.data["train"], batch_size=8, rank=-1, mode="val")
    raw = next(iter(loader))
    paths = [str(path) for path in raw.get("im_file", [])]
    return trainer.preprocess_batch(raw), trainer.args, paths


def capture_features(model: DetectionModel, image: torch.Tensor) -> dict[str, torch.Tensor]:
    captured = {}
    handles = []
    for name, index in {"backbone_p3": 4, "backbone_p4": 6, "backbone_p5": 10, "detect_p3": 16, "detect_p4": 19, "detect_p5": 22}.items():
        def hook(_module, _inputs, output, key=name):
            captured[key] = output.detach()
        handles.append(model.model[index].register_forward_hook(hook))
    model.eval()
    with torch.inference_mode():
        model(image)
    for handle in handles:
        handle.remove()
    return captured


def raw_classification(model: DetectionModel, image: torch.Tensor) -> dict:
    model.eval()
    model.model[-1].train()
    with torch.inference_mode():
        predictions = model(image)
    model.model[-1].eval()
    result = {}
    for branch in ("one2one", "one2many"):
        logits = predictions[branch]["scores"].float()
        confidence = logits.sigmoid()
        result[branch] = {
            "logits_mean": logits.mean().item(),
            "logits_std": logits.std(unbiased=False).item(),
            "confidence_mean": confidence.mean().item(),
            "confidence_max": confidence.max().item(),
            "locations_above_0.001_per_image": (confidence.amax(1) > 0.001).sum(1).float().tolist(),
        }
    return result


def real_batch_loss(model: DetectionModel, batch: dict, trainer_args) -> dict:
    model = model.train()
    model.args = trainer_args
    model.nc = 4
    model.names = {0: "D00", 1: "D10", 2: "D20", 3: "D40"}
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        total, items = model.loss(batch)
    return {"total": total.detach().float().sum().item(), "box": items[0].item(), "cls": items[1].item(), "dfl": items[2].item(), "finite": bool(torch.isfinite(total).all() and torch.isfinite(items).all())}


def initial_interface_audit(mshc: DetectionModel, weights: Path, data: Path, device: torch.device, report_dir: Path) -> dict:
    torch.manual_seed(42)
    b0 = DetectionModel("yolo26n.yaml", ch=3, nc=4, verbose=False).float()
    b0_transfer, b0_matched = transfer_audit(b0, weights)
    load_matched(b0, b0_matched)
    b0 = b0.to(device)
    batch, trainer_args, paths = fixed_train_batch(mshc, data, device, report_dir)
    image = batch["img"]
    b0_features = capture_features(b0, image)
    mshc_features = capture_features(mshc, image)
    comparisons = {}
    for name in b0_features:
        a = b0_features[name].float().flatten()
        b = mshc_features[name].float().flatten()
        comparisons[name] = {
            "cosine": torch.nn.functional.cosine_similarity(a, b, dim=0).item(),
            "relative_l2": (torch.linalg.vector_norm(b - a) / torch.linalg.vector_norm(a).clamp_min(1e-12)).item(),
        }
    return {
        "fixed_train_images": paths,
        "b0_pretrained_transfer_fraction": b0_transfer["matched_target_parameter_fraction"],
        "b0_features": {name: feature_stats(value) for name, value in b0_features.items()},
        "mshc_features": {name: feature_stats(value) for name, value in mshc_features.items()},
        "mshc_vs_b0": comparisons,
        "b0_loss": real_batch_loss(copy.deepcopy(b0), copy.deepcopy(batch), trainer_args),
        "mshc_loss": real_batch_loss(copy.deepcopy(mshc), copy.deepcopy(batch), trainer_args),
        "b0_classification": raw_classification(b0, image),
        "mshc_classification": raw_classification(mshc, image),
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
    device = torch.device(args.device)
    report_dir = args.report.parent
    report_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(42)
    model = DetectionModel(str(MODEL_YAML), ch=3, nc=4, verbose=False).float()
    model.args = get_cfg()
    model.args.epochs = 30
    transfer, matched = transfer_audit(model, args.weights)
    safety = semantic_safety(model, args.weights, matched)
    if not safety["passed"]:
        raise AssertionError(safety)
    component_coverage = coverage(model, matched)
    load_matched(model, matched)
    rebuild = trainer_rebuild(copy.deepcopy(model), args.data.resolve(), report_dir)
    if not rebuild["passed"]:
        raise AssertionError(rebuild)

    model = model.to(device).eval()
    sample = torch.zeros(1, 3, args.imgsz, args.imgsz, device=device)
    with torch.inference_mode():
        output, detect_shapes, top_level_shapes = capture_detect_shapes(model, sample)
    forward_finite = all(torch.isfinite(value).all() for value in tensors(output))
    if not forward_finite:
        raise FloatingPointError("non-finite forward output")
    gflops = get_flops(model, args.imgsz) or get_flops_with_torch_profiler(model, args.imgsz)
    branch = branch_audit(copy.deepcopy(model), device, args.imgsz)
    amp = amp_detection_step(copy.deepcopy(model), device, args.imgsz)
    measured_latency = latency(model, device, args.imgsz, args.warmup, args.runs)
    onnx = export_onnx(model, args.imgsz, args.onnx.resolve())
    interface = initial_interface_audit(model, args.weights, args.data.resolve(), device, report_dir)

    result = {
        "model": str(MODEL_YAML.resolve()),
        "scope": {"split_read": "fixed non-augmented train batch", "test_read": False, "training_started": False},
        "structure": {
            "top_level_modules": [type(module).__name__ for module in model.model],
            "replaced_layers": {"4": "B0 C3k2 -> MSHCBlock at P3", "6": "B0 C3k2 -> MSHCBlock at P4"},
            "unchanged_layers": [index for index in range(24) if index not in {4, 6}],
            "mshc": branch["block_details"],
        },
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "gflops": gflops,
        "pretrained_transfer": transfer,
        "semantic_transfer_safety": safety,
        "component_coverage": component_coverage,
        "trainer_rebuild": rebuild,
        "forward": {
            "passed": True,
            "finite": forward_finite,
            "output_shapes": shape_tree(output),
            "detect_input_shapes_p3_p4_p5": detect_shapes,
            "top_level_shapes": top_level_shapes,
        },
        "mshc_branch_audit": branch,
        "amp_detection_loss_backward": amp,
        "latency": measured_latency,
        "peak_vram_mib": amp["peak_allocated_mib"],
        "onnx": onnx,
        "initial_pretrained_interface": interface,
        "static_go": safety["passed"] and rebuild["passed"] and forward_finite and branch["all_mshc_parameters_finite_nonzero"],
    }
    args.report.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
