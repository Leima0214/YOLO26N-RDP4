"""Val/Test-free static audit for RoadAdaptiveScaleFusion-N1."""

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
from ultralytics.nn.yolo26_cvpr_improvements import RoadAdaptiveScaleFusionN1  # noqa: E402
from ultralytics.utils.torch_utils import get_flops, get_flops_with_torch_profiler  # noqa: E402

N1_YAML = ROOT / "ultralytics/cfg/models/26/yolo26-RoadAdaptiveScaleFusion-N1.yaml"
B0_YAML = "yolo26n.yaml"
NAMES = {0: "D00", 1: "D10", 2: "D20", 3: "D40"}
B0_FEATURES = {"backbone_p3": 4, "backbone_p4": 6, "backbone_p5": 10, "detect_p3": 16, "detect_p4": 19, "detect_p5": 22}
N1_FEATURES = {"backbone_p3": 4, "backbone_p4": 6, "backbone_p5": 10, "detect_p3": 16, "detect_p4": 23, "detect_p5": 22}


def split_layer_key(name: str) -> tuple[int, str] | None:
    parts = name.split(".", 2)
    return (int(parts[1]), parts[2]) if len(parts) == 3 and parts[0] == "model" and parts[1].isdigit() else None


def state_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def n1_semantic_transfer(model: DetectionModel, weights: Path) -> tuple[dict, dict[str, torch.Tensor]]:
    """Map unchanged B0 layers 0..22 directly and Detect 23 to N1 Detect 24."""
    checkpoint, _ = torch_safe_load(weights)
    source_model = (checkpoint.get("ema") or checkpoint["model"]).float()
    source, target = source_model.state_dict(), model.state_dict()
    matched, mismatches = {}, []
    for source_name, value in source.items():
        parsed = split_layer_key(source_name)
        if parsed is None:
            continue
        source_index, suffix = parsed
        target_index = source_index if source_index <= 22 else 24 if source_index == 23 else None
        if target_index is None:
            continue
        target_name = f"model.{target_index}.{suffix}"
        if target_name not in target:
            continue
        same_type = type(source_model.model[source_index]) is type(model.model[target_index])
        same_shape = target[target_name].shape == value.shape
        if same_type and same_shape:
            matched[target_name] = value
        else:
            mismatches.append({
                "source": source_name, "target": target_name,
                "source_type": type(source_model.model[source_index]).__name__,
                "target_type": type(model.model[target_index]).__name__,
                "source_shape": list(value.shape), "target_shape": list(target[target_name].shape),
            })
    parameters = dict(model.named_parameters())
    loaded_parameters = sum(parameters[name].numel() for name in matched if name in parameters)
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    adapter_loaded = [name for name in matched if name.startswith("model.23.")]
    return {
        "rule": "same semantic module, suffix and shape; B0 0..22 unchanged; B0 Detect 23 maps to N1 Detect 24",
        "matched_state_items": len(matched), "target_state_items": len(target),
        "matched_target_parameters": loaded_parameters, "target_parameters": total_parameters,
        "matched_target_parameter_fraction": loaded_parameters / total_parameters,
        "expected_incompatible_items": mismatches, "adapter_loaded_items": adapter_loaded,
    }, matched


def component_coverage(model: DetectionModel, matched: dict[str, torch.Tensor]) -> dict:
    parameters = dict(model.named_parameters())
    groups = {
        "backbone_0_10": [n for n in parameters if (p := split_layer_key(n)) and p[0] <= 10],
        "neck_11_22": [n for n in parameters if (p := split_layer_key(n)) and 11 <= p[0] <= 22],
        "adapter_23": [n for n in parameters if n.startswith("model.23.")],
        "detect_regression": [n for n in parameters if n.startswith(("model.24.cv2.", "model.24.one2one_cv2."))],
        "detect_classification": [n for n in parameters if n.startswith(("model.24.cv3.", "model.24.one2one_cv3."))],
    }
    result = {}
    for group, names in groups.items():
        total = sum(parameters[name].numel() for name in names)
        loaded = sum(parameters[name].numel() for name in names if name in matched)
        result[group] = {"loaded_parameters": loaded, "total_parameters": total, "coverage": loaded / total if total else 0.0,
                         "random_parameter_names": [name for name in names if name not in matched]}
    return result


def load_and_verify(model: DetectionModel, matched: dict[str, torch.Tensor]) -> None:
    model.load_state_dict(matched, strict=False)
    actual = model.state_dict()
    changed = [name for name, value in matched.items() if not torch.equal(actual[name].cpu(), value.cpu())]
    if changed:
        raise AssertionError(f"Transferred tensors changed: {changed[:10]}")


def trainer_rebuild(model: DetectionModel, data: Path, report_dir: Path) -> dict:
    before = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    trainer = DetectionTrainer(overrides={
        "model": str(N1_YAML), "data": str(data), "epochs": 30, "imgsz": 640, "batch": 8,
        "workers": 0, "device": "0", "seed": 42, "deterministic": True, "optimizer": "auto", "amp": True,
        "project": str(report_dir / "trainer_probe"), "name": "n1_rebuild", "exist_ok": True, "plots": False, "verbose": False,
    })
    rebuilt = trainer.get_model(weights=model, cfg=model.yaml, verbose=False).float()
    after = rebuilt.state_dict()
    mismatched = [name for name, value in before.items() if not torch.equal(value, after[name].cpu())]
    return {"passed": not mismatched, "before_sha256": state_sha256(before), "after_sha256": state_sha256(after), "mismatched": mismatched}


def fixed_train_batch(model: DetectionModel, data: Path, device: torch.device, report_dir: Path):
    trainer = DetectionTrainer(overrides={
        "model": str(N1_YAML), "data": str(data), "epochs": 30, "imgsz": 640, "batch": 8,
        "workers": 0, "device": "0", "seed": 42, "deterministic": True, "optimizer": "auto", "amp": True,
        "augment": False, "rect": True, "project": str(report_dir / "fixed_batch_probe"), "name": "loader",
        "exist_ok": True, "plots": False, "verbose": False,
    })
    trainer.model = model.to(device)
    trainer.set_model_attributes()
    loader = trainer.get_dataloader(trainer.data["train"], batch_size=8, rank=-1, mode="val")
    raw = next(iter(loader))
    return trainer.preprocess_batch(raw), trainer.args, [str(path) for path in raw.get("im_file", [])]


def capture_features(model: DetectionModel, image: torch.Tensor, indices: dict[str, int]) -> dict[str, torch.Tensor]:
    captured, handles = {}, []
    for name, index in indices.items():
        handles.append(model.model[index].register_forward_hook(lambda _m, _i, output, key=name: captured.__setitem__(key, output.detach())))
    model.eval()
    with torch.inference_mode():
        model(image)
    for handle in handles:
        handle.remove()
    return captured


def feature_stats(value: torch.Tensor) -> dict:
    value = value.detach().float()
    return {"shape": list(value.shape), "mean": value.mean().item(), "std": value.std(unbiased=False).item(),
            "abs_mean": value.abs().mean().item(), "l2": torch.linalg.vector_norm(value).item(),
            "zero_fraction": (value == 0).float().mean().item()}


def compare_features(candidate: dict[str, torch.Tensor], reference: dict[str, torch.Tensor]) -> dict:
    result = {}
    for name in reference:
        a, b = reference[name].float().flatten(), candidate[name].float().flatten()
        result[name] = {"cosine": torch.nn.functional.cosine_similarity(a, b, dim=0).item(),
                        "relative_l2": (torch.linalg.vector_norm(b - a) / torch.linalg.vector_norm(a).clamp_min(1e-12)).item()}
    return result


def real_batch_probe(model: DetectionModel, batch: dict, trainer_args, backward: bool = False) -> dict:
    model.train().zero_grad(set_to_none=True)
    model.args, model.nc, model.names = trainer_args, 4, NAMES
    adapter = model.model[23]
    captured_inputs: list[torch.Tensor] = []
    handle = None
    if backward:
        def retain_inputs(_module, args):
            captured_inputs[:] = list(args[0])
            for value in captured_inputs:
                value.retain_grad()
        handle = adapter.register_forward_pre_hook(retain_inputs)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        total, items = model.loss(batch)
    if backward:
        total.sum().backward()
    if handle is not None:
        handle.remove()
    selected = {}
    if backward:
        for name, parameter in adapter.named_parameters():
            gradient = parameter.grad
            selected[name] = {"present": gradient is not None, "finite": gradient is not None and bool(torch.isfinite(gradient).all()),
                              "nonzero": gradient is not None and bool(gradient.detach().abs().sum() > 0),
                              "l2": torch.linalg.vector_norm(gradient.detach().float()).item() if gradient is not None else None}
    input_gradients = []
    for value in captured_inputs:
        gradient = value.grad
        input_gradients.append({"present": gradient is not None, "finite": gradient is not None and bool(torch.isfinite(gradient).all()),
                                "nonzero": gradient is not None and bool(gradient.detach().abs().sum() > 0),
                                "l2": torch.linalg.vector_norm(gradient.detach().float()).item() if gradient is not None else None})
    all_gradients_ok = all(x["present"] and x["finite"] and x["nonzero"] for x in selected.values()) and all(
        x["present"] and x["finite"] and x["nonzero"] for x in input_gradients
    ) if backward else None
    return {"total": total.detach().float().sum().item(), "box": items[0].item(), "cls": items[1].item(), "dfl": items[2].item(),
            "finite": bool(torch.isfinite(total).all() and torch.isfinite(items).all()), "backward": backward,
            "all_adapter_and_input_gradients_finite_nonzero": all_gradients_ok, "adapter_gradients": selected,
            "input_gradients_p3_p4_p5": input_gradients}


def gate_probe(model: DetectionModel, image: torch.Tensor) -> dict:
    adapter: RoadAdaptiveScaleFusionN1 = model.model[23]
    captured = {}
    handle = adapter.scale_logits.register_forward_hook(lambda _m, _i, output: captured.__setitem__("logits", output.detach()))
    model.eval()
    with torch.inference_mode():
        model(image)
    handle.remove()
    weights = captured["logits"].float().softmax(1)
    return {"gamma": adapter.gamma.item(), "alpha_mean_p3_p4_p5": weights.mean((0, 2, 3)).tolist(),
            "alpha_std_p3_p4_p5": weights.std((0, 2, 3), unbiased=False).tolist(),
            "entropy_mean": (-(weights * weights.clamp_min(1e-12).log()).sum(1)).mean().item()}


def build_pretrained(yaml_path: str | Path, weights: Path, mode: str) -> tuple[DetectionModel, dict]:
    torch.manual_seed(42)
    model = DetectionModel(str(yaml_path), ch=3, nc=4, verbose=False).float()
    model.args = get_cfg()
    model.args.epochs = 30
    if mode == "n1":
        transfer, matched = n1_semantic_transfer(model, weights)
        load_and_verify(model, matched)
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
    n1, n1_init = build_pretrained(N1_YAML, args.weights, "n1")
    coverage = component_coverage(n1, n1_init["matched"])
    if coverage["backbone_0_10"]["coverage"] != 1.0 or coverage["neck_11_22"]["coverage"] != 1.0:
        raise AssertionError(f"B0 main path was not fully inherited: {coverage}")
    if n1_init["transfer"]["adapter_loaded_items"]:
        raise AssertionError("New adapter unexpectedly received B0 weights")
    rebuild = trainer_rebuild(copy.deepcopy(n1), args.data.resolve(), report_dir)
    if not rebuild["passed"]:
        raise AssertionError(rebuild)

    n1 = n1.to(device).eval()
    sample = torch.zeros(1, 3, args.imgsz, args.imgsz, device=device)
    with torch.inference_mode():
        output, detect_shapes, top_shapes = capture_detect_shapes(n1, sample)
    forward_finite = all(torch.isfinite(value).all() for value in tensors(output))
    amp = amp_detection_step(copy.deepcopy(n1), device, args.imgsz)
    gflops = get_flops(n1, args.imgsz) or get_flops_with_torch_profiler(n1, args.imgsz)
    measured_latency = latency(n1, device, args.imgsz, args.warmup, args.runs)
    onnx = export_onnx(n1, args.imgsz, args.onnx.resolve())

    batch, trainer_args, paths = fixed_train_batch(n1, args.data.resolve(), device, report_dir)
    b0 = b0.to(device)
    b0_features = capture_features(b0, batch["img"], B0_FEATURES)
    n1_features = capture_features(n1, batch["img"], N1_FEATURES)
    losses = {"b0": real_batch_probe(copy.deepcopy(b0), copy.deepcopy(batch), trainer_args),
              "n1": real_batch_probe(copy.deepcopy(n1), copy.deepcopy(batch), trainer_args, backward=True)}
    gates = gate_probe(n1, batch["img"])

    result = {
        "model": str(N1_YAML.resolve()),
        "scope": {"split_read": "fixed non-augmented train batch only", "test_read": False, "training_started": False},
        "structure": {"definition": "unchanged B0 layers 0..22; one P4-centered adapter at 23; Detect at 24 reads [16,23,22]",
                      "adapter_inputs": [16, 19, 22], "detect_inputs": [16, 23, 22],
                      "top_level_modules": [type(module).__name__ for module in n1.model]},
        "parameters": sum(parameter.numel() for parameter in n1.parameters()), "gflops": gflops,
        "pretrained_transfer": n1_init["transfer"], "component_coverage": coverage, "trainer_rebuild": rebuild,
        "forward": {"passed": True, "finite": forward_finite, "output_shapes": shape_tree(output),
                    "detect_input_shapes_p3_p4_p5": detect_shapes, "top_level_shapes": top_shapes},
        "amp_detection_loss_backward": amp,
        "real_batch": {"images": paths, "losses": losses},
        "features": {"b0": {name: feature_stats(value) for name, value in b0_features.items()},
                     "n1": {name: feature_stats(value) for name, value in n1_features.items()},
                     "n1_vs_b0": compare_features(n1_features, b0_features)},
        "gate_initialization": gates, "latency": measured_latency, "peak_vram_mib": amp["peak_allocated_mib"], "onnx": onnx,
        "engineering_pass": forward_finite and rebuild["passed"] and losses["n1"]["finite"]
                            and losses["n1"]["all_adapter_and_input_gradients_finite_nonzero"] and bool(onnx.get("passed", False)),
    }
    args.report.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
