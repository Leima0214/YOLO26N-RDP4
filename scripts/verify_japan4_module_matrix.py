"""Verify the Japan4 road-damage YAML matrix before GPU training."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ultralytics import YOLO
from ultralytics.nn.japan4_adapters import FreqFusionConcat, GatedDySample, OCEC3k2
from ultralytics.nn.modules.head import Detect
from ultralytics.nn.yolo26_cvpr_improvements import MAFConcat, SOMC3k2, WTCC3k2


MODEL_DIR = ROOT / "ultralytics/cfg/models/26"
VARIANTS = {
    "B0": MODEL_DIR / "yolo26.yaml",
    "SOM": MODEL_DIR / "yolo26n-japan4-som.yaml",
    "MAF": MODEL_DIR / "yolo26n-japan4-maf.yaml",
    "WTC": MODEL_DIR / "yolo26n-japan4-wtc.yaml",
    "SOM+MAF": MODEL_DIR / "yolo26n-japan4-som-maf.yaml",
    "SOM+WTC": MODEL_DIR / "yolo26n-japan4-som-wtc.yaml",
    "MAF+WTC": MODEL_DIR / "yolo26n-japan4-maf-wtc.yaml",
    "SOM+MAF+WTC": MODEL_DIR / "yolo26n-japan4-som-maf-wtc.yaml",
    "C2 OCE+MAF": MODEL_DIR / "yolo26n-japan4-c2-oce-maf.yaml",
    "C3 DySample": MODEL_DIR / "yolo26n-japan4-c3-dysample.yaml",
    "C4 FreqFusion": MODEL_DIR / "yolo26n-japan4-c4-freqfusion.yaml",
}

EXPECTED_MODULES = {
    "B0": {},
    "SOM": {"SOM": 2},
    "MAF": {"MAF": 4},
    "WTC": {"WTC": 3},
    "SOM+MAF": {"SOM": 2, "MAF": 4},
    "SOM+WTC": {"SOM": 2, "WTC": 3},
    "MAF+WTC": {"MAF": 4, "WTC": 3},
    "SOM+MAF+WTC": {"SOM": 2, "MAF": 4, "WTC": 3},
    "C2 OCE+MAF": {"OCE": 2, "MAF": 4},
    "C3 DySample": {"DySample": 2},
    "C4 FreqFusion": {"FreqFusion": 1},
}


def tensors(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from tensors(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from tensors(child)


def raw_prediction_tensors(output):
    if isinstance(output, tuple) and len(output) == 2 and isinstance(output[1], dict):
        output = output[1]
    return list(tensors(output))


def transfer_coverage(source: torch.nn.Module, target: torch.nn.Module) -> dict[str, float | int]:
    source_state, target_state = source.state_dict(), target.state_dict()
    matched = {
        name: value
        for name, value in source_state.items()
        if name in target_state and target_state[name].shape == value.shape
    }
    source_parameter_names = dict(source.named_parameters())
    matched_parameters = sum(source_parameter_names[name].numel() for name in matched if name in source_parameter_names)
    total_parameters = sum(parameter.numel() for parameter in source.parameters())
    return {
        "matched_state_items": len(matched),
        "source_state_items": len(source_state),
        "matched_parameters": matched_parameters,
        "source_parameters": total_parameters,
        "parameter_percent": 100.0 * matched_parameters / total_parameters,
    }


def assert_module_counts(name: str, model: torch.nn.Module) -> dict[str, int]:
    placements = {
        "SOM": (SOMC3k2, {4, 6}),
        "MAF": (MAFConcat, {12, 15, 18, 21}),
        "WTC": (WTCC3k2, {16, 19, 22}),
        "OCE": (OCEC3k2, {4, 6}),
        "DySample": (GatedDySample, {11, 14}),
        "FreqFusion": (FreqFusionConcat, {15}),
    }
    counts = {
        family: sum(isinstance(module, module_type) for module in model.modules())
        for family, (module_type, _) in placements.items()
    }
    expected = {family: EXPECTED_MODULES[name].get(family, 0) for family in placements}
    assert counts == expected, (name, counts, expected)
    for family, (module_type, indices) in placements.items():
        actual = {index for index, module in enumerate(model.model) if isinstance(module, module_type)}
        assert actual == (indices if expected[family] else set()), (name, family, actual)
    return counts


def check_new_gradients(name: str, model: torch.nn.Module) -> dict[str, int]:
    prefixes = {
        "SOM": ("som_",),
        "MAF": ("source_logits", "offsets.", "deforms.", "spatial_gates."),
        "WTC": ("wtc_filters.", "wtc_band_logits"),
        "OCE": ("oce_",),
        "DySample": ("dysample.", "dysample_scale"),
        "FreqFusion": ("freq_",),
    }
    result = {}
    for family, markers in prefixes.items():
        selected = [
            parameter
            for parameter_name, parameter in model.named_parameters()
            if any(marker in parameter_name for marker in markers)
        ]
        if EXPECTED_MODULES[name].get(family, 0):
            assert selected, (name, family, "no parameters")
            active = sum(
                parameter.grad is not None
                and torch.isfinite(parameter.grad).all()
                and bool(parameter.grad.abs().sum() > 0)
                for parameter in selected
            )
            assert active, (name, family, "no active gradient")
            result[family] = active
        else:
            assert not selected, (name, family, "unexpected parameters")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, default=ROOT / "yolo26n.pt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--imgsz", type=int, default=128)
    parser.add_argument("--report", type=Path, default=ROOT / "reports/japan4_module_matrix_verification.json")
    args = parser.parse_args()
    assert args.weights.is_file(), args.weights
    assert args.imgsz >= 64 and args.imgsz % 32 == 0

    torch.manual_seed(42)
    device = torch.device(args.device)
    checkpoint = YOLO(str(args.weights), task="detect", verbose=False).model.float().to(device).eval()
    sample = torch.randn(1, 3, args.imgsz, args.imgsz, device=device)
    with torch.no_grad():
        reference_tensors = [tensor.detach().cpu() for tensor in raw_prediction_tensors(checkpoint(sample))]
    results = {}

    for name, yaml_path in VARIANTS.items():
        assert yaml_path.is_file(), yaml_path
        wrapper = YOLO(str(yaml_path), task="detect", verbose=False)
        model = wrapper.model.float()
        coverage = transfer_coverage(checkpoint, model)
        assert coverage["parameter_percent"] == 100.0, (name, coverage)
        model.load(checkpoint, verbose=False)
        model.to(device)
        detect = model.model[-1]
        assert isinstance(detect, Detect)
        assert detect.end2end and detect.reg_max == 1
        assert model.stride.tolist() == [8.0, 16.0, 32.0]
        assert len(model.model) == 24
        module_counts = assert_module_counts(name, model)

        model.eval()
        with torch.no_grad():
            candidate_tensors = [tensor.detach().cpu() for tensor in raw_prediction_tensors(model(sample))]
        assert len(candidate_tensors) == len(reference_tensors)
        differences = [candidate - reference for candidate, reference in zip(candidate_tensors, reference_tensors)]
        max_initial_difference = max(difference.abs().max().item() for difference in differences)
        squared_error = sum(difference.double().square().sum() for difference in differences)
        reference_energy = sum(reference.double().square().sum() for reference in reference_tensors)
        relative_initial_l2 = float((squared_error / reference_energy).sqrt())
        relative_limit = 3e-3 if EXPECTED_MODULES[name].keys() & {"SOM", "OCE", "DySample", "FreqFusion"} else 1e-6
        assert relative_initial_l2 <= relative_limit, (name, relative_initial_l2, relative_limit)

        model.train()
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            output = model(sample)
            raw_tensors = list(tensors(output))
            assert raw_tensors and all(torch.isfinite(tensor).all() for tensor in raw_tensors)
            loss = sum(tensor.float().square().mean() for tensor in raw_tensors)
        loss.backward()
        gradients = check_new_gradients(name, model)
        parameters = sum(parameter.numel() for parameter in model.parameters())
        results[name] = {
            "yaml": str(yaml_path.relative_to(ROOT)),
            "parameters": parameters,
            "transfer": coverage,
            "modules": module_counts,
            "active_new_gradient_tensors": gradients,
            "strides": model.stride.tolist(),
            "end2end": detect.end2end,
            "reg_max": detect.reg_max,
            "max_initial_difference_from_pretrained_b0": max_initial_difference,
            "relative_initial_l2_from_pretrained_b0": relative_initial_l2,
        }
        print(
            f"{name:15s} PASS params={parameters:,} "
            f"transfer={coverage['parameter_percent']:.6f}% "
            f"relative_init_l2={relative_initial_l2:.3g} gradients={gradients}"
        )
        del output, raw_tensors, candidate_tensors, loss, wrapper, model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"REPORT {args.report}")


if __name__ == "__main__":
    main()
