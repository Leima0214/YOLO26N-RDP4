"""Semantic YOLO26 initialization for full-backbone Japan4 parent candidates."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import torch

from ultralytics.nn.tasks import DetectionModel, torch_safe_load


# Explicit source YOLO26 layer -> candidate layer mappings. Backbone layers are
# deliberately absent for full-backbone replacements.
PARENT_LAYER_MAPS = {
    "b0": {index: index for index in range(24)},
    "starnet": {index: index for index in range(9, 24)},
    "mobilemamba": {index: index - 5 for index in range(9, 24)},
}

TARGET_COMPONENTS = {
    "b0": {"backbone": range(0, 9), "tail": range(9, 11), "neck": range(11, 23), "head": (23,)},
    "starnet": {"backbone": range(0, 9), "tail": range(9, 11), "neck": range(11, 23), "head": (23,)},
    "mobilemamba": {"backbone": range(0, 4), "tail": range(4, 6), "neck": range(6, 18), "head": (18,)},
}

BACKBONE_PRETRAIN = {
    "b0": "YOLO26 official checkpoint, exact name and shape",
    "starnet": "none: local variant is not an exact StarNet S1-S4 checkpoint match",
    "mobilemamba": "none: local nano variant is not an exact MobileMamba T2/T4/S6/B1/B2/B4 match",
}


def _split_layer_key(key: str) -> tuple[int, str] | None:
    parts = key.split(".", 2)
    if len(parts) != 3 or parts[0] != "model" or not parts[1].isdigit():
        return None
    return int(parts[1]), parts[2]


def _component_for_layer(candidate: str, layer: int) -> str:
    for component, indices in TARGET_COMPONENTS[candidate].items():
        if layer in indices:
            return component
    raise KeyError(f"Layer {layer} has no component declaration for {candidate}")


def load_yolo26_source(weights: str | Path) -> tuple[torch.nn.Module, dict[str, torch.Tensor]]:
    """Load the official YOLO26 checkpoint state without modifying a candidate."""
    checkpoint, _ = torch_safe_load(Path(weights))
    source_model = (checkpoint.get("ema") or checkpoint["model"]).float()
    return source_model, source_model.state_dict()


def semantic_yolo26_state(
    model: DetectionModel, candidate: str, weights: str | Path
) -> tuple[dict[str, torch.Tensor], dict]:
    """Build an explicit semantic-and-shape matched state dictionary and audit."""
    if candidate not in PARENT_LAYER_MAPS:
        raise KeyError(f"Unsupported candidate: {candidate}")
    source_model, source_state = load_yolo26_source(weights)
    target_state = model.state_dict()
    source_parameters = dict(source_model.named_parameters())
    target_parameters = dict(model.named_parameters())
    layer_map = PARENT_LAYER_MAPS[candidate]
    matched: dict[str, torch.Tensor] = {}
    records = []

    for source_key, value in source_state.items():
        parsed = _split_layer_key(source_key)
        if parsed is None:
            continue
        source_layer, suffix = parsed
        if source_layer not in layer_map:
            continue
        target_layer = layer_map[source_layer]
        target_key = f"model.{target_layer}.{suffix}"
        if target_key not in target_state or target_state[target_key].shape != value.shape:
            continue
        # Semantic equivalence is declared by the layer map; shape equality is
        # an additional necessary check, never a substitute for semantics.
        matched[target_key] = value.detach().clone()
        records.append(
            {
                "source": source_key,
                "target": target_key,
                "shape": list(value.shape),
                "component": _component_for_layer(candidate, target_layer),
            }
        )

    loaded_parameter_names = {key for key in matched if key in target_parameters}
    component_totals = defaultdict(int)
    component_loaded = defaultdict(int)
    for key, parameter in target_parameters.items():
        parsed = _split_layer_key(key)
        if parsed is None:
            continue
        component = _component_for_layer(candidate, parsed[0])
        component_totals[component] += parameter.numel()
        if key in loaded_parameter_names:
            component_loaded[component] += parameter.numel()

    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    loaded_parameters = sum(target_parameters[key].numel() for key in loaded_parameter_names)
    components = {}
    for component in ("backbone", "tail", "neck", "head"):
        total = component_totals[component]
        loaded = component_loaded[component]
        components[component] = {
            "loaded_parameters": loaded,
            "total_parameters": total,
            "coverage": loaded / total if total else 0.0,
        }
    mapped_source_keys = {record["source"] for record in records}
    audit = {
        "candidate": candidate,
        "backbone_pretrained": BACKBONE_PRETRAIN[candidate],
        "semantic_layer_map": {str(source): target for source, target in layer_map.items()},
        "matched_state_items": len(matched),
        "matched_parameter_tensors": len(loaded_parameter_names),
        "loaded_parameters": loaded_parameters,
        "total_parameters": total_parameters,
        "loaded_parameter_fraction": loaded_parameters / total_parameters,
        "random_parameters": total_parameters - loaded_parameters,
        "random_parameter_fraction": 1.0 - loaded_parameters / total_parameters,
        "components": components,
        "records": records,
        "missing_target_parameter_keys": [key for key in target_parameters if key not in loaded_parameter_names],
        "unused_source_parameter_keys": [key for key in source_parameters if key not in mapped_source_keys],
    }
    return matched, audit


def initialize_parent(model: DetectionModel, candidate: str, weights: str | Path) -> dict:
    """Apply semantic initialization and verify every mapped tensor exactly."""
    matched, audit = semantic_yolo26_state(model, candidate, weights)
    incompatible = model.load_state_dict(matched, strict=False)
    state = model.state_dict()
    changed = [key for key, value in matched.items() if not torch.equal(state[key].cpu(), value.cpu())]
    if changed:
        raise AssertionError(f"Semantic tensors changed during load: {changed[:10]}")
    audit["load_verified"] = True
    audit["missing_state_items_after_load"] = list(incompatible.missing_keys)
    audit["unexpected_state_items_after_load"] = list(incompatible.unexpected_keys)
    return audit
