"""Semantic YOLO26 initialization for full-backbone Japan4 parent candidates."""

from __future__ import annotations

from collections import defaultdict
import hashlib
from pathlib import Path

import torch

from ultralytics.nn.tasks import DetectionModel, torch_safe_load
from ultralytics.nn.official_starnet import OfficialStarNetS1BackboneYOLO

OFFICIAL_STARNET_S1_SHA256 = "4d00fd7acab420dfe93d443222ebf5e89fac5caa4c6d43ebf423fcd8957ed23b"


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


def file_sha256(path: str | Path) -> str:
    """Return a streaming SHA256 digest."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def state_sha256(state: dict[str, torch.Tensor], names: list[str] | None = None) -> str:
    """Hash named tensors including names, dtypes, shapes, and raw values."""
    digest = hashlib.sha256()
    selected = sorted(names if names is not None else state)
    for name in selected:
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


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


def _parameter_coverage(
    parameters: dict[str, torch.nn.Parameter], names: list[str], loaded: set[str]
) -> dict[str, int | float]:
    total = sum(parameters[name].numel() for name in names)
    loaded_count = sum(parameters[name].numel() for name in names if name in loaded)
    return {
        "loaded_parameters": loaded_count,
        "total_parameters": total,
        "coverage": loaded_count / total if total else 0.0,
    }


def initialize_official_starnet_s1_yolo26(
    model: DetectionModel,
    star_checkpoint: str | Path,
    yolo_weights: str | Path,
) -> dict:
    """Initialize the official S1 backbone and retained YOLO26 modules from separate sources."""
    backbone = model.model[0]
    if not isinstance(backbone, OfficialStarNetS1BackboneYOLO):
        raise TypeError(f"Expected OfficialStarNetS1BackboneYOLO at model.0, got {type(backbone).__name__}")
    star_checkpoint = Path(star_checkpoint)
    checkpoint_hash = file_sha256(star_checkpoint)
    if checkpoint_hash != OFFICIAL_STARNET_S1_SHA256:
        raise ValueError(f"Unexpected StarNet-S1 checkpoint SHA256: {checkpoint_hash}")
    star_package = torch.load(star_checkpoint, map_location="cpu", weights_only=False)
    if star_package.get("arch") != "starnet_s1" or "state_dict" not in star_package:
        raise ValueError(f"Not the official StarNet-S1 checkpoint: keys={list(star_package)}")
    star_source = star_package["state_dict"]
    target_state = model.state_dict()
    target_parameters = dict(model.named_parameters())

    official_prefixes = ("model.0.stem.", "model.0.stages.")
    adapter_prefix = "model.0.adapters."
    official_target_state_names = [name for name in target_state if name.startswith(official_prefixes)]
    official_target_parameter_names = [name for name in target_parameters if name.startswith(official_prefixes)]
    adapter_parameter_names = [name for name in target_parameters if name.startswith(adapter_prefix)]
    matched: dict[str, torch.Tensor] = {}
    records = []
    for source_name, value in star_source.items():
        if source_name.startswith(("norm.", "head.")):
            continue
        target_name = f"model.0.{source_name}"
        if target_name not in target_state:
            raise KeyError(f"Official S1 tensor has no semantic target: {source_name}")
        if target_state[target_name].shape != value.shape:
            raise ValueError(
                f"Official S1 shape mismatch {source_name}: {tuple(value.shape)} != {tuple(target_state[target_name].shape)}"
            )
        matched[target_name] = value.detach().clone()
        records.append({"source": source_name, "target": target_name, "source_family": "official_starnet_s1"})
    missing_official_state = sorted(set(official_target_state_names) - set(matched))
    if missing_official_state:
        raise AssertionError(f"Official S1 backbone state is not fully covered: {missing_official_state[:10]}")

    yolo_source_model, yolo_source = load_yolo26_source(yolo_weights)
    semantic_groups = {
        "tail": ((9, 4), (10, 5)),
        "neck": tuple((source, source - 5) for source in range(11, 23)),
        "head": ((23, 18),),
    }
    yolo_records = []
    for group, layer_pairs in semantic_groups.items():
        for source_layer, target_layer in layer_pairs:
            source_module = yolo_source_model.model[source_layer]
            target_module = model.model[target_layer]
            if type(source_module).__name__ != type(target_module).__name__:
                raise TypeError(
                    f"Semantic module mismatch {group}: source {source_layer} {type(source_module).__name__} "
                    f"!= target {target_layer} {type(target_module).__name__}"
                )
            source_prefix = f"model.{source_layer}."
            for source_name, value in yolo_source.items():
                if not source_name.startswith(source_prefix):
                    continue
                suffix = source_name[len(source_prefix) :]
                target_name = f"model.{target_layer}.{suffix}"
                if target_name not in target_state or target_state[target_name].shape != value.shape:
                    continue
                if target_name in matched:
                    raise AssertionError(f"Two initialization sources target {target_name}")
                matched[target_name] = value.detach().clone()
                yolo_records.append(
                    {
                        "source": source_name,
                        "target": target_name,
                        "source_family": "yolo26n",
                        "component": group,
                    }
                )

    incompatible = model.load_state_dict(matched, strict=False)
    loaded_parameter_names = {name for name in matched if name in target_parameters}
    state_after = model.state_dict()
    changed = [name for name, value in matched.items() if not torch.equal(state_after[name].cpu(), value.cpu())]
    if changed:
        raise AssertionError(f"Initialized tensors changed during load: {changed[:10]}")

    def layer_parameter_names(indices: range | tuple[int, ...]) -> list[str]:
        prefixes = tuple(f"model.{index}." for index in indices)
        return [name for name in target_parameters if name.startswith(prefixes)]

    tail_names = layer_parameter_names((4, 5))
    neck_names = layer_parameter_names(range(6, 18))
    head_names = layer_parameter_names((18,))
    regression_names = [
        name for name in head_names if name.startswith(("model.18.cv2.", "model.18.one2one_cv2."))
    ]
    classification_names = [
        name for name in head_names if name.startswith(("model.18.cv3.", "model.18.one2one_cv3."))
    ]
    other_head_names = [name for name in head_names if name not in set(regression_names + classification_names)]
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    loaded_parameters = sum(target_parameters[name].numel() for name in loaded_parameter_names)
    random_parameter_names = [name for name in target_parameters if name not in loaded_parameter_names]
    audit = {
        "candidate": "starnet_s1_official_yolo26",
        "official_starnet_checkpoint": str(star_checkpoint.resolve()),
        "official_starnet_checkpoint_sha256": checkpoint_hash,
        "official_starnet_arch": star_package["arch"],
        "official_backbone": _parameter_coverage(
            target_parameters, official_target_parameter_names, loaded_parameter_names
        ),
        "adapter": {
            **_parameter_coverage(target_parameters, adapter_parameter_names, loaded_parameter_names),
            "expected_random": True,
        },
        "tail_sppf_c2psa": _parameter_coverage(target_parameters, tail_names, loaded_parameter_names),
        "neck": _parameter_coverage(target_parameters, neck_names, loaded_parameter_names),
        "detect": {
            "all": _parameter_coverage(target_parameters, head_names, loaded_parameter_names),
            "regression": _parameter_coverage(target_parameters, regression_names, loaded_parameter_names),
            "classification": _parameter_coverage(target_parameters, classification_names, loaded_parameter_names),
            "other": _parameter_coverage(target_parameters, other_head_names, loaded_parameter_names),
            "random_parameter_names": [name for name in head_names if name not in loaded_parameter_names],
        },
        "loaded_parameters": loaded_parameters,
        "total_parameters": total_parameters,
        "loaded_parameter_fraction": loaded_parameters / total_parameters,
        "random_parameters": total_parameters - loaded_parameters,
        "random_parameter_fraction": 1.0 - loaded_parameters / total_parameters,
        "random_parameter_names": random_parameter_names,
        "matched_state_items": len(matched),
        "official_state_items": len(records),
        "yolo_state_items": len(yolo_records),
        "missing_state_items": list(incompatible.missing_keys),
        "unexpected_state_items": list(incompatible.unexpected_keys),
        "records": records + yolo_records,
        "initialized_state_sha256": state_sha256(state_after),
        "official_backbone_state_sha256": state_sha256(state_after, official_target_state_names),
        "adapter_state_sha256": state_sha256(
            state_after, [name for name in state_after if name.startswith(adapter_prefix)]
        ),
    }
    if audit["official_backbone"]["coverage"] != 1.0:
        raise AssertionError(audit["official_backbone"])
    return audit
