"""Build deterministic B0-initialized RoadLite-26 A1/A2 checkpoints.

The transfer is dependency-aware: every selected output channel is reused by
the corresponding downstream input, residual, concat, depthwise convolution,
and Detect-scale branch. BN magnitude is the primary selection score; Conv L1
is the fallback when a convolution has no BN.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ultralytics import YOLO, __version__
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.nn.modules.block import Attention, Bottleneck, C2PSA, C3k, C3k2, PSABlock, SPPF
from ultralytics.nn.modules.conv import Concat, Conv
from ultralytics.nn.modules.head import Detect
from ultralytics.nn.tasks import DetectionModel


DEFAULT_YAMLS = (
    ROOT / "ultralytics/cfg/models/26/roadlite26-a1.yaml",
    ROOT / "ultralytics/cfg/models/26/roadlite26-a2.yaml",
)


def idx(values, *, device="cpu") -> torch.Tensor:
    return torch.as_tensor(values, dtype=torch.long, device=device)


def tensor_hash(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ChannelInheritor:
    """Copy a thinner YOLO graph from a source graph with consistent channel dependencies."""

    def __init__(self, source: DetectionModel, target: DetectionModel):
        self.source = source
        self.target = target
        self.names = {id(module): name for name, module in target.named_modules()}
        self.touched: set[str] = set()
        self.selection_log: list[dict] = []
        self.intentional_random: set[str] = set()

    def key(self, module: nn.Module, suffix: str) -> str:
        prefix = self.names[id(module)]
        return f"{prefix}.{suffix}" if prefix else suffix

    @torch.no_grad()
    def put(self, module: nn.Module, suffix: str, value: torch.Tensor) -> None:
        owner, attr = module, suffix
        if "." in suffix:
            path, attr = suffix.rsplit(".", 1)
            for part in path.split("."):
                owner = getattr(owner, part)
        destination = getattr(owner, attr)
        destination.copy_(value.to(device=destination.device, dtype=destination.dtype))
        self.touched.add(self.key(module, suffix))

    def select(self, source: Conv | nn.Conv2d, count: int, candidates=None, label="") -> torch.Tensor:
        convolution = source.conv if isinstance(source, Conv) else source
        candidates = idx(range(convolution.out_channels)) if candidates is None else idx(candidates)
        if count > candidates.numel():
            raise ValueError(f"{label}: requested {count} of only {candidates.numel()} source channels")
        if count == candidates.numel():
            chosen = candidates.clone()
            rule = "identity"
        else:
            if isinstance(source, Conv) and hasattr(source, "bn"):
                scores = source.bn.weight.detach().abs().cpu()[candidates]
                rule = "bn_abs_gamma"
            else:
                scores = convolution.weight.detach().abs().flatten(1).sum(1).cpu()[candidates]
                rule = "conv_filter_l1"
            order = torch.argsort(scores, descending=True, stable=True)[:count]
            chosen = candidates[order].sort().values
        self.selection_log.append(
            {"node": label, "rule": rule, "source": int(candidates.numel()), "target": count, "indices": chosen.tolist()}
        )
        return chosen

    @torch.no_grad()
    def copy_bn(self, source: nn.BatchNorm2d, target: nn.BatchNorm2d, out_map: torch.Tensor) -> None:
        for name in ("weight", "bias", "running_mean", "running_var"):
            self.put(target, name, getattr(source, name).detach().cpu()[out_map])
        self.put(target, "num_batches_tracked", source.num_batches_tracked.detach().cpu())

    @torch.no_grad()
    def copy_raw_conv(
        self,
        source: nn.Conv2d,
        target: nn.Conv2d,
        in_map: torch.Tensor,
        out_map: torch.Tensor | None = None,
        label: str = "",
    ) -> torch.Tensor:
        depthwise = source.groups == source.in_channels == source.out_channels
        if depthwise:
            if not (target.groups == target.in_channels == target.out_channels == in_map.numel()):
                raise ValueError(f"{label}: incompatible depthwise convolution")
            out_map = in_map
            weight = source.weight.detach().cpu()[out_map]
        else:
            if source.groups != 1 or target.groups != 1:
                raise NotImplementedError(f"{label}: grouped non-depthwise convolution is unsupported")
            out_map = self.select(source, target.out_channels, label=label) if out_map is None else idx(out_map)
            if out_map.numel() and int(out_map.max()) >= source.out_channels:
                raise ValueError(f"{label}: output index {int(out_map.max())} >= {source.out_channels}")
            if in_map.numel() and int(in_map.max()) >= source.in_channels:
                raise ValueError(f"{label}: input index {int(in_map.max())} >= {source.in_channels}")
            weight = source.weight.detach().cpu()[out_map][:, in_map]
        if tuple(weight.shape) != tuple(target.weight.shape):
            raise ValueError(f"{label}: mapped weight {tuple(weight.shape)} != target {tuple(target.weight.shape)}")
        self.put(target, "weight", weight)
        if target.bias is not None:
            if source.bias is None:
                raise ValueError(f"{label}: target has bias but source does not")
            self.put(target, "bias", source.bias.detach().cpu()[out_map])
        return out_map

    def copy_conv(
        self,
        source: Conv,
        target: Conv,
        in_map: torch.Tensor,
        out_map: torch.Tensor | None = None,
        label: str = "",
    ) -> torch.Tensor:
        depthwise = source.conv.groups == source.conv.in_channels == source.conv.out_channels
        if depthwise:
            out_map = in_map
        elif out_map is None:
            out_map = self.select(source, target.conv.out_channels, label=label)
        out_map = self.copy_raw_conv(source.conv, target.conv, in_map, out_map, label)
        self.copy_bn(source.bn, target.bn, out_map)
        return out_map

    def bottleneck(self, source: Bottleneck, target: Bottleneck, in_map: torch.Tensor, label: str) -> torch.Tensor:
        hidden = self.copy_conv(source.cv1, target.cv1, in_map, label=f"{label}.cv1")
        forced = in_map if target.add else None
        return self.copy_conv(source.cv2, target.cv2, hidden, forced, f"{label}.cv2")

    def c3k(self, source: C3k, target: C3k, in_map: torch.Tensor, label: str) -> torch.Tensor:
        current = self.copy_conv(source.cv1, target.cv1, in_map, label=f"{label}.cv1")
        source_hidden = source.cv1.conv.out_channels
        for i, (source_block, target_block) in enumerate(zip(source.m, target.m)):
            current = self.bottleneck(source_block, target_block, current, f"{label}.m.{i}")
        branch = self.copy_conv(source.cv2, target.cv2, in_map, label=f"{label}.cv2")
        return self.copy_conv(
            source.cv3, target.cv3, torch.cat((current, source_hidden + branch)), label=f"{label}.cv3"
        )

    def attention(self, source: Attention, target: Attention, in_map: torch.Tensor, label: str) -> torch.Tensor:
        source_span = 2 * source.key_dim + source.head_dim
        q_raw, k_raw, v_raw, v_channels = [], [], [], []
        for head in range(source.num_heads):
            base = head * source_span
            q_raw.extend(range(base, base + source.key_dim))
            k_raw.extend(range(base + source.key_dim, base + 2 * source.key_dim))
            v_raw.extend(range(base + 2 * source.key_dim, base + source_span))
            v_channels.extend(range(head * source.head_dim, (head + 1) * source.head_dim))

        q = self.select(source.qkv, target.num_heads * target.key_dim, q_raw, f"{label}.q")
        k = self.select(source.qkv, target.num_heads * target.key_dim, k_raw, f"{label}.k")
        v_choice = self.select(source.qkv, target.num_heads * target.head_dim, v_raw, f"{label}.v")
        v_lookup = {raw: channel for raw, channel in zip(v_raw, v_channels)}
        v_map = idx([v_lookup[int(raw)] for raw in v_choice])
        raw_map = []
        for head in range(target.num_heads):
            q0, q1 = head * target.key_dim, (head + 1) * target.key_dim
            v0, v1 = head * target.head_dim, (head + 1) * target.head_dim
            raw_map.extend(q[q0:q1].tolist())
            raw_map.extend(k[q0:q1].tolist())
            raw_map.extend(v_choice[v0:v1].tolist())
        self.copy_conv(source.qkv, target.qkv, in_map, idx(raw_map), f"{label}.qkv")
        self.copy_conv(source.pe, target.pe, v_map, v_map, f"{label}.pe")
        self.copy_conv(source.proj, target.proj, v_map, in_map, f"{label}.proj")
        return in_map

    def psa(self, source: PSABlock, target: PSABlock, in_map: torch.Tensor, label: str) -> torch.Tensor:
        self.attention(source.attn, target.attn, in_map, f"{label}.attn")
        hidden = self.copy_conv(source.ffn[0], target.ffn[0], in_map, label=f"{label}.ffn.0")
        self.copy_conv(source.ffn[1], target.ffn[1], hidden, in_map, f"{label}.ffn.1")
        return in_map

    def c3k2(self, source: C3k2, target: C3k2, in_map: torch.Tensor, label: str) -> torch.Tensor:
        source_hidden, target_hidden = source.c, target.c
        first = self.select(source.cv1, target_hidden, range(source_hidden), f"{label}.cv1.a")
        second_raw = self.select(
            source.cv1, target_hidden, range(source_hidden, 2 * source_hidden), f"{label}.cv1.b"
        )
        second = second_raw - source_hidden
        self.copy_conv(source.cv1, target.cv1, in_map, torch.cat((first, second_raw)), f"{label}.cv1")
        pieces, current = [first, source_hidden + second], second
        for i, (source_block, target_block) in enumerate(zip(source.m, target.m)):
            current = self.module(source_block, target_block, current, f"{label}.m.{i}")
            pieces.append((i + 2) * source_hidden + current)
        return self.copy_conv(source.cv2, target.cv2, torch.cat(pieces), label=f"{label}.cv2")

    def sppf(self, source: SPPF, target: SPPF, in_map: torch.Tensor, label: str) -> torch.Tensor:
        hidden = self.copy_conv(source.cv1, target.cv1, in_map, label=f"{label}.cv1")
        source_hidden = source.cv1.conv.out_channels
        pooled = torch.cat([i * source_hidden + hidden for i in range(source.n + 1)])
        forced = in_map if target.add else None
        return self.copy_conv(source.cv2, target.cv2, pooled, forced, f"{label}.cv2")

    def c2psa(self, source: C2PSA, target: C2PSA, in_map: torch.Tensor, label: str) -> torch.Tensor:
        source_hidden, target_hidden = source.c, target.c
        first = self.select(source.cv1, target_hidden, range(source_hidden), f"{label}.cv1.a")
        second_raw = self.select(
            source.cv1, target_hidden, range(source_hidden, 2 * source_hidden), f"{label}.cv1.b"
        )
        second = second_raw - source_hidden
        self.copy_conv(source.cv1, target.cv1, in_map, torch.cat((first, second_raw)), f"{label}.cv1")
        current = second
        for i, (source_block, target_block) in enumerate(zip(source.m, target.m)):
            current = self.psa(source_block, target_block, current, f"{label}.m.{i}")
        return self.copy_conv(
            source.cv2, target.cv2, torch.cat((first, source_hidden + current)), label=f"{label}.cv2"
        )

    def sequence(self, source: nn.Sequential, target: nn.Sequential, in_map: torch.Tensor, label: str) -> torch.Tensor:
        current = in_map
        for i, (source_block, target_block) in enumerate(zip(source, target)):
            current = self.module(source_block, target_block, current, f"{label}.{i}")
        return current

    def detect_branch(
        self, source: nn.Sequential, target: nn.Sequential, in_map: torch.Tensor, label: str, classifier=False
    ) -> None:
        current = in_map
        for i, (source_block, target_block) in enumerate(zip(source[:-1], target[:-1])):
            current = self.module(source_block, target_block, current, f"{label}.{i}")
        source_last, target_last = source[-1], target[-1]
        if classifier and source_last.out_channels != target_last.out_channels:
            self.intentional_random.update({self.key(target_last, "weight"), self.key(target_last, "bias")})
        else:
            self.copy_raw_conv(source_last, target_last, current, label=f"{label}.{len(target) - 1}")

    def detect(self, source: Detect, target: Detect, inputs: list[torch.Tensor], label: str) -> None:
        for family in ("cv2", "one2one_cv2"):
            if hasattr(target, family):
                for scale, (source_branch, target_branch, in_map) in enumerate(
                    zip(getattr(source, family), getattr(target, family), inputs)
                ):
                    self.detect_branch(source_branch, target_branch, in_map, f"{label}.{family}.{scale}")
        for family in ("cv3", "one2one_cv3"):
            if hasattr(target, family):
                for scale, (source_branch, target_branch, in_map) in enumerate(
                    zip(getattr(source, family), getattr(target, family), inputs)
                ):
                    self.detect_branch(source_branch, target_branch, in_map, f"{label}.{family}.{scale}", True)
        self.put(target, "stride", source.stride.detach().cpu())

    def module(self, source: nn.Module, target: nn.Module, in_map: torch.Tensor, label: str) -> torch.Tensor:
        if type(source) is not type(target):
            raise TypeError(f"{label}: {type(source).__name__} != {type(target).__name__}")
        if isinstance(target, Conv):
            return self.copy_conv(source, target, in_map, label=label)
        if isinstance(target, Bottleneck):
            return self.bottleneck(source, target, in_map, label)
        if isinstance(target, C3k):
            return self.c3k(source, target, in_map, label)
        if isinstance(target, C3k2):
            return self.c3k2(source, target, in_map, label)
        if isinstance(target, SPPF):
            return self.sppf(source, target, in_map, label)
        if isinstance(target, C2PSA):
            return self.c2psa(source, target, in_map, label)
        if isinstance(target, PSABlock):
            return self.psa(source, target, in_map, label)
        if isinstance(target, nn.Sequential):
            return self.sequence(source, target, in_map, label)
        if isinstance(target, nn.Conv2d):
            return self.copy_raw_conv(source, target, in_map, label=label)
        raise TypeError(f"{label}: unsupported {type(target).__name__}")

    @staticmethod
    def source_channels(module: nn.Module, fallback: int) -> int:
        if isinstance(module, Conv):
            return module.conv.out_channels
        if isinstance(module, (C3k2, SPPF, C2PSA)):
            return module.cv2.conv.out_channels
        if isinstance(module, nn.Upsample):
            return fallback
        raise TypeError(f"cannot infer output channels for {type(module).__name__}")

    def run(self) -> dict:
        source_layers, target_layers = self.source.model, self.target.model
        if len(source_layers) != len(target_layers):
            raise ValueError("source and target layer counts differ")
        output_maps: list[torch.Tensor | None] = []
        source_widths: list[int | None] = []

        def resolve(layer_from):
            if layer_from == -1:
                return idx(range(3)) if not output_maps else output_maps[-1]
            return output_maps[layer_from]

        def resolve_width(layer_from):
            if layer_from == -1:
                return 3 if not source_widths else source_widths[-1]
            return source_widths[layer_from]

        for layer_index, (source_layer, target_layer) in enumerate(zip(source_layers, target_layers)):
            label = f"model.{layer_index}"
            if isinstance(target_layer, Detect):
                inputs = [resolve(i) for i in target_layer.f]
                self.detect(source_layer, target_layer, inputs, label)
                output_maps.append(None)
                source_widths.append(None)
            elif isinstance(target_layer, Concat):
                maps, widths, offset = [], [resolve_width(i) for i in target_layer.f], 0
                for channel_map, width in zip((resolve(i) for i in target_layer.f), widths):
                    maps.append(offset + channel_map)
                    offset += width
                output_maps.append(torch.cat(maps))
                source_widths.append(sum(widths))
            elif isinstance(target_layer, nn.Upsample):
                output_maps.append(resolve(target_layer.f))
                source_widths.append(resolve_width(target_layer.f))
            else:
                input_map = resolve(target_layer.f)
                result = self.module(source_layer, target_layer, input_map, label)
                output_maps.append(result)
                source_widths.append(self.source_channels(source_layer, resolve_width(target_layer.f)))

        all_keys = set(self.target.state_dict())
        missing = sorted(all_keys - self.touched - self.intentional_random)
        if missing:
            raise AssertionError(f"unmapped target tensors: {missing}")
        total = sum(value.numel() for value in self.target.state_dict().values())
        random_numel = sum(self.target.state_dict()[name].numel() for name in self.intentional_random)
        return {
            "selection_rule": "BN abs(gamma), Conv-filter L1 fallback; stable top-k then ascending canonical order",
            "mapped_tensor_keys": len(self.touched),
            "intentional_random_keys": sorted(self.intentional_random),
            "mapped_numel_fraction": (total - random_numel) / total,
            "selections": self.selection_log,
        }


def finite_tensors(value) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(finite_tensors(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_tensors(item) for item in value)
    return True


def tensor_leaves(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from tensor_leaves(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from tensor_leaves(item)


def verify_forward_and_gradients(model: DetectionModel, image_size: int) -> dict:
    model.float().train()
    model.zero_grad(set_to_none=True)
    output = model(torch.randn(2, 3, image_size, image_size))
    if not finite_tensors(output):
        raise AssertionError("non-finite first forward")
    loss = sum(tensor.float().mean() for tensor in tensor_leaves(output))
    loss.backward()
    named = dict(model.named_parameters())
    probes = [
        "model.0.conv.weight",
        "model.13.cv1.conv.weight",
        "model.23.cv2.0.0.conv.weight",
        "model.23.one2one_cv2.0.0.conv.weight",
    ]
    for name in probes:
        gradient = named[name].grad
        if gradient is None or not torch.isfinite(gradient).all():
            raise AssertionError(f"missing/non-finite gradient: {name}")
    return {"image_size": image_size, "loss": float(loss.detach()), "finite": True, "gradient_probes": probes}


def verify_trainer_rebuild(yaml_path: Path, inherited: DetectionModel) -> dict:
    trainer = object.__new__(DetectionTrainer)
    trainer.data = {"nc": 4, "channels": 3}
    rebuilt = DetectionTrainer.get_model(trainer, cfg=str(yaml_path), weights=inherited, verbose=False)
    expected, actual = inherited.float().state_dict(), rebuilt.state_dict()
    mismatched = [name for name in expected if name not in actual or not torch.equal(expected[name], actual[name])]
    if mismatched:
        raise AssertionError(f"Trainer reconstruction changed weights: {mismatched[:10]}")
    return {"exact_tensor_keys": len(expected), "all_preserved": True}


def build_target(source: DetectionModel, yaml_path: Path, seed: int) -> tuple[DetectionModel, dict]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    target = DetectionModel(str(yaml_path), nc=4, ch=3, verbose=False).float()
    transfer = ChannelInheritor(source, target).run()
    return target, transfer


def save_checkpoint(model: DetectionModel, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model": deepcopy(model).half(),
        "date": datetime.now().isoformat(),
        "version": __version__,
        "license": "AGPL-3.0 License (https://ultralytics.com/license)",
        "docs": "https://docs.ultralytics.com",
    }
    torch.save(checkpoint, output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, default=ROOT / "yolo26n.pt")
    parser.add_argument("--yamls", nargs="+", type=Path, default=list(DEFAULT_YAMLS))
    parser.add_argument("--output-dir", type=Path, default=ROOT / "weights/roadlite26_inherited")
    parser.add_argument("--report-dir", type=Path, default=ROOT / "reports/roadlite26_inheritance")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=64)
    args = parser.parse_args()

    source = YOLO(str(args.weights)).model.float()
    args.report_dir.mkdir(parents=True, exist_ok=True)
    for yaml_path in args.yamls:
        yaml_path = yaml_path.resolve()
        target, transfer = build_target(source, yaml_path, args.seed)
        first_hash = tensor_hash(target.state_dict())
        duplicate, duplicate_transfer = build_target(source, yaml_path, args.seed)
        second_hash = tensor_hash(duplicate.state_dict())
        if first_hash != second_hash or transfer["selections"] != duplicate_transfer["selections"]:
            raise AssertionError(f"non-deterministic inheritance for {yaml_path.name}")
        del duplicate

        forward = verify_forward_and_gradients(target, args.image_size)
        target.zero_grad(set_to_none=True)
        output = args.output_dir / f"{yaml_path.stem}-b0inherit-nc4.pt"
        save_checkpoint(target, output)
        loaded = YOLO(str(output)).model.float()
        saved_reference = deepcopy(target).half().float()
        if tensor_hash(loaded.state_dict()) != tensor_hash(saved_reference.state_dict()):
            raise AssertionError(f"saved checkpoint reload mismatch: {output}")
        rebuild = verify_trainer_rebuild(yaml_path, loaded)

        report = {
            "yaml": str(yaml_path),
            "yaml_sha256": file_hash(yaml_path),
            "source_weights": str(args.weights.resolve()),
            "source_sha256": file_hash(args.weights.resolve()),
            "output_weights": str(output.resolve()),
            "output_sha256": file_hash(output),
            "seed": args.seed,
            "state_sha256": first_hash,
            "deterministic_repeat": True,
            "transfer": transfer,
            "forward_gradient_check": forward,
            "trainer_rebuild_check": rebuild,
            "attention_note": "Q/K/V roles are preserved; source heads may be merged when target head count is smaller.",
        }
        report_path = args.report_dir / f"{yaml_path.stem}.json"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps({"model": yaml_path.stem, "weights": str(output), "report": str(report_path), "ok": True}))


if __name__ == "__main__":
    main()
