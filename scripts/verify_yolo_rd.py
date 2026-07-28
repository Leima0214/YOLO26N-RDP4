"""Preflight the clean YOLO26n P3 Retriever-Dictionary candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) in sys.path:
    sys.path.remove(str(ROOT))
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from ultralytics import YOLO  # noqa: E402
from ultralytics.cfg import get_cfg  # noqa: E402
from ultralytics.nn.rd_adapter import RDP3Stage  # noqa: E402

BASELINE_YAML = ROOT / "ultralytics/cfg/models/26/yolo26.yaml"
RD_YAML = ROOT / "ultralytics/cfg/models/26/yolo26n-rd-p3-japan7.yaml"
RD_LAYER = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weights",
        default="yolo26n.pt",
        help="Official YOLO26n checkpoint, relative to repository root.",
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument(
        "--dictionary",
        default=None,
        help="Optional Japan7 atom package; omit to check random R1.",
    )
    parser.add_argument("--report", default="reports/yolo_rd_preflight.json")
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensors(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, dict):
        for key in sorted(value):
            yield from tensors(value[key])
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from tensors(item)


def max_output_error(left, right) -> float:
    left_tensors, right_tensors = list(tensors(left)), list(tensors(right))
    require(
        len(left_tensors) == len(right_tensors),
        "baseline and RD outputs expose different tensor structures",
    )
    errors = []
    for lhs, rhs in zip(left_tensors, right_tensors):
        require(
            lhs.shape == rhs.shape,
            f"output shape mismatch: {tuple(lhs.shape)} != {tuple(rhs.shape)}",
        )
        errors.append((lhs.float() - rhs.float()).abs().max().item())
    return max(errors, default=0.0)


def max_fused_inference_error(left, right) -> float:
    """Compare the deployed prediction only; YOLO26 fuse removes the unused O2M branch."""
    require(
        isinstance(left, tuple)
        and isinstance(right, tuple)
        and left
        and right,
        "fused model did not return an inference prediction",
    )
    return max_output_error(left[0], right[0])


def synthetic_batch(
    batch_size: int, imgsz: int, device: torch.device
) -> dict[str, torch.Tensor]:
    return {
        "img": torch.randn(batch_size, 3, imgsz, imgsz, device=device),
        "batch_idx": torch.arange(batch_size, device=device, dtype=torch.long),
        "cls": (torch.arange(batch_size, device=device) % 7).view(-1, 1).float(),
        "bboxes": torch.tensor([[0.5, 0.5, 0.25, 0.10]] * batch_size, device=device),
    }


def load(yaml: Path, weights: Path) -> YOLO:
    model = YOLO(str(yaml), task="detect")
    model.load(str(weights))
    return model


def rd_stage(model):
    stage = model.model[RD_LAYER]
    require(
        isinstance(stage, RDP3Stage),
        f"expected model.{RD_LAYER} to be RDP3Stage, got {type(stage).__name__}",
    )
    require(abs(stage.rd.gamma.item() - 1e-3) < 1e-8, "RD gamma must start at 1e-3")
    return stage


def load_dictionary(path: Path, weights: Path) -> torch.Tensor:
    try:
        package = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        package = torch.load(path, map_location="cpu")
    require(
        isinstance(package, dict)
        and isinstance(package.get("centroids"), torch.Tensor),
        f"invalid dictionary: {path}",
    )
    meta = package.get("meta", {})
    require(
        meta.get("layer") == 4 and meta.get("stride") == 8,
        f"dictionary is not a P3/8 package: {meta}",
    )
    require(
        meta.get("weights_sha256") == sha256(weights),
        "dictionary/checkpoint SHA256 mismatch",
    )
    return package["centroids"]


def check_transfer(baseline, candidate) -> dict[str, int]:
    baseline_state, candidate_state = baseline.state_dict(), candidate.state_dict()
    missing = set(baseline_state) - set(candidate_state)
    require(not missing, f"candidate lost pretrained keys: {sorted(missing)[:5]}")
    mismatched = [
        name
        for name in baseline_state
        if not torch.equal(baseline_state[name], candidate_state[name])
    ]
    require(not mismatched, f"shared pretrained tensors differ: {mismatched[:5]}")
    added = sorted(set(candidate_state) - set(baseline_state))
    require(
        added and all(name.startswith(f"model.{RD_LAYER}.rd.") for name in added),
        f"unexpected new state keys: {added}",
    )
    return {
        "shared_state_items": len(baseline_state),
        "rd_added_state_items": len(added),
    }


def run_backward(
    candidate, device: torch.device, imgsz: int, batch_size: int, amp: bool
) -> dict[str, float]:
    model = candidate.model.to(device).train()
    model.args = get_cfg()
    model.criterion = None
    stage = rd_stage(model)
    batch = synthetic_batch(batch_size, imgsz, device)
    autocast = torch.autocast(
        device_type=device.type,
        dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
        enabled=amp,
    )
    model.zero_grad(set_to_none=True)
    with autocast:
        loss, _ = model.loss(batch)
    require(torch.isfinite(loss).all(), "loss is non-finite")
    loss.sum().backward()
    gamma_grad = stage.rd.gamma.grad
    require(
        gamma_grad is not None
        and torch.isfinite(gamma_grad).all()
        and gamma_grad.abs().item() > 0,
        "gamma lacks gradient",
    )

    required_core = {
        "coefficient": stage.rd.coefficient.conv.weight,
        "exchange": stage.rd.exchange.conv.weight,
        "dictionary": stage.rd.dictionary.weight,
        "pono_scale": stage.rd.pono_scale,
    }
    core_grad_sums = {}
    for name, parameter in required_core.items():
        gradient = parameter.grad
        require(
            gradient is not None
            and torch.isfinite(gradient).all()
            and gradient.abs().sum().item() > 0,
            f"RD {name} lacks a first-batch gradient",
        )
        core_grad_sums[f"{name}_grad_abs_sum"] = float(
            gradient.detach().abs().sum().cpu()
        )
    return {
        "loss": float(loss.detach().cpu()),
        "gamma_grad_abs": float(gamma_grad.detach().abs().cpu()),
        **core_grad_sums,
    }


def main() -> None:
    args = parse_args()
    weights = resolve_path(args.weights)
    dictionary = resolve_path(args.dictionary) if args.dictionary else None
    report = resolve_path(args.report)
    require(BASELINE_YAML.is_file(), f"missing baseline YAML: {BASELINE_YAML}")
    require(RD_YAML.is_file(), f"missing RD YAML: {RD_YAML}")
    require(weights.is_file(), f"missing checkpoint: {weights}")
    if dictionary:
        require(dictionary.is_file(), f"dictionary must exist: {dictionary}")
    require(
        args.imgsz > 0 and args.imgsz % 32 == 0,
        f"imgsz must be a positive multiple of 32, got {args.imgsz}",
    )
    require(args.batch > 0, f"batch must be positive, got {args.batch}")
    device = torch.device(
        "cuda"
        if args.device == "cuda"
        or (args.device == "auto" and torch.cuda.is_available())
        else "cpu"
    )
    if args.device == "cuda":
        require(
            torch.cuda.is_available(), "--device cuda requested but CUDA is unavailable"
        )

    torch.manual_seed(0)
    baseline, candidate = load(BASELINE_YAML, weights), load(RD_YAML, weights)
    transfer = check_transfer(baseline.model, candidate.model)
    stage = rd_stage(candidate.model)
    if dictionary:
        stage.rd.initialize_atoms(load_dictionary(dictionary, weights))
    initial_gamma = float(stage.rd.gamma.detach())
    initial_mix = float(stage.rd.effective_mix().detach())
    atom_norm_error = float(
        (stage.rd.dictionary.normalized_weight().flatten(2).norm(dim=0) - 1).abs().max()
    )
    require(
        0.0 < initial_mix < 1e-3,
        f"RD initial mix must be small and active, got {initial_mix}",
    )
    require(atom_norm_error <= 1e-5, f"dictionary atom norm error {atom_norm_error}")

    baseline.model.to(device).eval()
    candidate.model.to(device).eval()
    image = torch.randn(args.batch, 3, args.imgsz, args.imgsz, device=device)
    saved_gamma = stage.rd.gamma.detach().clone()
    with torch.no_grad():
        stage.rd.gamma.zero_()
    with torch.inference_mode():
        baseline_output, candidate_output = (
            baseline.model(image),
            candidate.model(image),
        )
    identity_error = max_output_error(baseline_output, candidate_output)
    require(
        identity_error == 0.0,
        f"gamma=0 must be bit-exact, observed max error {identity_error}",
    )
    with torch.no_grad():
        stage.rd.gamma.copy_(saved_gamma)
    with torch.inference_mode():
        active_output = candidate.model(image)
    initial_output_delta = max_output_error(baseline_output, active_output)
    require(
        torch.isfinite(torch.tensor(initial_output_delta)),
        "initial RD output delta is non-finite",
    )

    fused = load(RD_YAML, weights).model.to(device).eval()
    fused_stage = rd_stage(fused)
    if dictionary:
        fused_stage.rd.initialize_atoms(load_dictionary(dictionary, weights))
    with torch.inference_mode():
        unfused_output = fused(image)
    fused.fuse()
    with torch.inference_mode():
        fused_output = fused(image)
    fuse_error = max_fused_inference_error(unfused_output, fused_output)
    baseline_fused = load(BASELINE_YAML, weights).model.to(device).eval()
    baseline_fused.fuse()
    with torch.inference_mode():
        baseline_fused_output = baseline_fused(image)
    baseline_fuse_error = max_fused_inference_error(baseline_output, baseline_fused_output)
    require(
        fuse_error <= baseline_fuse_error + 1e-4,
        f"fused RD drift {fuse_error} exceeds native baseline fuse drift {baseline_fuse_error}",
    )

    backward = run_backward(
        candidate, device, args.imgsz, args.batch, amp=device.type == "cuda"
    )
    report.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "PASS",
        "root": str(ROOT),
        "device": str(device),
        "variant": "R2" if dictionary else "R1",
        "rd_layer": RD_LAYER,
        "dictionary": str(dictionary) if dictionary else None,
        "weights": str(weights),
        "weights_sha256": sha256(weights),
        "imgsz": args.imgsz,
        "batch": args.batch,
        "rd_atoms": stage.rd.coefficient.conv.out_channels,
        "rd_params": sum(parameter.numel() for parameter in stage.rd.parameters()),
        "initial_gamma": initial_gamma,
        "initial_mix": initial_mix,
        "initial_output_max_delta": initial_output_delta,
        "atom_norm_max_error": atom_norm_error,
        "identity_max_error": identity_error,
        "fuse_max_error": fuse_error,
        "baseline_fuse_max_error": baseline_fuse_error,
        **transfer,
        **backward,
    }
    report.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
