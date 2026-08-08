"""Verify C12 transfer, near-identity inference, isolated gradients, and finite loss."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ultralytics import YOLO  # noqa: E402
from ultralytics.cfg import get_cfg  # noqa: E402
from ultralytics.nn.japan4_adapters import QualityAwareDetect  # noqa: E402
from ultralytics.utils import YAML  # noqa: E402
from ultralytics.utils.loss import QualityAwareE2ELoss  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, default=ROOT / "yolo26n.pt")
    parser.add_argument(
        "--model",
        type=Path,
        default=ROOT / "ultralytics/cfg/models/26/yolo26n-japan4-c12-quality-o2o.yaml",
    )
    parser.add_argument(
        "--parent",
        type=Path,
        default=ROOT / "ultralytics/cfg/models/26/yolo26.yaml",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--imgsz", type=int, default=128)
    args = parser.parse_args()
    yaml_path = args.model if args.model.is_absolute() else ROOT / args.model
    parent_yaml_path = args.parent if args.parent.is_absolute() else ROOT / args.parent
    assert args.weights.is_file() and yaml_path.is_file() and parent_yaml_path.is_file()
    assert args.imgsz >= 64 and args.imgsz % 32 == 0
    parent_yaml, c12_yaml = YAML.load(parent_yaml_path), YAML.load(yaml_path)
    assert c12_yaml["backbone"] == parent_yaml["backbone"]
    assert c12_yaml["head"][:-1] == parent_yaml["head"][:-1]
    assert c12_yaml["head"][-1][:2] == parent_yaml["head"][-1][:2]

    device = torch.device(args.device)
    checkpoint = YOLO(str(args.weights), task="detect", verbose=False).model.float().to(device).eval()
    source = YOLO(str(parent_yaml_path), task="detect", verbose=False).model.float().to(device).eval()
    source.load(checkpoint, verbose=False)
    candidate = YOLO(str(yaml_path), task="detect", verbose=False).model.float().to(device)
    source_state, candidate_state = source.state_dict(), candidate.state_dict()
    assert all(name in candidate_state and candidate_state[name].shape == value.shape for name, value in source_state.items())
    candidate.load(source, verbose=False)
    candidate.args = get_cfg()
    candidate.args.epochs = 100
    head = candidate.model[-1]
    assert isinstance(head, QualityAwareDetect)
    assert candidate.stride.tolist() == [8.0, 16.0, 32.0]

    sample = torch.randn(1, 3, args.imgsz, args.imgsz, device=device)
    candidate.eval()
    with torch.no_grad():
        source_output = source(sample)[0]
        candidate_inference = candidate(sample)
        candidate_output = candidate_inference[0]
        quality = candidate_inference[1]["one2one"]["quality"].sigmoid()
    torch.testing.assert_close(quality, torch.full_like(quality, 0.99), atol=1e-6, rtol=0)
    torch.testing.assert_close(candidate_output[..., :4], source_output[..., :4], atol=0, rtol=0)
    torch.testing.assert_close(candidate_output[..., 5], source_output[..., 5], atol=0, rtol=0)
    torch.testing.assert_close(candidate_output[..., 4], source_output[..., 4] * 0.99, atol=1e-6, rtol=1e-5)

    candidate.train()
    preds = candidate(sample)
    batch = {
        "batch_idx": torch.zeros(1, device=device),
        "cls": torch.zeros(1, 1, device=device),
        "bboxes": torch.tensor([[0.5, 0.5, 0.4, 0.4]], device=device),
    }
    criterion = candidate.init_criterion()
    assert isinstance(criterion, QualityAwareE2ELoss)
    assigned = criterion.one2one.get_assigned_targets_and_loss(preds["one2one"], batch)[0]
    quality_loss = criterion.quality_loss(preds["one2one"], assigned)
    candidate.zero_grad(set_to_none=True)
    quality_loss.backward()
    quality_parameters = dict(head.one2one_quality.named_parameters())
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in quality_parameters.values())
    assert all(
        parameter.grad is None
        for name, parameter in candidate.named_parameters()
        if "one2one_quality" not in name
    )

    candidate.zero_grad(set_to_none=True)
    losses, loss_items = criterion(candidate(sample), batch)
    assert losses.shape == loss_items.shape == (4,)
    assert torch.isfinite(losses).all() and torch.isfinite(loss_items).all()
    losses.sum().backward()
    assert all(parameter.grad is not None for parameter in quality_parameters.values())
    print(
        "QUALITY COMBINATION PASS "
        f"model={yaml_path.stem} parent={parent_yaml_path.stem} "
        f"source_items={len(source_state)} candidate_items={len(candidate_state)} "
        f"quality_params={sum(p.numel() for p in quality_parameters.values())} "
        f"quality_init={quality.mean().item():.6f} loss_items={loss_items.detach().cpu().tolist()}"
    )


if __name__ == "__main__":
    main()
