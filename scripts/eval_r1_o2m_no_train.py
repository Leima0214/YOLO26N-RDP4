"""No-training paired O2O versus O2M evaluation for the frozen R1 checkpoint."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ultralytics import YOLO
from ultralytics.utils.torch_utils import init_seeds


PARENT_SHA = "9ebdf1229ed31701283e39bab5094c443a04077f82cb1b74a42a8dd63dfea351"


def emit(event: str, **values) -> None:
    print(json.dumps({"event": event, **values}, ensure_ascii=False, sort_keys=True), flush=True)


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def tensor_sha(parameters) -> str:
    h = hashlib.sha256()
    for name, value in parameters:
        tensor = value.detach().cpu().contiguous()
        h.update(name.encode())
        h.update(str(tensor.dtype).encode())
        h.update(str(tuple(tensor.shape)).encode())
        h.update(tensor.numpy().tobytes())
    return h.hexdigest()


def make_o2m_artifact(checkpoint: Path, output: Path) -> dict:
    wrapper = YOLO(str(checkpoint))
    source = wrapper.model.float().cpu().eval()
    head = source.model[-1]
    assert head.end2end and hasattr(head, "one2one_cv2") and hasattr(head, "one2one_cv3")
    o2m_state_before = tensor_sha((name, value) for name, value in source.state_dict().items() if ".cv2." in name or ".cv3." in name)
    source_params = sum(x.numel() for x in source.parameters())

    sample = torch.zeros(2, 3, 640, 640)
    with torch.inference_mode():
        source_raw = source(sample)[1]["one2many"]
        source_box_logits = source_raw["boxes"].clone()
        source_cls_logits = source_raw["scores"].clone()

    deployed = copy.deepcopy(source)
    deployed_head = deployed.model[-1]
    delattr(deployed_head, "one2one_cv2")
    delattr(deployed_head, "one2one_cv3")
    assert not deployed_head.end2end
    assert not deployed.end2end
    deployed_params = sum(x.numel() for x in deployed.parameters())
    o2m_state_after = tensor_sha((name, value) for name, value in deployed.state_dict().items() if ".cv2." in name or ".cv3." in name)
    assert o2m_state_after == o2m_state_before, "O2M parameters changed while removing O2O"
    train_args = copy.deepcopy(wrapper.ckpt.get("train_args", {}))
    torch.save({
        "model": deployed,
        "ema": None,
        "optimizer": None,
        "epoch": -1,
        "train_args": train_args,
        "o2m_eval_artifact": {
            "parent": str(checkpoint),
            "parent_sha256": PARENT_SHA,
            "training_steps": 0,
            "removed": ["one2one_cv2", "one2one_cv3"],
            "inference": "one2many plus standard NMS",
        },
    }, output)

    reloaded = YOLO(str(output)).model.float().cpu().eval()
    assert not reloaded.model[-1].end2end and not reloaded.end2end
    with torch.inference_mode():
        reloaded_output = reloaded(sample)
        assert isinstance(reloaded_output, tuple)
        reloaded_raw = reloaded_output[1]
    assert torch.equal(source_box_logits, reloaded_raw["boxes"])
    assert torch.equal(source_cls_logits, reloaded_raw["scores"])
    report = {
        "parent_sha256": PARENT_SHA,
        "artifact_sha256": sha256(output),
        "training_steps": 0,
        "o2m_raw_box_logits_max_abs_diff": 0.0,
        "o2m_raw_class_logits_max_abs_diff": 0.0,
        "o2m_state_sha256": o2m_state_after,
        "source_parameters_with_both_heads": source_params,
        "o2m_deployment_parameters": deployed_params,
        "removed_o2o_parameters": source_params - deployed_params,
        "artifact": str(output),
    }
    emit("O2M_ARTIFACT_PASS", **report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    args.checkpoint = args.checkpoint.resolve()
    args.data = args.data.resolve()
    args.output = args.output.resolve()
    assert sha256(args.checkpoint) == PARENT_SHA
    args.output.mkdir(parents=True, exist_ok=False)
    init_seeds(args.seed, deterministic=True)
    protocol = {
        "question": "How does the already-trained R1 O2M head perform at inference versus its paired O2O head?",
        "boundary": "No training or fine-tuning; same R1 seed42 100E best checkpoint; Val only; Test untouched.",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": PARENT_SHA,
        "O2O": "native end-to-end NMS-free Top300 path",
        "O2M": "existing jointly-trained one2many head with standard NMS",
        "shared_evaluator": {"imgsz": 640, "batch": args.batch, "workers": args.workers, "conf": 0.001, "iou": 0.7, "max_det": 300, "coco_max_det": 100, "rect": True},
        "comparison_boundary": "O2O and O2M are different inference/postprocessing protocols; report separately.",
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
    }
    write_json(args.output / "protocol.json", protocol)
    emit("PROTOCOL", **protocol)
    artifact = args.output / "R1_O2M_ONLY.pt"
    static = make_o2m_artifact(args.checkpoint, artifact)
    write_json(args.output / "static_verification.json", static)
    if args.smoke:
        write_json(args.output / "smoke_pass.json", {"passed": True, "static": static})
        emit("SMOKE_PASS")
        return

    val_output = args.output / "val"
    command = [
        sys.executable, "-u", str(ROOT / "scripts/eval_japan4_candidate.py"),
        "--checkpoint", f"R1_O2O={args.checkpoint}",
        "--checkpoint", f"R1_O2M={artifact}",
        "--data", str(args.data), "--output", str(val_output),
        "--imgsz", "640", "--batch", str(args.batch), "--workers", str(args.workers), "--device", "0",
    ]
    emit("VAL_START", command=command, log=str(args.output / "val.log"))
    with (args.output / "val.log").open("x") as handle:
        subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, check=True)
    metrics = json.loads((val_output / "metrics.json").read_text())
    rows = {row["model"]: row for row in metrics["main"]}
    classes = {(row["model"], row["class"]): row for row in metrics["per_class"]}
    known = json.loads((ROOT / "reports/roadsnake_r1_100e_val_20260818/metrics.json").read_text())
    known_main = known["main"][0]
    known_classes = {row["class"]: row for row in known["per_class"]}
    parity = {key: abs(rows["R1_O2O"][key] - known_main[key]) for key in ("AP50_95", "AP75", "AP_small", "AR100")}
    for name in ("D00", "D10", "D20", "D40"):
        parity[f"{name}_AP75"] = abs(classes[("R1_O2O", name)]["AP75"] - known_classes[name]["AP75"])
    assert max(parity.values()) < 1e-7, parity
    delta = {key: rows["R1_O2M"][key] - rows["R1_O2O"][key] for key in ("P", "R", "AP50", "AP50_95", "AP75", "AP_small", "AP_medium", "AP_large", "AR100", "params", "GFLOPs", "pytorch_batch1_latency_ms")}
    per_class_delta = {
        name: {key: classes[("R1_O2M", name)][key] - classes[("R1_O2O", name)][key] for key in ("P", "R", "AP50", "AP50_95", "AP75", "AR100")}
        for name in ("D00", "D10", "D20", "D40")
    }
    summary = {
        "protocol": protocol,
        "static": static,
        "metrics": metrics,
        "O2M_minus_O2O": delta,
        "per_class_O2M_minus_O2O": per_class_delta,
        "parent_parity_max_abs_error": max(parity.values()),
        "decision_boundary": "Diagnostic only; no claim that O2M preserves YOLO26 NMS-free deployment identity.",
    }
    write_json(args.output / "summary.json", summary)
    write_json(args.output / "complete.json", {"complete": True, "O2M_minus_O2O": delta})
    emit("COMPLETE", delta=delta, per_class_delta=per_class_delta, parent_parity_max_abs_error=max(parity.values()))


if __name__ == "__main__":
    main()
