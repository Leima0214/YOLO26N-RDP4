"""No-training 2x2 R1/GBRG by O2O/O2M inference-head diagnostic."""
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


SHA = {
    "R1_O2O": "9ebdf1229ed31701283e39bab5094c443a04077f82cb1b74a42a8dd63dfea351",
    "R1_O2M": "684b81c92b7fbc44d7233d4e36179a585d388ab5deace742cc088eb732018475",
    "GBRG_O2O": "fdb4d3ce289ebc2511b36f847c8f615f0a79be64edad3c598615cb5744c72ddd",
}


def emit(event: str, **values) -> None:
    print(json.dumps({"event": event, **values}, ensure_ascii=False, sort_keys=True), flush=True)


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def tensor_digest(items) -> str:
    h = hashlib.sha256()
    for name, value in items:
        tensor = value.detach().cpu().contiguous()
        h.update(name.encode())
        h.update(str(tensor.dtype).encode())
        h.update(str(tuple(tensor.shape)).encode())
        h.update(tensor.numpy().tobytes())
    return h.hexdigest()


def make_gbrg_o2m(checkpoint: Path, output: Path) -> dict:
    wrapper = YOLO(str(checkpoint))
    source = wrapper.model.float().cpu().eval()
    head = source.model[-1]
    assert type(head).__name__ == "RoadSnakeGBRGDetect" and head.end2end
    source_params = sum(x.numel() for x in source.parameters())
    o2m_hash = tensor_digest((name, value) for name, value in source.state_dict().items() if ".cv2." in name or ".cv3." in name)
    sample = torch.zeros(2, 3, 640, 640)
    with torch.inference_mode():
        raw = source(sample)[1]["one2many"]
        box_logits, cls_logits = raw["boxes"].clone(), raw["scores"].clone()

    deployed = copy.deepcopy(source)
    deployed_head = deployed.model[-1]
    delattr(deployed_head, "one2one_cv2")
    delattr(deployed_head, "one2one_cv3")
    assert not deployed_head.end2end and not deployed.end2end
    after_hash = tensor_digest((name, value) for name, value in deployed.state_dict().items() if ".cv2." in name or ".cv3." in name)
    assert after_hash == o2m_hash
    train_args = copy.deepcopy(wrapper.ckpt.get("train_args", {}))
    torch.save({
        "model": deployed, "ema": None, "optimizer": None, "epoch": -1, "train_args": train_args,
        "o2m_eval_artifact": {"parent": str(checkpoint), "parent_sha256": SHA["GBRG_O2O"], "training_steps": 0,
                              "removed": ["one2one_cv2", "one2one_cv3"], "inference": "one2many plus standard NMS"},
    }, output)
    reloaded = YOLO(str(output)).model.float().cpu().eval()
    assert not reloaded.end2end
    with torch.inference_mode():
        reloaded_raw = reloaded(sample)[1]
    assert torch.equal(box_logits, reloaded_raw["boxes"])
    assert torch.equal(cls_logits, reloaded_raw["scores"])
    report = {
        "parent_sha256": SHA["GBRG_O2O"], "artifact_sha256": digest(output), "training_steps": 0,
        "raw_box_logits_max_abs_diff": 0.0, "raw_class_logits_max_abs_diff": 0.0,
        "o2m_state_sha256": after_hash, "source_parameters_with_both_heads": source_params,
        "o2m_deployment_parameters": sum(x.numel() for x in deployed.parameters()),
        "removed_o2o_parameters": source_params - sum(x.numel() for x in deployed.parameters()),
        "artifact": str(output),
    }
    emit("GBRG_O2M_ARTIFACT_PASS", **report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r1-o2o", type=Path, required=True)
    parser.add_argument("--r1-o2m", type=Path, required=True)
    parser.add_argument("--gbrg-o2o", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    args.r1_o2o, args.r1_o2m, args.gbrg_o2o = args.r1_o2o.resolve(), args.r1_o2m.resolve(), args.gbrg_o2o.resolve()
    args.data, args.output = args.data.resolve(), args.output.resolve()
    for name, path in {"R1_O2O": args.r1_o2o, "R1_O2M": args.r1_o2m, "GBRG_O2O": args.gbrg_o2o}.items():
        assert digest(path) == SHA[name], (name, digest(path))
    args.output.mkdir(parents=True, exist_ok=False)
    init_seeds(args.seed, deterministic=True)
    protocol = {
        "question": "Does GBRG change O2M inference relative to R1, and is its O2M trade-off different from O2O?",
        "boundary": "No training; seed42 100E best checkpoints; Val only; Test untouched.",
        "design": "2x2: R1 versus GBRG, crossed with O2O NMS-free versus existing O2M plus standard NMS.",
        "shared_evaluator": {"imgsz": 640, "batch": args.batch, "workers": args.workers, "conf": 0.001, "iou": 0.7, "max_det": 300, "coco_max_det": 100, "rect": True},
        "causal_boundary": "Within-head R1-to-GBRG differences estimate GBRG association; head differences also include NMS/postprocessing semantics.",
        "checkpoints": {"R1_O2O": str(args.r1_o2o), "R1_O2M": str(args.r1_o2m), "GBRG_O2O": str(args.gbrg_o2o)},
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
    }
    write_json(args.output / "protocol.json", protocol)
    emit("PROTOCOL", **protocol)
    gbrg_o2m = args.output / "GBRG_O2M_ONLY.pt"
    static = make_gbrg_o2m(args.gbrg_o2o, gbrg_o2m)
    write_json(args.output / "static_verification.json", static)
    if args.smoke:
        write_json(args.output / "smoke_pass.json", {"passed": True, "static": static})
        emit("SMOKE_PASS")
        return

    val = args.output / "val"
    model_paths = {"R1_O2O": args.r1_o2o, "R1_O2M": args.r1_o2m, "GBRG_O2O": args.gbrg_o2o, "GBRG_O2M": gbrg_o2m}
    command = [sys.executable, "-u", str(ROOT / "scripts/eval_japan4_candidate.py"), "--data", str(args.data),
               "--output", str(val), "--imgsz", "640", "--batch", str(args.batch), "--workers", str(args.workers), "--device", "0"]
    for name, path in model_paths.items():
        command.extend(["--checkpoint", f"{name}={path}"])
    emit("VAL_START", command=command, log=str(args.output / "val.log"))
    with (args.output / "val.log").open("x") as handle:
        subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, check=True)
    metrics = json.loads((val / "metrics.json").read_text())
    rows = {row["model"]: row for row in metrics["main"]}
    classes = {(row["model"], row["class"]): row for row in metrics["per_class"]}
    r1_known = json.loads((ROOT / "reports/r1_o2m_no_train_20260827/formal/val/metrics.json").read_text())
    r1_rows = {row["model"]: row for row in r1_known["main"]}
    gbrg_known = json.loads((ROOT / "reports/roadsnake_gbrg_100e_eval_20260823/metrics.json").read_text())["main"][0]
    parity = {"R1_O2O_AP": abs(rows["R1_O2O"]["AP50_95"] - r1_rows["R1_O2O"]["AP50_95"]),
              "R1_O2M_AP": abs(rows["R1_O2M"]["AP50_95"] - r1_rows["R1_O2M"]["AP50_95"]),
              "GBRG_O2O_AP": abs(rows["GBRG_O2O"]["AP50_95"] - gbrg_known["AP50_95"])}
    assert max(parity.values()) < 1e-7, parity
    keys = ("P", "R", "AP50", "AP50_95", "AP75", "AP_small", "AP_medium", "AP_large", "AR100")
    def subtract(a, b): return {key: rows[a][key] - rows[b][key] for key in keys}
    comparisons = {
        "GBRG_O2M_minus_GBRG_O2O": subtract("GBRG_O2M", "GBRG_O2O"),
        "GBRG_O2M_minus_R1_O2M": subtract("GBRG_O2M", "R1_O2M"),
        "R1_O2M_minus_R1_O2O": subtract("R1_O2M", "R1_O2O"),
    }
    interaction = {key: comparisons["GBRG_O2M_minus_R1_O2M"][key] - (rows["GBRG_O2O"][key] - rows["R1_O2O"][key]) for key in keys}
    per_class = {}
    for cls in ("D00", "D10", "D20", "D40"):
        per_class[cls] = {
            "GBRG_O2M_minus_GBRG_O2O": {key: classes[("GBRG_O2M", cls)][key] - classes[("GBRG_O2O", cls)][key] for key in ("AP50", "AP50_95", "AP75", "AR100")},
            "GBRG_O2M_minus_R1_O2M": {key: classes[("GBRG_O2M", cls)][key] - classes[("R1_O2M", cls)][key] for key in ("AP50", "AP50_95", "AP75", "AR100")},
        }
    summary = {"protocol": protocol, "static": static, "metrics": metrics, "comparisons": comparisons,
               "interaction_difference_of_differences": interaction, "per_class": per_class,
               "parent_parity_max_abs_error": max(parity.values()),
               "boundary": "One seed, no training; O2M uses NMS and is not the native YOLO26 end-to-end path."}
    write_json(args.output / "summary.json", summary)
    write_json(args.output / "complete.json", {"complete": True, "comparisons": comparisons, "interaction": interaction})
    emit("COMPLETE", comparisons=comparisons, interaction=interaction, per_class=per_class)


if __name__ == "__main__":
    main()
