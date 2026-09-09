"""Evaluate an official YOLO11/YOLO12 checkpoint with the shared SVRDD7 O2M protocol."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT / "scripts"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=ROOT / "configs/svrdd7_rs_mid_remote.yaml")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--voting", action="store_true",
                        help="Also compute the shared score Box Voting output")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    weights = args.weights.expanduser().resolve()
    data = args.data.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not weights.is_file() or not data.is_file():
        raise FileNotFoundError(f"Missing checkpoint or SVRDD7 data YAML: {weights}, {data}")
    from rs_mid_bootstrap import guard_optional_visualization_imports

    guard_optional_visualization_imports()
    from ultralytics.nn.modules.head import Detect
    from ultralytics.nn.tasks import load_checkpoint
    from ultralytics.utils.torch_utils import select_device
    from rs_mid_o2m import ValCOCOEvaluator

    model, _ = load_checkpoint(str(weights), device=select_device(args.device, verbose=False))
    head = model.model[-1]
    if type(head) is not Detect or head.end2end or int(head.nc) != 7:
        raise ValueError("Official YOLO11/YOLO12 evaluation requires a native seven-class O2M Detect")
    evaluator = ValCOCOEvaluator(str(data), 640, args.batch, args.workers, native_o2m=True)
    payload = evaluator.evaluate(model, voting=args.voting, compute_loss=False, output=output)
    print(json.dumps({"weights": str(weights), "output": str(output), "metrics": payload["metrics"]},
                     indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
