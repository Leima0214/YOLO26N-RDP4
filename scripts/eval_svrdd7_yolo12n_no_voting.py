"""Evaluate one native YOLO12 checkpoint on frozen SVRDD7 Val without Voting."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT / "scripts"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from rs_mid_bootstrap import guard_optional_visualization_imports

guard_optional_visualization_imports()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()

    from rs_mid_experiment import file_hash
    from rs_mid_o2m import ValCOCOEvaluator, write_json
    from ultralytics.nn.tasks import load_checkpoint
    from ultralytics.utils.torch_utils import select_device

    checkpoint = args.checkpoint.expanduser().resolve()
    data = args.data.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not checkpoint.is_file() or not data.is_file():
        raise FileNotFoundError(f"Missing checkpoint or data snapshot: {checkpoint}, {data}")
    net, _ = load_checkpoint(str(checkpoint), device=select_device(args.device, verbose=False))
    payload = ValCOCOEvaluator(data, native_o2m=True).evaluate(net, voting=False, output=output)
    if set(payload["metrics"]) != {"current"}:
        raise AssertionError("No-Voting evaluator emitted an unexpected prediction branch")
    payload.update(
        checkpoint=str(checkpoint),
        checkpoint_sha256=file_hash(checkpoint),
        deployment="native_YOLO12_O2M_no_voting",
        test_sealed=True,
    )
    write_json(output / "metrics.json", payload)
    overall = payload["metrics"]["current"]["overall"]
    print(
        "YOLO12_NO_VOTING_COMPLETED "
        f"AP={overall['AP']:.6f} AP50={overall['AP50']:.6f} AP75={overall['AP75']:.6f}"
    )


if __name__ == "__main__":
    main()
