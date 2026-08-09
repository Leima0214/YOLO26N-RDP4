"""Create JSON, CSV, Markdown, and epoch-curve comparisons for Japan4 runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def named_paths(values: list[str]) -> dict[str, Path]:
    output = {}
    for value in values:
        name, separator, raw_path = value.partition("=")
        path = Path(raw_path).expanduser().resolve()
        if not separator or not name or name in output or not path.exists():
            raise ValueError(f"Expected unique NAME=PATH, got {value}")
        output[name] = path
    return output


def read_results(path: Path) -> list[dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = []
        for row in csv.DictReader(handle):
            rows.append({key.strip(): float(value) for key, value in row.items() if value not in (None, "")})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, metavar="NAME=RUN_DIR")
    parser.add_argument("--eval", action="append", default=[], metavar="NAME=EVAL_DIR")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    runs, evaluations = named_paths(args.run), named_paths(args.eval) if args.eval else {}
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)

    raw, summary = {}, []
    for name, run_dir in runs.items():
        results_path = run_dir / "results.csv"
        args_path = run_dir / "args.yaml"
        if not results_path.is_file() or not args_path.is_file():
            raise FileNotFoundError(f"Missing results.csv or args.yaml in {run_dir}")
        rows = read_results(results_path)
        if not rows:
            raise ValueError(f"Empty results: {results_path}")
        key = "metrics/mAP50-95(B)"
        best = max(rows, key=lambda row: row.get(key, float("-inf")))
        record = {
            "model": name,
            "run_dir": str(run_dir),
            "best_epoch": int(best["epoch"]),
            "P": best.get("metrics/precision(B)"),
            "R": best.get("metrics/recall(B)"),
            "AP50": best.get("metrics/mAP50(B)"),
            "AP50_95": best.get(key),
        }
        if name in evaluations:
            metrics = json.loads((evaluations[name] / "metrics.json").read_text(encoding="utf-8"))
            main = next(row for row in metrics["main"] if row["model"] == name)
            record.update(main)
            for row in metrics["per_class"]:
                if row["model"] == name:
                    for metric in ("AP50_95", "AP50", "AP75", "AR100"):
                        record[f"{row['class']}_{metric}"] = row[metric]
        summary.append(record)
        raw[name] = {"epochs": rows, "summary": record}

    fields = list(dict.fromkeys(key for row in summary for key in row))
    with (args.output / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary)
    (args.output / "comparison.json").write_text(json.dumps(raw, indent=2), encoding="utf-8")

    selected = [
        "model",
        "best_epoch",
        "P",
        "R",
        "AP50",
        "AP50_95",
        "AP75",
        "AP_small",
        "AP_medium",
        "AP_large",
        "AR100",
        "params",
        "GFLOPs",
        "pytorch_batch1_latency_ms",
        "checkpoint_MB",
    ]
    selected.extend(f"{class_name}_{metric}" for class_name in ("D00", "D10", "D20", "D40") for metric in ("AP50_95", "AP50", "AP75"))
    selected = [field for field in selected if any(field in row for row in summary)]
    lines = ["# Japan4 Val-only comparison", "", "| " + " | ".join(selected) + " |", "|" + "---|" * len(selected)]
    for row in summary:
        values = [row.get(field, "") for field in selected]
        lines.append("| " + " | ".join(f"{value:.5f}" if isinstance(value, float) else str(value) for value in values) + " |")
    (args.output / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    curves = (
        "metrics/mAP50-95(B)",
        "metrics/mAP50(B)",
        "metrics/precision(B)",
        "metrics/recall(B)",
        "train/region_p3_loss",
        "train/region_p4_loss",
    )
    figure, axes = plt.subplots(3, 2, figsize=(12, 12))
    for axis, metric in zip(axes.flat, curves):
        for name, content in raw.items():
            points = [(row["epoch"], row[metric]) for row in content["epochs"] if metric in row]
            if points:
                axis.plot([point[0] for point in points], [point[1] for point in points], label=name)
        axis.set_title(metric)
        axis.set_xlabel("epoch")
        axis.grid(alpha=0.25)
        if axis.lines:
            axis.legend()
    figure.tight_layout()
    figure.savefig(args.output / "epoch_curves.png", dpi=160)
    plt.close(figure)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
