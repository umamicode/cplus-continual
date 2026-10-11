#!/usr/bin/env python3
"""Aggregate completed Qwen3 reproduction runs into a summary CSV."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


MODEL_FROM_SLUG = {
    "qwen3-0p6b": "0.6B",
    "qwen3-1p7b": "1.7B",
    "qwen3-4b": "4B",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    return parser.parse_args()


def metric(summary: dict[str, object], *keys: str) -> float:
    for key in keys:
        value = summary.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    raise KeyError(f"none of the metric keys are present: {keys}")


def parse_run_path(path: Path, root: Path) -> tuple[str, str, str, int, int]:
    relative = path.relative_to(root)
    parts = relative.parts
    try:
        runs_index = parts.index("runs")
        model_slug, task, method, anchor_part, seed_part = parts[runs_index + 1 : runs_index + 6]
    except (ValueError, IndexError) as exc:
        raise ValueError(f"unexpected run path: {path}") from exc
    if model_slug not in MODEL_FROM_SLUG:
        raise ValueError(f"unknown model slug in {path}")
    return (
        MODEL_FROM_SLUG[model_slug],
        task,
        method,
        int(anchor_part.removeprefix("a")),
        int(seed_part.removeprefix("seed")),
    )


def standard_error(values: list[float]) -> float:
    return statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0


def main() -> None:
    args = parse_args()
    rows: list[dict[str, object]] = []
    for path in sorted(args.input_root.rglob("final_summary.json")):
        if "runs" not in path.parts:
            continue
        model, task, method, anchors, seed = parse_run_path(path, args.input_root)
        summary = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            {
                "model": model,
                "task": task,
                "method": method,
                "anchors": anchors,
                "seed": seed,
                "retention": metric(summary, "correct_accuracy_all", "correct_accuracy"),
                "acquisition": metric(summary, "wrong_accuracy", "acquisition_accuracy"),
            }
        )
    if not rows:
        raise SystemExit(f"no completed runs found under {args.input_root}")

    groups: dict[tuple[str, str, str, int], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        anchor_key = 0 if row["method"] == "sft" else int(row["anchors"])
        groups[(str(row["model"]), str(row["task"]), str(row["method"]), anchor_key)].append(row)

    output_rows: list[dict[str, object]] = []
    for (model, task, method, anchors), group in sorted(groups.items()):
        seeds = [int(row["seed"]) for row in group]
        if len(seeds) != len(set(seeds)):
            raise ValueError(f"duplicate seeds for {(model, task, method, anchors)}")
        retention = [100.0 * float(row["retention"]) for row in group]
        acquisition = [100.0 * float(row["acquisition"]) for row in group]
        output_rows.append(
            {
                "model": model,
                "task": task,
                "method": method,
                "anchors": "" if method == "sft" else anchors,
                "n": len(group),
                "seeds": ",".join(str(seed) for seed in sorted(seeds)),
                "retention_mean_pct": statistics.fmean(retention),
                "retention_se_pct": standard_error(retention),
                "acquisition_mean_pct": statistics.fmean(acquisition),
                "acquisition_se_pct": standard_error(acquisition),
            }
        )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(output_rows[0])
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"wrote {len(output_rows)} groups to {args.output_csv}")


if __name__ == "__main__":
    main()
