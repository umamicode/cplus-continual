#!/usr/bin/env python3
"""Print the fixed TPU command plan for the released CSQA comparison."""

from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "configs" / "reproduction_matrix.json").read_text(encoding="utf-8"))
REPRODUCE = ROOT / "scripts" / "reproduce.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-jsonl", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--tp-size", type=int, default=4)
    return parser.parse_args()


def command(
    phase: str,
    anchors: int,
    seed: int,
    args: argparse.Namespace,
    method: str | None = None,
) -> str:
    suite = CONFIG["main_csqa"]
    model_size = suite["model_size"]
    model_id = CONFIG["models"][model_size]["id"]
    parts = [
        "python",
        str(REPRODUCE),
        phase,
        "--model-size",
        model_size,
        "--model-path",
        str(args.model_root / model_id),
        "--task",
        suite["task"],
        "--anchors",
        str(anchors),
        "--seed",
        str(seed),
        "--source-jsonl",
        str(args.source_jsonl),
        "--output-root",
        str(args.output_root),
        "--tp-size",
        str(args.tp_size),
        "--execute",
    ]
    if method is not None:
        parts.extend(("--method", method))
    return shlex.join(parts)


def main() -> None:
    args = parse_args()
    suite = CONFIG["main_csqa"]
    baseline_anchor = suite["anchors"][0]

    print("#!/usr/bin/env bash")
    print("set -euo pipefail")
    print()
    print("# Fixed CSQA main-result plan: 9 TPU jobs.")
    print("# Inspect paths and request one TPU allocation before execution.")
    for seed in suite["seeds"]:
        # SFT is anchor-independent, so it is evaluated once per seed.
        print(command("train", baseline_anchor, seed, args, "sft"))
    for anchors in suite["anchors"]:
        for seed in suite["seeds"]:
            print(command("train", anchors, seed, args, "replay"))
            print(command("train", anchors, seed, args, "cpl"))
            print(command("probes", anchors, seed, args))
            print(command("train", anchors, seed, args, "cplus"))


if __name__ == "__main__":
    main()
