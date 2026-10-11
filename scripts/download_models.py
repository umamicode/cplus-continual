#!/usr/bin/env python3
"""Download the pinned public Qwen3 checkpoints used by the TPU suite."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import snapshot_download


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "reproduction_matrix.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=ROOT / "weights")
    parser.add_argument(
        "--model-size",
        action="append",
        dest="model_sizes",
        help="Download only this size; repeat for multiple sizes.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    selected = args.model_sizes or list(config["models"])
    unknown = sorted(set(selected) - set(config["models"]))
    if unknown:
        raise SystemExit(f"unknown model sizes: {', '.join(unknown)}")

    manifest = {"schema_version": 1, "models": {}}
    for size in selected:
        spec = config["models"][size]
        destination = args.output_root / spec["id"]
        destination.mkdir(parents=True, exist_ok=True)
        resolved = snapshot_download(
            repo_id=spec["id"],
            revision=spec["revision"],
            local_dir=destination,
        )
        manifest["models"][size] = {
            **spec,
            "local_path": str(Path(resolved).resolve()),
        }
        print(f"{size}: {resolved}")

    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_root / "qwen3-model-manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
