#!/usr/bin/env python3
"""Build or execute one protocol-locked Qwen3 TPU command."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "reproduction_matrix.json"
RUNNER = ROOT / "jax_sft" / "run_qwen3_tpu.sh"
UNSET_ENV = (
    "CORRECT_COUNT",
    "MAX_ROWS",
    "MAX_CORRECT",
    "MAX_WRONG",
    "SYNTHETIC_CORRECT_JSONLS",
    "SYNTHETIC_CORRECT_SOURCE_JSONLS",
    "REPLAY_JSONLS",
    "REPLAY_UPDATE_MATCHED",
    "REPLAY_WEIGHT",
    "REAL_ANCHOR_JSONLS",
    "WANDB_PROJECT",
    "WANDB_ENTITY",
    "WANDB_MODE",
)


def parse_args() -> argparse.Namespace:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("probes", "train"))
    parser.add_argument("--model-size", choices=tuple(config["models"]), required=True)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--task", choices=(config["main_csqa"]["task"],), required=True)
    parser.add_argument("--method", choices=("sft", "replay", "cpl", "cplus"))
    parser.add_argument("--anchors", type=int, choices=config["main_csqa"]["anchors"], required=True)
    parser.add_argument("--seed", type=int, choices=config["main_csqa"]["seeds"], required=True)
    parser.add_argument("--source-jsonl", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=ROOT / "artifacts" / "csqa-main")
    parser.add_argument("--probe-bank", type=Path)
    parser.add_argument("--tp-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.phase == "train" and args.method is None:
        parser.error("--method is required for phase=train")
    if args.phase == "probes" and args.method is not None:
        parser.error("--method is not used for phase=probes")
    return args


def model_slug(size: str) -> str:
    return {"0.6B": "qwen3-0p6b", "1.7B": "qwen3-1p7b", "4B": "qwen3-4b"}[size]


def default_probe_bank(args: argparse.Namespace) -> Path:
    return (
        args.output_root
        / "probes"
        / model_slug(args.model_size)
        / args.task
        / f"a{args.anchors}"
        / f"seed{args.seed}"
        / "jsonl"
        / "synthetic_correct_reference.jsonl"
    )


def paths_for(args: argparse.Namespace) -> dict[str, Path]:
    slug = model_slug(args.model_size)
    shared = args.output_root / "splits" / slug / args.task
    probe_dir = default_probe_bank(args).parents[1]
    if args.phase == "probes":
        out_dir = probe_dir
    else:
        out_dir = (
            args.output_root
            / "runs"
            / slug
            / args.task
            / str(args.method)
            / f"a{args.anchors}"
            / f"seed{args.seed}"
        )
    return {
        "out_dir": out_dir,
        "correct": shared / "initially_correct.jsonl",
        "wrong": shared / "initially_incorrect.jsonl",
        "probe_bank": args.probe_bank or default_probe_bank(args),
    }


def build_environment(args: argparse.Namespace) -> tuple[dict[str, str], dict[str, Path]]:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    spec = config["models"][args.model_size]
    paths = paths_for(args)
    model_name = str(args.model_path.resolve()) if args.model_path else spec["id"]
    model_tag = spec["id"].split("/")[-1]

    env = {
        "MODEL_NAME": model_name,
        "MODEL_TAG": model_tag,
        "SOURCE_JSONLS": str(args.source_jsonl.resolve()),
        "OUT_CORRECT": str(paths["correct"].resolve()),
        "OUT_WRONG": str(paths["wrong"].resolve()),
        "OUT_DIR": str(paths["out_dir"].resolve()),
        "LR": f"{args.learning_rate:g}",
        "EPOCHS": str(args.epochs),
        "TP_SIZE": str(args.tp_size),
        "OPTIMIZER": "adamw",
        "WEIGHT_DECAY": "1e-1",
        "MAX_LENGTH": "256",
        "CORRECT_RATIO": "100",
        "CORRECT_SEED": str(args.seed),
        "TINY_ANCHOR_COUNT": str(args.anchors),
        "TINY_ANCHOR_SEED": str(args.seed),
        "SYNTHETIC_CORRECT_N": "512",
        "SYNTHETIC_SEED": str(args.seed),
        "SYNTHETIC_ICL_EXAMPLES": "2",
        "SYNTHETIC_ICL_RESAMPLE": "1",
        "SYNTHETIC_ICL_STRATEGY": "random",
        "SYNTHETIC_ICL_MAX_COMBINATIONS": "700000",
        "SYNTHETIC_ICL_ANCHOR_SCHEDULE": "random",
        "SYNTHETIC_GENERATION_TEMPERATURE": "0.8",
        "SYNTHETIC_GENERATION_BATCH_SIZE": "2",
        "SYNTHETIC_GENERATION_MAX_LENGTH": "2048",
        "SYNTHETIC_GENERATION_MAX_NEW_TOKENS": "192",
        "SYNTHETIC_GENERATION_ENABLE_THINKING": "false",
        "SYNTHETIC_GENERATION_ANSWER_AWARE": "false",
        "REFERENCE_REDUCTION": "mean",
        "NO_CVAR": "1",
        "REFERENCE_BATCH_SIZE": "1",
        "CNL_MASK_EMA_BETA": "0.9",
        "ANCHOR_SHRINKAGE_FIXED_REAL_WEIGHT": "0.5",
        "ANCHOR_BLEND_NORM_ALIGN": "0",
        "WANDB_MODE": "disabled",
        "SAVE_FINAL_MODEL": "0",
        "SKIP_SPLIT": "0",
    }

    if args.smoke:
        env.update(
            {
                "MAX_ROWS": "64",
                "MAX_CORRECT": "32",
                "MAX_WRONG": "8",
                "EPOCHS": "1",
                "SYNTHETIC_CORRECT_N": "16",
            }
        )

    if args.phase == "probes":
        env.update(
            {
                "METHOD_NAME": "cnl_synth_teacher_anchor_fixed_blend_ema",
                "USE_FREEZE": "1",
                "SYNTHETIC_CORRECT_MODE": "anchor_icl",
                "SYNTHETIC_GENERATION_ONLY": "true",
                "EPOCHS": "0",
                "RUN_NAME": f"probes-{model_slug(args.model_size)}-{args.task}-a{args.anchors}-seed{args.seed}",
            }
        )
        env.pop("SYNTHETIC_CORRECT_JSONLS", None)
        return env, paths

    assert args.method is not None
    env["RUN_NAME"] = (
        f"csqa-main-{model_slug(args.model_size)}-{args.task}-{args.method}"
        f"-a{args.anchors}-seed{args.seed}"
    )
    if args.method == "sft":
        env.update({"METHOD_NAME": "sft", "USE_FREEZE": "0", "CNL_MASK_EMA_BETA": "0"})
    elif args.method == "replay":
        env.update(
            {
                "METHOD_NAME": "replay",
                "USE_FREEZE": "0",
                "CORRECT_COUNT": str(args.anchors),
                "REPLAY_UPDATE_MATCHED": "false",
                "REPLAY_WEIGHT": "0.5",
                "CNL_MASK_EMA_BETA": "0",
            }
        )
    elif args.method == "cpl":
        env.update(
            {
                "METHOD_NAME": "cnl",
                "USE_FREEZE": "1",
                "CORRECT_COUNT": str(args.anchors),
                "TINY_ANCHOR_AUDIT": "0",
                "CNL_MASK_EMA_BETA": "0",
            }
        )
    else:
        env.update(
            {
                "METHOD_NAME": "cnl_synth_teacher_anchor_fixed_blend_ema",
                "USE_FREEZE": "1",
                "TINY_ANCHOR_AUDIT": "1",
                "SYNTHETIC_CORRECT_MODE": "anchor_icl",
                "SYNTHETIC_CORRECT_JSONLS": str(paths["probe_bank"].resolve()),
            }
        )
    return env, paths


def printable_command(env: dict[str, str], task: str) -> str:
    assignments = [f"{key}={shlex.quote(value)}" for key, value in sorted(env.items())]
    command = ["bash", str(RUNNER), task]
    separator = " \\" + "\n  "
    unset = ["env", *(piece for name in UNSET_ENV for piece in ("-u", name))]
    return separator.join([shlex.join(unset), *assignments, shlex.join(command)])


def main() -> None:
    args = parse_args()
    env, paths = build_environment(args)
    print(printable_command(env, args.task))
    if not args.execute:
        return

    if not args.source_jsonl.is_file():
        raise SystemExit(f"missing source JSONL: {args.source_jsonl}")
    if args.model_path is not None and not args.model_path.is_dir():
        raise SystemExit(f"missing model directory: {args.model_path}")
    if args.phase == "train" and args.method == "cplus" and not paths["probe_bank"].is_file():
        raise SystemExit(f"missing probe bank: {paths['probe_bank']}")

    paths["out_dir"].mkdir(parents=True, exist_ok=True)
    paths["correct"].parent.mkdir(parents=True, exist_ok=True)
    process_env = os.environ.copy()
    for name in UNSET_ENV:
        process_env.pop(name, None)
    process_env.update(env)
    process_env["PYTHONPATH"] = str(ROOT)
    subprocess.run(["bash", str(RUNNER), args.task], cwd=ROOT, env=process_env, check=True)


if __name__ == "__main__":
    main()
