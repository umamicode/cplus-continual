#!/usr/bin/env python3
"""Validate the normalized four-option CSQA input used by this release."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


OPTION_RE = re.compile(r"(?m)^([A-D]):\\s+\\S")
EXTRA_OPTION_RE = re.compile(r"(?m)^[E-Z]:\\s+\\S")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_jsonl", type=Path)
    parser.add_argument("--expected-rows", type=int, default=1096)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input_jsonl.is_file():
        raise SystemExit(f"missing input JSONL: {args.input_jsonl}")
    rows = 0
    for line_number, raw_line in enumerate(args.input_jsonl.read_text(encoding="utf-8").splitlines(), 1):
        if not raw_line.strip():
            raise SystemExit(f"line {line_number}: blank rows are not permitted")
        try:
            row = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"line {line_number}: invalid JSON ({exc.msg})") from exc
        question = row.get("question")
        label = row.get("label")
        if not isinstance(question, str) or not isinstance(label, str):
            raise SystemExit(f"line {line_number}: expected string fields question and label")
        if label not in {"A", "B", "C", "D"}:
            raise SystemExit(f"line {line_number}: label must be A, B, C, or D")
        if OPTION_RE.findall(question) != ["A", "B", "C", "D"] or EXTRA_OPTION_RE.search(question):
            raise SystemExit(f"line {line_number}: expected exactly one A-D option line and no extra options")
        rows += 1
    if args.expected_rows is not None and rows != args.expected_rows:
        raise SystemExit(f"expected {args.expected_rows} rows, found {rows}")
    digest = hashlib.sha256(args.input_jsonl.read_bytes()).hexdigest()
    print(f"validated {rows} rows")
    print(f"SHA256 {digest}")


if __name__ == "__main__":
    main()
