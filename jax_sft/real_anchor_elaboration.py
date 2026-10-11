"""Real-anchor question-answer elaborations used as CNL references."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
from tqdm import tqdm


def parse_mcq_options(question: str) -> dict[str, str]:
    options: dict[str, str] = {}
    for line in question.splitlines():
        match = re.match(r"^\s*([A-D])\s*[:.)]\s*(.+?)\s*$", line)
        if match:
            options[match.group(1)] = match.group(2).strip()
    if set(options) != set("ABCD"):
        raise ValueError("expected exactly four A-D options in the anchor")
    return options


def semantic_anchor_answer(row: dict[str, Any]) -> tuple[str, str]:
    label = str(row.get("label", "")).strip().upper()
    if label not in "ABCD" or len(label) != 1:
        raise ValueError(f"invalid real-anchor label: {row.get('label')!r}")
    return label, parse_mcq_options(str(row["question"]))[label]


def strip_answer_only_instruction(question: str) -> str:
    lines = question.strip().splitlines()
    prefixes = (
        "please reply with only",
        "please answer with only",
        "reply with only",
        "answer with only",
        "respond with only",
    )
    while lines and not lines[-1].strip():
        lines.pop()
    if lines and lines[-1].strip().lower().startswith(prefixes):
        lines.pop()
    return "\n".join(lines).strip()


def make_real_anchor_elaboration_prompt(row: dict[str, Any]) -> str:
    """Ask the frozen model to explain a mastered question-answer pair."""
    label, answer_text = semantic_anchor_answer(row)
    question = strip_answer_only_instruction(str(row["question"]))
    return (
        "The following multiple-choice question and answer are verified. "
        "Explain why the answer is correct using the underlying knowledge, "
        "then state one concise reusable fact.\n\n"
        f"{question}\n\n"
        f"Verified answer: {label}: {answer_text}\n\n"
        "Return exactly these two fields:\n"
        "Explanation: <concise reasoning>\n"
        "Key fact: <one reusable fact>"
    )


def parse_real_anchor_elaboration(text: str) -> dict[str, str] | None:
    cleaned = text.strip()
    if "</think>" in cleaned:
        cleaned = cleaned.split("</think>", 1)[1].strip()
    cleaned = re.sub(
        r"(?im)^\s*(?:[-+]\s*)?(?:#{1,6}\s*)?"
        r"\*{0,2}(Explanation|Key\s+fact)\*{0,2}\s*[-:]",
        lambda match: f"{match.group(1)}:",
        cleaned,
    )
    match = re.search(
        r"Explanation:\s*(.+?)\s*Key fact:\s*(.+)\s*$",
        cleaned,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not match:
        return None
    explanation = re.sub(r"\s+", " ", match.group(1)).strip()
    key_fact = re.sub(r"\s+", " ", match.group(2)).strip()
    if len(explanation) < 8 or len(key_fact) < 4:
        return None
    return {"explanation": explanation, "key_fact": key_fact}


def fallback_real_anchor_elaboration(
    text: str,
    anchor: dict[str, Any],
) -> dict[str, str]:
    """Recover useful content when only the requested field format failed."""
    _, answer_text = semantic_anchor_answer(anchor)
    cleaned = text.strip()
    if "</think>" in cleaned:
        cleaned = cleaned.split("</think>", 1)[1].strip()
    cleaned = re.sub(r"[*_`#]+", "", cleaned)
    parts = re.split(
        r"(?i)\bKey\s+fact\s*[-:]",
        cleaned,
        maxsplit=1,
    )
    explanation = re.sub(
        r"(?i)^\s*Explanation\s*[-:]\s*",
        "",
        parts[0],
    )
    explanation = re.sub(r"\s+", " ", explanation).strip()
    if len(explanation) < 8:
        explanation = (
            f"The verified answer is {answer_text}, as stated in the item."
        )
    key_fact = ""
    if len(parts) == 2:
        key_fact = re.sub(r"\s+", " ", parts[1]).strip()
    if len(key_fact) < 4:
        key_fact = f"The verified answer for this item is {answer_text}."
    return {
        "explanation": explanation,
        "key_fact": key_fact,
    }


def build_real_anchor_elaboration_reference_row(
    anchor: dict[str, Any],
    elaboration: dict[str, Any],
) -> dict[str, str]:
    """Build an LM row whose response is the frozen model's elaboration."""
    label, answer_text = semantic_anchor_answer(anchor)
    question = strip_answer_only_instruction(str(anchor["question"]))
    return {
        "instruction": (
            f"{question}\n\n"
            f"Verified answer: {label}: {answer_text}\n"
            "Explain the answer and state the key fact."
        ),
        "response": (
            f"Explanation: {elaboration['explanation']}\n"
            f"Key fact: {elaboration['key_fact']}"
        ),
    }


def validate_real_anchor_elaboration_cache(
    rows: list[dict[str, Any]],
    anchors: list[dict[str, Any]],
) -> None:
    if len(rows) != len(anchors):
        raise ValueError(
            f"real-anchor elaboration cache has {len(rows)} rows; "
            f"expected {len(anchors)}"
        )
    for index, (cached, anchor) in enumerate(zip(rows, anchors)):
        if int(cached.get("anchor_index", -1)) != index:
            raise ValueError(
                "real-anchor elaboration cache indices are not contiguous"
            )
        if cached.get("question") != anchor.get("question"):
            raise ValueError(
                f"real-anchor elaboration question mismatch at row {index}"
            )
        if cached.get("label") != anchor.get("label"):
            raise ValueError(
                f"real-anchor elaboration label mismatch at row {index}"
            )
        if not cached.get("explanation") or not cached.get("key_fact"):
            raise ValueError(
                f"real-anchor elaboration row {index} is incomplete"
            )


def load_real_anchor_elaboration_cache(
    path: str | Path,
    anchors: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    validate_real_anchor_elaboration_cache(rows, anchors)
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")


def generate_real_anchor_elaboration_cache(
    anchors: list[dict[str, Any]],
    complete: Callable[
        [list[str], int],
        tuple[list[str], list[dict[str, Any]]],
    ],
    *,
    output_path: str | Path,
    batch_size: int,
    retries: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Generate one parsed elaboration for every ordered real anchor."""
    if batch_size <= 0 or retries <= 0:
        raise ValueError("generation batch size and retries must be positive")
    started = time.perf_counter()
    completed: list[dict[str, Any] | None] = [None] * len(anchors)
    pending = list(range(len(anchors)))
    trace_rows: list[dict[str, Any]] = []
    attempt_count = 0
    parse_success_count = 0
    truncated_count = 0
    fallback_count = 0
    latest_generations: list[
        tuple[str, dict[str, Any]] | None
    ] = [None] * len(anchors)

    for retry in range(retries):
        if not pending:
            break
        next_pending: list[int] = []
        for start in tqdm(
            range(0, len(pending), batch_size),
            desc=f"real-anchor elaborations retry {retry + 1}",
        ):
            indices = pending[start : start + batch_size]
            prompts = [
                make_real_anchor_elaboration_prompt(anchors[index])
                for index in indices
            ]
            completions, metadata = complete(
                prompts,
                seed + retry * 1_000_003 + start,
            )
            if len(completions) != len(indices) or len(metadata) != len(indices):
                raise RuntimeError(
                    "real-anchor elaboration sampler returned a wrong-sized batch"
                )
            for index, completion, generation_meta in zip(
                indices,
                completions,
                metadata,
            ):
                attempt_count += 1
                truncated_count += int(generation_meta.get("truncated", False))
                latest_generations[index] = (
                    completion,
                    dict(generation_meta),
                )
                parsed = parse_real_anchor_elaboration(completion)
                parse_success_count += int(parsed is not None)
                trace_rows.append(
                    {
                        "anchor_index": index,
                        "retry": retry,
                        "raw_generation": completion,
                        "parsed": parsed,
                        **generation_meta,
                    }
                )
                if parsed is None:
                    next_pending.append(index)
                    continue
                label, answer_text = semantic_anchor_answer(anchors[index])
                completed[index] = {
                    "anchor_index": index,
                    "question": anchors[index]["question"],
                    "label": label,
                    "gold_answer_text": answer_text,
                    "explanation": parsed["explanation"],
                    "key_fact": parsed["key_fact"],
                    "raw_generation": completion,
                    "generated_token_count": int(
                        generation_meta.get("generated_token_count", 0)
                    ),
                    "truncated": bool(
                        generation_meta.get("truncated", False)
                    ),
                    "used_fallback": False,
                }
        pending = next_pending

    output_path = Path(output_path)
    trace_path = output_path.with_name(
        output_path.stem + "_trace.jsonl"
    )
    _write_jsonl(trace_path, trace_rows)
    if pending:
        for index in pending:
            latest = latest_generations[index]
            if latest is None:
                raise RuntimeError(
                    "real-anchor elaboration generation returned no output "
                    f"for anchor {index}; inspect {trace_path}"
                )
            completion, generation_meta = latest
            parsed = fallback_real_anchor_elaboration(
                completion,
                anchors[index],
            )
            label, answer_text = semantic_anchor_answer(anchors[index])
            completed[index] = {
                "anchor_index": index,
                "question": anchors[index]["question"],
                "label": label,
                "gold_answer_text": answer_text,
                "explanation": parsed["explanation"],
                "key_fact": parsed["key_fact"],
                "raw_generation": completion,
                "generated_token_count": int(
                    generation_meta.get("generated_token_count", 0)
                ),
                "truncated": bool(
                    generation_meta.get("truncated", False)
                ),
                "used_fallback": True,
            }
            fallback_count += 1
    result = [row for row in completed if row is not None]
    validate_real_anchor_elaboration_cache(result, anchors)
    _write_jsonl(output_path, result)
    return result, {
        "real_anchor_elaboration/generation_seconds": float(
            time.perf_counter() - started
        ),
        "real_anchor_elaboration/elaboration_count": float(len(result)),
        "real_anchor_elaboration/generation_attempt_count": float(
            attempt_count
        ),
        "real_anchor_elaboration/parser_yield": float(
            parse_success_count / max(attempt_count, 1)
        ),
        "real_anchor_elaboration/truncated_count": float(truncated_count),
        "real_anchor_elaboration/fallback_count": float(fallback_count),
        "real_anchor_elaboration/cache_reused": 0.0,
    }


def build_paired_anchor_reference_batches(
    anchor_batches: list[dict[str, Any]],
    elaboration_batches: list[dict[str, Any]],
    *,
    elaboration_weight: float,
) -> list[dict[str, Any]]:
    """Give each original/elaboration family a total reference weight of one."""
    if len(anchor_batches) != len(elaboration_batches):
        raise ValueError("anchor and elaboration reference batches must align")
    if not 0 < elaboration_weight < 1:
        raise ValueError("elaboration weight must lie strictly between 0 and 1")
    original_weight = 1.0 - elaboration_weight
    originals = [
        {
            **batch,
            "base_example_index": index,
            "reference_weight": original_weight,
            "real_anchor_reference_view": "original_mcq",
        }
        for index, batch in enumerate(anchor_batches)
    ]
    elaborations = [
        {
            **batch,
            "base_example_index": index,
            "reference_weight": elaboration_weight,
            "real_anchor_reference_view": "qa_elaboration",
        }
        for index, batch in enumerate(elaboration_batches)
    ]
    return originals + elaborations


def paired_anchor_group_assignments(
    anchor_count: int,
    *,
    groups: int,
    seed: int,
) -> list[int]:
    """Place both views of each real anchor in the same CVaR group."""
    if anchor_count <= 0 or groups <= 0:
        raise ValueError("anchor count and group count must be positive")
    groups = min(groups, anchor_count)
    order = np.random.default_rng(seed).permutation(anchor_count)
    anchor_groups = [0 for _ in range(anchor_count)]
    for rank, anchor_index in enumerate(order.tolist()):
        anchor_groups[int(anchor_index)] = int(rank % groups)
    return anchor_groups + anchor_groups
