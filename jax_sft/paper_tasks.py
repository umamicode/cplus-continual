from __future__ import annotations

import hashlib
import json
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable, Mapping


TASK_CATEGORICAL = "categorical"
TASK_GENERATION = "generation"
SUPPORTED_TASK_TYPES = (TASK_CATEGORICAL, TASK_GENERATION)

CLASSIFICATION_LABELS: dict[str, tuple[str, ...]] = {
    "ag_news": ("World", "Sports", "Business", "Sci/Tech"),
    "yahoo_answers": (
        "Society & Culture",
        "Science & Mathematics",
        "Health",
        "Education & Reference",
        "Computers & Internet",
        "Sports",
        "Business & Finance",
        "Entertainment & Music",
        "Family & Relationships",
        "Politics & Government",
    ),
    "dbpedia": (
        "Company",
        "Educational Institution",
        "Artist",
        "Athlete",
        "Office Holder",
        "Means of Transportation",
        "Building",
        "Natural Place",
        "Village",
        "Animal",
        "Plant",
        "Album",
        "Film",
        "Written Work",
    ),
}


def choice_labels(count: int) -> tuple[str, ...]:
    if not 1 <= count <= 26:
        raise ValueError(f"candidate count must be in [1, 26], got {count}")
    return tuple(chr(ord("A") + index) for index in range(count))


def canonical_task_type(row: dict[str, Any], fallback: str = "categorical") -> str:
    task_type = str(row.get("task_type") or fallback).strip().lower()
    if task_type not in SUPPORTED_TASK_TYPES:
        raise ValueError(f"unsupported task_type={task_type!r}")
    return task_type


def candidate_labels_from_row(
    row: dict[str, Any],
    fallback: Iterable[str] = ("A", "B", "C", "D"),
) -> tuple[str, ...]:
    labels = row.get("candidate_labels")
    if labels is None:
        labels = tuple(fallback)
    elif isinstance(labels, str):
        labels = tuple(part.strip() for part in labels.split(",") if part.strip())
    else:
        labels = tuple(str(part).strip() for part in labels)
    if not labels or any(not label for label in labels):
        raise ValueError("candidate_labels must contain nonempty strings")
    if len(set(labels)) != len(labels):
        raise ValueError("candidate_labels must be unique")
    return tuple(labels)


def validate_homogeneous_task_rows(
    rows: list[dict[str, Any]],
    *,
    fallback_task_type: str = TASK_CATEGORICAL,
    fallback_candidate_labels: Iterable[str] = ("A", "B", "C", "D"),
) -> tuple[str, tuple[str, ...]]:
    if not rows:
        raise ValueError("task rows are empty")
    task_types = {
        canonical_task_type(row, fallback_task_type)
        for row in rows
    }
    if len(task_types) != 1:
        raise ValueError(f"a run cannot mix task types: {sorted(task_types)}")
    task_type = next(iter(task_types))
    if task_type == TASK_GENERATION:
        return task_type, ()
    label_sets = {
        candidate_labels_from_row(row, fallback_candidate_labels)
        for row in rows
    }
    if len(label_sets) != 1:
        raise ValueError("all categorical rows must use the same candidate_labels")
    labels = next(iter(label_sets))
    invalid = sorted(
        {
            str(row.get("label"))
            for row in rows
            if str(row.get("label")) not in labels
        }
    )
    if invalid:
        raise ValueError(f"labels are outside candidate_labels: {invalid}")
    return task_type, labels


def normalize_exact_answer(text: str) -> str:
    """Normalize generation output without changing answer semantics."""
    text = str(text).replace("\r\n", "\n").replace("\r", "\n").strip()
    return text


def exact_answer_match(prediction: str, target: str) -> bool:
    return normalize_exact_answer(prediction) == normalize_exact_answer(target)


_NUMERIC_TOKEN_PATTERN = (
    r"[+-]?(?:(?:\d[\d,]*)(?:\.\d+)?|\.\d+)"
    r"(?:[eE][+-]?\d+)?(?:\s*/\s*[+-]?(?:(?:\d[\d,]*)(?:\.\d+)?|\.\d+)"
    r"(?:[eE][+-]?\d+)?)?%?"
)
_GSM8K_MARKED_ANSWER_RE = re.compile(
    rf"####\s*({_NUMERIC_TOKEN_PATTERN})",
    flags=re.IGNORECASE,
)
_GSM8K_ANSWER_PHRASE_RE = re.compile(
    r"(?:final\s+answer|answer)\s*(?:is|:|=)\s*"
    rf"({_NUMERIC_TOKEN_PATTERN})",
    flags=re.IGNORECASE,
)
_NUMBER_RE = re.compile(_NUMERIC_TOKEN_PATTERN)


def normalize_numeric_answer(text: str) -> str | None:
    value = str(text).strip().replace(",", "").replace("%", "")
    if not value:
        return None
    if "/" in value:
        numerator_text, denominator_text = value.split("/", 1)
        try:
            numerator = Decimal(numerator_text.strip())
            denominator = Decimal(denominator_text.strip())
        except InvalidOperation:
            return None
        if not numerator.is_finite() or not denominator.is_finite() or denominator == 0:
            return None
        number = numerator / denominator
    else:
        try:
            number = Decimal(value)
        except InvalidOperation:
            return None
    if not number.is_finite():
        return None
    normalized = format(number.normalize(), "f")
    if "." in normalized:
        normalized = normalized.rstrip("0").rstrip(".")
    return "0" if normalized in {"-0", "+0", ""} else normalized


def extract_gsm8k_final_answer(text: str) -> str | None:
    """Extract a normalized final number from a GSM8K-style completion."""
    text = strip_generation_envelope(text)
    marked = _GSM8K_MARKED_ANSWER_RE.findall(text)
    if marked:
        return normalize_numeric_answer(marked[-1])
    phrased = _GSM8K_ANSWER_PHRASE_RE.findall(text)
    if phrased:
        return normalize_numeric_answer(phrased[-1])
    numbers = _NUMBER_RE.findall(text)
    return normalize_numeric_answer(numbers[-1]) if numbers else None


def extract_numeric_answer(text: str) -> str | None:
    """Extract the final normalized number from a short numeric-QA response."""
    text = strip_generation_envelope(text)
    phrased = _GSM8K_ANSWER_PHRASE_RE.findall(text)
    if phrased:
        return normalize_numeric_answer(phrased[-1])
    numbers = _NUMBER_RE.findall(text)
    return normalize_numeric_answer(numbers[-1]) if numbers else None


_SHORT_ANSWER_PHRASE_RE = re.compile(
    r"(?:final\s+answer|answer)\s*(?:is|:|=)\s*([^\n]+)",
    flags=re.IGNORECASE,
)


def _normalize_numeric_target(text: str) -> str | None:
    value = strip_generation_envelope(text).strip().strip("$£€")
    if _NUMBER_RE.fullmatch(value) is None:
        return None
    return normalize_numeric_answer(value)


def canonicalize_numglue_answer(text: str) -> str | None:
    """Canonicalize a NumGLUE answer while preserving nonnumeric answers."""
    value = strip_generation_envelope(text).strip()
    if not value:
        return None
    numeric = _normalize_numeric_target(value)
    return numeric if numeric is not None else value


def extract_numglue_short_answer(text: str) -> str | None:
    """Extract a concise number, date, name, or phrase from a model response."""
    value = strip_generation_envelope(text)
    phrased = _SHORT_ANSWER_PHRASE_RE.findall(value)
    if phrased:
        value = phrased[-1]
    else:
        lines = [line.strip() for line in value.splitlines() if line.strip()]
        if not lines:
            return None
        value = lines[-1]
    value = re.sub(r"^####\s*", "", value).strip().strip("`\"'")
    value = value.rstrip(".!,;:").strip()
    return canonicalize_numglue_answer(value)


def normalize_short_text_answer(text: str) -> str:
    value = str(text).strip().casefold()
    return re.sub(r"\s+", " ", value)


def task_answer_match(
    prediction: str,
    target: str,
    metadata: Mapping[str, Any] | None = None,
) -> bool:
    """Evaluate a free-form answer using the row's declared functional matcher."""
    metadata = metadata or {}
    default_matcher = (
        "numglue_short_answer"
        if str(metadata.get("dataset") or "").strip().lower()
        in {"numglue_cm", "numglue_ds"}
        else "exact"
    )
    matcher = str(metadata.get("answer_match") or default_matcher).strip().lower()
    if matcher == "exact":
        return exact_answer_match(prediction, target)
    if matcher == "gsm8k_final_number":
        predicted_answer = extract_gsm8k_final_answer(prediction)
        target_source = str((metadata or {}).get("gold_answer") or target)
        target_answer = extract_gsm8k_final_answer(target_source)
        return predicted_answer is not None and predicted_answer == target_answer
    if matcher == "numeric":
        predicted_answer = extract_numeric_answer(prediction)
        target_source = str(metadata.get("gold_answer") or target)
        target_answer = extract_numeric_answer(target_source)
        return predicted_answer is not None and predicted_answer == target_answer
    if matcher == "numglue_short_answer":
        target_source = str(metadata.get("gold_answer") or target)
        target_numeric = _normalize_numeric_target(target_source)
        if target_numeric is not None:
            predicted_numeric = extract_numeric_answer(prediction)
            return predicted_numeric is not None and predicted_numeric == target_numeric
        predicted_answer = extract_numglue_short_answer(prediction)
        target_answer = canonicalize_numglue_answer(target_source)
        return (
            predicted_answer is not None
            and target_answer is not None
            and normalize_short_text_answer(predicted_answer)
            == normalize_short_text_answer(target_answer)
        )
    raise ValueError(f"unsupported answer_match={matcher!r}")


def classification_prompt(
    dataset: str,
    text: str,
    *,
    title: str = "",
) -> dict[str, Any]:
    if dataset not in CLASSIFICATION_LABELS:
        raise ValueError(f"unknown classification dataset: {dataset}")
    names = CLASSIFICATION_LABELS[dataset]
    labels = choice_labels(len(names))
    choices = "\n".join(
        f"{label}: {name}" for label, name in zip(labels, names)
    )
    if dataset == "ag_news":
        payload_name = "Article"
        payload = text.strip()
        answer_cue = "Category:"
        instruction = "Classify the news article into one category."
    elif dataset == "yahoo_answers":
        payload_name = "Question and answer"
        payload = "\n".join(part for part in (title.strip(), text.strip()) if part)
        answer_cue = "Topic:"
        instruction = "Classify the Yahoo Answers item into one topic."
    else:
        payload_name = "Entity description"
        payload = "\n".join(part for part in (title.strip(), text.strip()) if part)
        answer_cue = "Type:"
        instruction = "Classify the entity into one DBPedia type."
    question = (
        f"{instruction}\n"
        f"Categories:\n{choices}\n\n"
        f"{payload_name}:\n{payload}\n\n"
        f"Reply with only {', '.join(labels)}.\n"
        f"{answer_cue}"
    )
    return {
        "question": question,
        "candidate_labels": list(labels),
        "answer_cue": answer_cue,
    }


def canonicalize_classification_row(
    dataset: str,
    source_row: dict[str, Any],
    *,
    source_index: int,
) -> dict[str, Any]:
    raw_label = source_row.get("label", source_row.get("topic"))
    if raw_label is None:
        raise ValueError(f"{dataset} row {source_index} has no label")
    label_index = int(raw_label)
    labels = choice_labels(len(CLASSIFICATION_LABELS[dataset]))
    if not 0 <= label_index < len(labels):
        raise ValueError(f"{dataset} label is out of range: {label_index}")
    if dataset == "ag_news":
        prompt = classification_prompt(dataset, str(source_row["text"]))
    elif dataset == "yahoo_answers":
        title = str(source_row.get("question_title") or source_row.get("title") or "")
        body_parts = (
            source_row.get("question_content"),
            source_row.get("best_answer"),
            source_row.get("text"),
        )
        body_values: list[str] = []
        for part in body_parts:
            value = str(part).strip() if part else ""
            if value and value != title.strip() and value not in body_values:
                body_values.append(value)
        body = "\n".join(body_values)
        prompt = classification_prompt(dataset, body, title=title)
    elif dataset == "dbpedia":
        prompt = classification_prompt(
            dataset,
            str(source_row.get("content") or source_row.get("text") or ""),
            title=str(source_row.get("title") or ""),
        )
    else:
        raise ValueError(f"unknown classification dataset: {dataset}")
    return {
        "dataset": dataset,
        "source_index": source_index,
        "task_type": TASK_CATEGORICAL,
        "task_family": "text_classification",
        "label": labels[label_index],
        **prompt,
    }


def canonicalize_trace_row(
    dataset: str,
    source_row: dict[str, Any],
    *,
    source_index: int,
) -> dict[str, Any]:
    question = str(source_row.get("prompt") or "").strip()
    answer = str(source_row.get("answer") or "").strip()
    if not question or not answer:
        raise ValueError(f"{dataset} row {source_index} lacks prompt/answer")
    if dataset == "c_stance":
        labels = ("A", "B", "C")
        if answer not in labels:
            raise ValueError(f"C-STANCE answer is not A/B/C: {answer!r}")
        return {
            "dataset": dataset,
            "source_index": source_index,
            "task_type": TASK_CATEGORICAL,
            "task_family": "stance_classification",
            "question": question,
            "label": answer,
            "candidate_labels": list(labels),
            "answer_cue": "态度：",
        }
    if dataset not in {"numglue_cm", "numglue_ds"}:
        raise ValueError(f"unknown TRACE dataset: {dataset}")
    normalized_answer = extract_numglue_short_answer(answer)
    if normalized_answer is None:
        raise ValueError(f"{dataset} row {source_index} has an empty answer")
    return {
        "dataset": dataset,
        "source_index": source_index,
        "task_type": TASK_GENERATION,
        "task_family": "numeric_reasoning",
        "question": question,
        "answer": normalized_answer,
        "label": normalized_answer,
        "answer_cue": "Answer:",
        "answer_match": "numglue_short_answer",
        "gold_answer": normalized_answer,
    }


def canonicalize_gsm8k_row(
    source_row: dict[str, Any],
    *,
    source_index: int,
) -> dict[str, Any]:
    problem = str(source_row.get("question") or "").strip()
    raw_solution = str(source_row.get("answer") or "").strip()
    if not problem or not raw_solution:
        raise ValueError(f"gsm8k row {source_index} lacks question/answer")
    gold_answer = extract_gsm8k_final_answer(raw_solution)
    if gold_answer is None:
        raise ValueError(f"gsm8k row {source_index} has no parseable final answer")
    # Calculator annotations are supervision artifacts, not part of the desired
    # natural-language trajectory.
    solution = re.sub(r"<<[^<>]*>>", "", raw_solution).strip()
    question = (
        "Solve the following problem. Show the reasoning needed to reach the "
        "answer. End the response with a separate line in the exact format "
        "#### <number>.\n\n"
        f"Problem:\n{problem}\n\n"
        "Solution:"
    )
    return {
        "dataset": "gsm8k",
        "source_index": source_index,
        "task_type": TASK_GENERATION,
        "task_family": "longform_math_reasoning",
        "question": question,
        "answer": solution,
        "label": solution,
        "answer_cue": "Solution:",
        "answer_match": "gsm8k_final_number",
        "gold_answer": gold_answer,
    }


def strip_generation_envelope(text: str) -> str:
    text = str(text).strip()
    text = re.sub(r"(?is)^.*?</think>\s*", "", text, count=1)
    text = re.sub(r"(?is)^```(?:text|markdown)?\s*", "", text)
    text = re.sub(r"(?is)\s*```$", "", text)
    return text.strip()


def parse_task_icl_prompt(text: str, answer_cue: str) -> str | None:
    text = strip_generation_envelope(text)
    cue_index = text.rfind(answer_cue)
    if cue_index < 0:
        return None
    prompt = text[: cue_index + len(answer_cue)].strip()
    if len(prompt.split()) < 5:
        return None
    return prompt


def make_task_icl_generation_prompt(
    examples: list[dict[str, Any]],
    *,
    answer_cue: str,
) -> str:
    if not examples:
        raise ValueError("task ICL generation requires at least one example")
    rendered = []
    for index, example in enumerate(examples, start=1):
        rendered.append(
            f"EXAMPLE {index}\n{example['question']}\n{example['label']}"
        )
    return (
        "Create one new and distinct task example matching the format, domain, "
        "and difficulty of the demonstrations below. Return only the complete "
        "unanswered prompt. It must end exactly at the answer cue "
        f"{answer_cue!r}; do not provide an answer.\n\n"
        + "\n\n".join(rendered)
        + "\n\nNEW UNANSWERED PROMPT\n"
    )


_NUMERIC_PROBLEM_BLOCK_RE = re.compile(
    r"BEGIN_PROBLEM\s*(.*?)\s*END_PROBLEM",
    flags=re.IGNORECASE | re.DOTALL,
)
_NUMERIC_ANSWER_BLOCK_RE = re.compile(
    r"BEGIN_ANSWER\s*(.*?)\s*END_ANSWER",
    flags=re.IGNORECASE | re.DOTALL,
)


def parse_task_icl_short_answer_pair(
    text: str,
    answer_cue: str,
) -> tuple[str, str] | None:
    """Parse a generated problem and its independently declared short answer."""
    text = strip_generation_envelope(text)
    problem_match = _NUMERIC_PROBLEM_BLOCK_RE.search(text)
    answer_match = _NUMERIC_ANSWER_BLOCK_RE.search(text)
    if problem_match is None or answer_match is None:
        return None
    problem = parse_task_icl_prompt(problem_match.group(1), answer_cue)
    declared_answer = extract_numglue_short_answer(answer_match.group(1))
    if problem is None or declared_answer is None:
        return None
    return problem, declared_answer


def make_task_icl_short_answer_pair_generation_prompt(
    examples: list[dict[str, Any]],
    *,
    answer_cue: str,
) -> str:
    """Ask for one new NumGLUE-style problem plus a declared short answer."""
    if not examples:
        raise ValueError("short-answer task ICL generation requires at least one example")
    rendered = []
    for index, example in enumerate(examples, start=1):
        answer = canonicalize_numglue_answer(str(example["label"]))
        if answer is None:
            raise ValueError("task ICL demonstration has no short answer")
        rendered.append(
            f"EXAMPLE {index}\n{example['question']}\n{answer}"
        )
    return (
        "Create one new and distinct reasoning problem matching the format, "
        "domain, and difficulty of the demonstrations. Compute its answer yourself. "
        "The problem must be self-contained and its unanswered prompt must end exactly "
        f"at the answer cue {answer_cue!r}. Return exactly these four markers, with no "
        "text outside them:\n"
        "BEGIN_PROBLEM\n"
        f"<complete unanswered prompt ending in {answer_cue}>\n"
        "END_PROBLEM\n"
        "BEGIN_ANSWER\n"
        "<short answer only: a number, date, name, or brief phrase>\n"
        "END_ANSWER\n\n"
        + "\n\n".join(rendered)
        + "\n\nNEW EXAMPLE\n"
    )


def jsonl_sha256(rows: Iterable[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(
            json.dumps(
                row,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count
