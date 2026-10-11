"""Synthetic Anchor-ICL scheduling, permutation, and embedding utilities."""

from __future__ import annotations

import math
import re
from typing import Any, Callable

import numpy as np
from tqdm import tqdm


def normalized_embedding(value: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    return value / max(float(np.linalg.norm(value)), eps)


def frozen_batch_embeddings(
    weights: Any,
    batches: list[dict[str, Any]],
    encode_fn: Any,
    batch_to_arrays: Callable[[dict[str, Any]], dict[str, Any]],
    *,
    desc: str,
) -> np.ndarray:
    """Pool final frozen-model hidden states over non-padding tokens."""
    embeddings = []
    for batch in tqdm(batches, desc=desc):
        hidden = np.asarray(encode_fn(weights, batch_to_arrays(batch)), dtype=np.float32)[0]
        mask = np.asarray(batch["attention_mask"], dtype=np.float32)[0]
        pooled = (hidden * mask[:, None]).sum(axis=0) / max(float(mask.sum()), 1.0)
        embeddings.append(normalized_embedding(pooled))
    if not embeddings:
        return np.zeros((0, 0), dtype=np.float32)
    return np.stack(embeddings, axis=0)


def choose_farthest_indices(
    candidate_indices: list[int],
    k: int,
    rng: np.random.Generator,
    embeddings: np.ndarray,
) -> list[int]:
    if k <= 0:
        return []
    if k >= len(candidate_indices):
        return list(candidate_indices)
    first = int(rng.choice(candidate_indices))
    selected = [first]
    remaining = set(candidate_indices) - {first}
    while len(selected) < k:
        best_distance = -math.inf
        best = []
        for idx in remaining:
            distance = min(
                1.0 - float(np.dot(embeddings[idx], embeddings[chosen]))
                for chosen in selected
            )
            if distance > best_distance + 1e-12:
                best_distance = distance
                best = [idx]
            elif abs(distance - best_distance) <= 1e-12:
                best.append(idx)
        chosen = int(rng.choice(best))
        selected.append(chosen)
        remaining.remove(chosen)
    return selected


def choose_balanced_demo_indices(
    exposure: np.ndarray,
    k: int,
    rng: np.random.Generator,
    *,
    embeddings: np.ndarray | None = None,
    used_combinations: set[tuple[int, ...]] | None = None,
) -> list[int]:
    """Choose a low-exposure, non-repeating, optionally diverse anchor set."""
    n = int(exposure.shape[0])
    if k <= 0 or n == 0:
        return []
    k = min(k, n)
    best_indices = None
    best_score = None
    for _ in range(min(64, max(8, n * 2))):
        selected: list[int] = []
        while len(selected) < k:
            remaining = [idx for idx in range(n) if idx not in selected]
            min_exposure = min(float(exposure[idx]) for idx in remaining)
            candidates = [idx for idx in remaining if float(exposure[idx]) <= min_exposure]
            if embeddings is not None and selected:
                distances = {
                    idx: min(
                        1.0 - float(np.dot(embeddings[idx], embeddings[chosen]))
                        for chosen in selected
                    )
                    for idx in candidates
                }
                max_distance = max(distances.values())
                candidates = [
                    idx for idx in candidates if abs(distances[idx] - max_distance) <= 1e-12
                ]
            selected.append(int(rng.choice(candidates)))
        combo = tuple(sorted(selected))
        repeated = 1 if used_combinations is not None and combo in used_combinations else 0
        diversity = 0.0
        if embeddings is not None and len(selected) > 1:
            diversity = min(
                1.0 - float(np.dot(embeddings[a], embeddings[b]))
                for pos, a in enumerate(selected)
                for b in selected[pos + 1 :]
            )
        score = (repeated, max(float(exposure[idx]) for idx in selected), -diversity)
        if best_score is None or score < best_score:
            best_indices = selected
            best_score = score
        if repeated == 0:
            break
    assert best_indices is not None
    exposure[best_indices] += 1
    if used_combinations is not None:
        used_combinations.add(tuple(sorted(best_indices)))
    return best_indices


def split_mcq_options(question: str) -> tuple[str, list[str]] | None:
    """Extract the stem and semantic A-D choices from a normalized MCQ prompt."""
    lines = question.replace("\r\n", "\n").replace("\r", "\n").splitlines()
    stem_lines = []
    options: dict[str, str] = {}
    current = None
    for raw in lines:
        line = raw.strip()
        match = re.match(r"^([A-D])[\.:\)]\s*(.+)$", line)
        if match:
            current = match.group(1)
            options[current] = match.group(2).strip()
            continue
        if current is not None and line and not re.match(
            r"(?i)^(?:please reply|answer with|respond with|return only)", line
        ):
            options[current] = f"{options[current]} {line}".strip()
        elif not options and line.lower() not in {"options:", "options"}:
            stem_lines.append(line)
    if set(options) != set("ABCD"):
        return None
    stem = "\n".join(stem_lines).strip()
    if stem.lower().startswith("question:"):
        stem = stem.split(":", 1)[1].strip()
    if not stem:
        return None
    return stem, [options[label] for label in "ABCD"]


def permute_mcq_choices(question: str, permutation: np.ndarray) -> tuple[str, np.ndarray] | None:
    parsed = split_mcq_options(question)
    if parsed is None:
        return None
    stem, options = parsed
    permutation = np.asarray(permutation, dtype=np.int32)
    if sorted(permutation.tolist()) != [0, 1, 2, 3]:
        raise ValueError(f"invalid choice permutation: {permutation.tolist()}")
    permuted = [options[int(original_idx)] for original_idx in permutation]
    text = (
        f"Question: {stem}\n"
        "Options:\n"
        f"A: {permuted[0]}\n"
        f"B: {permuted[1]}\n"
        f"C: {permuted[2]}\n"
        f"D: {permuted[3]}\n\n"
        "Please reply with only A, B, C, or D. Do not provide any explanation."
    )
    return text, permutation


def filter_synthetic_rows_by_anchor_support(
    rows: list[dict[str, Any]],
    tokenizer: Any,
    weights: Any,
    encode_fn: Any,
    tokenize_prompt_fn: Callable[[Any, str, int], dict[str, Any] | None],
    anchor_embeddings: np.ndarray,
    max_length: int,
    *,
    enabled: bool,
    similarity_min: float,
    similarity_max: float,
    dedup_similarity_max: float,
    stats_out: dict[str, float],
) -> list[dict[str, Any]]:
    if anchor_embeddings.size == 0:
        raise ValueError("anchor-support filtering requires anchor embeddings")
    accepted = []
    accepted_embeddings: list[np.ndarray] = []
    similarities = []
    rejected = 0
    for row in tqdm(rows, desc="synthetic anchor-support audit"):
        batch = tokenize_prompt_fn(tokenizer, row.get("question", ""), max_length)
        if batch is None:
            rejected += 1
            continue
        hidden = np.asarray(encode_fn(weights, batch), dtype=np.float32)[0]
        mask = np.asarray(batch["attention_mask"], dtype=np.float32)[0]
        embedding = normalized_embedding(
            (hidden * mask[:, None]).sum(axis=0) / max(float(mask.sum()), 1.0)
        )
        nearest_anchor = float(np.clip(np.max(anchor_embeddings @ embedding), -1.0, 1.0))
        nearest_synth = (
            max(
                float(np.clip(np.dot(previous, embedding), -1.0, 1.0))
                for previous in accepted_embeddings
            )
            if accepted_embeddings
            else -1.0
        )
        similarities.append(nearest_anchor)
        keep = (
            not enabled
            or (
                similarity_min <= nearest_anchor <= similarity_max
                and nearest_synth <= dedup_similarity_max
            )
        )
        if keep:
            accepted.append(row)
            accepted_embeddings.append(embedding)
        else:
            rejected += 1
    values = np.asarray(similarities, dtype=np.float32)
    stats_out.update(
        {
            "synthetic/anchor_similarity_mean": float(values.mean()) if values.size else 0.0,
            "synthetic/anchor_similarity_min": float(values.min()) if values.size else 0.0,
            "synthetic/anchor_similarity_max": float(values.max()) if values.size else 0.0,
            "synthetic/anchor_support_rejected": float(rejected),
        }
    )
    return accepted
