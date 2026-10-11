"""Forward-only causal FFN-neuron routing for prompt-invariant anchors."""

from __future__ import annotations

from typing import Any, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from jax_sft.cnl import tree_l2_norm
from jax_sft.ffn_memory import FfnMemorySlotLayout


def _percentile_ranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("percentile ranks require a vector")
    order = np.argsort(values, kind="stable")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(values.size, dtype=np.float64)
    return ranks / max(values.size - 1, 1)


def select_causal_routing_candidates(
    injection_activity: np.ndarray,
    anchor_family_activity: np.ndarray,
    *,
    candidates_per_layer: int,
    free_fraction: float,
    anchor_quantile: float,
) -> tuple[np.ndarray, tuple[str, ...], dict[str, Any]]:
    """Shortlist injection-relevant free and overlap FFN neurons.

    ``anchor_family_activity`` contains one prompt-invariant activity vector
    per semantic anchor family. Each family should already have been reduced
    conservatively across its retained prompt views.
    """
    injection = np.asarray(injection_activity, dtype=np.float64)
    anchor_families = np.asarray(anchor_family_activity, dtype=np.float64)
    if injection.ndim != 2:
        raise ValueError("injection activity must have [layer, neuron] shape")
    if (
        anchor_families.ndim != 3
        or anchor_families.shape[1:] != injection.shape
        or anchor_families.shape[0] == 0
    ):
        raise ValueError(
            "anchor family activity must have [family, layer, neuron] shape"
        )
    if candidates_per_layer <= 1:
        raise ValueError("causal routing needs at least two candidates per layer")
    if candidates_per_layer > injection.shape[1]:
        raise ValueError("candidates per layer exceed the FFN width")
    if not 0 < free_fraction < 1:
        raise ValueError("causal routing free fraction must be in (0, 1)")
    if not 0 <= anchor_quantile <= 1:
        raise ValueError("anchor activity quantile must be in [0, 1]")
    if not np.all(np.isfinite(injection)) or not np.all(
        np.isfinite(anchor_families)
    ):
        raise ValueError("causal routing activities must be finite")

    anchor = np.quantile(
        anchor_families,
        anchor_quantile,
        axis=0,
    )
    free_count = int(round(candidates_per_layer * free_fraction))
    free_count = min(max(free_count, 1), candidates_per_layer - 1)
    overlap_count = candidates_per_layer - free_count
    candidates: list[tuple[int, int]] = []
    strata: list[str] = []
    selected_scores: list[float] = []
    selected_injection_ranks: list[float] = []
    selected_anchor_ranks: list[float] = []

    for layer in range(injection.shape[0]):
        injection_rank = _percentile_ranks(injection[layer])
        anchor_rank = _percentile_ranks(anchor[layer])
        free_score = injection_rank * (1.0 - anchor_rank)
        overlap_score = injection_rank * anchor_rank

        free_indices = np.argsort(free_score, kind="stable")[-free_count:][::-1]
        selected = set(int(index) for index in free_indices)
        overlap_order = np.argsort(overlap_score, kind="stable")[::-1]
        overlap_indices = [
            int(index)
            for index in overlap_order
            if int(index) not in selected
        ][:overlap_count]
        if len(overlap_indices) != overlap_count:
            raise RuntimeError("causal routing candidate selection underflow")

        for index in free_indices:
            neuron = int(index)
            candidates.append((layer, neuron))
            strata.append("free")
            selected_scores.append(float(free_score[neuron]))
            selected_injection_ranks.append(float(injection_rank[neuron]))
            selected_anchor_ranks.append(float(anchor_rank[neuron]))
        for neuron in overlap_indices:
            candidates.append((layer, neuron))
            strata.append("overlap")
            selected_scores.append(float(overlap_score[neuron]))
            selected_injection_ranks.append(float(injection_rank[neuron]))
            selected_anchor_ranks.append(float(anchor_rank[neuron]))

    indices = np.asarray(candidates, dtype=np.int32)
    metadata = {
        "layer_count": int(injection.shape[0]),
        "ffn_width": int(injection.shape[1]),
        "candidate_count": int(indices.shape[0]),
        "candidates_per_layer": int(candidates_per_layer),
        "free_candidate_count": int(sum(value == "free" for value in strata)),
        "overlap_candidate_count": int(
            sum(value == "overlap" for value in strata)
        ),
        "free_fraction": float(free_fraction),
        "anchor_quantile": float(anchor_quantile),
        "candidate_scores": selected_scores,
        "candidate_injection_percentiles": selected_injection_ranks,
        "candidate_anchor_percentiles": selected_anchor_ranks,
        "anchor_activity_mean": float(np.mean(anchor)),
        "injection_activity_mean": float(np.mean(injection)),
    }
    return indices, tuple(strata), metadata


def make_balanced_causal_ablation_patterns(
    candidate_count: int,
    pattern_count: int,
    seed: int,
) -> np.ndarray:
    """Create complementary randomized interventions.

    Every candidate is ablated in exactly half of the patterns. Complementary
    pairs reduce main-effect variance while retaining group interactions.
    """
    if candidate_count <= 1:
        raise ValueError("causal routing needs at least two candidates")
    if pattern_count < 4 or pattern_count % 2:
        raise ValueError(
            "causal routing pattern count must be an even integer at least four"
        )
    rng = np.random.default_rng(seed)
    half = pattern_count // 2
    patterns = np.zeros((pattern_count, candidate_count), dtype=np.float32)
    active_count = max(1, candidate_count // 2)
    for row in range(half):
        active = rng.choice(
            candidate_count,
            size=active_count,
            replace=False,
        )
        patterns[2 * row, active] = 1.0
        patterns[2 * row + 1] = 1.0 - patterns[2 * row]
    return patterns[rng.permutation(pattern_count)]


def materialize_causal_ablation_scales(
    patterns: np.ndarray,
    candidate_indices: np.ndarray,
    *,
    layer_count: int,
    ffn_width: int,
    ablation_strength: float,
) -> np.ndarray:
    """Expand compact group-test patterns into FFN coefficient multipliers."""
    patterns = np.asarray(patterns, dtype=np.float32)
    candidates = np.asarray(candidate_indices, dtype=np.int32)
    if patterns.ndim != 2:
        raise ValueError("causal routing patterns must be a matrix")
    if candidates.shape != (patterns.shape[1], 2):
        raise ValueError("candidate indices do not match causal patterns")
    if not 0 < ablation_strength <= 1:
        raise ValueError("causal ablation strength must be in (0, 1]")
    if np.any(candidates[:, 0] < 0) or np.any(candidates[:, 0] >= layer_count):
        raise ValueError("causal candidate layer index is out of range")
    if np.any(candidates[:, 1] < 0) or np.any(candidates[:, 1] >= ffn_width):
        raise ValueError("causal candidate neuron index is out of range")

    scales = np.ones(
        (patterns.shape[0], layer_count, ffn_width),
        dtype=np.float32,
    )
    for candidate_index, (layer, neuron) in enumerate(candidates):
        scales[:, layer, neuron] = (
            1.0 - ablation_strength * patterns[:, candidate_index]
        )
    return scales


def estimate_prompt_invariant_causal_effects(
    patterns: np.ndarray,
    family_margin_drops: np.ndarray,
    *,
    confidence_z: float,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Estimate randomized ablation effects and conservative uncertainty."""
    patterns = np.asarray(patterns, dtype=np.float64)
    drops = np.asarray(family_margin_drops, dtype=np.float64)
    if patterns.ndim != 2:
        raise ValueError("causal routing patterns must be a matrix")
    if drops.ndim != 2 or drops.shape[0] != patterns.shape[0]:
        raise ValueError(
            "family margin drops must have [pattern, family] shape"
        )
    if drops.shape[1] == 0:
        raise ValueError("causal routing requires at least one anchor family")
    if confidence_z < 0:
        raise ValueError("causal routing confidence z must be nonnegative")

    candidate_count = patterns.shape[1]
    family_count = drops.shape[1]
    family_effects = np.empty(
        (candidate_count, family_count),
        dtype=np.float64,
    )
    pattern_variances = np.empty_like(family_effects)
    for candidate in range(candidate_count):
        active = patterns[:, candidate] > 0.5
        inactive = ~active
        if not np.any(active) or not np.any(inactive):
            raise ValueError("every causal candidate needs both interventions")
        active_values = drops[active]
        inactive_values = drops[inactive]
        family_effects[candidate] = (
            active_values.mean(axis=0) - inactive_values.mean(axis=0)
        )
        active_var = (
            active_values.var(axis=0, ddof=1)
            if active_values.shape[0] > 1
            else np.zeros((family_count,), dtype=np.float64)
        )
        inactive_var = (
            inactive_values.var(axis=0, ddof=1)
            if inactive_values.shape[0] > 1
            else np.zeros((family_count,), dtype=np.float64)
        )
        pattern_variances[candidate] = (
            active_var / active_values.shape[0]
            + inactive_var / inactive_values.shape[0]
        )

    effect_mean = family_effects.mean(axis=1)
    between_family_se = (
        family_effects.std(axis=1, ddof=1) / np.sqrt(family_count)
        if family_count > 1
        else np.zeros((candidate_count,), dtype=np.float64)
    )
    randomized_pattern_se = np.sqrt(
        np.mean(pattern_variances, axis=1) / family_count
    )
    standard_error = np.sqrt(
        np.square(between_family_se) + np.square(randomized_pattern_se)
    )
    lower_confidence_bound = effect_mean - confidence_z * standard_error
    positive_family_fraction = np.mean(family_effects > 0, axis=1)

    centered = drops - drops.mean(axis=0, keepdims=True)
    interaction_dispersion = float(np.sqrt(np.mean(np.square(centered))))
    effects = {
        "effect_mean": effect_mean.astype(np.float32),
        "standard_error": standard_error.astype(np.float32),
        "lower_confidence_bound": lower_confidence_bound.astype(np.float32),
        "positive_family_fraction": positive_family_fraction.astype(np.float32),
        "family_effects": family_effects.astype(np.float32),
    }
    stats = {
        "causal_routing/effect_mean": float(np.mean(effect_mean)),
        "causal_routing/effect_max": float(
            np.max(effect_mean, initial=0.0)
        ),
        "causal_routing/lcb_positive_fraction": float(
            np.mean(lower_confidence_bound > 0)
        ),
        "causal_routing/positive_family_fraction_mean": float(
            np.mean(positive_family_fraction)
        ),
        "causal_routing/pattern_family_margin_drop_mean": float(
            np.mean(drops)
        ),
        "causal_routing/pattern_family_margin_drop_max": float(
            np.max(drops, initial=0.0)
        ),
        "causal_routing/interaction_dispersion": interaction_dispersion,
    }
    return effects, stats


def build_causal_neuron_gates(
    candidate_indices: np.ndarray,
    effects: dict[str, np.ndarray],
    *,
    layer_count: int,
    ffn_width: int,
    min_gate: float,
    strength: float,
    min_family_fraction: float,
    min_effect: float,
    scale_quantile: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Turn conservative causal effects into static neuron-triplet gates."""
    candidates = np.asarray(candidate_indices, dtype=np.int32)
    effect_mean = np.asarray(effects["effect_mean"], dtype=np.float64)
    lcb = np.asarray(
        effects["lower_confidence_bound"],
        dtype=np.float64,
    )
    positive_fraction = np.asarray(
        effects["positive_family_fraction"],
        dtype=np.float64,
    )
    if candidates.shape != (effect_mean.size, 2):
        raise ValueError("causal effects do not match candidate indices")
    if lcb.shape != effect_mean.shape or positive_fraction.shape != effect_mean.shape:
        raise ValueError("causal effect vectors have inconsistent shapes")
    if not 0 <= min_gate <= 1:
        raise ValueError("causal routing minimum gate must be in [0, 1]")
    if not 0 <= strength <= 1:
        raise ValueError("causal routing strength must be in [0, 1]")
    if not 0 <= min_family_fraction <= 1:
        raise ValueError(
            "causal routing minimum family fraction must be in [0, 1]"
        )
    if min_effect < 0:
        raise ValueError("causal routing minimum effect must be nonnegative")
    if not 0 < scale_quantile <= 1:
        raise ValueError("causal routing scale quantile must be in (0, 1]")

    protected = (lcb > min_effect) & (
        positive_fraction >= min_family_fraction
    )
    positive_lcb = np.where(protected, lcb, 0.0)
    positive_values = positive_lcb[protected]
    scale = (
        float(np.quantile(positive_values, scale_quantile))
        if positive_values.size
        else 1.0
    )
    scale = max(scale, 1e-12)
    protection = np.clip(positive_lcb / scale, 0.0, 1.0)
    candidate_gates = np.clip(
        1.0 - strength * protection,
        min_gate,
        1.0,
    )
    gates = np.ones((layer_count, ffn_width), dtype=np.float32)
    for candidate_index, (layer, neuron) in enumerate(candidates):
        gates[layer, neuron] = float(candidate_gates[candidate_index])

    metadata = {
        "protected_candidate_count": int(np.sum(protected)),
        "candidate_count": int(candidates.shape[0]),
        "protected_candidate_fraction": float(np.mean(protected)),
        "candidate_gate_mean": float(np.mean(candidate_gates)),
        "candidate_gate_min": float(
            np.min(candidate_gates, initial=1.0)
        ),
        "candidate_gate_fraction_lt_0_5": float(
            np.mean(candidate_gates < 0.5)
        ),
        "whole_ffn_gate_mean": float(np.mean(gates)),
        "effect_scale": scale,
        "min_gate": float(min_gate),
        "strength": float(strength),
        "min_family_fraction": float(min_family_fraction),
        "min_effect": float(min_effect),
        "scale_quantile": float(scale_quantile),
        "protected_mask": protected.astype(bool).tolist(),
        "candidate_gates": candidate_gates.astype(float).tolist(),
        "candidate_effect_mean": effect_mean.astype(float).tolist(),
        "candidate_lcb": lcb.astype(float).tolist(),
        "candidate_positive_family_fraction": (
            positive_fraction.astype(float).tolist()
        ),
    }
    return gates, metadata


def apply_causal_neuron_gates(
    updates: dict[str, jax.Array],
    layouts: Sequence[FfnMemorySlotLayout],
    neuron_gates: jax.Array,
) -> tuple[dict[str, jax.Array], dict[str, jax.Array]]:
    """Apply one coherent gate to gate/up rows and the matching down column."""
    gates = jnp.asarray(neuron_gates, dtype=jnp.float32)
    if gates.ndim != 2 or gates.shape[0] != len(layouts):
        raise ValueError(
            "causal neuron gates must have [layer, intermediate] shape"
        )
    result = dict(updates)
    removed_energy = jnp.asarray(0.0, dtype=jnp.float32)
    raw_ffn_energy = jnp.asarray(0.0, dtype=jnp.float32)
    for layer_index, layout in enumerate(layouts):
        if gates.shape[1] != layout.slot_count:
            raise ValueError(
                f"causal gate width {gates.shape[1]} does not match "
                f"layer {layout.layer} width {layout.slot_count}"
            )
        layer_gate = gates[layer_index]
        for path, axis in (
            (layout.gate_path, layout.gate_slot_axis),
            (layout.up_path, layout.up_slot_axis),
            (layout.down_path, layout.down_slot_axis),
        ):
            update = updates[path]
            shape = [1] * update.ndim
            shape[axis] = layer_gate.shape[0]
            broadcast_gate = layer_gate.reshape(shape).astype(update.dtype)
            gated = update * broadcast_gate
            result[path] = gated
            raw_value = jnp.asarray(update, dtype=jnp.float32)
            removed_value = raw_value - jnp.asarray(gated, dtype=jnp.float32)
            raw_ffn_energy = raw_ffn_energy + jnp.sum(jnp.square(raw_value))
            removed_energy = removed_energy + jnp.sum(
                jnp.square(removed_value)
            )

    raw_norm = tree_l2_norm(updates)
    final_norm = tree_l2_norm(result)
    stats = {
        "causal_routing/gate_mean": jnp.mean(gates),
        "causal_routing/gate_min": jnp.min(gates),
        "causal_routing/gate_fraction_lt_0_5": jnp.mean(
            (gates < 0.5).astype(jnp.float32)
        ),
        "causal_routing/ffn_removed_energy_fraction": (
            removed_energy / jnp.maximum(raw_ffn_energy, 1e-30)
        ),
        "update_norm/raw": raw_norm,
        "update_norm/final_structured": final_norm,
        "update_norm/kept_fraction": (
            final_norm / jnp.maximum(raw_norm, 1e-30)
        ),
    }
    return result, stats
