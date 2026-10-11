"""Curvature-based alternatives to first-order elementwise CPL masks."""

from __future__ import annotations

import math
from typing import Any

import jax
import jax.numpy as jnp

from jax_sft.cnl import tree_l2_norm, tree_sum
from jax_sft.ffn_memory import tree_mean, update_mask_ema


PyTree = Any


def upper_cvar_group_curvature(
    reference_group_grads: tuple[PyTree, ...] | list[PyTree],
    *,
    tail_fraction: float,
) -> tuple[PyTree, int]:
    """Return an upper-tail grouped empirical-Fisher curvature proxy."""
    if not reference_group_grads:
        raise ValueError("curvature estimation requires reference gradient groups")
    if not 0 < tail_fraction <= 1:
        raise ValueError("curvature CVaR tail fraction must be in (0, 1]")
    tail_count = max(
        1,
        math.ceil(len(reference_group_grads) * tail_fraction),
    )

    def curvature_leaf(*group_grads: jax.Array) -> jax.Array:
        squared = jnp.stack(
            [
                jnp.square(jnp.asarray(grad, dtype=jnp.float32))
                for grad in group_grads
            ],
            axis=0,
        )
        return jnp.mean(
            jnp.sort(squared, axis=0)[-tail_count:],
            axis=0,
        )

    curvature = jax.tree_util.tree_map(
        curvature_leaf,
        *reference_group_grads,
    )
    return curvature, tail_count


def upper_cvar_first_order_harm(
    updates: PyTree,
    reference_group_grads: tuple[PyTree, ...] | list[PyTree],
    *,
    tail_fraction: float,
) -> tuple[PyTree, int]:
    """Return worst-tail first-order mastered-loss change per coordinate."""
    if not reference_group_grads:
        raise ValueError("first-order harm requires reference gradient groups")
    if not 0 < tail_fraction <= 1:
        raise ValueError("harm CVaR tail fraction must be in (0, 1]")
    tail_count = max(
        1,
        math.ceil(len(reference_group_grads) * tail_fraction),
    )

    def harm_leaf(
        update: jax.Array,
        *group_grads: jax.Array,
    ) -> jax.Array:
        additive_update = jnp.asarray(update, dtype=jnp.float32)
        harms = jnp.stack(
            [
                additive_update * jnp.asarray(grad, dtype=jnp.float32)
                for grad in group_grads
            ],
            axis=0,
        )
        return jnp.mean(
            jnp.sort(harms, axis=0)[-tail_count:],
            axis=0,
        )

    harm = jax.tree_util.tree_map(
        harm_leaf,
        updates,
        *reference_group_grads,
    )
    return harm, tail_count


def fisher_free_capacity_source_gate(
    curvature: PyTree,
    *,
    strength: float,
    min_gate: float,
    eps: float,
) -> tuple[PyTree, dict[str, jax.Array]]:
    """Turn robust diagonal curvature into a scale-free soft capacity gate."""
    if strength < 0:
        raise ValueError("Fisher free-capacity strength must be nonnegative")
    if not 0 <= min_gate <= 1:
        raise ValueError("Fisher free-capacity minimum gate must be in [0, 1]")
    if eps <= 0:
        raise ValueError("Fisher free-capacity epsilon must be positive")

    curvature_mean = tree_mean(curvature)
    scale = jnp.maximum(
        curvature_mean,
        jnp.asarray(eps, dtype=jnp.float32),
    )
    normalized = jax.tree_util.tree_map(
        lambda value: jnp.asarray(value, dtype=jnp.float32) / scale,
        curvature,
    )
    raw_gate = jax.tree_util.tree_map(
        lambda value: 1.0
        / (
            1.0
            + jnp.asarray(strength, dtype=jnp.float32)
            * jnp.asarray(value, dtype=jnp.float32)
        ),
        normalized,
    )
    gate = jax.tree_util.tree_map(
        lambda value: (
            jnp.asarray(min_gate, dtype=jnp.float32)
            + (1.0 - jnp.asarray(min_gate, dtype=jnp.float32))
            * jnp.asarray(value, dtype=jnp.float32)
        ),
        raw_gate,
    )
    stats = {
        "curvature_mean": curvature_mean,
        "curvature_scale": scale,
        "normalized_curvature_mean": tree_mean(normalized),
        "gate_mean": tree_mean(gate),
        "high_curvature_fraction": tree_mean(
            jax.tree_util.tree_map(
                lambda value: (
                    jnp.asarray(value, dtype=jnp.float32) >= 1.0
                ).astype(jnp.float32),
                normalized,
            )
        ),
    }
    return gate, stats


def _conservative_gate_ema(
    instantaneous_gate: PyTree,
    mask_ema: PyTree,
    mask_ema_count: jax.Array,
    *,
    beta: float,
) -> tuple[PyTree, PyTree, jax.Array, PyTree]:
    """Smooth gates without reopening a currently rejected coordinate."""
    effective_gate, mask_ema, mask_ema_count = update_mask_ema(
        instantaneous_gate,
        mask_ema,
        mask_ema_count,
        beta=beta,
    )
    final_gate = jax.tree_util.tree_map(
        lambda smoothed, current: jnp.minimum(
            jnp.asarray(smoothed, dtype=jnp.float32),
            jnp.asarray(current, dtype=jnp.float32),
        ),
        effective_gate,
        instantaneous_gate,
    )
    return final_gate, mask_ema, mask_ema_count, effective_gate


def combine_fisher_free_capacity_gates_ema(
    synthetic_gate: PyTree,
    real_gate: PyTree,
    mask_ema: PyTree,
    mask_ema_count: jax.Array,
    *,
    beta: float,
) -> tuple[PyTree, PyTree, jax.Array, dict[str, jax.Array]]:
    """Combine source gates and update their conservative epoch-level EMA."""
    instantaneous_gate = jax.tree_util.tree_map(
        lambda synthetic, real: jnp.minimum(
            jnp.asarray(synthetic, dtype=jnp.float32),
            jnp.asarray(real, dtype=jnp.float32),
        ),
        synthetic_gate,
        real_gate,
    )
    (
        final_gate,
        mask_ema,
        mask_ema_count,
        effective_gate,
    ) = _conservative_gate_ema(
        instantaneous_gate,
        mask_ema,
        mask_ema_count,
        beta=beta,
    )
    real_veto = jax.tree_util.tree_map(
        lambda synthetic, real: (
            jnp.asarray(real, dtype=jnp.float32)
            < jnp.asarray(synthetic, dtype=jnp.float32)
        ).astype(jnp.float32),
        synthetic_gate,
        real_gate,
    )
    ema_extra_veto = jax.tree_util.tree_map(
        lambda smoothed, current: (
            jnp.asarray(smoothed, dtype=jnp.float32)
            < jnp.asarray(current, dtype=jnp.float32)
        ).astype(jnp.float32),
        effective_gate,
        instantaneous_gate,
    )
    stats = {
        "fisher_free/synthetic_gate_mean": tree_mean(synthetic_gate),
        "fisher_free/real_gate_mean": tree_mean(real_gate),
        "fisher_free/instantaneous_gate_mean": tree_mean(
            instantaneous_gate
        ),
        "fisher_free/ema_gate_mean": tree_mean(effective_gate),
        "fisher_free/final_gate_mean": tree_mean(final_gate),
        "fisher_free/real_veto_fraction": tree_mean(real_veto),
        "fisher_free/ema_extra_veto_fraction": tree_mean(ema_extra_veto),
    }
    return final_gate, mask_ema, mask_ema_count, stats


def apply_fisher_free_capacity_gate(
    updates: PyTree,
    final_gate: PyTree,
) -> tuple[PyTree, dict[str, jax.Array]]:
    """Apply a fixed epoch-level Fisher gate to Optax updates."""
    final_updates = jax.tree_util.tree_map(
        lambda update, gate: (
            update * jnp.asarray(gate, dtype=update.dtype)
        ),
        updates,
        final_gate,
    )
    raw_norm = tree_l2_norm(updates)
    final_norm = tree_l2_norm(final_updates)
    stats = {
        "update_norm/raw": raw_norm,
        "update_norm/final_curvature": final_norm,
        "update_norm/kept_fraction": final_norm
        / jnp.maximum(raw_norm, 1e-30),
    }
    return final_updates, stats


def mask_optax_updates_fisher_free_capacity_ema(
    updates: PyTree,
    synthetic_gate: PyTree,
    real_gate: PyTree,
    mask_ema: PyTree,
    mask_ema_count: jax.Array,
    *,
    beta: float,
) -> tuple[PyTree, PyTree, jax.Array, dict[str, jax.Array]]:
    """Combine an EMA gate once and apply it to one Optax update."""
    final_gate, mask_ema, mask_ema_count, stats = (
        combine_fisher_free_capacity_gates_ema(
            synthetic_gate,
            real_gate,
            mask_ema,
            mask_ema_count,
            beta=beta,
        )
    )
    final_updates, update_stats = apply_fisher_free_capacity_gate(
        updates,
        final_gate,
    )
    stats.update(update_stats)
    return final_updates, mask_ema, mask_ema_count, stats


def second_order_source_gate(
    updates: PyTree,
    reference_group_grads: tuple[PyTree, ...] | list[PyTree],
    curvature: PyTree,
    *,
    tail_fraction: float,
    horizon: float,
    min_gate: float,
    eps: float,
) -> tuple[PyTree, PyTree, dict[str, jax.Array]]:
    """Compute the largest diagonal-quadratic-safe gate for one source."""
    if horizon <= 0:
        raise ValueError("second-order horizon must be positive")
    if not 0 <= min_gate <= 1:
        raise ValueError("second-order minimum gate must be in [0, 1]")
    if eps <= 0:
        raise ValueError("second-order epsilon must be positive")

    harm, tail_count = upper_cvar_first_order_harm(
        updates,
        reference_group_grads,
        tail_fraction=tail_fraction,
    )
    quadratic = jax.tree_util.tree_map(
        lambda update, fisher: (
            jnp.asarray(horizon, dtype=jnp.float32)
            * jnp.asarray(fisher, dtype=jnp.float32)
            * jnp.square(jnp.asarray(update, dtype=jnp.float32))
        ),
        updates,
        curvature,
    )

    def gate_leaf(
        update: jax.Array,
        linear_harm: jax.Array,
        quadratic_harm: jax.Array,
    ) -> jax.Array:
        update = jnp.asarray(update)
        linear_harm = jnp.asarray(linear_harm, dtype=jnp.float32)
        quadratic_harm = jnp.asarray(quadratic_harm, dtype=jnp.float32)
        safe_fraction = jnp.where(
            linear_harm > 0,
            0.0,
            jnp.where(
                quadratic_harm <= jnp.asarray(eps, dtype=jnp.float32),
                1.0,
                jnp.clip(
                    -2.0 * linear_harm / quadratic_harm,
                    0.0,
                    1.0,
                ),
            ),
        )
        safe_fraction = jnp.where(update == 0, 1.0, safe_fraction)
        return (
            jnp.asarray(min_gate, dtype=jnp.float32)
            + (1.0 - jnp.asarray(min_gate, dtype=jnp.float32))
            * safe_fraction
        )

    gate = jax.tree_util.tree_map(
        gate_leaf,
        updates,
        harm,
        quadratic,
    )
    full_step_risk = jax.tree_util.tree_map(
        lambda linear, quad: (
            jnp.asarray(linear, dtype=jnp.float32)
            + 0.5 * jnp.asarray(quad, dtype=jnp.float32)
        ),
        harm,
        quadratic,
    )
    gated_risk = jax.tree_util.tree_map(
        lambda linear, quad, value: (
            jnp.asarray(value, dtype=jnp.float32)
            * jnp.asarray(linear, dtype=jnp.float32)
            + 0.5
            * jnp.square(jnp.asarray(value, dtype=jnp.float32))
            * jnp.asarray(quad, dtype=jnp.float32)
        ),
        harm,
        quadratic,
        gate,
    )
    curvature_limited = jax.tree_util.tree_map(
        lambda linear, value: (
            (jnp.asarray(linear, dtype=jnp.float32) <= 0)
            & (jnp.asarray(value, dtype=jnp.float32) < 1.0 - 1e-6)
        ).astype(jnp.float32),
        harm,
        gate,
    )
    stats = {
        "tail_count": jnp.asarray(tail_count, dtype=jnp.float32),
        "gate_mean": tree_mean(gate),
        "first_order_safe_fraction": tree_mean(
            jax.tree_util.tree_map(
                lambda value: (
                    jnp.asarray(value, dtype=jnp.float32) <= 0
                ).astype(jnp.float32),
                harm,
            )
        ),
        "curvature_limited_fraction": tree_mean(curvature_limited),
        "predicted_risk_before": tree_sum(full_step_risk),
        "predicted_risk_after_source_gate": tree_sum(gated_risk),
        "linear_harm_mean": tree_mean(harm),
        "quadratic_harm_mean": tree_mean(quadratic),
    }
    return gate, harm, stats


def mask_optax_updates_second_order_cpl_ema(
    updates: PyTree,
    synthetic_group_grads: tuple[PyTree, ...] | list[PyTree],
    real_group_grads: tuple[PyTree, ...] | list[PyTree],
    synthetic_curvature: PyTree,
    real_curvature: PyTree,
    mask_ema: PyTree,
    mask_ema_count: jax.Array,
    *,
    tail_fraction: float,
    beta: float,
    horizon: float,
    min_gate: float,
    eps: float,
) -> tuple[PyTree, PyTree, jax.Array, dict[str, jax.Array]]:
    """Replace first-order CPL with source-balanced quadratic safe gates."""
    synthetic_gate, synthetic_harm, synthetic_stats = second_order_source_gate(
        updates,
        synthetic_group_grads,
        synthetic_curvature,
        tail_fraction=tail_fraction,
        horizon=horizon,
        min_gate=min_gate,
        eps=eps,
    )
    real_gate, real_harm, real_stats = second_order_source_gate(
        updates,
        real_group_grads,
        real_curvature,
        tail_fraction=tail_fraction,
        horizon=horizon,
        min_gate=min_gate,
        eps=eps,
    )
    instantaneous_gate = jax.tree_util.tree_map(
        lambda synthetic, real: jnp.minimum(
            jnp.asarray(synthetic, dtype=jnp.float32),
            jnp.asarray(real, dtype=jnp.float32),
        ),
        synthetic_gate,
        real_gate,
    )
    (
        final_gate,
        mask_ema,
        mask_ema_count,
        effective_gate,
    ) = _conservative_gate_ema(
        instantaneous_gate,
        mask_ema,
        mask_ema_count,
        beta=beta,
    )
    final_updates = jax.tree_util.tree_map(
        lambda update, gate: (
            update * jnp.asarray(gate, dtype=update.dtype)
        ),
        updates,
        final_gate,
    )
    first_order_combined = jax.tree_util.tree_map(
        lambda synthetic, real: (
            (jnp.asarray(synthetic, dtype=jnp.float32) <= 0)
            & (jnp.asarray(real, dtype=jnp.float32) <= 0)
        ).astype(jnp.float32),
        synthetic_harm,
        real_harm,
    )
    differs_from_hard = jax.tree_util.tree_map(
        lambda final, hard: (
            jnp.abs(
                jnp.asarray(final, dtype=jnp.float32)
                - jnp.asarray(hard, dtype=jnp.float32)
            )
            > 1e-6
        ).astype(jnp.float32),
        instantaneous_gate,
        first_order_combined,
    )
    real_veto = jax.tree_util.tree_map(
        lambda synthetic, real: (
            jnp.asarray(real, dtype=jnp.float32)
            < jnp.asarray(synthetic, dtype=jnp.float32)
        ).astype(jnp.float32),
        synthetic_gate,
        real_gate,
    )
    raw_norm = tree_l2_norm(updates)
    final_norm = tree_l2_norm(final_updates)
    stats = {
        "second_order/synthetic_gate_mean": tree_mean(synthetic_gate),
        "second_order/real_gate_mean": tree_mean(real_gate),
        "second_order/instantaneous_gate_mean": tree_mean(
            instantaneous_gate
        ),
        "second_order/ema_gate_mean": tree_mean(effective_gate),
        "second_order/final_gate_mean": tree_mean(final_gate),
        "second_order/real_veto_fraction": tree_mean(real_veto),
        "second_order/differs_from_first_order_fraction": tree_mean(
            differs_from_hard
        ),
        "second_order/synthetic_tail_count": synthetic_stats["tail_count"],
        "second_order/real_tail_count": real_stats["tail_count"],
        "second_order/synthetic_first_order_safe_fraction": synthetic_stats[
            "first_order_safe_fraction"
        ],
        "second_order/real_first_order_safe_fraction": real_stats[
            "first_order_safe_fraction"
        ],
        "second_order/synthetic_curvature_limited_fraction": synthetic_stats[
            "curvature_limited_fraction"
        ],
        "second_order/real_curvature_limited_fraction": real_stats[
            "curvature_limited_fraction"
        ],
        "second_order/synthetic_predicted_risk_before": synthetic_stats[
            "predicted_risk_before"
        ],
        "second_order/synthetic_predicted_risk_after_source_gate": (
            synthetic_stats["predicted_risk_after_source_gate"]
        ),
        "second_order/real_predicted_risk_before": real_stats[
            "predicted_risk_before"
        ],
        "second_order/real_predicted_risk_after_source_gate": real_stats[
            "predicted_risk_after_source_gate"
        ],
        "second_order/synthetic_linear_harm_mean": synthetic_stats[
            "linear_harm_mean"
        ],
        "second_order/real_linear_harm_mean": real_stats[
            "linear_harm_mean"
        ],
        "second_order/synthetic_quadratic_harm_mean": synthetic_stats[
            "quadratic_harm_mean"
        ],
        "second_order/real_quadratic_harm_mean": real_stats[
            "quadratic_harm_mean"
        ],
        "second_order/horizon": jnp.asarray(horizon, dtype=jnp.float32),
        "update_norm/raw": raw_norm,
        "update_norm/final_curvature": final_norm,
        "update_norm/kept_fraction": final_norm
        / jnp.maximum(raw_norm, 1e-30),
    }
    return final_updates, mask_ema, mask_ema_count, stats
