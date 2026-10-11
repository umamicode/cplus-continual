"""Anchor-calibrated quadratic minimum-veto updates.

The curvature input is an empirical diagonal Fisher, ``E[g ** 2]``.  It is a
positive-semidefinite local sensitivity proxy, not the raw neural-network
Hessian.  All risk calculations use the additive Optax update.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from jax_sft.cnl import (
    compute_cvar_masks,
    first_order_contribution,
    tree_dot,
    tree_l2_norm,
    tree_l2_sq,
    tree_sum,
)


PyTree = Any


def _tree_size(tree: PyTree) -> int:
    return sum(leaf.size for leaf in jax.tree_util.tree_leaves(tree))


def _tree_mean(tree: PyTree) -> jax.Array:
    total = tree_sum(tree)
    size = _tree_size(tree)
    return total / jnp.asarray(float(max(size, 1)), dtype=total.dtype)


def _tree_max(tree: PyTree) -> jax.Array:
    leaves = jax.tree_util.tree_leaves(tree)
    if not leaves:
        return jnp.asarray(0.0, dtype=jnp.float32)
    result = jnp.asarray(-jnp.inf, dtype=jnp.float32)
    for leaf in leaves:
        result = jnp.maximum(
            result,
            jnp.max(jnp.asarray(leaf, dtype=jnp.float32)),
        )
    return result


def _tree_min_positive(tree: PyTree) -> jax.Array:
    leaves = jax.tree_util.tree_leaves(tree)
    if not leaves:
        return jnp.asarray(jnp.inf, dtype=jnp.float32)
    result = jnp.asarray(jnp.inf, dtype=jnp.float32)
    for leaf in leaves:
        value = jnp.asarray(leaf, dtype=jnp.float32)
        result = jnp.minimum(
            result,
            jnp.min(jnp.where(value > 0, value, jnp.inf)),
        )
    return result


def _tree_cdf(tree: PyTree, thresholds: jax.Array) -> jax.Array:
    counts = jnp.zeros_like(thresholds, dtype=jnp.float32)
    for leaf in jax.tree_util.tree_leaves(tree):
        value = jnp.asarray(leaf, dtype=jnp.float32)
        counts = counts + jnp.stack(
            [jnp.sum(value <= threshold) for threshold in thresholds]
        )
    size = _tree_size(tree)
    return counts / jnp.asarray(float(max(size, 1)), dtype=counts.dtype)


def _tree_nonnegative_quantiles(
    tree: PyTree,
    *,
    iterations: int,
    eps: float,
) -> jax.Array:
    """Estimate p50/p90/p99 without sorting a model-sharded dimension."""
    if iterations <= 0:
        raise ValueError("Fisher quantile iterations must be positive")
    targets = jnp.asarray([0.50, 0.90, 0.99], dtype=jnp.float32)
    minimum = _tree_min_positive(tree)
    maximum = jnp.maximum(
        _tree_max(tree),
        jnp.asarray(eps, dtype=jnp.float32),
    )
    has_positive = jnp.isfinite(minimum)
    log_minimum = jnp.log(
        jnp.maximum(minimum, jnp.asarray(eps, dtype=jnp.float32))
    )
    log_maximum = jnp.log(maximum)

    def refine(
        _: int,
        bounds: tuple[jax.Array, jax.Array],
    ) -> tuple[jax.Array, jax.Array]:
        low, high = bounds
        middle = 0.5 * (low + high)
        cdf = _tree_cdf(tree, jnp.exp(middle))
        return (
            jnp.where(cdf < targets, middle, low),
            jnp.where(cdf < targets, high, middle),
        )

    _, high = jax.lax.fori_loop(
        0,
        iterations,
        refine,
        (
            jnp.full_like(targets, log_minimum),
            jnp.full_like(targets, log_maximum),
        ),
    )
    zero_fraction = _tree_cdf(
        tree,
        jnp.zeros_like(targets),
    )
    estimates = jnp.exp(high)
    return jnp.where(
        has_positive & (targets > zero_fraction),
        estimates,
        jnp.zeros_like(estimates),
    )


def _tree_cosine(tree_a: PyTree, tree_b: PyTree, eps: float) -> jax.Array:
    denominator = jnp.maximum(
        tree_l2_norm(tree_a) * tree_l2_norm(tree_b),
        jnp.asarray(eps, dtype=jnp.float32),
    )
    return tree_dot(tree_a, tree_b) / denominator


def empirical_fisher_summary(
    fisher: PyTree,
    *,
    eps: float = 1e-30,
    quantile_iterations: int = 10,
) -> dict[str, jax.Array]:
    """Return sharding-safe moments for a nonnegative diagonal Fisher tree."""
    count = max(_tree_size(fisher), 1)
    mean = tree_sum(fisher) / count
    second_moment = tree_sum(
        jax.tree_util.tree_map(
            lambda value: jnp.square(
                jnp.asarray(value, dtype=jnp.float32)
            ),
            fisher,
        )
    ) / count
    summary = {
        "mean": mean,
        "std": jnp.sqrt(jnp.maximum(second_moment - mean**2, 0.0)),
        "max": _tree_max(fisher),
        "nonnegative_fraction": _tree_mean(
            jax.tree_util.tree_map(
                lambda value: (
                    jnp.asarray(value, dtype=jnp.float32) >= 0
                ).astype(jnp.float32),
                fisher,
            )
        ),
        "l2_norm": jnp.sqrt(
            jnp.maximum(
                tree_l2_sq(fisher),
                jnp.asarray(eps, dtype=jnp.float32),
            )
        ),
    }
    quantiles = _tree_nonnegative_quantiles(
        fisher,
        iterations=quantile_iterations,
        eps=eps,
    )
    summary.update(
        {
            "p50": quantiles[0],
            "p90": quantiles[1],
            "p99": quantiles[2],
        }
    )
    return summary


def empirical_fisher_pair_summary(
    synthetic_fisher: PyTree,
    real_fisher: PyTree,
    *,
    eps: float = 1e-30,
    quantile_iterations: int = 10,
) -> dict[str, jax.Array]:
    """Return epoch-level diagnostics for a synthetic/real Fisher pair."""
    synthetic = empirical_fisher_summary(
        synthetic_fisher,
        eps=eps,
        quantile_iterations=quantile_iterations,
    )
    return {
        "quadratic/fisher_mean": synthetic["mean"],
        "quadratic/fisher_std": synthetic["std"],
        "quadratic/fisher_p50": synthetic["p50"],
        "quadratic/fisher_p90": synthetic["p90"],
        "quadratic/fisher_p99": synthetic["p99"],
        "quadratic/fisher_max": synthetic["max"],
        "quadratic/fisher_nonnegative_fraction": synthetic[
            "nonnegative_fraction"
        ],
        "quadratic/real_fisher_mean": _tree_mean(real_fisher),
        "quadratic/synth_real_fisher_cosine": _tree_cosine(
            synthetic_fisher,
            real_fisher,
            eps,
        ),
    }


def calibrate_fisher_by_tensor(
    synthetic_fisher: PyTree,
    real_fisher: PyTree,
    proposal: PyTree,
    *,
    enabled: bool,
    calibration_min: float = 0.25,
    calibration_max: float = 4.0,
    calibration_strength: float = 0.5,
    min_directional_risk: float = 1e-20,
    eps: float = 1e-30,
) -> tuple[PyTree, dict[str, jax.Array]]:
    """Calibrate synthetic Fisher with one shrunk directional ratio per leaf."""
    if calibration_min <= 0 or calibration_max < calibration_min:
        raise ValueError(
            "quadratic calibration bounds must satisfy 0 < min <= max"
        )
    if not 0 <= calibration_strength <= 1:
        raise ValueError("quadratic calibration strength must be in [0, 1]")
    if min_directional_risk < 0:
        raise ValueError(
            "quadratic calibration minimum directional risk must be nonnegative"
        )
    if eps <= 0:
        raise ValueError("quadratic epsilon must be positive")

    synthetic_leaves, treedef = jax.tree_util.tree_flatten(synthetic_fisher)
    real_leaves = jax.tree_util.tree_leaves(real_fisher)
    proposal_leaves = jax.tree_util.tree_leaves(proposal)
    if not (
        len(synthetic_leaves) == len(real_leaves) == len(proposal_leaves)
    ):
        raise ValueError("quadratic calibration trees must have matching leaves")

    calibrated_leaves = []
    alphas = []
    clipped_low = []
    clipped_high = []
    directional_synthetic = []
    directional_real = []
    for synthetic, real, delta in zip(
        synthetic_leaves,
        real_leaves,
        proposal_leaves,
    ):
        synthetic = jnp.maximum(
            jnp.asarray(synthetic, dtype=jnp.float32),
            0.0,
        )
        real = jnp.maximum(jnp.asarray(real, dtype=jnp.float32), 0.0)
        delta_sq = jnp.square(jnp.asarray(delta, dtype=jnp.float32))
        q_synthetic = jnp.sum(synthetic * delta_sq)
        q_real = jnp.sum(real * delta_sq)
        ratio = (q_real + eps) / (q_synthetic + eps)
        ratio_clipped = jnp.clip(
            ratio,
            jnp.asarray(calibration_min, dtype=jnp.float32),
            jnp.asarray(calibration_max, dtype=jnp.float32),
        )
        informative = (
            jnp.maximum(q_synthetic, q_real)
            >= jnp.asarray(min_directional_risk, dtype=jnp.float32)
        )
        alpha = jnp.where(
            enabled & informative,
            jnp.exp(
                jnp.asarray(calibration_strength, dtype=jnp.float32)
                * jnp.log(ratio_clipped)
            ),
            jnp.asarray(1.0, dtype=jnp.float32),
        )
        calibrated_leaves.append(synthetic * alpha)
        alphas.append(alpha)
        clipped_low.append(
            (
                enabled
                & informative
                & (ratio < calibration_min)
            ).astype(jnp.float32)
        )
        clipped_high.append(
            (
                enabled
                & informative
                & (ratio > calibration_max)
            ).astype(jnp.float32)
        )
        directional_synthetic.append(q_synthetic)
        directional_real.append(q_real)

    alpha_values = jnp.stack(alphas)
    alpha_mean = jnp.mean(alpha_values)
    stats = {
        "quadratic/calibration_enabled": jnp.asarray(
            float(enabled),
            dtype=jnp.float32,
        ),
        "quadratic/calibration_alpha_mean": alpha_mean,
        "quadratic/calibration_alpha_std": jnp.std(alpha_values),
        "quadratic/calibration_alpha_min": jnp.min(alpha_values),
        "quadratic/calibration_alpha_max": jnp.max(alpha_values),
        "quadratic/calibration_clipped_low_fraction": jnp.mean(
            jnp.stack(clipped_low)
        ),
        "quadratic/calibration_clipped_high_fraction": jnp.mean(
            jnp.stack(clipped_high)
        ),
        "quadratic/directional_synthetic_fisher": jnp.sum(
            jnp.stack(directional_synthetic)
        ),
        "quadratic/directional_real_fisher": jnp.sum(
            jnp.stack(directional_real)
        ),
    }
    return jax.tree_util.tree_unflatten(treedef, calibrated_leaves), stats


def quadratic_risk(
    first_order_contributions: PyTree,
    quadratic_contributions: PyTree,
    gate: PyTree,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return first-order, curvature, and total risk for a continuous gate."""
    first_order = tree_dot(first_order_contributions, gate)
    curvature = 0.5 * tree_sum(
        jax.tree_util.tree_map(
            lambda q, value: (
                jnp.asarray(q, dtype=jnp.float32)
                * jnp.square(jnp.asarray(value, dtype=jnp.float32))
            ),
            quadratic_contributions,
            gate,
        )
    )
    return first_order, curvature, first_order + curvature


def solve_quadratic_minimum_veto(
    proposal: PyTree,
    current_grads: PyTree,
    anchor_grad: PyTree,
    curvature: PyTree,
    *,
    first_order_enabled: bool = True,
    harm_reduction_fraction: float = 0.75,
    energy_cap: float = 0.10,
    weight_eps: float = 1e-12,
    bisection_iterations: int = 12,
    max_expansions: int = 32,
    lambda_initial_high: float = 1.0,
    bisection_tolerance: float = 1e-6,
) -> tuple[PyTree, dict[str, jax.Array]]:
    """Solve the utility-constrained diagonal quadratic minimum-veto."""
    if not 0 <= harm_reduction_fraction <= 1:
        raise ValueError("quadratic harm reduction fraction must be in [0, 1]")
    if not 0 <= energy_cap <= 1:
        raise ValueError("quadratic energy cap must be in [0, 1]")
    if weight_eps <= 0:
        raise ValueError("quadratic weight epsilon must be positive")
    if bisection_iterations <= 0:
        raise ValueError("quadratic bisection iterations must be positive")
    if max_expansions < 0:
        raise ValueError("quadratic max expansions must be nonnegative")
    if lambda_initial_high <= 0:
        raise ValueError("quadratic initial lambda high must be positive")
    if bisection_tolerance < 0:
        raise ValueError("quadratic bisection tolerance must be nonnegative")

    raw_first_order = first_order_contribution(anchor_grad, proposal)
    first_order = (
        raw_first_order
        if first_order_enabled
        else jax.tree_util.tree_map(jnp.zeros_like, raw_first_order)
    )
    quadratic = jax.tree_util.tree_map(
        lambda fisher, delta: (
            jnp.maximum(jnp.asarray(fisher, dtype=jnp.float32), 0.0)
            * jnp.square(jnp.asarray(delta, dtype=jnp.float32))
        ),
        curvature,
        proposal,
    )
    current_utility = jax.tree_util.tree_map(
        lambda grad, delta: jax.nn.relu(
            -jnp.asarray(grad, dtype=jnp.float32)
            * jnp.asarray(delta, dtype=jnp.float32)
        ),
        current_grads,
        proposal,
    )
    utility_weights = jax.tree_util.tree_map(
        lambda utility: (
            jnp.asarray(utility, dtype=jnp.float32)
            + jnp.asarray(weight_eps, dtype=jnp.float32)
        ),
        current_utility,
    )
    ones = jax.tree_util.tree_map(
        lambda value: jnp.ones_like(value, dtype=jnp.float32),
        proposal,
    )
    first_before, curvature_before, risk_before = quadratic_risk(
        first_order,
        quadratic,
        ones,
    )
    risk_target = jnp.where(
        risk_before > 0,
        (
            1.0
            - jnp.asarray(
                harm_reduction_fraction,
                dtype=jnp.float32,
            )
        )
        * risk_before,
        risk_before,
    )
    risk_tolerance = (
        jnp.asarray(bisection_tolerance, dtype=jnp.float32)
        * jnp.maximum(
            jnp.abs(risk_before),
            jnp.asarray(weight_eps, dtype=jnp.float32),
        )
    )
    needs_correction = (risk_before > 0) & (
        jnp.asarray(harm_reduction_fraction, dtype=jnp.float32) > 0
    )

    def gate_for_lambda(lam: jax.Array) -> PyTree:
        return jax.tree_util.tree_map(
            lambda weight, linear, curved: jnp.clip(
                (
                    jnp.asarray(weight, dtype=jnp.float32)
                    - jnp.asarray(lam, dtype=jnp.float32)
                    * jnp.asarray(linear, dtype=jnp.float32)
                )
                / jnp.maximum(
                    jnp.asarray(weight, dtype=jnp.float32)
                    + jnp.asarray(lam, dtype=jnp.float32)
                    * jnp.asarray(curved, dtype=jnp.float32),
                    jnp.asarray(weight_eps, dtype=jnp.float32),
                ),
                0.0,
                1.0,
            ),
            utility_weights,
            first_order,
            quadratic,
        )

    def predicted_risk(lam: jax.Array) -> jax.Array:
        return quadratic_risk(
            first_order,
            quadratic,
            gate_for_lambda(lam),
        )[2]

    proposal_energy = tree_l2_sq(proposal)

    def removed_energy_fraction(lam: jax.Array) -> jax.Array:
        gate = gate_for_lambda(lam)
        removed = jax.tree_util.tree_map(
            lambda delta, value: (
                (1.0 - jnp.asarray(value, dtype=jnp.float32))
                * jnp.asarray(delta, dtype=jnp.float32)
            ),
            proposal,
            gate,
        )
        return tree_l2_sq(removed) / jnp.maximum(
            proposal_energy,
            jnp.asarray(weight_eps, dtype=jnp.float32),
        )

    initial_high = jnp.asarray(lambda_initial_high, dtype=jnp.float32)

    def expansion_needed(
        state: tuple[jax.Array, jax.Array],
    ) -> jax.Array:
        high, count = state
        return (
            count < jnp.asarray(max_expansions, dtype=jnp.int32)
        ) & (predicted_risk(high) > risk_target)

    def expand_high(
        state: tuple[jax.Array, jax.Array],
    ) -> tuple[jax.Array, jax.Array]:
        high, count = state
        return high * 2.0, count + jnp.asarray(1, dtype=jnp.int32)

    def bisect_risk(
        _: int,
        bounds: tuple[jax.Array, jax.Array],
    ) -> tuple[jax.Array, jax.Array]:
        low, high = bounds
        middle = 0.5 * (low + high)
        too_risky = predicted_risk(middle) > risk_target
        return (
            jnp.where(too_risky, middle, low),
            jnp.where(too_risky, high, middle),
        )

    def solve_risk(_: None) -> tuple[jax.Array, jax.Array]:
        expanded_high, expansion_count = jax.lax.while_loop(
            expansion_needed,
            expand_high,
            (initial_high, jnp.asarray(0, dtype=jnp.int32)),
        )
        _, lambda_risk = jax.lax.fori_loop(
            0,
            bisection_iterations,
            bisect_risk,
            (jnp.asarray(0.0, dtype=jnp.float32), expanded_high),
        )
        return lambda_risk, expansion_count

    lambda_risk, expansion_count = jax.lax.cond(
        needs_correction,
        solve_risk,
        lambda _: (
            jnp.asarray(0.0, dtype=jnp.float32),
            jnp.asarray(0, dtype=jnp.int32),
        ),
        operand=None,
    )
    cap_binding = needs_correction & (
        removed_energy_fraction(lambda_risk)
        > jnp.asarray(
            energy_cap + bisection_tolerance,
            dtype=jnp.float32,
        )
    )

    def bisect_cap(
        _: int,
        bounds: tuple[jax.Array, jax.Array],
    ) -> tuple[jax.Array, jax.Array]:
        low, high = bounds
        middle = 0.5 * (low + high)
        under_cap = removed_energy_fraction(middle) <= jnp.asarray(
            energy_cap,
            dtype=jnp.float32,
        )
        return (
            jnp.where(under_cap, middle, low),
            jnp.where(under_cap, high, middle),
        )

    def solve_cap(_: None) -> jax.Array:
        lambda_cap, _ = jax.lax.fori_loop(
            0,
            bisection_iterations,
            bisect_cap,
            (jnp.asarray(0.0, dtype=jnp.float32), lambda_risk),
        )
        return lambda_cap

    lambda_cap = jax.lax.cond(
        cap_binding,
        solve_cap,
        lambda _: lambda_risk,
        operand=None,
    )
    lambda_final = jnp.minimum(lambda_risk, lambda_cap)
    gate = gate_for_lambda(lambda_final)
    first_after, curvature_after, risk_after = quadratic_risk(
        first_order,
        quadratic,
        gate,
    )

    active = jax.tree_util.tree_map(
        lambda delta: jnp.asarray(delta) != 0,
        proposal,
    )
    active_count = tree_sum(
        jax.tree_util.tree_map(
            lambda value: value.astype(jnp.float32),
            active,
        )
    )
    active_denominator = jnp.maximum(active_count, 1.0)

    def active_fraction(condition: PyTree) -> jax.Array:
        return tree_sum(
            jax.tree_util.tree_map(
                lambda selected, keep: (
                    selected.astype(jnp.float32)
                    * jnp.asarray(keep, dtype=jnp.float32)
                ),
                active,
                condition,
            )
        ) / active_denominator

    false_safe = jax.tree_util.tree_map(
        lambda linear, curved, selected: (
            selected
            & (jnp.asarray(linear, dtype=jnp.float32) <= 0)
            & (
                jnp.asarray(linear, dtype=jnp.float32)
                + 0.5 * jnp.asarray(curved, dtype=jnp.float32)
                > 0
            )
        ),
        first_order,
        quadratic,
        active,
    )
    false_safe_energy = tree_sum(
        jax.tree_util.tree_map(
            lambda delta, selected: (
                jnp.square(jnp.asarray(delta, dtype=jnp.float32))
                * selected.astype(jnp.float32)
            ),
            proposal,
            false_safe,
        )
    )
    false_safe_curvature = 0.5 * tree_sum(
        jax.tree_util.tree_map(
            lambda curved, selected: (
                jnp.asarray(curved, dtype=jnp.float32)
                * selected.astype(jnp.float32)
            ),
            quadratic,
            false_safe,
        )
    )
    utility_total = tree_sum(current_utility)
    utility_kept = tree_dot(current_utility, gate)
    risk_reduction = jnp.where(
        risk_before > 0,
        (risk_before - risk_after)
        / jnp.maximum(
            risk_before,
            jnp.asarray(weight_eps, dtype=jnp.float32),
        ),
        0.0,
    )
    stats = {
        "quadratic/first_order_risk_before": first_before,
        "quadratic/curvature_risk_before": curvature_before,
        "quadratic/total_risk_before": risk_before,
        "quadratic/first_order_risk_instant_after": first_after,
        "quadratic/curvature_risk_instant_after": curvature_after,
        "quadratic/total_risk_instant_after": risk_after,
        "quadratic/risk_target": risk_target,
        "quadratic/risk_target_shortfall": jax.nn.relu(
            risk_after - risk_target
        ),
        "quadratic/risk_reduction_fraction_achieved": risk_reduction,
        "quadratic/lambda_risk": lambda_risk,
        "quadratic/lambda_cap": lambda_cap,
        "quadratic/lambda_final": lambda_final,
        "quadratic/cap_binding": cap_binding.astype(jnp.float32),
        "quadratic/target_feasible_under_cap": (
            risk_after <= risk_target + risk_tolerance
        ).astype(jnp.float32),
        "quadratic/update_energy_removed_fraction": (
            removed_energy_fraction(lambda_final)
        ),
        "quadratic/current_utility_kept_fraction": (
            utility_kept
            / jnp.maximum(
                utility_total,
                jnp.asarray(weight_eps, dtype=jnp.float32),
            )
        ),
        "quadratic/bisection_iterations": jnp.where(
            needs_correction,
            jnp.asarray(bisection_iterations, dtype=jnp.float32)
            * (1.0 + cap_binding.astype(jnp.float32)),
            0.0,
        ),
        "quadratic/lambda_expansions": expansion_count.astype(jnp.float32),
        "quadratic/false_safe_fraction": active_fraction(false_safe),
        "quadratic/false_safe_update_energy_fraction": (
            false_safe_energy
            / jnp.maximum(
                proposal_energy,
                jnp.asarray(weight_eps, dtype=jnp.float32),
            )
        ),
        "quadratic/false_safe_curvature_risk_fraction": (
            false_safe_curvature
            / jnp.maximum(
                curvature_before,
                jnp.asarray(weight_eps, dtype=jnp.float32),
            )
        ),
        "quadratic/first_order_safe_fraction": active_fraction(
            jax.tree_util.tree_map(
                lambda linear: (
                    jnp.asarray(linear, dtype=jnp.float32) <= 0
                ),
                first_order,
            )
        ),
        "quadratic/quadratic_safe_fraction": active_fraction(
            jax.tree_util.tree_map(
                lambda linear, curved: (
                    jnp.asarray(linear, dtype=jnp.float32)
                    + 0.5 * jnp.asarray(curved, dtype=jnp.float32)
                    <= 0
                ),
                first_order,
                quadratic,
            )
        ),
    }
    return gate, stats


def _gate_summary(
    gate: PyTree,
    active: PyTree,
) -> dict[str, jax.Array]:
    active_count = tree_sum(
        jax.tree_util.tree_map(
            lambda value: value.astype(jnp.float32),
            active,
        )
    )
    denominator = jnp.maximum(active_count, 1.0)
    values = jax.tree_util.tree_map(
        lambda value, selected: (
            jnp.asarray(value, dtype=jnp.float32)
            * selected.astype(jnp.float32)
        ),
        gate,
        active,
    )
    mean = tree_sum(values) / denominator
    second_moment = tree_sum(
        jax.tree_util.tree_map(
            lambda value, selected: (
                jnp.square(jnp.asarray(value, dtype=jnp.float32))
                * selected.astype(jnp.float32)
            ),
            gate,
            active,
        )
    ) / denominator
    minimum = jnp.asarray(jnp.inf, dtype=jnp.float32)
    maximum = jnp.asarray(-jnp.inf, dtype=jnp.float32)
    for value, selected in zip(
        jax.tree_util.tree_leaves(gate),
        jax.tree_util.tree_leaves(active),
    ):
        value = jnp.asarray(value, dtype=jnp.float32)
        minimum = jnp.minimum(
            minimum,
            jnp.min(jnp.where(selected, value, jnp.inf)),
        )
        maximum = jnp.maximum(
            maximum,
            jnp.max(jnp.where(selected, value, -jnp.inf)),
        )
    tolerance = jnp.asarray(1e-6, dtype=jnp.float32)

    def fraction(predicate: PyTree) -> jax.Array:
        return tree_sum(
            jax.tree_util.tree_map(
                lambda keep, selected: (
                    jnp.asarray(keep, dtype=jnp.float32)
                    * selected.astype(jnp.float32)
                ),
                predicate,
                active,
            )
        ) / denominator

    return {
        "quadratic/gate_mean": mean,
        "quadratic/gate_std": jnp.sqrt(
            jnp.maximum(second_moment - mean**2, 0.0)
        ),
        "quadratic/gate_min": jnp.where(active_count > 0, minimum, 0.0),
        "quadratic/gate_max": jnp.where(active_count > 0, maximum, 0.0),
        "quadratic/gate_unchanged_fraction": fraction(
            jax.tree_util.tree_map(
                lambda value: (
                    jnp.asarray(value, dtype=jnp.float32)
                    >= 1.0 - tolerance
                ),
                gate,
            )
        ),
        "quadratic/gate_soft_fraction": fraction(
            jax.tree_util.tree_map(
                lambda value: (
                    (jnp.asarray(value, dtype=jnp.float32) > tolerance)
                    & (
                        jnp.asarray(value, dtype=jnp.float32)
                        < 1.0 - tolerance
                    )
                ),
                gate,
            )
        ),
        "quadratic/gate_zero_fraction": fraction(
            jax.tree_util.tree_map(
                lambda value: (
                    jnp.asarray(value, dtype=jnp.float32) <= tolerance
                ),
                gate,
            )
        ),
    }


def _update_gate_ema(
    instantaneous_gate: PyTree,
    mask_ema: PyTree,
    mask_ema_count: jax.Array,
    *,
    beta: float,
    mode: str,
) -> tuple[PyTree, PyTree, jax.Array]:
    if not 0 <= beta < 1:
        raise ValueError("quadratic EMA beta must be in [0, 1)")
    if mode not in {"symmetric", "asymmetric_safety"}:
        raise ValueError(
            "quadratic EMA mode must be symmetric or asymmetric_safety"
        )
    beta_arr = jnp.asarray(beta, dtype=jnp.float32)
    new_count = mask_ema_count + jnp.asarray(
        1,
        dtype=mask_ema_count.dtype,
    )
    if mode == "symmetric":
        new_state = jax.tree_util.tree_map(
            lambda ema, gate: (
                beta_arr * jnp.asarray(ema, dtype=jnp.float32)
                + (1.0 - beta_arr)
                * jnp.asarray(gate, dtype=jnp.float32)
            ),
            mask_ema,
            instantaneous_gate,
        )
        correction = 1.0 - beta_arr ** new_count.astype(jnp.float32)
        effective = jax.tree_util.tree_map(
            lambda state: jnp.clip(state / correction, 0.0, 1.0),
            new_state,
        )
        return effective, new_state, new_count

    previous = jax.tree_util.tree_map(
        lambda state: jnp.where(
            mask_ema_count == 0,
            jnp.ones_like(state, dtype=jnp.float32),
            jnp.asarray(state, dtype=jnp.float32),
        ),
        mask_ema,
    )
    effective = jax.tree_util.tree_map(
        lambda old, gate: jnp.where(
            jnp.asarray(gate, dtype=jnp.float32) < old,
            jnp.asarray(gate, dtype=jnp.float32),
            beta_arr * old
            + (1.0 - beta_arr)
            * jnp.asarray(gate, dtype=jnp.float32),
        ),
        previous,
        instantaneous_gate,
    )
    effective = jax.tree_util.tree_map(
        lambda value: jnp.clip(value, 0.0, 1.0),
        effective,
    )
    return effective, effective, new_count


def mask_optax_updates_quadratic_minveto_ema(
    updates: PyTree,
    current_grads: PyTree,
    synthetic_group_grads: list[PyTree] | tuple[PyTree, ...],
    anchor_grad: PyTree,
    synthetic_fisher: PyTree,
    real_fisher: PyTree,
    mask_ema: PyTree,
    mask_ema_count: jax.Array,
    *,
    quadratic_preview_ema: PyTree | None = None,
    quadratic_preview_ema_count: jax.Array | None = None,
    tail_fraction: float = 0.2,
    calibration_enabled: bool = True,
    calibration_min: float = 0.25,
    calibration_max: float = 4.0,
    calibration_strength: float = 0.5,
    calibration_min_directional_risk: float = 1e-20,
    first_order_enabled: bool = True,
    harm_reduction_fraction: float = 0.75,
    energy_cap: float = 0.10,
    weight_eps: float = 1e-12,
    bisection_iterations: int = 12,
    max_expansions: int = 32,
    lambda_initial_high: float = 1.0,
    bisection_tolerance: float = 1e-6,
    ema_mode: str = "symmetric",
    ema_beta: float = 0.9,
    normmatched_first_order_control: bool = False,
) -> tuple[
    PyTree,
    PyTree,
    jax.Array,
    PyTree | None,
    jax.Array | None,
    dict[str, jax.Array],
]:
    """Apply synthetic CVaR and a calibrated quadratic real-anchor veto.

    The two optional preview-state outputs are populated only by the
    norm-matched control, which must track the counterfactual quadratic EMA
    separately from its applied first-order EMA.
    """
    synthetic_masks, synthetic_scores, synthetic_tail_count = (
        compute_cvar_masks(
            updates,
            list(synthetic_group_grads),
            tail_fraction=tail_fraction,
        )
    )
    proposal = jax.tree_util.tree_map(
        lambda update, mask: (
            jnp.asarray(update, dtype=jnp.float32)
            * jnp.asarray(mask, dtype=jnp.float32)
        ),
        updates,
        synthetic_masks,
    )
    calibrated_fisher, calibration_stats = calibrate_fisher_by_tensor(
        synthetic_fisher,
        real_fisher,
        proposal,
        enabled=calibration_enabled,
        calibration_min=calibration_min,
        calibration_max=calibration_max,
        calibration_strength=calibration_strength,
        min_directional_risk=calibration_min_directional_risk,
        eps=weight_eps,
    )
    quadratic_gate, solver_stats = solve_quadratic_minimum_veto(
        proposal,
        current_grads,
        anchor_grad,
        calibrated_fisher,
        first_order_enabled=first_order_enabled,
        harm_reduction_fraction=harm_reduction_fraction,
        energy_cap=energy_cap,
        weight_eps=weight_eps,
        bisection_iterations=bisection_iterations,
        max_expansions=max_expansions,
        lambda_initial_high=lambda_initial_high,
        bisection_tolerance=bisection_tolerance,
    )
    instantaneous_gate = jax.tree_util.tree_map(
        lambda synthetic, real: (
            jnp.asarray(synthetic, dtype=jnp.float32)
            * jnp.asarray(real, dtype=jnp.float32)
        ),
        synthetic_masks,
        quadratic_gate,
    )
    if normmatched_first_order_control:
        if (
            quadratic_preview_ema is None
            or quadratic_preview_ema_count is None
        ):
            raise ValueError(
                "norm-matched control requires a separate quadratic "
                "preview EMA state"
            )
        quadratic_ema_input = quadratic_preview_ema
        quadratic_ema_count_input = quadratic_preview_ema_count
    else:
        quadratic_ema_input = mask_ema
        quadratic_ema_count_input = mask_ema_count

    (
        quadratic_effective_gate,
        quadratic_ema_state,
        quadratic_ema_count,
    ) = _update_gate_ema(
        instantaneous_gate,
        quadratic_ema_input,
        quadratic_ema_count_input,
        beta=ema_beta,
        mode=ema_mode,
    )
    quadratic_updates = jax.tree_util.tree_map(
        lambda update, gate: (
            update * jnp.asarray(gate, dtype=update.dtype)
        ),
        updates,
        quadratic_effective_gate,
    )

    norm_scale = jnp.asarray(1.0, dtype=jnp.float32)
    if normmatched_first_order_control:
        zero_curvature = jax.tree_util.tree_map(
            jnp.zeros_like,
            calibrated_fisher,
        )
        first_order_gate, _ = solve_quadratic_minimum_veto(
            proposal,
            current_grads,
            anchor_grad,
            zero_curvature,
            first_order_enabled=True,
            harm_reduction_fraction=harm_reduction_fraction,
            energy_cap=energy_cap,
            weight_eps=weight_eps,
            bisection_iterations=bisection_iterations,
            max_expansions=max_expansions,
            lambda_initial_high=lambda_initial_high,
            bisection_tolerance=bisection_tolerance,
        )
        first_order_instantaneous_gate = jax.tree_util.tree_map(
            lambda synthetic, real: (
                jnp.asarray(synthetic, dtype=jnp.float32)
                * jnp.asarray(real, dtype=jnp.float32)
            ),
            synthetic_masks,
            first_order_gate,
        )
        (
            first_order_effective_gate,
            mask_ema,
            mask_ema_count,
        ) = _update_gate_ema(
            first_order_instantaneous_gate,
            mask_ema,
            mask_ema_count,
            beta=ema_beta,
            mode=ema_mode,
        )
        first_order_updates = jax.tree_util.tree_map(
            lambda update, gate: (
                update * jnp.asarray(gate, dtype=update.dtype)
            ),
            updates,
            first_order_effective_gate,
        )
        target_norm = tree_l2_norm(quadratic_updates)
        control_norm = tree_l2_norm(first_order_updates)
        norm_scale = jnp.where(
            control_norm > 0,
            target_norm
            / jnp.maximum(
                control_norm,
                jnp.asarray(weight_eps, dtype=jnp.float32),
            ),
            0.0,
        )
        final_updates = jax.tree_util.tree_map(
            lambda update: (
                update
                * jnp.asarray(norm_scale, dtype=update.dtype)
            ),
            first_order_updates,
        )
    else:
        final_updates = quadratic_updates
        mask_ema = quadratic_ema_state
        mask_ema_count = quadratic_ema_count
        quadratic_ema_state = quadratic_preview_ema
        quadratic_ema_count = quadratic_preview_ema_count

    ema_first_order = (
        tree_dot(anchor_grad, quadratic_updates)
        if first_order_enabled
        else jnp.asarray(0.0, dtype=jnp.float32)
    )
    ema_curvature = 0.5 * tree_sum(
        jax.tree_util.tree_map(
            lambda fisher, update: (
                jnp.asarray(fisher, dtype=jnp.float32)
                * jnp.square(jnp.asarray(update, dtype=jnp.float32))
            ),
            calibrated_fisher,
            quadratic_updates,
        )
    )
    active = jax.tree_util.tree_map(
        lambda value: jnp.asarray(value) != 0,
        proposal,
    )
    stats = {
        **calibration_stats,
        **solver_stats,
        **_gate_summary(quadratic_gate, active),
        "quadratic/curvature_type_id": jnp.asarray(
            1.0,
            dtype=jnp.float32,
        ),
        "quadratic/first_order_risk_ema_after": ema_first_order,
        "quadratic/curvature_risk_ema_after": ema_curvature,
        "quadratic/total_risk_ema_after": (
            ema_first_order + ema_curvature
        ),
        "mask/synthetic_mean": _tree_mean(synthetic_masks),
        "mask/quadratic_instantaneous_mean": _tree_mean(
            instantaneous_gate
        ),
        "mask/quadratic_ema_mean": _tree_mean(
            quadratic_effective_gate
        ),
        "update_norm/raw": tree_l2_norm(updates),
        "update_norm/synthetic": tree_l2_norm(proposal),
        "update_norm/quadratic_instantaneous": tree_l2_norm(
            jax.tree_util.tree_map(
                lambda update, gate: (
                    jnp.asarray(update, dtype=jnp.float32)
                    * jnp.asarray(gate, dtype=jnp.float32)
                ),
                updates,
                instantaneous_gate,
            )
        ),
        "update_norm/quadratic_ema": tree_l2_norm(quadratic_updates),
        "update_norm/normmatched_control": (
            tree_l2_norm(final_updates)
            if normmatched_first_order_control
            else jnp.asarray(0.0, dtype=jnp.float32)
        ),
        "update_norm/kept_fraction": (
            tree_l2_norm(final_updates)
            / jnp.maximum(
                tree_l2_norm(updates),
                jnp.asarray(weight_eps, dtype=jnp.float32),
            )
        ),
        "control/norm_scale": norm_scale,
        "control/separate_quadratic_ema_state": jnp.asarray(
            float(normmatched_first_order_control),
            dtype=jnp.float32,
        ),
        "cnl/synth_teacher_cvar_score_mean": _tree_mean(
            synthetic_scores
        ),
        "cnl/synth_teacher_cvar_tail_count": jnp.asarray(
            synthetic_tail_count,
            dtype=jnp.float32,
        ),
    }
    return (
        final_updates,
        mask_ema,
        mask_ema_count,
        quadratic_ema_state,
        quadratic_ema_count,
        stats,
    )
