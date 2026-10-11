"""One-sided functional margin budgets for prompt-invariant anchor families."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from jax_sft.cnl import add_trees, scale_tree, tree_dot, tree_l2_norm

PyTree = Any


def _subtract_trees(left: PyTree, right: PyTree) -> PyTree:
    return jax.tree_util.tree_map(
        lambda left_value, right_value: left_value - right_value,
        left,
        right,
    )


def _linear_combination(
    basis: Sequence[PyTree],
    coefficients: jax.Array,
) -> PyTree:
    result = scale_tree(basis[0], coefficients[0])
    for direction, coefficient in zip(basis[1:], coefficients[1:]):
        result = add_trees(result, scale_tree(direction, coefficient))
    return result


def _orthogonalize(
    candidate: PyTree,
    basis: Sequence[PyTree],
) -> PyTree:
    residual = candidate
    # A second pass substantially improves orthogonality for bfloat16 leaves.
    for _ in range(2):
        for direction in basis:
            residual = _subtract_trees(
                residual,
                scale_tree(direction, tree_dot(direction, residual)),
            )
    return residual


def prepare_margin_budget_subspace(
    gradient_fn: Callable[[int], PyTree],
    *,
    initial_margins: Sequence[float],
    family_ids: Sequence[str],
    max_rank: int,
    retain_fraction: float,
    absolute_floor: float,
    seed: int,
    eps: float,
) -> tuple[tuple[PyTree, ...], jax.Array, jax.Array, dict[str, Any]]:
    """Compress per-family margin gradients with a randomized range finder.

    ``gradient_fn`` is intentionally called twice per family. The first pass
    builds balanced signed-hash sketches from unit-normalized family gradients.
    Each family updates one sketch, keeping compression linear in the number of
    parameters. The second pass records every family's raw coefficients in the
    resulting orthonormal basis without retaining all full-model gradients.
    """
    margins = np.asarray(initial_margins, dtype=np.float64)
    if margins.ndim != 1 or margins.size == 0:
        raise ValueError("margin-budget initial margins must be a nonempty vector")
    if len(family_ids) != margins.size:
        raise ValueError("margin-budget family IDs must match initial margins")
    if not np.all(np.isfinite(margins)):
        raise ValueError("margin-budget initial margins must be finite")
    if np.any(margins < 0):
        raise ValueError("margin-budget anchors must start with nonnegative margins")
    if max_rank <= 0:
        raise ValueError("margin-budget rank must be positive")
    if not 0 <= retain_fraction <= 1:
        raise ValueError("margin-budget retain fraction must be in [0, 1]")
    if absolute_floor < 0:
        raise ValueError("margin-budget absolute floor must be nonnegative")
    if eps <= 0:
        raise ValueError("margin-budget epsilon must be positive")

    requested_rank = min(int(max_rank), int(margins.size))
    rng = np.random.default_rng(seed)
    shuffled_families = rng.permutation(margins.size)
    sketch_buckets = np.empty((margins.size,), dtype=np.int32)
    for position, family_index in enumerate(shuffled_families.tolist()):
        sketch_buckets[family_index] = position % requested_rank
    sketch_signs = rng.choice(
        np.asarray([-1.0, 1.0], dtype=np.float32),
        size=(margins.size,),
    )
    sketches: list[PyTree | None] = [None] * requested_rank
    first_pass_norms: list[float] = []
    active_family_count = 0

    for family_index in range(margins.size):
        gradient = gradient_fn(family_index)
        gradient_norm = float(tree_l2_norm(gradient))
        first_pass_norms.append(gradient_norm)
        if not np.isfinite(gradient_norm) or gradient_norm <= eps:
            continue
        active_family_count += 1
        normalized = scale_tree(gradient, 1.0 / gradient_norm)
        sketch_index = int(sketch_buckets[family_index])
        contribution = scale_tree(
            normalized,
            float(sketch_signs[family_index]),
        )
        sketches[sketch_index] = (
            contribution
            if sketches[sketch_index] is None
            else add_trees(sketches[sketch_index], contribution)
        )

    basis: list[PyTree] = []
    sketch_residual_norms: list[float] = []
    for sketch in sketches:
        if sketch is None:
            sketch_residual_norms.append(0.0)
            continue
        residual = _orthogonalize(sketch, basis)
        residual_norm = float(tree_l2_norm(residual))
        sketch_residual_norms.append(residual_norm)
        if np.isfinite(residual_norm) and residual_norm > eps:
            basis.append(scale_tree(residual, 1.0 / residual_norm))
    if not basis:
        raise RuntimeError("margin-budget random sketches have zero numerical rank")

    coefficients: list[np.ndarray] = []
    raw_norms: list[float] = []
    captured_fractions: list[float] = []
    residual_norms: list[float] = []
    for family_index in range(margins.size):
        gradient = gradient_fn(family_index)
        gradient_norm = float(tree_l2_norm(gradient))
        raw_norms.append(gradient_norm)
        if not np.isfinite(gradient_norm) or gradient_norm <= eps:
            response = np.zeros((len(basis),), dtype=np.float32)
            captured_fraction = 0.0
            residual_norm = max(gradient_norm, 0.0)
        else:
            response = np.asarray(
                jax.block_until_ready(
                    jnp.stack(
                        [tree_dot(direction, gradient) for direction in basis]
                    )
                ),
                dtype=np.float32,
            )
            captured_sq = float(np.sum(response.astype(np.float64) ** 2))
            norm_sq = gradient_norm**2
            captured_fraction = float(
                np.clip(captured_sq / max(norm_sq, eps), 0.0, 1.0)
            )
            residual_norm = float(
                np.sqrt(max(norm_sq - captured_sq, 0.0))
            )
        coefficients.append(response)
        captured_fractions.append(captured_fraction)
        residual_norms.append(residual_norm)

    floors = np.minimum(
        margins,
        np.maximum(float(absolute_floor), retain_fraction * margins),
    )
    budgets = np.maximum(margins - floors, 0.0)
    coefficient_matrix = np.stack(coefficients).astype(np.float32)
    metadata = {
        "family_count": int(margins.size),
        "active_family_count": int(active_family_count),
        "requested_rank": int(requested_rank),
        "effective_rank": int(len(basis)),
        "gradient_pass_count": 2,
        "gradient_evaluation_count": int(2 * margins.size),
        "family_ids": list(family_ids),
        "initial_margins": margins.tolist(),
        "margin_floors": floors.tolist(),
        "margin_budgets": budgets.tolist(),
        "retain_fraction": float(retain_fraction),
        "absolute_floor": float(absolute_floor),
        "first_pass_gradient_norms": first_pass_norms,
        "raw_gradient_norms": raw_norms,
        "residual_gradient_norms": residual_norms,
        "captured_fractions": captured_fractions,
        "captured_fraction_mean": float(np.mean(captured_fractions)),
        "captured_fraction_min": float(np.min(captured_fractions)),
        "captured_fraction_p10": float(
            np.percentile(captured_fractions, 10)
        ),
        "gradient_norm_min": float(np.min(raw_norms)),
        "gradient_norm_max": float(np.max(raw_norms)),
        "gradient_norm_mean": float(np.mean(raw_norms)),
        "sketch_residual_norms": sketch_residual_norms,
        "sketch_type": "balanced_signed_hash",
        "sketch_buckets": sketch_buckets.astype(int).tolist(),
        "seed": int(seed),
    }
    return (
        tuple(basis),
        jnp.asarray(coefficient_matrix),
        jnp.asarray(budgets.astype(np.float32)),
        metadata,
    )


def project_optax_updates_margin_budget(
    updates: PyTree,
    basis: Sequence[PyTree],
    constraint_coefficients: jax.Array,
    budgets: jax.Array,
    spent: jax.Array,
    *,
    projection_passes: int,
    max_correction_ratio: float,
    eps: float,
) -> tuple[PyTree, jax.Array, dict[str, jax.Array]]:
    """Project only predicted margin-harming motion that exceeds a budget."""
    if not basis:
        raise ValueError("margin-budget basis is empty")
    if projection_passes <= 0:
        raise ValueError("margin-budget projection passes must be positive")
    if max_correction_ratio < 0:
        raise ValueError(
            "margin-budget maximum correction ratio must be nonnegative"
        )
    if eps <= 0:
        raise ValueError("margin-budget epsilon must be positive")

    coefficients = jnp.asarray(constraint_coefficients, dtype=jnp.float32)
    budgets = jnp.asarray(budgets, dtype=jnp.float32)
    spent = jnp.asarray(spent, dtype=jnp.float32)
    if coefficients.ndim != 2 or coefficients.shape[1] != len(basis):
        raise ValueError(
            "margin-budget coefficient matrix must have one column per basis"
        )
    if budgets.shape != (coefficients.shape[0],):
        raise ValueError("margin-budget budgets must have one value per family")
    if spent.shape != budgets.shape:
        raise ValueError("margin-budget state must match the budget vector")

    response_before = jnp.stack(
        [tree_dot(direction, updates) for direction in basis]
    ).astype(jnp.float32)
    remaining = budgets - spent
    family_count = coefficients.shape[0]

    dykstra_residuals = jnp.zeros_like(coefficients)

    def project_halfspace(
        iteration: int,
        carry: tuple[jax.Array, jax.Array],
    ) -> tuple[jax.Array, jax.Array]:
        response, residuals = carry
        family_index = iteration % family_count
        normal = coefficients[family_index]
        candidate = response + residuals[family_index]
        violation = (
            jnp.dot(normal, candidate) - remaining[family_index]
        )
        correction = (
            jnp.maximum(violation, 0.0)
            / jnp.maximum(jnp.dot(normal, normal), eps)
        )
        projected = candidate - correction * normal
        residuals = residuals.at[family_index].set(
            candidate - projected
        )
        return projected, residuals

    constrained_response, _ = jax.lax.fori_loop(
        0,
        projection_passes * family_count,
        project_halfspace,
        (response_before, dykstra_residuals),
    )
    requested_response_correction = response_before - constrained_response
    requested_correction = _linear_combination(
        basis,
        requested_response_correction,
    )
    update_norm = tree_l2_norm(updates)
    requested_correction_norm = tree_l2_norm(requested_correction)
    correction_cap = max_correction_ratio * update_norm
    correction_scale = jnp.minimum(
        jnp.asarray(1.0, dtype=jnp.float32),
        correction_cap / jnp.maximum(requested_correction_norm, eps),
    )
    applied_correction = scale_tree(requested_correction, correction_scale)
    projected = jax.tree_util.tree_map(
        lambda update, correction: (
            update - correction.astype(update.dtype)
        ),
        updates,
        applied_correction,
    )
    response_after = jnp.stack(
        [tree_dot(direction, projected) for direction in basis]
    ).astype(jnp.float32)
    predicted_change_before = coefficients @ response_before
    predicted_change_after = coefficients @ response_after
    violation_before = jnp.maximum(
        spent + predicted_change_before - budgets,
        0.0,
    )
    violation_after = jnp.maximum(
        spent + predicted_change_after - budgets,
        0.0,
    )
    # Helpful updates restore used budget but cannot create more than the
    # original allowance, which limits drift from an imperfect linear model.
    new_spent = jnp.maximum(spent + predicted_change_after, 0.0)
    applied_correction_norm = tree_l2_norm(applied_correction)
    projected_norm = tree_l2_norm(projected)
    budget_utilization = jnp.where(
        budgets > eps,
        new_spent / jnp.maximum(budgets, eps),
        (new_spent > eps).astype(jnp.float32),
    )
    stats = {
        "margin_budget/update_norm_before": update_norm,
        "margin_budget/update_norm_after": projected_norm,
        "margin_budget/correction_norm": applied_correction_norm,
        "margin_budget/correction_fraction": (
            applied_correction_norm / jnp.maximum(update_norm, eps)
        ),
        "margin_budget/correction_scale": correction_scale,
        "margin_budget/cap_binding": (
            correction_scale < 1.0
        ).astype(jnp.float32),
        "margin_budget/active_constraint_fraction": jnp.mean(
            (violation_before > eps).astype(jnp.float32)
        ),
        "margin_budget/violation_before_mean": jnp.mean(violation_before),
        "margin_budget/violation_before_max": jnp.max(violation_before),
        "margin_budget/violation_after_mean": jnp.mean(violation_after),
        "margin_budget/violation_after_max": jnp.max(violation_after),
        "margin_budget/predicted_change_before_mean": jnp.mean(
            predicted_change_before
        ),
        "margin_budget/predicted_change_after_mean": jnp.mean(
            predicted_change_after
        ),
        "margin_budget/spent_mean": jnp.mean(new_spent),
        "margin_budget/spent_max": jnp.max(new_spent),
        "margin_budget/utilization_mean": jnp.mean(budget_utilization),
        "margin_budget/utilization_max": jnp.max(budget_utilization),
        "margin_budget/exhausted_fraction": jnp.mean(
            (new_spent >= budgets - eps).astype(jnp.float32)
        ),
        "margin_budget/constraint_count": jnp.asarray(
            family_count,
            dtype=jnp.float32,
        ),
        "margin_budget/effective_rank": jnp.asarray(
            len(basis),
            dtype=jnp.float32,
        ),
    }
    return projected, new_spent, stats
