"""Low-rank functional constraints built from prompt-invariant references."""

from __future__ import annotations

from typing import Any, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from jax_sft.cnl import add_trees, scale_tree, tree_dot, tree_l2_norm

PyTree = Any


def family_balanced_constraint_batches(
    view_groups: list[dict[str, Any]],
    *,
    group_count: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[int], dict[str, float]]:
    """Flatten verified views into balanced groups of real-anchor families.

    Every retained view of one anchor stays in the same constraint group. Each
    anchor contributes unit total weight, so anchors with more retained prompt
    views do not dominate the resulting gradient.
    """
    if not view_groups:
        raise ValueError("prompt-invariant anchor families are empty")
    if group_count <= 0:
        raise ValueError("functional subspace group count must be positive")

    retained_groups = [
        group for group in view_groups if float(group.get("weight", 1.0)) > 0
    ]
    if not retained_groups:
        raise ValueError("prompt-invariant anchor families have no positive weight")
    group_count = min(group_count, len(retained_groups))
    family_order = np.random.default_rng(seed).permutation(
        len(retained_groups)
    )
    family_assignments = {
        int(family_index): int(rank % group_count)
        for rank, family_index in enumerate(family_order.tolist())
    }

    batches: list[dict[str, Any]] = []
    assignments: list[int] = []
    families_per_group = [0 for _ in range(group_count)]
    views_per_group = [0 for _ in range(group_count)]
    for family_index, family in enumerate(retained_groups):
        views = list(family.get("views") or [])
        if not views:
            raise ValueError(
                "prompt-invariant anchor family has no retained views"
            )
        family_weight = float(family.get("weight", 1.0))
        group_index = family_assignments[family_index]
        view_weight = family_weight / len(views)
        families_per_group[group_index] += 1
        views_per_group[group_index] += len(views)
        for view in views:
            if "batch" not in view:
                raise ValueError(
                    "prompt-invariant anchor view is missing its batch"
                )
            batch = dict(view["batch"])
            batch["reference_weight"] = view_weight
            batch["functional_constraint_family"] = family_index
            batch["functional_constraint_group"] = group_index
            batches.append(batch)
            assignments.append(group_index)

    return batches, assignments, {
        "reference_family_count": float(len(retained_groups)),
        "reference_view_count": float(len(batches)),
        "requested_group_count": float(group_count),
        "families_per_group_min": float(min(families_per_group)),
        "families_per_group_max": float(max(families_per_group)),
        "views_per_group_min": float(min(views_per_group)),
        "views_per_group_max": float(max(views_per_group)),
    }


def prepare_functional_constraint_subspace(
    group_grads: Sequence[PyTree],
    *,
    rcond: float,
    eps: float,
) -> tuple[tuple[PyTree, ...], jax.Array, dict[str, Any]]:
    """Normalize constraint gradients and form their stable Gram pseudoinverse."""
    if not group_grads:
        raise ValueError("functional constraint gradients are empty")
    if rcond < 0:
        raise ValueError("functional subspace rcond must be nonnegative")
    if eps <= 0:
        raise ValueError("functional subspace eps must be positive")

    raw_norms = [float(tree_l2_norm(grad)) for grad in group_grads]
    active = [
        (grad, norm)
        for grad, norm in zip(group_grads, raw_norms)
        if np.isfinite(norm) and norm > eps
    ]
    if not active:
        raise RuntimeError(
            "all prompt-invariant functional constraint gradients are zero"
        )
    basis = tuple(scale_tree(grad, 1.0 / norm) for grad, norm in active)
    gram = np.asarray(
        [
            [float(tree_dot(left, right)) for right in basis]
            for left in basis
        ],
        dtype=np.float64,
    )
    gram = 0.5 * (gram + gram.T)
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    largest = float(max(eigenvalues[-1], 0.0))
    threshold = max(eps, largest * rcond)
    keep = eigenvalues > threshold
    effective_rank = int(np.count_nonzero(keep))
    if effective_rank == 0:
        raise RuntimeError(
            "functional constraint Gram matrix has zero numerical rank"
        )
    inverse_eigenvalues = np.where(keep, 1.0 / eigenvalues, 0.0)
    gram_pinv = (
        eigenvectors
        @ np.diag(inverse_eigenvalues)
        @ eigenvectors.T
    ).astype(np.float32)
    retained_eigenvalues = eigenvalues[keep]
    condition = float(
        retained_eigenvalues[-1] / retained_eigenvalues[0]
    )
    return basis, jnp.asarray(gram_pinv), {
        "raw_group_count": len(group_grads),
        "active_group_count": len(basis),
        "dropped_zero_group_count": len(group_grads) - len(basis),
        "effective_rank": effective_rank,
        "rcond": float(rcond),
        "eigenvalue_threshold": float(threshold),
        "gram_eigenvalues": [float(value) for value in eigenvalues],
        "gram_eigenvalue_min": float(retained_eigenvalues[0]),
        "gram_eigenvalue_max": float(retained_eigenvalues[-1]),
        "gram_condition": condition,
        "group_gradient_norms": raw_norms,
        "group_gradient_norm_min": float(min(raw_norms)),
        "group_gradient_norm_max": float(max(raw_norms)),
    }


def project_optax_updates_functional_subspace(
    updates: PyTree,
    basis: Sequence[PyTree],
    gram_pinv: jax.Array,
    *,
    strength: float,
    max_projection_ratio: float,
    eps: float,
) -> tuple[PyTree, dict[str, jax.Array]]:
    """Project an additive optimizer update away from a functional subspace."""
    if not basis:
        raise ValueError("functional constraint basis is empty")
    if strength < 0:
        raise ValueError("functional subspace strength must be nonnegative")
    if max_projection_ratio < 0:
        raise ValueError(
            "functional subspace max projection ratio must be nonnegative"
        )
    if eps <= 0:
        raise ValueError("functional subspace eps must be positive")

    response_before = jnp.stack(
        [tree_dot(direction, updates) for direction in basis]
    )
    coefficients = (
        jnp.asarray(gram_pinv, dtype=jnp.float32) @ response_before
    )
    component = scale_tree(basis[0], coefficients[0])
    for direction, coefficient in zip(basis[1:], coefficients[1:]):
        component = add_trees(
            component,
            scale_tree(direction, coefficient),
        )

    requested_removal = scale_tree(component, strength)
    update_norm = tree_l2_norm(updates)
    requested_norm = tree_l2_norm(requested_removal)
    projection_cap = max_projection_ratio * update_norm
    projection_scale = jnp.minimum(
        jnp.asarray(1.0, dtype=jnp.float32),
        projection_cap / jnp.maximum(requested_norm, eps),
    )
    removal = scale_tree(requested_removal, projection_scale)
    projected = jax.tree_util.tree_map(
        lambda update, removed: update - removed.astype(update.dtype),
        updates,
        removal,
    )
    response_after = jnp.stack(
        [tree_dot(direction, projected) for direction in basis]
    )
    removed_norm = tree_l2_norm(removal)
    projected_norm = tree_l2_norm(projected)
    response_l2_before = jnp.linalg.norm(response_before)
    response_l2_after = jnp.linalg.norm(response_after)
    stats = {
        "functional_subspace/update_norm_before": update_norm,
        "functional_subspace/update_norm_after": projected_norm,
        "functional_subspace/removed_norm": removed_norm,
        "functional_subspace/removed_fraction": (
            removed_norm / jnp.maximum(update_norm, eps)
        ),
        "functional_subspace/response_l2_before": response_l2_before,
        "functional_subspace/response_l2_after": response_l2_after,
        "functional_subspace/response_reduction": (
            1.0
            - response_l2_after
            / jnp.maximum(response_l2_before, eps)
        ),
        "functional_subspace/projection_scale": projection_scale,
        "functional_subspace/cap_binding": (
            projection_scale < 1.0
        ).astype(jnp.float32),
        "functional_subspace/constraint_count": jnp.asarray(
            len(basis),
            dtype=jnp.float32,
        ),
    }
    return projected, stats
