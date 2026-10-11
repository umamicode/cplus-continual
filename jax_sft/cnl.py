"""CNL masking primitives for JAX and Optax.

The original PyTorch implementation updates a parameter element only when the
training gradient and the mastered-set reference gradient have nonnegative
product. These helpers keep that rule explicit and reusable with any Optax
optimizer.
"""

from __future__ import annotations

import math
from typing import Any, Literal

import jax
import jax.numpy as jnp
import optax

from jax_sft.gradient_blend_schema import FIXED_BLEND_TENSOR_STAT_FIELDS

PyTree = Any
MaskStage = Literal["gradient", "update"]
MaskMode = Literal["hard", "margin", "leaky"]


def _validate_mask_args(mask_mode: MaskMode, margin: float, leak: float) -> None:
    if mask_mode not in ("hard", "margin", "leaky"):
        raise ValueError("mask_mode must be 'hard', 'margin', or 'leaky'")
    if margin < 0:
        raise ValueError("margin must be nonnegative")
    if not 0 <= leak <= 1:
        raise ValueError("leak must be in [0, 1]")


def _apply_mask(
    value: jax.Array,
    score: jax.Array,
    mask_mode: MaskMode,
    margin: float,
    leak: float,
) -> jax.Array:
    if mask_mode == "margin":
        return jnp.where(score >= -margin, value, jnp.zeros_like(value))
    if mask_mode == "leaky":
        return jnp.where(score >= 0, value, value * leak)
    return jnp.where(score >= 0, value, jnp.zeros_like(value))


def mask_gradients(
    grads: PyTree,
    reference_grads: PyTree | None,
    *,
    mask_mode: MaskMode = "hard",
    margin: float = 0.0,
    leak: float = 0.0,
) -> PyTree:
    """Relax or zero gradient elements that conflict with reference gradients."""
    if reference_grads is None:
        return grads

    _validate_mask_args(mask_mode, margin, leak)
    return jax.tree_util.tree_map(
        lambda g, r: _apply_mask(g, g * r, mask_mode, margin, leak),
        grads,
        reference_grads,
    )


def mask_optax_updates(
    updates: PyTree,
    reference_grads: PyTree | None,
    *,
    mask_mode: MaskMode = "hard",
    margin: float = 0.0,
    leak: float = 0.0,
) -> PyTree:
    """Relax or zero Optax update elements whose gradient-like direction conflicts.

    Optax updates already include the descent sign, so the gradient-like
    direction is ``-updates``. This makes the mask condition match the PyTorch
    Adam/momentum variants, where the mask is computed on the positive
    optimizer direction and then subtracted from the parameters.
    """
    if reference_grads is None:
        return updates

    _validate_mask_args(mask_mode, margin, leak)
    return jax.tree_util.tree_map(
        lambda u, r: _apply_mask(u, (-u) * r, mask_mode, margin, leak),
        updates,
        reference_grads,
    )


def init_mask_ema_like(tree: PyTree) -> PyTree:
    """Create a zero-initialized EMA mask state matching a parameter/update tree."""
    return jax.tree_util.tree_map(lambda x: jnp.zeros_like(x, dtype=jnp.float32), tree)


def _divide_tree_total_by_count(total: jax.Array, count: int) -> jax.Array:
    """Divide by a static tree size without first narrowing it to int32."""
    denominator = jnp.asarray(float(max(count, 1)), dtype=total.dtype)
    return total / denominator


def _tree_mean(tree: PyTree) -> jax.Array:
    leaves = jax.tree_util.tree_leaves(tree)
    if not leaves:
        return jnp.asarray(0.0, dtype=jnp.float32)
    total = sum(
        (jnp.sum(jnp.asarray(x, dtype=jnp.float32)) for x in leaves),
        jnp.asarray(0.0, dtype=jnp.float32),
    )
    size = sum(x.size for x in leaves)
    return _divide_tree_total_by_count(total, size)


def tree_l2_norm(tree: PyTree) -> jax.Array:
    """Return the global float32 L2 norm of a pytree."""
    return jnp.sqrt(jnp.maximum(tree_l2_sq(tree), 0.0))


def tree_sum(tree: PyTree) -> jax.Array:
    """Return the float32 sum over every leaf in a pytree."""
    leaves = jax.tree_util.tree_leaves(tree)
    if not leaves:
        return jnp.asarray(0.0, dtype=jnp.float32)
    return sum(
        (jnp.sum(jnp.asarray(x, dtype=jnp.float32)) for x in leaves),
        jnp.asarray(0.0, dtype=jnp.float32),
    )


def tree_l2_sq(tree: PyTree) -> jax.Array:
    """Return the global float32 squared L2 norm of a pytree."""
    return tree_sum(
        jax.tree_util.tree_map(
            lambda x: jnp.square(jnp.asarray(x, dtype=jnp.float32)),
            tree,
        )
    )


def tree_dot(tree_a: PyTree, tree_b: PyTree) -> jax.Array:
    """Return the global float32 dot product of two matching pytrees."""
    return tree_sum(
        jax.tree_util.tree_map(
            lambda a, b: (
                jnp.asarray(a, dtype=jnp.float32)
                * jnp.asarray(b, dtype=jnp.float32)
            ),
            tree_a,
            tree_b,
        )
    )


def first_order_contribution(
    grad_tree: PyTree,
    additive_update_tree: PyTree,
) -> PyTree:
    """Return per-coordinate first-order loss changes for an additive update."""
    return jax.tree_util.tree_map(
        lambda grad, update: (
            jnp.asarray(grad, dtype=jnp.float32)
            * jnp.asarray(update, dtype=jnp.float32)
        ),
        grad_tree,
        additive_update_tree,
    )


def scale_tree(tree: PyTree, scale: jax.Array | float) -> PyTree:
    """Multiply every pytree leaf by a scalar while retaining leaf dtypes."""
    return jax.tree_util.tree_map(
        lambda x: x * jnp.asarray(scale, dtype=x.dtype),
        tree,
    )


def compute_cvar_masks(
    updates: PyTree,
    reference_group_grads: list[PyTree],
    *,
    tail_fraction: float,
) -> tuple[PyTree, PyTree, int]:
    """Compute hard coordinate masks from worst-group update alignment.

    Optax updates include the descent sign, so ``-update`` is the positive
    optimizer direction used by the CNL derivation.
    """
    if not reference_group_grads:
        raise ValueError("CVaR masking requires at least one reference group")
    if not 0 < tail_fraction <= 1:
        raise ValueError("tail_fraction must be in (0, 1]")
    tail_count = max(1, math.ceil(len(reference_group_grads) * tail_fraction))

    def score_leaf(update: jax.Array, *group_grads: jax.Array) -> jax.Array:
        direction = -jnp.asarray(update, dtype=jnp.float32)
        group_scores = jnp.stack(
            [direction * jnp.asarray(grad, dtype=jnp.float32) for grad in group_grads],
            axis=0,
        )
        return jnp.mean(jnp.sort(group_scores, axis=0)[:tail_count], axis=0)

    scores = jax.tree_util.tree_map(score_leaf, updates, *reference_group_grads)
    masks = jax.tree_util.tree_map(lambda score: (score >= 0).astype(jnp.float32), scores)
    return masks, scores, tail_count


def compute_group_consensus_gate(
    updates: PyTree,
    reference_group_grads: list[PyTree] | tuple[PyTree, ...],
    *,
    power: float = 1.0,
) -> tuple[PyTree, PyTree]:
    """Convert current group-wise safety agreement into a continuous update gate."""
    if not reference_group_grads:
        raise ValueError("group consensus requires at least one reference group")
    if power <= 0:
        raise ValueError("group consensus power must be positive")

    def safe_fraction_leaf(
        update: jax.Array,
        *group_grads: jax.Array,
    ) -> jax.Array:
        direction = -jnp.asarray(update, dtype=jnp.float32)
        safe = jnp.stack(
            [
                direction * jnp.asarray(grad, dtype=jnp.float32) >= 0
                for grad in group_grads
            ],
            axis=0,
        )
        return safe.astype(jnp.float32).mean(axis=0)

    safe_fractions = jax.tree_util.tree_map(
        safe_fraction_leaf,
        updates,
        *reference_group_grads,
    )
    gates = jax.tree_util.tree_map(
        lambda fraction: jnp.power(fraction, power),
        safe_fractions,
    )
    return gates, safe_fractions


def mask_optax_updates_group_consensus_soft_veto(
    updates: PyTree,
    synthetic_group_grads: list[PyTree] | tuple[PyTree, ...],
    anchor_group_grads: list[PyTree] | tuple[PyTree, ...],
    *,
    power: float = 1.0,
    anchor_strength: float = 1.0,
    min_gate: float = 0.0,
) -> tuple[PyTree, dict[str, jax.Array]]:
    """Apply an instantaneous soft proposal and real-anchor consensus veto."""
    if not 0 <= anchor_strength <= 1:
        raise ValueError("group consensus anchor strength must be in [0, 1]")
    if not 0 <= min_gate <= 1:
        raise ValueError("group consensus minimum gate must be in [0, 1]")

    synthetic_gates, synthetic_safe = compute_group_consensus_gate(
        updates,
        synthetic_group_grads,
        power=power,
    )
    anchor_gates, anchor_safe = compute_group_consensus_gate(
        updates,
        anchor_group_grads,
        power=power,
    )
    effective_anchor_gates = jax.tree_util.tree_map(
        lambda gate: (1.0 - anchor_strength) + anchor_strength * gate,
        anchor_gates,
    )
    combined_gates = jax.tree_util.tree_map(
        lambda synthetic, anchor: synthetic * anchor,
        synthetic_gates,
        effective_anchor_gates,
    )
    if min_gate:
        combined_gates = jax.tree_util.tree_map(
            lambda gate: min_gate + (1.0 - min_gate) * gate,
            combined_gates,
        )
    masked_updates = jax.tree_util.tree_map(
        lambda update, gate: update * gate.astype(update.dtype),
        updates,
        combined_gates,
    )
    stats = {
        "group_consensus/synthetic_gate_mean": _tree_mean(synthetic_gates),
        "group_consensus/anchor_gate_mean": _tree_mean(anchor_gates),
        "group_consensus/final_gate_mean": _tree_mean(combined_gates),
        "group_consensus/synthetic_safe_fraction_mean": _tree_mean(
            synthetic_safe
        ),
        "group_consensus/anchor_safe_fraction_mean": _tree_mean(anchor_safe),
        "group_consensus/synthetic_unanimous_safe_fraction": _tree_mean(
            jax.tree_util.tree_map(
                lambda fraction: (fraction >= 1.0).astype(jnp.float32),
                synthetic_safe,
            )
        ),
        "group_consensus/synthetic_unanimous_harmful_fraction": _tree_mean(
            jax.tree_util.tree_map(
                lambda fraction: (fraction <= 0.0).astype(jnp.float32),
                synthetic_safe,
            )
        ),
        "group_consensus/anchor_unanimous_safe_fraction": _tree_mean(
            jax.tree_util.tree_map(
                lambda fraction: (fraction >= 1.0).astype(jnp.float32),
                anchor_safe,
            )
        ),
        "group_consensus/anchor_unanimous_harmful_fraction": _tree_mean(
            jax.tree_util.tree_map(
                lambda fraction: (fraction <= 0.0).astype(jnp.float32),
                anchor_safe,
            )
        ),
        "group_consensus/synthetic_group_count": jnp.asarray(
            len(synthetic_group_grads),
            dtype=jnp.float32,
        ),
        "group_consensus/anchor_group_count": jnp.asarray(
            len(anchor_group_grads),
            dtype=jnp.float32,
        ),
        "group_consensus/anchor_strength": jnp.asarray(
            anchor_strength,
            dtype=jnp.float32,
        ),
        "group_consensus/power": jnp.asarray(power, dtype=jnp.float32),
        "group_consensus/min_gate": jnp.asarray(min_gate, dtype=jnp.float32),
        "update_norm/raw": tree_l2_norm(updates),
        "update_norm/final_structured": tree_l2_norm(masked_updates),
        "update_norm/kept_fraction": (
            tree_l2_norm(masked_updates)
            / jnp.maximum(tree_l2_norm(updates), 1e-30)
        ),
    }
    return masked_updates, stats


def _apply_soft_mask_ema(
    updates: PyTree,
    masks: PyTree,
    mask_ema: PyTree,
    mask_ema_count: jax.Array,
    *,
    beta: float,
) -> tuple[PyTree, PyTree, jax.Array, PyTree]:
    """Bias-correct a soft mask EMA and apply it to Optax updates."""
    _validate_mask_ema_beta(beta)
    beta_arr = jnp.asarray(beta, dtype=jnp.float32)
    mask_ema = jax.tree_util.tree_map(
        lambda ema, mask: (
            beta_arr * jnp.asarray(ema, dtype=jnp.float32)
            + (1.0 - beta_arr) * jnp.asarray(mask, dtype=jnp.float32)
        ),
        mask_ema,
        masks,
    )
    mask_ema_count = mask_ema_count + jnp.asarray(
        1,
        dtype=mask_ema_count.dtype,
    )
    correction = 1.0 - beta_arr ** mask_ema_count.astype(jnp.float32)
    effective_masks = jax.tree_util.tree_map(
        lambda ema: jnp.clip(
            ema / jnp.maximum(correction, jnp.asarray(1e-12, dtype=jnp.float32)),
            0.0,
            1.0,
        ),
        mask_ema,
    )
    masked_updates = jax.tree_util.tree_map(
        lambda update, mask: update * mask.astype(update.dtype),
        updates,
        effective_masks,
    )
    return masked_updates, mask_ema, mask_ema_count, effective_masks


def mask_optax_updates_lineage_veto_ema(
    updates: PyTree,
    synthetic_lineage_grads: list[PyTree] | tuple[PyTree, ...],
    anchor_lineage_grads: list[PyTree] | tuple[PyTree, ...],
    mask_ema: PyTree,
    mask_ema_count: jax.Array,
    *,
    tail_fraction: float = 0.2,
    beta: float = 0.9,
    strength: float = 1.0,
    power: float = 1.0,
    min_gate: float = 0.0,
    eps: float = 1e-12,
) -> tuple[PyTree, PyTree, jax.Array, dict[str, jax.Array]]:
    """Apply synthetic CVaR plus a lineage-supported real-anchor soft veto.

    Synthetic and real groups must use the same anchor-lineage buckets. The
    magnitude of each synthetic lineage gradient determines how much authority
    the corresponding real-anchor bucket receives at each coordinate. This
    discounts prompt-specific real vetoes in regions unsupported by that
    anchor's synthetic descendants.
    """
    if not synthetic_lineage_grads or not anchor_lineage_grads:
        raise ValueError("lineage veto requires synthetic and anchor groups")
    if len(synthetic_lineage_grads) != len(anchor_lineage_grads):
        raise ValueError("lineage veto requires equal synthetic and anchor groups")
    if not 0 <= strength <= 1:
        raise ValueError("lineage veto strength must be in [0, 1]")
    if power <= 0:
        raise ValueError("lineage veto power must be positive")
    if not 0 <= min_gate <= 1:
        raise ValueError("lineage veto minimum gate must be in [0, 1]")
    if eps <= 0:
        raise ValueError("lineage veto epsilon must be positive")

    synthetic_masks, synthetic_scores, tail_count = compute_cvar_masks(
        updates,
        list(synthetic_lineage_grads),
        tail_fraction=tail_fraction,
    )

    def lineage_gate_leaf(
        update: jax.Array,
        *group_grads: jax.Array,
    ) -> jax.Array:
        group_count = len(synthetic_lineage_grads)
        synthetic = group_grads[:group_count]
        anchors = group_grads[group_count:]
        direction = -jnp.asarray(update, dtype=jnp.float32)
        support = jnp.stack(
            [jnp.abs(jnp.asarray(grad, dtype=jnp.float32)) for grad in synthetic],
            axis=0,
        )
        anchor_harm = jnp.stack(
            [
                direction * jnp.asarray(grad, dtype=jnp.float32) < 0
                for grad in anchors
            ],
            axis=0,
        ).astype(jnp.float32)
        support_sum = jnp.sum(support, axis=0)
        support_weights = jnp.where(
            support_sum[None, ...] > eps,
            support / jnp.maximum(
                support_sum[None, ...],
                jnp.asarray(eps, dtype=jnp.float32),
            ),
            jnp.zeros_like(support),
        )
        weighted_harm = jnp.sum(support_weights * anchor_harm, axis=0)
        gate = 1.0 - jnp.asarray(strength, dtype=jnp.float32) * jnp.power(
            weighted_harm,
            jnp.asarray(power, dtype=jnp.float32),
        )
        gate = jnp.maximum(gate, jnp.asarray(min_gate, dtype=jnp.float32))
        return jnp.stack(
            [jnp.clip(gate, 0.0, 1.0), weighted_harm],
            axis=0,
        )

    gate_and_harm = jax.tree_util.tree_map(
        lineage_gate_leaf,
        updates,
        *synthetic_lineage_grads,
        *anchor_lineage_grads,
    )
    lineage_gates = jax.tree_util.tree_map(
        lambda values: values[0],
        gate_and_harm,
    )
    weighted_harm = jax.tree_util.tree_map(
        lambda values: values[1],
        gate_and_harm,
    )
    combined_masks = jax.tree_util.tree_map(
        lambda synthetic, lineage: synthetic * lineage,
        synthetic_masks,
        lineage_gates,
    )
    (
        masked_updates,
        mask_ema,
        mask_ema_count,
        effective_masks,
    ) = _apply_soft_mask_ema(
        updates,
        combined_masks,
        mask_ema,
        mask_ema_count,
        beta=beta,
    )
    raw_norm = tree_l2_norm(updates)
    final_norm = tree_l2_norm(masked_updates)
    stats = {
        "lineage_veto/synthetic_mask_mean": _tree_mean(synthetic_masks),
        "lineage_veto/synthetic_score_mean": _tree_mean(synthetic_scores),
        "lineage_veto/lineage_gate_mean": _tree_mean(lineage_gates),
        "lineage_veto/weighted_harm_mean": _tree_mean(weighted_harm),
        "lineage_veto/combined_mask_mean": _tree_mean(combined_masks),
        "lineage_veto/effective_ema_mean": _tree_mean(effective_masks),
        "lineage_veto/group_count": jnp.asarray(
            len(synthetic_lineage_grads),
            dtype=jnp.float32,
        ),
        "lineage_veto/tail_count": jnp.asarray(tail_count, dtype=jnp.float32),
        "lineage_veto/strength": jnp.asarray(strength, dtype=jnp.float32),
        "lineage_veto/power": jnp.asarray(power, dtype=jnp.float32),
        "lineage_veto/min_gate": jnp.asarray(min_gate, dtype=jnp.float32),
        "update_norm/raw": raw_norm,
        "update_norm/final_structured": final_norm,
        "update_norm/kept_fraction": final_norm / jnp.maximum(raw_norm, 1e-30),
    }
    return masked_updates, mask_ema, mask_ema_count, stats


def combine_primary_and_regularizer_gradients(
    primary_grads: PyTree,
    regularizer_grads: PyTree,
    *,
    strength: float = 0.25,
    max_norm_ratio: float = 1.0,
    eps: float = 1e-12,
) -> tuple[PyTree, dict[str, jax.Array]]:
    """Add a globally norm-capped auxiliary gradient to a primary gradient."""
    if strength < 0:
        raise ValueError("regularizer strength must be nonnegative")
    if max_norm_ratio < 0:
        raise ValueError("regularizer max norm ratio must be nonnegative")
    if eps <= 0:
        raise ValueError("regularizer epsilon must be positive")

    primary_norm = tree_l2_norm(primary_grads)
    regularizer_norm = tree_l2_norm(regularizer_grads)
    weighted_norm = jnp.asarray(strength, dtype=jnp.float32) * regularizer_norm
    allowed_norm = jnp.asarray(max_norm_ratio, dtype=jnp.float32) * primary_norm
    cap_scale = jnp.minimum(
        1.0,
        allowed_norm
        / jnp.maximum(weighted_norm, jnp.asarray(eps, dtype=jnp.float32)),
    )
    applied_strength = jnp.asarray(strength, dtype=jnp.float32) * cap_scale
    combined = jax.tree_util.tree_map(
        lambda primary, regularizer: (
            primary
            + jnp.asarray(applied_strength, dtype=primary.dtype)
            * regularizer.astype(primary.dtype)
        ),
        primary_grads,
        regularizer_grads,
    )
    dot = tree_dot(primary_grads, regularizer_grads)
    cosine = dot / jnp.maximum(
        primary_norm * regularizer_norm,
        jnp.asarray(eps, dtype=jnp.float32),
    )
    stats = {
        "on_policy_regularizer/primary_grad_norm": primary_norm,
        "on_policy_regularizer/rewrite_grad_norm": regularizer_norm,
        "on_policy_regularizer/gradient_cosine": cosine,
        "on_policy_regularizer/requested_strength": jnp.asarray(
            strength,
            dtype=jnp.float32,
        ),
        "on_policy_regularizer/applied_strength": applied_strength,
        "on_policy_regularizer/cap_binding": (cap_scale < 1.0).astype(
            jnp.float32
        ),
        "on_policy_regularizer/combined_grad_norm": tree_l2_norm(combined),
    }
    return combined, stats


def project_optax_updates_functional_trust_region_ema(
    updates: PyTree,
    functional_grad: PyTree,
    functional_loss: jax.Array,
    gradient_ema: PyTree,
    gradient_ema_count: jax.Array,
    *,
    radius: float,
    beta: float = 0.9,
    strength: float = 1.0,
    max_correction_ratio: float = 1.0,
    eps: float = 1e-12,
) -> tuple[PyTree, PyTree, jax.Array, dict[str, jax.Array]]:
    """Project a candidate update back toward a functional KL trust region.

    ``functional_grad`` is evaluated at the candidate parameters after applying
    ``updates``. The correction follows a bias-corrected EMA of functional
    gradients across prompt views. When that direction is not a descent
    direction for the current view, the instantaneous gradient is used.
    """
    _validate_mask_ema_beta(beta)
    if radius < 0:
        raise ValueError("functional trust-region radius must be nonnegative")
    if strength < 0:
        raise ValueError("functional trust-region strength must be nonnegative")
    if max_correction_ratio < 0:
        raise ValueError(
            "functional trust-region max correction ratio must be nonnegative"
        )
    if eps <= 0:
        raise ValueError("functional trust-region epsilon must be positive")

    beta_arr = jnp.asarray(beta, dtype=jnp.float32)
    gradient_ema = jax.tree_util.tree_map(
        lambda ema, grad: (
            beta_arr * jnp.asarray(ema, dtype=jnp.float32)
            + (1.0 - beta_arr) * jnp.asarray(grad, dtype=jnp.float32)
        ),
        gradient_ema,
        functional_grad,
    )
    gradient_ema_count = gradient_ema_count + jnp.asarray(
        1,
        dtype=gradient_ema_count.dtype,
    )
    correction = 1.0 - beta_arr ** gradient_ema_count.astype(jnp.float32)
    effective_gradient = jax.tree_util.tree_map(
        lambda ema: ema / jnp.maximum(correction, jnp.asarray(eps, dtype=jnp.float32)),
        gradient_ema,
    )

    ema_directional_derivative = tree_dot(
        functional_grad,
        effective_gradient,
    )
    instantaneous_gradient_norm_sq = tree_l2_sq(functional_grad)
    use_instantaneous = ema_directional_derivative <= jnp.asarray(
        eps,
        dtype=jnp.float32,
    )
    projection_direction = jax.tree_util.tree_map(
        lambda ema_grad, current_grad: jnp.where(
            use_instantaneous,
            jnp.asarray(current_grad, dtype=jnp.float32),
            jnp.asarray(ema_grad, dtype=jnp.float32),
        ),
        effective_gradient,
        functional_grad,
    )
    directional_derivative = jnp.where(
        use_instantaneous,
        instantaneous_gradient_norm_sq,
        ema_directional_derivative,
    )

    violation = jnp.maximum(
        jnp.asarray(functional_loss, dtype=jnp.float32)
        - jnp.asarray(radius, dtype=jnp.float32),
        0.0,
    )
    coefficient = (
        jnp.asarray(strength, dtype=jnp.float32)
        * violation
        / jnp.maximum(directional_derivative, jnp.asarray(eps, dtype=jnp.float32))
    )
    raw_correction = scale_tree(projection_direction, coefficient)
    raw_correction_norm = tree_l2_norm(raw_correction)
    update_norm = tree_l2_norm(updates)
    correction_cap = (
        jnp.asarray(max_correction_ratio, dtype=jnp.float32) * update_norm
    )
    cap_scale = jnp.minimum(
        1.0,
        correction_cap
        / jnp.maximum(raw_correction_norm, jnp.asarray(eps, dtype=jnp.float32)),
    )
    applied_correction = scale_tree(raw_correction, cap_scale)
    projected_updates = jax.tree_util.tree_map(
        lambda update, correction_value: (
            update - correction_value.astype(update.dtype)
        ),
        updates,
        applied_correction,
    )
    applied_correction_norm = tree_l2_norm(applied_correction)
    predicted_reduction = coefficient * cap_scale * directional_derivative
    linearized_loss_after = jnp.maximum(
        jnp.asarray(functional_loss, dtype=jnp.float32) - predicted_reduction,
        0.0,
    )
    stats = {
        "functional_trust/candidate_kl": jnp.asarray(
            functional_loss,
            dtype=jnp.float32,
        ),
        "functional_trust/radius": jnp.asarray(radius, dtype=jnp.float32),
        "functional_trust/violation": violation,
        "functional_trust/active": (violation > 0).astype(jnp.float32),
        "functional_trust/ema_fallback": use_instantaneous.astype(jnp.float32),
        "functional_trust/directional_derivative": directional_derivative,
        "functional_trust/correction_norm": applied_correction_norm,
        "functional_trust/correction_to_update_norm": (
            applied_correction_norm
            / jnp.maximum(update_norm, jnp.asarray(eps, dtype=jnp.float32))
        ),
        "functional_trust/cap_binding": (cap_scale < 1.0).astype(jnp.float32),
        "functional_trust/linearized_kl_after": linearized_loss_after,
        "update_norm/raw": update_norm,
        "update_norm/final_structured": tree_l2_norm(projected_updates),
        "update_norm/kept_fraction": (
            tree_l2_norm(projected_updates)
            / jnp.maximum(update_norm, jnp.asarray(eps, dtype=jnp.float32))
        ),
    }
    return (
        projected_updates,
        gradient_ema,
        gradient_ema_count,
        stats,
    )


def mask_optax_updates_cvar_ema(
    updates: PyTree,
    reference_group_grads: list[PyTree],
    mask_ema: PyTree,
    mask_ema_count: jax.Array,
    *,
    tail_fraction: float = 0.2,
    beta: float = 0.9,
) -> tuple[PyTree, PyTree, jax.Array, dict[str, jax.Array]]:
    """Apply a bias-corrected EMA of a grouped worst-tail CVaR mask."""
    _validate_mask_ema_beta(beta)
    hard_masks, scores, tail_count = compute_cvar_masks(
        updates,
        reference_group_grads,
        tail_fraction=tail_fraction,
    )
    beta_arr = jnp.asarray(beta, dtype=jnp.float32)
    mask_ema = jax.tree_util.tree_map(
        lambda ema, hard: beta_arr * ema + (1.0 - beta_arr) * hard,
        mask_ema,
        hard_masks,
    )
    mask_ema_count = mask_ema_count + jnp.asarray(1, dtype=mask_ema_count.dtype)
    correction = 1.0 - beta_arr ** mask_ema_count.astype(jnp.float32)
    effective_masks = jax.tree_util.tree_map(
        lambda ema: jnp.clip(ema / correction, 0.0, 1.0),
        mask_ema,
    )
    masked_updates = jax.tree_util.tree_map(
        lambda update, mask: update * mask.astype(update.dtype),
        updates,
        effective_masks,
    )
    stats = {
        "cnl/synth_teacher_cvar_mask_mean": _tree_mean(hard_masks),
        "cnl/synth_teacher_cvar_ema_mask_mean": _tree_mean(effective_masks),
        "cnl/synth_teacher_cvar_score_mean": _tree_mean(scores),
        "cnl/synth_teacher_cvar_tail_count": jnp.asarray(tail_count, dtype=jnp.float32),
        "cnl/synth_teacher_update_norm_before": tree_l2_norm(updates),
        "cnl/synth_teacher_update_norm_after": tree_l2_norm(masked_updates),
    }
    return masked_updates, mask_ema, mask_ema_count, stats


def mask_optax_updates_cvar_ema_reference_projection(
    updates: PyTree,
    reference_group_grads: list[PyTree],
    real_anchor_grad: PyTree,
    mask_ema: PyTree,
    mask_ema_count: jax.Array,
    *,
    tail_fraction: float = 0.2,
    beta: float = 0.9,
    eps: float = 1e-12,
) -> tuple[PyTree, PyTree, jax.Array, dict[str, jax.Array]]:
    """Apply synthetic CVaR+EMA, then project inside its allowed support.

    The correction direction is the real-anchor gradient multiplied by the
    effective synthetic mask. This prevents the global projection from
    reviving coordinates that synthetic CNL has frozen.
    """
    if eps <= 0:
        raise ValueError("reference projection epsilon must be positive")
    _validate_mask_ema_beta(beta)
    hard_masks, scores, tail_count = compute_cvar_masks(
        updates,
        reference_group_grads,
        tail_fraction=tail_fraction,
    )
    (
        synthetic_updates,
        mask_ema,
        mask_ema_count,
        effective_masks,
    ) = _apply_soft_mask_ema(
        updates,
        hard_masks,
        mask_ema,
        mask_ema_count,
        beta=beta,
    )
    allowed_anchor_direction = jax.tree_util.tree_map(
        lambda grad, mask: (
            jnp.asarray(grad, dtype=jnp.float32)
            * jnp.asarray(mask, dtype=jnp.float32)
        ),
        real_anchor_grad,
        effective_masks,
    )
    damage_before = tree_dot(real_anchor_grad, synthetic_updates)
    denominator = tree_dot(
        real_anchor_grad,
        allowed_anchor_direction,
    )
    projection_scale = jax.nn.relu(damage_before) / jnp.maximum(
        denominator,
        jnp.asarray(eps, dtype=jnp.float32),
    )
    projected_updates = jax.tree_util.tree_map(
        lambda update, direction: (
            update
            - jnp.asarray(projection_scale, dtype=update.dtype)
            * jnp.asarray(direction, dtype=update.dtype)
        ),
        synthetic_updates,
        allowed_anchor_direction,
    )
    damage_after = tree_dot(real_anchor_grad, projected_updates)
    correction = jax.tree_util.tree_map(
        lambda before, after: before - after,
        synthetic_updates,
        projected_updates,
    )
    raw_norm = tree_l2_norm(updates)
    final_norm = tree_l2_norm(projected_updates)
    stats = {
        "cnl/synth_teacher_cvar_mask_mean": _tree_mean(hard_masks),
        "cnl/synth_teacher_cvar_ema_mask_mean": _tree_mean(
            effective_masks
        ),
        "cnl/synth_teacher_cvar_score_mean": _tree_mean(scores),
        "cnl/synth_teacher_cvar_tail_count": jnp.asarray(
            tail_count,
            dtype=jnp.float32,
        ),
        "cnl/synth_teacher_update_norm_before": raw_norm,
        "cnl/synth_teacher_update_norm_after": tree_l2_norm(
            synthetic_updates
        ),
        "real_projection/damage_before": damage_before,
        "real_projection/damage_after": damage_after,
        "real_projection/conflict": (damage_before > 0).astype(jnp.float32),
        "real_projection/scale": projection_scale,
        "real_projection/correction_norm": tree_l2_norm(correction),
        "real_projection/correction_to_update_norm": (
            tree_l2_norm(correction)
            / jnp.maximum(raw_norm, jnp.asarray(eps, dtype=jnp.float32))
        ),
        "update_norm/raw": raw_norm,
        "update_norm/synthetic_masked": tree_l2_norm(synthetic_updates),
        "update_norm/final_structured": final_norm,
        "update_norm/kept_fraction": (
            final_norm
            / jnp.maximum(raw_norm, jnp.asarray(eps, dtype=jnp.float32))
        ),
    }
    return projected_updates, mask_ema, mask_ema_count, stats


def mask_optax_updates_anchor_veto_ema(
    updates: PyTree,
    synthetic_group_grads: list[PyTree],
    anchor_group_grads: list[PyTree],
    mask_ema: PyTree,
    mask_ema_count: jax.Array,
    *,
    tail_fraction: float = 0.2,
    beta: float = 0.9,
) -> tuple[PyTree, PyTree, jax.Array, dict[str, jax.Array]]:
    """Apply synthetic-CVaR AND anchor-CVaR veto followed by mask EMA."""
    _validate_mask_ema_beta(beta)
    synthetic_masks, synthetic_scores, synthetic_tail_count = compute_cvar_masks(
        updates,
        synthetic_group_grads,
        tail_fraction=tail_fraction,
    )
    anchor_masks, anchor_scores, anchor_tail_count = compute_cvar_masks(
        updates,
        anchor_group_grads,
        tail_fraction=tail_fraction,
    )
    hard_masks = jax.tree_util.tree_map(
        lambda synthetic, anchor: synthetic * anchor,
        synthetic_masks,
        anchor_masks,
    )
    veto_masks = jax.tree_util.tree_map(
        lambda synthetic, anchor: ((synthetic >= 0.5) & (anchor < 0.5)).astype(jnp.float32),
        synthetic_masks,
        anchor_masks,
    )

    beta_arr = jnp.asarray(beta, dtype=jnp.float32)
    mask_ema = jax.tree_util.tree_map(
        lambda ema, hard: beta_arr * ema + (1.0 - beta_arr) * hard,
        mask_ema,
        hard_masks,
    )
    mask_ema_count = mask_ema_count + jnp.asarray(1, dtype=mask_ema_count.dtype)
    correction = 1.0 - beta_arr ** mask_ema_count.astype(jnp.float32)
    effective_masks = jax.tree_util.tree_map(
        lambda ema: jnp.clip(ema / correction, 0.0, 1.0),
        mask_ema,
    )
    masked_updates = jax.tree_util.tree_map(
        lambda update, mask: update * mask.astype(update.dtype),
        updates,
        effective_masks,
    )
    stats = {
        "anchor_veto/synth_mask_mean": _tree_mean(synthetic_masks),
        "anchor_veto/anchor_mask_mean": _tree_mean(anchor_masks),
        "anchor_veto/combined_mask_mean": _tree_mean(hard_masks),
        "anchor_veto/ema_mask_mean": _tree_mean(effective_masks),
        "anchor_veto/ema_mask_frac_gt_0_5": _tree_mean(
            jax.tree_util.tree_map(
                lambda mask: (mask > 0.5).astype(jnp.float32),
                effective_masks,
            )
        ),
        "anchor_veto/veto_fraction": _tree_mean(veto_masks),
        "anchor_veto/synth_score_mean": _tree_mean(synthetic_scores),
        "anchor_veto/anchor_score_mean": _tree_mean(anchor_scores),
        "anchor_veto/synth_tail_count": jnp.asarray(synthetic_tail_count, dtype=jnp.float32),
        "anchor_veto/anchor_tail_count": jnp.asarray(anchor_tail_count, dtype=jnp.float32),
        "anchor_veto/update_norm_before": tree_l2_norm(updates),
        "anchor_veto/update_norm_after": tree_l2_norm(masked_updates),
    }
    return masked_updates, mask_ema, mask_ema_count, stats


def mask_optax_updates_on_policy_cpl_ema(
    updates: PyTree,
    original_grads: PyTree,
    on_policy_grads: PyTree,
    synthetic_group_grads: list[PyTree] | tuple[PyTree, ...],
    anchor_group_grads: list[PyTree] | tuple[PyTree, ...],
    mask_ema: PyTree,
    mask_ema_count: jax.Array,
    *,
    tail_fraction: float = 0.2,
    beta: float = 0.9,
    acquisition_floor: float = 0.8,
    support_eps: float = 1e-12,
) -> tuple[PyTree, PyTree, jax.Array, dict[str, jax.Array]]:
    """Transfer an on-policy compatibility mask to a gold SFT update.

    The on-policy injection gradient is used only to estimate compatibility
    with on-policy mastered gradients. The actual update remains the original
    gold-labelled SFT update. If masking removes too much first-order task
    utility, the smallest interpolation back toward the raw update that meets
    ``acquisition_floor`` is restored.
    """
    _validate_mask_ema_beta(beta)
    if not 0 <= acquisition_floor <= 1:
        raise ValueError("on-policy acquisition floor must be in [0, 1]")
    if support_eps < 0:
        raise ValueError("on-policy support epsilon must be nonnegative")
    if not synthetic_group_grads:
        raise ValueError("on-policy CPL requires synthetic reference groups")
    if not anchor_group_grads:
        raise ValueError("on-policy CPL requires real-anchor reference groups")

    proxy_updates = jax.tree_util.tree_map(
        lambda grad: -jnp.asarray(grad),
        on_policy_grads,
    )
    synthetic_masks, synthetic_scores, synthetic_tail_count = (
        compute_cvar_masks(
            proxy_updates,
            list(synthetic_group_grads),
            tail_fraction=tail_fraction,
        )
    )
    anchor_masks, anchor_scores, anchor_tail_count = compute_cvar_masks(
        proxy_updates,
        list(anchor_group_grads),
        tail_fraction=tail_fraction,
    )

    def transfer_mask(update: jax.Array, proxy_grad: jax.Array) -> jax.Array:
        direction = -jnp.asarray(update, dtype=jnp.float32)
        proxy = jnp.asarray(proxy_grad, dtype=jnp.float32)
        supported = jnp.abs(proxy) > jnp.asarray(
            support_eps,
            dtype=jnp.float32,
        )
        return ((direction * proxy >= 0) & supported).astype(jnp.float32)

    transfer_masks = jax.tree_util.tree_map(
        transfer_mask,
        updates,
        on_policy_grads,
    )
    compatibility_masks = jax.tree_util.tree_map(
        lambda synthetic, anchor, transfer: synthetic * anchor * transfer,
        synthetic_masks,
        anchor_masks,
        transfer_masks,
    )
    (
        compatibility_updates,
        mask_ema,
        mask_ema_count,
        effective_masks,
    ) = _apply_soft_mask_ema(
        updates,
        compatibility_masks,
        mask_ema,
        mask_ema_count,
        beta=beta,
    )

    raw_utility = -tree_dot(original_grads, updates)
    compatibility_utility = -tree_dot(
        original_grads,
        compatibility_updates,
    )
    target_utility = jnp.asarray(
        acquisition_floor,
        dtype=jnp.float32,
    ) * jnp.maximum(raw_utility, 0.0)
    available_utility = raw_utility - compatibility_utility
    missing_utility = jax.nn.relu(target_utility - compatibility_utility)
    rescue_alpha = jnp.where(
        (missing_utility > 0)
        & (available_utility > jnp.asarray(support_eps, dtype=jnp.float32)),
        jnp.clip(
            missing_utility
            / jnp.maximum(
                available_utility,
                jnp.asarray(support_eps, dtype=jnp.float32),
            ),
            0.0,
            1.0,
        ),
        0.0,
    )
    final_updates = jax.tree_util.tree_map(
        lambda compatible, raw: (
            compatible
            + jnp.asarray(rescue_alpha, dtype=raw.dtype) * (raw - compatible)
        ),
        compatibility_updates,
        updates,
    )
    final_utility = -tree_dot(original_grads, final_updates)
    raw_norm = tree_l2_norm(updates)
    compatibility_norm = tree_l2_norm(compatibility_updates)
    final_norm = tree_l2_norm(final_updates)
    original_norm = tree_l2_norm(original_grads)
    on_policy_norm = tree_l2_norm(on_policy_grads)
    gradient_cosine = tree_dot(original_grads, on_policy_grads) / jnp.maximum(
        original_norm * on_policy_norm,
        jnp.asarray(max(support_eps, 1e-30), dtype=jnp.float32),
    )
    stats = {
        "on_policy_mask/synthetic_mask_mean": _tree_mean(synthetic_masks),
        "on_policy_mask/anchor_mask_mean": _tree_mean(anchor_masks),
        "on_policy_mask/transfer_mask_mean": _tree_mean(transfer_masks),
        "on_policy_mask/compatibility_mask_mean": _tree_mean(
            compatibility_masks
        ),
        "on_policy_mask/effective_ema_mean": _tree_mean(effective_masks),
        "on_policy_mask/synthetic_score_mean": _tree_mean(synthetic_scores),
        "on_policy_mask/anchor_score_mean": _tree_mean(anchor_scores),
        "on_policy_mask/synthetic_tail_count": jnp.asarray(
            synthetic_tail_count,
            dtype=jnp.float32,
        ),
        "on_policy_mask/anchor_tail_count": jnp.asarray(
            anchor_tail_count,
            dtype=jnp.float32,
        ),
        "on_policy_mask/original_grad_norm": original_norm,
        "on_policy_mask/proxy_grad_norm": on_policy_norm,
        "on_policy_mask/original_proxy_cosine": gradient_cosine,
        "on_policy_mask/raw_task_utility": raw_utility,
        "on_policy_mask/compatibility_task_utility": compatibility_utility,
        "on_policy_mask/target_task_utility": target_utility,
        "on_policy_mask/final_task_utility": final_utility,
        "on_policy_mask/acquisition_floor": jnp.asarray(
            acquisition_floor,
            dtype=jnp.float32,
        ),
        "on_policy_mask/rescue_alpha": rescue_alpha,
        "on_policy_mask/rescue_binding": (rescue_alpha > 0).astype(
            jnp.float32
        ),
        "on_policy_mask/acquisition_floor_satisfied": (
            final_utility + jnp.asarray(max(support_eps, 1e-12))
            >= target_utility
        ).astype(jnp.float32),
        "update_norm/raw": raw_norm,
        "update_norm/on_policy_compatible": compatibility_norm,
        "update_norm/final_structured": final_norm,
        "update_norm/compatibility_kept_fraction": (
            compatibility_norm / jnp.maximum(raw_norm, 1e-30)
        ),
        "update_norm/kept_fraction": (
            final_norm / jnp.maximum(raw_norm, 1e-30)
        ),
    }
    return final_updates, mask_ema, mask_ema_count, stats


def mask_optax_updates_minveto_ema(
    updates: PyTree,
    current_grads: PyTree,
    synthetic_group_grads: list[PyTree],
    anchor_grad: PyTree,
    mask_ema: PyTree,
    mask_ema_count: jax.Array,
    *,
    tail_fraction: float = 0.2,
    beta: float = 0.9,
    harm_reduction_fraction: float = 0.75,
    energy_cap: float = 0.10,
    weight_eps: float = 1e-12,
    bisection_iterations: int = 12,
    max_expansions: int = 32,
    lambda_initial_high: float = 1.0,
    bisection_tolerance: float = 1e-6,
) -> tuple[PyTree, PyTree, jax.Array, dict[str, jax.Array]]:
    """Apply synthetic CVaR followed by a minimum-change real-anchor gate.

    The real-anchor constraint is solved on the instantaneous synthetic
    proposal. The bias-corrected EMA gate is then applied to the raw Optax
    update, matching the existing update-stage CNL state semantics.
    """
    _validate_mask_ema_beta(beta)
    if not 0 <= harm_reduction_fraction <= 1:
        raise ValueError("minveto harm reduction fraction must be in [0, 1]")
    if not 0 <= energy_cap <= 1:
        raise ValueError("minveto energy cap must be in [0, 1]")
    if weight_eps <= 0:
        raise ValueError("minveto weight epsilon must be positive")
    if bisection_iterations <= 0:
        raise ValueError("minveto bisection iterations must be positive")
    if max_expansions < 0:
        raise ValueError("minveto max expansions must be nonnegative")
    if lambda_initial_high <= 0:
        raise ValueError("minveto initial lambda high must be positive")
    if bisection_tolerance < 0:
        raise ValueError("minveto bisection tolerance must be nonnegative")

    synthetic_masks, synthetic_scores, synthetic_tail_count = compute_cvar_masks(
        updates,
        synthetic_group_grads,
        tail_fraction=tail_fraction,
    )
    proposal = jax.tree_util.tree_map(
        lambda update, mask: (
            jnp.asarray(update, dtype=jnp.float32)
            * jnp.asarray(mask, dtype=jnp.float32)
        ),
        updates,
        synthetic_masks,
    )
    anchor_contribution = first_order_contribution(anchor_grad, proposal)
    current_utility = jax.tree_util.tree_map(
        lambda grad, delta: jax.nn.relu(
            -jnp.asarray(grad, dtype=jnp.float32)
            * jnp.asarray(delta, dtype=jnp.float32)
        ),
        current_grads,
        proposal,
    )
    utility_weights = jax.tree_util.tree_map(
        lambda utility: utility + jnp.asarray(weight_eps, dtype=jnp.float32),
        current_utility,
    )

    damage_before = tree_sum(anchor_contribution)
    damage_tolerance = (
        jnp.asarray(bisection_tolerance, dtype=jnp.float32)
        * jnp.maximum(
            jnp.abs(damage_before),
            jnp.asarray(weight_eps, dtype=jnp.float32),
        )
    )
    needs_correction = (damage_before > 0) & (
        jnp.asarray(harm_reduction_fraction, dtype=jnp.float32) > 0
    )
    damage_target = jnp.where(
        damage_before > 0,
        (1.0 - jnp.asarray(harm_reduction_fraction, dtype=jnp.float32))
        * damage_before,
        damage_before,
    )

    def real_gate(lam: jax.Array) -> PyTree:
        return jax.tree_util.tree_map(
            lambda harm, weight: jnp.clip(
                1.0
                - jnp.asarray(lam, dtype=jnp.float32)
                * jnp.asarray(harm, dtype=jnp.float32)
                / jnp.asarray(weight, dtype=jnp.float32),
                0.0,
                1.0,
            ),
            anchor_contribution,
            utility_weights,
        )

    def predicted_damage(lam: jax.Array) -> jax.Array:
        return tree_dot(anchor_contribution, real_gate(lam))

    proposal_energy = tree_l2_sq(proposal)

    def removed_energy_fraction(lam: jax.Array) -> jax.Array:
        gate = real_gate(lam)
        removed = jax.tree_util.tree_map(
            lambda delta, real: (
                (1.0 - jnp.asarray(real, dtype=jnp.float32))
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
        ) & (
            predicted_damage(high) > damage_target
        )

    def expand_high(
        state: tuple[jax.Array, jax.Array],
    ) -> tuple[jax.Array, jax.Array]:
        high, count = state
        return high * 2.0, count + jnp.asarray(1, dtype=jnp.int32)

    def bisect_harm(
        _: int,
        bounds: tuple[jax.Array, jax.Array],
    ) -> tuple[jax.Array, jax.Array]:
        low, high = bounds
        middle = 0.5 * (low + high)
        too_harmful = predicted_damage(middle) > damage_target
        return (
            jnp.where(too_harmful, middle, low),
            jnp.where(too_harmful, high, middle),
        )

    def solve_harm(_: None) -> tuple[jax.Array, jax.Array]:
        expanded_high, expansion_count = jax.lax.while_loop(
            expansion_needed,
            expand_high,
            (initial_high, jnp.asarray(0, dtype=jnp.int32)),
        )
        _, lambda_harm = jax.lax.fori_loop(
            0,
            bisection_iterations,
            bisect_harm,
            (jnp.asarray(0.0, dtype=jnp.float32), expanded_high),
        )
        return lambda_harm, expansion_count

    lambda_harm, expansion_count = jax.lax.cond(
        needs_correction,
        solve_harm,
        lambda _: (
            jnp.asarray(0.0, dtype=jnp.float32),
            jnp.asarray(0, dtype=jnp.int32),
        ),
        operand=None,
    )
    cap_binding = needs_correction & (
        removed_energy_fraction(lambda_harm)
        > jnp.asarray(energy_cap + bisection_tolerance, dtype=jnp.float32)
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
            (jnp.asarray(0.0, dtype=jnp.float32), lambda_harm),
        )
        return lambda_cap

    lambda_cap = jax.lax.cond(
        cap_binding,
        solve_cap,
        lambda _: lambda_harm,
        operand=None,
    )
    lambda_final = jnp.minimum(lambda_harm, lambda_cap)
    final_real_gate = real_gate(lambda_final)
    instantaneous_gate = jax.tree_util.tree_map(
        lambda synthetic, real: (
            jnp.asarray(synthetic, dtype=jnp.float32)
            * jnp.asarray(real, dtype=jnp.float32)
        ),
        synthetic_masks,
        final_real_gate,
    )
    instantaneous_update = jax.tree_util.tree_map(
        lambda update, gate: (
            jnp.asarray(update, dtype=jnp.float32)
            * jnp.asarray(gate, dtype=jnp.float32)
        ),
        updates,
        instantaneous_gate,
    )

    beta_arr = jnp.asarray(beta, dtype=jnp.float32)
    mask_ema = jax.tree_util.tree_map(
        lambda ema, gate: (
            beta_arr * jnp.asarray(ema, dtype=jnp.float32)
            + (1.0 - beta_arr) * jnp.asarray(gate, dtype=jnp.float32)
        ),
        mask_ema,
        instantaneous_gate,
    )
    mask_ema_count = mask_ema_count + jnp.asarray(
        1,
        dtype=mask_ema_count.dtype,
    )
    correction = 1.0 - beta_arr ** mask_ema_count.astype(jnp.float32)
    effective_gate = jax.tree_util.tree_map(
        lambda ema: jnp.clip(ema / correction, 0.0, 1.0),
        mask_ema,
    )
    final_updates = jax.tree_util.tree_map(
        lambda update, gate: (
            update * jnp.asarray(gate, dtype=update.dtype)
        ),
        updates,
        effective_gate,
    )

    active = jax.tree_util.tree_map(
        lambda delta: (jnp.asarray(delta) != 0),
        proposal,
    )
    active_count = tree_sum(
        jax.tree_util.tree_map(
            lambda selected: selected.astype(jnp.float32),
            active,
        )
    )
    active_denominator = jnp.maximum(active_count, 1.0)
    coordinate_tolerance = jnp.asarray(1e-6, dtype=jnp.float32)

    def active_fraction(predicate: PyTree) -> jax.Array:
        return tree_sum(
            jax.tree_util.tree_map(
                lambda keep, selected: (
                    jnp.asarray(keep, dtype=jnp.float32)
                    * selected.astype(jnp.float32)
                ),
                predicate,
                active,
            )
        ) / active_denominator

    gate_sum = tree_sum(
        jax.tree_util.tree_map(
            lambda gate, selected: (
                jnp.asarray(gate, dtype=jnp.float32)
                * selected.astype(jnp.float32)
            ),
            final_real_gate,
            active,
        )
    )
    gate_sq_sum = tree_sum(
        jax.tree_util.tree_map(
            lambda gate, selected: (
                jnp.square(jnp.asarray(gate, dtype=jnp.float32))
                * selected.astype(jnp.float32)
            ),
            final_real_gate,
            active,
        )
    )
    gate_mean = gate_sum / active_denominator
    gate_std = jnp.sqrt(
        jnp.maximum(gate_sq_sum / active_denominator - gate_mean**2, 0.0)
    )
    gate_min = jnp.asarray(jnp.inf, dtype=jnp.float32)
    gate_max = jnp.asarray(-jnp.inf, dtype=jnp.float32)
    for gate, selected in zip(
        jax.tree_util.tree_leaves(final_real_gate),
        jax.tree_util.tree_leaves(active),
    ):
        gate = jnp.asarray(gate, dtype=jnp.float32)
        gate_min = jnp.minimum(
            gate_min,
            jnp.min(jnp.where(selected, gate, jnp.inf)),
        )
        gate_max = jnp.maximum(
            gate_max,
            jnp.max(jnp.where(selected, gate, -jnp.inf)),
        )
    gate_min = jnp.where(active_count > 0, gate_min, 0.0)
    gate_max = jnp.where(active_count > 0, gate_max, 0.0)

    damage_instant_after = tree_dot(anchor_grad, instantaneous_update)
    damage_ema_after = tree_dot(anchor_grad, final_updates)
    utility_total = tree_sum(current_utility)
    utility_removed = tree_sum(
        jax.tree_util.tree_map(
            lambda utility, gate: (
                jnp.asarray(utility, dtype=jnp.float32)
                * (1.0 - jnp.asarray(gate, dtype=jnp.float32))
            ),
            current_utility,
            final_real_gate,
        )
    )
    damage_reduction = jnp.where(
        damage_before > 0,
        (damage_before - damage_instant_after)
        / jnp.maximum(damage_before, jnp.asarray(weight_eps, dtype=jnp.float32)),
        0.0,
    )
    stats = {
        "minveto/enabled": jnp.asarray(1.0, dtype=jnp.float32),
        "minveto/harm_reduction_fraction": jnp.asarray(
            harm_reduction_fraction,
            dtype=jnp.float32,
        ),
        "minveto/energy_cap": jnp.asarray(energy_cap, dtype=jnp.float32),
        "minveto/lambda_harm": lambda_harm,
        "minveto/lambda_cap": lambda_cap,
        "minveto/lambda_final": lambda_final,
        "minveto/cap_binding": cap_binding.astype(jnp.float32),
        "minveto/target_feasible_under_cap": (
            damage_instant_after
            <= damage_target + damage_tolerance
        ).astype(jnp.float32),
        "minveto/predicted_damage_before": damage_before,
        "minveto/predicted_damage_target": damage_target,
        "minveto/predicted_damage_instant_after": damage_instant_after,
        "minveto/predicted_damage_ema_after": damage_ema_after,
        "minveto/damage_reduction_fraction_achieved": damage_reduction,
        "minveto/damage_target_shortfall": jax.nn.relu(
            damage_instant_after - damage_target
        ),
        "minveto/update_energy_removed_fraction": removed_energy_fraction(
            lambda_final
        ),
        "minveto/fraction_coordinates_unchanged": active_fraction(
            jax.tree_util.tree_map(
                lambda gate: gate >= 1.0 - coordinate_tolerance,
                final_real_gate,
            )
        ),
        "minveto/fraction_coordinates_softened": active_fraction(
            jax.tree_util.tree_map(
                lambda gate: (
                    (gate > coordinate_tolerance)
                    & (gate < 1.0 - coordinate_tolerance)
                ),
                final_real_gate,
            )
        ),
        "minveto/fraction_coordinates_zeroed": active_fraction(
            jax.tree_util.tree_map(
                lambda gate: gate <= coordinate_tolerance,
                final_real_gate,
            )
        ),
        "minveto/gate_mean": gate_mean,
        "minveto/gate_std": gate_std,
        "minveto/gate_min": gate_min,
        "minveto/gate_max": gate_max,
        "minveto/current_utility_removed": utility_removed,
        "minveto/current_utility_kept_fraction": (
            1.0
            - utility_removed
            / jnp.maximum(
                utility_total,
                jnp.asarray(weight_eps, dtype=jnp.float32),
            )
        ),
        "minveto/bisection_iterations": jnp.where(
            needs_correction,
            jnp.asarray(bisection_iterations, dtype=jnp.float32)
            * (1.0 + cap_binding.astype(jnp.float32)),
            0.0,
        ),
        "minveto/lambda_expansions": expansion_count.astype(jnp.float32),
        "mask/synthetic_mean": _tree_mean(synthetic_masks),
        "mask/instantaneous_final_mean": _tree_mean(instantaneous_gate),
        "mask/effective_ema_mean": _tree_mean(effective_gate),
        "update_norm/raw": tree_l2_norm(updates),
        "update_norm/synthetic": tree_l2_norm(proposal),
        "update_norm/instantaneous_minveto": tree_l2_norm(
            instantaneous_update
        ),
        "update_norm/final_ema": tree_l2_norm(final_updates),
        "update_norm/kept_fraction": (
            tree_l2_norm(final_updates)
            / jnp.maximum(
                tree_l2_norm(updates),
                jnp.asarray(weight_eps, dtype=jnp.float32),
            )
        ),
        "cnl/synth_teacher_cvar_score_mean": _tree_mean(synthetic_scores),
        "cnl/synth_teacher_cvar_tail_count": jnp.asarray(
            synthetic_tail_count,
            dtype=jnp.float32,
        ),
    }
    return final_updates, mask_ema, mask_ema_count, stats


def mask_optax_updates_behavior_multiveto_ema(
    updates: PyTree,
    current_grads: PyTree,
    behavior_group_grads: list[PyTree] | tuple[PyTree, ...],
    mask_ema: PyTree,
    mask_ema_count: jax.Array,
    *,
    synthetic_group_count: int,
    beta: float = 0.9,
    harm_reduction_fraction: float = 0.75,
    energy_cap: float = 0.10,
    weight_eps: float = 1e-12,
    dual_iterations: int = 24,
    dual_step_scale: float = 1.0,
    cap_bisection_iterations: int = 12,
    constraint_tolerance: float = 1e-6,
    normmatched_control: bool = False,
) -> tuple[PyTree, PyTree, jax.Array, dict[str, jax.Array]]:
    """Minimally gate an update under multiple behavior-level constraints.

    Each reference group contributes one first-order constraint on the complete
    update. This avoids turning a small reference set into independent binary
    decisions for every parameter coordinate. The low-dimensional dual is
    solved with projected gradient ascent; the primal gate remains elementwise
    so safe capacity inside a tensor is not discarded wholesale.
    """
    groups = tuple(behavior_group_grads)
    group_count = len(groups)
    if group_count <= 0:
        raise ValueError("behavior multiveto requires at least one group")
    if not 0 <= synthetic_group_count <= group_count:
        raise ValueError(
            "synthetic group count must lie within the behavior groups"
        )
    _validate_mask_ema_beta(beta)
    if not 0 <= harm_reduction_fraction <= 1:
        raise ValueError(
            "behavior multiveto harm reduction fraction must be in [0, 1]"
        )
    if not 0 <= energy_cap <= 1:
        raise ValueError("behavior multiveto energy cap must be in [0, 1]")
    if weight_eps <= 0:
        raise ValueError("behavior multiveto weight epsilon must be positive")
    if dual_iterations <= 0:
        raise ValueError("behavior multiveto dual iterations must be positive")
    if dual_step_scale <= 0:
        raise ValueError("behavior multiveto dual step scale must be positive")
    if cap_bisection_iterations <= 0:
        raise ValueError(
            "behavior multiveto cap bisection iterations must be positive"
        )
    if constraint_tolerance < 0:
        raise ValueError(
            "behavior multiveto constraint tolerance must be nonnegative"
        )

    contributions = tuple(
        first_order_contribution(group_grad, updates)
        for group_grad in groups
    )
    current_utility = jax.tree_util.tree_map(
        lambda grad, delta: jax.nn.relu(
            -jnp.asarray(grad, dtype=jnp.float32)
            * jnp.asarray(delta, dtype=jnp.float32)
        ),
        current_grads,
        updates,
    )
    utility_weights = jax.tree_util.tree_map(
        lambda utility: (
            jnp.asarray(utility, dtype=jnp.float32)
            + jnp.asarray(weight_eps, dtype=jnp.float32)
        ),
        current_utility,
    )
    damage_before = jnp.stack(
        [tree_sum(contribution) for contribution in contributions]
    )
    reduction = jnp.asarray(
        harm_reduction_fraction,
        dtype=jnp.float32,
    )
    damage_targets = jnp.where(
        damage_before > 0,
        (1.0 - reduction) * damage_before,
        jnp.zeros_like(damage_before),
    )
    damage_scales = jnp.maximum(
        jnp.abs(damage_before),
        jnp.asarray(weight_eps, dtype=jnp.float32),
    )
    damage_tolerances = (
        jnp.asarray(constraint_tolerance, dtype=jnp.float32)
        * damage_scales
    )

    def gate_for_lambdas(lambdas: jax.Array) -> PyTree:
        pressure = jax.tree_util.tree_map(
            lambda *values: sum(
                (
                    lambdas[index]
                    * jnp.asarray(value, dtype=jnp.float32)
                    for index, value in enumerate(values)
                ),
                jnp.asarray(0.0, dtype=jnp.float32),
            ),
            *contributions,
        )
        return jax.tree_util.tree_map(
            lambda dual_pressure, weight: (
                1.0
                - jnp.clip(
                    jnp.asarray(dual_pressure, dtype=jnp.float32)
                    / jnp.asarray(weight, dtype=jnp.float32),
                    0.0,
                    1.0,
                )
            ),
            pressure,
            utility_weights,
        )

    def damages_for_lambdas(lambdas: jax.Array) -> jax.Array:
        gate = gate_for_lambdas(lambdas)
        return jnp.stack(
            [
                tree_dot(contribution, gate)
                for contribution in contributions
            ]
        )

    gram_rows = []
    for left in contributions:
        gram_rows.append(
            jnp.stack(
                [
                    tree_sum(
                        jax.tree_util.tree_map(
                            lambda left_value, right_value, weight: (
                                jnp.asarray(
                                    left_value,
                                    dtype=jnp.float32,
                                )
                                * jnp.asarray(
                                    right_value,
                                    dtype=jnp.float32,
                                )
                                / jnp.asarray(weight, dtype=jnp.float32)
                            ),
                            left,
                            right,
                            utility_weights,
                        )
                    )
                    for right in contributions
                ]
            )
        )
    dual_gram = jnp.stack(gram_rows)
    dual_lipschitz = jnp.maximum(
        jnp.max(jnp.sum(jnp.abs(dual_gram), axis=1)),
        jnp.asarray(weight_eps, dtype=jnp.float32),
    )
    dual_step = (
        jnp.asarray(dual_step_scale, dtype=jnp.float32)
        / dual_lipschitz
    )

    def dual_step_body(
        _: int,
        lambdas: jax.Array,
    ) -> jax.Array:
        residual = damages_for_lambdas(lambdas) - damage_targets
        return jax.nn.relu(lambdas + dual_step * residual)

    lambda_unscaled = jax.lax.fori_loop(
        0,
        dual_iterations,
        dual_step_body,
        jnp.zeros((group_count,), dtype=jnp.float32),
    )
    unscaled_gate = gate_for_lambdas(lambda_unscaled)
    damage_unscaled = damages_for_lambdas(lambda_unscaled)
    raw_energy = tree_l2_sq(updates)

    def removed_energy_fraction_for_scale(scale: jax.Array) -> jax.Array:
        gate = gate_for_lambdas(lambda_unscaled * scale)
        removed = jax.tree_util.tree_map(
            lambda update, keep: (
                (1.0 - jnp.asarray(keep, dtype=jnp.float32))
                * jnp.asarray(update, dtype=jnp.float32)
            ),
            updates,
            gate,
        )
        return tree_l2_sq(removed) / jnp.maximum(
            raw_energy,
            jnp.asarray(weight_eps, dtype=jnp.float32),
        )

    cap_binding = (
        removed_energy_fraction_for_scale(
            jnp.asarray(1.0, dtype=jnp.float32)
        )
        > jnp.asarray(
            energy_cap + constraint_tolerance,
            dtype=jnp.float32,
        )
    )

    def cap_bisection_body(
        _: int,
        bounds: tuple[jax.Array, jax.Array],
    ) -> tuple[jax.Array, jax.Array]:
        low, high = bounds
        middle = 0.5 * (low + high)
        under_cap = (
            removed_energy_fraction_for_scale(middle)
            <= jnp.asarray(energy_cap, dtype=jnp.float32)
        )
        return (
            jnp.where(under_cap, middle, low),
            jnp.where(under_cap, high, middle),
        )

    def solve_cap(_: None) -> jax.Array:
        low, _ = jax.lax.fori_loop(
            0,
            cap_bisection_iterations,
            cap_bisection_body,
            (
                jnp.asarray(0.0, dtype=jnp.float32),
                jnp.asarray(1.0, dtype=jnp.float32),
            ),
        )
        return low

    lambda_scale = jax.lax.cond(
        cap_binding,
        solve_cap,
        lambda _: jnp.asarray(1.0, dtype=jnp.float32),
        operand=None,
    )
    lambdas = lambda_unscaled * lambda_scale
    instantaneous_gate = gate_for_lambdas(lambdas)
    instantaneous_updates = jax.tree_util.tree_map(
        lambda update, gate: (
            update * jnp.asarray(gate, dtype=update.dtype)
        ),
        updates,
        instantaneous_gate,
    )

    beta_arr = jnp.asarray(beta, dtype=jnp.float32)
    mask_ema = jax.tree_util.tree_map(
        lambda ema, gate: (
            beta_arr * jnp.asarray(ema, dtype=jnp.float32)
            + (1.0 - beta_arr) * jnp.asarray(gate, dtype=jnp.float32)
        ),
        mask_ema,
        instantaneous_gate,
    )
    mask_ema_count = mask_ema_count + jnp.asarray(
        1,
        dtype=mask_ema_count.dtype,
    )
    correction = 1.0 - beta_arr ** mask_ema_count.astype(jnp.float32)
    effective_gate = jax.tree_util.tree_map(
        lambda ema: jnp.clip(ema / correction, 0.0, 1.0),
        mask_ema,
    )
    protected_updates = jax.tree_util.tree_map(
        lambda update, gate: (
            update * jnp.asarray(gate, dtype=update.dtype)
        ),
        updates,
        effective_gate,
    )
    protected_norm = tree_l2_norm(protected_updates)
    raw_norm = tree_l2_norm(updates)
    normmatched_scale = protected_norm / jnp.maximum(
        raw_norm,
        jnp.asarray(weight_eps, dtype=jnp.float32),
    )
    final_updates = (
        jax.tree_util.tree_map(
            lambda update: update
            * jnp.asarray(normmatched_scale, dtype=update.dtype),
            updates,
        )
        if normmatched_control
        else protected_updates
    )

    damage_instant_after = jnp.stack(
        [
            tree_dot(group_grad, instantaneous_updates)
            for group_grad in groups
        ]
    )
    damage_ema_after = jnp.stack(
        [tree_dot(group_grad, protected_updates) for group_grad in groups]
    )
    violation_before = jax.nn.relu(
        damage_before - damage_targets - damage_tolerances
    )
    violation_unscaled = jax.nn.relu(
        damage_unscaled - damage_targets - damage_tolerances
    )
    violation_after = jax.nn.relu(
        damage_instant_after - damage_targets - damage_tolerances
    )
    violation_ema = jax.nn.relu(
        damage_ema_after - damage_targets - damage_tolerances
    )
    utility_total = tree_sum(current_utility)
    utility_kept = tree_sum(
        jax.tree_util.tree_map(
            lambda utility, gate: (
                jnp.asarray(utility, dtype=jnp.float32)
                * jnp.asarray(gate, dtype=jnp.float32)
            ),
            current_utility,
            instantaneous_gate,
        )
    )
    coordinate_tolerance = jnp.asarray(1e-6, dtype=jnp.float32)
    gate_mean = _tree_mean(instantaneous_gate)
    gate_sq_mean = _tree_mean(
        jax.tree_util.tree_map(
            lambda gate: jnp.square(
                jnp.asarray(gate, dtype=jnp.float32)
            ),
            instantaneous_gate,
        )
    )
    gate_min = jnp.asarray(jnp.inf, dtype=jnp.float32)
    gate_max = jnp.asarray(-jnp.inf, dtype=jnp.float32)
    gate_leaves = jax.tree_util.tree_leaves(instantaneous_gate)
    for gate in gate_leaves:
        gate_value = jnp.asarray(gate, dtype=jnp.float32)
        gate_min = jnp.minimum(gate_min, jnp.min(gate_value))
        gate_max = jnp.maximum(gate_max, jnp.max(gate_value))
    if not gate_leaves:
        gate_min = jnp.asarray(0.0, dtype=jnp.float32)
        gate_max = jnp.asarray(0.0, dtype=jnp.float32)

    def source_mean(values: jax.Array, start: int, stop: int) -> jax.Array:
        return (
            jnp.mean(values[start:stop])
            if stop > start
            else jnp.asarray(0.0, dtype=jnp.float32)
        )

    stats = {
        "behavior_veto/enabled": jnp.asarray(1.0, dtype=jnp.float32),
        "behavior_veto/constraint_count": jnp.asarray(
            group_count,
            dtype=jnp.float32,
        ),
        "behavior_veto/synthetic_constraint_count": jnp.asarray(
            synthetic_group_count,
            dtype=jnp.float32,
        ),
        "behavior_veto/real_constraint_count": jnp.asarray(
            group_count - synthetic_group_count,
            dtype=jnp.float32,
        ),
        "behavior_veto/harm_reduction_fraction": reduction,
        "behavior_veto/energy_cap": jnp.asarray(
            energy_cap,
            dtype=jnp.float32,
        ),
        "behavior_veto/dual_iterations": jnp.asarray(
            dual_iterations,
            dtype=jnp.float32,
        ),
        "behavior_veto/dual_step": dual_step,
        "behavior_veto/dual_lipschitz": dual_lipschitz,
        "behavior_veto/lambda_l1": jnp.sum(lambdas),
        "behavior_veto/lambda_max": jnp.max(lambdas),
        "behavior_veto/lambda_active_count": jnp.sum(
            (lambdas > coordinate_tolerance).astype(jnp.float32)
        ),
        "behavior_veto/lambda_scale_for_energy_cap": lambda_scale,
        "behavior_veto/cap_binding": cap_binding.astype(jnp.float32),
        "behavior_veto/target_feasible_before_cap": (
            jnp.max(violation_unscaled) <= 0
        ).astype(jnp.float32),
        "behavior_veto/target_feasible_under_cap": (
            jnp.max(violation_after) <= 0
        ).astype(jnp.float32),
        "behavior_veto/predicted_damage_before_mean": jnp.mean(
            damage_before
        ),
        "behavior_veto/predicted_damage_before_max": jnp.max(
            damage_before
        ),
        "behavior_veto/predicted_damage_target_mean": jnp.mean(
            damage_targets
        ),
        "behavior_veto/predicted_damage_instant_after_mean": jnp.mean(
            damage_instant_after
        ),
        "behavior_veto/predicted_damage_instant_after_max": jnp.max(
            damage_instant_after
        ),
        "behavior_veto/predicted_damage_ema_after_mean": jnp.mean(
            damage_ema_after
        ),
        "behavior_veto/predicted_damage_ema_after_max": jnp.max(
            damage_ema_after
        ),
        "behavior_veto/max_violation_before": jnp.max(violation_before),
        "behavior_veto/max_violation_instant_after": jnp.max(
            violation_after
        ),
        "behavior_veto/max_violation_ema_after": jnp.max(violation_ema),
        "behavior_veto/synthetic_damage_before_mean": source_mean(
            damage_before,
            0,
            synthetic_group_count,
        ),
        "behavior_veto/synthetic_damage_after_mean": source_mean(
            damage_instant_after,
            0,
            synthetic_group_count,
        ),
        "behavior_veto/real_damage_before_mean": source_mean(
            damage_before,
            synthetic_group_count,
            group_count,
        ),
        "behavior_veto/real_damage_after_mean": source_mean(
            damage_instant_after,
            synthetic_group_count,
            group_count,
        ),
        "behavior_veto/update_energy_removed_fraction": (
            removed_energy_fraction_for_scale(lambda_scale)
        ),
        "behavior_veto/gate_mean": gate_mean,
        "behavior_veto/gate_std": jnp.sqrt(
            jnp.maximum(gate_sq_mean - gate_mean**2, 0.0)
        ),
        "behavior_veto/gate_min": gate_min,
        "behavior_veto/gate_max": gate_max,
        "behavior_veto/fraction_coordinates_unchanged": _tree_mean(
            jax.tree_util.tree_map(
                lambda gate: (
                    jnp.asarray(gate, dtype=jnp.float32)
                    >= 1.0 - coordinate_tolerance
                ).astype(jnp.float32),
                instantaneous_gate,
            )
        ),
        "behavior_veto/fraction_coordinates_softened": _tree_mean(
            jax.tree_util.tree_map(
                lambda gate: (
                    (
                        jnp.asarray(gate, dtype=jnp.float32)
                        > coordinate_tolerance
                    )
                    & (
                        jnp.asarray(gate, dtype=jnp.float32)
                        < 1.0 - coordinate_tolerance
                    )
                ).astype(jnp.float32),
                instantaneous_gate,
            )
        ),
        "behavior_veto/fraction_coordinates_zeroed": _tree_mean(
            jax.tree_util.tree_map(
                lambda gate: (
                    jnp.asarray(gate, dtype=jnp.float32)
                    <= coordinate_tolerance
                ).astype(jnp.float32),
                instantaneous_gate,
            )
        ),
        "behavior_veto/current_utility_kept_fraction": (
            utility_kept
            / jnp.maximum(
                utility_total,
                jnp.asarray(weight_eps, dtype=jnp.float32),
            )
        ),
        "behavior_veto/normmatched_control": jnp.asarray(
            float(normmatched_control),
            dtype=jnp.float32,
        ),
        "behavior_veto/normmatched_scale": normmatched_scale,
        "mask/instantaneous_final_mean": gate_mean,
        "mask/effective_ema_mean": _tree_mean(effective_gate),
        "update_norm/raw": raw_norm,
        "update_norm/instantaneous_behavior_veto": tree_l2_norm(
            instantaneous_updates
        ),
        "update_norm/protected_ema": protected_norm,
        "update_norm/final_ema": tree_l2_norm(final_updates),
        "update_norm/kept_fraction": (
            tree_l2_norm(final_updates)
            / jnp.maximum(
                raw_norm,
                jnp.asarray(weight_eps, dtype=jnp.float32),
            )
        ),
    }
    for index in range(group_count):
        prefix = f"behavior_veto/constraint_{index}"
        stats[f"{prefix}_damage_before"] = damage_before[index]
        stats[f"{prefix}_target"] = damage_targets[index]
        stats[f"{prefix}_damage_after"] = damage_instant_after[index]
        stats[f"{prefix}_damage_ema_after"] = damage_ema_after[index]
        stats[f"{prefix}_violation_after"] = violation_after[index]
        stats[f"{prefix}_lambda"] = lambdas[index]
    return final_updates, mask_ema, mask_ema_count, stats


def _validate_mask_ema_beta(beta: float) -> None:
    if not 0 <= beta < 1:
        raise ValueError("cnl_mask_ema_beta must be in [0, 1)")


def mask_optax_updates_ema(
    updates: PyTree,
    reference_grads: PyTree | None,
    mask_ema: PyTree,
    mask_ema_count: jax.Array,
    *,
    beta: float = 0.9,
) -> tuple[PyTree, PyTree, jax.Array, dict[str, jax.Array]]:
    """Apply a bias-corrected EMA of the hard CNL update mask.

    The instantaneous mask is computed on the same Optax update direction as
    ``mask_optax_updates``. EMA state is zero-initialized and bias-corrected so
    the first effective mask equals the hard CNL mask exactly.
    """
    _validate_mask_ema_beta(beta)
    if reference_grads is None:
        stats = {
            "cnl/instant_mask_mean": jnp.asarray(0.0, dtype=jnp.float32),
            "cnl/ema_mask_mean": jnp.asarray(0.0, dtype=jnp.float32),
            "cnl/ema_mask_frac_gt_0_5": jnp.asarray(0.0, dtype=jnp.float32),
        }
        return updates, mask_ema, mask_ema_count, stats

    hard_mask = jax.tree_util.tree_map(
        lambda u, r: (((-u) * r) >= 0).astype(jnp.float32),
        updates,
        reference_grads,
    )
    beta_arr = jnp.asarray(beta, dtype=jnp.float32)
    mask_ema = jax.tree_util.tree_map(
        lambda e, m: beta_arr * e + (1.0 - beta_arr) * m,
        mask_ema,
        hard_mask,
    )
    mask_ema_count = mask_ema_count + jnp.asarray(1, dtype=mask_ema_count.dtype)
    correction = 1.0 - beta_arr ** mask_ema_count.astype(jnp.float32)
    effective_mask = jax.tree_util.tree_map(lambda e: jnp.clip(e / correction, 0.0, 1.0), mask_ema)
    updates = jax.tree_util.tree_map(lambda u, m: u * m.astype(u.dtype), updates, effective_mask)
    stats = {
        "cnl/instant_mask_mean": _tree_mean(hard_mask),
        "cnl/ema_mask_mean": _tree_mean(effective_mask),
        "cnl/ema_mask_frac_gt_0_5": _tree_mean(
            jax.tree_util.tree_map(lambda m: (m > 0.5).astype(jnp.float32), effective_mask)
        ),
    }
    return updates, mask_ema, mask_ema_count, stats


def add_trees(a: PyTree, b: PyTree) -> PyTree:
    return jax.tree_util.tree_map(lambda x, y: x + y, a, b)


def divide_tree(tree: PyTree, denom: float) -> PyTree:
    return jax.tree_util.tree_map(lambda x: x / denom, tree)


def prepare_blend_reference_sources(
    synthetic_reference_grads: PyTree,
    anchor_reference_grads: PyTree,
    *,
    norm_align_anchor: bool = False,
    eps: float = 1e-12,
) -> tuple[PyTree, PyTree, dict[str, jax.Array]]:
    """Return blend-ready source gradients and scale diagnostics."""
    if eps <= 0:
        raise ValueError("blend source eps must be positive")

    synthetic_norm = tree_l2_norm(synthetic_reference_grads)
    anchor_norm = tree_l2_norm(anchor_reference_grads)
    if norm_align_anchor:
        anchor_scale = jnp.where(
            anchor_norm > jnp.asarray(eps, dtype=jnp.float32),
            synthetic_norm
            / jnp.maximum(anchor_norm, jnp.asarray(eps, dtype=jnp.float32)),
            jnp.asarray(1.0, dtype=jnp.float32),
        )
    else:
        anchor_scale = jnp.asarray(1.0, dtype=jnp.float32)
    aligned_anchor_grads = scale_tree(anchor_reference_grads, anchor_scale)
    aligned_anchor_norm = tree_l2_norm(aligned_anchor_grads)
    source_cosine = tree_dot(
        synthetic_reference_grads,
        aligned_anchor_grads,
    ) / jnp.maximum(
        synthetic_norm * aligned_anchor_norm,
        jnp.asarray(eps, dtype=jnp.float32),
    )
    stats = {
        "anchor_shrinkage/source_cosine": source_cosine,
        "anchor_blend/norm_aligned": jnp.asarray(
            float(norm_align_anchor), dtype=jnp.float32
        ),
        "anchor_blend/anchor_norm_scale": anchor_scale,
        "anchor_blend/synthetic_gradient_norm": synthetic_norm,
        "anchor_blend/anchor_gradient_norm_before_alignment": anchor_norm,
        "anchor_blend/anchor_gradient_norm_after_alignment": (
            aligned_anchor_norm
        ),
    }
    return synthetic_reference_grads, aligned_anchor_grads, stats


def build_fixed_blend_reference(
    synthetic_reference_grads: PyTree,
    anchor_reference_grads: PyTree,
    *,
    real_weight: float,
    synthetic_group_count: int,
    anchor_group_count: int,
    norm_align_anchor: bool = False,
    eps: float = 1e-12,
) -> tuple[PyTree, dict[str, jax.Array]]:
    """Fuse source-weighted synthetic and real-anchor reference gradients.

    The caller must pass the weighted mean gradient for each complete source.
    Source groups are retained only for diagnostics; averaging their means would
    overweight a smaller final group. CVaR is deliberately not applied after
    the blend.
    """
    if not 0.0 <= real_weight <= 1.0:
        raise ValueError("fixed blend real_weight must be in [0, 1]")
    if synthetic_group_count <= 0 or anchor_group_count <= 0:
        raise ValueError("fixed blending requires positive source group counts")
    if eps <= 0:
        raise ValueError("fixed blend eps must be positive")

    synthetic_mean, aligned_anchor_mean, source_stats = (
        prepare_blend_reference_sources(
            synthetic_reference_grads,
            anchor_reference_grads,
            norm_align_anchor=norm_align_anchor,
            eps=eps,
        )
    )

    synthetic_weight = 1.0 - float(real_weight)
    fused_reference = jax.tree_util.tree_map(
        lambda synthetic, anchor: (
            synthetic * jnp.asarray(synthetic_weight, dtype=synthetic.dtype)
            + anchor * jnp.asarray(real_weight, dtype=anchor.dtype)
        ),
        synthetic_mean,
        aligned_anchor_mean,
    )

    synthetic_norm = source_stats["anchor_blend/synthetic_gradient_norm"]
    aligned_anchor_norm = source_stats[
        "anchor_blend/anchor_gradient_norm_after_alignment"
    ]
    fused_norm = tree_l2_norm(fused_reference)
    synthetic_component_norm = (
        jnp.asarray(synthetic_weight, dtype=jnp.float32) * synthetic_norm
    )
    real_component_norm = (
        jnp.asarray(real_weight, dtype=jnp.float32) * aligned_anchor_norm
    )
    real_component_fraction = real_component_norm / jnp.maximum(
        synthetic_component_norm + real_component_norm,
        jnp.asarray(eps, dtype=jnp.float32),
    )
    weight = jnp.asarray(real_weight, dtype=jnp.float32)
    stats = {
        "anchor_shrinkage/real_weight_mean": weight,
        "anchor_shrinkage/real_weight_tensor_mean": weight,
        "anchor_shrinkage/real_weight_min": weight,
        "anchor_shrinkage/real_weight_max": weight,
        "anchor_shrinkage/synthetic_groups": jnp.asarray(
            synthetic_group_count, dtype=jnp.float32
        ),
        "anchor_shrinkage/anchor_groups": jnp.asarray(
            anchor_group_count, dtype=jnp.float32
        ),
        "anchor_shrinkage/fixed_blend": jnp.asarray(1.0, dtype=jnp.float32),
        "anchor_blend/fused_gradient_norm": fused_norm,
        "anchor_blend/effective_real_component_norm_fraction": (
            real_component_fraction
        ),
        **source_stats,
    }
    return fused_reference, stats


def fixed_blend_tensor_sufficient_statistics(
    updates: PyTree,
    synthetic_reference_grads: PyTree,
    real_reference_grads: PyTree,
    fused_reference_grads: PyTree,
    *,
    real_weight: float,
) -> PyTree:
    """Reduce fixed-blend behavior to one statistic vector per tensor.

    The returned tree has the same structure as ``updates``. Each leaf is a
    float32 vector ordered by ``FIXED_BLEND_TENSOR_STAT_FIELDS``. Keeping only
    reductions makes this suitable for TPU audits without transferring model-
    sized gradients or masks to the host.
    """
    if not 0.0 <= real_weight <= 1.0:
        raise ValueError("fixed blend real_weight must be in [0, 1]")

    synthetic_weight = 1.0 - float(real_weight)

    def summarize(
        update: jax.Array,
        synthetic: jax.Array,
        real: jax.Array,
        fused: jax.Array,
    ) -> jax.Array:
        update = jnp.asarray(update, dtype=jnp.float32)
        synthetic = jnp.asarray(synthetic, dtype=jnp.float32)
        real = jnp.asarray(real, dtype=jnp.float32)
        fused = jnp.asarray(fused, dtype=jnp.float32)

        descent = -update
        synthetic_allow = descent * synthetic >= 0
        real_allow = descent * real >= 0
        blended_allow = descent * fused >= 0
        source_active = (synthetic != 0) | (real != 0)
        source_conflict = synthetic * real < 0
        source_agreement = source_active & ~source_conflict
        synthetic_flip = synthetic_allow != blended_allow
        real_flip = real_allow != blended_allow

        update_sq = jnp.square(update)
        weighted_synthetic = synthetic * jnp.asarray(
            synthetic_weight, dtype=jnp.float32
        )
        weighted_real = real * jnp.asarray(
            real_weight, dtype=jnp.float32
        )
        synthetic_damage = synthetic * update
        real_damage = real * update
        blended_update = update * blended_allow.astype(jnp.float32)

        return jnp.stack(
            (
                jnp.asarray(float(update.size), dtype=jnp.float32),
                jnp.sum(update_sq),
                jnp.sum(update_sq * synthetic_allow),
                jnp.sum(update_sq * real_allow),
                jnp.sum(update_sq * blended_allow),
                jnp.sum(synthetic_allow, dtype=jnp.float32),
                jnp.sum(real_allow, dtype=jnp.float32),
                jnp.sum(blended_allow, dtype=jnp.float32),
                jnp.sum(source_active, dtype=jnp.float32),
                jnp.sum(source_agreement, dtype=jnp.float32),
                jnp.sum(source_conflict, dtype=jnp.float32),
                jnp.sum(synthetic_flip, dtype=jnp.float32),
                jnp.sum(real_flip, dtype=jnp.float32),
                jnp.sum(update_sq * synthetic_flip),
                jnp.sum(update_sq * real_flip),
                jnp.sum(jnp.abs(weighted_synthetic)),
                jnp.sum(jnp.abs(weighted_real)),
                jnp.sum(jnp.abs(fused)),
                jnp.sum(jnp.abs(weighted_synthetic * update)),
                jnp.sum(jnp.abs(weighted_real * update)),
                jnp.sum(jnp.abs(fused * update)),
                jnp.sum(synthetic * real),
                jnp.sum(jnp.square(synthetic)),
                jnp.sum(jnp.square(real)),
                jnp.sum(jnp.square(fused)),
                jnp.sum(jnp.maximum(synthetic_damage, 0.0)),
                jnp.sum(
                    jnp.maximum(synthetic * blended_update, 0.0)
                ),
                jnp.sum(jnp.maximum(real_damage, 0.0)),
                jnp.sum(jnp.maximum(real * blended_update, 0.0)),
            ),
            dtype=jnp.float32,
        )

    return jax.tree_util.tree_map(
        summarize,
        updates,
        synthetic_reference_grads,
        real_reference_grads,
        fused_reference_grads,
    )


def build_reference_gradient_variance_of_mean(
    reference_mean: PyTree,
    reference_group_grads: list[PyTree] | tuple[PyTree, ...],
) -> PyTree:
    """Estimate coordinatewise variance of a source mean from group means.

    With ``K`` balanced groups, the unbiased sample variance of the group
    gradients divided by ``K`` estimates the variance of their mean. The
    returned float32 tree can later be weighted by the squared optimizer update
    to measure uncertainty in the same interaction space used by CPL.
    """
    group_count = len(reference_group_grads)
    if group_count < 2:
        raise ValueError(
            "bias-variance shrinkage requires at least two groups per source"
        )
    denominator = jnp.asarray(
        float(group_count * (group_count - 1)),
        dtype=jnp.float32,
    )

    def variance_of_mean(
        mean: jax.Array,
        *group_values: jax.Array,
    ) -> jax.Array:
        mean_f32 = jnp.asarray(mean, dtype=jnp.float32)
        squared_deviation = sum(
            (
                jnp.square(
                    jnp.asarray(group, dtype=jnp.float32) - mean_f32
                )
                for group in group_values
            ),
            jnp.zeros_like(mean_f32),
        )
        return squared_deviation / denominator

    return jax.tree_util.tree_map(
        variance_of_mean,
        reference_mean,
        *reference_group_grads,
    )


def build_paired_reference_gradient_covariance_of_mean(
    synthetic_reference_mean: PyTree,
    anchor_reference_mean: PyTree,
    synthetic_group_grads: list[PyTree] | tuple[PyTree, ...],
    anchor_group_grads: list[PyTree] | tuple[PyTree, ...],
) -> PyTree:
    """Estimate covariance between two paired reference-mean estimators.

    Group ``k`` from both sources must describe the same real-anchor lineage
    bucket. With ``K`` paired groups, sample covariance divided by ``K`` is the
    covariance of the two source means. Pairing matters here because synthetic
    anchor-ICL examples are generated from the real anchors and the two
    estimators are therefore not independent.
    """
    group_count = len(synthetic_group_grads)
    if group_count < 2:
        raise ValueError(
            "paired reference covariance requires at least two groups"
        )
    if len(anchor_group_grads) != group_count:
        raise ValueError(
            "paired reference covariance requires equal source group counts"
        )
    denominator = jnp.asarray(
        float(group_count * (group_count - 1)),
        dtype=jnp.float32,
    )

    def covariance_of_mean(
        synthetic_mean: jax.Array,
        anchor_mean: jax.Array,
        *group_values: jax.Array,
    ) -> jax.Array:
        synthetic_values = group_values[:group_count]
        anchor_values = group_values[group_count:]
        synthetic_mean_f32 = jnp.asarray(
            synthetic_mean,
            dtype=jnp.float32,
        )
        anchor_mean_f32 = jnp.asarray(anchor_mean, dtype=jnp.float32)
        cross_deviation = sum(
            (
                (
                    jnp.asarray(synthetic_group, dtype=jnp.float32)
                    - synthetic_mean_f32
                )
                * (
                    jnp.asarray(anchor_group, dtype=jnp.float32)
                    - anchor_mean_f32
                )
                for synthetic_group, anchor_group in zip(
                    synthetic_values,
                    anchor_values,
                )
            ),
            jnp.zeros_like(synthetic_mean_f32),
        )
        return cross_deviation / denominator

    return jax.tree_util.tree_map(
        covariance_of_mean,
        synthetic_reference_mean,
        anchor_reference_mean,
        *synthetic_group_grads,
        *anchor_group_grads,
    )


def tensor_pairwise_reference_reliability(
    reference_group_grads: list[PyTree] | tuple[PyTree, ...],
    *,
    eps: float,
) -> PyTree:
    """Measure positive pairwise cosine agreement independently per tensor."""
    if len(reference_group_grads) < 2:
        raise ValueError("tensor reliability requires at least two groups")
    if eps <= 0:
        raise ValueError("tensor reliability eps must be positive")
    pairs = tuple(
        (left, right)
        for left in range(len(reference_group_grads))
        for right in range(left + 1, len(reference_group_grads))
    )

    def reliability(*group_values: jax.Array) -> jax.Array:
        total = jnp.asarray(0.0, dtype=jnp.float32)
        for left, right in pairs:
            lhs = jnp.asarray(group_values[left], dtype=jnp.float32)
            rhs = jnp.asarray(group_values[right], dtype=jnp.float32)
            lhs_norm = jnp.sqrt(jnp.sum(jnp.square(lhs)))
            rhs_norm = jnp.sqrt(jnp.sum(jnp.square(rhs)))
            denominator = lhs_norm * rhs_norm
            cosine = jnp.sum(lhs * rhs) / jnp.maximum(
                denominator,
                jnp.asarray(eps, dtype=jnp.float32),
            )
            total = total + jnp.where(
                denominator > jnp.asarray(eps, dtype=jnp.float32),
                jnp.clip(cosine, 0.0, 1.0),
                jnp.asarray(0.0, dtype=jnp.float32),
            )
        return total / jnp.asarray(float(len(pairs)), dtype=jnp.float32)

    return jax.tree_util.tree_map(reliability, *reference_group_grads)


def _tensor_scalar_summary(
    scalar_tree: PyTree,
    template_tree: PyTree,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return parameter-count-weighted mean plus tensor-level min and max."""
    scalar_leaves = jax.tree_util.tree_leaves(scalar_tree)
    template_leaves = jax.tree_util.tree_leaves(template_tree)
    if not scalar_leaves or len(scalar_leaves) != len(template_leaves):
        raise ValueError("scalar and template trees must be non-empty and match")
    weighted_total = sum(
        (
            jnp.asarray(value, dtype=jnp.float32)
            * jnp.asarray(float(template.size), dtype=jnp.float32)
            for value, template in zip(scalar_leaves, template_leaves)
        ),
        jnp.asarray(0.0, dtype=jnp.float32),
    )
    parameter_count = sum(float(template.size) for template in template_leaves)
    values = jnp.stack(
        [jnp.asarray(value, dtype=jnp.float32) for value in scalar_leaves]
    )
    return (
        weighted_total
        / jnp.asarray(max(parameter_count, 1.0), dtype=jnp.float32),
        jnp.min(values),
        jnp.max(values),
    )


def build_local_reliability_blend_reference(
    synthetic_reference_grads: PyTree,
    anchor_reference_grads: PyTree,
    synthetic_group_grads: list[PyTree] | tuple[PyTree, ...],
    anchor_group_grads: list[PyTree] | tuple[PyTree, ...],
    *,
    norm_align_anchor: bool = True,
    reliability_prior: float = 1.0,
    eps: float = 1e-12,
) -> tuple[PyTree, dict[str, jax.Array]]:
    """Blend sources per tensor using group agreement and a symmetric prior.

    Positive pairwise cosine agreement supplies one reliability value per
    parameter tensor and source. A symmetric prior shrinks uncertain tensor
    weights toward 0.5; with the default prior, weights lie in [0.25, 0.75].
    """
    synthetic_reliability = tensor_pairwise_reference_reliability(
        synthetic_group_grads,
        eps=eps,
    )
    anchor_reliability = tensor_pairwise_reference_reliability(
        anchor_group_grads,
        eps=eps,
    )
    return build_local_reliability_blend_from_scores(
        synthetic_reference_grads,
        anchor_reference_grads,
        synthetic_reliability,
        anchor_reliability,
        synthetic_group_count=len(synthetic_group_grads),
        anchor_group_count=len(anchor_group_grads),
        norm_align_anchor=norm_align_anchor,
        reliability_prior=reliability_prior,
        eps=eps,
    )


def build_local_reliability_blend_from_scores(
    synthetic_reference_grads: PyTree,
    anchor_reference_grads: PyTree,
    synthetic_reliability: PyTree,
    anchor_reliability: PyTree,
    *,
    synthetic_group_count: int,
    anchor_group_count: int,
    norm_align_anchor: bool = True,
    reliability_prior: float = 1.0,
    eps: float = 1e-12,
) -> tuple[PyTree, dict[str, jax.Array]]:
    """Build the local blend from precomputed per-tensor source scores."""
    if reliability_prior <= 0:
        raise ValueError("local reliability prior must be positive")
    if synthetic_group_count < 2 or anchor_group_count < 2:
        raise ValueError("local reliability blending requires two groups per source")
    synthetic_mean, aligned_anchor_mean, source_stats = (
        prepare_blend_reference_sources(
            synthetic_reference_grads,
            anchor_reference_grads,
            norm_align_anchor=norm_align_anchor,
            eps=eps,
        )
    )
    prior = jnp.asarray(reliability_prior, dtype=jnp.float32)
    real_weight_tree = jax.tree_util.tree_map(
        lambda synthetic, anchor: (
            0.5 * prior + jnp.asarray(anchor, dtype=jnp.float32)
        )
        / (
            prior
            + jnp.asarray(synthetic, dtype=jnp.float32)
            + jnp.asarray(anchor, dtype=jnp.float32)
        ),
        synthetic_reliability,
        anchor_reliability,
    )
    fused_reference = jax.tree_util.tree_map(
        lambda synthetic, anchor, real_weight: (
            synthetic
            * (jnp.asarray(1.0, dtype=synthetic.dtype) - real_weight).astype(
                synthetic.dtype
            )
            + anchor * real_weight.astype(anchor.dtype)
        ),
        synthetic_mean,
        aligned_anchor_mean,
        real_weight_tree,
    )
    real_weight_mean, real_weight_min, real_weight_max = _tensor_scalar_summary(
        real_weight_tree,
        synthetic_mean,
    )
    synthetic_reliability_mean, _, _ = _tensor_scalar_summary(
        synthetic_reliability,
        synthetic_mean,
    )
    anchor_reliability_mean, _, _ = _tensor_scalar_summary(
        anchor_reliability,
        synthetic_mean,
    )
    stats = {
        "anchor_shrinkage/real_weight_mean": real_weight_mean,
        "anchor_shrinkage/real_weight_tensor_mean": real_weight_mean,
        "anchor_shrinkage/real_weight_min": real_weight_min,
        "anchor_shrinkage/real_weight_max": real_weight_max,
        "anchor_shrinkage/synthetic_groups": jnp.asarray(
            synthetic_group_count, dtype=jnp.float32
        ),
        "anchor_shrinkage/anchor_groups": jnp.asarray(
            anchor_group_count, dtype=jnp.float32
        ),
        "anchor_shrinkage/fixed_blend": jnp.asarray(0.0, dtype=jnp.float32),
        "anchor_blend/fused_gradient_norm": tree_l2_norm(fused_reference),
        "anchor_blend/local_synthetic_reliability_mean": (
            synthetic_reliability_mean
        ),
        "anchor_blend/local_real_reliability_mean": anchor_reliability_mean,
        "anchor_blend/local_reliability_prior": prior,
        **source_stats,
    }
    return fused_reference, stats


def coordinate_leave_one_group_out_sign_stability(
    reference_mean: PyTree,
    reference_group_grads: list[PyTree] | tuple[PyTree, ...],
) -> PyTree:
    """Return where a source sign survives removal of every reference group.

    The grouped gradients are deterministic, approximately balanced group
    means. A coordinate is stable only when the complete-source mean is
    nonzero and every leave-one-group-out sum has the same sign. The output is
    boolean, so it is substantially smaller than retaining all group gradients
    during injection training.
    """
    if len(reference_group_grads) < 2:
        raise ValueError(
            "coordinate sign stability requires at least two source groups"
        )

    def stable(mean: jax.Array, *groups: jax.Array) -> jax.Array:
        mean_f32 = jnp.asarray(mean, dtype=jnp.float32)
        source_sign = jnp.sign(mean_f32)
        group_values = tuple(
            jnp.asarray(group, dtype=jnp.float32) for group in groups
        )
        group_sum = sum(group_values, jnp.zeros_like(mean_f32))
        is_stable = source_sign != 0
        for group in group_values:
            is_stable = is_stable & (
                jnp.sign(group_sum - group) == source_sign
            )
        return is_stable

    return jax.tree_util.tree_map(
        stable,
        reference_mean,
        *reference_group_grads,
    )


def build_anchor_certified_reference(
    synthetic_reference_grads: PyTree,
    anchor_reference_grads: PyTree,
    synthetic_stability: PyTree,
    anchor_stability: PyTree,
    *,
    synthetic_group_count: int,
    anchor_group_count: int,
    eps: float = 1e-12,
) -> tuple[PyTree, dict[str, jax.Array]]:
    """Elect a preservation sign using broad synthetic and certified real evidence.

    Matching source signs are used when at least one source is reproducible.
    On disagreement, the real-anchor sign is used only when it is reproducible.
    Every unresolved coordinate receives a zero reference, which means abstain
    rather than silently treating zero as permission to update.
    """
    if synthetic_group_count < 2 or anchor_group_count < 2:
        raise ValueError(
            "anchor-certified merging requires at least two groups per source"
        )
    if eps <= 0:
        raise ValueError("anchor-certified merge eps must be positive")

    synthetic_mean, anchor_mean, source_stats = prepare_blend_reference_sources(
        synthetic_reference_grads,
        anchor_reference_grads,
        norm_align_anchor=False,
        eps=eps,
    )

    def source_sign(value: jax.Array) -> jax.Array:
        return jnp.sign(jnp.asarray(value, dtype=jnp.float32)).astype(jnp.int8)

    synthetic_signs = jax.tree_util.tree_map(source_sign, synthetic_mean)
    anchor_signs = jax.tree_util.tree_map(source_sign, anchor_mean)
    source_agreement = jax.tree_util.tree_map(
        lambda synthetic, anchor: synthetic == anchor,
        synthetic_signs,
        anchor_signs,
    )
    any_stable = jax.tree_util.tree_map(
        lambda synthetic, anchor: synthetic | anchor,
        synthetic_stability,
        anchor_stability,
    )
    certified_agreement = jax.tree_util.tree_map(
        lambda agreement, stable: agreement & stable,
        source_agreement,
        any_stable,
    )
    real_certified_conflict = jax.tree_util.tree_map(
        lambda agreement, stable: (~agreement) & stable,
        source_agreement,
        anchor_stability,
    )
    elected_reference = jax.tree_util.tree_map(
        lambda synthetic, anchor, agreement, real_conflict: jnp.where(
            agreement,
            synthetic,
            jnp.where(real_conflict, anchor, jnp.zeros_like(synthetic)),
        ),
        synthetic_signs,
        anchor_signs,
        certified_agreement,
        real_certified_conflict,
    )
    elected = jax.tree_util.tree_map(
        lambda reference: reference != 0,
        elected_reference,
    )
    both_stable = jax.tree_util.tree_map(
        lambda synthetic, anchor: synthetic & anchor,
        synthetic_stability,
        anchor_stability,
    )

    stats = {
        "anchor_shrinkage/synthetic_groups": jnp.asarray(
            synthetic_group_count, dtype=jnp.float32
        ),
        "anchor_shrinkage/anchor_groups": jnp.asarray(
            anchor_group_count, dtype=jnp.float32
        ),
        "anchor_shrinkage/fixed_blend": jnp.asarray(0.0, dtype=jnp.float32),
        "anchor_certified/synthetic_stable_fraction": _tree_mean(
            synthetic_stability
        ),
        "anchor_certified/real_stable_fraction": _tree_mean(anchor_stability),
        "anchor_certified/both_stable_fraction": _tree_mean(both_stable),
        "anchor_certified/source_sign_agreement_fraction": _tree_mean(
            source_agreement
        ),
        "anchor_certified/certified_agreement_fraction": _tree_mean(
            certified_agreement
        ),
        "anchor_certified/source_sign_conflict_fraction": (
            jnp.asarray(1.0, dtype=jnp.float32) - _tree_mean(source_agreement)
        ),
        "anchor_certified/real_certified_conflict_fraction": _tree_mean(
            real_certified_conflict
        ),
        "anchor_certified/elected_fraction": _tree_mean(elected),
        "anchor_certified/abstained_fraction": (
            jnp.asarray(1.0, dtype=jnp.float32) - _tree_mean(elected)
        ),
        "anchor_certified/elected_reference_norm": tree_l2_norm(
            elected_reference
        ),
        **source_stats,
    }
    return elected_reference, stats


def mask_optax_updates_fixed_blend_ema(
    updates: PyTree,
    fused_reference_grads: PyTree,
    mask_ema: PyTree,
    mask_ema_count: jax.Array,
    *,
    beta: float = 0.9,
) -> tuple[PyTree, PyTree, jax.Array, dict[str, jax.Array]]:
    """Apply one EMA-smoothed CPL mask from a pre-built blended reference."""
    _validate_mask_ema_beta(beta)
    hard_masks = jax.tree_util.tree_map(
        lambda update, reference: (
            ((-update) * reference >= 0).astype(jnp.float32)
        ),
        updates,
        fused_reference_grads,
    )
    beta_arr = jnp.asarray(beta, dtype=jnp.float32)
    mask_ema = jax.tree_util.tree_map(
        lambda ema, hard: beta_arr * ema + (1.0 - beta_arr) * hard,
        mask_ema,
        hard_masks,
    )
    mask_ema_count = mask_ema_count + jnp.asarray(
        1, dtype=mask_ema_count.dtype
    )
    correction = 1.0 - beta_arr ** mask_ema_count.astype(jnp.float32)
    effective_masks = jax.tree_util.tree_map(
        lambda ema: jnp.clip(ema / correction, 0.0, 1.0),
        mask_ema,
    )
    masked_updates = jax.tree_util.tree_map(
        lambda update, mask: update * mask.astype(update.dtype),
        updates,
        effective_masks,
    )
    original_norm = tree_l2_norm(updates)
    masked_norm = tree_l2_norm(masked_updates)
    stats = {
        "anchor_shrinkage/hard_mask_mean": _tree_mean(hard_masks),
        "anchor_shrinkage/ema_mask_mean": _tree_mean(effective_masks),
        "anchor_shrinkage/update_norm_original": original_norm,
        "anchor_shrinkage/update_norm_masked": masked_norm,
        "anchor_shrinkage/update_norm_kept_fraction": (
            masked_norm
            / jnp.maximum(original_norm, jnp.asarray(1e-30, dtype=jnp.float32))
        ),
    }
    return masked_updates, mask_ema, mask_ema_count, stats


def _blend_gate_stats(
    prefix: str,
    updates: PyTree,
    instant_gates: PyTree,
    effective_gates: PyTree,
    masked_updates: PyTree,
) -> dict[str, jax.Array]:
    original_norm = tree_l2_norm(updates)
    masked_norm = tree_l2_norm(masked_updates)
    gate_leaves = jax.tree_util.tree_leaves(instant_gates)
    fractional_total = sum(
        (
            jnp.sum(
                ((gate > 0.0) & (gate < 1.0)).astype(jnp.float32)
            )
            for gate in gate_leaves
        ),
        jnp.asarray(0.0, dtype=jnp.float32),
    )
    fractional_mean = _divide_tree_total_by_count(
        fractional_total,
        sum(gate.size for gate in gate_leaves),
    )
    return {
        "anchor_shrinkage/hard_mask_mean": _tree_mean(instant_gates),
        "anchor_shrinkage/ema_mask_mean": _tree_mean(effective_gates),
        "anchor_shrinkage/update_norm_original": original_norm,
        "anchor_shrinkage/update_norm_masked": masked_norm,
        "anchor_shrinkage/update_norm_kept_fraction": (
            masked_norm
            / jnp.maximum(
                original_norm,
                jnp.asarray(1e-30, dtype=jnp.float32),
            )
        ),
        f"anchor_blend/{prefix}_instant_gate_mean": _tree_mean(
            instant_gates
        ),
        f"anchor_blend/{prefix}_fractional_gate_fraction": fractional_mean,
    }


def mask_optax_updates_source_vote_ema(
    updates: PyTree,
    synthetic_reference_grads: PyTree,
    anchor_reference_grads: PyTree,
    mask_ema: PyTree,
    mask_ema_count: jax.Array,
    *,
    real_weight: float = 0.5,
    beta: float = 0.9,
) -> tuple[PyTree, PyTree, jax.Array, dict[str, jax.Array]]:
    """Average the two source-level CPL decisions into one soft gate."""
    if not 0.0 <= real_weight <= 1.0:
        raise ValueError("source-vote real_weight must be in [0, 1]")
    synthetic_weight = jnp.asarray(1.0 - real_weight, dtype=jnp.float32)
    anchor_weight = jnp.asarray(real_weight, dtype=jnp.float32)
    gates = jax.tree_util.tree_map(
        lambda update, synthetic, anchor: (
            synthetic_weight
            * (
                (-jnp.asarray(update, dtype=jnp.float32))
                * jnp.asarray(synthetic, dtype=jnp.float32)
                >= 0.0
            ).astype(jnp.float32)
            + anchor_weight
            * (
                (-jnp.asarray(update, dtype=jnp.float32))
                * jnp.asarray(anchor, dtype=jnp.float32)
                >= 0.0
            ).astype(jnp.float32)
        ),
        updates,
        synthetic_reference_grads,
        anchor_reference_grads,
    )
    masked_updates, mask_ema, mask_ema_count, effective_gates = (
        _apply_soft_mask_ema(
            updates,
            gates,
            mask_ema,
            mask_ema_count,
            beta=beta,
        )
    )
    stats = _blend_gate_stats(
        "source_vote",
        updates,
        gates,
        effective_gates,
        masked_updates,
    )
    stats["anchor_blend/source_vote_real_weight"] = jnp.asarray(
        real_weight,
        dtype=jnp.float32,
    )
    return masked_updates, mask_ema, mask_ema_count, stats


def _normalized_preservation_score(
    update: jax.Array,
    reference: jax.Array,
    *,
    eps: float,
) -> jax.Array:
    score = (
        -jnp.asarray(update, dtype=jnp.float32)
        * jnp.asarray(reference, dtype=jnp.float32)
    )
    rms = jnp.sqrt(jnp.mean(jnp.square(score)))
    return score / jnp.maximum(rms, jnp.asarray(eps, dtype=jnp.float32))


def mask_optax_updates_soft_ties_ema(
    updates: PyTree,
    synthetic_reference_grads: PyTree,
    anchor_reference_grads: PyTree,
    mask_ema: PyTree,
    mask_ema_count: jax.Array,
    *,
    real_weight: float = 0.5,
    temperature: float = 1.0,
    eps: float = 1e-12,
    beta: float = 0.9,
) -> tuple[PyTree, PyTree, jax.Array, dict[str, jax.Array]]:
    """Blend scale-normalized, one-sided soft CPL decisions by source."""
    if not 0.0 <= real_weight <= 1.0:
        raise ValueError("soft-TIES real_weight must be in [0, 1]")
    if temperature <= 0.0:
        raise ValueError("soft-TIES temperature must be positive")
    if eps <= 0.0:
        raise ValueError("soft-TIES eps must be positive")
    synthetic_weight = jnp.asarray(1.0 - real_weight, dtype=jnp.float32)
    anchor_weight = jnp.asarray(real_weight, dtype=jnp.float32)

    def soft_gate(
        update: jax.Array,
        synthetic: jax.Array,
        anchor: jax.Array,
    ) -> jax.Array:
        synthetic_score = _normalized_preservation_score(
            update,
            synthetic,
            eps=eps,
        )
        anchor_score = _normalized_preservation_score(
            update,
            anchor,
            eps=eps,
        )
        temperature_arr = jnp.asarray(temperature, dtype=jnp.float32)
        # Zero or positive evidence leaves an update fully available. Only
        # increasingly confident harmful evidence attenuates it.
        synthetic_gate = jnp.minimum(
            2.0 * jax.nn.sigmoid(temperature_arr * synthetic_score),
            1.0,
        )
        anchor_gate = jnp.minimum(
            2.0 * jax.nn.sigmoid(temperature_arr * anchor_score),
            1.0,
        )
        return (
            synthetic_weight * synthetic_gate
            + anchor_weight * anchor_gate
        )

    gates = jax.tree_util.tree_map(
        soft_gate,
        updates,
        synthetic_reference_grads,
        anchor_reference_grads,
    )
    masked_updates, mask_ema, mask_ema_count, effective_gates = (
        _apply_soft_mask_ema(
            updates,
            gates,
            mask_ema,
            mask_ema_count,
            beta=beta,
        )
    )
    stats = _blend_gate_stats(
        "soft_ties",
        updates,
        gates,
        effective_gates,
        masked_updates,
    )
    stats.update(
        {
            "anchor_blend/soft_ties_real_weight": jnp.asarray(
                real_weight,
                dtype=jnp.float32,
            ),
            "anchor_blend/soft_ties_temperature": jnp.asarray(
                temperature,
                dtype=jnp.float32,
            ),
        }
    )
    return masked_updates, mask_ema, mask_ema_count, stats


def mask_optax_updates_trimmed_ties_ema(
    updates: PyTree,
    synthetic_reference_grads: PyTree,
    anchor_reference_grads: PyTree,
    mask_ema: PyTree,
    mask_ema_count: jax.Array,
    *,
    trim_threshold: float = 0.25,
    elect_margin: float = 1.5,
    eps: float = 1e-12,
    beta: float = 0.9,
) -> tuple[PyTree, PyTree, jax.Array, dict[str, jax.Array]]:
    """Trim weak source evidence and elect a dominant sign on conflicts."""
    if trim_threshold < 0.0:
        raise ValueError("trimmed-TIES threshold must be non-negative")
    if elect_margin < 1.0:
        raise ValueError("trimmed-TIES elect_margin must be at least 1")
    if eps <= 0.0:
        raise ValueError("trimmed-TIES eps must be positive")

    def ties_gate(
        update: jax.Array,
        synthetic: jax.Array,
        anchor: jax.Array,
    ) -> jax.Array:
        synthetic_score = _normalized_preservation_score(
            update,
            synthetic,
            eps=eps,
        )
        anchor_score = _normalized_preservation_score(
            update,
            anchor,
            eps=eps,
        )
        synthetic_abs = jnp.abs(synthetic_score)
        anchor_abs = jnp.abs(anchor_score)
        threshold = jnp.asarray(trim_threshold, dtype=jnp.float32)
        margin = jnp.asarray(elect_margin, dtype=jnp.float32)
        synthetic_active = synthetic_abs >= threshold
        anchor_active = anchor_abs >= threshold
        synthetic_safe = synthetic_score >= 0.0
        anchor_safe = anchor_score >= 0.0
        both_active = synthetic_active & anchor_active
        conflict = both_active & (synthetic_safe != anchor_safe)
        synthetic_dominates = synthetic_abs >= margin * anchor_abs
        anchor_dominates = anchor_abs >= margin * synthetic_abs
        unresolved = conflict & ~synthetic_dominates & ~anchor_dominates

        elected = jnp.where(
            synthetic_active & ~anchor_active,
            synthetic_safe.astype(jnp.float32),
            jnp.where(
                anchor_active & ~synthetic_active,
                anchor_safe.astype(jnp.float32),
                jnp.where(
                    both_active & ~conflict,
                    synthetic_safe.astype(jnp.float32),
                    jnp.where(
                        conflict & synthetic_dominates,
                        synthetic_safe.astype(jnp.float32),
                        jnp.where(
                            conflict & anchor_dominates,
                            anchor_safe.astype(jnp.float32),
                            jnp.asarray(1.0, dtype=jnp.float32),
                        ),
                    ),
                ),
            ),
        )
        return jnp.where(
            unresolved,
            jnp.asarray(0.5, dtype=jnp.float32),
            elected,
        )

    gates = jax.tree_util.tree_map(
        ties_gate,
        updates,
        synthetic_reference_grads,
        anchor_reference_grads,
    )
    masked_updates, mask_ema, mask_ema_count, effective_gates = (
        _apply_soft_mask_ema(
            updates,
            gates,
            mask_ema,
            mask_ema_count,
            beta=beta,
        )
    )
    stats = _blend_gate_stats(
        "trimmed_ties",
        updates,
        gates,
        effective_gates,
        masked_updates,
    )
    stats.update(
        {
            "anchor_blend/trimmed_ties_threshold": jnp.asarray(
                trim_threshold,
                dtype=jnp.float32,
            ),
            "anchor_blend/trimmed_ties_elect_margin": jnp.asarray(
                elect_margin,
                dtype=jnp.float32,
            ),
        }
    )
    return masked_updates, mask_ema, mask_ema_count, stats


def mask_optax_updates_anchor_certified_ema(
    updates: PyTree,
    elected_reference: PyTree,
    mask_ema: PyTree,
    mask_ema_count: jax.Array,
    *,
    beta: float = 0.9,
) -> tuple[PyTree, PyTree, jax.Array, dict[str, jax.Array]]:
    """Apply EMA-smoothed CPL while enforcing current merge abstentions."""
    _validate_mask_ema_beta(beta)
    eligible = jax.tree_util.tree_map(
        lambda reference: reference != 0,
        elected_reference,
    )
    hard_masks = jax.tree_util.tree_map(
        lambda update, reference, is_eligible: (
            is_eligible & (((-update) * reference) >= 0)
        ).astype(jnp.float32),
        updates,
        elected_reference,
        eligible,
    )
    beta_arr = jnp.asarray(beta, dtype=jnp.float32)
    mask_ema = jax.tree_util.tree_map(
        lambda ema, hard: beta_arr * ema + (1.0 - beta_arr) * hard,
        mask_ema,
        hard_masks,
    )
    mask_ema_count = mask_ema_count + jnp.asarray(
        1, dtype=mask_ema_count.dtype
    )
    correction = 1.0 - beta_arr ** mask_ema_count.astype(jnp.float32)
    effective_masks = jax.tree_util.tree_map(
        lambda ema, is_eligible: (
            jnp.clip(ema / correction, 0.0, 1.0)
            * is_eligible.astype(jnp.float32)
        ),
        mask_ema,
        eligible,
    )
    masked_updates = jax.tree_util.tree_map(
        lambda update, mask: update * mask.astype(update.dtype),
        updates,
        effective_masks,
    )
    original_norm = tree_l2_norm(updates)
    masked_norm = tree_l2_norm(masked_updates)
    stats = {
        "anchor_shrinkage/hard_mask_mean": _tree_mean(hard_masks),
        "anchor_shrinkage/ema_mask_mean": _tree_mean(effective_masks),
        "anchor_shrinkage/update_norm_original": original_norm,
        "anchor_shrinkage/update_norm_masked": masked_norm,
        "anchor_shrinkage/update_norm_kept_fraction": (
            masked_norm
            / jnp.maximum(original_norm, jnp.asarray(1e-30, dtype=jnp.float32))
        ),
        "anchor_certified/current_eligible_fraction": _tree_mean(eligible),
    }
    return masked_updates, mask_ema, mask_ema_count, stats


def mask_optax_updates_global_bv_shrinkage_ema(
    updates: PyTree,
    synthetic_reference_grads: PyTree,
    anchor_reference_grads: PyTree,
    synthetic_variance_of_mean: PyTree,
    anchor_variance_of_mean: PyTree,
    mask_ema: PyTree,
    mask_ema_count: jax.Array,
    blend_weight_ema: jax.Array,
    blend_weight_ema_count: jax.Array,
    *,
    beta: float = 0.9,
    blend_weight_beta: float = 0.9,
    eps: float = 1e-12,
    cross_covariance_of_mean: PyTree | None = None,
) -> tuple[
    PyTree,
    PyTree,
    jax.Array,
    jax.Array,
    jax.Array,
    dict[str, jax.Array],
]:
    """Apply global update-aware bias-variance shrinkage before CPL.

    Synthetic references are modeled as potentially biased but stable, while
    the small real-anchor estimate is modeled as less biased but noisier. For
    the current Optax update ``u``, all moments are measured in CPL interaction
    space, ``h = (-u) * g``. A single global real-anchor weight minimizes the
    resulting plug-in mean-squared-error estimate; no source norm alignment is
    applied. Passing paired cross-covariance accounts for the fact that
    anchor-ICL samples descend from the same real anchors. Leaving it unset
    exactly recovers the original independent-source estimator.
    """
    _validate_mask_ema_beta(beta)
    _validate_mask_ema_beta(blend_weight_beta)
    if eps <= 0:
        raise ValueError("bias-variance shrinkage eps must be positive")

    synthetic_variance = tree_sum(
        jax.tree_util.tree_map(
            lambda update, variance: (
                jnp.square(jnp.asarray(update, dtype=jnp.float32))
                * jnp.asarray(variance, dtype=jnp.float32)
            ),
            updates,
            synthetic_variance_of_mean,
        )
    )
    anchor_variance = tree_sum(
        jax.tree_util.tree_map(
            lambda update, variance: (
                jnp.square(jnp.asarray(update, dtype=jnp.float32))
                * jnp.asarray(variance, dtype=jnp.float32)
            ),
            updates,
            anchor_variance_of_mean,
        )
    )
    if cross_covariance_of_mean is None:
        raw_cross_covariance = jnp.asarray(0.0, dtype=jnp.float32)
    else:
        raw_cross_covariance = tree_sum(
            jax.tree_util.tree_map(
                lambda update, covariance: (
                    jnp.square(jnp.asarray(update, dtype=jnp.float32))
                    * jnp.asarray(covariance, dtype=jnp.float32)
                ),
                updates,
                cross_covariance_of_mean,
            )
        )
    covariance_bound = jnp.sqrt(
        jnp.maximum(synthetic_variance, 0.0)
        * jnp.maximum(anchor_variance, 0.0)
    )
    cross_covariance = jnp.clip(
        raw_cross_covariance,
        -covariance_bound,
        covariance_bound,
    )
    source_discrepancy = tree_sum(
        jax.tree_util.tree_map(
            lambda update, synthetic, anchor: jnp.square(
                jnp.asarray(update, dtype=jnp.float32)
                * (
                    jnp.asarray(synthetic, dtype=jnp.float32)
                    - jnp.asarray(anchor, dtype=jnp.float32)
                )
            ),
            updates,
            synthetic_reference_grads,
            anchor_reference_grads,
        )
    )
    estimated_synthetic_bias_sq = jnp.maximum(
        source_discrepancy
        - synthetic_variance
        - anchor_variance
        + 2.0 * cross_covariance,
        jnp.asarray(0.0, dtype=jnp.float32),
    )
    synthetic_error = estimated_synthetic_bias_sq + synthetic_variance
    numerator = synthetic_error - cross_covariance
    denominator = (
        synthetic_error + anchor_variance - 2.0 * cross_covariance
    )
    instant_real_weight = jnp.where(
        denominator > jnp.asarray(eps, dtype=jnp.float32),
        numerator
        / jnp.maximum(denominator, jnp.asarray(eps, dtype=jnp.float32)),
        jnp.asarray(0.5, dtype=jnp.float32),
    )
    instant_real_weight = jnp.clip(instant_real_weight, 0.0, 1.0)

    weight_beta = jnp.asarray(blend_weight_beta, dtype=jnp.float32)
    blend_weight_ema = (
        weight_beta * jnp.asarray(blend_weight_ema, dtype=jnp.float32)
        + (1.0 - weight_beta) * instant_real_weight
    )
    blend_weight_ema_count = blend_weight_ema_count + jnp.asarray(
        1,
        dtype=blend_weight_ema_count.dtype,
    )
    weight_correction = 1.0 - weight_beta ** blend_weight_ema_count.astype(
        jnp.float32
    )
    real_weight = jnp.clip(
        blend_weight_ema / weight_correction,
        0.0,
        1.0,
    )
    fused_reference = jax.tree_util.tree_map(
        lambda synthetic, anchor: (
            synthetic
            * (jnp.asarray(1.0, dtype=synthetic.dtype) - real_weight).astype(
                synthetic.dtype
            )
            + anchor * real_weight.astype(anchor.dtype)
        ),
        synthetic_reference_grads,
        anchor_reference_grads,
    )
    (
        masked_updates,
        mask_ema,
        mask_ema_count,
        stats,
    ) = mask_optax_updates_fixed_blend_ema(
        updates,
        fused_reference,
        mask_ema,
        mask_ema_count,
        beta=beta,
    )

    synthetic_influence = tree_sum(
        jax.tree_util.tree_map(
            lambda update, reference: jnp.abs(
                jnp.asarray(update, dtype=jnp.float32)
                * jnp.asarray(reference, dtype=jnp.float32)
            ),
            updates,
            synthetic_reference_grads,
        )
    )
    anchor_influence = tree_sum(
        jax.tree_util.tree_map(
            lambda update, reference: jnp.abs(
                jnp.asarray(update, dtype=jnp.float32)
                * jnp.asarray(reference, dtype=jnp.float32)
            ),
            updates,
            anchor_reference_grads,
        )
    )
    synthetic_component_influence = (1.0 - real_weight) * synthetic_influence
    anchor_component_influence = real_weight * anchor_influence
    synthetic_component_norm = (
        (1.0 - real_weight) * tree_l2_norm(synthetic_reference_grads)
    )
    anchor_component_norm = real_weight * tree_l2_norm(anchor_reference_grads)
    stats.update(
        {
            "anchor_shrinkage/real_weight_mean": real_weight,
            "anchor_shrinkage/real_weight_tensor_mean": real_weight,
            "anchor_shrinkage/real_weight_min": real_weight,
            "anchor_shrinkage/real_weight_max": real_weight,
            "anchor_blend/fused_gradient_norm": tree_l2_norm(fused_reference),
            "anchor_blend/effective_real_component_norm_fraction": (
                anchor_component_norm
                / jnp.maximum(
                    synthetic_component_norm + anchor_component_norm,
                    jnp.asarray(eps, dtype=jnp.float32),
                )
            ),
            "anchor_blend/synthetic_update_influence": synthetic_influence,
            "anchor_blend/real_update_influence_before_alignment": (
                anchor_influence
            ),
            "anchor_blend/real_update_influence_after_alignment": (
                anchor_influence
            ),
            "anchor_blend/influence_scale": jnp.asarray(
                1.0, dtype=jnp.float32
            ),
            "anchor_blend/effective_real_update_influence_fraction": (
                anchor_component_influence
                / jnp.maximum(
                    synthetic_component_influence
                    + anchor_component_influence,
                    jnp.asarray(eps, dtype=jnp.float32),
                )
            ),
            "anchor_blend/bv_synthetic_variance_of_mean": (
                synthetic_variance
            ),
            "anchor_blend/bv_real_variance_of_mean": anchor_variance,
            "anchor_blend/bv_source_discrepancy": source_discrepancy,
            "anchor_blend/bv_estimated_synthetic_bias_sq": (
                estimated_synthetic_bias_sq
            ),
            "anchor_blend/bv_raw_cross_covariance_of_mean": (
                raw_cross_covariance
            ),
            "anchor_blend/bv_cross_covariance_of_mean": cross_covariance,
            "anchor_blend/bv_cross_correlation": (
                cross_covariance
                / jnp.maximum(
                    covariance_bound,
                    jnp.asarray(eps, dtype=jnp.float32),
                )
            ),
            "anchor_blend/bv_covariance_aware": jnp.asarray(
                float(cross_covariance_of_mean is not None),
                dtype=jnp.float32,
            ),
            "anchor_blend/bv_instant_real_weight": instant_real_weight,
            "anchor_blend/bv_ema_real_weight": real_weight,
            "anchor_blend/bv_weight_ema_count": (
                blend_weight_ema_count.astype(jnp.float32)
            ),
        }
    )
    return (
        masked_updates,
        mask_ema,
        mask_ema_count,
        blend_weight_ema,
        blend_weight_ema_count,
        stats,
    )


def bias_correct_layer_blend_weight_ema(
    blend_weight_ema: jax.Array,
    blend_weight_ema_count: jax.Array,
    *,
    baseline_real_weight: float = 0.5,
    max_deviation: float = 0.25,
    beta: float = 0.9,
) -> jax.Array:
    """Return the bounded, bias-corrected per-layer blend weights."""
    _validate_mask_ema_beta(beta)
    if not 0.0 <= baseline_real_weight <= 1.0:
        raise ValueError("layer blend baseline real weight must be in [0, 1]")
    if not 0.0 <= max_deviation <= 1.0:
        raise ValueError("layer blend max deviation must be in [0, 1]")

    lower = max(0.0, baseline_real_weight - max_deviation)
    upper = min(1.0, baseline_real_weight + max_deviation)
    weight_beta = jnp.asarray(beta, dtype=jnp.float32)
    count = jnp.asarray(blend_weight_ema_count)
    correction = 1.0 - weight_beta ** count.astype(jnp.float32)
    valid = (count > 0) & (correction > 0)
    safe_correction = jnp.where(
        valid,
        correction,
        jnp.asarray(1.0, dtype=jnp.float32),
    )
    corrected = jnp.where(
        valid,
        jnp.asarray(blend_weight_ema, dtype=jnp.float32) / safe_correction,
        jnp.asarray(baseline_real_weight, dtype=jnp.float32),
    )
    return jnp.clip(corrected, lower, upper)


def mask_optax_updates_layer_bv_shrinkage_ema(
    updates: PyTree,
    synthetic_reference_grads: PyTree,
    anchor_reference_grads: PyTree,
    synthetic_variance_of_mean: PyTree,
    anchor_variance_of_mean: PyTree,
    layer_group_ids: PyTree,
    layer_group_labels: tuple[str, ...],
    layer_group_parameter_counts: tuple[float, ...],
    mask_ema: PyTree,
    mask_ema_count: jax.Array,
    blend_weight_ema: jax.Array,
    blend_weight_ema_count: jax.Array,
    *,
    baseline_real_weight: float = 0.5,
    max_deviation: float = 0.25,
    beta: float = 0.9,
    blend_weight_beta: float = 0.9,
    eps: float = 1e-12,
    layer_variance_is_aggregate: bool = False,
) -> tuple[
    PyTree,
    PyTree,
    jax.Array,
    jax.Array,
    jax.Array,
    dict[str, jax.Array],
]:
    """Apply bounded update-aware bias-variance blending per model layer.

    This is the layer-local counterpart of global bias-variance shrinkage.
    For every current optimizer update, it estimates synthetic error and real
    sampling variance in CPL interaction space independently for each layer.
    The plug-in MSE-optimal real weight is bounded around the fixed-blend
    baseline and smoothed with a small replicated EMA vector.
    """
    _validate_mask_ema_beta(beta)
    _validate_mask_ema_beta(blend_weight_beta)
    if eps <= 0:
        raise ValueError("layer bias-variance shrinkage eps must be positive")
    if not 0.0 <= baseline_real_weight <= 1.0:
        raise ValueError("layer blend baseline real weight must be in [0, 1]")
    if not 0.0 <= max_deviation <= 1.0:
        raise ValueError("layer blend max deviation must be in [0, 1]")
    group_count = len(layer_group_labels)
    if group_count <= 0:
        raise ValueError("layer bias-variance shrinkage requires layer groups")
    expected_structure = jax.tree_util.tree_structure(updates)
    trees_to_validate = [
        ("synthetic reference", synthetic_reference_grads),
        ("real reference", anchor_reference_grads),
        ("layer group ids", layer_group_ids),
    ]
    if not layer_variance_is_aggregate:
        trees_to_validate.extend(
            [
                ("synthetic variance", synthetic_variance_of_mean),
                ("real variance", anchor_variance_of_mean),
            ]
        )
    for name, tree in trees_to_validate:
        if jax.tree_util.tree_structure(tree) != expected_structure:
            raise ValueError(f"{name} tree must match the update tree")
    if tuple(blend_weight_ema.shape) != (group_count,):
        raise ValueError(
            "layer blend weight EMA must have one scalar per layer group"
        )
    if len(layer_group_parameter_counts) != group_count:
        raise ValueError(
            "layer blend parameter counts must match the layer groups"
        )
    if layer_variance_is_aggregate:
        if tuple(synthetic_variance_of_mean.shape) != (group_count,):
            raise ValueError(
                "synthetic layer variance must have one scalar per group"
            )
        if tuple(anchor_variance_of_mean.shape) != (group_count,):
            raise ValueError(
                "real layer variance must have one scalar per group"
            )

    group_id_leaves = jax.tree_util.tree_leaves(layer_group_ids)

    def group_sum(mapper, *trees: PyTree) -> jax.Array:
        tree_leaves = [jax.tree_util.tree_leaves(tree) for tree in trees]
        totals = jnp.zeros((group_count,), dtype=jnp.float32)
        for leaf_index, group_id in enumerate(group_id_leaves):
            group_index = int(group_id)
            if not 0 <= group_index < group_count:
                raise ValueError("layer group id is outside the group range")
            values = [leaves[leaf_index] for leaves in tree_leaves]
            totals = totals.at[group_index].add(
                jnp.sum(jnp.asarray(mapper(*values), dtype=jnp.float32))
            )
        return totals

    parameter_count = jnp.asarray(
        layer_group_parameter_counts,
        dtype=jnp.float32,
    )
    if layer_variance_is_aggregate:
        update_energy = group_sum(
            lambda update: jnp.square(
                jnp.asarray(update, dtype=jnp.float32)
            ),
            updates,
        )
        safe_parameter_count = jnp.maximum(
            parameter_count,
            jnp.asarray(1.0, dtype=jnp.float32),
        )
        synthetic_variance = update_energy * (
            jnp.asarray(synthetic_variance_of_mean, dtype=jnp.float32)
            / safe_parameter_count
        )
        anchor_variance = update_energy * (
            jnp.asarray(anchor_variance_of_mean, dtype=jnp.float32)
            / safe_parameter_count
        )
    else:
        synthetic_variance = group_sum(
            lambda update, variance: (
                jnp.square(jnp.asarray(update, dtype=jnp.float32))
                * jnp.asarray(variance, dtype=jnp.float32)
            ),
            updates,
            synthetic_variance_of_mean,
        )
        anchor_variance = group_sum(
            lambda update, variance: (
                jnp.square(jnp.asarray(update, dtype=jnp.float32))
                * jnp.asarray(variance, dtype=jnp.float32)
            ),
            updates,
            anchor_variance_of_mean,
        )
    source_discrepancy = group_sum(
        lambda update, synthetic, anchor: jnp.square(
            jnp.asarray(update, dtype=jnp.float32)
            * (
                jnp.asarray(synthetic, dtype=jnp.float32)
                - jnp.asarray(anchor, dtype=jnp.float32)
            )
        ),
        updates,
        synthetic_reference_grads,
        anchor_reference_grads,
    )
    statistic_scale = jnp.maximum(
        source_discrepancy,
        jnp.maximum(synthetic_variance, anchor_variance),
    )
    valid_statistic = statistic_scale > 0
    valid_layer_count = jnp.sum(valid_statistic.astype(jnp.float32))
    safe_statistic_scale = jnp.where(
        valid_statistic,
        statistic_scale,
        jnp.asarray(1.0, dtype=jnp.float32),
    )
    normalized_source_discrepancy = (
        source_discrepancy / safe_statistic_scale
    )
    normalized_synthetic_variance = (
        synthetic_variance / safe_statistic_scale
    )
    normalized_anchor_variance = anchor_variance / safe_statistic_scale
    normalized_synthetic_bias_sq = jnp.maximum(
        normalized_source_discrepancy
        - normalized_synthetic_variance
        - normalized_anchor_variance,
        jnp.asarray(0.0, dtype=jnp.float32),
    )
    normalized_synthetic_error = (
        normalized_synthetic_bias_sq + normalized_synthetic_variance
    )
    normalized_denominator = (
        normalized_synthetic_error + normalized_anchor_variance
    )
    raw_real_weight = jnp.where(
        valid_statistic & (normalized_denominator > 0),
        normalized_synthetic_error
        / jnp.where(
            normalized_denominator > 0,
            normalized_denominator,
            jnp.asarray(1.0, dtype=jnp.float32),
        ),
        jnp.asarray(baseline_real_weight, dtype=jnp.float32),
    )
    estimated_synthetic_bias_sq = (
        normalized_synthetic_bias_sq * statistic_scale
    )
    lower = max(0.0, baseline_real_weight - max_deviation)
    upper = min(1.0, baseline_real_weight + max_deviation)
    bounded_real_weight = jnp.clip(raw_real_weight, lower, upper)

    weight_beta = jnp.asarray(blend_weight_beta, dtype=jnp.float32)
    blend_weight_ema = (
        weight_beta * jnp.asarray(blend_weight_ema, dtype=jnp.float32)
        + (1.0 - weight_beta) * bounded_real_weight
    )
    blend_weight_ema_count = blend_weight_ema_count + jnp.asarray(
        1,
        dtype=blend_weight_ema_count.dtype,
    )
    real_weight = bias_correct_layer_blend_weight_ema(
        blend_weight_ema,
        blend_weight_ema_count,
        baseline_real_weight=baseline_real_weight,
        max_deviation=max_deviation,
        beta=blend_weight_beta,
    )

    fused_reference = jax.tree_util.tree_map(
        lambda synthetic, anchor, group_id: (
            synthetic
            * (
                jnp.asarray(1.0, dtype=synthetic.dtype)
                - real_weight[int(group_id)].astype(synthetic.dtype)
            )
            + anchor
            * real_weight[int(group_id)].astype(anchor.dtype)
        ),
        synthetic_reference_grads,
        anchor_reference_grads,
        layer_group_ids,
    )
    (
        masked_updates,
        mask_ema,
        mask_ema_count,
        stats,
    ) = mask_optax_updates_fixed_blend_ema(
        updates,
        fused_reference,
        mask_ema,
        mask_ema_count,
        beta=beta,
    )

    synthetic_influence = group_sum(
        lambda update, reference: jnp.abs(
            jnp.asarray(update, dtype=jnp.float32)
            * jnp.asarray(reference, dtype=jnp.float32)
        ),
        updates,
        synthetic_reference_grads,
    )
    anchor_influence = group_sum(
        lambda update, reference: jnp.abs(
            jnp.asarray(update, dtype=jnp.float32)
            * jnp.asarray(reference, dtype=jnp.float32)
        ),
        updates,
        anchor_reference_grads,
    )
    weighted_synthetic_influence = (1.0 - real_weight) * synthetic_influence
    weighted_anchor_influence = real_weight * anchor_influence
    total_weighted_influence = jnp.sum(
        weighted_synthetic_influence + weighted_anchor_influence
    )
    parameter_total = jnp.maximum(
        jnp.sum(parameter_count),
        jnp.asarray(1.0, dtype=jnp.float32),
    )
    real_weight_mean = jnp.sum(real_weight * parameter_count) / parameter_total
    raw_weight_mean = jnp.sum(raw_real_weight * parameter_count) / parameter_total
    bounded_weight_mean = (
        jnp.sum(bounded_real_weight * parameter_count) / parameter_total
    )
    synthetic_component_sq = group_sum(
        lambda synthetic, group_id: jnp.square(
            jnp.asarray(synthetic, dtype=jnp.float32)
            * (1.0 - real_weight[int(group_id)])
        ),
        synthetic_reference_grads,
        layer_group_ids,
    )
    anchor_component_sq = group_sum(
        lambda anchor, group_id: jnp.square(
            jnp.asarray(anchor, dtype=jnp.float32)
            * real_weight[int(group_id)]
        ),
        anchor_reference_grads,
        layer_group_ids,
    )
    synthetic_component_norm = jnp.sqrt(jnp.sum(synthetic_component_sq))
    anchor_component_norm = jnp.sqrt(jnp.sum(anchor_component_sq))

    stats.update(
        {
            "anchor_shrinkage/real_weight_mean": real_weight_mean,
            "anchor_shrinkage/real_weight_tensor_mean": real_weight_mean,
            "anchor_shrinkage/real_weight_min": jnp.min(real_weight),
            "anchor_shrinkage/real_weight_max": jnp.max(real_weight),
            "anchor_blend/fused_gradient_norm": tree_l2_norm(fused_reference),
            "anchor_blend/effective_real_component_norm_fraction": (
                anchor_component_norm
                / jnp.maximum(
                    synthetic_component_norm + anchor_component_norm,
                    jnp.asarray(eps, dtype=jnp.float32),
                )
            ),
            "anchor_blend/synthetic_update_influence": jnp.sum(
                synthetic_influence
            ),
            "anchor_blend/real_update_influence_before_alignment": jnp.sum(
                anchor_influence
            ),
            "anchor_blend/real_update_influence_after_alignment": jnp.sum(
                anchor_influence
            ),
            "anchor_blend/influence_scale": jnp.asarray(
                1.0, dtype=jnp.float32
            ),
            "anchor_blend/effective_real_update_influence_fraction": (
                jnp.sum(weighted_anchor_influence)
                / jnp.maximum(
                    total_weighted_influence,
                    jnp.asarray(eps, dtype=jnp.float32),
                )
            ),
            "anchor_blend/layer_bv_raw_real_weight_mean": raw_weight_mean,
            "anchor_blend/layer_bv_bounded_real_weight_mean": (
                bounded_weight_mean
            ),
            "anchor_blend/layer_bv_real_weight_mean": real_weight_mean,
            "anchor_blend/layer_bv_real_weight_min": jnp.min(real_weight),
            "anchor_blend/layer_bv_real_weight_max": jnp.max(real_weight),
            "anchor_blend/layer_bv_synthetic_variance": jnp.sum(
                synthetic_variance
            ),
            "anchor_blend/layer_bv_real_variance": jnp.sum(anchor_variance),
            "anchor_blend/layer_bv_source_discrepancy": jnp.sum(
                source_discrepancy
            ),
            "anchor_blend/layer_bv_estimated_synthetic_bias_sq": jnp.sum(
                estimated_synthetic_bias_sq
            ),
            "anchor_blend/layer_bv_weight_ema_count": (
                blend_weight_ema_count.astype(jnp.float32)
            ),
            "anchor_blend/layer_bv_valid_layer_fraction": jnp.mean(
                valid_statistic.astype(jnp.float32)
            ),
            "anchor_blend/layer_bv_statistic_scale_min": jnp.where(
                valid_layer_count > 0,
                jnp.min(
                    jnp.where(valid_statistic, statistic_scale, jnp.inf)
                ),
                jnp.asarray(0.0, dtype=jnp.float32),
            ),
            "anchor_blend/layer_bv_statistic_scale_max": jnp.max(
                statistic_scale
            ),
            "anchor_blend/layer_bv_compact_variance": jnp.asarray(
                float(layer_variance_is_aggregate),
                dtype=jnp.float32,
            ),
        }
    )
    metric_labels = tuple(
        label.replace(" ", "_").lower() if " " in label else label
        for label in layer_group_labels
    )
    stats.update(
        {
            "anchor_blend/layer_bv_real_weight/"
            f"{label}": real_weight[index]
            for index, label in enumerate(metric_labels)
        }
    )
    stats.update(
        {
            "anchor_blend/layer_bv_raw_real_weight/"
            f"{label}": raw_real_weight[index]
            for index, label in enumerate(metric_labels)
        }
    )
    stats.update(
        {
            "anchor_blend/layer_bv_bounded_real_weight/"
            f"{label}": bounded_real_weight[index]
            for index, label in enumerate(metric_labels)
        }
    )
    return (
        masked_updates,
        mask_ema,
        mask_ema_count,
        blend_weight_ema,
        blend_weight_ema_count,
        stats,
    )


def mask_optax_updates_influence_blend_ema(
    updates: PyTree,
    synthetic_reference_grads: PyTree,
    anchor_reference_grads: PyTree,
    mask_ema: PyTree,
    mask_ema_count: jax.Array,
    *,
    real_weight: float = 0.5,
    max_scale: float = 2.0,
    eps: float = 1e-12,
    beta: float = 0.9,
) -> tuple[PyTree, PyTree, jax.Array, dict[str, jax.Array]]:
    """Apply a clipped update-influence-aligned synthetic/real blend.

    Alignment uses the quantity that drives the CNL decision,
    ``sum(abs(update * reference))``. Clipping prevents a nearly-zero source
    from being amplified without bound.
    """
    if not 0.0 <= real_weight <= 1.0:
        raise ValueError("influence blend real_weight must be in [0, 1]")
    if max_scale < 1.0:
        raise ValueError("influence blend max_scale must be at least 1")
    if eps <= 0:
        raise ValueError("influence blend eps must be positive")

    synthetic_influence = tree_sum(
        jax.tree_util.tree_map(
            lambda update, reference: jnp.abs(
                jnp.asarray(update, dtype=jnp.float32)
                * jnp.asarray(reference, dtype=jnp.float32)
            ),
            updates,
            synthetic_reference_grads,
        )
    )
    anchor_influence = tree_sum(
        jax.tree_util.tree_map(
            lambda update, reference: jnp.abs(
                jnp.asarray(update, dtype=jnp.float32)
                * jnp.asarray(reference, dtype=jnp.float32)
            ),
            updates,
            anchor_reference_grads,
        )
    )
    valid = (
        synthetic_influence > jnp.asarray(eps, dtype=jnp.float32)
    ) & (anchor_influence > jnp.asarray(eps, dtype=jnp.float32))
    raw_scale = synthetic_influence / jnp.maximum(
        anchor_influence,
        jnp.asarray(eps, dtype=jnp.float32),
    )
    influence_scale = jnp.where(
        valid,
        jnp.clip(raw_scale, 1.0 / max_scale, max_scale),
        jnp.asarray(1.0, dtype=jnp.float32),
    )
    aligned_anchor_grads = scale_tree(anchor_reference_grads, influence_scale)
    synthetic_weight = 1.0 - float(real_weight)
    fused_reference = jax.tree_util.tree_map(
        lambda synthetic, anchor: (
            synthetic * jnp.asarray(synthetic_weight, dtype=synthetic.dtype)
            + anchor * jnp.asarray(real_weight, dtype=anchor.dtype)
        ),
        synthetic_reference_grads,
        aligned_anchor_grads,
    )
    (
        masked_updates,
        mask_ema,
        mask_ema_count,
        stats,
    ) = mask_optax_updates_fixed_blend_ema(
        updates,
        fused_reference,
        mask_ema,
        mask_ema_count,
        beta=beta,
    )
    aligned_anchor_influence = anchor_influence * influence_scale
    synthetic_component = (
        jnp.asarray(synthetic_weight, dtype=jnp.float32)
        * synthetic_influence
    )
    real_component = (
        jnp.asarray(real_weight, dtype=jnp.float32)
        * aligned_anchor_influence
    )
    stats.update(
        {
            "anchor_blend/synthetic_update_influence": synthetic_influence,
            "anchor_blend/real_update_influence_before_alignment": (
                anchor_influence
            ),
            "anchor_blend/real_update_influence_after_alignment": (
                aligned_anchor_influence
            ),
            "anchor_blend/influence_scale": influence_scale,
            "anchor_blend/effective_real_update_influence_fraction": (
                real_component
                / jnp.maximum(
                    synthetic_component + real_component,
                    jnp.asarray(eps, dtype=jnp.float32),
                )
            ),
        }
    )
    return masked_updates, mask_ema, mask_ema_count, stats


def mask_optax_updates_marginalized_blend_ema(
    updates: PyTree,
    synthetic_reference_grads: PyTree,
    anchor_reference_grads: PyTree,
    mask_ema: PyTree,
    mask_ema_count: jax.Array,
    *,
    candidate_real_weights: tuple[float, ...] = (0.25, 0.5, 0.75),
    beta: float = 0.9,
) -> tuple[PyTree, PyTree, jax.Array, dict[str, jax.Array]]:
    """Average CPL decisions over plausible synthetic/real blend weights."""
    if not candidate_real_weights:
        raise ValueError("marginalized blend requires candidate weights")
    if any(weight < 0.0 or weight > 1.0 for weight in candidate_real_weights):
        raise ValueError("marginalized blend weights must be in [0, 1]")

    def marginal_mask(
        update: jax.Array,
        synthetic: jax.Array,
        anchor: jax.Array,
    ) -> jax.Array:
        total = jnp.zeros_like(update, dtype=jnp.float32)
        for real_weight in candidate_real_weights:
            synthetic_weight = 1.0 - float(real_weight)
            reference = (
                synthetic
                * jnp.asarray(synthetic_weight, dtype=synthetic.dtype)
                + anchor * jnp.asarray(real_weight, dtype=anchor.dtype)
            )
            total = total + (
                ((-update) * reference >= 0).astype(jnp.float32)
            )
        return total / jnp.asarray(
            float(len(candidate_real_weights)), dtype=jnp.float32
        )

    marginal_masks = jax.tree_util.tree_map(
        marginal_mask,
        updates,
        synthetic_reference_grads,
        anchor_reference_grads,
    )
    (
        masked_updates,
        mask_ema,
        mask_ema_count,
        effective_masks,
    ) = _apply_soft_mask_ema(
        updates,
        marginal_masks,
        mask_ema,
        mask_ema_count,
        beta=beta,
    )
    original_norm = tree_l2_norm(updates)
    masked_norm = tree_l2_norm(masked_updates)
    uncertain = jax.tree_util.tree_map(
        lambda mask: ((mask > 0.0) & (mask < 1.0)).astype(jnp.float32),
        marginal_masks,
    )
    unanimous_safe = jax.tree_util.tree_map(
        lambda mask: (mask == 1.0).astype(jnp.float32),
        marginal_masks,
    )
    unanimous_harmful = jax.tree_util.tree_map(
        lambda mask: (mask == 0.0).astype(jnp.float32),
        marginal_masks,
    )
    candidate_array = jnp.asarray(candidate_real_weights, dtype=jnp.float32)
    stats = {
        "anchor_shrinkage/hard_mask_mean": _tree_mean(marginal_masks),
        "anchor_shrinkage/ema_mask_mean": _tree_mean(effective_masks),
        "anchor_shrinkage/update_norm_original": original_norm,
        "anchor_shrinkage/update_norm_masked": masked_norm,
        "anchor_shrinkage/update_norm_kept_fraction": (
            masked_norm
            / jnp.maximum(original_norm, jnp.asarray(1e-30, dtype=jnp.float32))
        ),
        "anchor_blend/marginal_candidate_count": jnp.asarray(
            len(candidate_real_weights), dtype=jnp.float32
        ),
        "anchor_blend/marginal_candidate_mean": jnp.mean(candidate_array),
        "anchor_blend/marginal_candidate_min": jnp.min(candidate_array),
        "anchor_blend/marginal_candidate_max": jnp.max(candidate_array),
        "anchor_blend/marginal_uncertain_fraction": _tree_mean(uncertain),
        "anchor_blend/marginal_unanimous_safe_fraction": _tree_mean(
            unanimous_safe
        ),
        "anchor_blend/marginal_unanimous_harmful_fraction": _tree_mean(
            unanimous_harmful
        ),
    }
    return masked_updates, mask_ema, mask_ema_count, stats


def cnl_optax_step(
    params: PyTree,
    grads: PyTree,
    reference_grads: PyTree | None,
    optimizer: optax.GradientTransformation,
    opt_state: optax.OptState,
    *,
    mask_stage: MaskStage = "gradient",
    mask_mode: MaskMode = "hard",
    margin: float = 0.0,
    leak: float = 0.0,
) -> tuple[PyTree, optax.OptState, PyTree]:
    """Apply one Optax step with optional CNL masking.

    Args:
        params: Current model parameters.
        grads: Gradients for the injection/wrong sample or batch.
        reference_grads: Mean gradients from the mastered/correct set. If
            ``None``, this is a normal Optax update.
        optimizer: Any Optax gradient transformation, such as SGD or AdamW.
        opt_state: Current optimizer state.
        mask_stage: ``"gradient"`` matches the main PyTorch ``sft.py`` rule.
            ``"update"`` matches the Adam/momentum variants, masking the final
            optimizer update direction instead.
        mask_mode: ``"hard"`` is original CNL. ``"margin"`` accepts mildly
            negative similarity scores ``>= -margin``. ``"leaky"`` scales
            conflicting coordinates by ``leak`` instead of freezing them.
    """
    if mask_stage not in ("gradient", "update"):
        raise ValueError("mask_stage must be 'gradient' or 'update'")
    _validate_mask_args(mask_mode, margin, leak)

    opt_grads = (
        mask_gradients(
            grads,
            reference_grads,
            mask_mode=mask_mode,
            margin=margin,
            leak=leak,
        )
        if mask_stage == "gradient"
        else grads
    )
    updates, opt_state = optimizer.update(opt_grads, opt_state, params)

    if mask_stage == "update":
        updates = mask_optax_updates(
            updates,
            reference_grads,
            mask_mode=mask_mode,
            margin=margin,
            leak=leak,
        )

    params = optax.apply_updates(params, updates)
    return params, opt_state, updates


def cnl_optax_step_mask_ema(
    params: PyTree,
    grads: PyTree,
    reference_grads: PyTree | None,
    optimizer: optax.GradientTransformation,
    opt_state: optax.OptState,
    mask_ema: PyTree,
    mask_ema_count: jax.Array,
    *,
    mask_stage: MaskStage = "update",
    beta: float = 0.9,
) -> tuple[PyTree, optax.OptState, PyTree, jax.Array, PyTree, dict[str, jax.Array]]:
    """Apply one Optax step with a bias-corrected EMA-smoothed CNL update mask."""
    if mask_stage != "update":
        raise ValueError("cnl_mask_ema supports only mask_stage='update'")
    _validate_mask_ema_beta(beta)

    updates, opt_state = optimizer.update(grads, opt_state, params)
    updates, mask_ema, mask_ema_count, stats = mask_optax_updates_ema(
        updates,
        reference_grads,
        mask_ema,
        mask_ema_count,
        beta=beta,
    )
    params = optax.apply_updates(params, updates)
    return params, opt_state, mask_ema, mask_ema_count, updates, stats
