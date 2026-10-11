"""Candidate-policy objectives for support expansion and consolidation."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np


PyTree = Any


def candidate_on_policy_gold_loss(
    candidate_logits: jax.Array,
    target_index: jax.Array,
    *,
    probability_power: float = 1.0,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return a variance-free on-policy reward loss for a finite action set.

    With ``probability_power=1``, the gradient equals the gradient of negative
    expected exact-match reward, ``-p(y | x)``. The stopped probability weight
    makes the connection to reward-weighted on-policy maximum likelihood
    explicit while avoiding action-sampling variance.
    """
    if probability_power <= 0:
        raise ValueError("on-policy probability power must be positive")
    logits = jnp.asarray(candidate_logits, dtype=jnp.float32)
    targets = jnp.asarray(target_index, dtype=jnp.int32)
    log_probabilities = jax.nn.log_softmax(logits, axis=-1)
    target_log_probability = jnp.take_along_axis(
        log_probabilities,
        targets[..., None],
        axis=-1,
    )[..., 0]
    target_probability = jnp.exp(target_log_probability)
    reward_weight = jax.lax.stop_gradient(
        jnp.power(
            target_probability,
            jnp.asarray(probability_power, dtype=jnp.float32),
        )
    )
    loss = jnp.mean(-reward_weight * target_log_probability)
    return loss, jnp.mean(target_probability), jnp.mean(reward_weight)


def update_support_gate(
    previous_supported: np.ndarray,
    original_gold_probability: np.ndarray,
    invariant_view_correct_fraction: np.ndarray,
    *,
    probability_threshold: float,
    view_fraction_threshold: float,
    exit_ratio: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Update per-example support state with symmetric exit hysteresis."""
    previous = np.asarray(previous_supported, dtype=bool)
    probabilities = np.asarray(
        original_gold_probability,
        dtype=np.float64,
    )
    view_fractions = np.asarray(
        invariant_view_correct_fraction,
        dtype=np.float64,
    )
    if (
        previous.shape != probabilities.shape
        or previous.shape != view_fractions.shape
    ):
        raise ValueError("support-gate arrays must have matching shapes")
    if not 0 <= probability_threshold <= 1:
        raise ValueError("support probability threshold must be in [0, 1]")
    if not 0 <= view_fraction_threshold <= 1:
        raise ValueError("support view threshold must be in [0, 1]")
    if not 0 < exit_ratio <= 1:
        raise ValueError("support exit ratio must be in (0, 1]")
    if (
        np.any(~np.isfinite(probabilities))
        or np.any(probabilities < 0)
        or np.any(probabilities > 1)
    ):
        raise ValueError("support probabilities must be finite and in [0, 1]")
    if (
        np.any(~np.isfinite(view_fractions))
        or np.any(view_fractions < 0)
        or np.any(view_fractions > 1)
    ):
        raise ValueError("support view fractions must be finite and in [0, 1]")

    enter = (
        (probabilities >= probability_threshold)
        & (view_fractions >= view_fraction_threshold)
    )
    stay = (
        (probabilities >= probability_threshold * exit_ratio)
        & (
            view_fractions
            >= view_fraction_threshold * exit_ratio
        )
    )
    supported = np.where(previous, stay, enter)
    entered = supported & ~previous
    exited = previous & ~supported
    return supported, entered, exited


def candidate_reverse_kl(
    student_logits: jax.Array,
    teacher_logits: jax.Array,
    *,
    temperature: float = 1.0,
) -> jax.Array:
    """Return mean reverse KL(student || stop-gradient teacher)."""
    if temperature <= 0:
        raise ValueError("candidate KL temperature must be positive")
    temperature_arr = jnp.asarray(temperature, dtype=jnp.float32)
    student_logp = jax.nn.log_softmax(
        jnp.asarray(student_logits, dtype=jnp.float32) / temperature_arr,
        axis=-1,
    )
    teacher_logp = jax.lax.stop_gradient(
        jax.nn.log_softmax(
            jnp.asarray(teacher_logits, dtype=jnp.float32) / temperature_arr,
            axis=-1,
        )
    )
    per_example = jnp.sum(
        jnp.exp(student_logp) * (student_logp - teacher_logp),
        axis=-1,
    )
    return jnp.mean(per_example) * temperature_arr**2


def candidate_forward_kl(
    student_logits: jax.Array,
    teacher_logits: jax.Array,
    *,
    temperature: float = 1.0,
) -> jax.Array:
    """Return mean forward KL(stop-gradient teacher || student)."""
    if temperature <= 0:
        raise ValueError("candidate KL temperature must be positive")
    temperature_arr = jnp.asarray(temperature, dtype=jnp.float32)
    student_logp = jax.nn.log_softmax(
        jnp.asarray(student_logits, dtype=jnp.float32) / temperature_arr,
        axis=-1,
    )
    teacher_logp = jax.lax.stop_gradient(
        jax.nn.log_softmax(
            jnp.asarray(teacher_logits, dtype=jnp.float32) / temperature_arr,
            axis=-1,
        )
    )
    per_example = jnp.sum(
        jnp.exp(teacher_logp) * (teacher_logp - student_logp),
        axis=-1,
    )
    return jnp.mean(per_example) * temperature_arr**2


def semantic_align_candidate_logits(
    view_logits: jax.Array,
    semantic_map: jax.Array,
) -> jax.Array:
    """Put choice-permuted view logits back in original semantic order.

    ``semantic_map[j]`` gives the original semantic choice represented by
    position ``j`` in the transformed prompt.
    """
    semantic_map = jnp.asarray(semantic_map, dtype=jnp.int32)
    inverse_map = jnp.argsort(semantic_map, axis=-1)
    return jnp.take_along_axis(
        jnp.asarray(view_logits, dtype=jnp.float32),
        inverse_map,
        axis=-1,
    )


def candidate_js_divergence(
    first_logits: jax.Array,
    second_logits: jax.Array,
    *,
    temperature: float = 1.0,
) -> jax.Array:
    """Return mean Jensen-Shannon divergence between candidate policies."""
    if temperature <= 0:
        raise ValueError("candidate JS temperature must be positive")
    temperature_arr = jnp.asarray(temperature, dtype=jnp.float32)
    first_logp = jax.nn.log_softmax(
        jnp.asarray(first_logits, dtype=jnp.float32) / temperature_arr,
        axis=-1,
    )
    second_logp = jax.nn.log_softmax(
        jnp.asarray(second_logits, dtype=jnp.float32) / temperature_arr,
        axis=-1,
    )
    mixture_logp = jnp.logaddexp(first_logp, second_logp) - jnp.log(
        jnp.asarray(2.0, dtype=jnp.float32)
    )
    first_kl = jnp.sum(
        jnp.exp(first_logp) * (first_logp - mixture_logp),
        axis=-1,
    )
    second_kl = jnp.sum(
        jnp.exp(second_logp) * (second_logp - mixture_logp),
        axis=-1,
    )
    return jnp.mean(0.5 * (first_kl + second_kl)) * temperature_arr**2


def nearby_answer_conditioned_teacher_logits(
    student_logits: jax.Array,
    conditioned_logits: jax.Array,
    *,
    mix: float,
) -> jax.Array:
    """Interpolate from the on-policy student toward an answer-conditioned view."""
    if not 0 <= mix <= 1:
        raise ValueError("answer-conditioned teacher mix must be in [0, 1]")
    student = jax.lax.stop_gradient(
        jnp.asarray(student_logits, dtype=jnp.float32)
    )
    conditioned = jax.lax.stop_gradient(
        jnp.asarray(conditioned_logits, dtype=jnp.float32)
    )
    return student + jnp.asarray(mix, dtype=jnp.float32) * (
        conditioned - student
    )


def gold_tilted_teacher_logits(
    student_logits: jax.Array,
    target_index: jax.Array,
    *,
    beta: float,
) -> jax.Array:
    """Construct q(a) proportional to p(a) exp(beta * 1[a=y])."""
    if beta < 0:
        raise ValueError("gold-tilt beta must be nonnegative")
    student = jax.lax.stop_gradient(
        jnp.asarray(student_logits, dtype=jnp.float32)
    )
    tilt = jax.nn.one_hot(
        jnp.asarray(target_index, dtype=jnp.int32),
        student.shape[-1],
        dtype=jnp.float32,
    )
    return student + jnp.asarray(beta, dtype=jnp.float32) * tilt


def update_ema_teacher(
    teacher: PyTree,
    student: PyTree,
    *,
    beta: float,
) -> PyTree:
    """Update an EMA teacher after the student step."""
    if not 0 <= beta < 1:
        raise ValueError("teacher EMA beta must be in [0, 1)")
    beta_arr = jnp.asarray(beta, dtype=jnp.float32)
    return jax.tree_util.tree_map(
        lambda old, new: (
            beta_arr * jnp.asarray(old, dtype=jnp.float32)
            + (1.0 - beta_arr) * jnp.asarray(new, dtype=jnp.float32)
        ).astype(new.dtype),
        teacher,
        student,
    )


def consolidation_ce_weight(
    epoch: int,
    transition_epoch: int,
    *,
    initial_weight: float,
    final_weight: float,
    decay_epochs: int,
) -> float:
    """Linearly decay CE after consolidation begins, never below the floor."""
    if epoch <= 0 or transition_epoch <= 0:
        raise ValueError("epoch and transition_epoch must be positive")
    if initial_weight < 0 or final_weight < 0:
        raise ValueError("CE weights must be nonnegative")
    if final_weight > initial_weight:
        raise ValueError("final CE weight cannot exceed initial CE weight")
    if decay_epochs < 0:
        raise ValueError("CE decay epochs must be nonnegative")
    if decay_epochs == 0:
        return float(final_weight)
    elapsed = max(epoch - transition_epoch, 0)
    progress = min(float(elapsed) / float(decay_epochs), 1.0)
    return float(initial_weight + progress * (final_weight - initial_weight))


def should_enter_consolidation(
    completed_epoch: int,
    injection_accuracy: float,
    *,
    min_support_epochs: int,
    max_support_epochs: int,
    accuracy_threshold: float,
) -> tuple[bool, str]:
    """Decide an epoch-boundary transition using current-task metrics only."""
    if min_support_epochs <= 0:
        raise ValueError("minimum support epochs must be positive")
    if max_support_epochs < min_support_epochs:
        raise ValueError("maximum support epochs must be >= minimum")
    if not 0 <= accuracy_threshold <= 1:
        raise ValueError("support accuracy threshold must be in [0, 1]")
    if completed_epoch >= max_support_epochs:
        return True, "max_support_epochs"
    if (
        completed_epoch >= min_support_epochs
        and injection_accuracy >= accuracy_threshold
    ):
        return True, "accuracy_threshold"
    return False, ""


def resolve_teacher_gated_transition(
    completed_epoch: int,
    *,
    transition_requested: bool,
    transition_reason: str,
    teacher_accuracy: float,
    minimum_teacher_accuracy: float,
    max_support_epochs: int,
) -> tuple[bool, str, bool]:
    """Accept, defer, or terminally reject a requested OPD transition."""
    if completed_epoch <= 0:
        raise ValueError("completed_epoch must be positive")
    if max_support_epochs < completed_epoch:
        raise ValueError("completed_epoch cannot exceed max_support_epochs")
    if not 0 <= teacher_accuracy <= 1:
        raise ValueError("teacher_accuracy must be in [0, 1]")
    if not 0 <= minimum_teacher_accuracy <= 1:
        raise ValueError("minimum_teacher_accuracy must be in [0, 1]")
    if not transition_requested:
        return False, "", False
    if teacher_accuracy >= minimum_teacher_accuracy:
        return True, transition_reason, False
    if completed_epoch >= max_support_epochs:
        return False, "teacher_audit_failed", True
    return False, "teacher_audit_deferred", False
