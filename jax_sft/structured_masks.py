"""Prompt-invariant activation and causal block masks for JAX CNL."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec as P

from jax_sft.cnl import compute_cvar_masks, tree_l2_norm
from jax_sft.dual_circuit import DualCircuitLayerLayout
from jax_sft.ffn_memory import (
    apply_tree_mask,
    right_subspace_components,
    tree_mean,
    update_mask_ema,
)


PyTree = Any
ACTIVATION_BASIS_KEYS = (
    "real_address",
    "synthetic_address",
    "real_value",
    "synthetic_value",
    "real_attention",
    "synthetic_attention",
)


def _named_array_sharding(value: jax.Array) -> NamedSharding | None:
    sharding = getattr(value, "sharding", None)
    if sharding is None:
        sharding = getattr(getattr(value, "aval", None), "sharding", None)
    return sharding if isinstance(sharding, NamedSharding) else None


def structured_projection_paths(
    layouts: tuple[DualCircuitLayerLayout, ...],
) -> tuple[str, ...]:
    """Return gate/up/down/O paths in stable layer-major order."""
    return tuple(
        path
        for layout in layouts
        for path in (
            layout.gate_path,
            layout.up_path,
            layout.down_path,
            layout.o_path,
        )
    )


def structured_projection_labels(
    layouts: tuple[DualCircuitLayerLayout, ...],
) -> tuple[str, ...]:
    return tuple(
        f"layer_{layout.layer}/{component}"
        for layout in layouts
        for component in ("gate_proj", "up_proj", "down_proj", "o_proj")
    )


def _project_right_subspace(
    matrix: jax.Array,
    basis: jax.Array,
    strength: float,
) -> tuple[jax.Array, jax.Array]:
    original_dtype = matrix.dtype
    value = jnp.asarray(matrix, dtype=jnp.float32)
    basis = jnp.asarray(basis, dtype=jnp.float32)
    _, component, _ = right_subspace_components(value, basis)
    removed = jnp.asarray(strength, dtype=jnp.float32) * component
    return (value - removed).astype(original_dtype), jnp.sum(jnp.square(removed))


def project_optax_updates_activation_nullspace_ema(
    updates: dict[str, jax.Array],
    synthetic_group_grads: tuple[dict[str, jax.Array], ...],
    anchor_group_grads: tuple[dict[str, jax.Array], ...],
    layouts: tuple[DualCircuitLayerLayout, ...],
    bases: dict[str, tuple[jax.Array, ...]],
    mask_ema: dict[str, jax.Array],
    mask_ema_count: jax.Array,
    *,
    tail_fraction: float,
    beta: float,
    real_strength: float = 1.0,
    synthetic_strength: float = 0.5,
) -> tuple[
    dict[str, jax.Array],
    dict[str, jax.Array],
    jax.Array,
    dict[str, jax.Array],
]:
    """Project protected map updates away from invariant activation inputs.

    The protected maps use no real-anchor parameter gradient. All parameters
    outside gate/up/down/O retain the ordinary synthetic-and-real CVaR veto.
    """
    if not 0 <= real_strength <= 1 or not 0 <= synthetic_strength <= 1:
        raise ValueError("activation-nullspace strengths must be in [0, 1]")
    if set(bases) != set(ACTIVATION_BASIS_KEYS):
        raise ValueError("activation-nullspace basis dictionary is incomplete")
    if any(len(bases[key]) != len(layouts) for key in ACTIVATION_BASIS_KEYS):
        raise ValueError("activation-nullspace basis layer count is inconsistent")

    synthetic_masks, synthetic_scores, synthetic_tail_count = compute_cvar_masks(
        updates,
        list(synthetic_group_grads),
        tail_fraction=tail_fraction,
    )
    anchor_masks, anchor_scores, anchor_tail_count = compute_cvar_masks(
        updates,
        list(anchor_group_grads),
        tail_fraction=tail_fraction,
    )
    target_paths = set(structured_projection_paths(layouts))
    hard_masks = {
        path: (
            jnp.ones_like(update, dtype=jnp.float32)
            if path in target_paths
            else synthetic_masks[path] * anchor_masks[path]
        )
        for path, update in updates.items()
    }
    effective_masks, mask_ema, mask_ema_count = update_mask_ema(
        hard_masks,
        mask_ema,
        mask_ema_count,
        beta=beta,
    )
    projected_updates = dict(apply_tree_mask(updates, effective_masks))
    real_removed = jnp.asarray(0.0, dtype=jnp.float32)
    synthetic_removed = jnp.asarray(0.0, dtype=jnp.float32)
    component_removed = {
        "address": jnp.asarray(0.0, dtype=jnp.float32),
        "value": jnp.asarray(0.0, dtype=jnp.float32),
        "attention": jnp.asarray(0.0, dtype=jnp.float32),
    }

    def project_two(
        update: jax.Array,
        real_basis: jax.Array,
        synthetic_basis: jax.Array,
        component: str,
    ) -> jax.Array:
        nonlocal real_removed, synthetic_removed
        value, removed_real = _project_right_subspace(
            update,
            real_basis,
            real_strength,
        )
        value, removed_synthetic = _project_right_subspace(
            value,
            synthetic_basis,
            synthetic_strength,
        )
        real_removed = real_removed + removed_real
        synthetic_removed = synthetic_removed + removed_synthetic
        component_removed[component] = (
            component_removed[component] + removed_real + removed_synthetic
        )
        return value

    for layer_index, layout in enumerate(layouts):
        for path in (layout.gate_path, layout.up_path):
            projected_updates[path] = project_two(
                updates[path],
                bases["real_address"][layer_index],
                bases["synthetic_address"][layer_index],
                "address",
            )
        projected_updates[layout.down_path] = project_two(
            updates[layout.down_path],
            bases["real_value"][layer_index],
            bases["synthetic_value"][layer_index],
            "value",
        )
        o_update = updates[layout.o_path]
        o_matrix = o_update.reshape((o_update.shape[0], -1))
        projected_o = project_two(
            o_matrix,
            bases["real_attention"][layer_index],
            bases["synthetic_attention"][layer_index],
            "attention",
        )
        o_sharding = _named_array_sharding(o_update)
        projected_updates[layout.o_path] = jnp.reshape(
            projected_o,
            o_update.shape,
            out_sharding=o_sharding,
        )

    raw_norm = tree_l2_norm(updates)
    final_norm = tree_l2_norm(projected_updates)
    total_removed = real_removed + synthetic_removed
    stats = {
        "activation_nullspace/outside_combined_mask_mean": tree_mean(
            {
                path: hard_masks[path]
                for path in updates
                if path not in target_paths
            }
        ),
        "activation_nullspace/effective_ema_mean": tree_mean(effective_masks),
        "activation_nullspace/synthetic_score_mean": tree_mean(synthetic_scores),
        "activation_nullspace/anchor_score_mean": tree_mean(anchor_scores),
        "activation_nullspace/synthetic_tail_count": jnp.asarray(
            synthetic_tail_count,
            dtype=jnp.float32,
        ),
        "activation_nullspace/anchor_tail_count": jnp.asarray(
            anchor_tail_count,
            dtype=jnp.float32,
        ),
        "activation_nullspace/real_removed_energy": real_removed,
        "activation_nullspace/synthetic_removed_energy": synthetic_removed,
        "activation_nullspace/address_removed_energy": component_removed["address"],
        "activation_nullspace/value_removed_energy": component_removed["value"],
        "activation_nullspace/attention_removed_energy": component_removed[
            "attention"
        ],
        "activation_nullspace/removed_norm_fraction": jnp.sqrt(total_removed)
        / jnp.maximum(raw_norm, 1e-30),
        "activation_nullspace/real_strength": jnp.asarray(
            real_strength,
            dtype=jnp.float32,
        ),
        "activation_nullspace/synthetic_strength": jnp.asarray(
            synthetic_strength,
            dtype=jnp.float32,
        ),
        "update_norm/raw": raw_norm,
        "update_norm/final_structured": final_norm,
        "update_norm/kept_fraction": final_norm / jnp.maximum(raw_norm, 1e-30),
    }
    return projected_updates, mask_ema, mask_ema_count, stats


def apply_structured_block_gates(
    updates: dict[str, jax.Array],
    layouts: tuple[DualCircuitLayerLayout, ...],
    gates: jax.Array,
    *,
    zero_outside: bool,
) -> dict[str, jax.Array]:
    """Apply one scalar to each layer-projection block."""
    paths = structured_projection_paths(layouts)
    gates = jnp.asarray(gates, dtype=jnp.float32)
    if gates.shape != (len(paths),):
        raise ValueError(
            f"structured gates have shape {gates.shape}, expected {(len(paths),)}"
        )
    path_to_index = {path: index for index, path in enumerate(paths)}
    result = {}
    for path, update in updates.items():
        if path in path_to_index:
            result[path] = update * gates[path_to_index[path]].astype(update.dtype)
        elif zero_outside:
            result[path] = jnp.zeros_like(update)
        else:
            result[path] = update
    return result


def structured_block_energies(
    updates: dict[str, jax.Array],
    layouts: tuple[DualCircuitLayerLayout, ...],
) -> jax.Array:
    return jnp.stack(
        [
            jnp.sum(jnp.square(jnp.asarray(updates[path], dtype=jnp.float32)))
            for path in structured_projection_paths(layouts)
        ]
    )


def synthetic_structured_probe_update(
    updates: dict[str, jax.Array],
    synthetic_group_grads: tuple[dict[str, jax.Array], ...],
    layouts: tuple[DualCircuitLayerLayout, ...],
    *,
    tail_fraction: float,
) -> dict[str, jax.Array]:
    """Keep only synthetic-CVaR-safe gate/up/down/O proposal blocks."""
    synthetic_masks, _, _ = compute_cvar_masks(
        updates,
        list(synthetic_group_grads),
        tail_fraction=tail_fraction,
    )
    proposal = apply_tree_mask(updates, synthetic_masks)
    return apply_structured_block_gates(
        proposal,
        layouts,
        jnp.ones((len(structured_projection_paths(layouts)),), dtype=jnp.float32),
        zero_outside=True,
    )


def make_balanced_group_test_patterns(
    group_count: int,
    pattern_count: int,
    seed: int,
) -> np.ndarray:
    """Build deterministic binary interventions with all-on and singleton support."""
    if group_count <= 0 or pattern_count < 2:
        raise ValueError("group tests require positive groups and at least two patterns")
    rng = np.random.default_rng(seed)
    patterns = np.zeros((pattern_count, group_count), dtype=np.float32)
    patterns[0] = 1.0
    # Reserve most of a small probe budget for overlapping interventions; the
    # singleton rows anchor scale without consuming all tests when G is large.
    singleton_count = min(group_count, max(1, (pattern_count - 1) // 3))
    singleton_order = rng.permutation(group_count)[:singleton_count]
    patterns[1 + np.arange(singleton_count), singleton_order] = 1.0
    for row in range(1 + singleton_count, pattern_count):
        active_count = max(1, group_count // 2)
        active = rng.choice(group_count, size=active_count, replace=False)
        patterns[row, active] = 1.0
    return patterns


def fit_causal_group_harm(
    patterns: np.ndarray,
    loss_deltas: np.ndarray,
    *,
    ridge: float,
) -> tuple[np.ndarray, dict[str, float]]:
    """Fit additive block effects to measured mastered-loss changes."""
    x = np.asarray(patterns, dtype=np.float64)
    y = np.asarray(loss_deltas, dtype=np.float64).reshape((-1,))
    if x.ndim != 2 or y.shape != (x.shape[0],):
        raise ValueError("group-test patterns and outcomes have incompatible shapes")
    if ridge < 0:
        raise ValueError("causal group-test ridge must be nonnegative")
    scale = max(float(np.trace(x.T @ x)) / max(x.shape[1], 1), 1.0)
    system = x.T @ x + ridge * scale * np.eye(x.shape[1], dtype=np.float64)
    rhs = x.T @ y
    try:
        coefficients = np.linalg.solve(system, rhs)
    except np.linalg.LinAlgError:
        coefficients = np.linalg.lstsq(system, rhs, rcond=None)[0]
    prediction = x @ coefficients
    residual = float(np.sum(np.square(y - prediction)))
    centered = float(np.sum(np.square(y - y.mean())))
    r2 = 1.0 - residual / max(centered, 1e-30)
    return coefficients.astype(np.float32), {
        "causal_group/fit_r2": float(r2),
        "causal_group/loss_delta_mean": float(y.mean()),
        "causal_group/loss_delta_std": float(y.std()),
        "causal_group/loss_delta_max": float(y.max(initial=0.0)),
        "causal_group/positive_coefficient_fraction": float(
            np.mean(coefficients > 0)
        ),
    }


def solve_causal_soft_gates(
    coefficients: np.ndarray,
    update_energies: np.ndarray,
    *,
    harm_reduction_fraction: float,
    energy_cap: float,
    min_gate: float,
) -> tuple[np.ndarray, dict[str, float]]:
    """Find minimum-energy soft block attenuation under a fitted harm budget."""
    if not 0 <= harm_reduction_fraction <= 1:
        raise ValueError("harm reduction fraction must be in [0, 1]")
    if not 0 <= energy_cap <= 1:
        raise ValueError("causal group energy cap must be in [0, 1]")
    if not 0 <= min_gate <= 1:
        raise ValueError("causal group minimum gate must be in [0, 1]")
    harm = np.maximum(np.asarray(coefficients, dtype=np.float64), 0.0)
    energy = np.maximum(np.asarray(update_energies, dtype=np.float64), 0.0)
    if harm.ndim != 1 or energy.shape != harm.shape:
        raise ValueError("causal coefficients and energies must be matching vectors")
    total_harm = float(harm.sum())
    total_energy = float(energy.sum())
    gates = np.ones_like(harm)
    target = (1.0 - harm_reduction_fraction) * total_harm

    if total_harm > 0 and total_energy > 0:
        denominator = np.maximum(2.0 * energy, 1e-30)

        def gates_for(lam: float) -> np.ndarray:
            return np.clip(1.0 - lam * harm / denominator, min_gate, 1.0)

        high = 1.0
        while float(harm @ gates_for(high)) > target and high < 2.0**40:
            high *= 2.0
        low = 0.0
        for _ in range(64):
            middle = 0.5 * (low + high)
            if float(harm @ gates_for(middle)) > target:
                low = middle
            else:
                high = middle
        gates = gates_for(high)

        removed_energy = float(energy @ np.square(1.0 - gates))
        allowed = energy_cap * total_energy
        if removed_energy > allowed and removed_energy > 0:
            attenuation_scale = np.sqrt(allowed / removed_energy)
            gates = 1.0 - attenuation_scale * (1.0 - gates)

    damage_after = float(harm @ gates)
    removed_energy = float(energy @ np.square(1.0 - gates))
    achieved = (
        (total_harm - damage_after) / total_harm if total_harm > 0 else 0.0
    )
    return gates.astype(np.float32), {
        "causal_group/predicted_harm_before": total_harm,
        "causal_group/predicted_harm_after": damage_after,
        "causal_group/harm_reduction_achieved": float(achieved),
        "causal_group/update_energy_removed_fraction": (
            removed_energy / total_energy if total_energy > 0 else 0.0
        ),
        "causal_group/gate_mean": float(gates.mean()),
        "causal_group/gate_min": float(gates.min(initial=1.0)),
        "causal_group/gate_fraction_lt_0_5": float(np.mean(gates < 0.5)),
    }


def mask_optax_updates_causal_group_test_ema(
    updates: dict[str, jax.Array],
    synthetic_group_grads: tuple[dict[str, jax.Array], ...],
    anchor_group_grads: tuple[dict[str, jax.Array], ...],
    layouts: tuple[DualCircuitLayerLayout, ...],
    block_gates: jax.Array,
    mask_ema: dict[str, jax.Array],
    mask_ema_count: jax.Array,
    *,
    tail_fraction: float,
    beta: float,
) -> tuple[
    dict[str, jax.Array],
    dict[str, jax.Array],
    jax.Array,
    dict[str, jax.Array],
]:
    """Use measured block gates inside target maps and real vetoes elsewhere."""
    synthetic_masks, synthetic_scores, synthetic_tail_count = compute_cvar_masks(
        updates,
        list(synthetic_group_grads),
        tail_fraction=tail_fraction,
    )
    anchor_masks, anchor_scores, anchor_tail_count = compute_cvar_masks(
        updates,
        list(anchor_group_grads),
        tail_fraction=tail_fraction,
    )
    target_paths = set(structured_projection_paths(layouts))
    hard_masks = {
        path: (
            synthetic_masks[path]
            if path in target_paths
            else synthetic_masks[path] * anchor_masks[path]
        )
        for path in updates
    }
    effective_masks, mask_ema, mask_ema_count = update_mask_ema(
        hard_masks,
        mask_ema,
        mask_ema_count,
        beta=beta,
    )
    for path in target_paths:
        effective_masks[path] = effective_masks[path] * synthetic_masks[path]
    masked = dict(apply_tree_mask(updates, effective_masks))
    gated = apply_structured_block_gates(
        masked,
        layouts,
        block_gates,
        zero_outside=False,
    )
    raw_norm = tree_l2_norm(updates)
    final_norm = tree_l2_norm(gated)
    gates = jnp.asarray(block_gates, dtype=jnp.float32)
    stats = {
        "causal_group/outside_combined_mask_mean": tree_mean(
            {
                path: hard_masks[path]
                for path in updates
                if path not in target_paths
            }
        ),
        "causal_group/synthetic_mask_mean": tree_mean(synthetic_masks),
        "causal_group/effective_ema_mean": tree_mean(effective_masks),
        "causal_group/synthetic_score_mean": tree_mean(synthetic_scores),
        "causal_group/anchor_score_mean": tree_mean(anchor_scores),
        "causal_group/synthetic_tail_count": jnp.asarray(
            synthetic_tail_count,
            dtype=jnp.float32,
        ),
        "causal_group/anchor_tail_count": jnp.asarray(
            anchor_tail_count,
            dtype=jnp.float32,
        ),
        "causal_group/gate_mean": gates.mean(),
        "causal_group/gate_min": gates.min(),
        "causal_group/gate_fraction_lt_0_5": (gates < 0.5).mean(),
        "update_norm/raw": raw_norm,
        "update_norm/final_structured": final_norm,
        "update_norm/kept_fraction": final_norm / jnp.maximum(raw_norm, 1e-30),
    }
    return gated, mask_ema, mask_ema_count, stats


def save_activation_subspace_bases(
    path: str | Path,
    bases: dict[str, np.ndarray],
    metadata: dict[str, Any],
) -> None:
    path = Path(path)
    if path.suffix != ".npz":
        raise ValueError("activation-nullspace basis path must end in .npz")
    if set(bases) != set(ACTIVATION_BASIS_KEYS):
        raise ValueError("activation-nullspace basis dictionary is incomplete")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        **{key: np.asarray(value, dtype=np.float32) for key, value in bases.items()},
    )
    path.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_activation_subspace_bases(
    path: str | Path,
    layouts: tuple[DualCircuitLayerLayout, ...],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    path = Path(path)
    with np.load(path, allow_pickle=False) as payload:
        bases = {
            key: np.asarray(payload[key], dtype=np.float32)
            for key in ACTIVATION_BASIS_KEYS
        }
    layer_count = len(layouts)
    expected_features = {
        "address": tuple(layout.hidden_size for layout in layouts),
        "value": tuple(layout.intermediate_size for layout in layouts),
        "attention": tuple(
            layout.query_head_count * layout.head_dim for layout in layouts
        ),
    }
    for key, value in bases.items():
        family = key.split("_", 1)[1]
        if value.ndim != 3 or value.shape[0] != layer_count:
            raise ValueError(f"cached {key} basis has invalid shape {value.shape}")
        if any(value.shape[1] != width for width in expected_features[family]):
            raise ValueError(f"cached {key} feature width does not match model layout")
        gram = np.einsum("lfr,lfs->lrs", value, value)
        identity = np.eye(value.shape[-1], dtype=np.float32)[None, :, :]
        if not np.allclose(gram, identity, rtol=2e-3, atol=2e-3):
            raise ValueError(f"cached {key} bases are not orthonormal")
    metadata_path = path.with_suffix(".json")
    metadata = (
        json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata_path.exists()
        else {}
    )
    return bases, metadata


def shard_activation_subspace_bases(
    bases: dict[str, np.ndarray],
    layouts: tuple[DualCircuitLayerLayout, ...],
    weights: dict[str, jax.Array],
) -> dict[str, tuple[jax.Array, ...]]:
    """Place basis feature axes consistently with their protected map inputs."""
    result: dict[str, tuple[jax.Array, ...]] = {}
    for key, stacked in bases.items():
        family = key.split("_", 1)[1]
        layer_values = []
        for layer_index, layout in enumerate(layouts):
            value = jnp.asarray(stacked[layer_index], dtype=jnp.float32)
            reference_path = (
                layout.gate_path
                if family == "address"
                else layout.down_path
                if family == "value"
                else layout.o_path
            )
            sharding = getattr(weights[reference_path], "sharding", None)
            if isinstance(sharding, NamedSharding):
                feature_spec = None if family == "address" else "model"
                value = jax.device_put(
                    value,
                    NamedSharding(sharding.mesh, P(feature_spec, None)),
                )
            layer_values.append(value)
        result[key] = tuple(layer_values)
    return result
