"""FFN memory-slot masking and diagnostics for the JAX Qwen3 trainer."""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec as P

from jax_sft.cnl import compute_cvar_masks


PyTree = Any
MemoryGranularity = Literal[
    "element",
    "memory_vector",
    "memory_slot",
    "hierarchical",
    "confidence",
    "collision",
]


@dataclass(frozen=True)
class FfnMemorySlotLayout:
    """Parameter paths and intermediate axes for one SwiGLU layer."""

    layer: int
    gate_path: str
    up_path: str
    down_path: str
    gate_shape: tuple[int, ...]
    up_shape: tuple[int, ...]
    down_shape: tuple[int, ...]
    gate_slot_axis: int
    up_slot_axis: int
    down_slot_axis: int
    slot_count: int


_MLP_PATH = re.compile(
    r"^model\.layers\.(?P<layer>\d+)\.mlp\.(?P<projection>gate_proj|up_proj|down_proj)\.weight$"
)


def audit_ffn_memory_slot_layout(
    weights: dict[str, Any],
) -> tuple[tuple[FfnMemorySlotLayout, ...], dict[str, Any]]:
    """Infer and validate the Qwen SwiGLU memory-slot mapping from real weights."""
    by_layer: dict[int, dict[str, str]] = {}
    unsupported = []
    for path in weights:
        match = _MLP_PATH.match(path)
        if match is None:
            if ".mlp." in path and path.endswith(".weight"):
                unsupported.append(path)
            continue
        layer = int(match.group("layer"))
        by_layer.setdefault(layer, {})[match.group("projection")] = path

    if not by_layer:
        raise ValueError("No Qwen gate/up/down MLP weights were found for the FFN layout audit")

    layouts = []
    artifact: dict[str, Any] = {}
    expected_layers = list(range(max(by_layer) + 1))
    if sorted(by_layer) != expected_layers:
        raise ValueError(
            f"FFN layers are not contiguous: found={sorted(by_layer)} expected={expected_layers}"
        )

    for layer in expected_layers:
        paths = by_layer[layer]
        missing = {"gate_proj", "up_proj", "down_proj"} - set(paths)
        if missing:
            raise ValueError(f"layer {layer} is missing MLP projections: {sorted(missing)}")
        gate = weights[paths["gate_proj"]]
        up = weights[paths["up_proj"]]
        down = weights[paths["down_proj"]]
        gate_shape = tuple(int(x) for x in gate.shape)
        up_shape = tuple(int(x) for x in up.shape)
        down_shape = tuple(int(x) for x in down.shape)
        if len(gate_shape) != 2 or len(up_shape) != 2 or len(down_shape) != 2:
            raise ValueError(
                f"layer {layer} MLP projections must be matrices: "
                f"gate={gate_shape} up={up_shape} down={down_shape}"
            )
        if gate_shape != up_shape:
            raise ValueError(f"layer {layer} gate/up shapes differ: {gate_shape} vs {up_shape}")

        # Vendored Qwen forward equations are x @ gate.T, x @ up.T, coeff @ down.T.
        slot_count, hidden_size = gate_shape
        if down_shape != (hidden_size, slot_count):
            raise ValueError(
                f"layer {layer} does not match Qwen [F,D]/[D,F] layout: "
                f"gate={gate_shape} up={up_shape} down={down_shape}"
            )
        layout = FfnMemorySlotLayout(
            layer=layer,
            gate_path=paths["gate_proj"],
            up_path=paths["up_proj"],
            down_path=paths["down_proj"],
            gate_shape=gate_shape,
            up_shape=up_shape,
            down_shape=down_shape,
            gate_slot_axis=0,
            up_slot_axis=0,
            down_slot_axis=1,
            slot_count=slot_count,
        )
        layouts.append(layout)
        artifact[f"layer_{layer}"] = {
            "gate_proj": {
                "path": layout.gate_path,
                "shape": list(layout.gate_shape),
                "slot_axis": layout.gate_slot_axis,
                "input_feature_axis": 1,
                "output_feature_axis": 0,
                "memory_slot_slice": "row[k, :]",
                "forward_equation": "btd,fd->btf",
            },
            "up_proj": {
                "path": layout.up_path,
                "shape": list(layout.up_shape),
                "slot_axis": layout.up_slot_axis,
                "input_feature_axis": 1,
                "output_feature_axis": 0,
                "memory_slot_slice": "row[k, :]",
                "forward_equation": "btd,fd->btf",
            },
            "down_proj": {
                "path": layout.down_path,
                "shape": list(layout.down_shape),
                "slot_axis": layout.down_slot_axis,
                "input_feature_axis": 1,
                "output_feature_axis": 0,
                "memory_slot_slice": "column[:, k]",
                "forward_equation": "btf,df->btd",
            },
            "slot_count": layout.slot_count,
            "shared_ffn_intermediate_index": "k",
            "group_complete": True,
        }

    artifact["summary"] = {
        "layout_audit_passed": True,
        "layer_count": len(layouts),
        "slot_count_total": sum(layout.slot_count for layout in layouts),
        "group_completeness_passed": True,
        "unsupported_parameter_count": len(unsupported),
        "unsupported_parameters": sorted(unsupported),
        "forward_equations": {
            "gate_proj": "btd,fd->btf",
            "up_proj": "btd,fd->btf",
            "down_proj": "btf,df->btd",
        },
    }
    return tuple(layouts), artifact


def write_ffn_memory_slot_layout(path: str | Path, artifact: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def ffn_parameter_paths(
    layouts: tuple[FfnMemorySlotLayout, ...],
) -> tuple[str, ...]:
    """Return every gate/up/down path in stable layer-major order."""
    return tuple(
        path
        for layout in layouts
        for path in (
            layout.gate_path,
            layout.up_path,
            layout.down_path,
        )
    )


def select_ffn_parameters(
    tree: dict[str, Any],
    layouts: tuple[FfnMemorySlotLayout, ...],
) -> dict[str, Any]:
    """Select the Qwen FFN leaves used by FFN-only reference gradients."""
    paths = ffn_parameter_paths(layouts)
    missing = [path for path in paths if path not in tree]
    if missing:
        raise ValueError(f"parameter tree is missing FFN paths: {missing[:3]}")
    return {path: tree[path] for path in paths}


def merge_ffn_parameters(
    non_ffn: dict[str, Any],
    ffn: dict[str, Any],
) -> dict[str, Any]:
    """Merge disjoint flat parameter dictionaries for a model forward pass."""
    overlap = set(non_ffn) & set(ffn)
    if overlap:
        raise ValueError(f"FFN and non-FFN parameter trees overlap: {sorted(overlap)[:3]}")
    return {**non_ffn, **ffn}


def first_order_loss_change(reference_grad: PyTree, additive_update: PyTree) -> jax.Array:
    """Return the Taylor loss change dot(g_ref, additive_update)."""
    products = jax.tree_util.tree_leaves(
        elementwise_first_order_change(reference_grad, additive_update)
    )
    return sum(
        (jnp.sum(product) for product in products),
        jnp.asarray(0.0, dtype=jnp.float32),
    )


def elementwise_first_order_change(reference_grad: PyTree, additive_update: PyTree) -> PyTree:
    return jax.tree_util.tree_map(
        lambda grad, update: jnp.asarray(grad, dtype=jnp.float32)
        * jnp.asarray(update, dtype=jnp.float32),
        reference_grad,
        additive_update,
    )


def grouped_first_order_loss_changes(
    reference_parts: tuple[jax.Array, ...],
    update_parts: tuple[jax.Array, ...],
    slot_axes: tuple[int, ...],
) -> jax.Array:
    """Vectorize the canonical Taylor score over matching functional slices."""
    if not (len(reference_parts) == len(update_parts) == len(slot_axes)):
        raise ValueError("reference parts, update parts, and slot axes must have equal length")
    if not reference_parts:
        raise ValueError("at least one functional slice is required")
    references = tuple(
        jnp.moveaxis(jnp.asarray(part), axis, 0)
        for part, axis in zip(reference_parts, slot_axes)
    )
    updates = tuple(
        jnp.moveaxis(jnp.asarray(part), axis, 0)
        for part, axis in zip(update_parts, slot_axes)
    )
    return jax.vmap(first_order_loss_change)(references, updates)


def elementwise_safe_mask(reference_grad: PyTree, additive_update: PyTree) -> PyTree:
    return jax.tree_util.tree_map(
        lambda change: (change <= 0).astype(jnp.float32),
        elementwise_first_order_change(reference_grad, additive_update),
    )


def apply_tree_mask(tree: PyTree, mask: PyTree) -> PyTree:
    return jax.tree_util.tree_map(lambda value, gate: value * gate.astype(value.dtype), tree, mask)


def tree_mean(tree: PyTree) -> jax.Array:
    leaves = jax.tree_util.tree_leaves(tree)
    if not leaves:
        return jnp.asarray(0.0, dtype=jnp.float32)
    total = sum(
        (jnp.sum(jnp.asarray(leaf, dtype=jnp.float32)) for leaf in leaves),
        jnp.asarray(0.0, dtype=jnp.float32),
    )
    size = sum(leaf.size for leaf in leaves)
    return total / jnp.asarray(float(max(size, 1)), dtype=total.dtype)


def tree_l2_norm(tree: PyTree) -> jax.Array:
    leaves = jax.tree_util.tree_leaves(tree)
    if not leaves:
        return jnp.asarray(0.0, dtype=jnp.float32)
    norm_sq = sum(
        (jnp.sum(jnp.square(jnp.asarray(leaf, dtype=jnp.float32))) for leaf in leaves),
        jnp.asarray(0.0, dtype=jnp.float32),
    )
    return jnp.sqrt(jnp.maximum(norm_sq, 0.0))


def _broadcast_slot(values: jax.Array, axis: int, ndim: int = 2) -> jax.Array:
    shape = [1] * ndim
    shape[axis] = values.shape[0]
    return values.reshape(shape)


def _slot_contributions(
    layout: FfnMemorySlotLayout,
    anchor_grad: dict[str, jax.Array],
    synthetic_update: dict[str, jax.Array],
) -> tuple[jax.Array, jax.Array]:
    changes = elementwise_first_order_change(
        {
            layout.gate_path: anchor_grad[layout.gate_path],
            layout.up_path: anchor_grad[layout.up_path],
            layout.down_path: anchor_grad[layout.down_path],
        },
        {
            layout.gate_path: synthetic_update[layout.gate_path],
            layout.up_path: synthetic_update[layout.up_path],
            layout.down_path: synthetic_update[layout.down_path],
        },
    )
    gate = changes[layout.gate_path]
    up = changes[layout.up_path]
    down = changes[layout.down_path]
    signed = grouped_first_order_loss_changes(
        (
            anchor_grad[layout.gate_path],
            anchor_grad[layout.up_path],
            anchor_grad[layout.down_path],
        ),
        (
            synthetic_update[layout.gate_path],
            synthetic_update[layout.up_path],
            synthetic_update[layout.down_path],
        ),
        (
            layout.gate_slot_axis,
            layout.up_slot_axis,
            layout.down_slot_axis,
        ),
    )
    absolute = jnp.abs(gate).sum(axis=1) + jnp.abs(up).sum(axis=1) + jnp.abs(down).sum(axis=0)
    return signed, absolute


def _replace_slot_slices(
    masks: dict[str, jax.Array],
    layout: FfnMemorySlotLayout,
    gate_values: jax.Array,
) -> None:
    masks[layout.gate_path] = _broadcast_slot(gate_values, layout.gate_slot_axis)
    masks[layout.up_path] = _broadcast_slot(gate_values, layout.up_slot_axis)
    masks[layout.down_path] = _broadcast_slot(gate_values, layout.down_slot_axis)


def _slot_mask_fraction(
    layout: FfnMemorySlotLayout,
    masks: dict[str, jax.Array],
) -> jax.Array:
    """Return each slot's mean gate over its gate/up/down parameter slices."""
    gate = jnp.asarray(masks[layout.gate_path], dtype=jnp.float32)
    up = jnp.asarray(masks[layout.up_path], dtype=jnp.float32)
    down = jnp.asarray(masks[layout.down_path], dtype=jnp.float32)
    gate_sum = gate.sum(axis=1)
    up_sum = up.sum(axis=1)
    down_sum = down.sum(axis=0)
    denominator = gate.shape[1] + up.shape[1] + down.shape[0]
    return (gate_sum + up_sum + down_sum) / max(denominator, 1)


def compute_memory_hard_masks(
    updates: dict[str, jax.Array],
    synthetic_grad: dict[str, jax.Array],
    anchor_grad: dict[str, jax.Array],
    layouts: tuple[FfnMemorySlotLayout, ...],
    *,
    granularity: MemoryGranularity,
    tau: float = 0.25,
    leak: float = 0.10,
    anchor_group_grads: tuple[dict[str, jax.Array], ...] = (),
    confidence_kappa: float = 1.0,
    old_occupancy: tuple[jax.Array, ...] | None = None,
    current_occupancy: tuple[jax.Array, ...] | None = None,
    collision_leak: float = 0.25,
    collision_beta: float = 4.0,
    collision_mode: str = "exp",
    collision_threshold: float = 0.25,
    collision_temperature: float = 0.1,
    occupancy_active_threshold: float = 0.75,
    eps: float = 1e-12,
) -> tuple[dict[str, jax.Array], dict[str, jax.Array], dict[str, jax.Array]]:
    """Build an instantaneous synthetic proposal plus structured real-anchor veto."""
    if granularity not in (
        "element",
        "memory_vector",
        "memory_slot",
        "hierarchical",
        "confidence",
        "collision",
    ):
        raise ValueError(f"unknown memory granularity: {granularity}")
    if not 0 <= leak <= 1:
        raise ValueError("memory-slot leak must be in [0, 1]")
    if tau < 0:
        raise ValueError("memory-slot tau must be nonnegative")
    if granularity == "confidence" and not anchor_group_grads:
        raise ValueError("confidence memory-slot masking requires anchor groups")
    if granularity == "collision" and (old_occupancy is None or current_occupancy is None):
        raise ValueError("collision masking requires old and current occupancy")

    synthetic_masks = dict(elementwise_safe_mask(synthetic_grad, updates))
    synthetic_update = dict(apply_tree_mask(updates, synthetic_masks))
    anchor_element_masks = dict(elementwise_safe_mask(anchor_grad, synthetic_update))
    final_masks = {
        path: synthetic_masks[path] * anchor_element_masks[path]
        for path in updates
    }

    slot_scores = []
    signed_masses = []
    safe_indicators = []
    harmful_indicators = []
    mixed_indicators = []
    collision_values = []
    collision_gates = []
    slot_gates = []

    for layout_idx, layout in enumerate(layouts):
        signed, absolute = _slot_contributions(layout, anchor_grad, synthetic_update)
        q = signed / (absolute + eps)
        slot_scores.append(signed)
        signed_masses.append(q)

        if granularity == "element":
            slot_gates.append(_slot_mask_fraction(layout, final_masks))
            continue
        if granularity == "memory_vector":
            vector_score = grouped_first_order_loss_changes(
                (anchor_grad[layout.down_path],),
                (synthetic_update[layout.down_path],),
                (layout.down_slot_axis,),
            )
            vector_gate = (vector_score <= 0).astype(jnp.float32)
            final_masks[layout.down_path] = synthetic_masks[layout.down_path] * _broadcast_slot(
                vector_gate, layout.down_slot_axis
            )
            safe_indicators.append(vector_gate)
            harmful_indicators.append(1.0 - vector_gate)
            mixed_indicators.append(jnp.zeros_like(vector_gate))
            slot_gates.append(vector_gate)
            continue

        if granularity == "memory_slot":
            slot_gate = (signed <= 0).astype(jnp.float32)
            safe_indicators.append(slot_gate)
            harmful_indicators.append(1.0 - slot_gate)
            mixed_indicators.append(jnp.zeros_like(slot_gate))
            slot_gates.append(slot_gate)
            _replace_slot_slices(
                final_masks,
                layout,
                slot_gate,
            )
            for path in (layout.gate_path, layout.up_path, layout.down_path):
                final_masks[path] = final_masks[path] * synthetic_masks[path]
            continue

        if granularity == "confidence":
            group_scores = []
            for group_grad in anchor_group_grads:
                group_signed, _ = _slot_contributions(layout, group_grad, synthetic_update)
                group_scores.append(group_signed)
            stacked = jnp.stack(group_scores, axis=0)
            mu = stacked.mean(axis=0)
            std = stacked.std(axis=0)
            lower = mu - confidence_kappa * std
            upper = mu + confidence_kappa * std
            safe = upper <= 0
            harmful = lower > 0
            mixed = ~(safe | harmful)
        else:
            safe = q <= -tau
            harmful = q >= tau
            mixed = ~(safe | harmful)

        safe_indicators.append(safe.astype(jnp.float32))
        harmful_indicators.append(harmful.astype(jnp.float32))
        mixed_indicators.append(mixed.astype(jnp.float32))

        if granularity == "collision":
            assert old_occupancy is not None and current_occupancy is not None
            collision = (
                jnp.asarray(old_occupancy[layout_idx], dtype=jnp.float32)
                * jnp.asarray(current_occupancy[layout_idx], dtype=jnp.float32)
                * jnp.maximum(q, 0.0)
            )
            if collision_mode == "exp":
                slot_gate = collision_leak + (1.0 - collision_leak) * jnp.exp(
                    -collision_beta * collision
                )
            elif collision_mode == "sigmoid":
                slot_gate = collision_leak + (1.0 - collision_leak) * jax.nn.sigmoid(
                    (collision_threshold - collision) / collision_temperature
                )
            else:
                raise ValueError("collision mode must be 'exp' or 'sigmoid'")
            collision_values.append(collision)
            collision_gates.append(slot_gate)
            slot_gates.append(slot_gate)
            for path, axis in (
                (layout.gate_path, layout.gate_slot_axis),
                (layout.up_path, layout.up_slot_axis),
                (layout.down_path, layout.down_slot_axis),
            ):
                final_masks[path] = synthetic_masks[path] * _broadcast_slot(slot_gate, axis)
            continue

        # Hierarchical and confidence modes keep the synthetic proposal in safe
        # slots, leak harmful slots, and use elementwise real veto when mixed.
        mixed_gate = _slot_mask_fraction(layout, anchor_element_masks)
        slot_gates.append(
            jnp.where(safe, 1.0, jnp.where(harmful, leak, mixed_gate)).astype(jnp.float32)
        )
        for path, axis in (
            (layout.gate_path, layout.gate_slot_axis),
            (layout.up_path, layout.up_slot_axis),
            (layout.down_path, layout.down_slot_axis),
        ):
            safe_b = _broadcast_slot(safe, axis)
            harmful_b = _broadcast_slot(harmful, axis)
            final_masks[path] = jnp.where(
                safe_b,
                synthetic_masks[path],
                jnp.where(
                    harmful_b,
                    leak * synthetic_masks[path],
                    synthetic_masks[path] * anchor_element_masks[path],
                ),
            ).astype(jnp.float32)

    def concat_or_zero(values: list[jax.Array]) -> jax.Array:
        if not values:
            return jnp.zeros((1,), dtype=jnp.float32)
        return jnp.concatenate([jnp.ravel(value).astype(jnp.float32) for value in values])

    flat_scores = concat_or_zero(slot_scores)
    flat_q = concat_or_zero(signed_masses)
    flat_safe = concat_or_zero(safe_indicators)
    flat_harmful = concat_or_zero(harmful_indicators)
    flat_mixed = concat_or_zero(mixed_indicators)
    flat_collision = concat_or_zero(collision_values)
    flat_collision_gate = concat_or_zero(collision_gates)
    flat_slot_gate = concat_or_zero(slot_gates)
    if old_occupancy is None:
        flat_old_occupancy = jnp.zeros_like(flat_slot_gate)
        protected_occupied = jnp.asarray(0.0, dtype=jnp.float32)
        plastic_free = jnp.asarray(0.0, dtype=jnp.float32)
    else:
        flat_old_occupancy = concat_or_zero(list(old_occupancy))
        protected_occupied = (
            (flat_old_occupancy >= occupancy_active_threshold) & (flat_slot_gate < 0.5)
        ).mean()
        plastic_free = (
            (flat_old_occupancy < 1.0 - occupancy_active_threshold)
            & (flat_slot_gate > 0.75)
        ).mean()
    diagnostics = {
        "mask/synthetic_mean": tree_mean(synthetic_masks),
        "mask/anchor_element_mean": tree_mean(anchor_element_masks),
        "mask/final_mean": tree_mean(final_masks),
        "memory/slot_score_mean": flat_scores.mean(),
        "memory/slot_score_std": flat_scores.std(),
        "memory/signed_mass_mean": flat_q.mean(),
        "memory/signed_mass_abs_mean": jnp.abs(flat_q).mean(),
        "memory/confident_safe_fraction": flat_safe.mean(),
        "memory/confident_harmful_fraction": flat_harmful.mean(),
        "memory/mixed_fraction": flat_mixed.mean(),
        "memory/slot_gate_mean": flat_slot_gate.mean(),
        "memory/slot_gate_std": flat_slot_gate.std(),
        "memory/slot_gate_frac_lt_025": (flat_slot_gate < 0.25).mean(),
        "memory/slot_gate_frac_gt_075": (flat_slot_gate > 0.75).mean(),
        "memory/collision_mean": flat_collision.mean(),
        "memory/collision_max": flat_collision.max(),
        "memory/collision_high_fraction": (flat_collision >= collision_threshold).mean(),
        "memory/collision_gate_mean": flat_collision_gate.mean(),
        "memory/protected_occupied_slot_fraction": protected_occupied,
        "memory/plastic_free_slot_fraction": plastic_free,
    }
    auxiliary = {
        "synthetic_masks": synthetic_masks,
        "anchor_element_masks": anchor_element_masks,
        "synthetic_update": synthetic_update,
    }
    return final_masks, diagnostics, auxiliary


def update_mask_ema(
    hard_masks: PyTree,
    mask_ema: PyTree,
    mask_ema_count: jax.Array,
    *,
    beta: float,
) -> tuple[PyTree, PyTree, jax.Array]:
    if not 0 <= beta < 1:
        raise ValueError("mask EMA beta must be in [0, 1)")
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
    return effective_masks, mask_ema, mask_ema_count


def mask_optax_updates_neuron_aligned_soft_cpl_ema(
    updates: dict[str, jax.Array],
    synthetic_group_grads: tuple[dict[str, jax.Array], ...],
    anchor_group_grads: tuple[dict[str, jax.Array], ...],
    layouts: tuple[FfnMemorySlotLayout, ...],
    mask_ema: dict[str, jax.Array],
    mask_ema_count: jax.Array,
    *,
    tail_fraction: float = 0.2,
    beta: float = 0.9,
    down_strength: float = 1.0,
    address_strength: float = 0.5,
    power: float = 1.0,
    min_gate: float = 0.10,
) -> tuple[
    dict[str, jax.Array],
    dict[str, jax.Array],
    jax.Array,
    dict[str, jax.Array],
]:
    """Use separate gate/up rows and down columns for real-anchor soft CPL.

    Synthetic CVaR remains elementwise. Outside FFNs, the ordinary elementwise
    synthetic-plus-real intersection is retained. Within FFNs, real-anchor
    evidence is pooled independently over gate rows, up rows, and down columns;
    no whole-slot bit is formed.
    """
    if not synthetic_group_grads or not anchor_group_grads:
        raise ValueError("neuron-aligned CPL requires synthetic and anchor groups")
    if not 0 <= down_strength <= 1:
        raise ValueError("down strength must be in [0, 1]")
    if not 0 <= address_strength <= 1:
        raise ValueError("address strength must be in [0, 1]")
    if power <= 0:
        raise ValueError("neuron-aligned CPL power must be positive")
    if not 0 <= min_gate <= 1:
        raise ValueError("neuron-aligned CPL minimum gate must be in [0, 1]")

    synthetic_masks, synthetic_scores, synthetic_tail_count = compute_cvar_masks(
        updates,
        list(synthetic_group_grads),
        tail_fraction=tail_fraction,
    )
    ffn_paths = {
        path
        for layout in layouts
        for path in (
            layout.gate_path,
            layout.up_path,
            layout.down_path,
        )
    }
    outside_updates = {
        path: value for path, value in updates.items() if path not in ffn_paths
    }
    if outside_updates:
        outside_anchor_groups = [
            {
                path: group[path]
                for path in outside_updates
            }
            for group in anchor_group_grads
        ]
        outside_anchor_masks, _, anchor_tail_count = compute_cvar_masks(
            outside_updates,
            outside_anchor_groups,
            tail_fraction=tail_fraction,
        )
    else:
        outside_anchor_masks = {}
        anchor_tail_count = max(
            1,
            math.ceil(len(anchor_group_grads) * tail_fraction),
        )
    masks = dict(synthetic_masks)
    for path in outside_updates:
        masks[path] = synthetic_masks[path] * outside_anchor_masks[path]
    gate_values = []
    up_values = []
    down_values = []

    def component_gate(
        update: jax.Array,
        path: str,
        slot_axis: int,
        strength: float,
    ) -> jax.Array:
        direction = -jnp.moveaxis(
            jnp.asarray(update, dtype=jnp.float32),
            slot_axis,
            0,
        )
        grouped = _stack_ffn_vector_group_grads(
            anchor_group_grads,
            path,
            slot_axis,
        )
        scores = jnp.sum(direction[:, None, :] * grouped, axis=-1)
        safe_fraction = (scores >= 0).astype(jnp.float32).mean(axis=1)
        gate = (
            1.0
            - jnp.asarray(strength, dtype=jnp.float32)
            + jnp.asarray(strength, dtype=jnp.float32)
            * jnp.power(safe_fraction, jnp.asarray(power, dtype=jnp.float32))
        )
        return jnp.maximum(
            jnp.clip(gate, 0.0, 1.0),
            jnp.asarray(min_gate, dtype=jnp.float32),
        )

    for layout in layouts:
        gate = component_gate(
            updates[layout.gate_path],
            layout.gate_path,
            layout.gate_slot_axis,
            address_strength,
        )
        up = component_gate(
            updates[layout.up_path],
            layout.up_path,
            layout.up_slot_axis,
            address_strength,
        )
        down = component_gate(
            updates[layout.down_path],
            layout.down_path,
            layout.down_slot_axis,
            down_strength,
        )
        gate_values.append(gate)
        up_values.append(up)
        down_values.append(down)
        masks[layout.gate_path] = (
            synthetic_masks[layout.gate_path]
            * _broadcast_slot(gate, layout.gate_slot_axis)
        )
        masks[layout.up_path] = (
            synthetic_masks[layout.up_path]
            * _broadcast_slot(up, layout.up_slot_axis)
        )
        masks[layout.down_path] = (
            synthetic_masks[layout.down_path]
            * _broadcast_slot(down, layout.down_slot_axis)
        )

    effective_masks, mask_ema, mask_ema_count = update_mask_ema(
        masks,
        mask_ema,
        mask_ema_count,
        beta=beta,
    )
    # A stale structured gate may soften a decision, but it cannot reactivate a
    # coordinate rejected by the current synthetic mastered set.
    effective_masks = jax.tree_util.tree_map(
        lambda effective, synthetic: effective * synthetic,
        effective_masks,
        synthetic_masks,
    )
    masked_updates = apply_tree_mask(updates, effective_masks)
    raw_norm = tree_l2_norm(updates)
    final_norm = tree_l2_norm(masked_updates)

    def component_mean(values: list[jax.Array]) -> jax.Array:
        total = sum(
            (
                jnp.sum(jnp.asarray(value, dtype=jnp.float32))
                for value in values
            ),
            jnp.asarray(0.0, dtype=jnp.float32),
        )
        size = sum(value.size for value in values)
        return total / jnp.asarray(float(max(size, 1)), dtype=total.dtype)

    stats = {
        "neuron_soft/synthetic_mask_mean": tree_mean(synthetic_masks),
        "neuron_soft/synthetic_score_mean": tree_mean(synthetic_scores),
        "neuron_soft/outside_element_mask_mean": tree_mean(
            {
                path: masks[path]
                for path in outside_updates
            }
        ),
        "neuron_soft/gate_row_mean": component_mean(gate_values),
        "neuron_soft/up_row_mean": component_mean(up_values),
        "neuron_soft/down_column_mean": component_mean(down_values),
        "neuron_soft/combined_mask_mean": tree_mean(masks),
        "neuron_soft/effective_ema_mean": tree_mean(effective_masks),
        "neuron_soft/synthetic_tail_count": jnp.asarray(
            synthetic_tail_count,
            dtype=jnp.float32,
        ),
        "neuron_soft/anchor_tail_count": jnp.asarray(
            anchor_tail_count,
            dtype=jnp.float32,
        ),
        "neuron_soft/down_strength": jnp.asarray(
            down_strength,
            dtype=jnp.float32,
        ),
        "neuron_soft/address_strength": jnp.asarray(
            address_strength,
            dtype=jnp.float32,
        ),
        "update_norm/raw": raw_norm,
        "update_norm/final_structured": final_norm,
        "update_norm/kept_fraction": final_norm / jnp.maximum(raw_norm, 1e-30),
    }
    return masked_updates, mask_ema, mask_ema_count, stats


def mask_optax_updates_ffn_only_neuron_aligned_soft_cpl_ema(
    updates: dict[str, jax.Array],
    synthetic_group_grads: tuple[dict[str, jax.Array], ...],
    anchor_group_grads: tuple[dict[str, jax.Array], ...],
    layouts: tuple[FfnMemorySlotLayout, ...],
    mask_ema: dict[str, jax.Array],
    mask_ema_count: jax.Array,
    *,
    tail_fraction: float = 0.2,
    beta: float = 0.9,
    down_strength: float = 1.0,
    address_strength: float = 0.5,
    power: float = 1.0,
    min_gate: float = 0.10,
) -> tuple[
    dict[str, jax.Array],
    dict[str, jax.Array],
    jax.Array,
    dict[str, jax.Array],
]:
    """Protect only FFN rows/columns and pass every other update through.

    Reference groups may contain only FFN leaves. Synthetic protection remains
    elementwise within the FFN, while real-anchor evidence is pooled separately
    over gate rows, up rows, and down columns.
    """
    if not synthetic_group_grads or not anchor_group_grads:
        raise ValueError("FFN-only neuron CPL requires synthetic and anchor groups")
    if not 0 <= down_strength <= 1:
        raise ValueError("down strength must be in [0, 1]")
    if not 0 <= address_strength <= 1:
        raise ValueError("address strength must be in [0, 1]")
    if power <= 0:
        raise ValueError("neuron-aligned CPL power must be positive")
    if not 0 <= min_gate <= 1:
        raise ValueError("neuron-aligned CPL minimum gate must be in [0, 1]")

    ffn_paths = {
        path
        for layout in layouts
        for path in (
            layout.gate_path,
            layout.up_path,
            layout.down_path,
        )
    }
    ffn_updates = {path: updates[path] for path in ffn_paths}
    for source, groups in (
        ("synthetic", synthetic_group_grads),
        ("anchor", anchor_group_grads),
    ):
        for group in groups:
            missing = ffn_paths - set(group)
            if missing:
                raise ValueError(
                    f"{source} FFN reference group is missing paths: "
                    f"{sorted(missing)[:3]}"
                )

    synthetic_masks, synthetic_scores, synthetic_tail_count = compute_cvar_masks(
        ffn_updates,
        [
            {path: group[path] for path in ffn_paths}
            for group in synthetic_group_grads
        ],
        tail_fraction=tail_fraction,
    )
    ffn_masks = dict(synthetic_masks)
    gate_values = []
    up_values = []
    down_values = []

    def component_gate(
        update: jax.Array,
        path: str,
        slot_axis: int,
        strength: float,
    ) -> jax.Array:
        direction = -jnp.moveaxis(
            jnp.asarray(update, dtype=jnp.float32),
            slot_axis,
            0,
        )
        grouped = _stack_ffn_vector_group_grads(
            anchor_group_grads,
            path,
            slot_axis,
        )
        scores = jnp.sum(direction[:, None, :] * grouped, axis=-1)
        safe_fraction = (scores >= 0).astype(jnp.float32).mean(axis=1)
        gate = (
            1.0
            - jnp.asarray(strength, dtype=jnp.float32)
            + jnp.asarray(strength, dtype=jnp.float32)
            * jnp.power(safe_fraction, jnp.asarray(power, dtype=jnp.float32))
        )
        return jnp.maximum(
            jnp.clip(gate, 0.0, 1.0),
            jnp.asarray(min_gate, dtype=jnp.float32),
        )

    for layout in layouts:
        gate = component_gate(
            updates[layout.gate_path],
            layout.gate_path,
            layout.gate_slot_axis,
            address_strength,
        )
        up = component_gate(
            updates[layout.up_path],
            layout.up_path,
            layout.up_slot_axis,
            address_strength,
        )
        down = component_gate(
            updates[layout.down_path],
            layout.down_path,
            layout.down_slot_axis,
            down_strength,
        )
        gate_values.append(gate)
        up_values.append(up)
        down_values.append(down)
        ffn_masks[layout.gate_path] = (
            synthetic_masks[layout.gate_path]
            * _broadcast_slot(gate, layout.gate_slot_axis)
        )
        ffn_masks[layout.up_path] = (
            synthetic_masks[layout.up_path]
            * _broadcast_slot(up, layout.up_slot_axis)
        )
        ffn_masks[layout.down_path] = (
            synthetic_masks[layout.down_path]
            * _broadcast_slot(down, layout.down_slot_axis)
        )

    ffn_mask_ema = {path: mask_ema[path] for path in ffn_paths}
    effective_ffn_masks, ffn_mask_ema, mask_ema_count = update_mask_ema(
        ffn_masks,
        ffn_mask_ema,
        mask_ema_count,
        beta=beta,
    )
    effective_ffn_masks = jax.tree_util.tree_map(
        lambda effective, synthetic: effective * synthetic,
        effective_ffn_masks,
        synthetic_masks,
    )
    masked_updates = dict(updates)
    updated_mask_ema = dict(mask_ema)
    for path in ffn_paths:
        masked_updates[path] = (
            updates[path] * effective_ffn_masks[path].astype(updates[path].dtype)
        )
        updated_mask_ema[path] = ffn_mask_ema[path]

    def component_mean(values: list[jax.Array]) -> jax.Array:
        total = sum(
            (
                jnp.sum(jnp.asarray(value, dtype=jnp.float32))
                for value in values
            ),
            jnp.asarray(0.0, dtype=jnp.float32),
        )
        size = sum(value.size for value in values)
        return total / jnp.asarray(float(max(size, 1)), dtype=total.dtype)

    non_ffn_updates = {
        path: value for path, value in updates.items() if path not in ffn_paths
    }
    raw_norm = tree_l2_norm(updates)
    final_norm = tree_l2_norm(masked_updates)
    stats = {
        "ffn_only_neuron_soft/synthetic_mask_mean": tree_mean(
            synthetic_masks
        ),
        "ffn_only_neuron_soft/synthetic_score_mean": tree_mean(
            synthetic_scores
        ),
        "ffn_only_neuron_soft/gate_row_mean": component_mean(gate_values),
        "ffn_only_neuron_soft/up_row_mean": component_mean(up_values),
        "ffn_only_neuron_soft/down_column_mean": component_mean(down_values),
        "ffn_only_neuron_soft/ffn_mask_mean": tree_mean(ffn_masks),
        "ffn_only_neuron_soft/effective_ema_mean": tree_mean(
            effective_ffn_masks
        ),
        "ffn_only_neuron_soft/synthetic_tail_count": jnp.asarray(
            synthetic_tail_count,
            dtype=jnp.float32,
        ),
        "ffn_only_neuron_soft/anchor_tail_count": jnp.asarray(
            max(1, math.ceil(len(anchor_group_grads) * tail_fraction)),
            dtype=jnp.float32,
        ),
        "ffn_only_neuron_soft/ffn_update_norm_raw": tree_l2_norm(
            ffn_updates
        ),
        "ffn_only_neuron_soft/ffn_update_norm_after": tree_l2_norm(
            {path: masked_updates[path] for path in ffn_paths}
        ),
        "ffn_only_neuron_soft/non_ffn_update_norm": tree_l2_norm(
            non_ffn_updates
        ),
        "ffn_only_neuron_soft/non_ffn_passthrough": jnp.asarray(
            1.0,
            dtype=jnp.float32,
        ),
        "update_norm/raw": raw_norm,
        "update_norm/final_structured": final_norm,
        "update_norm/kept_fraction": final_norm / jnp.maximum(raw_norm, 1e-30),
    }
    return (
        masked_updates,
        updated_mask_ema,
        mask_ema_count,
        stats,
    )


def _stack_ffn_vector_group_grads(
    group_grads: tuple[dict[str, jax.Array], ...],
    path: str,
    slot_axis: int,
) -> jax.Array:
    if not group_grads:
        raise ValueError("FFN vector CVaR requires at least one real-anchor group")
    # [slot, group, vector-coordinate] keeps the group axis local under the
    # model-parallel FFN sharding used by Qwen.
    return jnp.stack(
        [
            jnp.moveaxis(jnp.asarray(group[path], dtype=jnp.float32), slot_axis, 0)
            for group in group_grads
        ],
        axis=1,
    )


def ffn_value_attribution_scores(
    weights: dict[str, jax.Array],
    anchor_grads: tuple[dict[str, jax.Array], ...],
    layouts: tuple[FfnMemorySlotLayout, ...],
) -> tuple[jax.Array, ...]:
    """Return supportive gradient-times-value scores for every FFN value slot.

    Scaling one down-projection column scales that slot's complete FFN
    sub-update. Its directional derivative is therefore the dot product of the
    column and its loss gradient, equivalent to gradient-times-activation for
    that sub-update after summing over tokens.
    """
    if not anchor_grads:
        raise ValueError("FFN value attribution requires at least one anchor gradient")
    scores = []
    for layout in layouts:
        value_vectors = jnp.moveaxis(
            jnp.asarray(weights[layout.down_path], dtype=jnp.float32),
            layout.down_slot_axis,
            0,
        )
        layer_scores = []
        for grad in anchor_grads:
            grad_vectors = jnp.moveaxis(
                jnp.asarray(grad[layout.down_path], dtype=jnp.float32),
                layout.down_slot_axis,
                0,
            )
            # Gradients are taken from CE loss, so a negative scaling
            # derivative means the value supports the mastered answer.
            layer_scores.append(
                jax.nn.relu(-jnp.sum(value_vectors * grad_vectors, axis=-1))
            )
        scores.append(jnp.stack(layer_scores, axis=0))
    return tuple(scores)


def select_ffn_value_attribution_slots(
    scores_by_layer: tuple[np.ndarray, ...],
    *,
    fraction: float,
    tail_fraction: float,
    probe_label: str = "real anchors",
    eps: float = 1e-12,
) -> tuple[tuple[np.ndarray, ...], dict[str, Any]]:
    """Select an exact global fraction from per-anchor value attributions."""
    if not scores_by_layer:
        raise ValueError("FFN value attribution selection requires layer scores")
    if not 0 < fraction <= 1:
        raise ValueError("FFN value attribution fraction must be in (0, 1]")
    if not 0 < tail_fraction <= 1:
        raise ValueError("FFN value attribution tail fraction must be in (0, 1]")
    if eps <= 0:
        raise ValueError("FFN value attribution epsilon must be positive")

    normalized_layers = []
    aggregated_layers = []
    group_count = None
    for scores in scores_by_layer:
        scores = np.asarray(scores, dtype=np.float32)
        if scores.ndim != 2 or scores.shape[1] == 0:
            raise ValueError(
                "FFN value attribution scores must have shape [probe, slot]"
            )
        if group_count is None:
            group_count = scores.shape[0]
        elif scores.shape[0] != group_count:
            raise ValueError("FFN value attribution probe counts differ across layers")
        if not np.all(np.isfinite(scores)) or np.any(scores < 0):
            raise ValueError("FFN value attribution scores must be finite and nonnegative")
        # Equalize layer scale for each anchor without forcing an equal number
        # of selected slots per layer. Peaky layers can still win more slots.
        normalized = scores / np.maximum(scores.mean(axis=1, keepdims=True), eps)
        normalized_layers.append(normalized)

    assert group_count is not None and group_count > 0
    tail_count = max(1, math.ceil(group_count * tail_fraction))
    for normalized in normalized_layers:
        top_tail = np.partition(
            normalized,
            normalized.shape[0] - tail_count,
            axis=0,
        )[-tail_count:]
        aggregated_layers.append(top_tail.mean(axis=0, dtype=np.float64))

    flat_scores = np.concatenate(aggregated_layers)
    if not np.any(flat_scores > eps):
        raise ValueError("FFN value attribution produced no positive slot scores")
    selected_count = max(1, math.ceil(flat_scores.size * fraction))
    selected_flat = np.zeros(flat_scores.size, dtype=np.float32)
    order = np.argsort(-flat_scores, kind="stable")
    selected_flat[order[:selected_count]] = 1.0

    selectors = []
    layer_metadata = []
    offset = 0
    for layer_idx, aggregated in enumerate(aggregated_layers):
        selector = selected_flat[offset : offset + aggregated.size]
        selected_indices = np.flatnonzero(selector).astype(int).tolist()
        selectors.append(selector)
        layer_metadata.append(
            {
                "layer": layer_idx,
                "slot_count": int(aggregated.size),
                "selected_count": len(selected_indices),
                "selected_slots": selected_indices,
                "aggregated_score_mean": float(np.mean(aggregated)),
                "aggregated_score_max": float(np.max(aggregated)),
            }
        )
        offset += aggregated.size

    metadata = {
        "method": "positive correct-answer gradient-times-value attribution",
        "normalization": "per-probe per-layer mean",
        "aggregation": f"largest-tail mean across {probe_label}",
        "probe_label": probe_label,
        "summary": {
            "layer_count": len(scores_by_layer),
            "probe_count": int(group_count),
            "tail_fraction": float(tail_fraction),
            "tail_count": int(tail_count),
            "requested_fraction": float(fraction),
            "slot_count_total": int(flat_scores.size),
            "selected_count": int(selected_count),
            "selected_fraction": float(selected_count / flat_scores.size),
            "selection_threshold": float(flat_scores[order[selected_count - 1]]),
        },
        "layers": layer_metadata,
    }
    return tuple(selectors), metadata


def build_ffn_value_memory_map(
    real_selectors: tuple[np.ndarray, ...],
    synthetic_selectors: tuple[np.ndarray, ...],
) -> tuple[dict[str, tuple[np.ndarray, ...]], dict[str, Any]]:
    """Partition FFN values by real and synthetic attribution support."""
    if not real_selectors or len(real_selectors) != len(synthetic_selectors):
        raise ValueError("real and synthetic FFN selectors must have equal layers")

    categories: dict[str, list[np.ndarray]] = {
        "shared": [],
        "real_only": [],
        "synthetic_only": [],
        "free": [],
    }
    layer_metadata = []
    totals = {name: 0 for name in categories}
    slot_count_total = 0
    for layer_idx, (real, synthetic) in enumerate(
        zip(real_selectors, synthetic_selectors, strict=True)
    ):
        real = np.asarray(real, dtype=np.float32)
        synthetic = np.asarray(synthetic, dtype=np.float32)
        if real.ndim != 1 or real.shape != synthetic.shape or real.size == 0:
            raise ValueError(
                "real and synthetic FFN selectors must be nonempty matching vectors"
            )
        if not np.all(np.isin(real, (0.0, 1.0))) or not np.all(
            np.isin(synthetic, (0.0, 1.0))
        ):
            raise ValueError("FFN value selectors must be binary")

        shared = real * synthetic
        real_only = real * (1.0 - synthetic)
        synthetic_only = synthetic * (1.0 - real)
        free = (1.0 - real) * (1.0 - synthetic)
        layer_categories = {
            "shared": shared,
            "real_only": real_only,
            "synthetic_only": synthetic_only,
            "free": free,
        }
        layer_summary: dict[str, Any] = {
            "layer": layer_idx,
            "slot_count": int(real.size),
        }
        for name, selector in layer_categories.items():
            categories[name].append(selector.astype(np.float32, copy=False))
            count = int(selector.sum())
            totals[name] += count
            layer_summary[f"{name}_count"] = count
            if name != "free":
                layer_summary[f"{name}_slots"] = (
                    np.flatnonzero(selector).astype(int).tolist()
                )
        layer_metadata.append(layer_summary)
        slot_count_total += int(real.size)

    summary: dict[str, Any] = {
        "layer_count": len(real_selectors),
        "slot_count_total": slot_count_total,
    }
    for name, count in totals.items():
        summary[f"{name}_count"] = count
        summary[f"{name}_fraction"] = count / slot_count_total
    summary["attributed_union_count"] = slot_count_total - totals["free"]
    summary["attributed_union_fraction"] = 1.0 - summary["free_fraction"]

    metadata = {
        "method": "joint real/synthetic FFN value attribution map",
        "categories": {
            "shared": "selected by both real and synthetic support maps",
            "real_only": "selected only by real-anchor support",
            "synthetic_only": "selected only by synthetic-mastered support",
            "free": "selected by neither support map",
        },
        "summary": summary,
        "layers": layer_metadata,
    }
    return (
        {name: tuple(values) for name, values in categories.items()},
        metadata,
    )


def shard_ffn_value_attribution_selectors(
    selectors: tuple[np.ndarray, ...],
    layouts: tuple[FfnMemorySlotLayout, ...],
    weights: dict[str, jax.Array],
) -> tuple[jax.Array, ...]:
    """Place each slot selector on the down-projection FFN sharding."""
    if len(selectors) != len(layouts):
        raise ValueError("FFN value selector layer count does not match layouts")
    sharded = []
    for selector, layout in zip(selectors, layouts, strict=True):
        selector_array = jnp.asarray(selector, dtype=jnp.float32)
        if selector_array.shape != (layout.slot_count,):
            raise ValueError(
                f"layer {layout.layer} selector has shape {selector_array.shape}, "
                f"expected {(layout.slot_count,)}"
            )
        down_sharding = getattr(weights[layout.down_path], "sharding", None)
        if isinstance(down_sharding, NamedSharding):
            selector_array = jax.device_put(
                selector_array,
                NamedSharding(down_sharding.mesh, P("model")),
            )
        sharded.append(selector_array)
    return tuple(sharded)


def write_ffn_value_attribution_selection(
    path: str | Path,
    metadata: dict[str, Any],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")


def _ffn_vector_cvar_tail(
    moved_update: jax.Array,
    grouped_vectors: jax.Array,
    tail_count: int,
) -> tuple[jax.Array, jax.Array]:
    group_changes = jnp.einsum(
        "sgd,sd->sg",
        grouped_vectors,
        jnp.asarray(moved_update, dtype=jnp.float32),
    )
    top_changes, top_indices = jax.lax.top_k(group_changes, tail_count)
    selected = jnp.take_along_axis(
        grouped_vectors,
        top_indices[..., None],
        axis=1,
    )
    return selected.mean(axis=1), top_changes.mean(axis=1)


def _ffn_vector_real_cvar_gate(
    update: jax.Array,
    group_grads: tuple[dict[str, jax.Array], ...],
    path: str,
    slot_axis: int,
    tail_count: int,
) -> tuple[jax.Array, jax.Array]:
    moved_update = jnp.moveaxis(jnp.asarray(update, dtype=jnp.float32), slot_axis, 0)
    grouped_vectors = _stack_ffn_vector_group_grads(group_grads, path, slot_axis)
    _, tail_change = _ffn_vector_cvar_tail(
        moved_update,
        grouped_vectors,
        tail_count,
    )
    return (tail_change <= 0).astype(jnp.float32), tail_change


def _project_ffn_vectors_against_real_cvar(
    update: jax.Array,
    group_grads: tuple[dict[str, jax.Array], ...],
    path: str,
    slot_axis: int,
    tail_count: int,
    projection_passes: int,
    eps: float,
    slot_selector: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    original_dtype = update.dtype
    moved = jnp.moveaxis(jnp.asarray(update, dtype=jnp.float32), slot_axis, 0)
    original = moved
    grouped_vectors = _stack_ffn_vector_group_grads(group_grads, path, slot_axis)
    selector = None
    if slot_selector is not None:
        selector = jnp.asarray(slot_selector, dtype=jnp.float32)
        if selector.shape != (moved.shape[0],):
            raise ValueError(
                f"slot selector has shape {selector.shape}, expected {(moved.shape[0],)}"
            )
    _, before = _ffn_vector_cvar_tail(moved, grouped_vectors, tail_count)
    for _ in range(projection_passes):
        tail_grad, violation = _ffn_vector_cvar_tail(moved, grouped_vectors, tail_count)
        denominator = jnp.sum(jnp.square(tail_grad), axis=-1)
        scale = jax.nn.relu(violation) / jnp.maximum(denominator, eps)
        if selector is not None:
            scale = scale * selector
        moved = moved - scale[:, None] * tail_grad
    _, after = _ffn_vector_cvar_tail(moved, grouped_vectors, tail_count)
    removed_energy = jnp.sum(jnp.square(original - moved))
    return (
        jnp.moveaxis(moved, 0, slot_axis).astype(original_dtype),
        before,
        after,
        removed_energy,
    )


def mask_optax_updates_ffn_vector_cvar_veto_ema(
    updates: dict[str, jax.Array],
    synthetic_group_grads: tuple[dict[str, jax.Array], ...],
    anchor_group_grads: tuple[dict[str, jax.Array], ...],
    layouts: tuple[FfnMemorySlotLayout, ...],
    mask_ema: dict[str, jax.Array],
    mask_ema_count: jax.Array,
    *,
    tail_fraction: float,
    beta: float,
) -> tuple[dict[str, jax.Array], dict[str, jax.Array], jax.Array, dict[str, jax.Array]]:
    """Apply independent real-CVaR gates to every gate/up/down FFN vector."""
    synthetic_masks, synthetic_scores, synthetic_tail_count = compute_cvar_masks(
        updates,
        list(synthetic_group_grads),
        tail_fraction=tail_fraction,
    )
    anchor_element_masks, anchor_scores, anchor_tail_count = compute_cvar_masks(
        updates,
        list(anchor_group_grads),
        tail_fraction=tail_fraction,
    )
    synthetic_update = dict(apply_tree_mask(updates, synthetic_masks))
    structured_anchor_masks = dict(anchor_element_masks)
    vector_gates: dict[str, list[jax.Array]] = {"gate": [], "up": [], "down": []}
    vector_changes: list[jax.Array] = []

    for layout in layouts:
        for component, path, slot_axis in (
            ("gate", layout.gate_path, layout.gate_slot_axis),
            ("up", layout.up_path, layout.up_slot_axis),
            ("down", layout.down_path, layout.down_slot_axis),
        ):
            gate, tail_change = _ffn_vector_real_cvar_gate(
                synthetic_update[path],
                anchor_group_grads,
                path,
                slot_axis,
                anchor_tail_count,
            )
            structured_anchor_masks[path] = _broadcast_slot(
                gate,
                slot_axis,
                updates[path].ndim,
            )
            vector_gates[component].append(gate)
            vector_changes.append(tail_change)

    hard_masks = {
        path: synthetic_masks[path] * structured_anchor_masks[path]
        for path in updates
    }
    effective_masks, mask_ema, mask_ema_count = update_mask_ema(
        hard_masks,
        mask_ema,
        mask_ema_count,
        beta=beta,
    )
    masked_updates = dict(apply_tree_mask(updates, effective_masks))
    flat_changes = jnp.concatenate([jnp.ravel(value) for value in vector_changes])
    stats = {
        "mask/synthetic_mean": tree_mean(synthetic_masks),
        "mask/anchor_element_mean": tree_mean(anchor_element_masks),
        "mask/final_mean": tree_mean(hard_masks),
        "mask/effective_ema_mean": tree_mean(effective_masks),
        "ffn_vector/real_cvar_change_mean": flat_changes.mean(),
        "ffn_vector/real_cvar_harmful_fraction": (flat_changes > 0).mean(),
        "ffn_vector/real_gate_keep_mean": jnp.mean(
            jnp.concatenate(
                [jnp.ravel(value) for values in vector_gates.values() for value in values]
            )
        ),
        "ffn_vector/gate_keep_mean": jnp.mean(jnp.concatenate(vector_gates["gate"])),
        "ffn_vector/up_keep_mean": jnp.mean(jnp.concatenate(vector_gates["up"])),
        "ffn_vector/down_keep_mean": jnp.mean(jnp.concatenate(vector_gates["down"])),
        "ffn_vector/synthetic_tail_count": jnp.asarray(
            synthetic_tail_count, dtype=jnp.float32
        ),
        "ffn_vector/anchor_tail_count": jnp.asarray(anchor_tail_count, dtype=jnp.float32),
        "ffn_vector/synthetic_score_mean": tree_mean(synthetic_scores),
        "ffn_vector/anchor_element_score_mean": tree_mean(anchor_scores),
        "update_norm/raw": tree_l2_norm(updates),
        "update_norm/synthetic_masked": tree_l2_norm(synthetic_update),
        "update_norm/final_structured": tree_l2_norm(masked_updates),
        "update_norm/kept_fraction": tree_l2_norm(masked_updates)
        / jnp.maximum(tree_l2_norm(updates), 1e-30),
    }
    return masked_updates, mask_ema, mask_ema_count, stats


def mask_optax_updates_ffn_vector_cvar_projection_ema(
    updates: dict[str, jax.Array],
    synthetic_group_grads: tuple[dict[str, jax.Array], ...],
    anchor_group_grads: tuple[dict[str, jax.Array], ...],
    layouts: tuple[FfnMemorySlotLayout, ...],
    mask_ema: dict[str, jax.Array],
    mask_ema_count: jax.Array,
    *,
    tail_fraction: float,
    beta: float,
    projection_passes: int = 4,
    eps: float = 1e-12,
) -> tuple[dict[str, jax.Array], dict[str, jax.Array], jax.Array, dict[str, jax.Array]]:
    """Project independent FFN vectors while retaining elementwise vetoes elsewhere."""
    if projection_passes <= 0 or eps <= 0:
        raise ValueError("projection_passes and eps must be positive")
    synthetic_masks, synthetic_scores, synthetic_tail_count = compute_cvar_masks(
        updates,
        list(synthetic_group_grads),
        tail_fraction=tail_fraction,
    )
    anchor_element_masks, anchor_scores, anchor_tail_count = compute_cvar_masks(
        updates,
        list(anchor_group_grads),
        tail_fraction=tail_fraction,
    )

    # Outside FFNs this is exactly the elementwise combined-CVaR baseline. On
    # FFN matrices, EMA tracks only the synthetic proposal; real anchors act
    # afterward through a minimum-change vector projection.
    hard_masks = {
        path: synthetic_masks[path] * anchor_element_masks[path]
        for path in updates
    }
    for layout in layouts:
        for path in (layout.gate_path, layout.up_path, layout.down_path):
            hard_masks[path] = synthetic_masks[path]
    effective_masks, mask_ema, mask_ema_count = update_mask_ema(
        hard_masks,
        mask_ema,
        mask_ema_count,
        beta=beta,
    )
    synthetic_masked = dict(apply_tree_mask(updates, synthetic_masks))
    pre_projection_update = dict(apply_tree_mask(updates, effective_masks))
    projected_updates = dict(pre_projection_update)
    before_by_component: dict[str, list[jax.Array]] = {"gate": [], "up": [], "down": []}
    after_by_component: dict[str, list[jax.Array]] = {"gate": [], "up": [], "down": []}
    removed_energy = jnp.asarray(0.0, dtype=jnp.float32)

    for layout in layouts:
        for component, path, slot_axis in (
            ("gate", layout.gate_path, layout.gate_slot_axis),
            ("up", layout.up_path, layout.up_slot_axis),
            ("down", layout.down_path, layout.down_slot_axis),
        ):
            projected, before, after, removed = _project_ffn_vectors_against_real_cvar(
                pre_projection_update[path],
                anchor_group_grads,
                path,
                slot_axis,
                anchor_tail_count,
                projection_passes,
                eps,
            )
            projected_updates[path] = projected
            before_by_component[component].append(before)
            after_by_component[component].append(after)
            removed_energy += removed

    before_all = jnp.concatenate(
        [jnp.ravel(value) for values in before_by_component.values() for value in values]
    )
    after_all = jnp.concatenate(
        [jnp.ravel(value) for values in after_by_component.values() for value in values]
    )
    proposal_norm = tree_l2_norm(pre_projection_update)
    final_norm = tree_l2_norm(projected_updates)
    stats = {
        "mask/synthetic_mean": tree_mean(synthetic_masks),
        "mask/anchor_element_mean": tree_mean(anchor_element_masks),
        "mask/final_mean": tree_mean(hard_masks),
        "mask/effective_ema_mean": tree_mean(effective_masks),
        "ffn_vector/projection_cvar_before_mean": before_all.mean(),
        "ffn_vector/projection_cvar_after_mean": after_all.mean(),
        "ffn_vector/projection_violation_before_mean": jax.nn.relu(before_all).mean(),
        "ffn_vector/projection_violation_after_mean": jax.nn.relu(after_all).mean(),
        "ffn_vector/projection_harmful_fraction_before": (before_all > 0).mean(),
        "ffn_vector/projection_harmful_fraction_after": (after_all > 1e-7).mean(),
        "ffn_vector/projection_removed_norm_fraction": jnp.sqrt(
            jnp.maximum(removed_energy, 0.0)
        )
        / jnp.maximum(proposal_norm, 1e-30),
        "ffn_vector/gate_violation_after_mean": jax.nn.relu(
            jnp.concatenate(after_by_component["gate"])
        ).mean(),
        "ffn_vector/up_violation_after_mean": jax.nn.relu(
            jnp.concatenate(after_by_component["up"])
        ).mean(),
        "ffn_vector/down_violation_after_mean": jax.nn.relu(
            jnp.concatenate(after_by_component["down"])
        ).mean(),
        "ffn_vector/synthetic_tail_count": jnp.asarray(
            synthetic_tail_count, dtype=jnp.float32
        ),
        "ffn_vector/anchor_tail_count": jnp.asarray(anchor_tail_count, dtype=jnp.float32),
        "ffn_vector/synthetic_score_mean": tree_mean(synthetic_scores),
        "ffn_vector/anchor_element_score_mean": tree_mean(anchor_scores),
        "update_norm/raw": tree_l2_norm(updates),
        "update_norm/synthetic_masked": tree_l2_norm(synthetic_masked),
        "update_norm/final_structured": final_norm,
        "update_norm/kept_fraction": final_norm
        / jnp.maximum(tree_l2_norm(updates), 1e-30),
    }
    return projected_updates, mask_ema, mask_ema_count, stats


def mask_optax_updates_ffn_value_attribution_projection_ema(
    updates: dict[str, jax.Array],
    synthetic_group_grads: tuple[dict[str, jax.Array], ...],
    anchor_group_grads: tuple[dict[str, jax.Array], ...],
    layouts: tuple[FfnMemorySlotLayout, ...],
    value_selectors: tuple[jax.Array, ...],
    mask_ema: dict[str, jax.Array],
    mask_ema_count: jax.Array,
    *,
    tail_fraction: float,
    beta: float,
    projection_passes: int = 4,
    eps: float = 1e-12,
) -> tuple[dict[str, jax.Array], dict[str, jax.Array], jax.Array, dict[str, jax.Array]]:
    """Project real-anchor conflicts only in attributed FFN value vectors.

    Non-FFN parameters retain the combined elementwise synthetic-plus-real
    baseline. Within every FFN, the synthetic proposal remains elementwise;
    real anchors constrain only selected down-projection columns.
    """
    if len(value_selectors) != len(layouts):
        raise ValueError("FFN value selector layer count does not match layouts")
    if projection_passes <= 0 or eps <= 0:
        raise ValueError("projection_passes and eps must be positive")
    synthetic_masks, synthetic_scores, synthetic_tail_count = compute_cvar_masks(
        updates,
        list(synthetic_group_grads),
        tail_fraction=tail_fraction,
    )
    anchor_element_masks, anchor_scores, anchor_tail_count = compute_cvar_masks(
        updates,
        list(anchor_group_grads),
        tail_fraction=tail_fraction,
    )

    hard_masks = {
        path: synthetic_masks[path] * anchor_element_masks[path]
        for path in updates
    }
    for layout in layouts:
        for path in (layout.gate_path, layout.up_path, layout.down_path):
            hard_masks[path] = synthetic_masks[path]
    effective_masks, mask_ema, mask_ema_count = update_mask_ema(
        hard_masks,
        mask_ema,
        mask_ema_count,
        beta=beta,
    )
    synthetic_masked = dict(apply_tree_mask(updates, synthetic_masks))
    pre_projection_update = dict(apply_tree_mask(updates, effective_masks))
    projected_updates = dict(pre_projection_update)
    before_by_layer = []
    after_by_layer = []
    selectors = []
    removed_energy = jnp.asarray(0.0, dtype=jnp.float32)

    for layout, selector in zip(layouts, value_selectors, strict=True):
        projected, before, after, removed = _project_ffn_vectors_against_real_cvar(
            pre_projection_update[layout.down_path],
            anchor_group_grads,
            layout.down_path,
            layout.down_slot_axis,
            anchor_tail_count,
            projection_passes,
            eps,
            slot_selector=selector,
        )
        projected_updates[layout.down_path] = projected
        before_by_layer.append(before)
        after_by_layer.append(after)
        selectors.append(jnp.asarray(selector, dtype=jnp.float32))
        removed_energy += removed

    before_all = jnp.concatenate([jnp.ravel(value) for value in before_by_layer])
    after_all = jnp.concatenate([jnp.ravel(value) for value in after_by_layer])
    selector_all = jnp.concatenate([jnp.ravel(value) for value in selectors])
    selected_count = jnp.sum(selector_all)
    selected_denominator = jnp.maximum(selected_count, 1.0)

    def selected_mean(value: jax.Array) -> jax.Array:
        return jnp.sum(value * selector_all) / selected_denominator

    proposal_norm = tree_l2_norm(pre_projection_update)
    final_norm = tree_l2_norm(projected_updates)
    stats = {
        "mask/synthetic_mean": tree_mean(synthetic_masks),
        "mask/anchor_element_mean": tree_mean(anchor_element_masks),
        "mask/final_mean": tree_mean(hard_masks),
        "mask/effective_ema_mean": tree_mean(effective_masks),
        "ffn_value/attribution_selected_count": selected_count,
        "ffn_value/attribution_selected_fraction": jnp.mean(selector_all),
        "ffn_value/projection_cvar_before_mean": selected_mean(before_all),
        "ffn_value/projection_cvar_after_mean": selected_mean(after_all),
        "ffn_value/projection_violation_before_mean": selected_mean(
            jax.nn.relu(before_all)
        ),
        "ffn_value/projection_violation_after_mean": selected_mean(
            jax.nn.relu(after_all)
        ),
        "ffn_value/projection_harmful_fraction_before": selected_mean(
            (before_all > 0).astype(jnp.float32)
        ),
        "ffn_value/projection_harmful_fraction_after": selected_mean(
            (after_all > 1e-7).astype(jnp.float32)
        ),
        "ffn_value/projection_removed_norm_fraction": jnp.sqrt(
            jnp.maximum(removed_energy, 0.0)
        )
        / jnp.maximum(proposal_norm, 1e-30),
        "ffn_value/synthetic_tail_count": jnp.asarray(
            synthetic_tail_count, dtype=jnp.float32
        ),
        "ffn_value/anchor_tail_count": jnp.asarray(
            anchor_tail_count, dtype=jnp.float32
        ),
        "ffn_value/synthetic_score_mean": tree_mean(synthetic_scores),
        "ffn_value/anchor_element_score_mean": tree_mean(anchor_scores),
        "update_norm/raw": tree_l2_norm(updates),
        "update_norm/synthetic_masked": tree_l2_norm(synthetic_masked),
        "update_norm/final_structured": final_norm,
        "update_norm/kept_fraction": final_norm
        / jnp.maximum(tree_l2_norm(updates), 1e-30),
    }
    return projected_updates, mask_ema, mask_ema_count, stats


def _project_ffn_values_against_joint_memory_map(
    update: jax.Array,
    synthetic_group_grads: tuple[dict[str, jax.Array], ...],
    anchor_group_grads: tuple[dict[str, jax.Array], ...],
    path: str,
    slot_axis: int,
    synthetic_tail_count: int,
    anchor_tail_count: int,
    category_selectors: dict[str, jax.Array],
    projection_passes: int,
    synthetic_projection_strength: float,
    eps: float,
) -> tuple[
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
]:
    """Project mapped FFN values against their source-specific CVaR constraints."""
    original_dtype = update.dtype
    moved = jnp.moveaxis(jnp.asarray(update, dtype=jnp.float32), slot_axis, 0)
    original = moved
    synthetic_vectors = _stack_ffn_vector_group_grads(
        synthetic_group_grads,
        path,
        slot_axis,
    )
    anchor_vectors = _stack_ffn_vector_group_grads(
        anchor_group_grads,
        path,
        slot_axis,
    )
    selectors = {
        name: jnp.asarray(category_selectors[name], dtype=jnp.float32)
        for name in ("shared", "real_only", "synthetic_only", "free")
    }
    for name, selector in selectors.items():
        if selector.shape != (moved.shape[0],):
            raise ValueError(
                f"{name} selector has shape {selector.shape}, "
                f"expected {(moved.shape[0],)}"
            )
    synthetic_selector = selectors["shared"] + selectors["synthetic_only"]
    anchor_selector = selectors["shared"] + selectors["real_only"]

    _, synthetic_before = _ffn_vector_cvar_tail(
        moved,
        synthetic_vectors,
        synthetic_tail_count,
    )
    _, anchor_before = _ffn_vector_cvar_tail(
        moved,
        anchor_vectors,
        anchor_tail_count,
    )

    def project_source(
        current: jax.Array,
        grouped_vectors: jax.Array,
        tail_count: int,
        selector: jax.Array,
    ) -> jax.Array:
        tail_grad, violation = _ffn_vector_cvar_tail(
            current,
            grouped_vectors,
            tail_count,
        )
        denominator = jnp.sum(jnp.square(tail_grad), axis=-1)
        scale = jax.nn.relu(violation) / jnp.maximum(denominator, eps)
        return current - (scale * selector)[:, None] * tail_grad

    # Alternating the order avoids always privileging one side of the shared
    # intersection when the two half-spaces are not orthogonal.
    for pass_idx in range(projection_passes):
        if pass_idx % 2 == 0:
            moved = project_source(
                moved,
                synthetic_vectors,
                synthetic_tail_count,
                synthetic_selector,
            )
            moved = project_source(
                moved,
                anchor_vectors,
                anchor_tail_count,
                anchor_selector,
            )
        else:
            moved = project_source(
                moved,
                anchor_vectors,
                anchor_tail_count,
                anchor_selector,
            )
            moved = project_source(
                moved,
                synthetic_vectors,
                synthetic_tail_count,
                synthetic_selector,
            )

    full_strength = selectors["shared"] + selectors["real_only"]
    slot_strength = (
        full_strength
        + synthetic_projection_strength * selectors["synthetic_only"]
    )
    moved = original + slot_strength[:, None] * (moved - original)
    _, synthetic_after = _ffn_vector_cvar_tail(
        moved,
        synthetic_vectors,
        synthetic_tail_count,
    )
    _, anchor_after = _ffn_vector_cvar_tail(
        moved,
        anchor_vectors,
        anchor_tail_count,
    )
    removed_energy = jnp.sum(jnp.square(original - moved))
    return (
        jnp.moveaxis(moved, 0, slot_axis).astype(original_dtype),
        synthetic_before,
        synthetic_after,
        anchor_before,
        anchor_after,
        removed_energy,
    )


def mask_optax_updates_ffn_joint_memory_map_projection_ema(
    updates: dict[str, jax.Array],
    synthetic_group_grads: tuple[dict[str, jax.Array], ...],
    anchor_group_grads: tuple[dict[str, jax.Array], ...],
    layouts: tuple[FfnMemorySlotLayout, ...],
    value_memory_map: dict[str, tuple[jax.Array, ...]],
    mask_ema: dict[str, jax.Array],
    mask_ema_count: jax.Array,
    *,
    tail_fraction: float,
    beta: float,
    projection_passes: int = 4,
    synthetic_projection_strength: float = 0.5,
    eps: float = 1e-12,
) -> tuple[dict[str, jax.Array], dict[str, jax.Array], jax.Array, dict[str, jax.Array]]:
    """Apply a joint real/synthetic memory map to FFN value-vector updates.

    Outside FFNs this is the combined elementwise CVaR baseline. Gate/up rows
    retain only the synthetic elementwise proposal. Down-projection columns are
    governed entirely by their frozen joint-map category.
    """
    required_categories = {"shared", "real_only", "synthetic_only", "free"}
    if set(value_memory_map) != required_categories:
        raise ValueError(
            "joint FFN value memory map must contain shared, real_only, "
            "synthetic_only, and free selectors"
        )
    if any(len(value_memory_map[name]) != len(layouts) for name in required_categories):
        raise ValueError("joint FFN value memory-map layer count does not match layouts")
    if projection_passes <= 0 or eps <= 0:
        raise ValueError("projection_passes and eps must be positive")
    if not 0 <= synthetic_projection_strength <= 1:
        raise ValueError("synthetic_projection_strength must be in [0, 1]")

    synthetic_masks, synthetic_scores, synthetic_tail_count = compute_cvar_masks(
        updates,
        list(synthetic_group_grads),
        tail_fraction=tail_fraction,
    )
    anchor_element_masks, anchor_scores, anchor_tail_count = compute_cvar_masks(
        updates,
        list(anchor_group_grads),
        tail_fraction=tail_fraction,
    )

    hard_masks = {
        path: synthetic_masks[path] * anchor_element_masks[path]
        for path in updates
    }
    for layout in layouts:
        hard_masks[layout.gate_path] = synthetic_masks[layout.gate_path]
        hard_masks[layout.up_path] = synthetic_masks[layout.up_path]
        hard_masks[layout.down_path] = jnp.ones_like(
            synthetic_masks[layout.down_path]
        )
    effective_masks, mask_ema, mask_ema_count = update_mask_ema(
        hard_masks,
        mask_ema,
        mask_ema_count,
        beta=beta,
    )
    synthetic_masked = dict(apply_tree_mask(updates, synthetic_masks))
    pre_projection_update = dict(apply_tree_mask(updates, effective_masks))
    projected_updates = dict(pre_projection_update)
    synthetic_before_by_layer = []
    synthetic_after_by_layer = []
    anchor_before_by_layer = []
    anchor_after_by_layer = []
    selectors_by_category: dict[str, list[jax.Array]] = {
        name: [] for name in required_categories
    }
    removed_energy = jnp.asarray(0.0, dtype=jnp.float32)

    for layer_idx, layout in enumerate(layouts):
        layer_selectors = {
            name: value_memory_map[name][layer_idx]
            for name in required_categories
        }
        (
            projected,
            synthetic_before,
            synthetic_after,
            anchor_before,
            anchor_after,
            removed,
        ) = _project_ffn_values_against_joint_memory_map(
            pre_projection_update[layout.down_path],
            synthetic_group_grads,
            anchor_group_grads,
            layout.down_path,
            layout.down_slot_axis,
            synthetic_tail_count,
            anchor_tail_count,
            layer_selectors,
            projection_passes,
            synthetic_projection_strength,
            eps,
        )
        projected_updates[layout.down_path] = projected
        synthetic_before_by_layer.append(synthetic_before)
        synthetic_after_by_layer.append(synthetic_after)
        anchor_before_by_layer.append(anchor_before)
        anchor_after_by_layer.append(anchor_after)
        for name, selector in layer_selectors.items():
            selectors_by_category[name].append(
                jnp.asarray(selector, dtype=jnp.float32)
            )
        removed_energy += removed

    synthetic_before = jnp.concatenate(
        [jnp.ravel(value) for value in synthetic_before_by_layer]
    )
    synthetic_after = jnp.concatenate(
        [jnp.ravel(value) for value in synthetic_after_by_layer]
    )
    anchor_before = jnp.concatenate(
        [jnp.ravel(value) for value in anchor_before_by_layer]
    )
    anchor_after = jnp.concatenate(
        [jnp.ravel(value) for value in anchor_after_by_layer]
    )
    flat_selectors = {
        name: jnp.concatenate([jnp.ravel(value) for value in selectors])
        for name, selectors in selectors_by_category.items()
    }
    synthetic_selector = (
        flat_selectors["shared"] + flat_selectors["synthetic_only"]
    )
    anchor_selector = flat_selectors["shared"] + flat_selectors["real_only"]

    def selected_mean(value: jax.Array, selector: jax.Array) -> jax.Array:
        return jnp.sum(value * selector) / jnp.maximum(jnp.sum(selector), 1.0)

    proposal_norm = tree_l2_norm(pre_projection_update)
    final_norm = tree_l2_norm(projected_updates)
    shared_before = jnp.maximum(
        jax.nn.relu(synthetic_before),
        jax.nn.relu(anchor_before),
    )
    shared_after = jnp.maximum(
        jax.nn.relu(synthetic_after),
        jax.nn.relu(anchor_after),
    )
    stats = {
        "mask/synthetic_mean": tree_mean(synthetic_masks),
        "mask/anchor_element_mean": tree_mean(anchor_element_masks),
        "mask/final_mean": tree_mean(hard_masks),
        "mask/effective_ema_mean": tree_mean(effective_masks),
        "ffn_joint/shared_count": jnp.sum(flat_selectors["shared"]),
        "ffn_joint/real_only_count": jnp.sum(flat_selectors["real_only"]),
        "ffn_joint/synthetic_only_count": jnp.sum(
            flat_selectors["synthetic_only"]
        ),
        "ffn_joint/free_count": jnp.sum(flat_selectors["free"]),
        "ffn_joint/shared_fraction": jnp.mean(flat_selectors["shared"]),
        "ffn_joint/real_only_fraction": jnp.mean(flat_selectors["real_only"]),
        "ffn_joint/synthetic_only_fraction": jnp.mean(
            flat_selectors["synthetic_only"]
        ),
        "ffn_joint/free_fraction": jnp.mean(flat_selectors["free"]),
        "ffn_joint/synthetic_projection_strength": jnp.asarray(
            synthetic_projection_strength,
            dtype=jnp.float32,
        ),
        "ffn_joint/synthetic_violation_before_mean": selected_mean(
            jax.nn.relu(synthetic_before),
            synthetic_selector,
        ),
        "ffn_joint/synthetic_violation_after_mean": selected_mean(
            jax.nn.relu(synthetic_after),
            synthetic_selector,
        ),
        "ffn_joint/real_violation_before_mean": selected_mean(
            jax.nn.relu(anchor_before),
            anchor_selector,
        ),
        "ffn_joint/real_violation_after_mean": selected_mean(
            jax.nn.relu(anchor_after),
            anchor_selector,
        ),
        "ffn_joint/shared_joint_violation_before_mean": selected_mean(
            shared_before,
            flat_selectors["shared"],
        ),
        "ffn_joint/shared_joint_violation_after_mean": selected_mean(
            shared_after,
            flat_selectors["shared"],
        ),
        "ffn_joint/synthetic_only_violation_after_mean": selected_mean(
            jax.nn.relu(synthetic_after),
            flat_selectors["synthetic_only"],
        ),
        "ffn_joint/real_only_violation_after_mean": selected_mean(
            jax.nn.relu(anchor_after),
            flat_selectors["real_only"],
        ),
        "ffn_joint/projection_removed_norm_fraction": jnp.sqrt(
            jnp.maximum(removed_energy, 0.0)
        )
        / jnp.maximum(proposal_norm, 1e-30),
        "ffn_joint/synthetic_tail_count": jnp.asarray(
            synthetic_tail_count,
            dtype=jnp.float32,
        ),
        "ffn_joint/real_tail_count": jnp.asarray(
            anchor_tail_count,
            dtype=jnp.float32,
        ),
        "ffn_joint/synthetic_score_mean": tree_mean(synthetic_scores),
        "ffn_joint/real_element_score_mean": tree_mean(anchor_scores),
        "update_norm/raw": tree_l2_norm(updates),
        "update_norm/synthetic_masked": tree_l2_norm(synthetic_masked),
        "update_norm/final_structured": final_norm,
        "update_norm/kept_fraction": final_norm
        / jnp.maximum(tree_l2_norm(updates), 1e-30),
    }
    return projected_updates, mask_ema, mask_ema_count, stats


def mask_optax_updates_synthetic_ema(
    updates: dict[str, jax.Array],
    synthetic_grad: dict[str, jax.Array],
    mask_ema: dict[str, jax.Array],
    mask_ema_count: jax.Array,
    *,
    beta: float,
) -> tuple[dict[str, jax.Array], dict[str, jax.Array], jax.Array, dict[str, jax.Array]]:
    """Apply the mean synthetic proposal with EMA and no real-anchor veto."""
    hard_masks = dict(elementwise_safe_mask(synthetic_grad, updates))
    effective_masks, mask_ema, mask_ema_count = update_mask_ema(
        hard_masks,
        mask_ema,
        mask_ema_count,
        beta=beta,
    )
    # EMA history may soften accepted coordinates, but it cannot override the
    # current synthetic proposal and reactivate a currently rejected update.
    effective_masks = jax.tree_util.tree_map(
        lambda effective, hard: effective * hard,
        effective_masks,
        hard_masks,
    )
    masked_updates = apply_tree_mask(updates, effective_masks)
    raw_norm = tree_l2_norm(updates)
    final_norm = tree_l2_norm(masked_updates)
    stats = {
        "mask/synthetic_mean": tree_mean(hard_masks),
        "mask/final_mean": tree_mean(effective_masks),
        "mask/effective_ema_mean": tree_mean(effective_masks),
        "update_norm/raw": raw_norm,
        "update_norm/synthetic_masked": final_norm,
        "update_norm/final_structured": final_norm,
        "update_norm/kept_fraction": final_norm / jnp.maximum(raw_norm, 1e-30),
    }
    return masked_updates, mask_ema, mask_ema_count, stats


def mask_optax_updates_memory_ema(
    updates: dict[str, jax.Array],
    synthetic_grad: dict[str, jax.Array],
    anchor_grad: dict[str, jax.Array],
    layouts: tuple[FfnMemorySlotLayout, ...],
    mask_ema: dict[str, jax.Array],
    mask_ema_count: jax.Array,
    *,
    granularity: MemoryGranularity,
    beta: float,
    tau: float = 0.25,
    leak: float = 0.10,
    anchor_group_grads: tuple[dict[str, jax.Array], ...] = (),
    confidence_kappa: float = 1.0,
    old_occupancy: tuple[jax.Array, ...] | None = None,
    current_occupancy: tuple[jax.Array, ...] | None = None,
    collision_leak: float = 0.25,
    collision_beta: float = 4.0,
    collision_mode: str = "exp",
    collision_threshold: float = 0.25,
    collision_temperature: float = 0.1,
    occupancy_active_threshold: float = 0.75,
) -> tuple[dict[str, jax.Array], dict[str, jax.Array], jax.Array, dict[str, jax.Array]]:
    hard_masks, stats, auxiliary = compute_memory_hard_masks(
        updates,
        synthetic_grad,
        anchor_grad,
        layouts,
        granularity=granularity,
        tau=tau,
        leak=leak,
        anchor_group_grads=anchor_group_grads,
        confidence_kappa=confidence_kappa,
        old_occupancy=old_occupancy,
        current_occupancy=current_occupancy,
        collision_leak=collision_leak,
        collision_beta=collision_beta,
        collision_mode=collision_mode,
        collision_threshold=collision_threshold,
        collision_temperature=collision_temperature,
        occupancy_active_threshold=occupancy_active_threshold,
    )
    effective_masks, mask_ema, mask_ema_count = update_mask_ema(
        hard_masks,
        mask_ema,
        mask_ema_count,
        beta=beta,
    )
    effective_masks = jax.tree_util.tree_map(
        lambda effective, synthetic: effective * synthetic,
        effective_masks,
        auxiliary["synthetic_masks"],
    )
    masked_updates = apply_tree_mask(updates, effective_masks)
    stats.update(
        {
            "mask/effective_ema_mean": tree_mean(effective_masks),
            "update_norm/raw": tree_l2_norm(updates),
            "update_norm/synthetic_masked": tree_l2_norm(auxiliary["synthetic_update"]),
            "update_norm/final_structured": tree_l2_norm(masked_updates),
            "update_norm/kept_fraction": tree_l2_norm(masked_updates)
            / jnp.maximum(tree_l2_norm(updates), 1e-30),
        }
    )
    return masked_updates, mask_ema, mask_ema_count, stats


def mask_optax_updates_memory_instant(
    updates: dict[str, jax.Array],
    synthetic_grad: dict[str, jax.Array],
    anchor_grad: dict[str, jax.Array],
    layouts: tuple[FfnMemorySlotLayout, ...],
    *,
    granularity: MemoryGranularity,
    tau: float = 0.25,
    leak: float = 0.10,
    anchor_group_grads: tuple[dict[str, jax.Array], ...] = (),
    confidence_kappa: float = 1.0,
    old_occupancy: tuple[jax.Array, ...] | None = None,
    current_occupancy: tuple[jax.Array, ...] | None = None,
    collision_leak: float = 0.25,
    collision_beta: float = 4.0,
    collision_mode: str = "exp",
    collision_threshold: float = 0.25,
    collision_temperature: float = 0.1,
    occupancy_active_threshold: float = 0.75,
) -> tuple[dict[str, jax.Array], dict[str, jax.Array]]:
    """Apply the current synthetic proposal and real veto without mask history."""
    if granularity == "element":
        synthetic_masks = dict(elementwise_safe_mask(synthetic_grad, updates))
        synthetic_update = dict(apply_tree_mask(updates, synthetic_masks))
        anchor_masks = dict(elementwise_safe_mask(anchor_grad, synthetic_update))
        hard_masks = {
            path: synthetic_masks[path] * anchor_masks[path]
            for path in updates
        }
        stats = {
            "mask/synthetic_mean": tree_mean(synthetic_masks),
            "mask/anchor_element_mean": tree_mean(anchor_masks),
            "mask/final_mean": tree_mean(hard_masks),
        }
        auxiliary = {"synthetic_update": synthetic_update}
    else:
        hard_masks, stats, auxiliary = compute_memory_hard_masks(
            updates,
            synthetic_grad,
            anchor_grad,
            layouts,
            granularity=granularity,
            tau=tau,
            leak=leak,
            anchor_group_grads=anchor_group_grads,
            confidence_kappa=confidence_kappa,
            old_occupancy=old_occupancy,
            current_occupancy=current_occupancy,
            collision_leak=collision_leak,
            collision_beta=collision_beta,
            collision_mode=collision_mode,
            collision_threshold=collision_threshold,
            collision_temperature=collision_temperature,
            occupancy_active_threshold=occupancy_active_threshold,
        )
    masked_updates = apply_tree_mask(updates, hard_masks)
    raw_norm = tree_l2_norm(updates)
    final_norm = tree_l2_norm(masked_updates)
    stats.update(
        {
            "mask/instantaneous_final_mean": tree_mean(hard_masks),
            "update_norm/raw": raw_norm,
            "update_norm/synthetic_masked": tree_l2_norm(
                auxiliary["synthetic_update"]
            ),
            "update_norm/final_structured": final_norm,
            "update_norm/kept_fraction": final_norm
            / jnp.maximum(raw_norm, 1e-30),
        }
    )
    return masked_updates, stats


def mask_optax_updates_normmatched_control(
    updates: dict[str, jax.Array],
    synthetic_grad: dict[str, jax.Array],
    anchor_grad: dict[str, jax.Array],
    layouts: tuple[FfnMemorySlotLayout, ...],
    element_ema: dict[str, jax.Array],
    structured_ema: dict[str, jax.Array],
    mask_ema_count: jax.Array,
    *,
    target_granularity: MemoryGranularity,
    beta: float,
    tau: float,
    leak: float,
) -> tuple[
    dict[str, jax.Array],
    dict[str, jax.Array],
    dict[str, jax.Array],
    jax.Array,
    dict[str, jax.Array],
]:
    element_hard, element_stats, element_auxiliary = compute_memory_hard_masks(
        updates,
        synthetic_grad,
        anchor_grad,
        layouts,
        granularity="element",
        tau=tau,
        leak=leak,
    )
    structured_hard, _, structured_auxiliary = compute_memory_hard_masks(
        updates,
        synthetic_grad,
        anchor_grad,
        layouts,
        granularity=target_granularity,
        tau=tau,
        leak=leak,
    )
    element_effective, element_ema, next_count = update_mask_ema(
        element_hard, element_ema, mask_ema_count, beta=beta
    )
    structured_effective, structured_ema, structured_count = update_mask_ema(
        structured_hard, structured_ema, mask_ema_count, beta=beta
    )
    del structured_count
    element_effective = jax.tree_util.tree_map(
        lambda effective, synthetic: effective * synthetic,
        element_effective,
        element_auxiliary["synthetic_masks"],
    )
    structured_effective = jax.tree_util.tree_map(
        lambda effective, synthetic: effective * synthetic,
        structured_effective,
        structured_auxiliary["synthetic_masks"],
    )
    element_update = apply_tree_mask(updates, element_effective)
    structured_update = apply_tree_mask(updates, structured_effective)
    element_norm = tree_l2_norm(element_update)
    structured_norm = tree_l2_norm(structured_update)
    scale = jnp.where(element_norm > 1e-30, structured_norm / element_norm, 0.0)
    matched_update = jax.tree_util.tree_map(lambda update: update * scale.astype(update.dtype), element_update)
    element_stats.update(
        {
            "mask/effective_ema_mean": tree_mean(element_effective),
            "update_norm/raw": tree_l2_norm(updates),
            "update_norm/synthetic_masked": tree_l2_norm(
                element_auxiliary["synthetic_update"]
            ),
            "update_norm/final_structured": structured_norm,
            "update_norm/normmatched_control": tree_l2_norm(matched_update),
            "update_norm/kept_fraction": tree_l2_norm(matched_update)
            / jnp.maximum(tree_l2_norm(updates), 1e-30),
            "control/norm_scale": scale,
        }
    )
    return matched_update, element_ema, structured_ema, next_count, element_stats


def estimate_right_activation_subspace(
    coefficients: np.ndarray,
    rank: int,
    *,
    oversample: int = 8,
    power_iterations: int = 0,
    seed: int = 0,
) -> tuple[np.ndarray, dict[str, float]]:
    """Estimate the leading signed FFN-address subspace with randomized SVD."""
    matrix = np.asarray(coefficients, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError(f"coefficients must have shape [probe, feature], got {matrix.shape}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("coefficients contain non-finite values")
    if rank <= 0:
        raise ValueError("subspace rank must be positive")
    if oversample < 0 or power_iterations < 0:
        raise ValueError("oversample and power_iterations must be nonnegative")

    n_rows, n_features = matrix.shape
    available_rank = min(n_rows, n_features)
    if rank > available_rank:
        raise ValueError(
            f"subspace rank {rank} exceeds available rank {available_rank} "
            f"for coefficient matrix {matrix.shape}"
        )
    sketch_rank = min(available_rank, rank + oversample)
    rng = np.random.default_rng(seed)
    omega = rng.standard_normal((n_features, sketch_rank), dtype=np.float32)
    sketch = matrix @ omega
    for _ in range(power_iterations):
        sketch, _ = np.linalg.qr(sketch, mode="reduced")
        sketch = matrix @ (matrix.T @ sketch)
    q, _ = np.linalg.qr(sketch, mode="reduced")
    compressed = q.T @ matrix
    _, singular_values, right_vectors_t = np.linalg.svd(
        compressed,
        full_matrices=False,
    )
    basis = right_vectors_t[:rank].T.astype(np.float32, copy=False)

    # Canonical signs make cached artifacts deterministic across LAPACK builds.
    max_indices = np.argmax(np.abs(basis), axis=0)
    signs = np.sign(basis[max_indices, np.arange(rank)])
    basis *= np.where(signs == 0, 1.0, signs)[None, :]
    projected = matrix @ basis
    total_energy = float(np.sum(np.square(matrix), dtype=np.float64))
    captured_energy = float(np.sum(np.square(projected), dtype=np.float64))
    stats = {
        "rank": float(rank),
        "probe_count": float(n_rows),
        "feature_count": float(n_features),
        "captured_energy_fraction": captured_energy / max(total_energy, 1e-30),
        "leading_singular_value": float(singular_values[0]) if singular_values.size else 0.0,
        "retained_singular_value": float(singular_values[rank - 1]),
    }
    return basis, stats


def save_memory_subspace_bases(
    path: str | Path,
    bases: np.ndarray,
    metadata: dict[str, Any],
) -> None:
    """Persist a nested-rank basis cache and a human-readable audit artifact."""
    path = Path(path)
    if path.suffix != ".npz":
        raise ValueError("memory subspace basis path must end in .npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, bases=np.asarray(bases, dtype=np.float32))
    metadata_path = path.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")


def load_memory_subspace_bases(
    path: str | Path,
    layouts: tuple[FfnMemorySlotLayout, ...],
    rank: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Load and validate a cached basis, slicing a nested prefix if requested."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as payload:
        bases = np.asarray(payload["bases"], dtype=np.float32)
    expected_features = tuple(layout.slot_count for layout in layouts)
    if bases.ndim != 3 or bases.shape[0] != len(layouts):
        raise ValueError(
            f"cached bases must have shape [layer, feature, rank], got {bases.shape}"
        )
    if any(bases.shape[1] != feature_count for feature_count in expected_features):
        raise ValueError(
            f"cached feature count {bases.shape[1]} does not match layouts {expected_features}"
        )
    if rank <= 0 or rank > bases.shape[2]:
        raise ValueError(
            f"requested rank {rank} is unavailable in cached rank {bases.shape[2]}"
        )
    gram = np.einsum("lfr,lfs->lrs", bases[:, :, :rank], bases[:, :, :rank])
    identity = np.eye(rank, dtype=np.float32)[None, :, :]
    if not np.allclose(gram, identity, rtol=1e-4, atol=1e-4):
        raise ValueError("cached memory subspace bases are not orthonormal")
    metadata_path = path.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    return bases[:, :, :rank], metadata


def shard_memory_subspace_bases(
    bases: np.ndarray,
    layouts: tuple[FfnMemorySlotLayout, ...],
    weights: dict[str, jax.Array],
) -> tuple[jax.Array, ...]:
    """Shard each [FFN, rank] basis like the down-projection FFN axis."""
    if bases.shape[0] != len(layouts):
        raise ValueError("basis layer count does not match FFN layout")
    sharded = []
    for layer_idx, layout in enumerate(layouts):
        basis = jnp.asarray(bases[layer_idx], dtype=jnp.float32)
        down_sharding = getattr(weights[layout.down_path], "sharding", None)
        if isinstance(down_sharding, NamedSharding):
            basis = jax.device_put(
                basis,
                NamedSharding(down_sharding.mesh, P("model", None)),
            )
        sharded.append(basis)
    return tuple(sharded)


def right_subspace_components(
    matrix: jax.Array,
    basis: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return low-rank coordinates, in-subspace matrix, and orthogonal remainder."""
    coordinates = _right_subspace_coordinates(matrix, basis)
    in_subspace = _reconstruct_right_subspace(coordinates, basis, matrix)
    return coordinates, in_subspace, jnp.asarray(matrix, dtype=jnp.float32) - in_subspace


def _named_array_sharding(value: jax.Array) -> NamedSharding | None:
    sharding = getattr(value, "sharding", None)
    if sharding is None:
        sharding = getattr(getattr(value, "aval", None), "sharding", None)
    return sharding if isinstance(sharding, NamedSharding) else None


def _right_subspace_coordinates(
    matrix: jax.Array,
    basis: jax.Array,
) -> jax.Array:
    matrix = jnp.asarray(matrix, dtype=jnp.float32)
    basis = jnp.asarray(basis, dtype=jnp.float32)
    matrix_sharding = _named_array_sharding(matrix)
    if matrix_sharding is None:
        return matrix @ basis
    leading_spec = (
        matrix_sharding.spec[0]
        if len(matrix_sharding.spec) > 0
        else None
    )
    return jnp.einsum(
        "df,fr->dr",
        matrix,
        basis,
        out_sharding=NamedSharding(matrix_sharding.mesh, P(leading_spec, None)),
    )


def _reconstruct_right_subspace(
    coordinates: jax.Array,
    basis: jax.Array,
    target: jax.Array,
) -> jax.Array:
    coordinates = jnp.asarray(coordinates, dtype=jnp.float32)
    basis = jnp.asarray(basis, dtype=jnp.float32)
    target_sharding = _named_array_sharding(target)
    if target_sharding is None:
        return coordinates @ basis.T
    return jnp.einsum(
        "dr,fr->df",
        coordinates,
        basis,
        out_sharding=target_sharding,
    )


def project_reference_groups_to_memory_subspaces(
    reference_group_grads: tuple[dict[str, jax.Array], ...],
    layouts: tuple[FfnMemorySlotLayout, ...],
    bases: tuple[jax.Array, ...],
) -> tuple[tuple[jax.Array, ...], ...]:
    """Cache fixed epoch reference gradients in low-rank down coordinates."""
    if not reference_group_grads:
        raise ValueError("at least one reference group is required")
    if len(layouts) != len(bases):
        raise ValueError("one memory subspace basis is required per FFN layer")
    return tuple(
        tuple(
            _right_subspace_coordinates(group[layout.down_path], basis)
            for group in reference_group_grads
        )
        for layout, basis in zip(layouts, bases)
    )


def _matrix_inner(left: jax.Array, right: jax.Array) -> jax.Array:
    return jnp.sum(
        jnp.asarray(left, dtype=jnp.float32)
        * jnp.asarray(right, dtype=jnp.float32)
    )


def _cvar_tail_gradient(
    update: jax.Array,
    group_grads: tuple[jax.Array, ...],
    tail_count: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    scores = jnp.stack([_matrix_inner(grad, update) for grad in group_grads])
    tail_scores, tail_indices = jax.lax.top_k(scores, tail_count)
    stacked_grads = jnp.stack(group_grads, axis=0)
    tail_grad = jnp.mean(stacked_grads[tail_indices], axis=0)
    return tail_grad, jnp.mean(tail_scores), jnp.max(scores)


def _project_one_cvar_halfspace(
    update: jax.Array,
    group_grads: tuple[jax.Array, ...],
    tail_count: int,
    eps: float,
) -> jax.Array:
    tail_grad, violation, _ = _cvar_tail_gradient(update, group_grads, tail_count)
    scale = jax.nn.relu(violation) / jnp.maximum(
        _matrix_inner(tail_grad, tail_grad),
        eps,
    )
    return update - scale.astype(update.dtype) * tail_grad.astype(update.dtype)


def project_optax_updates_memory_subspace_cvar(
    updates: dict[str, jax.Array],
    elementwise_fallback_updates: dict[str, jax.Array],
    synthetic_subspace_group_grads: tuple[tuple[jax.Array, ...], ...],
    anchor_subspace_group_grads: tuple[tuple[jax.Array, ...], ...],
    layouts: tuple[FfnMemorySlotLayout, ...],
    bases: tuple[jax.Array, ...],
    *,
    tail_fraction: float,
    projection_passes: int = 4,
    eps: float = 1e-12,
) -> tuple[dict[str, jax.Array], dict[str, jax.Array]]:
    """Protect down-projection address subspaces with synthetic and real CVaR.

    Non-down parameters retain the combined elementwise CVaR+EMA fallback. For
    each down projection, the raw update outside the frozen synthetic address
    subspace is preserved, while only harmful in-subspace components are
    removed by minimum-norm half-space projection.
    """
    if not synthetic_subspace_group_grads or not anchor_subspace_group_grads:
        raise ValueError("memory subspace CVaR requires synthetic and anchor groups")
    if len(layouts) != len(bases):
        raise ValueError("one memory subspace basis is required per FFN layer")
    if len(synthetic_subspace_group_grads) != len(layouts):
        raise ValueError("synthetic subspace gradients must match the layer count")
    if len(anchor_subspace_group_grads) != len(layouts):
        raise ValueError("anchor subspace gradients must match the layer count")
    if not 0 < tail_fraction <= 1:
        raise ValueError("tail_fraction must be in (0, 1]")
    if projection_passes <= 0 or eps <= 0:
        raise ValueError("projection_passes and eps must be positive")

    synthetic_group_count = len(synthetic_subspace_group_grads[0])
    anchor_group_count = len(anchor_subspace_group_grads[0])
    if synthetic_group_count <= 0 or anchor_group_count <= 0:
        raise ValueError("each memory subspace layer requires reference groups")
    if any(len(groups) != synthetic_group_count for groups in synthetic_subspace_group_grads):
        raise ValueError("synthetic subspace group counts must match across layers")
    if any(len(groups) != anchor_group_count for groups in anchor_subspace_group_grads):
        raise ValueError("anchor subspace group counts must match across layers")
    synthetic_tail_count = max(
        1,
        math.ceil(synthetic_group_count * tail_fraction),
    )
    anchor_tail_count = max(1, math.ceil(anchor_group_count * tail_fraction))
    final_updates = dict(elementwise_fallback_updates)
    raw_down_energy = jnp.asarray(0.0, dtype=jnp.float32)
    old_energy = jnp.asarray(0.0, dtype=jnp.float32)
    free_energy = jnp.asarray(0.0, dtype=jnp.float32)
    removed_energy = jnp.asarray(0.0, dtype=jnp.float32)
    synthetic_before = []
    synthetic_after = []
    anchor_before = []
    anchor_after = []
    synthetic_max_after = []
    anchor_max_after = []

    for layer_idx, (layout, basis) in enumerate(zip(layouts, bases)):
        raw_update = updates[layout.down_path]
        coordinates, _, free_update = right_subspace_components(
            raw_update,
            basis,
        )
        synthetic_coordinates = synthetic_subspace_group_grads[layer_idx]
        anchor_coordinates = anchor_subspace_group_grads[layer_idx]
        _, synth_score_before, _ = _cvar_tail_gradient(
            coordinates,
            synthetic_coordinates,
            synthetic_tail_count,
        )
        _, anchor_score_before, _ = _cvar_tail_gradient(
            coordinates,
            anchor_coordinates,
            anchor_tail_count,
        )

        safe_coordinates = coordinates
        for _ in range(projection_passes):
            safe_coordinates = _project_one_cvar_halfspace(
                safe_coordinates,
                synthetic_coordinates,
                synthetic_tail_count,
                eps,
            )
            safe_coordinates = _project_one_cvar_halfspace(
                safe_coordinates,
                anchor_coordinates,
                anchor_tail_count,
                eps,
            )
        _, synth_score_after, synth_max = _cvar_tail_gradient(
            safe_coordinates,
            synthetic_coordinates,
            synthetic_tail_count,
        )
        _, anchor_score_after, anchor_max = _cvar_tail_gradient(
            safe_coordinates,
            anchor_coordinates,
            anchor_tail_count,
        )
        safe_old_update = _reconstruct_right_subspace(
            safe_coordinates,
            basis,
            raw_update,
        )
        final_updates[layout.down_path] = (
            free_update + safe_old_update
        ).astype(raw_update.dtype)

        raw_down_energy += _matrix_inner(raw_update, raw_update)
        old_energy += _matrix_inner(coordinates, coordinates)
        free_energy += _matrix_inner(free_update, free_update)
        removed = coordinates - safe_coordinates
        removed_energy += _matrix_inner(removed, removed)
        synthetic_before.append(synth_score_before)
        synthetic_after.append(synth_score_after)
        anchor_before.append(anchor_score_before)
        anchor_after.append(anchor_score_after)
        synthetic_max_after.append(synth_max)
        anchor_max_after.append(anchor_max)

    old_norm = jnp.sqrt(jnp.maximum(old_energy, 0.0))
    stats = {
        "subspace/current_update_energy_fraction_in_old_space": old_energy
        / jnp.maximum(raw_down_energy, 1e-30),
        "subspace/free_update_energy_fraction": free_energy
        / jnp.maximum(raw_down_energy, 1e-30),
        "subspace/harmful_component_removed_fraction": jnp.sqrt(
            jnp.maximum(removed_energy, 0.0)
        )
        / jnp.maximum(old_norm, 1e-30),
        "subspace/synthetic_cvar_before": jnp.mean(jnp.stack(synthetic_before)),
        "subspace/synthetic_cvar_after": jnp.mean(jnp.stack(synthetic_after)),
        "subspace/anchor_cvar_before": jnp.mean(jnp.stack(anchor_before)),
        "subspace/anchor_cvar_after": jnp.mean(jnp.stack(anchor_after)),
        "subspace/synthetic_max_group_change_after": jnp.max(
            jnp.stack(synthetic_max_after)
        ),
        "subspace/anchor_max_group_change_after": jnp.max(
            jnp.stack(anchor_max_after)
        ),
        "subspace/synthetic_tail_count": jnp.asarray(
            synthetic_tail_count,
            dtype=jnp.float32,
        ),
        "subspace/anchor_tail_count": jnp.asarray(
            anchor_tail_count,
            dtype=jnp.float32,
        ),
        "subspace/down_update_norm_before": jnp.sqrt(
            jnp.maximum(raw_down_energy, 0.0)
        ),
        "subspace/down_update_norm_after": jnp.sqrt(
            sum(
                (
                    _matrix_inner(final_updates[layout.down_path], final_updates[layout.down_path])
                    for layout in layouts
                ),
                jnp.asarray(0.0, dtype=jnp.float32),
            )
        ),
        "subspace/full_update_norm_after": tree_l2_norm(final_updates),
    }
    return final_updates, stats


def percentile_normalize(activity: jax.Array) -> jax.Array:
    """Convert each layer's slot activity to tie-aware percentile ranks."""
    activity = jnp.asarray(activity, dtype=jnp.float32)
    if activity.ndim != 2:
        raise ValueError(f"memory activity must have shape [layers, slots], got {activity.shape}")

    # JAX sort requires its sorting dimension to be local. Qwen tensor-parallel
    # activity is P(None, "model"), so replicate this small [layers, slots]
    # diagnostic while ranking and restore its original sharding afterward.
    original_sharding = getattr(activity, "sharding", None)
    restore_sharding = None
    if isinstance(original_sharding, NamedSharding):
        sort_spec = list(original_sharding.spec)
        sort_spec.extend([None] * (activity.ndim - len(sort_spec)))
        if sort_spec[-1] is not None:
            sort_spec[-1] = None
            activity = jax.device_put(
                activity,
                NamedSharding(original_sharding.mesh, P(*sort_spec)),
            )
            restore_sharding = original_sharding

    order = jnp.argsort(activity, axis=-1, stable=True)
    sorted_activity = jnp.take_along_axis(activity, order, axis=-1)
    slot_count = activity.shape[-1]
    indices = jnp.arange(slot_count, dtype=jnp.int32)[None, :]
    changed = sorted_activity[:, 1:] != sorted_activity[:, :-1]
    group_starts = jnp.concatenate(
        [jnp.ones((activity.shape[0], 1), dtype=bool), changed],
        axis=-1,
    )
    group_ends = jnp.concatenate(
        [changed, jnp.ones((activity.shape[0], 1), dtype=bool)],
        axis=-1,
    )
    start_indices = jnp.where(group_starts, indices, 0)
    end_indices = jnp.where(group_ends, indices, slot_count - 1)
    starts = jax.lax.associative_scan(jnp.maximum, start_indices, axis=-1)
    ends = jnp.flip(
        jax.lax.associative_scan(jnp.minimum, jnp.flip(end_indices, axis=-1), axis=-1),
        axis=-1,
    )
    sorted_ranks = 0.5 * (starts.astype(jnp.float32) + ends.astype(jnp.float32))
    inverse_order = jnp.argsort(order, axis=-1, stable=True)
    ranks = jnp.take_along_axis(sorted_ranks, inverse_order, axis=-1)
    percentiles = ranks / jnp.maximum(activity.shape[-1] - 1, 1)
    if restore_sharding is not None:
        percentiles = jax.device_put(percentiles, restore_sharding)
    return percentiles


def normalized_memory_activity(activity: jax.Array, eps: float = 1e-12) -> jax.Array:
    activity = jnp.asarray(activity, dtype=jnp.float32)
    return activity / jnp.maximum(activity.sum(axis=-1, keepdims=True), eps)


def robust_old_occupancy(synthetic_activity: jax.Array, anchor_activity: jax.Array) -> jax.Array:
    synthetic = percentile_normalize(normalized_memory_activity(synthetic_activity))
    anchor = percentile_normalize(normalized_memory_activity(anchor_activity))
    return jax.lax.stop_gradient(jnp.maximum(synthetic, anchor))


def blended_old_occupancy(
    synthetic_activity: jax.Array,
    anchor_activity: jax.Array,
    anchor_weight: float,
) -> jax.Array:
    """Rank a calibrated blend instead of taking two independently ranked maxima."""
    if not 0 <= anchor_weight <= 1:
        raise ValueError("anchor occupancy weight must be in [0, 1]")
    synthetic = normalized_memory_activity(synthetic_activity)
    anchor = normalized_memory_activity(anchor_activity)
    combined = (1.0 - anchor_weight) * synthetic + anchor_weight * anchor
    return jax.lax.stop_gradient(percentile_normalize(combined))


def free_memory_distribution(
    old_occupancy: jax.Array,
    gamma: float,
    eps: float = 1e-12,
    *,
    floor: float = 0.0,
) -> jax.Array:
    if gamma < 0:
        raise ValueError("free-memory gamma must be nonnegative")
    if floor < 0:
        raise ValueError("free-memory floor must be nonnegative")
    free = floor + jnp.power(jnp.maximum(1.0 - old_occupancy, 0.0), gamma)
    return jax.lax.stop_gradient(free / jnp.maximum(free.sum(axis=-1, keepdims=True), eps))


def memory_occupancy_summary(
    old_occupancy: jax.Array,
    current_occupancy: jax.Array | None = None,
    *,
    active_threshold: float = 0.75,
    eps: float = 1e-12,
) -> dict[str, jax.Array]:
    old_prob = normalized_memory_activity(old_occupancy, eps)
    old_entropy = -(old_prob * jnp.log(jnp.maximum(old_prob, eps))).sum(axis=-1)
    stats = {
        "memory/old_entropy_mean": old_entropy.mean(),
        "memory/old_active_slot_fraction": (old_occupancy >= active_threshold).mean(),
        "memory/free_slot_fraction": (old_occupancy < 1.0 - active_threshold).mean(),
    }
    if current_occupancy is None:
        stats.update(
            {
                "memory/current_entropy_mean": jnp.asarray(0.0, dtype=jnp.float32),
                "memory/current_active_slot_fraction": jnp.asarray(0.0, dtype=jnp.float32),
                "memory/old_current_overlap_mean": jnp.asarray(0.0, dtype=jnp.float32),
                "memory/contested_slot_fraction": jnp.asarray(0.0, dtype=jnp.float32),
            }
        )
        return stats
    current_prob = normalized_memory_activity(current_occupancy, eps)
    current_entropy = -(current_prob * jnp.log(jnp.maximum(current_prob, eps))).sum(axis=-1)
    contested = (old_occupancy >= active_threshold) & (current_occupancy >= active_threshold)
    stats.update(
        {
            "memory/current_entropy_mean": current_entropy.mean(),
            "memory/current_active_slot_fraction": (current_occupancy >= active_threshold).mean(),
            "memory/old_current_overlap_mean": jnp.minimum(old_occupancy, current_occupancy).mean(),
            "memory/contested_slot_fraction": contested.mean(),
        }
    )
    return stats


def layout_as_dict(layouts: tuple[FfnMemorySlotLayout, ...]) -> list[dict[str, Any]]:
    return [asdict(layout) for layout in layouts]
