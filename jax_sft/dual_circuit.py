"""Prompt-consistent MLP/attention circuit helpers for JAX CNL.

The circuit is estimated once from a frozen model. During training it only
localizes where real-anchor gradients may constrain the existing synthetic
CVaR proposal; it never replaces or expands that proposal's support.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec as P

from jax_sft.cnl import compute_cvar_masks, tree_l2_norm


PyTree = Any

TIER_CORE = 0
TIER_SYNTHETIC_ONLY = 1
TIER_REAL_RESIDUAL = 2
TIER_UNSUPPORTED = 3
TIER_NAMES = ("core", "synthetic_only", "real_residual", "unsupported")


@dataclass(frozen=True)
class DualCircuitLayerLayout:
    """Qwen projection paths and component axes for one transformer layer."""

    layer: int
    gate_path: str
    up_path: str
    down_path: str
    q_path: str
    k_path: str
    v_path: str
    o_path: str
    hidden_size: int
    intermediate_size: int
    query_head_count: int
    kv_head_count: int
    head_dim: int
    gate_component_axis: int = 0
    up_component_axis: int = 0
    down_component_axis: int = 1
    q_head_axis: int = 0
    k_head_axis: int = 0
    v_head_axis: int = 0
    o_head_axis: int = 1
    query_to_kv: tuple[int, ...] = ()


_CIRCUIT_PATH = re.compile(
    r"^model\.layers\.(?P<layer>\d+)\."
    r"(?P<family>mlp|self_attn)\."
    r"(?P<projection>gate_proj|up_proj|down_proj|q_proj|k_proj|v_proj|o_proj)\.weight$"
)


def audit_dual_circuit_parameter_layout(
    weights: dict[str, Any],
) -> tuple[tuple[DualCircuitLayerLayout, ...], dict[str, Any]]:
    """Infer and validate MLP plus GQA component slices from actual weights."""
    by_layer: dict[int, dict[str, str]] = {}
    for path in weights:
        match = _CIRCUIT_PATH.match(path)
        if match is not None:
            by_layer.setdefault(int(match.group("layer")), {})[
                match.group("projection")
            ] = path
    if not by_layer:
        raise ValueError("No Qwen MLP/attention projections were found")
    expected_layers = list(range(max(by_layer) + 1))
    if sorted(by_layer) != expected_layers:
        raise ValueError(
            f"circuit layers are not contiguous: found={sorted(by_layer)} "
            f"expected={expected_layers}"
        )

    layouts = []
    layer_artifacts = []
    supported_paths = set()
    required = {
        "gate_proj",
        "up_proj",
        "down_proj",
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    }
    for layer in expected_layers:
        paths = by_layer[layer]
        missing = required - set(paths)
        if missing:
            raise ValueError(
                f"layer {layer} is missing circuit projections: {sorted(missing)}"
            )
        shapes = {
            name: tuple(int(dim) for dim in weights[path].shape)
            for name, path in paths.items()
        }
        gate_shape = shapes["gate_proj"]
        up_shape = shapes["up_proj"]
        down_shape = shapes["down_proj"]
        q_shape = shapes["q_proj"]
        k_shape = shapes["k_proj"]
        v_shape = shapes["v_proj"]
        o_shape = shapes["o_proj"]
        if gate_shape != up_shape or len(gate_shape) != 2:
            raise ValueError(
                f"layer {layer} gate/up layout mismatch: {gate_shape} vs {up_shape}"
            )
        intermediate_size, hidden_size = gate_shape
        if down_shape != (hidden_size, intermediate_size):
            raise ValueError(
                f"layer {layer} down projection {down_shape} does not match "
                f"[D,F]=({hidden_size},{intermediate_size})"
            )
        if len(q_shape) != 3 or len(k_shape) != 3 or len(v_shape) != 3:
            raise ValueError(f"layer {layer} Q/K/V projections must be rank three")
        query_heads, head_dim, q_hidden = q_shape
        kv_heads, k_head_dim, k_hidden = k_shape
        if v_shape != k_shape:
            raise ValueError(f"layer {layer} K/V shapes differ: {k_shape} vs {v_shape}")
        if q_hidden != hidden_size or k_hidden != hidden_size or k_head_dim != head_dim:
            raise ValueError(f"layer {layer} attention dimensions do not match hidden size")
        if o_shape != (hidden_size, query_heads, head_dim):
            raise ValueError(
                f"layer {layer} O projection {o_shape} does not match "
                f"[D,N,H]=({hidden_size},{query_heads},{head_dim})"
            )
        if query_heads % kv_heads:
            raise ValueError(
                f"layer {layer} query heads {query_heads} are not divisible by "
                f"KV heads {kv_heads}"
            )
        group_size = query_heads // kv_heads
        query_to_kv = tuple(index // group_size for index in range(query_heads))
        layout = DualCircuitLayerLayout(
            layer=layer,
            gate_path=paths["gate_proj"],
            up_path=paths["up_proj"],
            down_path=paths["down_proj"],
            q_path=paths["q_proj"],
            k_path=paths["k_proj"],
            v_path=paths["v_proj"],
            o_path=paths["o_proj"],
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            query_head_count=query_heads,
            kv_head_count=kv_heads,
            head_dim=head_dim,
            query_to_kv=query_to_kv,
        )
        layouts.append(layout)
        supported_paths.update(paths.values())
        layer_artifacts.append(
            {
                **asdict(layout),
                "query_to_kv": list(query_to_kv),
                "component_slices": {
                    "mlp": {
                        "gate": "gate_proj.weight[k, :]",
                        "up": "up_proj.weight[k, :]",
                        "down": "down_proj.weight[:, k]",
                    },
                    "attention": {
                        "q": "q_proj.weight[h, :, :]",
                        "k": "k_proj.weight[kv(h), :, :] (shared by GQA group)",
                        "v": "v_proj.weight[kv(h), :, :] (shared by GQA group)",
                        "o": "o_proj.weight[:, h, :]",
                    },
                },
                "forward_equations": {
                    "mlp": "coeff=silu(x@gate.T)*(x@up.T); out=coeff@down.T",
                    "attention_o": "b t n h, d n h -> b t d",
                },
            }
        )

    unsupported = sorted(path for path in weights if path not in supported_paths)
    artifact = {
        "method": "Qwen prompt-consistent dual-circuit parameter layout audit",
        "summary": {
            "layout_audit_passed": True,
            "layer_count": len(layouts),
            "mlp_component_count": sum(x.intermediate_size for x in layouts),
            "attention_component_count": sum(x.query_head_count for x in layouts),
            "supported_parameter_count": len(supported_paths),
            "unsupported_parameter_count": len(unsupported),
            "unsupported_parameters": unsupported,
            "gqa_shared_kv_projection_once": True,
        },
        "layers": layer_artifacts,
    }
    return tuple(layouts), artifact


def write_dual_circuit_parameter_layout(
    path: str | Path,
    artifact: dict[str, Any],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")


def answer_margin(candidate_logits: jax.Array, target_index: jax.Array) -> jax.Array:
    """Return target logit minus the strongest non-target candidate logit."""
    candidate_logits = jnp.asarray(candidate_logits, dtype=jnp.float32)
    target_index = jnp.asarray(target_index, dtype=jnp.int32).reshape((-1,))
    target_mask = jax.nn.one_hot(
        target_index,
        candidate_logits.shape[-1],
        dtype=bool,
    )
    target = jnp.sum(
        candidate_logits
        * target_mask.astype(candidate_logits.dtype),
        axis=-1,
    )
    strongest_other = jnp.max(
        jnp.where(target_mask, -jnp.inf, candidate_logits),
        axis=-1,
    )
    return target - strongest_other


def normalize_view_attribution(
    attribution: np.ndarray,
    *,
    sign_mode: str,
    eps: float = 1e-12,
) -> np.ndarray:
    """Normalize each layer while preserving the requested attribution sign rule."""
    values = np.asarray(attribution, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("component attribution must have shape [layer, component]")
    if not np.all(np.isfinite(values)):
        raise ValueError("component attribution contains non-finite values")
    if sign_mode == "absolute":
        values = np.abs(values)
    elif sign_mode in {"signed", "positive_signed"}:
        # Only components that positively support the answer margin belong to
        # a retained circuit. Negative signed values are still available in
        # raw artifacts, but are not promoted to protected components.
        values = np.maximum(values, 0.0)
    else:
        raise ValueError(f"unknown circuit attribution sign mode: {sign_mode}")
    layer_scale = np.max(values, axis=-1, keepdims=True)
    return np.divide(
        values,
        np.maximum(layer_scale, eps),
        out=np.zeros_like(values),
        where=layer_scale > eps,
    )


def top_fraction_mask(values: np.ndarray, fraction: float) -> np.ndarray:
    """Select an exact top fraction independently in each layer."""
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("top-fraction values must have shape [layer, component]")
    if not 0 < fraction <= 1:
        raise ValueError("component top fraction must be in (0, 1]")
    count = max(1, math.ceil(values.shape[-1] * fraction))
    selected = np.zeros_like(values, dtype=bool)
    for layer in range(values.shape[0]):
        if not np.any(values[layer] > 0):
            continue
        order = np.argsort(-values[layer], kind="stable")
        positive = order[values[layer, order] > 0]
        selected[layer, positive[:count]] = True
    return selected


def refine_prompt_consistent_attribution(
    view_attributions: Iterable[np.ndarray],
    *,
    top_fraction: float,
    consistency_threshold: float,
    sign_mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return soft support, consistency, and stable mask for one example."""
    if not 0 < consistency_threshold <= 1:
        raise ValueError("prompt consistency threshold must be in (0, 1]")
    normalized = [
        normalize_view_attribution(value, sign_mode=sign_mode)
        for value in view_attributions
    ]
    if not normalized:
        raise ValueError("prompt consistency requires at least one view")
    if any(value.shape != normalized[0].shape for value in normalized[1:]):
        raise ValueError("prompt-view attribution shapes differ")
    top_masks = np.stack(
        [top_fraction_mask(value, top_fraction) for value in normalized],
        axis=0,
    )
    consistency = top_masks.mean(axis=0, dtype=np.float32)
    stable = consistency >= consistency_threshold
    magnitude = np.stack(normalized, axis=0).mean(axis=0, dtype=np.float32)
    soft_support = stable.astype(np.float32) * consistency * magnitude
    return soft_support, consistency, stable


class PromptCircuitAccumulator:
    """Streaming cross-example aggregation for one component family."""

    def __init__(
        self,
        component_shape: tuple[int, int],
        *,
        top_fraction: float,
        consistency_threshold: float,
        sign_mode: str,
    ) -> None:
        if len(component_shape) != 2 or min(component_shape) <= 0:
            raise ValueError("component shape must be positive [layer, component]")
        self.component_shape = component_shape
        self.top_fraction = top_fraction
        self.consistency_threshold = consistency_threshold
        self.sign_mode = sign_mode
        self.weighted_support = np.zeros(component_shape, dtype=np.float64)
        self.weighted_activation = np.zeros(component_shape, dtype=np.float64)
        self.total_weight = 0.0
        self.consistency_sum = 0.0
        self.stable_sum = 0.0
        self.example_count = 0

    def add(
        self,
        view_attributions: Iterable[np.ndarray],
        *,
        weight: float = 1.0,
        view_activations: Iterable[np.ndarray] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if weight < 0:
            raise ValueError("circuit reliability weights must be nonnegative")
        views = [np.asarray(value, dtype=np.float32) for value in view_attributions]
        soft, consistency, stable = refine_prompt_consistent_attribution(
            views,
            top_fraction=self.top_fraction,
            consistency_threshold=self.consistency_threshold,
            sign_mode=self.sign_mode,
        )
        if soft.shape != self.component_shape:
            raise ValueError(
                f"component attribution has shape {soft.shape}, "
                f"expected {self.component_shape}"
            )
        if weight > 0:
            self.weighted_support += weight * soft
            if view_activations is not None:
                activations = [
                    np.asarray(value, dtype=np.float32) for value in view_activations
                ]
                if not activations or any(
                    value.shape != self.component_shape for value in activations
                ):
                    raise ValueError("component activation shapes differ")
                if any(not np.all(np.isfinite(value)) for value in activations):
                    raise ValueError("component activation contains non-finite values")
                self.weighted_activation += weight * np.mean(activations, axis=0)
            self.total_weight += weight
        self.consistency_sum += float(np.mean(consistency))
        self.stable_sum += float(np.mean(stable))
        self.example_count += 1
        return soft, consistency, stable

    def finalize(self, eps: float = 1e-12) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
        if self.total_weight <= 0 or self.example_count <= 0:
            raise ValueError("circuit accumulator has no positive-weight examples")
        raw_support = (self.weighted_support / self.total_weight).astype(np.float32)
        layer_max = np.max(raw_support, axis=-1, keepdims=True)
        support = np.divide(
            raw_support,
            np.maximum(layer_max, eps),
            out=np.zeros_like(raw_support),
            where=layer_max > eps,
        )
        activation = (self.weighted_activation / self.total_weight).astype(np.float32)
        stats = {
            "prompt_consistency_mean": self.consistency_sum / self.example_count,
            "stable_fraction": self.stable_sum / self.example_count,
            "support_raw_mean": float(raw_support.mean()),
            "support_mean": float(support.mean()),
            "example_count": float(self.example_count),
            "effective_weight_sum": float(self.total_weight),
        }
        return support, activation, stats


def _tiered_family_map(
    synthetic_support: np.ndarray,
    real_support: np.ndarray,
    *,
    synthetic_threshold: float,
    real_threshold: float,
    core_strength: float,
    synthetic_only_strength: float,
    real_residual_strength: float,
    real_completion: bool,
    enabled: bool,
) -> tuple[np.ndarray, np.ndarray]:
    synthetic_support = np.asarray(synthetic_support, dtype=np.float32)
    real_support = np.asarray(real_support, dtype=np.float32)
    if synthetic_support.shape != real_support.shape:
        raise ValueError("synthetic and real circuit support shapes differ")
    if not enabled:
        return (
            np.zeros_like(synthetic_support),
            np.full(synthetic_support.shape, TIER_UNSUPPORTED, dtype=np.int8),
        )
    synth = synthetic_support >= synthetic_threshold
    real = real_support >= real_threshold
    if not real_completion:
        real = np.zeros_like(real)
    tier = np.full(synthetic_support.shape, TIER_UNSUPPORTED, dtype=np.int8)
    tier[synth & real] = TIER_CORE
    tier[synth & ~real] = TIER_SYNTHETIC_ONLY
    tier[~synth & real] = TIER_REAL_RESIDUAL
    weights = np.zeros_like(synthetic_support)
    weights[tier == TIER_CORE] = (
        np.maximum(synthetic_support, real_support)[tier == TIER_CORE]
        * core_strength
    )
    weights[tier == TIER_SYNTHETIC_ONLY] = (
        synthetic_support[tier == TIER_SYNTHETIC_ONLY] * synthetic_only_strength
    )
    weights[tier == TIER_REAL_RESIDUAL] = (
        real_support[tier == TIER_REAL_RESIDUAL] * real_residual_strength
    )
    return np.clip(weights, 0.0, 1.0), tier


def build_dual_circuit_map(
    synthetic_mlp_support: np.ndarray,
    real_mlp_support: np.ndarray,
    synthetic_attention_support: np.ndarray,
    real_attention_support: np.ndarray,
    *,
    synthetic_threshold: float = 0.5,
    real_threshold: float = 0.5,
    core_strength: float = 1.0,
    synthetic_only_strength: float = 0.5,
    real_residual_strength: float = 1.0,
    real_completion: bool = True,
    enable_mlp: bool = True,
    enable_attention: bool = True,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Build fixed weighted component tiers from synthetic and real support."""
    for name, value in (
        ("synthetic_threshold", synthetic_threshold),
        ("real_threshold", real_threshold),
        ("core_strength", core_strength),
        ("synthetic_only_strength", synthetic_only_strength),
        ("real_residual_strength", real_residual_strength),
    ):
        if not 0 <= value <= 1:
            raise ValueError(f"{name} must be in [0, 1]")
    mlp_weight, mlp_tier = _tiered_family_map(
        synthetic_mlp_support,
        real_mlp_support,
        synthetic_threshold=synthetic_threshold,
        real_threshold=real_threshold,
        core_strength=core_strength,
        synthetic_only_strength=synthetic_only_strength,
        real_residual_strength=real_residual_strength,
        real_completion=real_completion,
        enabled=enable_mlp,
    )
    attention_weight, attention_tier = _tiered_family_map(
        synthetic_attention_support,
        real_attention_support,
        synthetic_threshold=synthetic_threshold,
        real_threshold=real_threshold,
        core_strength=core_strength,
        synthetic_only_strength=synthetic_only_strength,
        real_residual_strength=real_residual_strength,
        real_completion=real_completion,
        enabled=enable_attention,
    )
    circuit = {
        "mlp_weight": mlp_weight,
        "mlp_tier": mlp_tier,
        "attention_weight": attention_weight,
        "attention_tier": attention_tier,
        "synthetic_mlp_support": np.asarray(synthetic_mlp_support, dtype=np.float32),
        "real_mlp_support": np.asarray(real_mlp_support, dtype=np.float32),
        "synthetic_attention_support": np.asarray(
            synthetic_attention_support, dtype=np.float32
        ),
        "real_attention_support": np.asarray(real_attention_support, dtype=np.float32),
    }
    flat_tiers = np.concatenate([mlp_tier.ravel(), attention_tier.ravel()])
    synth_selected = np.concatenate(
        [
            (np.asarray(synthetic_mlp_support) >= synthetic_threshold).ravel(),
            (np.asarray(synthetic_attention_support) >= synthetic_threshold).ravel(),
        ]
    )
    real_selected = np.concatenate(
        [
            (np.asarray(real_mlp_support) >= real_threshold).ravel(),
            (np.asarray(real_attention_support) >= real_threshold).ravel(),
        ]
    )
    intersection = np.sum(synth_selected & real_selected)
    union = np.sum(synth_selected | real_selected)
    summary = {
        "mlp_component_count": int(mlp_weight.size),
        "attention_component_count": int(attention_weight.size),
        "synthetic_core_fraction": float(np.mean(flat_tiers == TIER_CORE)),
        "synthetic_only_fraction": float(
            np.mean(flat_tiers == TIER_SYNTHETIC_ONLY)
        ),
        "real_residual_fraction": float(
            np.mean(flat_tiers == TIER_REAL_RESIDUAL)
        ),
        "unsupported_fraction": float(np.mean(flat_tiers == TIER_UNSUPPORTED)),
        "synth_real_component_jaccard": float(intersection / max(union, 1)),
        "real_completion_count": int(np.sum(flat_tiers == TIER_REAL_RESIDUAL)),
        "real_completion_enabled": bool(real_completion),
        "mlp_enabled": bool(enable_mlp),
        "attention_enabled": bool(enable_attention),
    }
    return circuit, {"summary": summary}


def build_crossfit_ffn_value_circuit(
    synthetic_mlp_support: np.ndarray,
    attention_shape: tuple[int, int],
    *,
    threshold: float = 0.5,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Build a binary synthetic-only FFN value circuit.

    Cross-fitted value projection deliberately excludes attention and real-only
    map completion. Real anchors constrain updates later; they do not select
    which value vectors are protected.
    """
    support = np.asarray(synthetic_mlp_support, dtype=np.float32)
    if support.ndim != 2:
        raise ValueError("cross-fit MLP support must have shape [layer, channel]")
    if len(attention_shape) != 2 or min(attention_shape) <= 0:
        raise ValueError("cross-fit attention shape must be positive [layer, head]")
    if attention_shape[0] != support.shape[0]:
        raise ValueError("cross-fit MLP and attention layer counts differ")
    if not 0 < threshold <= 1:
        raise ValueError("cross-fit support threshold must be in (0, 1]")

    selected = support >= threshold
    mlp_weight = selected.astype(np.float32)
    mlp_tier = np.full(support.shape, TIER_UNSUPPORTED, dtype=np.int8)
    mlp_tier[selected] = TIER_SYNTHETIC_ONLY
    attention_weight = np.zeros(attention_shape, dtype=np.float32)
    attention_tier = np.full(
        attention_shape,
        TIER_UNSUPPORTED,
        dtype=np.int8,
    )
    total_count = mlp_weight.size + attention_weight.size
    selected_count = int(np.sum(selected))
    circuit = {
        "mlp_weight": mlp_weight,
        "mlp_tier": mlp_tier,
        "attention_weight": attention_weight,
        "attention_tier": attention_tier,
        "synthetic_mlp_support": support,
    }
    summary = {
        "mlp_component_count": int(mlp_weight.size),
        "attention_component_count": int(attention_weight.size),
        "mlp_selected_component_count": selected_count,
        "attention_selected_component_count": 0,
        "synthetic_core_fraction": 0.0,
        "synthetic_only_fraction": float(selected_count / max(total_count, 1)),
        "real_residual_fraction": 0.0,
        "unsupported_fraction": float(1.0 - selected_count / max(total_count, 1)),
        "synth_real_component_jaccard": 0.0,
        "real_completion_count": 0,
        "real_completion_enabled": False,
        "mlp_enabled": True,
        "attention_enabled": False,
        "value_projection_only": True,
    }
    return circuit, {"summary": summary}


def build_prompt_invariant_output_circuit(
    synthetic_mlp_support: np.ndarray,
    synthetic_attention_support: np.ndarray,
    *,
    mlp_threshold: float = 0.5,
    attention_threshold: float = 0.5,
    enable_mlp: bool = True,
    enable_attention: bool = True,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Build a synthetic-only circuit over FFN values and attention outputs.

    The support values remain continuous after thresholding. They are later
    used as projection strengths, not as elementwise parameter masks.
    """
    mlp_support = np.asarray(synthetic_mlp_support, dtype=np.float32)
    attention_support = np.asarray(
        synthetic_attention_support,
        dtype=np.float32,
    )
    if mlp_support.ndim != 2 or attention_support.ndim != 2:
        raise ValueError("prompt-invariant support must have [layer, component] shape")
    if mlp_support.shape[0] != attention_support.shape[0]:
        raise ValueError("prompt-invariant MLP and attention layer counts differ")
    for name, threshold in (
        ("MLP", mlp_threshold),
        ("attention", attention_threshold),
    ):
        if not 0 < threshold <= 1:
            raise ValueError(f"{name} support threshold must be in (0, 1]")

    mlp_selected = (mlp_support >= mlp_threshold) & enable_mlp
    attention_selected = (
        attention_support >= attention_threshold
    ) & enable_attention
    mlp_weight = np.where(mlp_selected, mlp_support, 0.0).astype(np.float32)
    attention_weight = np.where(
        attention_selected,
        attention_support,
        0.0,
    ).astype(np.float32)
    mlp_tier = np.full(mlp_support.shape, TIER_UNSUPPORTED, dtype=np.int8)
    attention_tier = np.full(
        attention_support.shape,
        TIER_UNSUPPORTED,
        dtype=np.int8,
    )
    mlp_tier[mlp_selected] = TIER_SYNTHETIC_ONLY
    attention_tier[attention_selected] = TIER_SYNTHETIC_ONLY
    selected_count = int(np.sum(mlp_selected) + np.sum(attention_selected))
    total_count = mlp_weight.size + attention_weight.size
    circuit = {
        "mlp_weight": mlp_weight,
        "mlp_tier": mlp_tier,
        "attention_weight": attention_weight,
        "attention_tier": attention_tier,
        "synthetic_mlp_support": mlp_support,
        "synthetic_attention_support": attention_support,
    }
    summary = {
        "mlp_component_count": int(mlp_weight.size),
        "attention_component_count": int(attention_weight.size),
        "mlp_selected_component_count": int(np.sum(mlp_selected)),
        "attention_selected_component_count": int(np.sum(attention_selected)),
        "synthetic_core_fraction": 0.0,
        "synthetic_only_fraction": float(selected_count / max(total_count, 1)),
        "real_residual_fraction": 0.0,
        "unsupported_fraction": float(1.0 - selected_count / max(total_count, 1)),
        "synth_real_component_jaccard": 0.0,
        "real_completion_count": 0,
        "real_completion_enabled": False,
        "mlp_enabled": bool(enable_mlp),
        "attention_enabled": bool(enable_attention),
        "output_projection_only": True,
    }
    return circuit, {"summary": summary}


def save_dual_circuit_cache(
    path: str | Path,
    circuit: dict[str, np.ndarray],
    metadata: dict[str, Any],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **circuit)
    path.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )


def load_dual_circuit_cache(
    path: str | Path,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        circuit = {name: np.asarray(data[name]) for name in data.files}
    metadata = json.loads(path.with_suffix(".json").read_text())
    return circuit, metadata


def shard_dual_circuit_map(
    circuit: dict[str, np.ndarray],
    layouts: tuple[DualCircuitLayerLayout, ...],
    weights: dict[str, jax.Array],
) -> dict[str, jax.Array]:
    """Place component maps on the Qwen tensor-parallel component axes."""
    if not layouts:
        raise ValueError("cannot shard a dual circuit without layouts")
    mlp_shape = (len(layouts), layouts[0].intermediate_size)
    attention_shape = (len(layouts), layouts[0].query_head_count)
    for name in ("mlp_weight", "mlp_tier"):
        if np.asarray(circuit[name]).shape != mlp_shape:
            raise ValueError(f"{name} shape does not match circuit layout")
    for name in ("attention_weight", "attention_tier"):
        if np.asarray(circuit[name]).shape != attention_shape:
            raise ValueError(f"{name} shape does not match circuit layout")

    down_sharding = getattr(weights[layouts[0].down_path], "sharding", None)
    o_sharding = getattr(weights[layouts[0].o_path], "sharding", None)
    v_sharding = getattr(weights[layouts[0].v_path], "sharding", None)
    mlp_sharding = None
    attention_sharding = None
    kv_sharding = None
    if isinstance(down_sharding, NamedSharding):
        mlp_sharding = NamedSharding(down_sharding.mesh, P(None, "model"))
    if isinstance(o_sharding, NamedSharding):
        attention_sharding = NamedSharding(o_sharding.mesh, P(None, "model"))
    if isinstance(v_sharding, NamedSharding):
        kv_sharding = NamedSharding(v_sharding.mesh, P(None, "model"))

    def put(value: np.ndarray, sharding: NamedSharding | None) -> jax.Array:
        array = jnp.asarray(value)
        return jax.device_put(array, sharding) if sharding is not None else array

    sharded = {
        "mlp_weight": put(np.asarray(circuit["mlp_weight"], dtype=np.float32), mlp_sharding),
        "mlp_tier": put(np.asarray(circuit["mlp_tier"], dtype=np.int8), mlp_sharding),
        "attention_weight": put(
            np.asarray(circuit["attention_weight"], dtype=np.float32),
            attention_sharding,
        ),
        "attention_tier": put(
            np.asarray(circuit["attention_tier"], dtype=np.int8),
            attention_sharding,
        ),
    }
    kv_weights = []
    kv_tiers = []
    host_attention_weight = np.asarray(circuit["attention_weight"], dtype=np.float32)
    host_attention_tier = np.asarray(circuit["attention_tier"], dtype=np.int8)
    for layout in layouts:
        layer_weights = []
        layer_tiers = []
        for kv_head in range(layout.kv_head_count):
            indices = [
                index
                for index, mapped in enumerate(layout.query_to_kv)
                if mapped == kv_head
            ]
            values = host_attention_weight[layout.layer, indices]
            winner = int(np.argmax(values))
            layer_weights.append(values[winner])
            layer_tiers.append(host_attention_tier[layout.layer, indices[winner]])
        kv_weights.append(layer_weights)
        kv_tiers.append(layer_tiers)
    sharded["attention_kv_weight"] = put(
        np.asarray(kv_weights, dtype=np.float32),
        kv_sharding,
    )
    sharded["attention_kv_tier"] = put(
        np.asarray(kv_tiers, dtype=np.int8),
        kv_sharding,
    )
    return sharded


def matched_random_component_mask(
    selected: np.ndarray,
    activation: np.ndarray,
    *,
    seed: int,
) -> np.ndarray:
    """Match selected components by layer, count, and approximate activation rank."""
    selected = np.asarray(selected, dtype=bool)
    activation = np.asarray(activation, dtype=np.float32)
    if selected.shape != activation.shape or selected.ndim != 2:
        raise ValueError("selected and activation must be matching [layer, component]")
    rng = np.random.default_rng(seed)
    random_mask = np.zeros_like(selected)
    for layer in range(selected.shape[0]):
        chosen = np.flatnonzero(selected[layer])
        candidates = np.flatnonzero(~selected[layer])
        if not len(chosen):
            continue
        if len(candidates) < len(chosen):
            # Exact count matching takes precedence when a very permissive
            # threshold protects over half a layer. Overlap is then
            # unavoidable and should be visible in the circuit fractions.
            candidates = np.arange(selected.shape[1])
        available = candidates.tolist()
        rng.shuffle(available)
        for component in chosen:
            if not available:
                break
            values = np.asarray(
                [
                    abs(float(activation[layer, candidate]) - float(activation[layer, component]))
                    for candidate in available
                ]
            )
            match_idx = int(np.argmin(values))
            match = available.pop(match_idx)
            random_mask[layer, match] = True
    return random_mask


def compare_circuit_support_maps(
    primary_support: np.ndarray,
    replication_support: np.ndarray,
    *,
    threshold: float,
) -> dict[str, float]:
    """Measure whether a thresholded circuit is rediscovered on disjoint data."""
    primary = np.asarray(primary_support, dtype=np.float32)
    replication = np.asarray(replication_support, dtype=np.float32)
    if primary.shape != replication.shape or primary.ndim != 2:
        raise ValueError(
            "circuit support maps must be matching [layer, component] arrays"
        )
    if not 0 < threshold <= 1:
        raise ValueError("circuit support threshold must be in (0, 1]")
    if not np.all(np.isfinite(primary)) or not np.all(np.isfinite(replication)):
        raise ValueError("circuit support maps contain non-finite values")

    primary_selected = primary >= threshold
    replication_selected = replication >= threshold
    intersection = primary_selected & replication_selected
    union = primary_selected | replication_selected
    primary_count = int(np.sum(primary_selected))
    replication_count = int(np.sum(replication_selected))
    intersection_count = int(np.sum(intersection))
    union_count = int(np.sum(union))
    smaller_count = min(primary_count, replication_count)

    active_layer_jaccards = []
    for layer in range(primary.shape[0]):
        layer_union = primary_selected[layer] | replication_selected[layer]
        if np.any(layer_union):
            layer_intersection = (
                primary_selected[layer] & replication_selected[layer]
            )
            active_layer_jaccards.append(
                float(np.sum(layer_intersection) / np.sum(layer_union))
            )

    primary_flat = primary.ravel().astype(np.float64)
    replication_flat = replication.ravel().astype(np.float64)
    primary_norm = float(np.linalg.norm(primary_flat))
    replication_norm = float(np.linalg.norm(replication_flat))
    if np.std(primary_flat) > 1e-12 and np.std(replication_flat) > 1e-12:
        pearson = float(np.corrcoef(primary_flat, replication_flat)[0, 1])
    else:
        pearson = 0.0
    cosine = (
        float(np.dot(primary_flat, replication_flat))
        / (primary_norm * replication_norm)
        if primary_norm > 1e-12 and replication_norm > 1e-12
        else 0.0
    )
    return {
        "primary_selected_count": float(primary_count),
        "replication_selected_count": float(replication_count),
        "intersection_count": float(intersection_count),
        "union_count": float(union_count),
        "selected_jaccard": float(
            intersection_count / union_count if union_count else 0.0
        ),
        "selected_overlap_coefficient": float(
            intersection_count / smaller_count if smaller_count else 0.0
        ),
        "active_layer_jaccard_mean": float(
            np.mean(active_layer_jaccards) if active_layer_jaccards else 0.0
        ),
        "support_pearson": pearson,
        "support_cosine": cosine,
    }


def _bootstrap_mean_interval(
    values: np.ndarray,
    *,
    repeats: int,
    seed: int,
    confidence: float = 0.95,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("bootstrap values must be one-dimensional")
    if repeats <= 0:
        raise ValueError("bootstrap repeats must be positive")
    if not 0 < confidence < 1:
        raise ValueError("bootstrap confidence must be in (0, 1)")
    if not len(values):
        return 0.0, 0.0
    if len(values) == 1:
        value = float(values[0])
        return value, value
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(repeats, len(values)))
    means = np.mean(values[indices], axis=1)
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(means, [alpha, 1.0 - alpha])
    return float(low), float(high)


def summarize_circuit_interventions(
    records: Iterable[dict[str, Any]],
    *,
    bootstrap_repeats: int,
    seed: int,
) -> dict[str, float]:
    """Summarize grouped necessity, specificity, and retained-only rescue."""
    rows = list(records)
    zero_summary = {
        "top_margin_drop": 0.0,
        "random_margin_drop": 0.0,
        "specificity_ratio": 0.0,
        "paired_margin_advantage": 0.0,
        "paired_margin_advantage_ci_low": 0.0,
        "paired_margin_advantage_ci_high": 0.0,
        "positive_view_rate": 0.0,
        "all_views_positive_family_rate": 0.0,
        "family_effect_std_mean": 0.0,
        "baseline_correct_rate": 0.0,
        "suppressed_correct_rate": 0.0,
        "correct_to_incorrect_flip_rate": 0.0,
        "random_correct_to_incorrect_flip_rate": 0.0,
        "family_zeroed_margin": 0.0,
        "selected_only_margin": 0.0,
        "selected_only_correct_rate": 0.0,
        "selected_only_rescue": 0.0,
        "random_only_rescue": 0.0,
        "selected_only_rescue_advantage": 0.0,
        "selected_only_rescue_advantage_ci_low": 0.0,
        "selected_only_rescue_advantage_ci_high": 0.0,
        "family_total_margin_contribution": 0.0,
        "normalized_rescue_mean": 0.0,
        "normalized_rescue_median": 0.0,
        "normalized_rescue_eligible_rate": 0.0,
        "family_count": 0.0,
        "view_count": 0.0,
    }
    if not rows:
        return zero_summary

    required = {
        "family_id",
        "baseline_margin",
        "suppressed_margin",
        "random_suppressed_margins",
        "family_zeroed_margin",
        "selected_only_margin",
        "random_only_margins",
    }
    if any(not required.issubset(row) for row in rows):
        raise ValueError("circuit intervention record is missing required fields")

    baseline = np.asarray(
        [row["baseline_margin"] for row in rows], dtype=np.float64
    )
    suppressed = np.asarray(
        [row["suppressed_margin"] for row in rows], dtype=np.float64
    )
    zeroed = np.asarray(
        [row["family_zeroed_margin"] for row in rows], dtype=np.float64
    )
    selected_only = np.asarray(
        [row["selected_only_margin"] for row in rows], dtype=np.float64
    )
    random_suppressed = np.asarray(
        [row["random_suppressed_margins"] for row in rows],
        dtype=np.float64,
    )
    random_only = np.asarray(
        [row["random_only_margins"] for row in rows],
        dtype=np.float64,
    )
    if random_suppressed.ndim != 2 or random_only.shape != random_suppressed.shape:
        raise ValueError("random intervention records must have matching repeat counts")
    arrays = (
        baseline,
        suppressed,
        zeroed,
        selected_only,
        random_suppressed,
        random_only,
    )
    if any(not np.all(np.isfinite(value)) for value in arrays):
        raise ValueError("circuit intervention records contain non-finite values")

    top_drop = baseline - suppressed
    random_drop = np.mean(baseline[:, None] - random_suppressed, axis=1)
    paired_advantage = top_drop - random_drop
    selected_rescue = selected_only - zeroed
    random_rescue = np.mean(random_only - zeroed[:, None], axis=1)
    rescue_advantage = selected_rescue - random_rescue
    total_family_contribution = baseline - zeroed

    family_indices: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        family_indices.setdefault(str(row["family_id"]), []).append(index)

    def family_means(values: np.ndarray) -> np.ndarray:
        return np.asarray(
            [
                np.mean(values[np.asarray(indices, dtype=np.int32)])
                for indices in family_indices.values()
            ],
            dtype=np.float64,
        )

    family_paired = family_means(paired_advantage)
    family_rescue_advantage = family_means(rescue_advantage)
    paired_low, paired_high = _bootstrap_mean_interval(
        family_paired,
        repeats=bootstrap_repeats,
        seed=seed,
    )
    rescue_low, rescue_high = _bootstrap_mean_interval(
        family_rescue_advantage,
        repeats=bootstrap_repeats,
        seed=seed + 104729,
    )
    family_all_positive = [
        bool(np.all(top_drop[np.asarray(indices, dtype=np.int32)] > 0))
        for indices in family_indices.values()
    ]
    family_effect_std = [
        float(np.std(top_drop[np.asarray(indices, dtype=np.int32)]))
        for indices in family_indices.values()
    ]
    baseline_correct = baseline > 0
    random_flips = np.mean(
        baseline_correct[:, None] & (random_suppressed <= 0)
    )
    eligible = total_family_contribution > 1e-6
    normalized_rescue = np.divide(
        selected_rescue,
        total_family_contribution,
        out=np.zeros_like(selected_rescue),
        where=eligible,
    )

    top_mean = float(np.mean(top_drop))
    random_mean = float(np.mean(random_drop))
    return {
        "top_margin_drop": top_mean,
        "random_margin_drop": random_mean,
        "specificity_ratio": top_mean / max(abs(random_mean), 1e-8),
        "paired_margin_advantage": float(np.mean(paired_advantage)),
        "paired_margin_advantage_ci_low": paired_low,
        "paired_margin_advantage_ci_high": paired_high,
        "positive_view_rate": float(np.mean(top_drop > 0)),
        "all_views_positive_family_rate": float(np.mean(family_all_positive)),
        "family_effect_std_mean": float(np.mean(family_effect_std)),
        "baseline_correct_rate": float(np.mean(baseline_correct)),
        "suppressed_correct_rate": float(np.mean(suppressed > 0)),
        "correct_to_incorrect_flip_rate": float(
            np.mean(baseline_correct & (suppressed <= 0))
        ),
        "random_correct_to_incorrect_flip_rate": float(random_flips),
        "family_zeroed_margin": float(np.mean(zeroed)),
        "selected_only_margin": float(np.mean(selected_only)),
        "selected_only_correct_rate": float(np.mean(selected_only > 0)),
        "selected_only_rescue": float(np.mean(selected_rescue)),
        "random_only_rescue": float(np.mean(random_rescue)),
        "selected_only_rescue_advantage": float(np.mean(rescue_advantage)),
        "selected_only_rescue_advantage_ci_low": rescue_low,
        "selected_only_rescue_advantage_ci_high": rescue_high,
        "family_total_margin_contribution": float(
            np.mean(total_family_contribution)
        ),
        "normalized_rescue_mean": float(
            np.mean(normalized_rescue[eligible]) if np.any(eligible) else 0.0
        ),
        "normalized_rescue_median": float(
            np.median(normalized_rescue[eligible]) if np.any(eligible) else 0.0
        ),
        "normalized_rescue_eligible_rate": float(np.mean(eligible)),
        "family_count": float(len(family_indices)),
        "view_count": float(len(rows)),
    }


def _tree_mean(tree: PyTree) -> jax.Array:
    leaves = jax.tree_util.tree_leaves(tree)
    total = sum(
        (jnp.sum(jnp.asarray(value, dtype=jnp.float32)) for value in leaves),
        jnp.asarray(0.0, dtype=jnp.float32),
    )
    size = sum(value.size for value in leaves)
    return total / jnp.asarray(float(max(size, 1)), dtype=total.dtype)


def _apply_mask(tree: PyTree, masks: PyTree) -> PyTree:
    return jax.tree_util.tree_map(
        lambda value, mask: value * jnp.asarray(mask, dtype=value.dtype),
        tree,
        masks,
    )


def _update_mask_ema(
    hard_masks: PyTree,
    mask_ema: PyTree,
    mask_ema_count: jax.Array,
    *,
    beta: float,
) -> tuple[PyTree, PyTree, jax.Array]:
    if not 0 <= beta < 1:
        raise ValueError("circuit mask EMA beta must be in [0, 1)")
    beta_value = jnp.asarray(beta, dtype=jnp.float32)
    mask_ema = jax.tree_util.tree_map(
        lambda ema, hard: beta_value * ema + (1.0 - beta_value) * hard,
        mask_ema,
        hard_masks,
    )
    mask_ema_count = mask_ema_count + jnp.asarray(1, dtype=mask_ema_count.dtype)
    correction = 1.0 - beta_value ** mask_ema_count.astype(jnp.float32)
    effective = jax.tree_util.tree_map(
        lambda ema: jnp.clip(ema / correction, 0.0, 1.0),
        mask_ema,
    )
    return effective, mask_ema, mask_ema_count


def _kv_component_values(
    query_values: jax.Array,
    query_tiers: jax.Array,
    mapping: tuple[int, ...],
    kv_head_count: int,
) -> tuple[jax.Array, jax.Array]:
    """Collapse GQA query components into one strongest constraint per KV head."""
    values = []
    tiers = []
    for kv_head in range(kv_head_count):
        indices = tuple(index for index, mapped in enumerate(mapping) if mapped == kv_head)
        group_values = query_values[jnp.asarray(indices, dtype=jnp.int32)]
        winner = jnp.argmax(group_values)
        values.append(group_values[winner])
        tiers.append(query_tiers[jnp.asarray(indices, dtype=jnp.int32)][winner])
    return jnp.stack(values), jnp.stack(tiers)


def _project_component_blocks(
    update: jax.Array,
    synthetic_mask: jax.Array,
    correction_ema: jax.Array,
    correction_ema_count: jax.Array,
    anchor_group_grads: tuple[dict[str, jax.Array], ...],
    path: str,
    component_axis: int,
    component_strength: jax.Array,
    component_tier: jax.Array,
    *,
    consensus_threshold: float,
    projection_passes: int,
    hard_veto: bool,
    correction_beta: float,
    eps: float,
) -> tuple[jax.Array, jax.Array, dict[str, jax.Array]]:
    """Project disjoint parameter blocks against fixed anchor-group half-spaces."""
    original_dtype = update.dtype
    moved = jnp.moveaxis(jnp.asarray(update, dtype=jnp.float32), component_axis, 0)
    moved_shape = moved.shape
    moved = moved.reshape((moved.shape[0], -1))
    original = moved
    moved_correction_ema = jnp.moveaxis(
        jnp.asarray(correction_ema, dtype=jnp.float32), component_axis, 0
    ).reshape(moved.shape)
    support = jnp.moveaxis(
        jnp.asarray(synthetic_mask, dtype=jnp.float32), component_axis, 0
    ).reshape(moved.shape)
    grouped = jnp.stack(
        [
            jnp.moveaxis(
                jnp.asarray(group[path], dtype=jnp.float32), component_axis, 0
            ).reshape(moved.shape)
            * support
            for group in anchor_group_grads
        ],
        axis=0,
    )
    strength = jnp.asarray(component_strength, dtype=jnp.float32)
    tier = jnp.asarray(component_tier, dtype=jnp.int32)
    if strength.shape != (moved.shape[0],) or tier.shape != strength.shape:
        raise ValueError(
            f"component map for {path} has shape {strength.shape}, "
            f"expected {(moved.shape[0],)}"
        )
    before_group = jnp.einsum("gcd,cd->gc", grouped, moved)
    vote_fraction = jnp.mean(before_group > 0, axis=0)
    protected = strength > 0
    harmful = protected & (vote_fraction >= consensus_threshold)

    def apply_instantaneous_projection(current: jax.Array) -> jax.Array:
        if hard_veto:
            return jnp.where(harmful[:, None], jnp.zeros_like(current), current)
        for _ in range(projection_passes):
            for group_idx in range(len(anchor_group_grads)):
                gradient = grouped[group_idx]
                violation = jnp.sum(gradient * current, axis=-1)
                denominator = jnp.sum(jnp.square(gradient), axis=-1)
                scale = (
                    harmful.astype(jnp.float32)
                    * strength
                    * jax.nn.relu(violation)
                    / jnp.maximum(denominator, eps)
                )
                current = current - scale[:, None] * gradient
        return current

    instantaneous = apply_instantaneous_projection(original) * support
    instantaneous_correction = instantaneous - original
    beta_value = jnp.asarray(correction_beta, dtype=jnp.float32)
    moved_correction_ema = (
        beta_value * moved_correction_ema
        + (1.0 - beta_value) * instantaneous_correction
    )
    bias_correction = 1.0 - beta_value ** correction_ema_count.astype(jnp.float32)
    moved = original + moved_correction_ema / jnp.maximum(bias_correction, eps)
    moved = moved * support

    # EMA can lag a newly harmful constraint. Remove only the excess over the
    # instantaneous soft projection's allowed violation, avoiding a second
    # application of the configured tier strength.
    instantaneous_group = jnp.einsum("gcd,cd->gc", grouped, instantaneous)
    if hard_veto:
        moved = jnp.where(harmful[:, None], jnp.zeros_like(moved), moved)
    else:
        for _ in range(projection_passes):
            for group_idx in range(len(anchor_group_grads)):
                gradient = grouped[group_idx]
                current_violation = jnp.sum(gradient * moved, axis=-1)
                allowed_violation = instantaneous_group[group_idx]
                excess = jax.nn.relu(current_violation - allowed_violation)
                denominator = jnp.sum(jnp.square(gradient), axis=-1)
                scale = (
                    harmful.astype(jnp.float32)
                    * excess
                    / jnp.maximum(denominator, eps)
                )
                moved = moved - scale[:, None] * gradient
    moved = moved * support
    after_group = jnp.einsum("gcd,cd->gc", grouped, moved)
    correction_energy_by_component = jnp.sum(jnp.square(original - moved), axis=-1)
    final_energy = jnp.sum(jnp.square(moved))
    before_violation = jnp.max(jax.nn.relu(before_group), axis=0)
    after_violation = jnp.max(jax.nn.relu(after_group), axis=0)
    protected_count = jnp.sum(protected.astype(jnp.float32))
    stats = {
        "protected_count": protected_count,
        "harmful_count": jnp.sum(harmful.astype(jnp.float32)),
        "consensus_harmful_count": jnp.sum(
            (protected & (vote_fraction >= consensus_threshold)).astype(jnp.float32)
        ),
        "before_violation_sum": jnp.sum(before_violation * protected),
        "after_violation_sum": jnp.sum(after_violation * protected),
        "after_violation_max": jnp.max(
            jnp.where(protected, after_violation, jnp.zeros_like(after_violation))
        ),
        "correction_energy": jnp.sum(correction_energy_by_component),
        "final_energy": final_energy,
    }
    for tier_id, tier_name in enumerate(TIER_NAMES[:3]):
        stats[f"{tier_name}_correction_energy"] = jnp.sum(
            jnp.where(
                tier == tier_id,
                correction_energy_by_component,
                jnp.zeros_like(correction_energy_by_component),
            )
        )
    return (
        jnp.moveaxis(moved.reshape(moved_shape), 0, component_axis).astype(
            original_dtype
        ),
        jnp.moveaxis(
            moved_correction_ema.reshape(moved_shape), 0, component_axis
        ).astype(correction_ema.dtype),
        stats,
    )


def _project_crossfit_value_blocks(
    update: jax.Array,
    synthetic_mask: jax.Array,
    anchor_group_grads: tuple[dict[str, jax.Array], ...],
    path: str,
    component_axis: int,
    selected_components: jax.Array,
    *,
    consensus_threshold: float,
    projection_passes: int,
    eps: float,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Project selected value vectors without a correction EMA or hard veto."""
    original_dtype = update.dtype
    moved = jnp.moveaxis(jnp.asarray(update, dtype=jnp.float32), component_axis, 0)
    moved_shape = moved.shape
    moved = moved.reshape((moved.shape[0], -1))
    original = moved
    support = jnp.moveaxis(
        jnp.asarray(synthetic_mask, dtype=jnp.float32), component_axis, 0
    ).reshape(moved.shape)
    grouped = jnp.stack(
        [
            jnp.moveaxis(
                jnp.asarray(group[path], dtype=jnp.float32), component_axis, 0
            ).reshape(moved.shape)
            * support
            for group in anchor_group_grads
        ],
        axis=0,
    )
    selected = jnp.asarray(selected_components) > 0
    if selected.shape != (moved.shape[0],):
        raise ValueError(
            f"cross-fit selector for {path} has shape {selected.shape}, "
            f"expected {(moved.shape[0],)}"
        )
    before_group = jnp.einsum("gcd,cd->gc", grouped, moved)
    vote_fraction = jnp.mean(before_group > 0, axis=0)
    harmful = selected & (vote_fraction >= consensus_threshold)
    for _ in range(projection_passes):
        for group_idx in range(len(anchor_group_grads)):
            gradient = grouped[group_idx]
            violation = jnp.sum(gradient * moved, axis=-1)
            denominator = jnp.sum(jnp.square(gradient), axis=-1)
            scale = (
                harmful.astype(jnp.float32)
                * jax.nn.relu(violation)
                / jnp.maximum(denominator, eps)
            )
            moved = moved - scale[:, None] * gradient
    moved = moved * support
    after_group = jnp.einsum("gcd,cd->gc", grouped, moved)
    correction_energy = jnp.sum(jnp.square(original - moved), axis=-1)
    before_violation = jnp.max(jax.nn.relu(before_group), axis=0)
    after_violation = jnp.max(jax.nn.relu(after_group), axis=0)
    selected_count = jnp.sum(selected.astype(jnp.float32))
    stats = {
        "selected_count": selected_count,
        "harmful_count": jnp.sum(harmful.astype(jnp.float32)),
        "before_violation_sum": jnp.sum(before_violation * selected),
        "after_violation_sum": jnp.sum(after_violation * selected),
        "after_violation_max": jnp.max(
            jnp.where(selected, after_violation, jnp.zeros_like(after_violation))
        ),
        "correction_energy": jnp.sum(correction_energy),
    }
    return (
        jnp.moveaxis(moved.reshape(moved_shape), 0, component_axis).astype(
            original_dtype
        ),
        stats,
    )


def mask_optax_updates_crossfit_ffn_value_projection_ema(
    updates: dict[str, jax.Array],
    synthetic_group_grads: tuple[dict[str, jax.Array], ...],
    anchor_group_grads: tuple[dict[str, jax.Array], ...],
    layouts: tuple[DualCircuitLayerLayout, ...],
    value_selectors: jax.Array,
    mask_ema: dict[str, jax.Array],
    mask_ema_count: jax.Array,
    *,
    tail_fraction: float,
    beta: float,
    consensus_threshold: float = 0.75,
    projection_passes: int = 1,
    eps: float = 1e-12,
) -> tuple[
    dict[str, jax.Array],
    dict[str, jax.Array],
    jax.Array,
    dict[str, jax.Array],
]:
    """Apply synthetic CVaR globally and real projection to FFN values only."""
    if not synthetic_group_grads or not anchor_group_grads:
        raise ValueError("cross-fit value projection requires synthetic and anchor groups")
    if not layouts:
        raise ValueError("cross-fit value projection requires FFN layouts")
    if not 0 < consensus_threshold <= 1:
        raise ValueError("cross-fit anchor consensus must be in (0, 1]")
    if projection_passes <= 0 or eps <= 0:
        raise ValueError("cross-fit projection passes and epsilon must be positive")
    expected_shape = (len(layouts), layouts[0].intermediate_size)
    if tuple(value_selectors.shape) != expected_shape:
        raise ValueError(
            f"cross-fit value selectors have shape {value_selectors.shape}, "
            f"expected {expected_shape}"
        )

    synthetic_masks, _, synthetic_tail_count = compute_cvar_masks(
        updates,
        list(synthetic_group_grads),
        tail_fraction=tail_fraction,
    )
    effective_masks, mask_ema, mask_ema_count = _update_mask_ema(
        synthetic_masks,
        mask_ema,
        mask_ema_count,
        beta=beta,
    )
    proposal_masks = jax.tree_util.tree_map(
        lambda effective, hard: effective * hard,
        effective_masks,
        synthetic_masks,
    )
    synthetic_update = dict(_apply_mask(updates, proposal_masks))
    projected = dict(synthetic_update)
    aggregate = {
        "selected_count": jnp.asarray(0.0, dtype=jnp.float32),
        "harmful_count": jnp.asarray(0.0, dtype=jnp.float32),
        "before_violation_sum": jnp.asarray(0.0, dtype=jnp.float32),
        "after_violation_sum": jnp.asarray(0.0, dtype=jnp.float32),
        "after_violation_max": jnp.asarray(0.0, dtype=jnp.float32),
        "correction_energy": jnp.asarray(0.0, dtype=jnp.float32),
    }
    for layout in layouts:
        value, block_stats = _project_crossfit_value_blocks(
            projected[layout.down_path],
            synthetic_masks[layout.down_path],
            anchor_group_grads,
            layout.down_path,
            layout.down_component_axis,
            value_selectors[layout.layer],
            consensus_threshold=consensus_threshold,
            projection_passes=projection_passes,
            eps=eps,
        )
        projected[layout.down_path] = value
        for key, stat in block_stats.items():
            if key == "after_violation_max":
                aggregate[key] = jnp.maximum(aggregate[key], stat)
            else:
                aggregate[key] = aggregate[key] + stat

    projected = dict(_apply_mask(projected, synthetic_masks))
    raw_norm = tree_l2_norm(updates)
    synthetic_norm = tree_l2_norm(synthetic_update)
    final_norm = tree_l2_norm(projected)
    selected_denominator = jnp.maximum(aggregate["selected_count"], 1.0)
    stats = {
        "mask/synthetic_mean": _tree_mean(synthetic_masks),
        "mask/final_effective_mean": _tree_mean(proposal_masks),
        "update_norm/raw": raw_norm,
        "update_norm/synthetic_masked": synthetic_norm,
        "update_norm/final_structured": final_norm,
        "update_norm/kept_fraction": final_norm / jnp.maximum(raw_norm, eps),
        "crossfit/selected_value_count": aggregate["selected_count"],
        "crossfit/harmful_value_fraction": aggregate["harmful_count"]
        / selected_denominator,
        "crossfit/projection_violation_before_mean": aggregate[
            "before_violation_sum"
        ]
        / selected_denominator,
        "crossfit/projection_violation_after_mean": aggregate[
            "after_violation_sum"
        ]
        / selected_denominator,
        "crossfit/projection_violation_after_max": aggregate[
            "after_violation_max"
        ],
        "crossfit/value_correction_norm": jnp.sqrt(
            jnp.maximum(aggregate["correction_energy"], 0.0)
        ),
        "crossfit/synthetic_tail_count": jnp.asarray(
            synthetic_tail_count, dtype=jnp.float32
        ),
    }
    return projected, mask_ema, mask_ema_count, stats


def mask_optax_updates_prompt_invariant_circuit_veto_ema(
    updates: dict[str, jax.Array],
    synthetic_group_grads: tuple[dict[str, jax.Array], ...],
    anchor_group_grads: tuple[dict[str, jax.Array], ...],
    layouts: tuple[DualCircuitLayerLayout, ...],
    value_selectors: jax.Array,
    mask_ema: dict[str, jax.Array],
    mask_ema_count: jax.Array,
    *,
    tail_fraction: float,
    beta: float,
    eps: float = 1e-12,
) -> tuple[
    dict[str, jax.Array],
    dict[str, jax.Array],
    jax.Array,
    dict[str, jax.Array],
]:
    """Apply real CVaR vetoes only inside a frozen FFN value circuit."""
    if not synthetic_group_grads or not anchor_group_grads:
        raise ValueError(
            "prompt-invariant circuit veto requires synthetic and anchor groups"
        )
    if not layouts:
        raise ValueError("prompt-invariant circuit veto requires FFN layouts")
    if eps <= 0:
        raise ValueError("prompt-invariant circuit veto epsilon must be positive")
    expected_shape = (len(layouts), layouts[0].intermediate_size)
    if tuple(value_selectors.shape) != expected_shape:
        raise ValueError(
            f"prompt-invariant value selectors have shape {value_selectors.shape}, "
            f"expected {expected_shape}"
        )

    synthetic_masks, synthetic_scores, synthetic_tail_count = compute_cvar_masks(
        updates,
        list(synthetic_group_grads),
        tail_fraction=tail_fraction,
    )
    down_paths = tuple(layout.down_path for layout in layouts)
    down_updates = {path: updates[path] for path in down_paths}
    down_anchor_groups = [
        {path: group[path] for path in down_paths}
        for group in anchor_group_grads
    ]
    anchor_masks, anchor_scores, anchor_tail_count = compute_cvar_masks(
        down_updates,
        down_anchor_groups,
        tail_fraction=tail_fraction,
    )

    hard_masks = dict(synthetic_masks)
    selected_parameter_count = jnp.asarray(0.0, dtype=jnp.float32)
    selected_value_count = jnp.asarray(0.0, dtype=jnp.float32)
    real_keep_sum = jnp.asarray(0.0, dtype=jnp.float32)
    anchor_score_sum = jnp.asarray(0.0, dtype=jnp.float32)
    synthetic_eligible_count = jnp.asarray(0.0, dtype=jnp.float32)
    active_veto_count = jnp.asarray(0.0, dtype=jnp.float32)
    for layout in layouts:
        selector = (
            jnp.asarray(value_selectors[layout.layer], dtype=jnp.float32) > 0
        ).astype(jnp.float32)
        selector_shape = [1] * updates[layout.down_path].ndim
        selector_shape[layout.down_component_axis] = selector.shape[0]
        circuit = jnp.broadcast_to(
            selector.reshape(tuple(selector_shape)),
            updates[layout.down_path].shape,
        )
        synthetic = jnp.asarray(
            synthetic_masks[layout.down_path], dtype=jnp.float32
        )
        anchor = jnp.asarray(anchor_masks[layout.down_path], dtype=jnp.float32)
        localized_anchor = 1.0 - circuit + circuit * anchor
        hard_masks[layout.down_path] = synthetic * localized_anchor

        selected_parameter_count = selected_parameter_count + jnp.sum(circuit)
        selected_value_count = selected_value_count + jnp.sum(selector)
        real_keep_sum = real_keep_sum + jnp.sum(anchor * circuit)
        anchor_score_sum = anchor_score_sum + jnp.sum(
            jnp.asarray(anchor_scores[layout.down_path], dtype=jnp.float32)
            * circuit
        )
        synthetic_eligible_count = synthetic_eligible_count + jnp.sum(
            synthetic * circuit
        )
        active_veto_count = active_veto_count + jnp.sum(
            synthetic * circuit * (1.0 - anchor)
        )

    effective_masks, mask_ema, mask_ema_count = _update_mask_ema(
        hard_masks,
        mask_ema,
        mask_ema_count,
        beta=beta,
    )
    masked_updates = dict(_apply_mask(updates, effective_masks))
    total_parameter_count = sum(value.size for value in updates.values())
    selected_denominator = jnp.maximum(selected_parameter_count, 1.0)
    eligible_denominator = jnp.maximum(synthetic_eligible_count, 1.0)
    raw_norm = tree_l2_norm(updates)
    final_norm = tree_l2_norm(masked_updates)
    stats = {
        "anchor_veto/synth_mask_mean": _tree_mean(synthetic_masks),
        "anchor_veto/anchor_mask_mean": real_keep_sum / selected_denominator,
        "anchor_veto/combined_mask_mean": _tree_mean(hard_masks),
        "anchor_veto/ema_mask_mean": _tree_mean(effective_masks),
        "anchor_veto/ema_mask_frac_gt_0_5": _tree_mean(
            jax.tree_util.tree_map(
                lambda mask: (mask > 0.5).astype(jnp.float32),
                effective_masks,
            )
        ),
        "anchor_veto/veto_fraction": active_veto_count
        / max(total_parameter_count, 1),
        "anchor_veto/synth_score_mean": _tree_mean(synthetic_scores),
        "anchor_veto/anchor_score_mean": anchor_score_sum
        / selected_denominator,
        "anchor_veto/synth_tail_count": jnp.asarray(
            synthetic_tail_count, dtype=jnp.float32
        ),
        "anchor_veto/anchor_tail_count": jnp.asarray(
            anchor_tail_count, dtype=jnp.float32
        ),
        "anchor_veto/update_norm_before": raw_norm,
        "anchor_veto/update_norm_after": final_norm,
        "circuit_veto/selected_value_count": selected_value_count,
        "circuit_veto/selected_parameter_fraction": selected_parameter_count
        / max(total_parameter_count, 1),
        "circuit_veto/real_keep_inside": real_keep_sum / selected_denominator,
        "circuit_veto/real_veto_inside": 1.0
        - real_keep_sum / selected_denominator,
        "circuit_veto/active_veto_fraction_inside": active_veto_count
        / eligible_denominator,
        "mask/synthetic_mean": _tree_mean(synthetic_masks),
        "mask/final_effective_mean": _tree_mean(effective_masks),
        "update_norm/raw": raw_norm,
        "update_norm/final_structured": final_norm,
        "update_norm/kept_fraction": final_norm / jnp.maximum(raw_norm, eps),
    }
    return masked_updates, mask_ema, mask_ema_count, stats


def _tree_dot(left: PyTree, right: PyTree) -> jax.Array:
    products = jax.tree_util.tree_leaves(
        jax.tree_util.tree_map(
            lambda x, y: jnp.sum(
                jnp.asarray(x, dtype=jnp.float32)
                * jnp.asarray(y, dtype=jnp.float32)
            ),
            left,
            right,
        )
    )
    return sum(products, jnp.asarray(0.0, dtype=jnp.float32))


def _upper_tail_membership(values: jax.Array, tail_count: int) -> jax.Array:
    """Return an exact stable top-tail mask along the first dimension."""
    order = jnp.argsort(values, axis=0, stable=True)
    ranks = jnp.argsort(order, axis=0, stable=True)
    return ranks >= values.shape[0] - tail_count


def _project_tree_against_source_tail(
    updates: PyTree,
    group_grads: tuple[PyTree, ...],
    *,
    tail_fraction: float,
    projection_passes: int,
    eps: float,
) -> tuple[PyTree, dict[str, jax.Array]]:
    """Project a full update against one source's independently chosen tail."""
    tail_count = max(1, math.ceil(len(group_grads) * tail_fraction))
    before = jnp.stack([_tree_dot(group, updates) for group in group_grads])
    selected = _upper_tail_membership(before, tail_count)
    projected = updates
    for _ in range(projection_passes):
        for group_index, group in enumerate(group_grads):
            violation = _tree_dot(group, projected)
            denominator = _tree_dot(group, group)
            scale = (
                selected[group_index].astype(jnp.float32)
                * jax.nn.relu(violation)
                / jnp.maximum(denominator, eps)
            )
            projected = jax.tree_util.tree_map(
                lambda update, gradient: update
                - jnp.asarray(scale, dtype=update.dtype)
                * jnp.asarray(gradient, dtype=update.dtype),
                projected,
                group,
            )
    after = jnp.stack([_tree_dot(group, projected) for group in group_grads])
    selected_float = selected.astype(jnp.float32)
    return projected, {
        "tail_count": jnp.asarray(tail_count, dtype=jnp.float32),
        "violation_before_mean": jnp.sum(
            jax.nn.relu(before) * selected_float
        )
        / jnp.maximum(jnp.sum(selected_float), 1.0),
        "violation_after_mean": jnp.sum(
            jax.nn.relu(after) * selected_float
        )
        / jnp.maximum(jnp.sum(selected_float), 1.0),
        "violation_after_max": jnp.max(
            jnp.where(selected, jax.nn.relu(after), jnp.zeros_like(after))
        ),
    }


def _project_output_blocks_against_source_tail(
    update: jax.Array,
    group_grads: tuple[dict[str, jax.Array], ...],
    path: str,
    component_axis: int,
    component_strength: jax.Array,
    *,
    tail_fraction: float,
    projection_passes: int,
    parameter_support: jax.Array | None = None,
    agreement_weighting: bool = False,
    eps: float,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Apply blockwise CVaR projection for one source and component family."""
    original_dtype = update.dtype
    moved = jnp.moveaxis(jnp.asarray(update, dtype=jnp.float32), component_axis, 0)
    moved_shape = moved.shape
    moved = moved.reshape((moved.shape[0], -1))
    if parameter_support is None:
        support = jnp.ones_like(moved)
    else:
        support = jnp.moveaxis(
            jnp.asarray(parameter_support, dtype=jnp.float32),
            component_axis,
            0,
        ).reshape(moved.shape)
        moved = moved * support
    original = moved
    grouped = jnp.stack(
        [
            jnp.moveaxis(
                jnp.asarray(group[path], dtype=jnp.float32),
                component_axis,
                0,
            ).reshape(moved.shape)
            for group in group_grads
        ],
        axis=0,
    ) * support[None, :, :]
    strength = jnp.asarray(component_strength, dtype=jnp.float32)
    if strength.shape != (moved.shape[0],):
        raise ValueError(
            f"component strength for {path} has shape {strength.shape}, "
            f"expected {(moved.shape[0],)}"
        )
    selected = strength > 0
    tail_count = max(1, math.ceil(len(group_grads) * tail_fraction))
    before = jnp.einsum("gcd,cd->gc", grouped, moved)
    tail = _upper_tail_membership(before, tail_count)
    vote_fraction = jnp.mean(before > 0, axis=0)
    effective_strength = strength * (
        vote_fraction if agreement_weighting else 1.0
    )
    initially_harmful = tail & selected[None, :] & (before > 0)
    for _ in range(projection_passes):
        for group_index in range(len(group_grads)):
            gradient = grouped[group_index]
            violation = jnp.sum(gradient * moved, axis=-1)
            denominator = jnp.sum(jnp.square(gradient), axis=-1)
            scale = (
                tail[group_index].astype(jnp.float32)
                * selected.astype(jnp.float32)
                * effective_strength
                * jax.nn.relu(violation)
                / jnp.maximum(denominator, eps)
            )
            moved = moved - scale[:, None] * gradient
            moved = moved * support
    after = jnp.einsum("gcd,cd->gc", grouped, moved)
    active = tail & selected[None, :]
    active_float = active.astype(jnp.float32)
    selected_count = jnp.sum(selected.astype(jnp.float32))
    return (
        jnp.moveaxis((moved * support).reshape(moved_shape), 0, component_axis).astype(
            original_dtype
        ),
        {
            "selected_count": selected_count,
            "harmful_count": jnp.sum(
                jnp.any(initially_harmful, axis=0).astype(jnp.float32)
            ),
            "agreement_sum": jnp.sum(vote_fraction * selected),
            "violation_before_sum": jnp.sum(
                jax.nn.relu(before) * active_float
            ),
            "violation_after_sum": jnp.sum(
                jax.nn.relu(after) * active_float
            ),
            "active_constraint_count": jnp.sum(active_float),
            "violation_after_max": jnp.max(
                jnp.where(active, jax.nn.relu(after), jnp.zeros_like(after))
            ),
            "correction_energy": jnp.sum(jnp.square(original - moved)),
        },
    )


def mask_optax_updates_circuit_local_soft_projection_ema(
    updates: dict[str, jax.Array],
    synthetic_group_grads: tuple[dict[str, jax.Array], ...],
    anchor_group_grads: tuple[dict[str, jax.Array], ...],
    layouts: tuple[DualCircuitLayerLayout, ...],
    circuit: dict[str, jax.Array],
    mask_ema: dict[str, jax.Array],
    mask_ema_count: jax.Array,
    *,
    family: str,
    tail_fraction: float,
    beta: float,
    soft_projection_strength: float = 0.5,
    projection_passes: int = 1,
    eps: float = 1e-12,
) -> tuple[
    dict[str, jax.Array],
    dict[str, jax.Array],
    jax.Array,
    dict[str, jax.Array],
]:
    """Apply global synthetic masking and circuit-local real soft projection."""
    if not synthetic_group_grads or not anchor_group_grads:
        raise ValueError("circuit-local projection requires synthetic and real groups")
    if not layouts:
        raise ValueError("circuit-local projection requires circuit layouts")
    if family not in {"ffn", "attention"}:
        raise ValueError("circuit-local projection family must be ffn or attention")
    if not 0 <= soft_projection_strength <= 1:
        raise ValueError("soft projection strength must be in [0, 1]")
    if projection_passes <= 0 or eps <= 0:
        raise ValueError("projection passes and epsilon must be positive")

    synthetic_masks, synthetic_scores, synthetic_tail_count = compute_cvar_masks(
        updates,
        list(synthetic_group_grads),
        tail_fraction=tail_fraction,
    )
    effective_masks, mask_ema, mask_ema_count = _update_mask_ema(
        synthetic_masks,
        mask_ema,
        mask_ema_count,
        beta=beta,
    )
    proposal_masks = jax.tree_util.tree_map(
        lambda effective, hard: effective * hard,
        effective_masks,
        synthetic_masks,
    )
    synthetic_update = dict(_apply_mask(updates, proposal_masks))
    projected = dict(synthetic_update)
    aggregate = {
        "selected_count": jnp.asarray(0.0, dtype=jnp.float32),
        "harmful_count": jnp.asarray(0.0, dtype=jnp.float32),
        "agreement_sum": jnp.asarray(0.0, dtype=jnp.float32),
        "violation_before_sum": jnp.asarray(0.0, dtype=jnp.float32),
        "violation_after_sum": jnp.asarray(0.0, dtype=jnp.float32),
        "active_constraint_count": jnp.asarray(0.0, dtype=jnp.float32),
        "violation_after_max": jnp.asarray(0.0, dtype=jnp.float32),
        "correction_energy": jnp.asarray(0.0, dtype=jnp.float32),
    }

    def add_stats(values: dict[str, jax.Array]) -> None:
        for key, value in values.items():
            if key == "violation_after_max":
                aggregate[key] = jnp.maximum(aggregate[key], value)
            else:
                aggregate[key] = aggregate[key] + value

    for layout in layouts:
        if family == "ffn":
            blocks = (
                (
                    layout.down_path,
                    layout.down_component_axis,
                    circuit["mlp_weight"][layout.layer],
                ),
            )
        else:
            attention_weight = circuit["attention_weight"][layout.layer]
            if "attention_kv_weight" in circuit:
                kv_weight = circuit["attention_kv_weight"][layout.layer]
            else:
                kv_weight, _ = _kv_component_values(
                    attention_weight,
                    circuit["attention_tier"][layout.layer],
                    layout.query_to_kv,
                    layout.kv_head_count,
                )
            blocks = (
                (
                    layout.v_path,
                    layout.v_head_axis,
                    kv_weight,
                ),
                (
                    layout.o_path,
                    layout.o_head_axis,
                    attention_weight,
                ),
            )
        for path, axis, confidence in blocks:
            value, block_stats = _project_output_blocks_against_source_tail(
                projected[path],
                anchor_group_grads,
                path,
                axis,
                jnp.asarray(confidence, dtype=jnp.float32)
                * soft_projection_strength,
                tail_fraction=tail_fraction,
                projection_passes=projection_passes,
                parameter_support=synthetic_masks[path],
                agreement_weighting=True,
                eps=eps,
            )
            projected[path] = value
            add_stats(block_stats)

    projected = dict(_apply_mask(projected, synthetic_masks))
    selected = jnp.maximum(aggregate["selected_count"], 1.0)
    constraints = jnp.maximum(aggregate["active_constraint_count"], 1.0)
    raw_norm = tree_l2_norm(updates)
    synthetic_norm = tree_l2_norm(synthetic_update)
    final_norm = tree_l2_norm(projected)
    correction_norm = tree_l2_norm(
        jax.tree_util.tree_map(
            lambda before, after: before - after,
            synthetic_update,
            projected,
        )
    )
    stats = {
        "mask/synthetic_mean": _tree_mean(synthetic_masks),
        "mask/final_effective_mean": _tree_mean(proposal_masks),
        "local_soft/family_id": jnp.asarray(
            0 if family == "ffn" else 1,
            dtype=jnp.float32,
        ),
        "local_soft/strength": jnp.asarray(
            soft_projection_strength,
            dtype=jnp.float32,
        ),
        "local_soft/selected_block_count": aggregate["selected_count"],
        "local_soft/harmful_block_fraction": aggregate["harmful_count"] / selected,
        "local_soft/anchor_group_agreement_mean": (
            aggregate["agreement_sum"] / selected
        ),
        "local_soft/violation_before_mean": (
            aggregate["violation_before_sum"] / constraints
        ),
        "local_soft/violation_after_mean": (
            aggregate["violation_after_sum"] / constraints
        ),
        "local_soft/violation_after_max": aggregate["violation_after_max"],
        "local_soft/correction_norm": correction_norm,
        "local_soft/correction_to_synthetic_update_norm": (
            correction_norm / jnp.maximum(synthetic_norm, eps)
        ),
        "local_soft/synthetic_tail_count": jnp.asarray(
            synthetic_tail_count,
            dtype=jnp.float32,
        ),
        "local_soft/real_tail_count": jnp.asarray(
            max(1, math.ceil(len(anchor_group_grads) * tail_fraction)),
            dtype=jnp.float32,
        ),
        "local_soft/synthetic_score_mean": _tree_mean(synthetic_scores),
        "update_norm/raw": raw_norm,
        "update_norm/synthetic_masked": synthetic_norm,
        "update_norm/final_structured": final_norm,
        "update_norm/kept_fraction": final_norm / jnp.maximum(raw_norm, eps),
    }
    return projected, mask_ema, mask_ema_count, stats


def project_optax_updates_cached_source_balanced_circuit(
    updates: dict[str, jax.Array],
    synthetic_group_grads: tuple[dict[str, jax.Array], ...],
    anchor_group_grads: tuple[dict[str, jax.Array], ...],
    layouts: tuple[DualCircuitLayerLayout, ...],
    circuit: dict[str, jax.Array],
    *,
    tail_fraction: float,
    projection_passes: int = 1,
    minimum_kept_fraction: float = 0.25,
    eps: float = 1e-12,
) -> tuple[dict[str, jax.Array], dict[str, jax.Array]]:
    """Project against cached synthetic and current real constraints separately.

    Global source projections prevent cancellation outside the sparse circuit.
    Additional blockwise projections protect prompt-invariant FFN value vectors
    and attention output heads. No coordinate is selected by a real anchor and
    no elementwise mask is formed.
    """
    if not synthetic_group_grads or not anchor_group_grads:
        raise ValueError("cached source-balanced projection requires both sources")
    if not layouts:
        raise ValueError("cached source-balanced projection requires circuit layouts")
    if not 0 < tail_fraction <= 1:
        raise ValueError("source-balanced tail fraction must be in (0, 1]")
    if projection_passes <= 0 or eps <= 0:
        raise ValueError("projection passes and epsilon must be positive")
    if not 0 <= minimum_kept_fraction <= 1:
        raise ValueError("minimum kept fraction must be in [0, 1]")

    expected_mlp_shape = (len(layouts), layouts[0].intermediate_size)
    expected_attention_shape = (len(layouts), layouts[0].query_head_count)
    if tuple(circuit["mlp_weight"].shape) != expected_mlp_shape:
        raise ValueError("cached projector MLP circuit shape does not match layouts")
    if tuple(circuit["attention_weight"].shape) != expected_attention_shape:
        raise ValueError(
            "cached projector attention circuit shape does not match layouts"
        )

    raw_norm = tree_l2_norm(updates)
    projected, _ = _project_tree_against_source_tail(
        updates,
        synthetic_group_grads,
        tail_fraction=tail_fraction,
        projection_passes=projection_passes,
        eps=eps,
    )
    projected, _ = _project_tree_against_source_tail(
        projected,
        anchor_group_grads,
        tail_fraction=tail_fraction,
        projection_passes=projection_passes,
        eps=eps,
    )

    zero = jnp.asarray(0.0, dtype=jnp.float32)
    family_stats = {
        source: {
            family: {
                "selected_count": zero,
                "harmful_count": zero,
                "agreement_sum": zero,
                "violation_before_sum": zero,
                "violation_after_sum": zero,
                "active_constraint_count": zero,
                "violation_after_max": zero,
                "correction_energy": zero,
            }
            for family in ("mlp", "attention")
        }
        for source in ("synthetic", "real")
    }

    def add_block_stats(
        source: str,
        family: str,
        values: dict[str, jax.Array],
    ) -> None:
        target = family_stats[source][family]
        for key, value in values.items():
            if key == "violation_after_max":
                target[key] = jnp.maximum(target[key], value)
            else:
                target[key] = target[key] + value

    for layout in layouts:
        for family, path, axis, strength in (
            (
                "mlp",
                layout.down_path,
                layout.down_component_axis,
                circuit["mlp_weight"][layout.layer],
            ),
            (
                "attention",
                layout.o_path,
                layout.o_head_axis,
                circuit["attention_weight"][layout.layer],
            ),
        ):
            value, block_stats = _project_output_blocks_against_source_tail(
                projected[path],
                synthetic_group_grads,
                path,
                axis,
                strength,
                tail_fraction=tail_fraction,
                projection_passes=projection_passes,
                eps=eps,
            )
            projected[path] = value
            add_block_stats("synthetic", family, block_stats)
            value, block_stats = _project_output_blocks_against_source_tail(
                projected[path],
                anchor_group_grads,
                path,
                axis,
                strength,
                tail_fraction=tail_fraction,
                projection_passes=projection_passes,
                eps=eps,
            )
            projected[path] = value
            add_block_stats("real", family, block_stats)

    # Local projections can alter the global source dot products. Reapply the
    # two low-rank source constraints once; this is still independent of N.
    projected, synthetic_global_stats = _project_tree_against_source_tail(
        projected,
        synthetic_group_grads,
        tail_fraction=tail_fraction,
        projection_passes=projection_passes,
        eps=eps,
    )
    projected, real_global_stats = _project_tree_against_source_tail(
        projected,
        anchor_group_grads,
        tail_fraction=tail_fraction,
        projection_passes=projection_passes,
        eps=eps,
    )

    pre_floor_norm = tree_l2_norm(projected)
    desired_norm = minimum_kept_fraction * raw_norm
    floor_scale = jnp.where(
        (pre_floor_norm > eps) & (pre_floor_norm < desired_norm),
        desired_norm / jnp.maximum(pre_floor_norm, eps),
        1.0,
    )
    projected = dict(
        jax.tree_util.tree_map(
            lambda value: value * jnp.asarray(floor_scale, dtype=value.dtype),
            projected,
        )
    )
    final_norm = tree_l2_norm(projected)
    correction_norm = tree_l2_norm(
        jax.tree_util.tree_map(lambda raw, final: raw - final, updates, projected)
    )

    stats: dict[str, jax.Array] = {
        "cached_projector/synthetic_tail_count": synthetic_global_stats[
            "tail_count"
        ],
        "cached_projector/real_tail_count": real_global_stats["tail_count"],
        "cached_projector/synthetic_global_violation_before_mean": (
            synthetic_global_stats["violation_before_mean"]
        ),
        "cached_projector/synthetic_global_violation_after_mean": (
            synthetic_global_stats["violation_after_mean"]
        ),
        "cached_projector/synthetic_global_violation_after_max": (
            synthetic_global_stats["violation_after_max"]
        ),
        "cached_projector/real_global_violation_before_mean": real_global_stats[
            "violation_before_mean"
        ],
        "cached_projector/real_global_violation_after_mean": real_global_stats[
            "violation_after_mean"
        ],
        "cached_projector/real_global_violation_after_max": real_global_stats[
            "violation_after_max"
        ],
        "cached_projector/correction_norm": correction_norm,
        "cached_projector/minimum_kept_fraction": jnp.asarray(
            minimum_kept_fraction,
            dtype=jnp.float32,
        ),
        "cached_projector/norm_floor_scale": floor_scale,
        "update_norm/raw": raw_norm,
        "update_norm/final_structured": final_norm,
        "update_norm/kept_fraction": final_norm / jnp.maximum(raw_norm, eps),
    }
    for source in ("synthetic", "real"):
        for family in ("mlp", "attention"):
            values = family_stats[source][family]
            selected = jnp.maximum(values["selected_count"], 1.0)
            constraints = jnp.maximum(values["active_constraint_count"], 1.0)
            prefix = f"cached_projector/{source}_{family}"
            stats[f"{prefix}_selected_count"] = values["selected_count"]
            stats[f"{prefix}_harmful_fraction"] = (
                values["harmful_count"] / selected
            )
            stats[f"{prefix}_violation_before_mean"] = (
                values["violation_before_sum"] / constraints
            )
            stats[f"{prefix}_violation_after_mean"] = (
                values["violation_after_sum"] / constraints
            )
            stats[f"{prefix}_violation_after_max"] = values[
                "violation_after_max"
            ]
            stats[f"{prefix}_correction_norm"] = jnp.sqrt(
                jnp.maximum(values["correction_energy"], 0.0)
            )
    return projected, stats


def mask_optax_updates_dual_circuit_projection_ema(
    updates: dict[str, jax.Array],
    synthetic_group_grads: tuple[dict[str, jax.Array], ...],
    anchor_group_grads: tuple[dict[str, jax.Array], ...],
    layouts: tuple[DualCircuitLayerLayout, ...],
    circuit: dict[str, jax.Array],
    mask_ema: dict[str, jax.Array],
    correction_ema: dict[str, jax.Array],
    mask_ema_count: jax.Array,
    *,
    tail_fraction: float,
    beta: float,
    consensus_threshold: float = 0.75,
    projection_passes: int = 1,
    mlp_value_strength: float = 1.0,
    mlp_address_strength: float = 0.5,
    attention_vo_strength: float = 1.0,
    attention_qk_strength: float = 0.25,
    enable_attention_qk: bool = False,
    hard_veto: bool = False,
    normmatched_control: bool = False,
    eps: float = 1e-12,
) -> tuple[
    dict[str, jax.Array],
    dict[str, jax.Array],
    dict[str, jax.Array],
    jax.Array,
    dict[str, jax.Array],
]:
    """Apply prompt-consistent, update-aware component projection.

    Synthetic CVaR defines the allowed coordinate support. Real anchors only
    project harmful directions inside protected circuit components.
    """
    if not synthetic_group_grads or not anchor_group_grads:
        raise ValueError("dual-circuit projection requires synthetic and anchor groups")
    if not 0 < consensus_threshold <= 1:
        raise ValueError("circuit anchor consensus threshold must be in (0, 1]")
    if projection_passes <= 0 or eps <= 0:
        raise ValueError("circuit projection passes and epsilon must be positive")
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
    effective_masks, mask_ema, mask_ema_count = _update_mask_ema(
        synthetic_masks,
        mask_ema,
        mask_ema_count,
        beta=beta,
    )
    # The instantaneous hard synthetic support is reapplied after EMA and after
    # every projection, so neither EMA nor a dense anchor gradient can revive a
    # coordinate rejected by the current synthetic proposal.
    proposal_masks = jax.tree_util.tree_map(
        lambda effective, hard: effective * hard,
        effective_masks,
        synthetic_masks,
    )
    synthetic_update = dict(_apply_mask(updates, proposal_masks))
    projected = dict(synthetic_update)

    aggregate = {
        "protected_count": jnp.asarray(0.0, dtype=jnp.float32),
        "harmful_count": jnp.asarray(0.0, dtype=jnp.float32),
        "consensus_harmful_count": jnp.asarray(0.0, dtype=jnp.float32),
        "before_violation_sum": jnp.asarray(0.0, dtype=jnp.float32),
        "after_violation_sum": jnp.asarray(0.0, dtype=jnp.float32),
        "after_violation_max": jnp.asarray(0.0, dtype=jnp.float32),
        "correction_energy": jnp.asarray(0.0, dtype=jnp.float32),
        "final_energy": jnp.asarray(0.0, dtype=jnp.float32),
        "core_correction_energy": jnp.asarray(0.0, dtype=jnp.float32),
        "synthetic_only_correction_energy": jnp.asarray(0.0, dtype=jnp.float32),
        "real_residual_correction_energy": jnp.asarray(0.0, dtype=jnp.float32),
    }
    family = {
        "mlp": {key: jnp.asarray(0.0, dtype=jnp.float32) for key in aggregate},
        "attention": {
            key: jnp.asarray(0.0, dtype=jnp.float32) for key in aggregate
        },
    }

    def add_stats(target: dict[str, jax.Array], values: dict[str, jax.Array]) -> None:
        for key, value in values.items():
            if key == "after_violation_max":
                target[key] = jnp.maximum(target[key], value)
            else:
                target[key] = target[key] + value

    for layout in layouts:
        mlp_weight = jnp.asarray(circuit["mlp_weight"][layout.layer])
        mlp_tier = jnp.asarray(circuit["mlp_tier"][layout.layer])
        for path, axis, slice_strength in (
            (layout.gate_path, layout.gate_component_axis, mlp_address_strength),
            (layout.up_path, layout.up_component_axis, mlp_address_strength),
            (layout.down_path, layout.down_component_axis, mlp_value_strength),
        ):
            value, correction_value, block_stats = _project_component_blocks(
                projected[path],
                synthetic_masks[path],
                correction_ema[path],
                mask_ema_count,
                anchor_group_grads,
                path,
                axis,
                mlp_weight * slice_strength,
                mlp_tier,
                consensus_threshold=consensus_threshold,
                projection_passes=projection_passes,
                hard_veto=hard_veto,
                correction_beta=beta,
                eps=eps,
            )
            projected[path] = value
            correction_ema[path] = correction_value
            add_stats(family["mlp"], block_stats)
            add_stats(aggregate, block_stats)

        attention_weight = jnp.asarray(circuit["attention_weight"][layout.layer])
        attention_tier = jnp.asarray(circuit["attention_tier"][layout.layer])
        if "attention_kv_weight" in circuit:
            kv_weight = jnp.asarray(circuit["attention_kv_weight"][layout.layer])
            kv_tier = jnp.asarray(circuit["attention_kv_tier"][layout.layer])
        else:
            kv_weight, kv_tier = _kv_component_values(
                attention_weight,
                attention_tier,
                layout.query_to_kv,
                layout.kv_head_count,
            )
        attention_blocks = [
            (
                layout.v_path,
                layout.v_head_axis,
                kv_weight * attention_vo_strength,
                kv_tier,
            ),
            (
                layout.o_path,
                layout.o_head_axis,
                attention_weight * attention_vo_strength,
                attention_tier,
            ),
        ]
        if enable_attention_qk:
            attention_blocks.extend(
                [
                    (
                        layout.q_path,
                        layout.q_head_axis,
                        attention_weight * attention_qk_strength,
                        attention_tier,
                    ),
                    (
                        layout.k_path,
                        layout.k_head_axis,
                        kv_weight * attention_qk_strength,
                        kv_tier,
                    ),
                ]
            )
        for path, axis, strength, tiers in attention_blocks:
            value, correction_value, block_stats = _project_component_blocks(
                projected[path],
                synthetic_masks[path],
                correction_ema[path],
                mask_ema_count,
                anchor_group_grads,
                path,
                axis,
                strength,
                tiers,
                consensus_threshold=consensus_threshold,
                projection_passes=projection_passes,
                hard_veto=hard_veto,
                correction_beta=beta,
                eps=eps,
            )
            projected[path] = value
            correction_ema[path] = correction_value
            add_stats(family["attention"], block_stats)
            add_stats(aggregate, block_stats)

    projected = dict(_apply_mask(projected, synthetic_masks))
    target_norm = tree_l2_norm(projected)
    normmatched_norm = jnp.asarray(0.0, dtype=jnp.float32)
    if normmatched_control:
        element_masks = jax.tree_util.tree_map(
            lambda synthetic, anchor: synthetic * anchor,
            synthetic_masks,
            anchor_masks,
        )
        element_update = dict(_apply_mask(updates, element_masks))
        element_norm = tree_l2_norm(element_update)
        scale = target_norm / jnp.maximum(element_norm, eps)
        projected = {
            path: value * jnp.asarray(scale, dtype=value.dtype)
            for path, value in element_update.items()
        }
        normmatched_norm = tree_l2_norm(projected)

    protected_denominator = jnp.maximum(aggregate["protected_count"], 1.0)
    correction_norm = jnp.sqrt(jnp.maximum(aggregate["correction_energy"], 0.0))
    raw_norm = tree_l2_norm(updates)
    final_norm = tree_l2_norm(projected)
    stats = {
        "mask/synthetic_mean": _tree_mean(synthetic_masks),
        "mask/element_anchor_mean": _tree_mean(anchor_masks),
        "mask/final_effective_mean": _tree_mean(proposal_masks),
        "circuit/mlp_harmful_fraction": family["mlp"]["harmful_count"]
        / jnp.maximum(family["mlp"]["protected_count"], 1.0),
        "circuit/attention_harmful_fraction": family["attention"]["harmful_count"]
        / jnp.maximum(family["attention"]["protected_count"], 1.0),
        "circuit/anchor_consensus_harmful_fraction": aggregate[
            "consensus_harmful_count"
        ]
        / protected_denominator,
        "circuit/mlp_projected_update_norm": jnp.sqrt(
            jnp.maximum(family["mlp"]["final_energy"], 0.0)
        ),
        "circuit/attention_projected_update_norm": jnp.sqrt(
            jnp.maximum(family["attention"]["final_energy"], 0.0)
        ),
        "circuit/total_correction_norm": correction_norm,
        "circuit/correction_to_update_norm_ratio": correction_norm
        / jnp.maximum(tree_l2_norm(synthetic_update), eps),
        "circuit/core_correction_norm": jnp.sqrt(
            jnp.maximum(aggregate["core_correction_energy"], 0.0)
        ),
        "circuit/synth_only_correction_norm": jnp.sqrt(
            jnp.maximum(aggregate["synthetic_only_correction_energy"], 0.0)
        ),
        "circuit/real_residual_correction_norm": jnp.sqrt(
            jnp.maximum(aggregate["real_residual_correction_energy"], 0.0)
        ),
        "circuit/projection_violation_before_mean": aggregate[
            "before_violation_sum"
        ]
        / protected_denominator,
        "circuit/projection_violation_after_mean": aggregate[
            "after_violation_sum"
        ]
        / protected_denominator,
        "circuit/projection_violation_after_max": aggregate["after_violation_max"],
        "circuit/synthetic_tail_count": jnp.asarray(
            synthetic_tail_count, dtype=jnp.float32
        ),
        "circuit/anchor_tail_count": jnp.asarray(anchor_tail_count, dtype=jnp.float32),
        "circuit/synthetic_score_mean": _tree_mean(synthetic_scores),
        "circuit/anchor_score_mean": _tree_mean(anchor_scores),
        "update_norm/raw": raw_norm,
        "update_norm/synthetic_masked": tree_l2_norm(synthetic_update),
        "update_norm/circuit_projected": final_norm,
        "update_norm/normmatched_control": normmatched_norm,
        "update_norm/kept_fraction": final_norm / jnp.maximum(raw_norm, eps),
    }
    return projected, mask_ema, correction_ema, mask_ema_count, stats
