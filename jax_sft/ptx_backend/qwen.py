import json
import math
from collections import defaultdict
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
from huggingface_hub import snapshot_download
from jax.sharding import AxisType, PartitionSpec as P, reshard
from safetensors import safe_open
from safetensors.numpy import save_file
from transformers import AutoTokenizer


@dataclass
class Model:
    weights: dict[str, Any]
    forward: Callable
    forward_with_memory_activity: Callable
    memory_coefficients: Callable
    activation_subspace_features: Callable
    forward_with_circuit_scales: Callable
    forward_with_circuit_components: Callable
    encode: Callable
    init_kv: Callable
    tokenizer: Any
    config: dict[str, Any]
    save: Callable


def apply_rope(x, theta, pos=0):
    B, T, N, H = x.shape
    positions = pos + jnp.broadcast_to(jnp.arange(T)[None, :], [B, T])
    freq = 1.0 / (theta ** (jnp.arange(0, H, 2, dtype=jnp.float32) / H))
    inp = jnp.einsum("bt,h->bth", positions, freq, precision=jax.lax.Precision.HIGHEST)
    sin, cos = jnp.sin(inp).astype(x.dtype), jnp.cos(inp).astype(x.dtype)
    x1, x2 = x[:, :, :, :H // 2], x[:, :, :, H // 2:]
    sin, cos = sin[:, :, None, :], cos[:, :, None, :]
    return jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)


def rms_norm(x, gamma, eps, axis=-1):
    rms = jnp.sqrt(jnp.mean(x.astype(jnp.float32) ** 2, axis=axis, keepdims=True) + eps)
    return (gamma * x / rms).astype(x.dtype)


def mean_abs_memory_activity(coefficients, attention_mask=None):
    """Aggregate SwiGLU coefficient activity while excluding padding tokens."""
    coefficients = jnp.abs(coefficients).astype(jnp.float32)
    if attention_mask is None:
        return jnp.mean(coefficients, axis=(0, 1))
    mask = jnp.asarray(attention_mask, dtype=jnp.float32)[..., None]
    return jnp.sum(coefficients * mask, axis=(0, 1)) / jnp.maximum(mask.sum(), 1.0)


def coefficients_at_positions(coefficients, positions):
    """Select one SwiGLU coefficient vector per sequence."""
    positions = jnp.asarray(positions, dtype=jnp.int32).reshape((-1,))
    positions = reshard(positions, P("data"))
    return coefficients.at[jnp.arange(coefficients.shape[0]), positions, :].get(
        out_sharding=P("data", "model")
    )


def attention_at_positions(attention_outputs, positions):
    """Select one pre-output-projection head tensor per sequence."""
    positions = jnp.asarray(positions, dtype=jnp.int32).reshape((-1,))
    positions = reshard(positions, P("data"))
    return attention_outputs.at[
        jnp.arange(attention_outputs.shape[0]), positions, :, :
    ].get(out_sharding=P("data", "model", None))


def hidden_at_positions(hidden_states, positions):
    """Select one hidden-state vector per sequence."""
    positions = jnp.asarray(positions, dtype=jnp.int32).reshape((-1,))
    positions = reshard(positions, P("data"))
    return hidden_states.at[
        jnp.arange(hidden_states.shape[0]), positions, :
    ].get(out_sharding=P("data", None))


def per_example_abs_memory_activity(coefficients, attention_mask=None):
    """Return one padding-aware mean absolute coefficient vector per example."""
    coefficients = jnp.abs(coefficients).astype(jnp.float32)
    if attention_mask is None:
        return jnp.mean(coefficients, axis=1)
    mask = jnp.asarray(attention_mask, dtype=jnp.float32)[..., None]
    return jnp.sum(coefficients * mask, axis=1) / jnp.maximum(
        jnp.sum(mask, axis=1),
        1.0,
    )


def _forward_layer_core(
    cfg,
    x,
    w,
    kv=None,
    pos=0,
    activity_mask=None,
    coefficient_positions=None,
    mlp_component_scale=None,
    attention_component_scale=None,
    component_positions=None,
):
    x_norm = rms_norm(x, w["input_layernorm.weight"], cfg["rms_norm_eps"])

    q = jnp.einsum("btd,nhd->btnh", x_norm, w["self_attn.q_proj.weight"], preferred_element_type=x.dtype, out_sharding=P("data", None, "model", None))
    k = jnp.einsum("bsd,khd->bskh", x_norm, w["self_attn.k_proj.weight"], preferred_element_type=x.dtype, out_sharding=P("data", None, "model", None))
    v = jnp.einsum("bsd,khd->bskh", x_norm, w["self_attn.v_proj.weight"], preferred_element_type=x.dtype, out_sharding=P("data", None, "model", None))

    if "self_attn.q_proj.bias" in w:
        q += w["self_attn.q_proj.bias"]
        k += w["self_attn.k_proj.bias"]
        v += w["self_attn.v_proj.bias"]

    if "self_attn.q_norm.weight" in w:
        q = rms_norm(q, w["self_attn.q_norm.weight"], cfg["rms_norm_eps"])
        k = rms_norm(k, w["self_attn.k_norm.weight"], cfg["rms_norm_eps"])

    q = apply_rope(q, cfg["rope_theta"], pos)
    k = apply_rope(k, cfg["rope_theta"], pos)

    if kv is not None:
        kv = jax.lax.dynamic_update_slice(kv, jnp.stack([k, v]), (0, 0, pos, 0, 0))
        k, v = kv

    attn_mask = jnp.tri(q.shape[1], dtype=bool)[None] if kv is None else (jnp.arange(k.shape[1]) <= pos)[None, None]
    attn_out = jax.nn.dot_product_attention(q, k, v, mask=attn_mask)
    if attention_component_scale is not None:
        head_scale = jnp.asarray(attention_component_scale, dtype=attn_out.dtype)
        if component_positions is None:
            attn_out = attn_out * head_scale[:, None, :, None]
        else:
            position_mask = jax.nn.one_hot(
                jnp.asarray(component_positions, dtype=jnp.int32),
                attn_out.shape[1],
                dtype=attn_out.dtype,
            )[:, :, None]
            token_head_scale = 1.0 + position_mask * (
                head_scale[:, None, :] - 1.0
            )
            attn_out = attn_out * token_head_scale[..., None]
    selected_attention = None
    attention_component_magnitude = None
    if component_positions is not None:
        selected_attention = attention_at_positions(attn_out, component_positions)
        attention_component_magnitude = jnp.sqrt(
            jnp.sum(
                jnp.square(selected_attention.astype(jnp.float32)),
                axis=-1,
            )
        )
    x += jnp.einsum(
        "btnh,dnh->btd",
        attn_out,
        w["self_attn.o_proj.weight"],
        preferred_element_type=x.dtype,
        out_sharding=P("data", None, None),
    )

    x_norm = rms_norm(x, w["post_attention_layernorm.weight"], cfg["rms_norm_eps"])
    feature_positions = (
        coefficient_positions
        if coefficient_positions is not None
        else component_positions
    )
    selected_address_input = (
        hidden_at_positions(x_norm, feature_positions)
        if feature_positions is not None
        else None
    )
    gate = jax.nn.silu(jnp.einsum("btd,fd->btf", x_norm, w["mlp.gate_proj.weight"], preferred_element_type=jnp.float32, out_sharding=P("data", None, "model")))
    up = jnp.einsum("btd,fd->btf", x_norm, w["mlp.up_proj.weight"], preferred_element_type=x.dtype, out_sharding=P("data", None, "model"))
    coefficients = gate * up
    if mlp_component_scale is not None:
        coefficients = coefficients * jnp.asarray(
            mlp_component_scale,
            dtype=coefficients.dtype,
        )[:, None, :]
    x += jnp.einsum("btf,df->btd", coefficients, w["mlp.down_proj.weight"], preferred_element_type=x.dtype, out_sharding=P("data", None, None))
    activity = mean_abs_memory_activity(coefficients, activity_mask)
    selected_coefficients = (
        coefficients_at_positions(coefficients, coefficient_positions)
        if coefficient_positions is not None
        else None
    )
    mlp_component_magnitude = (
        per_example_abs_memory_activity(coefficients, activity_mask)
        if component_positions is not None
        else None
    )
    return (
        x,
        kv,
        activity,
        selected_coefficients,
        mlp_component_magnitude,
        attention_component_magnitude,
        selected_address_input,
        selected_attention,
    )


def forward_layer(cfg, x, w, kv=None, pos=0):
    x, kv, _, _, _, _, _, _ = _forward_layer_core(cfg, x, w, kv, pos)
    return x, kv


def forward(cfg, x, weights, kv=None, pos=0):
    weights = jax.tree.map(lambda w: w.astype(jnp.bfloat16) if w.ndim > 1 else w, weights)
    x = reshard(x, P("data", None))
    x = weights["model.embed_tokens.weight"].at[x, :].get(out_sharding=P("data", None, None)).astype(jnp.bfloat16)

    return_kv = kv is not None
    if kv is None: kv = defaultdict(lambda: None)
    for i in range(cfg["num_hidden_layers"]):
        layer_weights = {k.replace(prefix, ""): v for k, v in weights.items() if (prefix := f"model.layers.{i}.") in k}
        x, kv[i] = jax.remat(partial(forward_layer, cfg))(x, layer_weights, kv[i], pos)

    out_embed = weights["model.embed_tokens.weight"] if cfg["tie_word_embeddings"] else weights["lm_head.weight"]
    x = rms_norm(x, weights["model.norm.weight"], cfg["rms_norm_eps"])
    logits = jnp.einsum("btd,vd->btv", x, out_embed, preferred_element_type=x.dtype, out_sharding=P("data", None, "model"))
    return (logits, kv) if return_kv else logits


def forward_with_memory_activity(cfg, x, weights, attention_mask=None):
    """Return normal logits and padding-masked mean absolute SwiGLU coefficients."""
    weights = jax.tree.map(lambda w: w.astype(jnp.bfloat16) if w.ndim > 1 else w, weights)
    x = reshard(x, P("data", None))
    if attention_mask is not None:
        attention_mask = reshard(attention_mask, P("data", None))
    x = weights["model.embed_tokens.weight"].at[x, :].get(
        out_sharding=P("data", None, None)
    ).astype(jnp.bfloat16)

    activities = []
    for i in range(cfg["num_hidden_layers"]):
        prefix = f"model.layers.{i}."
        layer_weights = {k.replace(prefix, ""): v for k, v in weights.items() if prefix in k}
        x, _, activity, _, _, _, _, _ = jax.remat(partial(_forward_layer_core, cfg))(
            x,
            layer_weights,
            None,
            0,
            attention_mask,
        )
        activities.append(activity)

    out_embed = weights["model.embed_tokens.weight"] if cfg["tie_word_embeddings"] else weights["lm_head.weight"]
    x = rms_norm(x, weights["model.norm.weight"], cfg["rms_norm_eps"])
    logits = jnp.einsum("btd,vd->btv", x, out_embed, preferred_element_type=x.dtype, out_sharding=P("data", None, "model"))
    return logits, jnp.stack(activities, axis=0)


def memory_coefficients(cfg, x, weights, positions):
    """Return signed decision-token SwiGLU coefficients for every layer."""
    weights = jax.tree.map(lambda w: w.astype(jnp.bfloat16) if w.ndim > 1 else w, weights)
    x = reshard(x, P("data", None))
    positions = reshard(jnp.asarray(positions, dtype=jnp.int32), P("data"))
    x = weights["model.embed_tokens.weight"].at[x, :].get(
        out_sharding=P("data", None, None)
    ).astype(jnp.bfloat16)

    selected_by_layer = []
    for i in range(cfg["num_hidden_layers"]):
        prefix = f"model.layers.{i}."
        layer_weights = {k.replace(prefix, ""): v for k, v in weights.items() if prefix in k}
        x, _, _, selected, _, _, _, _ = jax.remat(partial(_forward_layer_core, cfg))(
            x,
            layer_weights,
            None,
            0,
            None,
            positions,
        )
        selected_by_layer.append(selected.astype(jnp.float32))
    return jnp.stack(selected_by_layer, axis=0)


def activation_subspace_features(cfg, x, weights, positions):
    """Return decision-token inputs to gate/up, down, and attention O maps."""
    weights = jax.tree.map(
        lambda w: w.astype(jnp.bfloat16) if w.ndim > 1 else w,
        weights,
    )
    x = reshard(x, P("data", None))
    positions = reshard(jnp.asarray(positions, dtype=jnp.int32), P("data"))
    x = weights["model.embed_tokens.weight"].at[x, :].get(
        out_sharding=P("data", None, None)
    ).astype(jnp.bfloat16)

    address_by_layer = []
    value_by_layer = []
    attention_by_layer = []
    for i in range(cfg["num_hidden_layers"]):
        prefix = f"model.layers.{i}."
        layer_weights = {
            k.replace(prefix, ""): v for k, v in weights.items() if prefix in k
        }
        (
            x,
            _,
            _,
            selected_value,
            _,
            _,
            selected_address,
            selected_attention,
        ) = jax.remat(partial(_forward_layer_core, cfg))(
            x,
            layer_weights,
            None,
            0,
            None,
            positions,
            None,
            None,
            positions,
        )
        address_by_layer.append(selected_address.astype(jnp.float32))
        value_by_layer.append(selected_value.astype(jnp.float32))
        attention_by_layer.append(
            selected_attention.reshape((selected_attention.shape[0], -1)).astype(
                jnp.float32
            )
        )
    return (
        jnp.stack(address_by_layer, axis=0),
        jnp.stack(value_by_layer, axis=0),
        jnp.stack(attention_by_layer, axis=0),
    )


def forward_with_circuit_scales(
    cfg,
    x,
    weights,
    positions,
    mlp_scales,
    attention_scales,
):
    """Forward with differentiable MLP-channel and answer-head multipliers.

    A derivative with respect to a multiplier at one is exactly the summed
    gradient-times-activation attribution for that component. MLP scales apply
    across all prompt tokens; attention scales apply only at ``positions``.
    """
    weights = jax.tree.map(
        lambda w: w.astype(jnp.bfloat16) if w.ndim > 1 else w,
        weights,
    )
    x = reshard(x, P("data", None))
    positions = reshard(jnp.asarray(positions, dtype=jnp.int32), P("data"))
    mlp_scales = reshard(
        jnp.asarray(mlp_scales, dtype=jnp.float32),
        P("data", None, "model"),
    )
    attention_scales = reshard(
        jnp.asarray(attention_scales, dtype=jnp.float32),
        P("data", None, "model"),
    )
    x = weights["model.embed_tokens.weight"].at[x, :].get(
        out_sharding=P("data", None, None)
    ).astype(jnp.bfloat16)

    for i in range(cfg["num_hidden_layers"]):
        prefix = f"model.layers.{i}."
        layer_weights = {
            k.replace(prefix, ""): v for k, v in weights.items() if prefix in k
        }
        layer_mlp_scale = mlp_scales.at[:, i, :].get(
            out_sharding=P("data", "model")
        )
        layer_attention_scale = attention_scales.at[:, i, :].get(
            out_sharding=P("data", "model")
        )
        x, _, _, _, _, _, _, _ = jax.remat(partial(_forward_layer_core, cfg))(
            x,
            layer_weights,
            None,
            0,
            None,
            None,
            layer_mlp_scale,
            layer_attention_scale,
            positions,
        )

    out_embed = (
        weights["model.embed_tokens.weight"]
        if cfg["tie_word_embeddings"]
        else weights["lm_head.weight"]
    )
    x = rms_norm(x, weights["model.norm.weight"], cfg["rms_norm_eps"])
    return jnp.einsum(
        "btd,vd->btv",
        x,
        out_embed,
        preferred_element_type=x.dtype,
        out_sharding=P("data", None, "model"),
    )


def forward_with_circuit_components(
    cfg,
    x,
    weights,
    attention_mask,
    positions,
):
    """Return logits and compact activation-only component controls."""
    weights = jax.tree.map(
        lambda w: w.astype(jnp.bfloat16) if w.ndim > 1 else w,
        weights,
    )
    x = reshard(x, P("data", None))
    attention_mask = reshard(attention_mask, P("data", None))
    positions = reshard(jnp.asarray(positions, dtype=jnp.int32), P("data"))
    x = weights["model.embed_tokens.weight"].at[x, :].get(
        out_sharding=P("data", None, None)
    ).astype(jnp.bfloat16)
    mlp_by_layer = []
    attention_by_layer = []
    for i in range(cfg["num_hidden_layers"]):
        prefix = f"model.layers.{i}."
        layer_weights = {
            k.replace(prefix, ""): v for k, v in weights.items() if prefix in k
        }
        x, _, _, _, mlp_magnitude, attention_magnitude, _, _ = jax.remat(
            partial(_forward_layer_core, cfg)
        )(
            x,
            layer_weights,
            None,
            0,
            attention_mask,
            None,
            None,
            None,
            positions,
        )
        mlp_by_layer.append(mlp_magnitude)
        attention_by_layer.append(attention_magnitude)

    out_embed = (
        weights["model.embed_tokens.weight"]
        if cfg["tie_word_embeddings"]
        else weights["lm_head.weight"]
    )
    x = rms_norm(x, weights["model.norm.weight"], cfg["rms_norm_eps"])
    logits = jnp.einsum(
        "btd,vd->btv",
        x,
        out_embed,
        preferred_element_type=x.dtype,
        out_sharding=P("data", None, "model"),
    )
    return (
        logits,
        jnp.stack(mlp_by_layer, axis=1),
        jnp.stack(attention_by_layer, axis=1),
    )


def encode(cfg, x, weights):
    """Return final normalized hidden states for frozen semantic embeddings."""
    weights = jax.tree.map(lambda w: w.astype(jnp.bfloat16) if w.ndim > 1 else w, weights)
    x = reshard(x, P("data", None))
    x = weights["model.embed_tokens.weight"].at[x, :].get(
        out_sharding=P("data", None, None)
    ).astype(jnp.bfloat16)
    for i in range(cfg["num_hidden_layers"]):
        prefix = f"model.layers.{i}."
        layer_weights = {k.replace(prefix, ""): v for k, v in weights.items() if prefix in k}
        x, _ = jax.remat(partial(forward_layer, cfg))(x, layer_weights, None, 0)
    return rms_norm(x, weights["model.norm.weight"], cfg["rms_norm_eps"])


def init_kv(L, K, H, B, T):
    sharding = P(None, "data", None, "model", None)
    return [jnp.zeros((2, B, T, K, H), dtype=jnp.bfloat16, out_sharding=sharding) for _ in range(L)]


def _head_dim(cfg):
    return cfg.get("head_dim", cfg["hidden_size"] // cfg["num_attention_heads"])


def _load_template(model_id, hf_ckpt_dir, allow_patterns=None):
    source = Path(model_id).expanduser()
    if not source.exists():
        source = Path(hf_ckpt_dir).expanduser() / model_id
        snapshot_download(repo_id=model_id, local_dir=source, allow_patterns=allow_patterns)
    cfg = json.loads((source / "config.json").read_text())
    if cfg.get("model_type") != "qwen3":
        raise ValueError(f"Only dense Qwen3 checkpoints are supported, got model_type={cfg.get('model_type')}")
    return source, AutoTokenizer.from_pretrained(source), cfg


def _make_random_cfg(template_cfg, hidden_size=None, num_hidden_layers=None):
    hidden_size = template_cfg["hidden_size"] if hidden_size is None else int(hidden_size)
    num_hidden_layers = template_cfg["num_hidden_layers"] if num_hidden_layers is None else int(num_hidden_layers)
    if hidden_size < 128 or hidden_size % 128 != 0:
        raise ValueError(f"hidden_size must be a positive multiple of 128, got {hidden_size}")
    if num_hidden_layers < 1:
        raise ValueError(f"num_hidden_layers must be >= 1, got {num_hidden_layers}")

    cfg = dict(template_cfg)
    cfg["hidden_size"] = hidden_size
    cfg["num_hidden_layers"] = num_hidden_layers
    cfg["num_attention_heads"] = hidden_size // 128
    cfg["num_key_value_heads"] = math.gcd(cfg["num_attention_heads"], 8)
    cfg["intermediate_size"] = 3 * hidden_size
    if "max_window_layers" in cfg: cfg["max_window_layers"] = min(cfg["max_window_layers"], num_hidden_layers)
    if "layer_types" in cfg: cfg["layer_types"] = ["full_attention"] * num_hidden_layers
    return cfg


def _get_sharding(key, dp_shard=False):
    if "norm" in key or "bias" in key: return P()
    if any(k in key for k in ("q_proj", "k_proj", "v_proj")):
        return P("model", None, "data") if dp_shard else P("model", None, None)
    if "o_proj" in key:
        return P("data", "model", None) if dp_shard else P(None, "model", None)
    if any(k in key for k in ("gate_proj", "up_proj", "embed_tokens", "lm_head")):
        return P("model", "data") if dp_shard else P("model")
    if "down_proj" in key:
        return P("data", "model") if dp_shard else P(None, "model")
    raise ValueError(f"Unrecognized key: {key}")


def _load_weights(model_ckpt_dir, cfg, dp_shard=False):
    N, K, D, H = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["hidden_size"], _head_dim(cfg)
    weights = {}
    for file in sorted(model_ckpt_dir.glob("*.safetensors")):
        with safe_open(file, framework="numpy") as f:
            for key in f.keys():
                value = f.get_tensor(key)
                if "q_proj.bias" in key: value = value.reshape([N, H])
                if "k_proj.bias" in key or "v_proj.bias" in key: value = value.reshape([K, H])
                if "q_proj.weight" in key: value = value.reshape([N, H, D])
                if "k_proj.weight" in key or "v_proj.weight" in key: value = value.reshape([K, H, D])
                if "o_proj.weight" in key: value = value.reshape([D, N, H])
                weights[key] = jax.device_put(value, _get_sharding(key, dp_shard))
    if not weights:
        raise ValueError(f"No safetensors files found in {model_ckpt_dir}")
    return weights


def _random_weights(cfg, key, dp_shard=False):
    std = cfg["initializer_range"]
    D, F, N, K, H, V = cfg["hidden_size"], cfg["intermediate_size"], cfg["num_attention_heads"], cfg["num_key_value_heads"], _head_dim(cfg), cfg["vocab_size"]
    weights = {}

    def add(name, shape, ones=False):
        nonlocal key
        if ones:
            value = jnp.ones(shape, dtype=jnp.bfloat16)
        else:
            key, subkey = jax.random.split(key)
            value = (jax.random.normal(subkey, shape, dtype=jnp.float32) * std).astype(jnp.bfloat16)
        weights[name] = jax.device_put(value, _get_sharding(name, dp_shard))

    add("model.embed_tokens.weight", (V, D))
    add("model.norm.weight", (D,), ones=True)
    if not cfg["tie_word_embeddings"]: add("lm_head.weight", (V, D))

    specs = [
        ("input_layernorm.weight", (D,), True),
        ("post_attention_layernorm.weight", (D,), True),
        ("self_attn.q_norm.weight", (H,), True),
        ("self_attn.k_norm.weight", (H,), True),
        ("self_attn.q_proj.weight", (N, H, D), False),
        ("self_attn.k_proj.weight", (K, H, D), False),
        ("self_attn.v_proj.weight", (K, H, D), False),
        ("self_attn.o_proj.weight", (D, N, H), False),
        ("mlp.gate_proj.weight", (F, D), False),
        ("mlp.up_proj.weight", (F, D), False),
        ("mlp.down_proj.weight", (D, F), False),
    ]
    for i in range(cfg["num_hidden_layers"]):
        prefix = f"model.layers.{i}."
        for name, shape, ones in specs: add(prefix + name, shape, ones)
    return weights


def load(model_id="Qwen/Qwen3-0.6B-Base", hf_ckpt_dir="~/weights", tp_size=1, dp_shard=False, init="pretrained", hidden_size=None, num_hidden_layers=None, key=None):
    if jax.device_count() % tp_size != 0:
        raise ValueError(f"tp_size={tp_size} does not divide device_count={jax.device_count()}")
    mesh = jax.make_mesh((jax.device_count() // tp_size, tp_size), ("data", "model"), axis_types=(AxisType.Explicit, AxisType.Explicit))
    jax.set_mesh(mesh)
    has_architecture_override = hidden_size is not None or num_hidden_layers is not None

    if init == "pretrained":
        if has_architecture_override:
            raise ValueError("hidden_size and num_hidden_layers overrides are only supported when init='random'")
        model_ckpt_dir, tokenizer, cfg = _load_template(model_id, hf_ckpt_dir)
        return Model(
            _load_weights(model_ckpt_dir, cfg, dp_shard),
            partial(forward, cfg),
            partial(forward_with_memory_activity, cfg),
            partial(memory_coefficients, cfg),
            partial(activation_subspace_features, cfg),
            partial(forward_with_circuit_scales, cfg),
            partial(forward_with_circuit_components, cfg),
            partial(encode, cfg),
            partial(init_kv, cfg["num_hidden_layers"], cfg["num_key_value_heads"], _head_dim(cfg)),
            tokenizer,
            cfg,
            save,
        )

    if init == "random":
        if key is None:
            raise ValueError("Random initialization requires a JAX PRNG key")
        _, tokenizer, template_cfg = _load_template(model_id, hf_ckpt_dir, ["*.json", "*.txt", "*.jinja"])
        cfg = _make_random_cfg(template_cfg, hidden_size=hidden_size, num_hidden_layers=num_hidden_layers)
        return Model(
            _random_weights(cfg, key, dp_shard),
            partial(forward, cfg),
            partial(forward_with_memory_activity, cfg),
            partial(memory_coefficients, cfg),
            partial(activation_subspace_features, cfg),
            partial(forward_with_circuit_scales, cfg),
            partial(forward_with_circuit_components, cfg),
            partial(encode, cfg),
            partial(init_kv, cfg["num_hidden_layers"], cfg["num_key_value_heads"], _head_dim(cfg)),
            tokenizer,
            cfg,
            save,
        )

    raise ValueError(f"Unknown init mode: {init}")


def save(model, output_dir, *, dtype="bfloat16"):
    if dtype not in {"bfloat16", "float32"}:
        raise ValueError("save dtype must be 'bfloat16' or 'float32'")
    save_dtype = jnp.bfloat16 if dtype == "bfloat16" else jnp.float32
    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    hf = {}
    for k, v in model.weights.items():
        v = np.asarray(jax.device_get(jnp.asarray(v, dtype=save_dtype)))
        if "q_proj" in k or "k_proj" in k or "v_proj" in k: v = v.reshape(-1, v.shape[-1]) if v.ndim == 3 else v.reshape(-1)
        if "o_proj.weight" in k: v = v.reshape(v.shape[0], -1)
        hf[k] = v

    save_file(hf, str(output_dir / "model.safetensors"))
    (output_dir / "config.json").write_text(json.dumps(model.config, indent=2) + "\n")
    model.tokenizer.save_pretrained(output_dir)
