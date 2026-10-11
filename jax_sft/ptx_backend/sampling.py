from __future__ import annotations

import inspect
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P, reshard


_CATEGORICAL_HAS_OUT_SHARDING = (
    "out_sharding" in inspect.signature(jax.random.categorical).parameters
)


class SamplingState(NamedTuple):
    step: int
    key: jax.Array
    tokens: jax.Array
    kv: jax.Array
    done: jax.Array


def _sample_step(state, forward, weights, tokenizer, temperature, eos_id):
    pad_id = tokenizer.pad_token_id
    key, key_sampling = jax.random.split(state.key)
    input_token = state.tokens[:, state.step, None]
    logits, kv = forward(input_token, weights, state.kv, state.step)
    next_token = state.tokens[:, state.step + 1]
    next_token_sharding = jax.typeof(next_token).sharding
    categorical_kwargs = (
        {"out_sharding": next_token_sharding}
        if _CATEGORICAL_HAS_OUT_SHARDING
        else {}
    )
    sampled_token = jax.random.categorical(
        key_sampling,
        logits[:, 0, :] / temperature,
        **categorical_kwargs,
    )
    if not _CATEGORICAL_HAS_OUT_SHARDING:
        sampled_token = reshard(
            sampled_token,
            next_token_sharding,
        )

    update_token = jnp.where((~state.done) & (next_token == pad_id), sampled_token, next_token)
    tokens = state.tokens.at[:, state.step + 1].set(update_token)
    done = state.done | ((next_token == pad_id) & (sampled_token == eos_id))
    return SamplingState(state.step + 1, key, tokens, kv, done)


def _greedy_step(state, forward, weights, tokenizer, eos_id):
    pad_id = tokenizer.pad_token_id
    input_token = state.tokens[:, state.step, None]
    logits, kv = forward(input_token, weights, state.kv, state.step)
    next_token = state.tokens[:, state.step + 1]
    next_token_sharding = jax.typeof(next_token).sharding
    selected_token = jnp.argmax(logits[:, 0, :], axis=-1).astype(next_token.dtype)
    selected_token = reshard(selected_token, next_token_sharding)
    use_selected = reshard(
        (~state.done) & (next_token == pad_id),
        next_token_sharding,
    )
    update_token = jnp.where(use_selected, selected_token, next_token)
    tokens = state.tokens.at[:, state.step + 1].set(update_token)
    done = state.done | ((next_token == pad_id) & (selected_token == eos_id))
    return SamplingState(state.step + 1, state.key, tokens, kv, done)


@partial(jax.jit, static_argnames=("forward", "init_kv", "tokenizer"))
def _sample(key, forward, init_kv, weights, tokenizer, tokens, temperature, eos_id):
    batch_size, seq_len = tokens.shape
    tokens = reshard(tokens, P("data", None))
    batch_sharding = jax.typeof(tokens[:, 0]).sharding
    state = SamplingState(
        step=0,
        key=key,
        tokens=tokens,
        kv=init_kv(batch_size, seq_len),
        done=jnp.zeros(
            [batch_size],
            dtype=bool,
            out_sharding=batch_sharding,
        ),
    )
    step_fn = lambda state: _sample_step(state, forward, weights, tokenizer, temperature, eos_id)
    cond_fn = lambda state: (state.step + 1 < seq_len) & jnp.any(~state.done)
    state = jax.lax.while_loop(cond_fn, step_fn, state)
    return state.tokens


@partial(jax.jit, static_argnames=("forward", "init_kv", "tokenizer"))
def _greedy(key, forward, init_kv, weights, tokenizer, tokens, eos_id):
    batch_size, seq_len = tokens.shape
    tokens = reshard(tokens, P("data", None))
    batch_sharding = jax.typeof(tokens[:, 0]).sharding
    state = SamplingState(
        step=0,
        key=key,
        tokens=tokens,
        kv=init_kv(batch_size, seq_len),
        done=jnp.zeros(
            [batch_size],
            dtype=bool,
            out_sharding=batch_sharding,
        ),
    )
    step_fn = lambda state: _greedy_step(
        state,
        forward,
        weights,
        tokenizer,
        eos_id,
    )
    cond_fn = lambda state: (state.step + 1 < seq_len) & jnp.any(~state.done)
    state = jax.lax.while_loop(cond_fn, step_fn, state)
    return state.tokens


def sample(key, model, tokens, temperature=1.0, weights=None, eos_id=None):
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if weights is None:
        weights = model.weights
    if eos_id is None:
        eos_id = model.tokenizer.eos_token_id or -1
    return _sample(key, model.forward, model.init_kv, weights, model.tokenizer, tokens, temperature, eos_id)


def greedy(key, model, tokens, temperature=1.0, weights=None, eos_id=None):
    del temperature
    if weights is None:
        weights = model.weights
    if eos_id is None:
        eos_id = model.tokenizer.eos_token_id or -1
    return _greedy(
        key,
        model.forward,
        model.init_kv,
        weights,
        model.tokenizer,
        tokens,
        eos_id,
    )
