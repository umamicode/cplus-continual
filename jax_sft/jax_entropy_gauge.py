"""Function-preserving SwiGLU channel gauges for the JAX Qwen backend."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, MutableMapping

import jax
import numpy as np


GAUGE_MODES = ("none", "resuscitate", "permuted", "inverse")
_UP_PATTERN = re.compile(r"^(model\.layers\.(\d+)\.mlp)\.up_proj\.weight$")


@dataclass(frozen=True)
class EntropyGaugeConfig:
    mode: str = "none"
    p_percent: float = 50.0
    q: float = 1.0
    eps: float = 1e-8
    max_scale: float = 4.0
    seed: int = 0
    source: str = "balanced_synth_real"
    token_scope: str = "all_nonpad"
    verify: bool = True
    verify_atol: float = 5e-5
    verify_rtol: float = 1e-5
    power_of_two: bool = True

    def validate(self) -> None:
        if self.mode not in GAUGE_MODES:
            raise ValueError(f"unsupported entropy gauge mode: {self.mode!r}")
        if not 0.0 < self.p_percent <= 100.0:
            raise ValueError("entropy gauge p_percent must be in (0, 100]")
        if self.q <= 0 or self.eps <= 0:
            raise ValueError("entropy gauge q and eps must be positive")
        if self.max_scale < 1:
            raise ValueError("entropy gauge max_scale must be at least 1")
        if self.power_of_two and not math.isclose(
            math.log2(self.max_scale),
            round(math.log2(self.max_scale)),
            abs_tol=1e-12,
        ):
            raise ValueError(
                "power-of-two entropy gauges require max_scale to be a power of two"
            )
        if self.source != "balanced_synth_real":
            raise ValueError(
                "the first entropy-gauge experiment supports only "
                "source=balanced_synth_real"
            )
        if self.token_scope != "all_nonpad":
            raise ValueError(
                "the first JAX entropy-gauge experiment supports only "
                "token_scope=all_nonpad"
            )
        if self.verify_atol < 0 or self.verify_rtol < 0:
            raise ValueError("entropy gauge verification tolerances must be nonnegative")


@dataclass(frozen=True)
class GaugeVerification:
    probe_max_abs_logit_diff: float
    probe_max_rel_logit_diff: float
    probe_loss_abs_diff: float
    prediction_changes: int
    allclose: bool


def entropy_metrics(activity: np.ndarray, eps: float = 1e-8) -> dict[str, float]:
    values = np.asarray(activity, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("activity must be a non-empty vector")
    if not np.all(np.isfinite(values)) or np.any(values < 0) or values.sum() <= 0:
        raise ValueError("activity must be finite, nonnegative, and have positive mass")
    probability = values / max(float(values.sum()), eps)
    entropy = -float(np.sum(probability * np.log(np.maximum(probability, eps))))
    top_count = max(1, int(math.ceil(values.size * 0.10)))
    normalizer = math.log(values.size) if values.size > 1 else 1.0
    return {
        "entropy": entropy,
        "normalized_entropy": entropy / normalizer if values.size > 1 else 0.0,
        "effective_channels": math.exp(entropy),
        "top10_mass": float(np.sort(probability)[-top_count:].sum()),
    }


def _power_of_two_scales(scales: np.ndarray, max_scale: float) -> np.ndarray:
    minimum = 1.0 / max_scale
    quantized = np.exp2(np.rint(np.log2(np.asarray(scales, dtype=np.float64))))
    return np.clip(quantized, minimum, max_scale).astype(np.float32)


def build_base_scales(
    activity: np.ndarray,
    p_percent: float,
    q: float,
    eps: float,
    max_scale: float,
    *,
    power_of_two: bool,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    values = np.asarray(activity, dtype=np.float32)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("activity must be a non-empty vector")
    if not np.all(np.isfinite(values)) or np.any(values < 0) or values.sum() <= 0:
        raise ValueError("activity must be finite, nonnegative, and have positive mass")
    if not 0.0 < p_percent <= 100.0:
        raise ValueError("p_percent must be in (0, 100]")
    if q <= 0 or eps <= 0 or max_scale < 1:
        raise ValueError("q and eps must be positive and max_scale must be at least 1")

    threshold = float(np.quantile(values, p_percent / 100.0))
    low = values <= threshold
    proposed = q * float(values.mean()) / np.maximum(values, eps)
    unclipped = np.ones_like(values)
    unclipped[low] = proposed[low]
    minimum = 1.0 / max_scale
    clipped = np.clip(unclipped, minimum, max_scale).astype(np.float32)
    clipped_mask = (unclipped < minimum) | (unclipped > max_scale)
    selected = (
        _power_of_two_scales(clipped, max_scale)
        if power_of_two
        else clipped
    )
    return selected, low, {
        "threshold": threshold,
        "low_fraction": float(low.mean()),
        "clipped_fraction": float(clipped_mask.mean()),
        "power_of_two_changed_fraction": float(np.mean(selected != clipped)),
        "base_scale_mean": float(selected.mean()),
        "base_scale_min": float(selected.min()),
        "base_scale_max": float(selected.max()),
    }


def deterministic_layer_seed(seed: int, layer_name: str) -> int:
    digest = hashlib.sha256(f"{int(seed)}\0{layer_name}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little")


def build_mode_scales(
    base_scales: np.ndarray,
    mode: str,
    *,
    seed: int = 0,
    layer_name: str = "",
) -> np.ndarray:
    base = np.asarray(base_scales, dtype=np.float32)
    if mode not in GAUGE_MODES:
        raise ValueError(f"unsupported entropy gauge mode: {mode!r}")
    if base.ndim != 1 or base.size == 0:
        raise ValueError("base_scales must be a non-empty vector")
    if not np.all(np.isfinite(base)) or np.any(base <= 0):
        raise ValueError("base_scales must be finite and positive")
    if mode == "none":
        return np.ones_like(base)
    if mode == "resuscitate":
        return base.copy()
    if mode == "inverse":
        return np.reciprocal(base).astype(np.float32)
    rng = np.random.default_rng(deterministic_layer_seed(seed, layer_name))
    return base[rng.permutation(base.size)].copy()


def qwen3_mlp_layer_names(weights: Mapping[str, Any]) -> tuple[str, ...]:
    indexed: list[tuple[int, str]] = []
    for key in weights:
        match = _UP_PATTERN.match(key)
        if match:
            indexed.append((int(match.group(2)), match.group(1)))
    names = tuple(name for _, name in sorted(indexed))
    if not names:
        raise ValueError("no Qwen3 SwiGLU MLP weights were found")
    for name in names:
        required = {
            f"{name}.gate_proj.weight",
            f"{name}.up_proj.weight",
            f"{name}.down_proj.weight",
        }
        missing = required - set(weights)
        if missing:
            raise ValueError(f"incomplete Qwen3 MLP {name}: {sorted(missing)}")
    return names


def combine_source_activity(
    layer_names: tuple[str, ...],
    synthetic: np.ndarray,
    real: np.ndarray,
    synthetic_token_counts: np.ndarray,
    real_token_counts: np.ndarray,
) -> dict[str, dict[str, Any]]:
    synthetic = np.asarray(synthetic, dtype=np.float32)
    real = np.asarray(real, dtype=np.float32)
    if synthetic.shape != real.shape or synthetic.ndim != 2:
        raise ValueError("synthetic and real activities must have matching [layer, channel] shapes")
    if synthetic.shape[0] != len(layer_names):
        raise ValueError("activity layer count does not match the Qwen MLP layout")
    synth_counts = np.asarray(synthetic_token_counts, dtype=np.int64)
    real_counts = np.asarray(real_token_counts, dtype=np.int64)
    if synth_counts.shape != (len(layer_names),) or real_counts.shape != (len(layer_names),):
        raise ValueError("token counts must contain one value per layer")
    return {
        name: {
            "synthetic": synthetic[index],
            "real": real[index],
            "combined": 0.5 * synthetic[index] + 0.5 * real[index],
            "synthetic_token_count": int(synth_counts[index]),
            "real_token_count": int(real_counts[index]),
        }
        for index, name in enumerate(layer_names)
    }


def build_gauge_scales(
    activities: Mapping[str, Mapping[str, Any]],
    config: EntropyGaugeConfig,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    config.validate()
    scales: dict[str, np.ndarray] = {}
    low_masks: dict[str, np.ndarray] = {}
    layers: dict[str, Any] = {}
    for name, activity in activities.items():
        base, low, base_stats = build_base_scales(
            np.asarray(activity["combined"]),
            config.p_percent,
            config.q,
            config.eps,
            config.max_scale,
            power_of_two=config.power_of_two,
        )
        selected = build_mode_scales(
            base,
            config.mode,
            seed=config.seed,
            layer_name=name,
        )
        scales[name] = selected
        low_masks[name] = low
        before = {
            source: entropy_metrics(np.asarray(activity[source]), config.eps)
            for source in ("synthetic", "real", "combined")
        }
        after = {
            source: entropy_metrics(
                np.asarray(activity[source]) * selected,
                config.eps,
            )
            for source in ("synthetic", "real", "combined")
        }
        layers[name] = {
            **base_stats,
            "scale_mean": float(selected.mean()),
            "scale_min": float(selected.min()),
            "scale_max": float(selected.max()),
            "synthetic_token_count": int(activity["synthetic_token_count"]),
            "real_token_count": int(activity["real_token_count"]),
            "before": before,
            "after": after,
        }
    return scales, low_masks, {"config": asdict(config), "layers": layers}


def apply_qwen3_swiglu_gauge(
    weights: MutableMapping[str, Any],
    scales_by_layer: Mapping[str, np.ndarray],
) -> MutableMapping[str, Any]:
    names = qwen3_mlp_layer_names(weights)
    if set(names) != set(scales_by_layer):
        raise ValueError("gauge scales and Qwen3 MLP layers do not match")
    for name in names:
        up_key = f"{name}.up_proj.weight"
        down_key = f"{name}.down_proj.weight"
        up = weights[up_key]
        down = weights[down_key]
        scale = np.asarray(scales_by_layer[name], dtype=np.float32)
        if scale.shape != (up.shape[0],) or scale.shape != (down.shape[1],):
            raise ValueError(f"gauge scale shape mismatch for {name}")
        if not np.all(np.isfinite(scale)) or np.any(scale <= 0):
            raise ValueError(f"gauge scale for {name} must be finite and positive")

        up_scale = jax.device_put(scale[:, None], up.sharding)
        down_scale = jax.device_put(np.reciprocal(scale)[None, :], down.sharding)
        new_up = up * up_scale.astype(up.dtype)
        new_down = down * down_scale.astype(down.dtype)
        jax.block_until_ready((new_up, new_down))
        weights[up_key] = new_up
        weights[down_key] = new_down
        bias_key = f"{name}.up_proj.bias"
        if bias_key in weights:
            bias = weights[bias_key]
            bias_scale = jax.device_put(scale, bias.sharding)
            new_bias = bias * bias_scale.astype(bias.dtype)
            jax.block_until_ready(new_bias)
            weights[bias_key] = new_bias
    return weights


def verify_probe_outputs(
    before_logits: np.ndarray,
    after_logits: np.ndarray,
    before_loss: float,
    after_loss: float,
    *,
    atol: float,
    rtol: float,
    abort: bool = True,
) -> GaugeVerification:
    before = np.asarray(before_logits, dtype=np.float32)
    after = np.asarray(after_logits, dtype=np.float32)
    if before.shape != after.shape:
        raise ValueError("before and after probe logits have different shapes")
    difference = np.abs(after - before)
    max_abs = float(difference.max(initial=0.0))
    max_rel = float(
        np.max(difference / np.maximum(np.abs(before), 1e-6), initial=0.0)
    )
    prediction_changes = int(np.count_nonzero(after.argmax(-1) != before.argmax(-1)))
    allclose = bool(
        np.allclose(after, before, atol=atol, rtol=rtol)
        and math.isclose(after_loss, before_loss, abs_tol=atol, rel_tol=rtol)
    )
    result = GaugeVerification(
        probe_max_abs_logit_diff=max_abs,
        probe_max_rel_logit_diff=max_rel,
        probe_loss_abs_diff=abs(float(after_loss) - float(before_loss)),
        prediction_changes=prediction_changes,
        allclose=allclose,
    )
    if abort and (prediction_changes or not allclose):
        raise RuntimeError(
            "entropy gauge function-preservation check failed: "
            f"max_abs={max_abs:.6g} max_rel={max_rel:.6g} "
            f"loss_abs={result.probe_loss_abs_diff:.6g} "
            f"prediction_changes={prediction_changes}"
        )
    return result


def save_activity_cache(
    path: Path,
    layer_names: tuple[str, ...],
    synthetic: np.ndarray,
    real: np.ndarray,
    synthetic_token_counts: np.ndarray,
    real_token_counts: np.ndarray,
    metadata: Mapping[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as handle:
        np.savez_compressed(
            handle,
            format_version=np.asarray(1, dtype=np.int32),
            layer_names=np.asarray(layer_names),
            synthetic=np.asarray(synthetic, dtype=np.float32),
            real=np.asarray(real, dtype=np.float32),
            synthetic_token_counts=np.asarray(synthetic_token_counts, dtype=np.int64),
            real_token_counts=np.asarray(real_token_counts, dtype=np.int64),
            metadata=np.asarray(json.dumps(dict(metadata), sort_keys=True)),
        )


def load_activity_cache(
    path: Path,
    expected_metadata: Mapping[str, Any],
) -> tuple[tuple[str, ...], np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    with np.load(path, allow_pickle=False) as payload:
        metadata = json.loads(str(payload["metadata"].item()))
        mismatches = {
            key: (metadata.get(key), expected)
            for key, expected in expected_metadata.items()
            if metadata.get(key) != expected
        }
        if mismatches:
            raise ValueError(f"entropy-gauge activity cache identity mismatch: {mismatches}")
        return (
            tuple(str(value) for value in payload["layer_names"].tolist()),
            np.asarray(payload["synthetic"], dtype=np.float32),
            np.asarray(payload["real"], dtype=np.float32),
            np.asarray(payload["synthetic_token_counts"], dtype=np.int64),
            np.asarray(payload["real_token_counts"], dtype=np.int64),
            metadata,
        )


def save_scale_cache(
    path: Path,
    scales: Mapping[str, np.ndarray],
    low_masks: Mapping[str, np.ndarray],
    metadata: Mapping[str, Any],
) -> None:
    names = tuple(scales)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as handle:
        np.savez_compressed(
            handle,
            format_version=np.asarray(1, dtype=np.int32),
            layer_names=np.asarray(names),
            scales=np.stack([scales[name] for name in names]),
            low_masks=np.stack([low_masks[name] for name in names]),
            metadata=np.asarray(json.dumps(dict(metadata), sort_keys=True)),
        )


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
