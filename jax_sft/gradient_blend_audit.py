"""Host-side aggregation and plots for fixed synthetic/real gradient blends."""

from __future__ import annotations

import csv
import html
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

try:
    from jax_sft.gradient_blend_schema import FIXED_BLEND_TENSOR_STAT_FIELDS
except ModuleNotFoundError as exc:
    if exc.name != "jax_sft":
        raise
    from gradient_blend_schema import FIXED_BLEND_TENSOR_STAT_FIELDS


_LAYER_PATTERN = re.compile(r"^model\.layers\.(\d+)\.(.+)$")


def parameter_location(name: str) -> tuple[str, int | None, str]:
    """Return a readable layer label, numeric layer, and tensor family."""
    match = _LAYER_PATTERN.match(name)
    if match:
        layer_index = int(match.group(1))
        suffix = match.group(2)
        family_rules = (
            ("self_attn.q_proj", "attn.q"),
            ("self_attn.k_proj", "attn.k"),
            ("self_attn.v_proj", "attn.v"),
            ("self_attn.o_proj", "attn.o"),
            ("mlp.gate_proj", "mlp.gate"),
            ("mlp.up_proj", "mlp.up"),
            ("mlp.down_proj", "mlp.down"),
            ("input_layernorm", "norm.input"),
            ("post_attention_layernorm", "norm.post_attn"),
        )
        family = next(
            (label for prefix, label in family_rules if suffix.startswith(prefix)),
            suffix.rsplit(".weight", 1)[0],
        )
        return f"L{layer_index:02d}", layer_index, family
    if name.startswith("model.embed_tokens"):
        return "Embedding", None, "embedding"
    if name.startswith("model.norm"):
        return "Final norm", None, "norm.final"
    if name.startswith("lm_head"):
        return "LM head", None, "lm_head"
    return "Other", None, name.rsplit(".weight", 1)[0]


def build_parameter_layer_groups(
    parameters: Mapping[str, Any],
) -> tuple[dict[str, int], tuple[str, ...], tuple[float, ...]]:
    """Assign every flat parameter tensor to one stable model-layer group."""
    if not parameters:
        raise ValueError("layer grouping requires a non-empty parameter mapping")

    locations = {
        name: parameter_location(name)[:2] for name in parameters
    }

    def sort_key(item: tuple[str, int | None]) -> tuple[int, int, str]:
        label, layer_index = item
        if label == "Embedding":
            return (0, -1, label)
        if layer_index is not None:
            return (1, layer_index, label)
        if label == "Final norm":
            return (2, 0, label)
        if label == "LM head":
            return (3, 0, label)
        return (4, 0, label)

    ordered_locations = sorted(set(locations.values()), key=sort_key)
    labels = tuple(label for label, _ in ordered_locations)
    label_to_group = {label: index for index, label in enumerate(labels)}
    parameter_counts = [0.0] * len(labels)
    for name, parameter in parameters.items():
        label, _ = locations[name]
        parameter_counts[label_to_group[label]] += float(parameter.size)
    return (
        {name: label_to_group[label] for name, (label, _) in locations.items()},
        labels,
        tuple(parameter_counts),
    )


def _safe_ratio(numerator: float, denominator: float) -> float:
    if abs(denominator) <= 1e-30:
        return 0.0
    return float(numerator / denominator)


def _bounded_percent(value: float) -> float:
    return 100.0 * min(1.0, max(0.0, value))


def _gradient_rms(source_sq: float, parameter_count: float) -> float:
    return math.sqrt(max(_safe_ratio(source_sq, parameter_count), 0.0))


def _gradient_scale_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 1e-30:
        return 1.0 if numerator <= 1e-30 else 2.0**20
    return numerator / denominator


def _derived_row(group: Mapping[str, float], **labels: Any) -> dict[str, Any]:
    parameter_count = group["parameter_count"]
    update_energy = group["update_energy"]
    active_count = group["source_active_count"]
    weighted_source_l1 = (
        group["weighted_synthetic_l1"] + group["weighted_real_l1"]
    )
    weighted_update_l1 = (
        group["weighted_synthetic_update_l1"]
        + group["weighted_real_update_l1"]
    )
    cosine_denom = math.sqrt(
        max(group["synthetic_source_sq"], 0.0)
        * max(group["real_source_sq"], 0.0)
    )
    synthetic_gradient_rms = _gradient_rms(
        group["synthetic_source_sq"], parameter_count
    )
    real_gradient_rms = _gradient_rms(
        group["real_source_sq"], parameter_count
    )
    blended_gradient_rms = _gradient_rms(
        group["blended_source_sq"], parameter_count
    )
    real_to_synthetic_rms_ratio = _gradient_scale_ratio(
        real_gradient_rms,
        synthetic_gradient_rms,
    )
    return {
        **labels,
        "parameter_count": parameter_count,
        "update_energy": update_energy,
        "synthetic_mask_allow_pct": 100.0
        * _safe_ratio(group["synthetic_allow_count"], parameter_count),
        "real_mask_allow_pct": 100.0
        * _safe_ratio(group["real_allow_count"], parameter_count),
        "blended_mask_allow_pct": 100.0
        * _safe_ratio(group["blended_allow_count"], parameter_count),
        "synthetic_update_energy_kept_pct": 100.0
        * _safe_ratio(group["synthetic_kept_energy"], update_energy),
        "real_update_energy_kept_pct": 100.0
        * _safe_ratio(group["real_kept_energy"], update_energy),
        "blended_update_energy_kept_pct": 100.0
        * _safe_ratio(group["blended_kept_energy"], update_energy),
        "source_sign_agreement_pct": 100.0
        * _safe_ratio(group["source_agreement_count"], active_count),
        "source_sign_conflict_pct": 100.0
        * _safe_ratio(group["source_conflict_count"], active_count),
        "synthetic_to_blend_flip_pct": 100.0
        * _safe_ratio(group["synthetic_blend_flip_count"], parameter_count),
        "real_to_blend_flip_pct": 100.0
        * _safe_ratio(group["real_blend_flip_count"], parameter_count),
        "synthetic_to_blend_flip_energy_pct": 100.0
        * _safe_ratio(group["synthetic_blend_flip_energy"], update_energy),
        "real_to_blend_flip_energy_pct": 100.0
        * _safe_ratio(group["real_blend_flip_energy"], update_energy),
        "source_cosine": _safe_ratio(group["source_dot"], cosine_denom),
        "synthetic_gradient_rms": synthetic_gradient_rms,
        "real_gradient_rms": real_gradient_rms,
        "blended_gradient_rms": blended_gradient_rms,
        "real_to_synthetic_gradient_rms_ratio": (
            real_to_synthetic_rms_ratio
        ),
        "real_to_synthetic_gradient_log2_ratio": math.log2(
            min(
                max(real_to_synthetic_rms_ratio, 2.0**-20),
                2.0**20,
            )
        ),
        "raw_gradient_cancellation_pct": _bounded_percent(
            1.0 - _safe_ratio(group["blended_l1"], weighted_source_l1)
        ),
        "update_weighted_cancellation_pct": _bounded_percent(
            1.0
            - _safe_ratio(group["blended_update_l1"], weighted_update_l1)
        ),
        "effective_real_update_influence_pct": 100.0
        * _safe_ratio(
            group["weighted_real_update_l1"], weighted_update_l1
        ),
        "synthetic_predicted_damage_kept_pct": 100.0
        * _safe_ratio(
            group["synthetic_damage_after_blend"],
            group["synthetic_damage_before"],
        ),
        "real_predicted_damage_kept_pct": 100.0
        * _safe_ratio(
            group["real_damage_after_blend"],
            group["real_damage_before"],
        ),
    }


def aggregate_tensor_statistics(
    tensor_statistics: Mapping[str, Sequence[float]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Build exact additive tensor, layer, and layer-family summaries."""
    field_count = len(FIXED_BLEND_TENSOR_STAT_FIELDS)
    tensor_rows: list[dict[str, Any]] = []
    layer_totals: dict[str, dict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    family_totals: dict[tuple[str, str], dict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    layer_indices: dict[str, int | None] = {}

    for name, raw_values in sorted(tensor_statistics.items()):
        values = np.asarray(raw_values, dtype=np.float64).reshape(-1)
        if len(values) != field_count:
            raise ValueError(
                f"{name}: expected {field_count} blend statistics, "
                f"received {len(values)}"
            )
        totals = dict(zip(FIXED_BLEND_TENSOR_STAT_FIELDS, values.tolist()))
        layer, layer_index, family = parameter_location(name)
        tensor_rows.append(
            _derived_row(
                totals,
                parameter=name,
                layer=layer,
                layer_index=layer_index,
                family=family,
            )
        )
        layer_indices[layer] = layer_index
        for field, value in totals.items():
            layer_totals[layer][field] += value
            family_totals[(layer, family)][field] += value

    def sort_key(label: str) -> tuple[int, int, str]:
        index = layer_indices.get(label)
        if label == "Embedding":
            return (0, -1, label)
        if index is not None:
            return (1, index, label)
        if label == "Final norm":
            return (2, 0, label)
        if label == "LM head":
            return (3, 0, label)
        return (4, 0, label)

    layer_rows = [
        _derived_row(
            layer_totals[layer],
            layer=layer,
            layer_index=layer_indices.get(layer),
        )
        for layer in sorted(layer_totals, key=sort_key)
    ]
    family_rows = [
        _derived_row(
            family_totals[(layer, family)],
            layer=layer,
            layer_index=layer_indices.get(layer),
            family=family,
        )
        for layer, family in sorted(
            family_totals,
            key=lambda item: (sort_key(item[0]), item[1]),
        )
    ]
    return tensor_rows, layer_rows, family_rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _color(value: float, minimum: float, maximum: float) -> str:
    ratio = 0.5 if maximum <= minimum else (value - minimum) / (maximum - minimum)
    ratio = min(1.0, max(0.0, ratio))
    low = (247, 249, 252)
    high = (15, 118, 110)
    rgb = tuple(round(a + ratio * (b - a)) for a, b in zip(low, high))
    return f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"


def _text_color(value: float, minimum: float, maximum: float) -> str:
    ratio = 0.5 if maximum <= minimum else (value - minimum) / (maximum - minimum)
    return "#ffffff" if ratio > 0.62 else "#17202a"


def render_layer_heatmap_svg(
    rows: Sequence[Mapping[str, Any]],
    path: Path,
) -> None:
    metrics = (
        ("synthetic_mask_allow_pct", "Synth allow", 0.0, 100.0),
        ("real_mask_allow_pct", "Real allow", 0.0, 100.0),
        ("blended_mask_allow_pct", "Blend allow", 0.0, 100.0),
        (
            "blended_update_energy_kept_pct",
            "Blend energy",
            0.0,
            100.0,
        ),
        ("source_sign_conflict_pct", "Source conflict", 0.0, 100.0),
        (
            "update_weighted_cancellation_pct",
            "Cancellation",
            0.0,
            100.0,
        ),
        (
            "effective_real_update_influence_pct",
            "Real influence",
            0.0,
            100.0,
        ),
        ("source_cosine", "Source cosine", -1.0, 1.0),
        (
            "real_to_synthetic_gradient_log2_ratio",
            "log2 R/S RMS",
            -4.0,
            4.0,
        ),
    )
    cell_w, cell_h = 116, 28
    left, top = 118, 82
    width = left + len(metrics) * cell_w + 24
    height = top + len(rows) * cell_h + 54
    pieces = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="20" y="28" font-family="Arial,sans-serif" '
        'font-size="18" font-weight="700" fill="#17202a">'
        "Fixed blend by layer</text>",
        '<text x="20" y="50" font-family="Arial,sans-serif" '
        'font-size="12" fill="#566573">Percent; darker means more</text>',
    ]
    for column, (_, label, _, _) in enumerate(metrics):
        x = left + column * cell_w + cell_w / 2
        pieces.append(
            f'<text x="{x:.1f}" y="72" text-anchor="middle" '
            'font-family="Arial,sans-serif" font-size="11" '
            f'fill="#34495e">{html.escape(label)}</text>'
        )
    for row_index, row in enumerate(rows):
        y = top + row_index * cell_h
        pieces.append(
            f'<text x="{left - 10}" y="{y + 19}" text-anchor="end" '
            'font-family="Arial,sans-serif" font-size="11" '
            f'fill="#34495e">{html.escape(str(row["layer"]))}</text>'
        )
        for column, (key, _, minimum, maximum) in enumerate(metrics):
            value = float(row[key])
            x = left + column * cell_w
            pieces.extend(
                (
                    f'<rect x="{x}" y="{y}" width="{cell_w - 2}" '
                    f'height="{cell_h - 2}" rx="2" fill="{_color(value, minimum, maximum)}"/>',
                    f'<text x="{x + (cell_w - 2) / 2:.1f}" y="{y + 18}" '
                    'text-anchor="middle" font-family="Arial,sans-serif" '
                    f'font-size="11" fill="{_text_color(value, minimum, maximum)}">'
                    f"{value:.1f}</text>",
                )
            )
    pieces.append(
        f'<text x="{left}" y="{height - 18}" font-family="Arial,sans-serif" '
        'font-size="11" fill="#566573">Conservative: low Blend energy. '
        "Plastic: high Blend energy.</text>"
    )
    pieces.append("</svg>")
    path.write_text("\n".join(pieces) + "\n", encoding="utf-8")


def _format_table_value(key: str, value: Any) -> str:
    if not isinstance(value, float):
        return str(value)
    if key.endswith("_gradient_rms"):
        return f"{value:.3e}"
    return f"{value:.2f}"


def _table(rows: Iterable[Mapping[str, Any]], columns: Sequence[tuple[str, str]]) -> str:
    body = []
    for row in rows:
        cells = []
        for key, _ in columns:
            value = row[key]
            rendered = _format_table_value(key, value)
            cells.append(f"<td>{html.escape(rendered)}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    headings = "".join(f"<th>{html.escape(label)}</th>" for _, label in columns)
    return f"<table><thead><tr>{headings}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def write_gradient_blend_audit(
    output_dir: str | Path,
    tensor_statistics: Mapping[str, Sequence[float]],
    *,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Write machine-readable tables and a self-contained audit report."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tensor_rows, layer_rows, family_rows = aggregate_tensor_statistics(
        tensor_statistics
    )
    tensor_path = output_dir / "gradient_blend_tensor_audit.csv"
    layer_path = output_dir / "gradient_blend_layer_audit.csv"
    family_path = output_dir / "gradient_blend_layer_family_audit.csv"
    svg_path = output_dir / "gradient_blend_layer_heatmap.svg"
    html_path = output_dir / "gradient_blend_layer_audit.html"
    metadata_path = output_dir / "gradient_blend_layer_audit.json"
    _write_csv(tensor_path, tensor_rows)
    _write_csv(layer_path, layer_rows)
    _write_csv(family_path, family_rows)
    render_layer_heatmap_svg(layer_rows, svg_path)

    transformer_rows = [row for row in layer_rows if row["layer_index"] is not None]
    most_conservative = min(
        transformer_rows,
        key=lambda row: row["blended_update_energy_kept_pct"],
    )
    most_plastic = max(
        transformer_rows,
        key=lambda row: row["blended_update_energy_kept_pct"],
    )
    overall_totals = defaultdict(float)
    for values in tensor_statistics.values():
        for field, value in zip(FIXED_BLEND_TENSOR_STAT_FIELDS, values):
            overall_totals[field] += float(value)
    overall = _derived_row(overall_totals, layer="All parameters")
    summary = {
        "metadata": dict(metadata),
        "overall": overall,
        "most_conservative_transformer_layer": most_conservative["layer"],
        "most_conservative_blended_update_energy_kept_pct": (
            most_conservative["blended_update_energy_kept_pct"]
        ),
        "most_plastic_transformer_layer": most_plastic["layer"],
        "most_plastic_blended_update_energy_kept_pct": (
            most_plastic["blended_update_energy_kept_pct"]
        ),
        "artifacts": {
            "tensor_csv": str(tensor_path),
            "layer_csv": str(layer_path),
            "layer_family_csv": str(family_path),
            "heatmap_svg": str(svg_path),
            "report_html": str(html_path),
        },
    }
    metadata_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    columns = (
        ("layer", "Layer"),
        ("synthetic_mask_allow_pct", "Synth allow %"),
        ("real_mask_allow_pct", "Real allow %"),
        ("blended_mask_allow_pct", "Blend allow %"),
        ("blended_update_energy_kept_pct", "Blend energy %"),
        ("source_sign_conflict_pct", "Conflict %"),
        ("update_weighted_cancellation_pct", "Cancellation %"),
        ("effective_real_update_influence_pct", "Real influence %"),
        ("source_cosine", "Source cosine"),
        ("synthetic_gradient_rms", "Synth grad RMS"),
        ("real_gradient_rms", "Real grad RMS"),
        ("blended_gradient_rms", "Blend grad RMS"),
        (
            "real_to_synthetic_gradient_log2_ratio",
            "log2 real/synth RMS",
        ),
    )
    report = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Gradient Blend Layer Audit</title>
<style>
body{{font:14px Arial,sans-serif;color:#17202a;margin:32px;max-width:1320px}}h1,h2{{margin:0 0 12px}}p{{line-height:1.5;color:#455a64}}img{{max-width:100%;height:auto;border:1px solid #dfe6e9}}table{{border-collapse:collapse;width:100%;font-size:12px}}th,td{{padding:7px 8px;border-bottom:1px solid #e5e7eb;text-align:right;white-space:nowrap}}th:first-child,td:first-child{{text-align:left}}th{{position:sticky;top:0;background:#f7f9fb}}.table-wrap{{overflow-x:auto}}.grid{{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin:18px 0}}.stat{{border-left:4px solid #0f766e;padding:10px 14px;background:#f7faf9}}code{{background:#f3f4f6;padding:2px 4px}}</style>
</head><body>
<h1>Gradient Blend Layer Audit</h1>
<p>This is a frozen-state audit of the current-task update against the synthetic, real, and blended preservation gradients. Low blend update energy means a layer is more constrained; high blend update energy means it retains more capacity to learn.</p>
<div class="grid"><div class="stat"><strong>Most conservative</strong><br>{html.escape(str(most_conservative['layer']))}: {most_conservative['blended_update_energy_kept_pct']:.2f}% update energy retained</div><div class="stat"><strong>Most plastic</strong><br>{html.escape(str(most_plastic['layer']))}: {most_plastic['blended_update_energy_kept_pct']:.2f}% update energy retained</div></div>
<img src="{html.escape(svg_path.name)}" alt="Layer heatmap">
<h2>Layer table</h2><div class="table-wrap">{_table(layer_rows, columns)}</div>
<h2>How to read it</h2>
<p><strong>Allow</strong> is the fraction of parameter coordinates permitted by a hard mask. <strong>Blend energy</strong> is the fraction of squared current-task update magnitude that actually survives, which is the better plasticity measure. <strong>Conflict</strong> measures opposite signs between synthetic and real preservation gradients. <strong>Cancellation</strong> measures how much weighted gradient magnitude disappears in the mixture. <strong>Real influence</strong> measures the real source's share of update-weighted preservation pressure.</p>
</body></html>"""
    html_path.write_text(report, encoding="utf-8")
    return summary


def _read_audit_csv(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            row: dict[str, Any] = {}
            for key, value in raw.items():
                if key in {"parameter", "layer", "family"}:
                    row[key] = value
                elif value in {None, ""}:
                    row[key] = None
                else:
                    row[key] = float(value)
            rows.append(row)
    return rows


def load_gradient_blend_audit(summary_path: str | Path) -> dict[str, Any]:
    """Load one detailed audit and its layer table."""
    summary_path = Path(summary_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    anchor_count = int(summary["metadata"]["anchor_count"])
    layer_path = summary_path.parent / "gradient_blend_layer_audit.csv"
    return {
        "anchor_count": anchor_count,
        "summary_path": str(summary_path),
        "summary": summary,
        "overall": summary["overall"],
        "layers": _read_audit_csv(layer_path),
    }


def write_gradient_blend_trajectory_index(
    summary_paths: Sequence[str | Path],
    output_path: str | Path,
) -> dict[str, Any]:
    """Index repeated layer audits from one training run by model state."""
    entries: list[dict[str, Any]] = []
    seen_epochs: set[int] = set()
    for summary_path in summary_paths:
        summary_path = Path(summary_path)
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        metadata = summary.get("metadata", {})
        audit_epoch = int(metadata.get("audit_epoch", 1))
        if audit_epoch in seen_epochs:
            raise ValueError(
                f"duplicate gradient-blend trajectory epoch: {audit_epoch}"
            )
        seen_epochs.add(audit_epoch)
        entries.append(
            {
                "audit_epoch": audit_epoch,
                "training_epochs_completed": int(
                    metadata.get("training_epochs_completed", audit_epoch - 1)
                ),
                "summary_path": str(summary_path),
                "overall": summary["overall"],
            }
        )
    entries.sort(key=lambda entry: entry["audit_epoch"])
    payload = {
        "audit_count": len(entries),
        "audit_epochs": [entry["audit_epoch"] for entry in entries],
        "training_epochs_completed": [
            entry["training_epochs_completed"] for entry in entries
        ],
        "entries": entries,
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def discover_gradient_blend_audits(root: str | Path) -> list[Path]:
    """Find per-run layer audits below a sweep root."""
    return sorted(
        Path(root).glob(
            "**/gradient_blend_layer_audit/gradient_blend_layer_audit.json"
        )
    )


def _interpolate_color(
    low: tuple[int, int, int],
    high: tuple[int, int, int],
    ratio: float,
) -> str:
    ratio = min(1.0, max(0.0, ratio))
    rgb = tuple(round(a + ratio * (b - a)) for a, b in zip(low, high))
    return f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"


def _diverging_color(value: float, minimum: float, maximum: float) -> str:
    midpoint = 0.0
    white = (248, 250, 252)
    if value <= midpoint:
        ratio = _safe_ratio(value - minimum, midpoint - minimum)
        return _interpolate_color((37, 99, 235), white, ratio)
    ratio = _safe_ratio(value - midpoint, maximum - midpoint)
    return _interpolate_color(white, (220, 38, 38), ratio)


def _cross_anchor_heatmap_svg(
    audits: Sequence[Mapping[str, Any]],
    path: Path,
    *,
    title: str,
    subtitle: str,
    metrics: Sequence[tuple[str, str, float, float, bool]],
) -> None:
    if not audits:
        raise ValueError("cross-anchor heatmap requires at least one audit")
    layer_names = [str(row["layer"]) for row in audits[0]["layers"]]
    for audit in audits[1:]:
        if [str(row["layer"]) for row in audit["layers"]] != layer_names:
            raise ValueError("cross-anchor audits have incompatible layer layouts")

    cell_w, cell_h = 82, 27
    left, top = 104, 102
    column_count = len(audits) * len(metrics)
    width = left + column_count * cell_w + 24
    height = top + len(layer_names) * cell_h + 50
    pieces = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="20" y="28" font-family="Arial,sans-serif" '
        'font-size="18" font-weight="700" fill="#17202a">'
        f"{html.escape(title)}</text>",
        '<text x="20" y="49" font-family="Arial,sans-serif" '
        'font-size="11" fill="#566573">'
        f"{html.escape(subtitle)}</text>",
    ]
    for audit_index, audit in enumerate(audits):
        group_x = left + audit_index * len(metrics) * cell_w
        group_width = len(metrics) * cell_w
        pieces.append(
            f'<text x="{group_x + group_width / 2:.1f}" y="70" '
            'text-anchor="middle" font-family="Arial,sans-serif" '
            'font-size="12" font-weight="700" fill="#17202a">'
            f'A={int(audit["anchor_count"])}</text>'
        )
        if audit_index:
            pieces.append(
                f'<line x1="{group_x}" y1="58" x2="{group_x}" '
                f'y2="{height - 35}" stroke="#cbd5e1" stroke-width="1"/>'
            )
        for metric_index, (_, label, _, _, _) in enumerate(metrics):
            x = group_x + metric_index * cell_w + cell_w / 2
            pieces.append(
                f'<text x="{x:.1f}" y="91" text-anchor="middle" '
                'font-family="Arial,sans-serif" font-size="10" '
                f'fill="#475569">{html.escape(label)}</text>'
            )

    for row_index, layer in enumerate(layer_names):
        y = top + row_index * cell_h
        pieces.append(
            f'<text x="{left - 9}" y="{y + 18}" text-anchor="end" '
            'font-family="Arial,sans-serif" font-size="10" '
            f'fill="#34495e">{html.escape(layer)}</text>'
        )
        for audit_index, audit in enumerate(audits):
            row = audit["layers"][row_index]
            for metric_index, (key, _, minimum, maximum, diverging) in enumerate(
                metrics
            ):
                value = float(row[key])
                x = left + (
                    audit_index * len(metrics) + metric_index
                ) * cell_w
                fill = (
                    _diverging_color(value, minimum, maximum)
                    if diverging
                    else _color(value, minimum, maximum)
                )
                normalized = _safe_ratio(value - minimum, maximum - minimum)
                text_fill = (
                    "#ffffff"
                    if normalized < 0.14 or normalized > 0.78
                    else "#17202a"
                )
                pieces.extend(
                    (
                        f'<rect x="{x}" y="{y}" width="{cell_w - 2}" '
                        f'height="{cell_h - 2}" rx="2" fill="{fill}"/>',
                        f'<text x="{x + (cell_w - 2) / 2:.1f}" '
                        f'y="{y + 17}" text-anchor="middle" '
                        'font-family="Arial,sans-serif" font-size="10" '
                        f'fill="{text_fill}">{value:.1f}</text>',
                    )
                )
    pieces.append("</svg>")
    path.write_text("\n".join(pieces) + "\n", encoding="utf-8")


def write_cross_anchor_gradient_blend_report(
    summary_paths: Sequence[str | Path],
    output_dir: str | Path,
) -> dict[str, Any]:
    """Compare frozen blend geometry across nested real-anchor counts."""
    audits = sorted(
        (load_gradient_blend_audit(path) for path in summary_paths),
        key=lambda audit: audit["anchor_count"],
    )
    if not audits:
        raise ValueError("no gradient blend audits were provided")
    anchor_counts = [int(audit["anchor_count"]) for audit in audits]
    if len(anchor_counts) != len(set(anchor_counts)):
        raise ValueError("cross-anchor report requires one audit per anchor count")

    bank_hash_sets: list[set[str]] = []
    for audit in audits:
        raw_hashes = audit["summary"]["metadata"].get(
            "synthetic_reference_sha256", {}
        )
        if isinstance(raw_hashes, Mapping):
            bank_hash_sets.append(
                {str(value) for value in raw_hashes.values() if value}
            )
        else:
            bank_hash_sets.append(set())
    known_bank_hashes = all(len(values) == 1 for values in bank_hash_sets)
    distinct_bank_hashes = sorted(set().union(*bank_hash_sets))
    if known_bank_hashes and len(distinct_bank_hashes) == 1:
        synthetic_bank_status = "verified_fixed"
        bank_note = (
            "All audits use the same verified synthetic bank "
            f"(SHA-256 {distinct_bank_hashes[0][:12]}...)."
        )
    elif known_bank_hashes:
        synthetic_bank_status = "verified_different"
        bank_note = (
            "The audits use different verified synthetic banks, so differences "
            "cannot be attributed solely to real-anchor count."
        )
    else:
        synthetic_bank_status = "unverified"
        bank_note = (
            "Synthetic-bank identity is unverified; inspect bank hashes before "
            "attributing differences solely to real-anchor count."
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mask_path = output_dir / "cross_anchor_mask_allow_heatmap.svg"
    geometry_path = output_dir / "cross_anchor_gradient_geometry_heatmap.svg"
    plasticity_path = output_dir / "cross_anchor_update_survival_heatmap.svg"
    html_path = output_dir / "cross_anchor_gradient_blend_report.html"
    json_path = output_dir / "cross_anchor_gradient_blend_report.json"

    _cross_anchor_heatmap_svg(
        audits,
        mask_path,
        title="Synthetic, real, and blended masks across anchor counts",
        subtitle="Percent of coordinates allowed for one fixed mean injection update",
        metrics=(
            ("synthetic_mask_allow_pct", "Synth", 0.0, 100.0, False),
            ("real_mask_allow_pct", "Real", 0.0, 100.0, False),
            ("blended_mask_allow_pct", "Blend", 0.0, 100.0, False),
        ),
    )
    _cross_anchor_heatmap_svg(
        audits,
        geometry_path,
        title="Synthetic and real preservation-gradient geometry",
        subtitle="Cosine and log2 scale ratio are centered at zero; other values are percent",
        metrics=(
            ("source_cosine", "Cosine", -1.0, 1.0, True),
            ("source_sign_conflict_pct", "Conflict", 0.0, 100.0, False),
            (
                "real_to_synthetic_gradient_log2_ratio",
                "log2 R/S",
                -4.0,
                4.0,
                True,
            ),
            (
                "update_weighted_cancellation_pct",
                "Cancel",
                0.0,
                100.0,
                False,
            ),
        ),
    )
    _cross_anchor_heatmap_svg(
        audits,
        plasticity_path,
        title="How much candidate-update energy survives each mask",
        subtitle="Higher values preserve more plasticity for the fixed injection update",
        metrics=(
            (
                "synthetic_update_energy_kept_pct",
                "Synth",
                0.0,
                100.0,
                False,
            ),
            ("real_update_energy_kept_pct", "Real", 0.0, 100.0, False),
            (
                "blended_update_energy_kept_pct",
                "Blend",
                0.0,
                100.0,
                False,
            ),
            (
                "effective_real_update_influence_pct",
                "Real infl.",
                0.0,
                100.0,
                False,
            ),
        ),
    )

    overall_rows = [
        {"anchor_count": audit["anchor_count"], **audit["overall"]}
        for audit in audits
    ]
    columns = (
        ("anchor_count", "A"),
        ("synthetic_mask_allow_pct", "Synth allow %"),
        ("real_mask_allow_pct", "Real allow %"),
        ("blended_mask_allow_pct", "Blend allow %"),
        ("blended_update_energy_kept_pct", "Blend energy %"),
        ("source_sign_conflict_pct", "Conflict %"),
        ("source_cosine", "Cosine"),
        ("update_weighted_cancellation_pct", "Cancellation %"),
        ("effective_real_update_influence_pct", "Real influence %"),
        (
            "real_to_synthetic_gradient_log2_ratio",
            "log2 real/synth RMS",
        ),
    )
    report = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Cross-Anchor Gradient Blend Audit</title>
<style>
body{{font:14px Arial,sans-serif;color:#17202a;margin:32px;max-width:1480px}}h1,h2{{margin:0 0 12px}}p{{line-height:1.5;color:#455a64;max-width:1050px}}img{{display:block;max-width:100%;height:auto;border:1px solid #dfe6e9;margin:12px 0 28px}}table{{border-collapse:collapse;width:100%;font-size:12px}}th,td{{padding:7px 8px;border-bottom:1px solid #e5e7eb;text-align:right;white-space:nowrap}}th:first-child,td:first-child{{text-align:left}}th{{background:#f7f9fb}}.table-wrap{{overflow-x:auto;margin-bottom:28px}}.note{{border-left:4px solid #0f766e;background:#f7faf9;padding:10px 14px;margin:18px 0}}</style>
</head><body>
<h1>Cross-Anchor Gradient Blend Audit</h1>
<p>This frozen-state diagnostic compares synthetic and real preservation-gradient geometry across anchor-count configurations using the same deterministic mean injection update. {html.escape(bank_note)}</p>
<div class="note">A mask is update-dependent: a coordinate may be safe for one injection update and harmful for another. These figures therefore use a deterministic mean update over the configured injection audit examples.</div>
<h2>Whole-model summary</h2><div class="table-wrap">{_table(overall_rows, columns)}</div>
<h2>Mask decisions</h2><img src="{html.escape(mask_path.name)}" alt="Cross-anchor mask heatmap">
<h2>Gradient geometry</h2><img src="{html.escape(geometry_path.name)}" alt="Cross-anchor gradient geometry heatmap">
<h2>Plasticity</h2><img src="{html.escape(plasticity_path.name)}" alt="Cross-anchor update survival heatmap">
</body></html>"""
    html_path.write_text(report, encoding="utf-8")

    summary = {
        "anchor_counts": anchor_counts,
        "synthetic_bank_status": synthetic_bank_status,
        "synthetic_bank_sha256": distinct_bank_hashes,
        "audits": [audit["summary_path"] for audit in audits],
        "overall": overall_rows,
        "artifacts": {
            "report_html": str(html_path),
            "mask_heatmap_svg": str(mask_path),
            "gradient_geometry_heatmap_svg": str(geometry_path),
            "update_survival_heatmap_svg": str(plasticity_path),
        },
    }
    json_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


_FORGETTING_PREDICTORS = (
    (
        "source_sign_conflict_pct",
        "Source sign conflict",
    ),
    (
        "source_cosine",
        "Source cosine",
    ),
    (
        "update_weighted_cancellation_pct",
        "Update-weighted cancellation",
    ),
    (
        "absolute_log2_scale_mismatch",
        "Absolute log2 real/synthetic RMS ratio",
    ),
    (
        "blended_update_energy_kept_pct",
        "Blended update energy kept",
    ),
    (
        "synthetic_to_blend_flip_energy_pct",
        "Synthetic-to-blend mask disagreement",
    ),
    (
        "real_to_blend_flip_energy_pct",
        "Real-to-blend mask disagreement",
    ),
    (
        "mean_source_to_blend_flip_energy_pct",
        "Mean source-to-blend mask disagreement",
    ),
    (
        "synthetic_predicted_damage_kept_pct",
        "Synthetic predicted damage kept",
    ),
    (
        "real_predicted_damage_kept_pct",
        "Real predicted damage kept",
    ),
)


def _numeric(row: Mapping[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = row.get(key)
        if value in {None, ""}:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            return number
    return None


def _average_tie_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def _pearson(values: np.ndarray, targets: np.ndarray) -> float | None:
    if len(values) < 2:
        return None
    centered_values = values - values.mean()
    centered_targets = targets - targets.mean()
    denominator = math.sqrt(
        float(np.dot(centered_values, centered_values))
        * float(np.dot(centered_targets, centered_targets))
    )
    if denominator <= 1e-30:
        return None
    return float(np.dot(centered_values, centered_targets) / denominator)


def _spearman(values: np.ndarray, targets: np.ndarray) -> float | None:
    return _pearson(_average_tie_ranks(values), _average_tie_ranks(targets))


def _bootstrap_spearman_interval(
    values: np.ndarray,
    targets: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> tuple[float | None, float | None, int]:
    if len(values) < 4 or samples <= 0:
        return None, None, 0
    rng = np.random.default_rng(seed)
    estimates: list[float] = []
    for _ in range(samples):
        indices = rng.integers(0, len(values), size=len(values))
        estimate = _spearman(values[indices], targets[indices])
        if estimate is not None and math.isfinite(estimate):
            estimates.append(estimate)
    if not estimates:
        return None, None, 0
    lower, upper = np.percentile(estimates, [2.5, 97.5])
    return float(lower), float(upper), len(estimates)


def load_gradient_blend_training_run(
    audit_summary_path: str | Path,
) -> dict[str, Any] | None:
    """Join one frozen blend audit to its own initial and final evaluations."""
    audit_summary_path = Path(audit_summary_path)
    audit = load_gradient_blend_audit(audit_summary_path)
    run_dir = audit_summary_path.parents[2]
    summary_csv = run_dir / "summary.csv"
    if not summary_csv.is_file():
        return None
    with summary_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    evaluated_rows: list[tuple[float, Mapping[str, Any]]] = []
    for row in rows:
        epoch = _numeric(row, "epoch")
        correct = _numeric(row, "correct_accuracy_all", "correct_accuracy")
        if epoch is not None and correct is not None:
            evaluated_rows.append((epoch, row))
    if not evaluated_rows:
        return None
    evaluated_rows.sort(key=lambda item: item[0])
    initial_epoch, initial_row = evaluated_rows[0]
    final_epoch, final_row = evaluated_rows[-1]
    if final_epoch <= initial_epoch:
        return None

    initial_correct = _numeric(
        initial_row,
        "correct_accuracy_all",
        "correct_accuracy",
    )
    final_correct = _numeric(
        final_row,
        "correct_accuracy_all",
        "correct_accuracy",
    )
    final_acquisition = _numeric(
        final_row,
        "acquisition_accuracy",
        "wrong_accuracy",
    )
    if initial_correct is None or final_correct is None:
        return None

    metadata = audit["summary"]["metadata"]
    overall = dict(audit["overall"])
    log2_ratio = float(overall["real_to_synthetic_gradient_log2_ratio"])
    overall.update(
        {
            "absolute_log2_scale_mismatch": abs(log2_ratio),
            "synthetic_blend_mask_agreement_pct": 100.0
            - float(overall["synthetic_to_blend_flip_pct"]),
            "real_blend_mask_agreement_pct": 100.0
            - float(overall["real_to_blend_flip_pct"]),
            "synthetic_blend_energy_agreement_pct": 100.0
            - float(overall["synthetic_to_blend_flip_energy_pct"]),
            "real_blend_energy_agreement_pct": 100.0
            - float(overall["real_to_blend_flip_energy_pct"]),
            "mean_source_to_blend_flip_energy_pct": 0.5
            * (
                float(overall["synthetic_to_blend_flip_energy_pct"])
                + float(overall["real_to_blend_flip_energy_pct"])
            ),
        }
    )
    return {
        "run_name": run_dir.name,
        "run_dir": str(run_dir),
        "audit_summary_path": str(audit_summary_path),
        "model_name": metadata.get("model_name", ""),
        "anchor_count": int(metadata.get("anchor_count", audit["anchor_count"])),
        "anchor_seed": int(metadata.get("anchor_seed", 0)),
        "synthetic_seed": int(metadata.get("synthetic_seed", 0)),
        "initial_epoch": initial_epoch,
        "final_epoch": final_epoch,
        "initial_correct_accuracy": initial_correct,
        "final_correct_accuracy": final_correct,
        "final_acquisition_accuracy": final_acquisition,
        "observed_forgetting": initial_correct - final_correct,
        "observed_forgetting_pct": 100.0 * (initial_correct - final_correct),
        "retention_ratio_pct": 100.0
        * final_correct
        / max(initial_correct, 1e-30),
        **overall,
    }


def discover_completed_gradient_blend_runs(
    root: str | Path,
) -> list[dict[str, Any]]:
    runs = []
    for summary_path in discover_gradient_blend_audits(root):
        run = load_gradient_blend_training_run(summary_path)
        if run is not None:
            runs.append(run)
    return sorted(
        runs,
        key=lambda row: (
            int(row["anchor_count"]),
            int(row["anchor_seed"]),
            str(row["run_name"]),
        ),
    )


def _correlation_rows(
    runs: Sequence[Mapping[str, Any]],
    *,
    bootstrap_samples: int,
    seed: int,
) -> list[dict[str, Any]]:
    targets = np.asarray(
        [float(run["observed_forgetting_pct"]) for run in runs],
        dtype=np.float64,
    )
    rows: list[dict[str, Any]] = []
    for predictor_index, (key, label) in enumerate(_FORGETTING_PREDICTORS):
        values = np.asarray([float(run[key]) for run in runs], dtype=np.float64)
        pearson = _pearson(values, targets)
        spearman = _spearman(values, targets)
        lower, upper, valid_bootstraps = _bootstrap_spearman_interval(
            values,
            targets,
            samples=bootstrap_samples,
            seed=seed + predictor_index * 104729,
        )
        rows.append(
            {
                "predictor": key,
                "label": label,
                "n": len(runs),
                "pearson_with_forgetting": pearson,
                "spearman_with_forgetting": spearman,
                "spearman_ci95_low": lower,
                "spearman_ci95_high": upper,
                "valid_bootstrap_samples": valid_bootstraps,
            }
        )
    return sorted(
        rows,
        key=lambda row: abs(row["spearman_with_forgetting"] or 0.0),
        reverse=True,
    )


def _scatter_svg(
    runs: Sequence[Mapping[str, Any]],
    path: Path,
) -> None:
    panels = (
        (
            "absolute_log2_scale_mismatch",
            "|log2 real/synthetic RMS|",
        ),
        (
            "update_weighted_cancellation_pct",
            "Update-weighted cancellation (%)",
        ),
        (
            "source_sign_conflict_pct",
            "Source sign conflict (%)",
        ),
        (
            "mean_source_to_blend_flip_energy_pct",
            "Mean mask disagreement energy (%)",
        ),
    )
    panel_w, panel_h = 410, 300
    width, height = panel_w * 2, panel_h * 2
    pieces = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
    ]
    y_values = np.asarray(
        [float(run["observed_forgetting_pct"]) for run in runs],
        dtype=np.float64,
    )
    for panel_index, (key, label) in enumerate(panels):
        column, row = panel_index % 2, panel_index // 2
        origin_x, origin_y = column * panel_w, row * panel_h
        left, right = origin_x + 62, origin_x + panel_w - 20
        top, bottom = origin_y + 42, origin_y + panel_h - 50
        x_values = np.asarray([float(run[key]) for run in runs], dtype=np.float64)
        x_min, x_max = float(x_values.min()), float(x_values.max())
        y_min, y_max = float(y_values.min()), float(y_values.max())
        if x_max <= x_min:
            x_min, x_max = x_min - 0.5, x_max + 0.5
        if y_max <= y_min:
            y_min, y_max = y_min - 0.5, y_max + 0.5
        x_pad, y_pad = 0.08 * (x_max - x_min), 0.1 * (y_max - y_min)
        x_min, x_max = x_min - x_pad, x_max + x_pad
        y_min, y_max = y_min - y_pad, y_max + y_pad

        def sx(value: float) -> float:
            return left + (value - x_min) / (x_max - x_min) * (right - left)

        def sy(value: float) -> float:
            return bottom - (value - y_min) / (y_max - y_min) * (bottom - top)

        pieces.extend(
            (
                f'<text x="{origin_x + 16}" y="{origin_y + 23}" '
                'font-family="Arial,sans-serif" font-size="14" '
                f'font-weight="700" fill="#17202a">{html.escape(label)}</text>',
                f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" '
                'stroke="#64748b"/>',
                f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" '
                'stroke="#64748b"/>',
                f'<text x="{(left + right) / 2:.1f}" y="{origin_y + panel_h - 13}" '
                'text-anchor="middle" font-family="Arial,sans-serif" '
                f'font-size="11" fill="#475569">{html.escape(label)}</text>',
                f'<text x="{origin_x + 15}" y="{(top + bottom) / 2:.1f}" '
                'text-anchor="middle" font-family="Arial,sans-serif" '
                'font-size="11" fill="#475569" '
                f'transform="rotate(-90 {origin_x + 15} {(top + bottom) / 2:.1f})">'
                'Observed forgetting (points)</text>',
                f'<text x="{left}" y="{bottom + 16}" text-anchor="middle" '
                f'font-family="Arial,sans-serif" font-size="9" fill="#64748b">{x_min:.2f}</text>',
                f'<text x="{right}" y="{bottom + 16}" text-anchor="middle" '
                f'font-family="Arial,sans-serif" font-size="9" fill="#64748b">{x_max:.2f}</text>',
                f'<text x="{left - 7}" y="{bottom + 3}" text-anchor="end" '
                f'font-family="Arial,sans-serif" font-size="9" fill="#64748b">{y_min:.1f}</text>',
                f'<text x="{left - 7}" y="{top + 3}" text-anchor="end" '
                f'font-family="Arial,sans-serif" font-size="9" fill="#64748b">{y_max:.1f}</text>',
            )
        )
        if len(runs) >= 2 and float(np.var(x_values)) > 1e-30:
            slope, intercept = np.polyfit(x_values, y_values, 1)
            pieces.append(
                f'<line x1="{sx(float(x_values.min())):.1f}" '
                f'y1="{sy(float(slope * x_values.min() + intercept)):.1f}" '
                f'x2="{sx(float(x_values.max())):.1f}" '
                f'y2="{sy(float(slope * x_values.max() + intercept)):.1f}" '
                'stroke="#94a3b8" stroke-width="1.5" stroke-dasharray="4 3"/>'
            )
        for run, x_value, y_value in zip(runs, x_values, y_values):
            point_x, point_y = sx(float(x_value)), sy(float(y_value))
            point_label = f'A{int(run["anchor_count"])}/s{int(run["anchor_seed"])}'
            pieces.extend(
                (
                    f'<circle cx="{point_x:.1f}" cy="{point_y:.1f}" r="4" fill="#0f766e"/>',
                    f'<text x="{point_x + 6:.1f}" y="{point_y - 5:.1f}" '
                    'font-family="Arial,sans-serif" font-size="9" '
                    f'fill="#334155">{html.escape(point_label)}</text>',
                )
            )
    pieces.append("</svg>")
    path.write_text("\n".join(pieces) + "\n", encoding="utf-8")


def _correlation_value(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.3f}"


def write_gradient_collision_prediction_report(
    audit_summary_paths: Sequence[str | Path],
    output_dir: str | Path,
    *,
    minimum_acquisition: float = 0.98,
    bootstrap_samples: int = 2000,
    seed: int = 0,
) -> dict[str, Any]:
    """Test whether initial blend geometry predicts later mastered-set loss."""
    all_runs = [
        run
        for path in audit_summary_paths
        if (run := load_gradient_blend_training_run(path)) is not None
    ]
    if not all_runs:
        raise ValueError(
            "no audits had both an epoch-0 and a later evaluation in the same run"
        )
    eligible_runs = [
        run
        for run in all_runs
        if run["final_acquisition_accuracy"] is not None
        and float(run["final_acquisition_accuracy"]) >= minimum_acquisition
    ]
    analysis_runs = eligible_runs if eligible_runs else all_runs
    correlations = _correlation_rows(
        analysis_runs,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    runs_path = output_dir / "gradient_collision_runs.csv"
    correlations_path = output_dir / "gradient_collision_correlations.csv"
    scatter_path = output_dir / "gradient_collision_prediction.svg"
    html_path = output_dir / "gradient_collision_prediction.html"
    json_path = output_dir / "gradient_collision_prediction.json"
    _write_csv(runs_path, analysis_runs)
    _write_csv(correlations_path, correlations)
    _scatter_svg(analysis_runs, scatter_path)

    strongest = correlations[0]
    correlation_table = "".join(
        "<tr>"
        f"<td>{html.escape(str(row['label']))}</td>"
        f"<td>{_correlation_value(row['pearson_with_forgetting'])}</td>"
        f"<td>{_correlation_value(row['spearman_with_forgetting'])}</td>"
        f"<td>{_correlation_value(row['spearman_ci95_low'])}</td>"
        f"<td>{_correlation_value(row['spearman_ci95_high'])}</td>"
        "</tr>"
        for row in correlations
    )
    run_table = "".join(
        "<tr>"
        f"<td>{html.escape(str(row['run_name']))}</td>"
        f"<td>{int(row['anchor_count'])}</td>"
        f"<td>{int(row['anchor_seed'])}</td>"
        f"<td>{100 * float(row['final_correct_accuracy']):.2f}</td>"
        f"<td>{100 * float(row['final_acquisition_accuracy']):.2f}</td>"
        f"<td>{float(row['observed_forgetting_pct']):.2f}</td>"
        "</tr>"
        for row in analysis_runs
    )
    caution = (
        "Fewer than eight independent runs are available, so correlations are "
        "exploratory and bootstrap intervals will be unstable."
        if len(analysis_runs) < 8
        else "Correlations remain observational; anchor count and seed should be controlled in confirmatory models."
    )
    report = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Gradient Collision and Forgetting</title>
<style>
body{{font:14px Arial,sans-serif;color:#17202a;margin:32px;max-width:1280px}}h1,h2{{margin:0 0 12px}}p{{line-height:1.5;color:#455a64;max-width:1050px}}img{{max-width:100%;height:auto;border:1px solid #dfe6e9}}table{{border-collapse:collapse;width:100%;font-size:12px;margin-bottom:28px}}th,td{{padding:7px 8px;border-bottom:1px solid #e5e7eb;text-align:right}}th:first-child,td:first-child{{text-align:left}}th{{background:#f7f9fb}}.note{{border-left:4px solid #0f766e;background:#f7faf9;padding:10px 14px;margin:18px 0}}</style>
</head><body>
<h1>Can gradient collision predict forgetting?</h1>
<p>Each predictor is measured before training from the synthetic preservation gradient, the real-anchor preservation gradient, their fixed blend, and one deterministic mean injection update. The target is the drop in full mastered-set accuracy from epoch 0 to the final epoch of that same run.</p>
<div class="note"><strong>Acquisition gate:</strong> primary analysis uses runs with final injection accuracy at least {100 * minimum_acquisition:.1f}%. {html.escape(caution)}</div>
<p><strong>Largest observed rank association:</strong> {html.escape(str(strongest['label']))}, Spearman {_correlation_value(strongest['spearman_with_forgetting'])}. This is a screening statistic, not evidence of causation.</p>
<img src="{html.escape(scatter_path.name)}" alt="Collision predictors versus forgetting">
<h2>Correlations with forgetting</h2>
<table><thead><tr><th>Predictor</th><th>Pearson</th><th>Spearman</th><th>Bootstrap 2.5%</th><th>Bootstrap 97.5%</th></tr></thead><tbody>{correlation_table}</tbody></table>
<h2>Included runs</h2>
<table><thead><tr><th>Run</th><th>A</th><th>Seed</th><th>Final retention %</th><th>Final acquisition %</th><th>Observed forgetting points</th></tr></thead><tbody>{run_table}</tbody></table>
<p>Layer rows should remain descriptive because every layer in one run shares the same global forgetting target. Treating layers as independent samples would inflate the sample size. Confirmatory prediction should use one aggregate row per run, then inspect layer heatmaps to localize the mechanism.</p>
</body></html>"""
    html_path.write_text(report, encoding="utf-8")

    summary = {
        "minimum_acquisition": minimum_acquisition,
        "all_completed_run_count": len(all_runs),
        "eligible_run_count": len(eligible_runs),
        "analysis_run_count": len(analysis_runs),
        "used_acquisition_gate": bool(eligible_runs),
        "correlations": correlations,
        "artifacts": {
            "runs_csv": str(runs_path),
            "correlations_csv": str(correlations_path),
            "scatter_svg": str(scatter_path),
            "report_html": str(html_path),
        },
    }
    json_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary
