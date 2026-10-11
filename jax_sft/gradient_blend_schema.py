"""Backend-neutral schema for fixed gradient-blend audit reductions."""

from __future__ import annotations


FIXED_BLEND_TENSOR_STAT_FIELDS = (
    "parameter_count",
    "update_energy",
    "synthetic_kept_energy",
    "real_kept_energy",
    "blended_kept_energy",
    "synthetic_allow_count",
    "real_allow_count",
    "blended_allow_count",
    "source_active_count",
    "source_agreement_count",
    "source_conflict_count",
    "synthetic_blend_flip_count",
    "real_blend_flip_count",
    "synthetic_blend_flip_energy",
    "real_blend_flip_energy",
    "weighted_synthetic_l1",
    "weighted_real_l1",
    "blended_l1",
    "weighted_synthetic_update_l1",
    "weighted_real_update_l1",
    "blended_update_l1",
    "source_dot",
    "synthetic_source_sq",
    "real_source_sq",
    "blended_source_sq",
    "synthetic_damage_before",
    "synthetic_damage_after_blend",
    "real_damage_before",
    "real_damage_after_blend",
)
