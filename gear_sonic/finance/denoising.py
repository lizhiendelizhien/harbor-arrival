"""Versioned financial encoder corruption, applied to stored normalized inputs."""

from __future__ import annotations

import torch

from .observations import FUTURE_FIELDS


ENCODER_MASK_WEIGHTS = (1.0, 1.0, 1.0, 0.1)
ENCODER_NOISE_BOUND = 0.05
ENCODER_MASK_DURATION_STEPS = (2, 5)
_VISIBLE_FEATURES = (
    tuple(range(15)), tuple(range(11)) + (13, 14), tuple(range(11)), tuple(range(8)),
)


def encoder_denoising_contract() -> dict:
    """Return independent serializable metadata for exact training compatibility."""
    return {
        "schema_version": 1,
        "name": "financial_encoder_denoising_v1",
        "horizon": 10,
        "feature_fields": list(FUTURE_FIELDS),
        "mask_modes": [
            {"id": mode, "visible_feature_indices": list(indices)}
            for mode, indices in enumerate(_VISIBLE_FEATURES)
        ],
        "mask_weights": list(ENCODER_MASK_WEIGHTS),
        "mask_probabilities": [weight / sum(ENCODER_MASK_WEIGHTS) for weight in ENCODER_MASK_WEIGHTS],
        "mask_scope": "per_environment_shared_across_horizon",
        "mask_duration_monthly_steps": {
            "distribution": "discrete_uniform", "min": ENCODER_MASK_DURATION_STEPS[0],
            "max": ENCODER_MASK_DURATION_STEPS[1], "resample": "expiry_or_episode_reset",
        },
        "noise": {
            "distribution": "uniform", "low": -ENCODER_NOISE_BOUND, "high": ENCODER_NOISE_BOUND,
            "units": "normalized_features", "independent": "per_environment_month_feature",
            "resample": "each_new_anchor",
        },
        "operation_order": ["normalize_clean_future", "add_noise", "zero_hidden_and_invalid"],
        "kin_target": "clean_normalized_future_all_valid_features",
        "future_mask": "data_validity_only",
        "cycle_input": "normalized_kin_reconstruction_without_corruption_renormalization_or_detach",
    }


def prepare_encoder_input(
    normalized_future: torch.Tensor, future_mask: torch.Tensor, *,
    encoder_noise: torch.Tensor | None = None, encoder_mask_type: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply a saved corruption snapshot, or leave clean inference uncorrupted."""
    if normalized_future.ndim < 3 or normalized_future.shape[-1] != len(FUTURE_FIELDS):
        raise ValueError("Normalized future observations must end in [H,15]")
    if (future_mask.shape != normalized_future.shape or future_mask.dtype != torch.bool
            or future_mask.device != normalized_future.device):
        raise ValueError("future_mask must match normalized future shape and device and be boolean")
    if (encoder_noise is None) != (encoder_mask_type is None):
        raise ValueError("encoder_noise and encoder_mask_type must be supplied together")
    if encoder_noise is None:
        return torch.where(future_mask, normalized_future, 0.0)
    if (encoder_noise.shape != normalized_future.shape or not encoder_noise.is_floating_point()
            or encoder_noise.dtype != normalized_future.dtype
            or encoder_noise.device != normalized_future.device
            or not torch.isfinite(encoder_noise).all()
            or (encoder_noise.abs() > ENCODER_NOISE_BOUND).any()):
        raise ValueError("encoder_noise must match normalized future shape, dtype and device and be finite in [-0.05,0.05]")
    if (encoder_mask_type.shape != (*normalized_future.shape[:-2], 1)
            or encoder_mask_type.dtype != torch.int64
            or encoder_mask_type.device != normalized_future.device
            or ((encoder_mask_type < 0) | (encoder_mask_type >= len(_VISIBLE_FEATURES))).any()):
        raise ValueError("encoder_mask_type must be int64 in [0,3], on the input device, with shape [...,1]")
    visibility = torch.tensor([
        [feature in indices for feature in range(len(FUTURE_FIELDS))]
        for indices in _VISIBLE_FEATURES
    ], dtype=torch.bool, device=normalized_future.device)
    visible = visibility[encoder_mask_type.squeeze(-1)].unsqueeze(-2) & future_mask
    return torch.where(visible, normalized_future + encoder_noise, 0.0)
