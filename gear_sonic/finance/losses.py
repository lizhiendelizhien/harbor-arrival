"""Supervised trajectory, auxiliary reconstruction and shared-latent cycle losses."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return torch.where(mask, values, 0.0).sum() / mask.sum().clamp_min(1)


def financial_auxiliary_losses(output: dict, *, mask: torch.Tensor | None = None) -> dict:
    """Only kin reconstruction and shared-encoder cycle supervision, for PPO."""
    if mask is None:
        mask = output["future_mask"]
    if mask.shape != output["future_mask"].shape or mask.dtype != torch.bool or not mask.any():
        raise ValueError("Auxiliary losses require a matching nonempty boolean mask")
    kin = _masked_mean((output["kin_normalized"] - output["future_normalized"]).square(), mask)
    cycle = _masked_mean(
        (output["reencoded_latent"] - output["latent"]).square().mean(dim=-1),
        mask.any(dim=(-1, -2)),
    )
    return {"kin": kin, "cycle": cycle}


def financial_sonic_loss(
    output: dict, future: torch.Tensor, *, future_mask: torch.Tensor | None = None,
    return_scale: torch.Tensor | float = 1.0, cumulative_weight: float = 1.0,
    kin_weight: float = 0.01, cycle_weight: float = 1.0, burn_in: int = 0,
) -> dict[str, torch.Tensor]:
    """Losses operate on [B,S,H,F] targets; burn_in masks initial anchor steps.

    Current and future normalization statistics must have been fitted on the
    training split. All cycle paths remain differentiable, including original z.
    A missing monthly return invalidates cumulative targets from that horizon on.
    """
    mask = output["future_mask"]
    if future_mask is not None:
        if future_mask.shape != mask.shape or future_mask.dtype != torch.bool:
            raise ValueError("future_mask must match future observations")
        mask = mask & future_mask
    if future.shape != mask.shape or not torch.isfinite(future[mask]).all():
        raise ValueError("Targets must match the future shape and be finite where valid")
    if not 0 <= burn_in < future.shape[1]:
        raise ValueError("burn_in must leave at least one supervised anchor step")
    mask = mask.clone()
    mask[:, :burn_in] = False
    if not mask.any():
        raise ValueError("Loss requires valid targets after burn_in")
    scale = torch.as_tensor(return_scale, dtype=future.dtype, device=future.device)
    if scale.numel() != 1 or not torch.isfinite(scale).all() or scale <= 0:
        raise ValueError("return_scale must be a finite positive scalar")
    return_mask = mask[..., 0]
    target = torch.where(return_mask, future[..., 0], 0.0)
    pred = output["log_returns"]
    dyn = _masked_mean(F.smooth_l1_loss(pred / scale, target / scale, reduction="none"), return_mask)
    cumulative_mask = return_mask.long().cumprod(dim=-1).bool()
    horizon_scale = torch.arange(1, future.shape[-2] + 1, device=future.device, dtype=future.dtype).sqrt() * scale
    cumulative = _masked_mean(F.smooth_l1_loss(
        pred.cumsum(-1) / horizon_scale, target.cumsum(-1) / horizon_scale, reduction="none",
    ), cumulative_mask)
    auxiliary = financial_auxiliary_losses(output, mask=mask)
    kin, cycle = auxiliary["kin"], auxiliary["cycle"]
    total = dyn + cumulative_weight * cumulative + kin_weight * kin + cycle_weight * cycle
    return {"total": total, "dyn": dyn, "cumulative": cumulative, "kin": kin, "cycle": cycle}
