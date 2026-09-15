"""Fixed reference-tracking rewards and shared scalar diagnostic summaries."""

from __future__ import annotations

import math

import torch


TRACKING_REWARD_CONTRACT = {
    "version": 2,
    "name": "financial_tracking_v1",
    "monthly_weight": 0.3,
    "cumulative_weight": 0.4,
    "change_weight": 0.2,
    "volatility_weight": 0.1,
    "tau_month": 1.0,
    "tau_path": 1.0,
    "tau_change": 1.0,
    "tau_vol": 1.0,
    "tau_roll": 1.0,
    "lambda_roll": 0.0,
    "volatility_correction": 0,
    "target": "unclipped_monthly_log_return_normalized_by_checkpoint_center_scale",
    "cumulative_normalization": "sqrt_horizon",
}

LEGACY_TRACKING_REWARD_CONTRACT = {
    "version": 1, "name": "monthly_and_cumulative_exponential_tracking",
    "monthly_weight": 0.5, "cumulative_weight": 0.5,
    "target": "unclipped_monthly_log_return_normalized_by_checkpoint_center_scale",
    "cumulative_normalization": "sqrt_horizon",
}

# Validate against independent snapshots even if a caller mutates an exported dict.
_V1_CONTRACT = dict(TRACKING_REWARD_CONTRACT)
_LEGACY_CONTRACT = dict(LEGACY_TRACKING_REWARD_CONTRACT)
_COMPONENTS = ("month", "path", "change", "vol")
_HORIZONS = (1, 3, 6, 10)
_ROLLING_METRICS = ("rolling_mse", "rolling_penalty", "previous_overlap_mse")


def validate_reward_contract(contract, *, allow_legacy=False):
    """Reject missing, altered or unsupported reward metadata without coercion."""
    supported = (_V1_CONTRACT, _LEGACY_CONTRACT) if allow_legacy else (_V1_CONTRACT,)
    if isinstance(contract, dict):
        for expected in supported:
            if contract.keys() == expected.keys() and all(
                type(contract[key]) is type(value) and contract[key] == value
                for key, value in expected.items()
            ):
                return None
    raise ValueError("Reward contract must exactly match a supported tracking objective")


@torch.no_grad()
def tracking_metrics(
    actions, future_returns, return_center, return_scale, *,
    previous_actions=None, rolling_valid=None, reward_contract=None,
):
    """Score matching [...,10] normalized actions and unclipped raw log returns.

    Only ``reward`` is float32; all numeric diagnostics remain float64. Rolling
    comparisons are calendar-aligned diagnostics and never alter the V1 reward.
    """
    contract = _V1_CONTRACT if reward_contract is None else reward_contract
    validate_reward_contract(contract, allow_legacy=True)
    actions = torch.as_tensor(actions)
    future_returns = torch.as_tensor(future_returns, device=actions.device)
    center = torch.as_tensor(return_center, dtype=torch.float32, device=actions.device)
    scale = torch.as_tensor(return_scale, dtype=torch.float32, device=actions.device)
    if actions.shape != future_returns.shape or actions.ndim < 1 or actions.shape[-1] != 10:
        raise ValueError("Tracking actions and future returns must have matching ten-month shapes")
    if (center.ndim != 0 or scale.ndim != 0 or not torch.isfinite(center)
            or not torch.isfinite(scale) or scale <= 0):
        raise ValueError("Tracking center and positive scale must be finite scalars")
    if not torch.isfinite(actions).all() or not torch.isfinite(future_returns).all():
        raise ValueError("Tracking actions and targets must be finite")

    # Widen before subtraction, normalization, sums, differences, or variances.
    actions, future_returns = actions.double(), future_returns.double()
    center, scale = center.double(), scale.double()
    targets = (future_returns - center) / scale
    errors = actions - targets
    monthly_mse = errors.square().mean(-1)
    sqrt_horizons = torch.arange(1, 11, dtype=torch.float64, device=actions.device).sqrt()
    cumulative_mse = (errors.cumsum(-1) / sqrt_horizons).square().mean(-1)
    if contract == _LEGACY_CONTRACT:
        reward = (0.5 * ((-monthly_mse).exp() + (-cumulative_mse).exp())).float()
        return {"reward": reward, "monthly_mse": monthly_mse, "cumulative_mse": cumulative_mse}

    if rolling_valid is None:
        valid = torch.full(actions.shape[:-1], previous_actions is not None,
                           dtype=torch.bool, device=actions.device)
    else:
        valid = torch.as_tensor(rolling_valid, device=actions.device)
        if valid.dtype != torch.bool or valid.shape != actions.shape[:-1]:
            raise ValueError("Rolling validity must be boolean with the action batch shape")
    if previous_actions is None:
        if valid.any():
            raise ValueError("Valid rolling comparisons require previous actions")
        previous = torch.zeros_like(actions)
    else:
        previous = torch.as_tensor(previous_actions, device=actions.device)
        if previous.shape != actions.shape:
            raise ValueError("Previous actions must match the current action shape")
        previous = torch.where(valid.unsqueeze(-1), previous.double(), 0.0)
        if not torch.isfinite(previous).all():
            raise ValueError("Valid previous actions must be finite")

    change_mse = (errors.diff(dim=-1).square() / 2).mean(-1)
    volatility_mse = (actions.std(-1, correction=0) - targets.std(-1, correction=0)).square()
    metrics = {"monthly_mse": monthly_mse, "cumulative_mse": cumulative_mse,
               "change_mse": change_mse, "volatility_mse": volatility_mse}
    reward_tracking = torch.zeros_like(monthly_mse)
    for component, error, weight in zip(
        _COMPONENTS, (monthly_mse, cumulative_mse, change_mse, volatility_mse),
        (0.3, 0.4, 0.2, 0.1),
    ):
        component_reward = (-error).exp()
        contribution = weight * component_reward
        metrics[f"reward_{component}"] = component_reward
        metrics[f"contribution_{component}"] = contribution
        reward_tracking += contribution
    metrics.update(reward=reward_tracking.float(), reward_tracking=reward_tracking,
                   reward_total=reward_tracking)

    current_overlap = torch.where(valid.unsqueeze(-1), actions[..., :9], 0.0)
    target_overlap = torch.where(valid.unsqueeze(-1), targets[..., :9], 0.0)
    rolling_mse = (current_overlap - previous[..., 1:]).square().mean(-1)
    metrics.update(
        rolling_valid=valid,
        rolling_mse=rolling_mse,
        rolling_penalty=-torch.expm1(-rolling_mse),
        previous_overlap_mse=(previous[..., 1:] - target_overlap).square().mean(-1),
    )

    predicted_cumulative = (center + scale * actions).cumsum(-1)
    target_cumulative = future_returns.cumsum(-1)
    for horizon in _HORIZONS:
        prediction = predicted_cumulative[..., horizon - 1]
        target = target_cumulative[..., horizon - 1]
        predicted_sign = (prediction > 1e-8).double() - (prediction < -1e-8).double()
        target_sign = (target > 1e-8).double() - (target < -1e-8).double()
        metrics[f"cumulative_squared_error_{horizon}m"] = (prediction - target).square()
        metrics[f"direction_correct_{horizon}m"] = (predicted_sign == target_sign).double()
        for label, sign in (("negative", -1), ("neutral", 0), ("positive", 1)):
            metrics[f"direction_{label}_{horizon}m"] = (target_sign == sign).double()
    if any(not torch.isfinite(value).all() for value in metrics.values()):
        raise ValueError("Tracking metrics must remain finite")
    return metrics


@torch.no_grad()
def summarize_tracking_metrics(metrics, *, include_quantiles=True):
    """Aggregate scalar per-transition metrics into flat JSON-safe values."""
    if "reward" not in metrics:
        raise ValueError("Tracking metrics must include reward")
    shape = torch.as_tensor(metrics["reward"]).shape
    values = {}
    for name, source in metrics.items():
        value = torch.as_tensor(source)
        if value.shape != shape:
            raise ValueError("Tracking metric shapes must match the reward shape")
        if name == "rolling_valid":
            if value.dtype != torch.bool:
                raise ValueError("Rolling validity must be boolean")
        elif not torch.isfinite(value).all():
            raise ValueError("Tracking metrics must be finite")
        else:
            value = value.double()
        values[name] = value.reshape(-1)

    count = values["reward"].numel()
    summary = {"sample_count": count}
    for name, value in values.items():
        if name == "rolling_valid" or name in _ROLLING_METRICS or name.startswith("direction_"):
            continue
        summary[name] = value.mean().item() if count else None
    if "rolling_valid" in values:
        valid = values["rolling_valid"]
        valid_count = valid.sum().item()
        summary["rolling_valid_count"] = valid_count
        for name in _ROLLING_METRICS:
            summary[name] = values[name][valid].mean().item() if valid_count else None
    if include_quantiles:
        for component in _COMPONENTS:
            name = f"reward_{component}"
            if name not in values:
                continue
            value = values[name]
            for quantile in (10, 50, 90):
                summary[f"{name}_p{quantile}"] = (
                    torch.quantile(value, quantile / 100).item() if count else None
                )
            summary[f"{name}_below_005"] = (value < 0.05).double().mean().item() if count else None
            summary[f"{name}_above_095"] = (value > 0.95).double().mean().item() if count else None
    for horizon in _HORIZONS:
        squared_error = f"cumulative_squared_error_{horizon}m"
        if squared_error not in summary:
            continue
        summary[f"cumulative_rmse_{horizon}m"] = math.sqrt(summary[squared_error]) if count else None
        summary[f"direction_accuracy_{horizon}m"] = (
            values[f"direction_correct_{horizon}m"].mean().item() if count else None
        )
        for label in ("negative", "neutral", "positive"):
            summary[f"direction_{label}_count_{horizon}m"] = int(
                values[f"direction_{label}_{horizon}m"].sum().item()
            )
    return summary
