import copy
import importlib.util
import json
import math

import pytest
import torch


COMPONENTS = ("month", "path", "change", "vol")
HORIZONS = (1, 3, 6, 10)


def test_shared_reward_interfaces_exist():
    assert importlib.util.find_spec("gear_sonic.finance.rewards") is not None
    from gear_sonic.finance import rewards

    for name in ("TRACKING_REWARD_CONTRACT", "LEGACY_TRACKING_REWARD_CONTRACT",
                 "validate_reward_contract", "tracking_metrics", "summarize_tracking_metrics"):
        assert hasattr(rewards, name)


def test_contract_is_fixed_explicit_and_rejects_mutations():
    from gear_sonic.finance.rewards import (
        LEGACY_TRACKING_REWARD_CONTRACT, TRACKING_REWARD_CONTRACT, validate_reward_contract,
    )

    contract = TRACKING_REWARD_CONTRACT
    assert contract["name"] == "financial_tracking_v1"
    assert contract["version"] == 2
    assert [contract[key] for key in (
        "monthly_weight", "cumulative_weight", "change_weight", "volatility_weight",
    )] == [0.3, 0.4, 0.2, 0.1]
    assert [contract[f"tau_{name}"] for name in (*COMPONENTS, "roll")] == [1.0] * 5
    assert contract["lambda_roll"] == 0.0
    assert contract["volatility_correction"] == 0
    assert validate_reward_contract(copy.deepcopy(contract)) is None
    assert validate_reward_contract(LEGACY_TRACKING_REWARD_CONTRACT, allow_legacy=True) is None
    bad_contracts = [None, {}, [], {**contract, "unexpected": 0}, LEGACY_TRACKING_REWARD_CONTRACT]
    for key in contract:
        missing = dict(contract)
        missing.pop(key)
        bad_contracts.append(missing)
        bad_contracts.append({**contract, key: "modified"})
    bad_contracts.extend({**contract, key: value} for key, value in (
        ("lambda_roll", 0.02), ("tau_month", math.nan), ("monthly_weight", math.inf),
        ("volatility_correction", False),
    ))
    for bad in bad_contracts:
        with pytest.raises(ValueError, match="[Rr]eward contract"):
            validate_reward_contract(bad)


@pytest.mark.parametrize("values", [
    [0.0] * 10, [-0.5] * 10,
    [0.1, 0.2, -0.1, -5.0, 0.8, -2.0, 0.1, 0.0, -0.2, 0.4],
])
def test_exact_flat_negative_and_crash_paths_maximize_every_component(values):
    from gear_sonic.finance.rewards import tracking_metrics

    target = torch.tensor([values], requires_grad=True)
    metrics = tracking_metrics(target, target, 0.0, 1.0)
    for name in ("reward", "reward_tracking", "reward_total", *(f"reward_{c}" for c in COMPONENTS)):
        torch.testing.assert_close(metrics[name], torch.ones_like(metrics[name]))
    for name, value in metrics.items():
        assert not value.requires_grad
        assert value.dtype == (torch.bool if name == "rolling_valid" else
                               torch.float32 if name == "reward" else torch.float64)
        assert value.shape == (1,)
        assert torch.isfinite(value).all()


def test_four_formulas_use_normalized_units_population_std_and_fixed_weights():
    from gear_sonic.finance.rewards import tracking_metrics

    actions = torch.tensor([[0.0, 0.3, -2.0, 1.5, 0.1, 0.2, -0.2, -0.9, 0.7, 1.3],
                            [-0.8, -0.4, 0.6, 1.2, 0.5, -0.3, -1.4, -0.8, 0.4, 1.0]])
    raw = torch.linspace(-0.3, 0.4, 20).reshape(2, 10)
    center, scale = torch.tensor(0.1), torch.tensor(0.2)
    target = (raw.double() - center.double()) / scale.double()
    errors = actions.double() - target
    expected_errors = (
        errors.square().mean(-1),
        (errors.cumsum(-1) / torch.arange(1, 11).double().sqrt()).square().mean(-1),
        (errors.diff(dim=-1).square() / 2).mean(-1),
        (actions.double().std(-1, correction=0) - target.std(-1, correction=0)).square(),
    )
    metrics = tracking_metrics(actions, raw, center, scale)
    expected_reward = torch.zeros(2, dtype=torch.float64)
    for component, error_name, weight, expected in zip(
        COMPONENTS, ("monthly_mse", "cumulative_mse", "change_mse", "volatility_mse"),
        (0.3, 0.4, 0.2, 0.1), expected_errors,
    ):
        torch.testing.assert_close(metrics[error_name], expected)
        torch.testing.assert_close(metrics[f"reward_{component}"], (-expected).exp())
        contribution = weight * (-expected).exp()
        torch.testing.assert_close(metrics[f"contribution_{component}"], contribution)
        expected_reward += contribution
    torch.testing.assert_close(metrics["reward_tracking"], expected_reward)
    torch.testing.assert_close(metrics["reward_total"], expected_reward)
    torch.testing.assert_close(metrics["reward"], expected_reward.float())


def test_imperfect_predictions_are_not_maximized_and_bias_keeps_change_and_vol():
    from gear_sonic.finance.rewards import tracking_metrics

    target = torch.tensor([[-0.8, -0.4, 0.6, 1.2, 0.5, -0.3, -1.4, -0.8, 0.4, 1.0]])
    alternatives = (-target, torch.zeros_like(target), target.roll(1, -1), 2 * target, target + 1)
    for actions in alternatives:
        assert tracking_metrics(actions, target, 0, 1)["reward"].item() < 1.0
    biased = tracking_metrics(target + 1, target, 0, 1)
    for component in ("change", "vol"):
        assert biased[f"reward_{component}"].item() == pytest.approx(1.0)
    for component in ("month", "path"):
        assert biased[f"reward_{component}"].item() < 1
    opposite = tracking_metrics(-target, target, 0, 1)
    assert opposite["reward_vol"].item() == pytest.approx(1.0)


def test_extreme_float32_inputs_are_widened_before_all_arithmetic_and_summary():
    from gear_sonic.finance.rewards import summarize_tracking_metrics, tracking_metrics

    largest = torch.finfo(torch.float32).max
    smallest = torch.nextafter(torch.tensor(0.0), torch.tensor(1.0))
    actions = torch.full((3, 10), -largest, requires_grad=True)
    raw = torch.full((3, 10), largest)
    raw[:, 1::2] = -largest
    raw.requires_grad_()
    metrics = tracking_metrics(actions, raw, -largest, smallest)
    assert metrics["monthly_mse"].min() > torch.finfo(torch.float32).max
    for name, value in metrics.items():
        assert torch.isfinite(value).all(), name
        assert not value.requires_grad
    summary = summarize_tracking_metrics(metrics)
    assert math.isfinite(summary["monthly_mse"])
    json.dumps(summary, allow_nan=False)


def test_unclipped_large_targets_and_original_unit_direction_deadband():
    from gear_sonic.finance.rewards import tracking_metrics

    actions = torch.tensor([[-0.25] * 10, [-0.75] * 10, [-0.5] * 10, [-0.5] * 10])
    raw = torch.tensor([[0.5] * 10, [0.5] * 10, [0.0] * 10, [-1e-6] * 10])
    metrics = tracking_metrics(actions, raw, 1.0, 2.0)
    for horizon in HORIZONS:
        assert metrics[f"direction_correct_{horizon}m"].tolist() == [1, 0, 1, 0]
        assert metrics[f"direction_positive_{horizon}m"].tolist() == [1, 1, 0, 0]
        assert metrics[f"direction_neutral_{horizon}m"].tolist() == [0, 0, 1, 0]
        assert metrics[f"direction_negative_{horizon}m"].tolist() == [0, 0, 0, 1]
        expected = ((1 + 2 * actions.double())[:, :horizon].sum(-1) - raw.double()[:, :horizon].sum(-1)).square()
        torch.testing.assert_close(metrics[f"cumulative_squared_error_{horizon}m"], expected)
    tiny = torch.full((1, 10), 1e-10)
    assert tracking_metrics(tiny, -tiny, 0, 1)["direction_correct_10m"].item() == 1
    assert tracking_metrics(torch.zeros(1, 10), torch.full((1, 10), 1000.0), 0, 1)["monthly_mse"].item() == 1e6


def test_rolling_alignment_comes_from_one_calendar_series_and_is_diagnostic_only():
    from gear_sonic.finance.rewards import tracking_metrics

    calendar = torch.tensor([-0.8, -0.4, 0.6, 1.2, 0.5, -0.3, -1.4, -0.8, 0.4, 1.0, 0.3])
    previous, current = calendar[:10].unsqueeze(0), calendar[1:].unsqueeze(0)
    aligned = tracking_metrics(current, current, 0, 1, previous_actions=previous)
    assert aligned["rolling_valid"].tolist() == [True]
    for name in ("rolling_mse", "rolling_penalty", "previous_overlap_mse"):
        assert aligned[name].item() == 0
    assert (current[:, :9] - previous[:, :9]).square().mean().item() > 0
    perturbed = tracking_metrics(current, current, 0, 1, previous_actions=previous + 10)
    assert perturbed["rolling_mse"].item() == pytest.approx(100)
    assert perturbed["previous_overlap_mse"].item() == pytest.approx(100)
    assert perturbed["rolling_penalty"].item() == pytest.approx(1 - math.exp(-100))
    torch.testing.assert_close(aligned["reward"], perturbed["reward"])


def test_invalid_overlap_is_masked_before_arithmetic_and_absent_previous_is_invalid():
    from gear_sonic.finance.rewards import tracking_metrics

    current = torch.zeros(2, 10)
    previous = torch.stack((torch.ones(10), torch.full((10,), math.nan)))
    metrics = tracking_metrics(current, current, 0, 1, previous_actions=previous,
                               rolling_valid=torch.tensor([True, False]))
    assert metrics["rolling_mse"].tolist() == [1, 0]
    assert metrics["previous_overlap_mse"].tolist() == [1, 0]
    assert torch.isfinite(metrics["rolling_penalty"]).all()
    absent = tracking_metrics(current, current, 0, 1)
    assert not absent["rolling_valid"].any()
    assert torch.count_nonzero(absent["rolling_mse"]) == 0


@pytest.mark.parametrize("kwargs", [
    {"return_center": math.nan}, {"return_center": math.inf}, {"return_center": [0]},
    {"return_scale": 0}, {"return_scale": -1}, {"return_scale": math.nan},
    {"return_scale": math.inf}, {"return_scale": [1]}, {"return_scale": 1e-50},
    {"actions": torch.zeros(2, 9)}, {"actions": torch.zeros(10)},
    {"actions": torch.full((2, 10), math.nan)},
    {"future_returns": torch.full((2, 10), math.inf)},
    {"previous_actions": torch.zeros(2, 9)},
    {"previous_actions": torch.full((2, 10), math.nan)},
    {"rolling_valid": torch.tensor([True, True])},
    {"rolling_valid": torch.ones(2), "previous_actions": torch.zeros(2, 10)},
    {"rolling_valid": torch.ones(2, 1, dtype=torch.bool), "previous_actions": torch.zeros(2, 10)},
    {"reward_contract": {}},
])
def test_invalid_inputs_and_configuration_are_rejected(kwargs):
    from gear_sonic.finance.rewards import tracking_metrics

    arguments = {"actions": torch.zeros(2, 10), "future_returns": torch.zeros(2, 10),
                 "return_center": 0, "return_scale": 1}
    arguments.update(kwargs)
    with pytest.raises(ValueError):
        tracking_metrics(**arguments)


def test_leading_dimensions_and_scalar_horizons_are_preserved():
    from gear_sonic.finance.rewards import tracking_metrics

    for shape in ((10,), (2, 3, 10)):
        values = torch.zeros(shape)
        metrics = tracking_metrics(values, values, 0, 1)
        assert all(value.shape == shape[:-1] for value in metrics.values())


def test_explicit_legacy_contract_preserves_exact_old_keys_and_formula():
    from gear_sonic.finance.rewards import LEGACY_TRACKING_REWARD_CONTRACT, tracking_metrics

    assert LEGACY_TRACKING_REWARD_CONTRACT == {
        "version": 1, "name": "monthly_and_cumulative_exponential_tracking",
        "monthly_weight": 0.5, "cumulative_weight": 0.5,
        "target": "unclipped_monthly_log_return_normalized_by_checkpoint_center_scale",
        "cumulative_normalization": "sqrt_horizon",
    }
    actions, raw = torch.arange(20).reshape(2, 10).float() / 7, torch.linspace(-2, 3, 20).reshape(2, 10)
    errors = actions.double() - (raw.double() - torch.tensor(0.2).double()) / torch.tensor(0.3).double()
    month = errors.square().mean(-1)
    path = (errors.cumsum(-1) / torch.arange(1, 11).double().sqrt()).square().mean(-1)
    expected = (0.5 * ((-month).exp() + (-path).exp())).float()
    metrics = tracking_metrics(actions, raw, 0.2, 0.3, reward_contract=LEGACY_TRACKING_REWARD_CONTRACT)
    assert set(metrics) == {"reward", "monthly_mse", "cumulative_mse"}
    assert torch.equal(metrics["reward"], expected)
    assert torch.equal(metrics["monthly_mse"], month)
    assert torch.equal(metrics["cumulative_mse"], path)


def test_summary_uses_sample_means_valid_only_rolling_and_component_quantiles():
    from gear_sonic.finance.rewards import summarize_tracking_metrics, tracking_metrics

    actions = torch.arange(6).reshape(2, 3, 1).expand(2, 3, 10).float()
    raw = torch.zeros_like(actions)
    previous = torch.zeros_like(actions)
    valid = torch.tensor([[False, True, False], [True, False, True]])
    metrics = tracking_metrics(actions, raw, 0, 1, previous_actions=previous, rolling_valid=valid)
    summary = summarize_tracking_metrics(metrics)
    assert summary["sample_count"] == 6
    assert summary["rolling_valid_count"] == 3
    for name in ("reward", "monthly_mse", "cumulative_mse", "change_mse", "volatility_mse",
                 "reward_tracking", "reward_total", *(f"contribution_{c}" for c in COMPONENTS)):
        assert summary[name] == pytest.approx(metrics[name].double().mean().item())
    for name in ("rolling_mse", "rolling_penalty", "previous_overlap_mse"):
        assert summary[name] == pytest.approx(metrics[name][valid].mean().item())
    for component in COMPONENTS:
        name = f"reward_{component}"
        values = metrics[name].flatten()
        for quantile in (10, 50, 90):
            assert summary[f"{name}_p{quantile}"] == pytest.approx(torch.quantile(values, quantile / 100).item())
        assert summary[f"{name}_below_005"] == (values < 0.05).double().mean().item()
        assert summary[f"{name}_above_095"] == (values > 0.95).double().mean().item()
    for horizon in HORIZONS:
        mse = metrics[f"cumulative_squared_error_{horizon}m"].mean().item()
        assert summary[f"cumulative_squared_error_{horizon}m"] == mse
        assert summary[f"cumulative_rmse_{horizon}m"] == pytest.approx(math.sqrt(mse))
        assert summary[f"direction_accuracy_{horizon}m"] == pytest.approx(1 / 6)
        assert summary[f"direction_neutral_count_{horizon}m"] == 6
        assert summary[f"direction_negative_count_{horizon}m"] == 0
        assert summary[f"direction_positive_count_{horizon}m"] == 0
    assert all(type(value) in (int, float) or value is None for value in summary.values())
    json.dumps(summary, allow_nan=False)
    without_quantiles = summarize_tracking_metrics(metrics, include_quantiles=False)
    assert all(not any(suffix in name for suffix in ("_p10", "_p50", "_p90", "_below_005", "_above_095"))
               for name in without_quantiles)


def test_summary_with_no_valid_overlap_serializes_null_and_legacy_remains_small():
    from gear_sonic.finance.rewards import (
        LEGACY_TRACKING_REWARD_CONTRACT, summarize_tracking_metrics, tracking_metrics,
    )

    values = torch.zeros(1, 10)
    summary = summarize_tracking_metrics(tracking_metrics(values, values, 0, 1))
    assert summary["rolling_valid_count"] == 0
    for name in ("rolling_mse", "rolling_penalty", "previous_overlap_mse"):
        assert summary[name] is None
    assert '"rolling_mse": null' in json.dumps(summary, allow_nan=False)
    legacy = tracking_metrics(values, values, 0, 1, reward_contract=LEGACY_TRACKING_REWARD_CONTRACT)
    assert summarize_tracking_metrics(legacy) == {
        "sample_count": 1, "reward": 1.0, "monthly_mse": 0.0, "cumulative_mse": 0.0,
    }


@pytest.mark.parametrize("field,value", [
    ("rolling_valid", torch.zeros(2)),
    ("monthly_mse", torch.zeros(3, dtype=torch.float64)),
    ("monthly_mse", torch.full((2,), math.inf, dtype=torch.float64)),
])
def test_summary_rejects_malformed_shapes_masks_and_nonfinite_values(field, value):
    from gear_sonic.finance.rewards import summarize_tracking_metrics, tracking_metrics

    values = torch.zeros(2, 10)
    metrics = tracking_metrics(values, values, 0, 1)
    metrics[field] = value
    with pytest.raises(ValueError):
        summarize_tracking_metrics(metrics)
