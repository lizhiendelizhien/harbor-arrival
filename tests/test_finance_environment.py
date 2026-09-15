import copy
import math

import pytest
import torch


def period(index):
    return f"{2020 + index // 12:04d}-{index % 12 + 1:02d}"


def sequence(length=4, offset=0, symbol="AAA"):
    current = torch.arange(length * 16, dtype=torch.float32).reshape(length, 16) + offset
    future = torch.arange(length * 150, dtype=torch.float32).reshape(length, 10, 15) / 100
    return {
        "symbol": symbol,
        "periods": [period(index) for index in range(length)],
        "target_end_period": period(length - 1 + 10),
        "current_state": current,
        "future_reference": future,
        "future_mask": torch.ones(length, 10, 15, dtype=torch.bool),
    }


def calendar_sequence(length=5):
    source = sequence(length=length)
    calendar_returns = torch.tensor([
        (index % 5 - 2) * 0.1 + index * 0.003 for index in range(length + 10)
    ])
    source["future_reference"][..., 0] = calendar_returns[1:].unfold(0, 10, 1)
    return source


def environment(sequences=None, **kwargs):
    from gear_sonic.finance.environment import MonthlyTrackingEnv

    arguments = {"num_envs": 2, "return_center": 0.1, "return_scale": 0.2}
    arguments.update(kwargs)
    return MonthlyTrackingEnv([sequence()] if sequences is None else sequences, **arguments)


def test_observation_shapes_and_initial_critic_order():
    source = sequence()
    env = environment([source])
    obs = env.observe()
    assert env.num_envs == 2
    assert env.device == torch.device("cpu")
    assert env.history_length == 10
    assert env.horizon == 10
    assert env.critic_obs_dim == 410
    assert {key: tuple(value.shape) for key, value in obs.items()} == {
        "actor_obs": (2, 16), "future_reference": (2, 10, 15),
        "future_mask": (2, 10, 15), "critic_obs": (2, 410),
        "encoder_mask_type": (2, 1), "encoder_noise": (2, 10, 15),
    }
    assert obs["future_mask"].dtype == torch.bool
    assert obs["future_mask"].all()
    critic = obs["critic_obs"]
    assert torch.count_nonzero(critic[:, :144]) == 0
    torch.testing.assert_close(critic[:, 144:160], source["current_state"][0].expand(2, -1))
    assert torch.count_nonzero(critic[:, 160:260]) == 0
    torch.testing.assert_close(critic[:, 260:], source["future_reference"][0].flatten().expand(2, -1))
    assert all(torch.isfinite(value).all() for value in obs.values())


def test_exact_normalized_path_has_unit_reward_and_detached_diagnostics():
    env = environment()
    target = (env.observe()["future_reference"][..., 0] - 0.1) / 0.2
    _, reward, done, info = env.step(target.requires_grad_())
    torch.testing.assert_close(reward, torch.ones(2))
    assert not done.any()
    assert done.dtype == torch.bool
    for name in ("monthly_mse", "cumulative_mse", "change_mse", "volatility_mse"):
        torch.testing.assert_close(info[name], torch.zeros_like(info[name]), atol=1e-10, rtol=0)
    for name in ("reward_month", "reward_path", "reward_change", "reward_vol",
                 "reward_tracking", "reward_total"):
        torch.testing.assert_close(info[name], torch.ones(2, dtype=torch.float64))
    assert "reward" not in info
    assert not info["rolling_valid"].any()
    for horizon in (1, 3, 6, 10):
        for metric in ("cumulative_squared_error", "direction_correct", "direction_negative",
                       "direction_neutral", "direction_positive"):
            assert info[f"{metric}_{horizon}m"].shape == (2,)
    for name, value in info.items():
        assert not value.requires_grad
        assert value.dtype == (torch.bool if name == "rolling_valid" else torch.float64)
    assert not reward.requires_grad


def test_reward_uses_four_independently_calculated_normalized_tracking_errors():
    env = environment()
    target = (env.observe()["future_reference"][..., 0] - 0.1) / 0.2
    errors = torch.stack((torch.ones(10), torch.tensor([1.0, -1.0] * 5)))
    actions = target + errors
    actual_target = (env.observe()["future_reference"][..., 0].double()
                     - env.return_center.double()) / env.return_scale.double()
    actual_errors = actions.double() - actual_target
    expected_errors = {
        "monthly_mse": actual_errors.square().mean(-1),
        "cumulative_mse": torch.stack([
            actual_errors[:, :horizon].sum(-1).square() / horizon
            for horizon in range(1, 11)
        ], dim=-1).mean(-1),
        "change_mse": (actual_errors[:, 1:] - actual_errors[:, :-1]).square().mean(-1) / 2,
        "volatility_mse": (
            (actions.double() - actions.double().mean(-1, keepdim=True)).square().mean(-1).sqrt()
            - (actual_target - actual_target.mean(-1, keepdim=True)).square().mean(-1).sqrt()
        ).square(),
    }
    _, reward, _, info = env.step(actions)
    expected = sum(
        weight * (-expected_errors[metric]).exp()
        for metric, weight in zip(expected_errors, (0.3, 0.4, 0.2, 0.1))
    )
    torch.testing.assert_close(reward, expected.float())
    for component, metric, weight in zip(
        ("month", "path", "change", "vol"), expected_errors, (0.3, 0.4, 0.2, 0.1),
    ):
        component_reward = (-expected_errors[metric]).exp()
        torch.testing.assert_close(info[metric], expected_errors[metric])
        torch.testing.assert_close(info[f"reward_{component}"], component_reward)
        torch.testing.assert_close(info[f"contribution_{component}"], weight * component_reward)
    torch.testing.assert_close(info["reward_tracking"], expected)
    torch.testing.assert_close(info["reward_total"], expected)
    assert (reward < 1).all()


@pytest.mark.parametrize("reference", [
    [0.0] * 10,
    [-0.2] * 10,
    [0.03, 0.01, -0.02, -1.5, -0.3, 0.5, -0.01, 0.01, 0.02, 0.0],
])
def test_exact_flat_negative_and_crash_references_maximize_all_four_components(reference):
    source = sequence()
    source["future_reference"][..., 0] = torch.tensor(reference)
    env = environment([source])
    target = (env.observe()["future_reference"][..., 0] - env.return_center) / env.return_scale
    _, reward, done, info = env.step(target)
    torch.testing.assert_close(reward, torch.ones(2))
    assert not done.any()
    for name in ("reward_month", "reward_path", "reward_change", "reward_vol"):
        torch.testing.assert_close(info[name], torch.ones(2, dtype=torch.float64))


def test_environment_exposes_independent_v1_contract_metadata_and_compatibility_imports():
    from gear_sonic.finance.environment import TRACKING_REWARD_CONTRACT, tracking_metrics

    left = environment()
    right = environment()
    assert left.reward_contract["name"] == "financial_tracking_v1"
    assert left.reward_contract == right.reward_contract == TRACKING_REWARD_CONTRACT
    left.reward_contract["name"] = "local_change"
    assert right.reward_contract == TRACKING_REWARD_CONTRACT
    assert right.reward_contract["name"] == "financial_tracking_v1"
    assert callable(tracking_metrics)


def test_finite_extreme_normalization_cannot_overflow_tracking_reward():
    source = sequence()
    source["future_reference"][..., 0] = torch.tensor([1.0, -1.0] * 5)
    env = environment([source], return_scale=1e-40)
    _, reward, _, info = env.step(torch.zeros(2, 10))
    assert reward.dtype == torch.float32
    assert torch.isfinite(reward).all()
    for name, value in info.items():
        assert torch.isfinite(value).all()
        assert value.dtype == (torch.bool if name == "rolling_valid" else torch.float64)
    torch.testing.assert_close(reward, torch.zeros_like(reward))


def test_calendar_aligned_exact_predictions_have_zero_rolling_error_after_first_step():
    env = environment([calendar_sequence()])
    previous_actions = None
    for step in range(3):
        target = (env.observe()["future_reference"][..., 0] - env.return_center) / env.return_scale
        _, reward, done, info = env.step(target)
        torch.testing.assert_close(reward, torch.ones(2))
        assert not done.any()
        assert info["rolling_valid"].tolist() == [step > 0, step > 0]
        for name in ("rolling_mse", "rolling_penalty", "previous_overlap_mse"):
            torch.testing.assert_close(info[name], torch.zeros(2, dtype=torch.float64), atol=1e-12, rtol=0)
        if previous_actions is not None:
            assert (target[:, :9] - previous_actions[:, :9]).square().mean() > 0.1
        previous_actions = target


def test_correcting_previous_overlap_is_diagnostic_only_and_uses_sampled_actions():
    env = environment([calendar_sequence()])
    target = (env.observe()["future_reference"][..., 0] - env.return_center) / env.return_scale
    previous_actions = target + 2
    env.step(previous_actions)
    target = (env.observe()["future_reference"][..., 0] - env.return_center) / env.return_scale
    _, reward, _, info = env.step(target)
    assert info["rolling_valid"].all()
    expected = (target[:, :9].double() - previous_actions[:, 1:].double()).square().mean(-1)
    torch.testing.assert_close(info["rolling_mse"], expected)
    torch.testing.assert_close(info["rolling_penalty"], 1 - (-expected).exp())
    torch.testing.assert_close(info["previous_overlap_mse"], torch.full((2,), 4.0, dtype=torch.float64))
    torch.testing.assert_close(reward, torch.ones(2))
    torch.testing.assert_close(info["reward_total"], info["reward_tracking"])


def test_terminal_diagnostics_precede_same_symbol_reset_and_full_reset_invalidates_overlap():
    env = environment([calendar_sequence(length=2)])
    env.step(torch.ones(2, 10))
    _, _, done, info = env.step(torch.full((2, 10), 3.0))
    assert done.all()
    assert info["rolling_valid"].all()
    torch.testing.assert_close(info["rolling_mse"], torch.full((2,), 4.0, dtype=torch.float64))
    assert env._episode_lengths.tolist() == [0, 0]
    _, _, _, info = env.step(torch.ones(2, 10))
    assert not info["rolling_valid"].any()
    assert not info["rolling_mse"].any()
    env.reset()
    _, _, _, info = env.step(torch.ones(2, 10))
    assert not info["rolling_valid"].any()
    assert not info["previous_overlap_mse"].any()


def test_partial_reset_invalidates_only_reset_slot_and_zero_actions_are_valid_history():
    env = environment([calendar_sequence()])
    env.step(torch.zeros(2, 10))
    env._reset_slots(torch.tensor([0]))
    _, _, _, info = env.step(torch.ones(2, 10))
    assert info["rolling_valid"].tolist() == [False, True]
    torch.testing.assert_close(info["rolling_mse"], torch.tensor([0.0, 1.0], dtype=torch.float64))
    torch.testing.assert_close(info["rolling_penalty"], torch.tensor([0.0, 1 - math.exp(-1)], dtype=torch.float64))
    assert env._episode_lengths.tolist() == [1, 2]
    # Observing or collecting a fresh rollout must not invalidate continuing clips.
    env.observe()
    _, _, _, info = env.step(torch.full((2, 10), 2.0))
    assert info["rolling_valid"].all()
    torch.testing.assert_close(info["rolling_mse"], torch.ones(2, dtype=torch.float64))


def test_reward_uses_unclipped_targets_and_preserves_effective_normalizers():
    source = sequence()
    source["future_reference"][..., 0] = torch.tensor([-10.0, 10.0] * 5)
    env = environment([source], return_center=0.7, return_scale=0.03)
    center_before, scale_before = env.return_center.clone(), env.return_scale.clone()
    target = (env.observe()["future_reference"][..., 0] - env.return_center) / env.return_scale
    assert target.abs().max() > 300
    _, reward, _, info = env.step(target)
    torch.testing.assert_close(reward, torch.ones(2))
    torch.testing.assert_close(info["monthly_mse"], torch.zeros(2, dtype=torch.float64), atol=1e-8, rtol=0)
    env.reset()
    torch.testing.assert_close(env.return_center, center_before)
    torch.testing.assert_close(env.return_scale, scale_before)


def test_market_transitions_ignore_actions_and_histories_are_chronological():
    source = sequence(length=5)
    env = environment([source], history_length=2)
    first = torch.stack((torch.full((10,), 2.0), torch.full((10,), -3.0)))
    obs, _, _, _ = env.step(first)
    assert env.critic_obs_dim == 202
    torch.testing.assert_close(obs["actor_obs"], source["current_state"][1].expand(2, -1))
    torch.testing.assert_close(obs["critic_obs"][:, :32], source["current_state"][:2].flatten().expand(2, -1))
    assert torch.count_nonzero(obs["critic_obs"][:, 32:42]) == 0
    torch.testing.assert_close(obs["critic_obs"][:, 42:52], first)
    second = torch.full((2, 10), 4.0)
    obs, _, _, _ = env.step(second)
    torch.testing.assert_close(obs["critic_obs"][:, :32], source["current_state"][1:3].flatten().expand(2, -1))
    torch.testing.assert_close(obs["critic_obs"][:, 32:52], torch.stack((first, second), dim=1).flatten(1))
    torch.testing.assert_close(obs["critic_obs"][:, 52:], source["future_reference"][2].flatten().expand(2, -1))


def test_fixed_clip_terminal_autoresets_all_histories_and_reports_episode():
    source = sequence(length=2)
    env = environment([source], history_length=2)
    _, first_reward, done, _ = env.step(torch.ones(2, 10))
    assert not done.any()
    obs, second_reward, done, info = env.step(torch.ones(2, 10) * 2)
    assert done.all()
    assert "time_outs" not in info
    torch.testing.assert_close(obs["actor_obs"], source["current_state"][0].expand(2, -1))
    assert torch.count_nonzero(obs["critic_obs"][:, :16]) == 0
    assert torch.count_nonzero(obs["critic_obs"][:, 32:52]) == 0
    torch.testing.assert_close(info["episode"]["env_ids"], torch.arange(2))
    torch.testing.assert_close(info["episode"]["return"], first_reward + second_reward)
    torch.testing.assert_close(info["episode"]["length"], torch.full((2,), 2, dtype=torch.long))


def test_terminal_reset_preserves_nonterminal_slot_history():
    source = calendar_sequence(length=3)
    env = environment([source], history_length=2)
    env.step(torch.full((2, 10), 0.5))
    # Stagger clips with a real partial reset, preserving calendar-contiguous actions.
    env._reset_slots(torch.tensor([1]))
    env.step(torch.full((2, 10), 1.0))
    obs, _, done, info = env.step(torch.full((2, 10), 2.0))
    assert done.tolist() == [True, False]
    torch.testing.assert_close(obs["actor_obs"], source["current_state"][[0, 2]])
    assert torch.count_nonzero(obs["critic_obs"][0, 32:52]) == 0
    torch.testing.assert_close(obs["critic_obs"][1, 32:52], torch.tensor([1.0] * 10 + [2.0] * 10))
    assert info["episode"]["env_ids"].tolist() == [0]
    assert info["rolling_valid"].all()
    torch.testing.assert_close(info["rolling_mse"], torch.ones(2, dtype=torch.float64))
    _, _, _, info = env.step(torch.full((2, 10), 3.0))
    assert info["rolling_valid"].tolist() == [False, True]
    torch.testing.assert_close(info["rolling_mse"], torch.tensor([0.0, 1.0], dtype=torch.float64))


def test_seeded_sampling_is_reproducible_independent_and_does_not_touch_global_rng():
    sources = [sequence(length=1, offset=index * 1000, symbol=f"S{index}") for index in range(5)]
    rng_before = torch.random.get_rng_state().clone()
    left = environment(sources, seed=31, num_envs=32)
    right = environment(sources, seed=31, num_envs=32)
    assert torch.equal(torch.random.get_rng_state(), rng_before)
    initial = left.observe()["actor_obs"].clone()
    assert initial[:, 0].unique().numel() > 1
    for _ in range(4):
        torch.testing.assert_close(left.observe()["actor_obs"], right.observe()["actor_obs"])
        left.step(torch.zeros(32, 10))
        right.step(torch.zeros(32, 10))
    assert not torch.equal(initial, left.observe()["actor_obs"])


def test_reset_returns_initial_history_and_observations_are_safe_to_store_or_mutate():
    env = environment()
    saved = env.observe()
    expected = {key: value.clone() for key, value in saved.items()}
    env.step(torch.ones(2, 10))
    for key in saved:
        torch.testing.assert_close(saved[key], expected[key])
    reset_obs = env.reset()
    for key in reset_obs:
        if not key.startswith("encoder_"):
            torch.testing.assert_close(reset_obs[key], expected[key])
    assert not torch.equal(reset_obs["encoder_noise"], expected["encoder_noise"])
    reset_expected = {key: value.clone() for key, value in reset_obs.items()}
    for value in reset_obs.values():
        value.zero_()
    for key, value in env.observe().items():
        torch.testing.assert_close(value, reset_expected[key])


def test_source_tensors_are_copied_into_the_environment_bank():
    source = sequence()
    env = environment([source])
    expected = env.observe()
    source["current_state"].zero_()
    source["future_reference"].zero_()
    source["future_mask"].zero_()
    for key, value in env.observe().items():
        torch.testing.assert_close(value, expected[key])


@pytest.mark.parametrize("arguments", [
    {"num_envs": 0}, {"num_envs": 1.5}, {"num_envs": True},
    {"history_length": 0}, {"history_length": 1.5},
    {"return_scale": 0}, {"return_scale": -1}, {"return_scale": math.nan},
    {"return_scale": math.inf}, {"return_center": math.nan},
    {"return_center": math.inf}, {"return_scale": [1, 2]},
])
def test_invalid_configuration_is_rejected(arguments):
    with pytest.raises(ValueError):
        environment(**arguments)


@pytest.mark.parametrize("field,value", [
    ("symbol", ["AAA", "BBB"]), ("symbol", ""),
    ("periods", ["2020-01", "2020-02", "2020-04", "2020-05"]),
    ("periods", ["2020-01", "2020-02", "2020-02", "2020-04"]),
    ("periods", ["2020-13", "2021-02", "2021-03", "2021-04"]),
    ("periods", ["2020-01"]), ("target_end_period", "2021-01"),
    ("current_state", torch.zeros(4, 15)),
    ("current_state", torch.full((4, 16), math.nan)),
    ("future_reference", torch.zeros(4, 9, 15)),
    ("future_reference", torch.full((4, 10, 15), math.inf)),
    ("future_mask", torch.zeros(4, 10, 15, dtype=torch.bool)),
    ("future_mask", torch.ones(4, 10, 15)),
])
def test_invalid_sequence_metadata_values_and_shapes_are_rejected(field, value):
    source = sequence()
    source[field] = value
    with pytest.raises(ValueError):
        environment([source])


def test_empty_and_mixed_length_sequence_pools_are_rejected():
    for sources in ([], [sequence(length=0)], [sequence(length=2), sequence(length=3)]):
        with pytest.raises(ValueError):
            environment(sources)


def test_one_missing_future_feature_is_rejected():
    source = copy.deepcopy(sequence())
    source["future_mask"][1, 3, 7] = False
    with pytest.raises(ValueError, match="complete"):
        environment([source])


@pytest.mark.parametrize("actions", [
    torch.zeros(2, 9), torch.zeros(1, 10), torch.full((2, 10), math.nan),
    torch.full((2, 10), math.inf),
])
def test_invalid_actions_fail_without_advancing(actions):
    env = environment()
    env.step(torch.ones(2, 10))
    before = env.observe()
    state_before = {
        name: getattr(env, name).clone() for name in (
            "_sequence_ids", "_step_ids", "_episode_returns", "_episode_lengths",
            "_state_history", "_action_history",
        )
    }
    rng_before = env._generator.get_state().clone()
    with pytest.raises(ValueError):
        env.step(actions)
    for key, value in env.observe().items():
        torch.testing.assert_close(value, before[key])
    for name, value in state_before.items():
        torch.testing.assert_close(getattr(env, name), value)
    assert torch.equal(env._generator.get_state(), rng_before)
