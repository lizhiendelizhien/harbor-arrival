import math

import pytest
import torch

from gear_sonic.finance.ppo import PPOConfig, clipped_ppo_losses, compute_gae


def test_reference_gae_terminal_mask_and_sample_normalization():
    rewards = torch.tensor([[1.0, 2.0, 3.0], [2.0, 1.0, 4.0]])
    values = torch.tensor([[0.5, 0.6, 0.7], [0.2, 0.4, 0.6]])
    dones = torch.tensor([[False, True, False], [False, False, True]])
    returns, advantages = compute_gae(rewards, values, dones, torch.tensor([0.8, 99.0]), 0.9, 0.8)
    raw = torch.tensor([[2.048, 1.4, 3.02], [4.74336, 3.588, 3.4]])
    torch.testing.assert_close(returns, raw + values)
    torch.testing.assert_close(advantages, (raw - raw.mean()) / (raw.std() + 1e-8))


def test_single_transition_advantage_is_finite_zero():
    returns, advantages = compute_gae(torch.ones(1, 1), torch.zeros(1, 1),
                                     torch.ones(1, 1, dtype=torch.bool), torch.ones(1), 0.99, 0.95)
    torch.testing.assert_close(returns, torch.ones(1, 1))
    torch.testing.assert_close(advantages, torch.zeros(1, 1))


def test_clipped_policy_value_entropy_and_reference_kl_math():
    config = PPOConfig()
    losses = clipped_ppo_losses(
        new_logprobs=torch.tensor([[math.log(1.5), math.log(0.5)]]),
        old_logprobs=torch.zeros(1, 2), advantages=torch.tensor([[2.0, -3.0]]),
        new_values=torch.tensor([[2.0, -1.0]]), old_values=torch.zeros(1, 2),
        returns=torch.tensor([[3.0, -2.0]]), entropy=torch.tensor([[2.0, 4.0]]),
        means=torch.tensor([[[0.0], [1.0]]]), old_means=torch.zeros(1, 2, 1),
        sigmas=torch.full((1, 2, 1), 2.0), old_sigmas=torch.ones(1, 2, 1),
        config=config,
    )
    assert losses["policy_loss"].item() == pytest.approx(0.0, abs=1e-6)
    assert losses["value_loss"].item() == pytest.approx((2.8**2 + 1.8**2) / 2)
    assert losses["entropy"].item() == pytest.approx(3.0)
    assert losses["kl"].item() == pytest.approx(math.log(2.0 + 1e-5) + 1.5 / 8 - 0.5)
    assert losses["ppo_loss"].item() == pytest.approx((2.8**2 + 1.8**2) / 2 - 0.03)


def test_config_matches_effective_stage5_defaults_and_rejects_invalid_bounds():
    config = PPOConfig()
    assert (config.rollout_steps, config.epochs, config.num_minibatches) == (24, 5, 4)
    assert (config.learning_rate, config.max_grad_norm) == (2e-5, 0.1)
    assert (config.adaptive_lr_min, config.adaptive_lr_max) == (1e-5, 2e-4)
    with pytest.raises(ValueError, match="positive"):
        PPOConfig(rollout_steps=0)
    with pytest.raises(ValueError, match="learning rate"):
        PPOConfig(adaptive_lr_min=1e-3)


def make_trainer(length=37, num_envs=4, **config_kwargs):
    from gear_sonic.finance.ppo import FinancialPPOTrainer
    from gear_sonic.finance.environment import MonthlyTrackingEnv
    from gear_sonic.finance.model import FinancialSonicConfig
    from gear_sonic.finance.policy import make_actor_critic

    torch.manual_seed(19)
    model_config = FinancialSonicConfig(mlp_hidden_dims=(32, 32), d_model=32,
                                      num_heads=2, num_layers=2, ffn_dim=64)
    actor, critic = make_actor_critic(model_config, critic_obs_dim=410, critic_hidden_dims=(32, 32))
    current, future = torch.randn(length, 16), torch.randn(length, 10, 15) * 0.1
    actor.actor_module.model.fit_normalizers(current, future)
    def period(index):
        return f"{2020 + index // 12:04d}-{index % 12 + 1:02d}"
    sequence = {
        "symbol": "AAA", "periods": [period(index) for index in range(length)],
        "target_end_period": period(length + 9), "current_state": current,
        "future_reference": future, "future_mask": torch.ones_like(future, dtype=torch.bool),
    }
    normalizer = actor.actor_module.model.future_normalizer
    second = {**sequence, "symbol": "BBB", "current_state": -current,
              "future_reference": future.flip(0)}
    env = MonthlyTrackingEnv([sequence, second], num_envs=num_envs,
                             return_center=normalizer.center[0], return_scale=normalizer.scale[0])
    kwargs = {"epochs": 1, "num_minibatches": 2}
    kwargs.update(config_kwargs)
    return FinancialPPOTrainer(actor, critic, env, PPOConfig(**kwargs))


def evaluate_batch_means(trainer, batch):
    from gear_sonic.trl.utils.rl import compute_episode_attnmask

    trainer.actor.eval()
    with torch.no_grad():
        trainer.actor.update_distribution(
            batch["obs"], episode_attnmask=compute_episode_attnmask(batch["dones"]),
            kv_prefix_state=batch["prefix"],
            kv_prefix_dones=batch["dones"] if batch["prefix"] is not None else None,
            loss_start_index=0,
        )
    return trainer.actor.action_mean.clone()


def test_rollout_preserves_time_and_detached_prefix_means_across_window_and_resets():
    trainer = make_trainer()
    rms_count = trainer.critic.running_mean_std.count.clone()
    first = trainer.collect_rollout()
    assert first["prefix"] is None
    assert first["obs"]["actor_obs"].shape == (4, 24, 16)
    assert first["actions"].shape == (4, 24, 10)
    assert first["values"].shape == first["returns"].shape == (4, 24)
    assert first["bootstrap"].shape == (4,)
    for _ in range(3):
        batch = trainer.collect_rollout()
        assert batch["prefix"] is not None
        assert batch["prefix"]["cache_valid"].shape[1] <= 31
        assert not batch["prefix"]["past_key_values"].requires_grad
        torch.testing.assert_close(evaluate_batch_means(trainer, batch), batch["means"],
                                   atol=3e-6, rtol=3e-5)
        for key in ("actions", "logprobs", "means", "sigmas", "values", "returns", "advantages"):
            assert not batch[key].requires_grad
    torch.testing.assert_close(trainer.critic.running_mean_std.count, rms_count)


def test_terminal_at_rollout_boundary_clears_prefix_before_capture():
    trainer = make_trainer(length=24)
    first = trainer.collect_rollout()
    assert first["dones"][:, -1].all()
    second = trainer.collect_rollout()
    prefix = second["prefix"]
    assert prefix is None or not prefix["cache_valid"].any()
    torch.testing.assert_close(evaluate_batch_means(trainer, second), second["means"],
                               atol=3e-6, rtol=3e-5)


def test_update_trains_actor_encoder_kin_and_critic_without_world_losses():
    trainer = make_trainer(epochs=2)
    trainer.collect_rollout()
    batch = trainer.collect_rollout()
    model = trainer.actor.actor_module.model
    modules = {"encoder": model.encoder, "kin": model.kin, "dyn": model.dyn,
               "critic": trainer.critic}
    before = {name: [p.detach().clone() for p in module.parameters()]
              for name, module in modules.items()}
    cache_before = trainer.actor.get_rollout_cache_state()
    count_before = trainer.critic.running_mean_std.count.item()
    metrics = trainer.update(batch)
    assert metrics["updates"] == 4
    assert all(isinstance(value, (int, float)) and math.isfinite(value) for value in metrics.values())
    assert set(trainer.actor.aux_losses) == {"kin", "cycle"}
    for name, module in modules.items():
        assert any(not torch.equal(left, right) for left, right in zip(before[name], module.parameters())), name
        assert all(torch.isfinite(p).all() for p in module.parameters())
        assert all(torch.isfinite(p.grad).all() for p in module.parameters() if p.grad is not None)
    assert trainer.critic.running_mean_std.count.item() > count_before
    for key, value in trainer.actor.get_rollout_cache_state().items():
        torch.testing.assert_close(value, cache_before[key])
    assert trainer.optimizer.defaults["weight_decay"] == 0
    assert len({group["lr"] for group in trainer.optimizer.param_groups}) == 1
    next_batch = trainer.collect_rollout()
    torch.testing.assert_close(evaluate_batch_means(trainer, next_batch), next_batch["means"],
                               atol=3e-6, rtol=3e-5)


def test_checkpoint_restores_models_optimizer_iteration_and_restarts_episode(tmp_path):
    trainer = make_trainer(rollout_steps=4)
    metrics = trainer.train_iteration()
    assert metrics["iteration"] == trainer.iteration == 1
    path = tmp_path / "checkpoint.pt"
    trainer.save_checkpoint(path)
    checkpoint = torch.load(path, weights_only=True)
    assert checkpoint["critic_config"] == {"input_dim": 410, "hidden_dims": (32, 32)}
    assert checkpoint["env_config"]["sequence_length"] == 37
    assert checkpoint["env_config"]["history_length"] == 10
    assert checkpoint["env_config"]["horizon"] == 10
    assert checkpoint["env_config"]["return_center"] == float(trainer.env.return_center)
    assert checkpoint["env_config"]["return_scale"] == float(trainer.env.return_scale)
    with pytest.raises(FileExistsError):
        trainer.save_checkpoint(path)
    restored = make_trainer(rollout_steps=4)
    restored.load_checkpoint(path)
    assert restored.iteration == 1
    assert restored.actor.steps == 0
    assert restored.actor.get_rollout_cache_state() is None
    assert restored.optimizer.state_dict()["param_groups"] == trainer.optimizer.state_dict()["param_groups"]
    original_optimizer = trainer.optimizer.state_dict()["state"]
    restored_optimizer = restored.optimizer.state_dict()["state"]
    assert original_optimizer
    assert original_optimizer.keys() == restored_optimizer.keys()
    for parameter_id, state in original_optimizer.items():
        for key, value in state.items():
            torch.testing.assert_close(value, restored_optimizer[parameter_id][key])
    for original, loaded in ((trainer.actor, restored.actor), (trainer.critic, restored.critic)):
        for key, value in original.state_dict().items():
            torch.testing.assert_close(value, loaded.state_dict()[key])
    batch = trainer.collect_rollout()
    torch.testing.assert_close(evaluate_batch_means(trainer, batch), evaluate_batch_means(restored, batch))
    with pytest.raises(ValueError, match="config"):
        make_trainer(rollout_steps=5).load_checkpoint(path)


def test_checkpoint_rejects_incompatible_environment_and_reward_normalization(tmp_path):
    trainer = make_trainer()
    path = tmp_path / "checkpoint.pt"
    trainer.save_checkpoint(path)
    for field, value in (("sequence_length", 36), ("return_center", 3.0),
                         ("return_scale", 2.0), ("history_length", 9)):
        restored = make_trainer()
        setattr(restored.env, field, value)
        with pytest.raises(ValueError, match="environment config"):
            restored.load_checkpoint(path)


def test_minibatches_cannot_exceed_environment_streams():
    with pytest.raises(ValueError, match="minibatch"):
        make_trainer(num_envs=1)


def test_single_sample_update_preserves_finite_critic_running_statistics():
    trainer = make_trainer(num_envs=1, rollout_steps=1, num_minibatches=1)
    metrics = trainer.train_iteration()
    assert all(math.isfinite(value) for value in metrics.values() if value is not None)
    assert metrics["rolling_valid_count"] == 0
    assert metrics["rolling_mse"] is None
    assert torch.isfinite(trainer.critic.running_mean_std.running_var).all()
    assert trainer.critic.running_mean_std.count.item() == 1


def test_reference_adaptive_kl_updates_all_optimizer_groups():
    trainer = make_trainer()
    trainer._adjust_learning_rate(0.1)
    assert trainer.learning_rate == pytest.approx(2e-5 / 1.5)
    trainer._adjust_learning_rate(0.1)
    assert trainer.learning_rate == pytest.approx(1e-5)
    trainer._adjust_learning_rate(0.001)
    assert trainer.learning_rate == pytest.approx(1.5e-5)
    for _ in range(10):
        trainer._adjust_learning_rate(0.001)
    assert trainer.learning_rate == pytest.approx(2e-4)
    assert all(group["lr"] == pytest.approx(trainer.learning_rate) for group in trainer.optimizer.param_groups)


def test_rollout_retains_sampled_reward_diagnostics_across_rollout_and_reset():
    trainer = make_trainer(length=5, rollout_steps=3)
    first = trainer.collect_rollout()
    diagnostics = first["tracking_metrics"]
    assert diagnostics["monthly_mse"].dtype == torch.float64
    assert diagnostics["monthly_mse"].shape == (4, 3)
    assert not diagnostics["rolling_valid"][:, 0].any()
    assert diagnostics["rolling_valid"][:, 1:].all()
    normalizer = trainer.actor.actor_module.model.future_normalizer
    target = (first["obs"]["future_reference"][..., 0].double()
              - normalizer.center[0].double()) / normalizer.scale[0].double()
    expected = (first["actions"].double() - target).square().mean(-1)
    torch.testing.assert_close(diagnostics["monthly_mse"], expected)
    assert not torch.equal(expected, (first["means"].double() - target).square().mean(-1))
    second = trainer.collect_rollout()
    assert second["tracking_metrics"]["rolling_valid"].tolist() == [[True, True, False]] * 4


def test_iteration_reports_v1_components_and_finite_json():
    import json

    trainer = make_trainer(length=5, rollout_steps=3)
    metrics = trainer.train_iteration()
    assert metrics["sample_count"] == 12
    assert metrics["rolling_valid_count"] == 8
    contributions = [metrics[f"contribution_{name}"] for name in ("month", "path", "change", "vol")]
    assert sum(contributions) == pytest.approx(metrics["reward_tracking"])
    assert metrics["reward"] == pytest.approx(metrics["reward_total"], abs=1e-7)
    for name in ("month", "path", "change", "vol"):
        assert 0 <= metrics[f"reward_{name}_p10"] <= metrics[f"reward_{name}_p90"] <= 1
    for horizon in (1, 3, 6, 10):
        assert metrics[f"cumulative_rmse_{horizon}m"] >= 0
        assert 0 <= metrics[f"direction_accuracy_{horizon}m"] <= 1
        assert sum(metrics[f"direction_{label}_count_{horizon}m"]
                   for label in ("negative", "neutral", "positive")) == 12
    json.dumps(metrics, allow_nan=False)


@pytest.mark.parametrize("field", ["return_center", "return_scale"])
def test_direct_trainer_rejects_environment_normalization_mismatch(field):
    from gear_sonic.finance.ppo import FinancialPPOTrainer

    trainer = make_trainer()
    setattr(trainer.env, field, getattr(trainer.env, field) + 0.5)
    before = trainer.env._step_ids.clone()
    with pytest.raises(ValueError, match="normalization"):
        FinancialPPOTrainer(trainer.actor, trainer.critic, trainer.env, trainer.config)
    torch.testing.assert_close(trainer.env._step_ids, before)


def test_checkpoint_has_v1_reward_contract_and_new_schema(tmp_path):
    trainer = make_trainer()
    path = tmp_path / "v1.pt"
    trainer.save_checkpoint(path)
    checkpoint = torch.load(path, weights_only=True)
    assert checkpoint["schema_version"] == 4
    assert checkpoint["reward_contract"] == trainer.env.reward_contract
    assert checkpoint["reward_contract"]["name"] == "financial_tracking_v1"


@pytest.mark.parametrize("case", ["schema1", "schema2", "schema3", "missing", "different", "actor_normalization"])
def test_bad_reward_checkpoint_rejected_before_loading_any_state(tmp_path, case):
    trainer = make_trainer()
    path = tmp_path / "original.pt"
    trainer.save_checkpoint(path)
    checkpoint = torch.load(path, weights_only=True)
    if case.startswith("schema"):
        checkpoint["schema_version"] = int(case[-1])
    elif case == "missing":
        checkpoint.pop("reward_contract", None)
    elif case == "different":
        checkpoint["reward_contract"] = {"name": "different_reward"}
    else:
        checkpoint["actor"]["actor_module.model.future_normalizer.center"][0] += 0.5
    for name in checkpoint["actor"]:
        if name.endswith("weight"):
            checkpoint["actor"][name].add_(1.0)
            break
    bad = tmp_path / "bad.pt"
    torch.save(checkpoint, bad)
    before = {name: value.clone() for name, value in trainer.actor.state_dict().items()}
    with pytest.raises(ValueError, match="[Rr]eward|schema|normalization"):
        trainer.load_checkpoint(bad)
    for name, value in trainer.actor.state_dict().items():
        torch.testing.assert_close(value, before[name], atol=0, rtol=0)


def test_checkpoint_records_encoder_denoising_contract(tmp_path):
    trainer = make_trainer()
    path = tmp_path / "denoising.pt"
    trainer.save_checkpoint(path)
    checkpoint = torch.load(path, weights_only=True)
    assert "encoder_denoising_contract" in checkpoint
    from gear_sonic.finance.denoising import encoder_denoising_contract

    assert checkpoint["encoder_denoising_contract"] == encoder_denoising_contract()
    assert checkpoint["env_config"]["encoder_denoising"] is True


@pytest.mark.parametrize("case", ["missing", "different", "disabled"])
def test_bad_denoising_checkpoint_rejected_before_loading_any_state(tmp_path, case):
    trainer = make_trainer()
    path = tmp_path / "original.pt"
    trainer.save_checkpoint(path)
    checkpoint = torch.load(path, weights_only=True)
    if case == "missing":
        checkpoint.pop("encoder_denoising_contract", None)
    elif case == "different":
        checkpoint["encoder_denoising_contract"] = {"version": 99}
    else:
        checkpoint["env_config"]["encoder_denoising"] = False
    for name, value in checkpoint["actor"].items():
        if name.endswith("weight"):
            value.add_(1.0)
            break
    bad = tmp_path / "bad.pt"
    torch.save(checkpoint, bad)
    before = {name: value.clone() for name, value in trainer.actor.state_dict().items()}
    optimizer_before = trainer.optimizer.state_dict()
    with pytest.raises(ValueError, match="denoising|environment config"):
        trainer.load_checkpoint(bad)
    for name, value in trainer.actor.state_dict().items():
        torch.testing.assert_close(value, before[name], atol=0, rtol=0)
    assert trainer.optimizer.state_dict() == optimizer_before


def test_augmented_rollout_replays_same_logprobs_without_random_draws():
    trainer = make_trainer(length=37)
    for _ in range(3):
        batch = trainer.collect_rollout()
        assert "encoder_noise" in batch["obs"]
        assert batch["obs"]["encoder_noise"].shape == (4, 24, 10, 15)
        assert batch["obs"]["encoder_mask_type"].shape == (4, 24, 1)
        rng = torch.random.get_rng_state().clone()
        means = evaluate_batch_means(trainer, batch)
        torch.testing.assert_close(means, batch["means"], atol=3e-6, rtol=3e-5)
        torch.testing.assert_close(trainer.actor.get_actions_log_prob(batch["actions"]),
                                   batch["logprobs"], atol=3e-5, rtol=3e-5)
        torch.testing.assert_close(torch.random.get_rng_state(), rng, atol=0, rtol=0)
        trainer.actor.train()
        from gear_sonic.trl.utils.rl import compute_episode_attnmask
        with torch.no_grad():
            trainer.actor.update_distribution(
                batch["obs"], episode_attnmask=compute_episode_attnmask(batch["dones"]),
                kv_prefix_state=batch["prefix"],
                kv_prefix_dones=batch["dones"] if batch["prefix"] is not None else None,
                loss_start_index=0, is_training=True,
            )
        torch.testing.assert_close(trainer.actor.action_mean, means, atol=0, rtol=0)
        torch.testing.assert_close(torch.random.get_rng_state(), rng, atol=0, rtol=0)


def test_explicit_cache_rebuild_uses_saved_corruption_across_episode_reset():
    from gear_sonic.finance.denoising import prepare_encoder_input

    trainer = make_trainer(length=5, rollout_steps=8)
    trainer.collect_rollout()
    actor, model = trainer.actor, trainer.actor.actor_module.model
    history = {key: value.clone() for key, value in actor.obs_dict_buffer.items()}
    cache = {key: value.clone() for key, value in actor.get_rollout_cache_state().items()}
    rng, encoder_rng = torch.random.get_rng_state().clone(), trainer.env._encoder_generator.get_state().clone()
    seen = []
    hook = model.encoder.register_forward_pre_hook(lambda _, args: seen.append(args[0].clone()))
    try:
        actor._rebuild_rollout_cache()
    finally:
        hook.remove()
    assert len(seen) == 8
    for step, actual in enumerate(seen):
        mask = history["future_mask"][:, step]
        expected = prepare_encoder_input(
            model.future_normalizer(history["future_reference"][:, step], mask), mask,
            encoder_noise=history["encoder_noise"][:, step],
            encoder_mask_type=history["encoder_mask_type"][:, step],
        )
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    for key, value in actor.get_rollout_cache_state().items():
        torch.testing.assert_close(value, cache[key], atol=0, rtol=0)
    torch.testing.assert_close(torch.random.get_rng_state(), rng, atol=0, rtol=0)
    torch.testing.assert_close(trainer.env._encoder_generator.get_state(), encoder_rng, atol=0, rtol=0)


def legacy_checkpoint_payload(checkpoint, schema):
    """Represent the metadata of a real pre-denoising checkpoint fixture."""
    from copy import deepcopy
    from gear_sonic.finance.rewards import LEGACY_TRACKING_REWARD_CONTRACT

    saved = deepcopy(checkpoint)
    saved["schema_version"] = schema
    saved.pop("encoder_denoising_contract")
    saved.pop("training_migrations", None)
    saved["env_config"].pop("encoder_denoising")
    if schema in (1, 2):
        saved.pop("reward_contract")
        if saved.get("reference_config") is not None:
            saved["reference_config"]["reward_contract"] = deepcopy(LEGACY_TRACKING_REWARD_CONTRACT)
    if schema == 1:
        saved.pop("reference_config", None)
        saved.pop("reference_provenance", None)
    return saved


@pytest.mark.parametrize("schema", [1, 2, 3])
def test_legacy_resume_restores_learning_state_and_continues_current_training(tmp_path, schema):
    from gear_sonic.finance.rewards import TRACKING_REWARD_CONTRACT

    source = make_trainer(rollout_steps=4)
    source.train_iteration()
    modern = tmp_path / "modern.pt"
    source.save_checkpoint(modern)
    payload = legacy_checkpoint_payload(torch.load(modern, weights_only=True), schema)
    legacy = tmp_path / "legacy.pt"
    torch.save(payload, legacy)
    original_bytes = legacy.read_bytes()
    restored = make_trainer(rollout_steps=4)
    with pytest.warns(UserWarning, match=f"schema {schema}"):
        restored.load_checkpoint(legacy)
    assert restored.iteration == payload["iteration"] == 1
    assert restored.learning_rate == payload["learning_rate"]
    assert restored.actor.get_rollout_cache_state() is None
    assert restored.actor.steps == 0
    for name in ("actor", "critic"):
        for key, value in getattr(restored, name).state_dict().items():
            torch.testing.assert_close(value, payload[name][key], atol=0, rtol=0)
    torch.testing.assert_close(restored.optimizer.state_dict(), payload["optimizer"], atol=0, rtol=0)
    assert restored.env.encoder_denoising is True
    assert restored.env.reward_contract == TRACKING_REWARD_CONTRACT
    history = restored.training_migrations
    assert len(history) == 1
    assert history[0]["source_schema"] == schema
    assert history[0]["source_iteration"] == 1
    assert history[0]["target_schema"] == 4
    metrics = restored.train_iteration()
    assert metrics["iteration"] == 2
    assert all(math.isfinite(value) for value in metrics.values() if value is not None)
    migrated = tmp_path / "migrated.pt"
    restored.save_checkpoint(migrated)
    saved = torch.load(migrated, weights_only=True)
    assert saved["schema_version"] == 4
    assert saved["training_migrations"] == history
    assert saved["reward_contract"] == TRACKING_REWARD_CONTRACT
    again = make_trainer(rollout_steps=4)
    again.load_checkpoint(migrated)
    assert again.training_migrations == history
    assert again.iteration == 2
    assert legacy.read_bytes() == original_bytes


@pytest.mark.parametrize("schema", [1, 2, 3, 4])
@pytest.mark.parametrize("configured_reference", [False, True])
def test_direct_resume_without_source_reference_keeps_provenance_unverified(tmp_path, schema, configured_reference):
    from contextlib import nullcontext
    from gear_sonic.finance.ppo import FinancialPPOTrainer
    from gear_sonic.finance.reference import feature_schema

    source = make_trainer(rollout_steps=4)
    path = tmp_path / "source.pt"
    source.save_checkpoint(path)
    payload = torch.load(path, weights_only=True)
    if schema < 4:
        payload = legacy_checkpoint_payload(payload, schema)
    payload.pop("reference_config", None)
    payload.pop("reference_provenance", None)
    source_path = tmp_path / "without_reference.pt"
    torch.save(payload, source_path)
    restored = make_trainer(rollout_steps=4)
    reference = None
    if configured_reference:
        reference = {
            "schema_version": 1, "dataset_partition": "none", "canonical": "/fixture/monthly.csv",
            "canonical_sha256": "a" * 64, "canonical_size": 1234,
            "symbols": ["AAA", "BBB"], "start": None, "end": None,
            "sequence_length": restored.env.sequence_length, "horizon": 10,
            "feature_schema": feature_schema(), "reward_contract": restored.env.reward_contract,
            "sequence_count": 2, "sequence_index_sha256": "b" * 64,
        }
        restored = FinancialPPOTrainer(restored.actor, restored.critic, restored.env, restored.config,
                                       reference_config=reference)
    with pytest.warns(UserWarning, match=f"schema {schema}") if schema < 4 else nullcontext():
        restored.load_checkpoint(source_path)
    assert restored.reference_provenance == "legacy_reference_unverified"
    assert restored.reference_config == reference
    migrated = tmp_path / "migrated.pt"
    restored.save_checkpoint(migrated)
    saved = torch.load(migrated, weights_only=True)
    assert saved["reference_provenance"] == "legacy_reference_unverified"
    assert saved["reference_config"] == reference
    restored.load_checkpoint(migrated)
    assert restored.reference_provenance == "legacy_reference_unverified"
