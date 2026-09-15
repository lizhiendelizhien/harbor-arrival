import pytest
import torch

from gear_sonic.finance.model import FinancialSonicConfig


@pytest.fixture(autouse=True)
def deterministic_cpu():
    torch.manual_seed(31)
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def policy_and_observations(length=8):
    from gear_sonic.finance.policy import make_actor_critic

    config = FinancialSonicConfig(
        mlp_hidden_dims=(32, 32), d_model=32, num_heads=2, num_layers=2, ffn_dim=64,
    )
    actor, critic = make_actor_critic(config, 410, critic_hidden_dims=(32, 32))
    obs = {
        "actor_obs": torch.randn(2, length, 16),
        "future_reference": torch.randn(2, length, 10, 15) * 0.1,
        "future_mask": torch.ones(2, length, 10, 15, dtype=torch.bool),
        "critic_obs": torch.randn(2, length, 410),
    }
    actor.actor_module.model.fit_normalizers(obs["actor_obs"], obs["future_reference"])
    return actor, critic, obs


def test_reuses_actor_critic_without_world_prediction_heads():
    actor, critic, obs = policy_and_observations()
    from gear_sonic.trl.modules.actor_critic_modules import Actor, Critic

    assert isinstance(actor, Actor)
    assert isinstance(critic, Critic)
    assert actor.running_mean_std is None
    assert critic.running_mean_std is not None
    output = actor.actor_module(obs, compute_aux_loss=True)
    assert output["action_mean"].shape == (2, 8, 10)
    assert set(output["aux_losses"]) == {"kin", "cycle"}
    assert output["aux_loss_coef"] == {"kin": 0.01, "cycle": 1.0}
    model = actor.actor_module.model
    assert model.dyn.auxiliary_head is None
    assert len(model.dyn.extra_output_heads) == 0
    assert model.dyn.action_head.out_features == 10
    assert not any("s_pred" in name or "z_gmm" in name for name in actor.state_dict())
    assert critic.evaluate(obs).shape == (2, 8, 1)


@pytest.mark.parametrize("term", ["policy", "kin", "cycle"])
def test_live_encoder_receives_policy_and_auxiliary_gradients(term):
    actor, _, obs = policy_and_observations()
    actor.update_distribution(obs, is_training=True)
    if term == "policy":
        action = actor.action_mean.detach() + 0.01
        loss = -actor.get_actions_log_prob(action).mean()
    else:
        loss = actor.aux_losses[term]
    loss.backward()
    model = actor.actor_module.model
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.encoder.parameters())
    if term != "policy":
        assert all(p.grad is None for p in model.dyn.parameters())


def test_normalized_actions_have_explicit_log_return_conversion():
    actor, _, obs = policy_and_observations()
    model = actor.actor_module.model
    output = model(obs["actor_obs"], obs["future_reference"])
    expected = output["normalized_actions"] * model.future_normalizer.scale[0] + model.future_normalizer.center[0]
    torch.testing.assert_close(output["log_returns"], expected)
    torch.testing.assert_close(actor(obs), output["normalized_actions"])


def test_rollout_skips_kin_and_matches_prefix_replay_with_episode_resets():
    actor, _, obs = policy_and_observations(length=48)
    actor.eval()
    model = actor.actor_module.model
    calls = []
    hook = model.kin.register_forward_hook(lambda *args: calls.append(True))
    with torch.no_grad():
        for index in range(36):
            actor.rollout({key: value[:, index] for key, value in obs.items()})
        prefix = actor.get_rollout_cache_state()
        assert prefix["cache_valid"].shape == (2, 31)
        assert not prefix["past_key_values"].requires_grad
        dones = torch.zeros(2, 12, dtype=torch.bool)
        dones[0, 3] = True
        means = []
        for index in range(12):
            output = actor.rollout({key: value[:, index + 36] for key, value in obs.items()})
            means.append(output["action_mean"])
            actor.reset(dones[:, index])
        assert not calls
        replay = actor.actor_module(
            {key: value[:, 36:] for key, value in obs.items()},
            kv_prefix_state=prefix, kv_prefix_dones=dones,
        )
        torch.testing.assert_close(replay, torch.stack(means, dim=1), atol=2e-6, rtol=2e-5)
    hook.remove()


def test_action_noise_clamps_and_critic_eval_keeps_statistics_fixed():
    actor, critic, obs = policy_and_observations()
    torch.testing.assert_close(actor.get_std, torch.full((10,), 0.05))
    with torch.no_grad():
        actor.std[0] = -1.0
        actor.std[1] = 2.0
    assert actor.get_std[0].item() == pytest.approx(0.001)
    assert actor.get_std[1].item() == pytest.approx(0.5)
    critic.eval()
    before = {key: value.clone() for key, value in critic.running_mean_std.state_dict().items()}
    critic.evaluate(obs)
    for key, value in before.items():
        torch.testing.assert_close(critic.running_mean_std.state_dict()[key], value)


def test_policy_rejects_nonzero_dropout_for_rollout_replay():
    from gear_sonic.finance.policy import make_actor_critic

    with pytest.raises(ValueError, match="dropout"):
        make_actor_critic(FinancialSonicConfig(dropout=0.1), 410)
