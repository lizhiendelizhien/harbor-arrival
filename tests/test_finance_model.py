import pytest
import torch

from gear_sonic.finance.losses import financial_sonic_loss
from gear_sonic.finance.model import FinancialSonic, FinancialSonicConfig


@pytest.fixture(autouse=True)
def deterministic_cpu():
    torch.manual_seed(12)
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def model_and_data(length=4):
    model = FinancialSonic(FinancialSonicConfig(
        mlp_hidden_dims=(32, 32), d_model=32, num_heads=2,
        num_layers=2, ffn_dim=64,
    ))
    current = torch.randn(2, length, 16)
    future = torch.randn(2, length, 10, 15) * 0.1
    model.fit_normalizers(current, future)
    return model, current, future


def encoder_has_gradient(model):
    return any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.encoder.parameters())


def test_shapes_and_shared_latent_branch_routing():
    model, current, future = model_and_data()
    captured = {}
    dyn_hook = model.dyn.register_forward_pre_hook(lambda module, args: captured.update(dyn=args[0]))
    kin_hook = model.kin.register_forward_pre_hook(lambda module, args: captured.update(kin=args[0]))
    output = model(current, future)
    dyn_hook.remove()
    kin_hook.remove()
    assert output["latent"].shape == (2, 4, 64)
    assert output["quantized_latent"].shape == (2, 4, 2, 32)
    assert output["kin_reconstruction"].shape == (2, 4, 10, 15)
    assert output["log_returns"].shape == (2, 4, 10)
    torch.testing.assert_close(captured["dyn"][..., 16:], output["latent"])
    torch.testing.assert_close(captured["kin"], output["quantized_latent"])
    torch.testing.assert_close(output["cumulative_log_returns"], output["log_returns"].cumsum(-1))
    assert not torch.allclose(output["latent"], output["quantized_latent"].flatten(-2))


@pytest.mark.parametrize("term", ["dyn", "kin", "cycle"])
def test_each_loss_branch_updates_encoder(term):
    model, current, future = model_and_data()
    output = model(current, future)
    losses = financial_sonic_loss(output, future, return_scale=model.future_normalizer.scale[0])
    losses[term].backward()
    assert encoder_has_gradient(model)
    if term == "kin":
        assert all(p.grad is None for p in model.dyn.parameters())
    if term == "cycle":
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.kin.parameters())


def test_streaming_matches_full_sequence_beyond_cache_window_and_resets():
    model, current, future = model_and_data(length=40)
    model.eval()
    with torch.no_grad():
        latent = model.encode(future)
        full = model.decode(current, latent)["log_returns"]
        streamed = torch.stack([
            model.predict_step(current[:, i], latent[:, i])["log_returns"] for i in range(40)
        ], dim=1)
        torch.testing.assert_close(streamed, full, atol=2e-6, rtol=2e-5)
        assert model.dyn._cache_valid.shape == (2, 32)
        fresh = model.decode(current[:, :1], latent[:, :1])["log_returns"][:, 0]
        reset = model.predict_step(current[:, 0], latent[:, 0], reset_mask=torch.ones(2, dtype=torch.bool))
        torch.testing.assert_close(reset["log_returns"], fresh, atol=2e-6, rtol=2e-5)


def test_attention_is_causal_and_episode_mask_blocks_other_symbols():
    model, current, future = model_and_data(length=8)
    model.eval()
    latent = model.encode(future)
    before = model.decode(current, latent)["log_returns"]
    changed = current.clone()
    changed[:, 4:] += 100
    after = model.decode(changed, latent)["log_returns"]
    torch.testing.assert_close(before[:, :4], after[:, :4])
    episode = torch.arange(8) // 4
    mask = (episode[:, None] != episode[None, :]).expand(2, -1, -1)
    together = model.decode(current, latent, episode_attnmask=mask)["log_returns"]
    separate = model.decode(current[:, 4:], latent[:, 4:])["log_returns"]
    torch.testing.assert_close(together[:, 4:], separate, atol=2e-6, rtol=2e-5)


def test_partial_cache_reset_preserves_other_stream():
    model, current, future = model_and_data(length=5)
    model.eval()
    with torch.no_grad():
        latent = model.encode(future)
        full = model.decode(current, latent)["log_returns"]
        for index in range(4):
            model.predict_step(current[:, index], latent[:, index])
        output = model.predict_step(current[:, 4], latent[:, 4], reset_mask=torch.tensor([True, False]))
        fresh = model.decode(current[:1, 4:], latent[:1, 4:])["log_returns"]
        torch.testing.assert_close(output["log_returns"][0], fresh[0, 0], atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(output["log_returns"][1], full[1, 4], atol=2e-6, rtol=2e-5)


def test_masked_nan_values_do_not_poison_losses_or_gradients():
    model, current, future = model_and_data()
    mask = torch.ones_like(future, dtype=torch.bool)
    mask[..., 3, 0] = False
    mask[..., 1, 12] = False
    future[~mask] = float("nan")
    output = model(current, future, future_mask=mask)
    losses = financial_sonic_loss(output, future, future_mask=mask,
                                  return_scale=model.future_normalizer.scale[0])
    assert all(torch.isfinite(value) for value in losses.values())
    losses["total"].backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_normalizers_are_checkpointed_and_require_explicit_training_fit():
    model, current, future = model_and_data()
    restored = FinancialSonic(model.config)
    with pytest.raises(RuntimeError, match="normalizer"):
        restored(current, future)
    restored.load_state_dict(model.state_dict())
    model.eval()
    restored.eval()
    torch.testing.assert_close(model(current, future)["log_returns"], restored(current, future)["log_returns"])


def test_default_architecture_and_empty_masks():
    config = FinancialSonicConfig()
    assert config.mlp_hidden_dims == (2048, 1024, 512, 512)
    assert (config.d_model, config.num_heads, config.num_layers,
            config.ffn_dim, config.window_size) == (256, 4, 6, 1024, 32)
    model, current, future = model_and_data()
    with pytest.raises(ValueError, match="valid"):
        model(current, future, future_mask=torch.zeros_like(future, dtype=torch.bool))


def test_burn_in_excludes_loss_but_keeps_full_sequence_context():
    model, current, future = model_and_data(length=5)
    output = model(current, future)
    burned = financial_sonic_loss(output, future, burn_in=2)
    cropped = {key: value[:, 2:] for key, value in output.items()}
    expected = financial_sonic_loss(cropped, future[:, 2:])
    for key in burned:
        torch.testing.assert_close(burned[key], expected[key])
