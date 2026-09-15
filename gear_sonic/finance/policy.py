"""Financial backbones for the retained Gaussian Actor and normalized Critic."""

from __future__ import annotations

from dataclasses import asdict
from types import SimpleNamespace

from omegaconf import OmegaConf
import torch
from torch import nn

from gear_sonic.finance.losses import financial_auxiliary_losses
from gear_sonic.finance.model import FinancialSonic, FinancialSonicConfig
from gear_sonic.trl.modules.actor_critic_modules import Actor, Critic
from gear_sonic.trl.modules.base_module import BaseModule
from gear_sonic.trl.utils.rl import compute_episode_attnmask


class FinancialActorBackbone(nn.Module):
    """Raw latent conditions dyn; only kin/cycle auxiliaries train the shared encoder."""

    def __init__(self, config, **kwargs):
        super().__init__()
        self.model = FinancialSonic(FinancialSonicConfig(**config))

    def forward(
        self, input_data, *, compute_aux_loss=False, episode_attnmask=None,
        kv_prefix_state=None, kv_prefix_dones=None, loss_start_index=0,
    ):
        if episode_attnmask is None and kv_prefix_dones is not None:
            episode_attnmask = compute_episode_attnmask(kv_prefix_dones)
        current, future = input_data["actor_obs"], input_data["future_reference"]
        mask = input_data.get("future_mask")
        encoder_kwargs = {
            "encoder_noise": input_data.get("encoder_noise"),
            "encoder_mask_type": input_data.get("encoder_mask_type"),
        }
        cache_kwargs = dict(kv_prefix_state=kv_prefix_state, kv_prefix_dones=kv_prefix_dones)
        if not compute_aux_loss:
            latent = self.model.encode(future, mask, **encoder_kwargs)
            return self.model.decode(
                current, latent, episode_attnmask, **cache_kwargs,
            )["normalized_actions"]
        output = self.model(current, future, future_mask=mask,
                            episode_attnmask=episode_attnmask, **encoder_kwargs, **cache_kwargs)
        loss_mask = output["future_mask"].clone()
        if not 0 <= loss_start_index < loss_mask.shape[1]:
            raise ValueError("loss_start_index must leave a trainable observation")
        loss_mask[:, :loss_start_index] = False
        return {
            "action_mean": output["normalized_actions"],
            "aux_losses": financial_auxiliary_losses(output, mask=loss_mask),
            "aux_loss_coef": {"kin": 0.01, "cycle": 1.0},
        }

    @torch.no_grad()
    def rollout_step(self, input_data, *, reset_mask=None):
        latent = self.model.encode(
            input_data["future_reference"], input_data.get("future_mask"),
            encoder_noise=input_data.get("encoder_noise"),
            encoder_mask_type=input_data.get("encoder_mask_type"),
        )
        current = self.model.current_normalizer(input_data["actor_obs"])
        return self.model.dyn.forward_step(torch.cat((current, latent), dim=-1), reset_mask=reset_mask)[:, 0]

    def get_cache_state(self):
        return self.model.dyn.get_cache_state()

    def reset_cache(self, reset_mask=None):
        self.model.reset_cache(reset_mask)


class FinancialCriticBackbone(nn.Module):
    """Reference MLP widths; financial privileged observation dimension only."""

    def __init__(self, input_dim, hidden_dims, **kwargs):
        super().__init__()
        if input_dim < 1 or not hidden_dims or any(width < 1 for width in hidden_dims):
            raise ValueError("Critic dimensions must be positive")
        self.mlp = BaseModule(
            input_dim=input_dim, output_dim=1,
            module_config_dict=SimpleNamespace(layer_config={
                "type": "MLP", "hidden_dims": list(hidden_dims), "activation": "SiLU",
            }),
        )

    def forward(self, input_data, **kwargs):
        return self.mlp(input_data["critic_obs"])


def make_actor_critic(
    model_config: FinancialSonicConfig, critic_obs_dim: int, *,
    critic_hidden_dims=(2048, 2048, 1024, 1024, 512, 512),
) -> tuple[Actor, Critic]:
    if model_config.dropout != 0:
        raise ValueError("PPO cached replay requires dropout=0, as in the reference configuration")
    algo = OmegaConf.create({
        "init_noise_std": 0.05, "use_log_std": False, "use_clampped_std": True,
        "std_clamp_min": 0.001, "std_clamp_max": 0.5,
    })
    actor = Actor(
        env_config=None, algo_config=algo,
        backbone={"_target_": "gear_sonic.finance.policy.FinancialActorBackbone",
                  "config": asdict(model_config)},
        obs_dim_dict={"actor_obs": 16}, input_obs_dict=True, has_aux_loss=True,
        max_rollout_history=model_config.window_size, action_dim=model_config.horizon,
    )
    critic = Critic(
        env_config=None, algo_config=algo,
        backbone={"_target_": "gear_sonic.finance.policy.FinancialCriticBackbone",
                  "input_dim": critic_obs_dim, "hidden_dims": tuple(critic_hidden_dims)},
        obs_dim_dict={"critic_obs": critic_obs_dim}, running_mean_std=True,
    )
    critic.finance_config = {"input_dim": critic_obs_dim, "hidden_dims": tuple(critic_hidden_dims)}
    return actor, critic
