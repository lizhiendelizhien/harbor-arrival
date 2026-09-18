"""Sonic-style PPO with optional explicit gradient synchronization."""

from dataclasses import asdict, dataclass
from copy import deepcopy
from pathlib import Path
import warnings

import torch

from gear_sonic.finance.denoising import encoder_denoising_contract
from gear_sonic.finance.distributed import DistributedContext
from gear_sonic.finance.rewards import summarize_tracking_metrics, validate_reward_contract
from gear_sonic.finance.resume import prepare_resume_checkpoint
from gear_sonic.trl.utils.rl import compute_episode_attnmask


@dataclass(frozen=True)
class PPOConfig:
    rollout_steps: int = 24
    epochs: int = 5
    num_minibatches: int = 4
    gamma: float = 0.99
    lam: float = 0.95
    clip_param: float = 0.2
    value_clip_param: float = 0.2
    value_loss_coef: float = 1.0
    entropy_coef: float = 0.01
    learning_rate: float = 2e-5
    desired_kl: float | None = 0.01
    adaptive_lr_min: float = 1e-5
    adaptive_lr_max: float = 2e-4
    max_grad_norm: float = 0.1

    def __post_init__(self):
        if min(self.rollout_steps, self.epochs, self.num_minibatches) < 1:
            raise ValueError("rollout steps, epochs, and minibatches must be positive")
        if not 0 <= self.gamma <= 1 or not 0 <= self.lam <= 1:
            raise ValueError("gamma and lam must be between zero and one")
        if not 0 < self.adaptive_lr_min <= self.learning_rate <= self.adaptive_lr_max:
            raise ValueError("learning rate must lie within positive adaptive bounds")
        if min(self.clip_param, self.value_clip_param, self.max_grad_norm) <= 0:
            raise ValueError("clipping limits must be positive")
        if self.desired_kl is not None and self.desired_kl <= 0:
            raise ValueError("desired KL must be positive or None")


def _normalize_advantages(advantages):
    scale = advantages.std() if advantages.numel() > 1 else advantages.new_zeros(())
    return (advantages - advantages.mean()) / (scale + 1e-8)


def compute_gae(rewards, values, dones, bootstrap, gamma, lam, *, normalize=True,
                normalizer=None):
    """Return GAE targets and optionally normalized advantages, all [env, step]."""
    returns = torch.zeros_like(values)
    advantage = torch.zeros_like(bootstrap)
    for step in reversed(range(rewards.shape[1])):
        next_value = bootstrap if step == rewards.shape[1] - 1 else values[:, step + 1]
        alive = (~dones[:, step].bool()).to(values.dtype)
        delta = rewards[:, step] + gamma * alive * next_value - values[:, step]
        advantage = delta + gamma * lam * alive * advantage
        returns[:, step] = advantage + values[:, step]
    advantages = returns - values
    if normalize:
        advantages = normalizer(advantages) if normalizer is not None else _normalize_advantages(advantages)
    return returns, advantages


def clipped_ppo_losses(*, new_logprobs, old_logprobs, advantages, new_values,
                       old_values, returns, entropy, means, old_means, sigmas,
                       old_sigmas, config):
    """Reference clipped policy/value losses, Gaussian KL, and entropy bonus."""
    ratio = (new_logprobs - old_logprobs).exp()
    policy_loss = torch.maximum(
        -advantages * ratio,
        -advantages * ratio.clamp(1 - config.clip_param, 1 + config.clip_param),
    ).mean()
    clipped_values = old_values + (new_values - old_values).clamp(
        -config.value_clip_param, config.value_clip_param
    )
    value_loss = torch.maximum((new_values - returns).square(),
                               (clipped_values - returns).square()).mean()
    with torch.no_grad():
        kl = (torch.log(sigmas / old_sigmas + 1e-5)
              + (old_sigmas.square() + (old_means - means).square())
              / (2 * sigmas.square()) - 0.5).sum(-1).mean()
    entropy_mean = entropy.mean()
    return {
        "ppo_loss": policy_loss + config.value_loss_coef * value_loss
                    - config.entropy_coef * entropy_mean,
        "policy_loss": policy_loss, "value_loss": value_loss,
        "entropy": entropy_mean, "kl": kl,
    }


class FinancialPPOTrainer:
    """Bounded on-policy updates; resume restores weights but starts fresh clips.

    With a distributed context each rank owns independent environment streams;
    gradients and stateful training statistics are synchronized explicitly so
    the custom Actor/KV-cache API remains unchanged.
    """

    def __init__(self, actor, critic, env, config: PPOConfig | None = None, *,
                 reference_config=None, reference_provenance="training_reference_pool",
                 distributed: DistributedContext | None = None,
                 local_reference_count: int | None = None):
        self.config = config or PPOConfig()
        self.distributed = distributed or DistributedContext()
        if (torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1
                and not self.distributed.enabled):
            raise ValueError("A distributed process group requires a DistributedContext")
        if self.config.num_minibatches > env.num_envs:
            raise ValueError("The minibatch count cannot exceed the number of environment streams")
        self.actor, self.critic, self.env = actor, critic, env
        validate_reward_contract(env.reward_contract)
        if env.encoder_denoising is not True:
            raise ValueError("Financial PPO training requires encoder denoising")
        self._validate_normalization()
        if reference_config is not None:
            from gear_sonic.finance.reference import validate_reference_config
            validate_reference_config(reference_config)
            if reference_config["reward_contract"] != env.reward_contract:
                raise ValueError("Reference reward contract differs from the training environment")
            expected_count = (len(env._current_bank) if local_reference_count is None
                              else local_reference_count)
            if (reference_config["sequence_length"] != env.sequence_length
                    or reference_config["horizon"] != env.horizon
                    or expected_count != len(env._current_bank)
                    or (local_reference_count is None
                        and reference_config["sequence_count"] != len(env._current_bank))):
                raise ValueError("Reference metadata does not match the training environment")
        if reference_provenance not in ("training_reference_pool", "legacy_reference_unverified"):
            raise ValueError("Unsupported reference provenance")
        self.reference_config = deepcopy(reference_config)
        self.reference_provenance = reference_provenance
        self.local_reference_count = len(env._current_bank)
        self.global_reference_count = (
            reference_config["sequence_count"] if reference_config is not None
            else self.local_reference_count
        )
        self.parameters = [parameter for module in (actor, critic)
                           for parameter in module.parameters() if parameter.requires_grad]
        self.device = next(actor.parameters()).device
        self.learning_rate = self.config.learning_rate
        # The inherited Transformers optimizer does not use critic_learning_rate.
        self.optimizer = torch.optim.AdamW(self.parameters, lr=self.learning_rate,
                                           betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
        self.iteration = 0
        self.training_migrations = []
        self.actor.init_rollout()
        self._obs = env.reset()
        self._last_dones = torch.zeros(env.num_envs, dtype=torch.bool, device=self.device)

    def _validate_normalization(self, actor_state=None):
        """Check effective model statistics before changing rollout or loaded state."""
        model = self.actor.actor_module.model
        for name in ("current_normalizer", "future_normalizer"):
            normalizer = getattr(model, name)
            if actor_state is None:
                state = normalizer.state_dict()
            else:
                prefix = f"actor_module.model.{name}."
                state = {key: actor_state[prefix + key] for key in ("center", "scale", "fitted")}
            center, scale = state["center"].float(), state["scale"].float()
            if (not bool(state["fitted"]) or not torch.isfinite(center).all()
                    or not torch.isfinite(scale).all() or not (scale > 0).all()):
                raise ValueError("Actor normalization must be fitted with finite centers and positive scales")
            if name == "future_normalizer":
                for field, value in (("return_center", center[0]), ("return_scale", scale[0])):
                    actual = torch.as_tensor(getattr(self.env, field), dtype=torch.float32)
                    if actual.ndim != 0 or not torch.isfinite(actual) or float(actual) != float(value):
                        raise ValueError(f"Environment config normalization {field} differs from actor statistics")

    def _sync_critic_statistics(self):
        self.distributed.sync_running_mean_std(self.critic.running_mean_std)

    def _reduce_tracking_metrics(self, metrics):
        """Gather equal local rollout tensors before producing global summaries."""
        if not self.distributed.enabled:
            return summarize_tracking_metrics(metrics)
        gathered = {
            name: self.distributed.gather(value)
            for name, value in metrics.items()
        }
        return summarize_tracking_metrics(gathered)

    def _reduce_update_metrics(self, metrics):
        if not self.distributed.enabled:
            return metrics
        reduced = {}
        for name, value in metrics.items():
            if name == "updates":
                # Every rank executes the same number of local minibatches.
                reduced[name] = value
            else:
                reduced[name] = float(self.distributed.mean(value, device=self.device).item())
        return reduced

    @torch.no_grad()
    def collect_rollout(self):
        """Snapshot [env, step, ...] transitions and pre-rollout detached KV state."""
        self.actor.eval()
        self.critic.eval()
        self.actor.init_rollout(preserve_history=True)
        self.actor.reset(self._last_dones)
        cache = self.actor.get_rollout_cache_state()
        prefix = None if cache is None else {key: value.detach().clone() for key, value in cache.items()}
        observations = {key: [] for key in self._obs}
        stored = {key: [] for key in ("actions", "logprobs", "means", "sigmas", "rewards", "dones")}
        diagnostics = {}
        for _ in range(self.config.rollout_steps):
            for key, value in self._obs.items():
                observations[key].append(value.detach().clone())
            policy = self.actor.rollout(self._obs, cur_dones=self._last_dones)
            actions = policy["actions"]
            stored["actions"].append(actions.detach().clone())
            stored["logprobs"].append(self.actor.get_actions_log_prob(actions).detach().clone())
            stored["means"].append(policy["action_mean"].detach().clone())
            stored["sigmas"].append(policy["action_sigma"].detach().clone())
            self._obs, rewards, dones, info = self.env.step(actions)
            for key, value in {"reward": rewards, **info}.items():
                if isinstance(value, torch.Tensor):
                    diagnostics.setdefault(key, []).append(value.detach().clone())
            stored["rewards"].append(rewards.detach().clone())
            stored["dones"].append(dones.detach().clone().bool())
            self._last_dones = dones.detach().clone().bool()
            self.actor.reset(self._last_dones)
            self.critic.reset(self._last_dones)
        self.actor.clear_rollout(preserve_history=True)
        obs = {key: torch.stack(value, dim=1) for key, value in observations.items()}
        critic_obs = torch.cat((obs["critic_obs"], self._obs["critic_obs"].unsqueeze(1)), dim=1)
        all_values = self.critic.evaluate({"critic_obs": critic_obs}).squeeze(-1)
        self._sync_critic_statistics()
        batch = {key: torch.stack(value, dim=1) for key, value in stored.items()}
        batch.update(obs=obs, prefix=prefix, values=all_values[:, :-1], bootstrap=all_values[:, -1])
        batch["tracking_metrics"] = {key: torch.stack(values, dim=1) for key, values in diagnostics.items()}
        batch["returns"], raw_advantages = compute_gae(
            batch["rewards"], batch["values"], batch["dones"], batch["bootstrap"],
            self.config.gamma, self.config.lam, normalize=False,
        )
        batch["advantages"] = (
            self.distributed.standardize(raw_advantages)
            if self.distributed.enabled else _normalize_advantages(raw_advantages)
        )
        return batch

    def _adjust_learning_rate(self, kl):
        desired = self.config.desired_kl
        if desired is None:
            return
        if kl > desired * 2:
            self.learning_rate = max(self.config.adaptive_lr_min, self.learning_rate / 1.5)
        elif 0 < kl < desired / 2:
            self.learning_rate = min(self.config.adaptive_lr_max, self.learning_rate * 1.5)
        for group in self.optimizer.param_groups:
            group["lr"] = self.learning_rate

    def update(self, batch):
        """Optimize every rollout step; shuffle complete environment streams only."""
        self.actor.train()
        self.critic.train()
        count, steps = batch["actions"].shape[:2]
        if count < self.config.num_minibatches or steps != self.config.rollout_steps:
            raise ValueError("Rollout dimensions do not match the configured minibatches and steps")
        totals = {}
        updates = 0
        for _ in range(self.config.epochs):
            permutation = torch.randperm(count, device=self.device)
            for indices in torch.tensor_split(permutation, self.config.num_minibatches):
                obs = {key: value[indices] for key, value in batch["obs"].items()}
                dones = batch["dones"][indices]
                prefix = batch["prefix"]
                if prefix is not None:
                    prefix = {
                        "past_key_values": prefix["past_key_values"][:, :, indices],
                        "cache_valid": prefix["cache_valid"][indices],
                        "abs_position": prefix["abs_position"][indices],
                    }
                self.actor.update_distribution(
                    obs, episode_attnmask=compute_episode_attnmask(dones),
                    kv_prefix_state=prefix, kv_prefix_dones=dones if prefix is not None else None,
                    loss_start_index=0, is_training=True,
                )
                rms = self.critic.running_mean_std
                # Retained RMS uses sample variance; one sample cannot update it.
                if rms is not None and indices.numel() * steps == 1:
                    rms.eval()
                try:
                    values = self.critic.evaluate(obs).squeeze(-1)
                finally:
                    if rms is not None:
                        rms.train()
                self._sync_critic_statistics()
                losses = clipped_ppo_losses(
                    new_logprobs=self.actor.get_actions_log_prob(batch["actions"][indices]),
                    old_logprobs=batch["logprobs"][indices], advantages=batch["advantages"][indices],
                    new_values=values, old_values=batch["values"][indices], returns=batch["returns"][indices],
                    entropy=self.actor.entropy, means=self.actor.action_mean, old_means=batch["means"][indices],
                    sigmas=self.actor.action_std, old_sigmas=batch["sigmas"][indices], config=self.config,
                )
                if set(self.actor.aux_losses) != {"kin", "cycle"}:
                    raise ValueError("Financial PPO requires only kin and cycle auxiliary losses")
                total_loss = losses["ppo_loss"]
                for name, value in self.actor.aux_losses.items():
                    losses[name] = value
                    total_loss = total_loss + self.actor.aux_loss_coef[name] * value
                losses["loss"] = total_loss
                if not all(bool(torch.isfinite(value).all()) for value in losses.values()):
                    raise FloatingPointError("Non-finite financial PPO loss")
                global_kl = (self.distributed.mean(losses["kl"], device=self.device)
                             if self.distributed.enabled else losses["kl"])
                self._adjust_learning_rate(float(global_kl))
                self.optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                self.distributed.sync_gradients(self.parameters)
                losses["grad_norm"] = torch.nn.utils.clip_grad_norm_(
                    self.parameters, self.config.max_grad_norm, error_if_nonfinite=True,
                )
                self.optimizer.step()
                weight = indices.numel() / (count * self.config.epochs)
                for name, value in losses.items():
                    totals[name] = totals.get(name, 0.0) + float(value.detach()) * weight
                updates += 1
        totals.update(updates=updates, learning_rate=self.learning_rate)
        return self._reduce_update_metrics(totals)

    def train_iteration(self):
        batch = self.collect_rollout()
        metrics = self.update(batch)
        metrics.update(self._reduce_tracking_metrics(batch["tracking_metrics"]))
        self.iteration += 1
        terminal_count = int(batch["dones"].sum())
        if self.distributed.enabled:
            terminal_count = int(self.distributed.sum(terminal_count, device=self.device).item())
        metrics.update(iteration=self.iteration, terminal_count=terminal_count)
        return metrics

    def _environment_config(self):
        return {
            "sequence_length": self.env.sequence_length,
            "history_length": self.env.history_length, "horizon": self.env.horizon,
            "return_center": float(self.env.return_center), "return_scale": float(self.env.return_scale),
            "encoder_denoising": self.env.encoder_denoising,
        }

    def save_checkpoint(self, path):
        """Write a new checkpoint exclusively; never overwrite an existing path."""
        if self.distributed.enabled and not self.distributed.is_main_process:
            raise RuntimeError("Only rank 0 may publish a finance PPO checkpoint")
        validate_reward_contract(self.env.reward_contract)
        if self.env.encoder_denoising is not True:
            raise ValueError("Financial PPO checkpoints require encoder denoising")
        self._validate_normalization()
        checkpoint = {
            "schema_version": 4, "config": asdict(self.config),
            "encoder_denoising_contract": encoder_denoising_contract(),
            "training_migrations": deepcopy(self.training_migrations),
            "reward_contract": deepcopy(self.env.reward_contract),
            "reference_config": deepcopy(self.reference_config),
            "reference_provenance": self.reference_provenance,
            "model_config": asdict(self.actor.actor_module.model.config),
            "critic_config": self.critic.finance_config,
            "env_config": self._environment_config(),
            "actor": self.actor.state_dict(), "critic": self.critic.state_dict(),
            "optimizer": self.optimizer.state_dict(), "iteration": self.iteration,
            "learning_rate": self.learning_rate,
            "distributed": self.distributed.metadata(
                reference_count_global=self.global_reference_count,
                reference_count_local=self.local_reference_count,
                env_count_local=self.env.num_envs,
            ),
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as handle:
            torch.save(checkpoint, handle)

    def load_checkpoint(self, path, *, reference_config=None):
        """Restore models/optimizer, then reset environment and actor history.

        This is not an exact mid-episode resume: rollout caches and sampled clip
        positions are intentionally discarded, while normalization buffers persist.
        Known legacy protocols migrate to current rewards and denoising while
        preserving model and optimizer state; their changes remain in metadata.
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        source_schema = checkpoint.get("schema_version")
        checkpoint = prepare_resume_checkpoint(checkpoint, reference_config=reference_config)
        if self.env.encoder_denoising is not True:
            raise ValueError("Checkpoint encoder denoising contract is incompatible with this trainer")
        validate_reward_contract(checkpoint.get("reward_contract"))
        if checkpoint["reward_contract"] != self.env.reward_contract:
            raise ValueError("Checkpoint reward contract differs from this training environment")
        if (checkpoint["config"] != asdict(self.config)
                or checkpoint["model_config"] != asdict(self.actor.actor_module.model.config)
                or checkpoint["critic_config"] != self.critic.finance_config):
            raise ValueError("Checkpoint config is incompatible with this trainer")
        if checkpoint["env_config"] != self._environment_config():
            raise ValueError("Checkpoint environment config is incompatible with this trainer")
        if checkpoint.get("reference_config") is not None:
            from gear_sonic.finance.reference import reference_identity, validate_reference_config
            validate_reference_config(checkpoint["reference_config"])
            if (self.reference_config is None or reference_identity(self.reference_config)
                    != reference_identity(checkpoint["reference_config"])):
                raise ValueError("Checkpoint reference identity differs from this trainer")
            if checkpoint.get("reference_provenance", "training_reference_pool") != self.reference_provenance:
                raise ValueError("Checkpoint reference provenance differs from this trainer")
        for name, module in (("actor", self.actor), ("critic", self.critic)):
            current, saved = module.state_dict(), checkpoint[name]
            if current.keys() != saved.keys() or any(current[key].shape != saved[key].shape for key in current):
                raise ValueError(f"Checkpoint {name} architecture config is incompatible")
        self._validate_normalization(checkpoint["actor"])
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.learning_rate = checkpoint["learning_rate"]
        self.iteration = checkpoint["iteration"]
        self.training_migrations = deepcopy(checkpoint["training_migrations"])
        if checkpoint.get("reference_config") is None:
            self.reference_provenance = "legacy_reference_unverified"
        self.actor.init_rollout()
        self._obs = self.env.reset()
        self._last_dones.zero_()
        if source_schema < 4:
            warnings.warn(
                f"Resuming checkpoint schema {source_schema} at iteration {self.iteration}: "
                "Actor, Critic, optimizer and normalization states are preserved; "
                "training now uses Reward V1 and encoder denoising. "
                "The Critic may need to adapt to changed reward targets.",
                UserWarning, stacklevel=2,
            )
