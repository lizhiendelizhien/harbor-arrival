"""Vector playback of fixed monthly reference clips for privileged tracking PPO."""

from __future__ import annotations

from copy import deepcopy
import re

import torch

from .denoising import ENCODER_MASK_DURATION_STEPS, ENCODER_MASK_WEIGHTS, ENCODER_NOISE_BOUND
from .rewards import TRACKING_REWARD_CONTRACT, tracking_metrics


def _month_number(period: str) -> int:
    if not isinstance(period, str) or re.fullmatch(r"[0-9]{4}-(0[1-9]|1[0-2])", period) is None:
        raise ValueError("Periods must be valid YYYY-MM strings")
    return int(period[:4]) * 12 + int(period[5:]) - 1


class MonthlyTrackingEnv:
    """Replay market anchors independently of normalized ten-month return actions.

    Rewards measure reference tracking, not PnL. True future descriptors are
    privileged actor-encoder and critic inputs, not deployable forecasts.
    Fixed-length clips terminate without timeout bootstrapping and automatically
    restart from an independently sampled clip's first anchor.
    """

    horizon = 10

    def __init__(
        self, sequences, num_envs: int, return_center: float, return_scale: float,
        history_length: int = 10, seed: int = 0, device: str | torch.device = "cpu",
        encoder_denoising: bool = True,
    ):
        for name, value in (("num_envs", num_envs), ("history_length", history_length)):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(encoder_denoising, bool):
            raise ValueError("encoder_denoising must be boolean")
        self.encoder_denoising = encoder_denoising
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.history_length = history_length
        self.critic_obs_dim = history_length * (16 + self.horizon) + self.horizon * 15
        self.reward_contract = deepcopy(TRACKING_REWARD_CONTRACT)
        self.return_center = torch.as_tensor(return_center, dtype=torch.float32, device=self.device).detach().clone()
        self.return_scale = torch.as_tensor(return_scale, dtype=torch.float32, device=self.device).detach().clone()
        if (self.return_center.ndim != 0 or self.return_scale.ndim != 0
                or not torch.isfinite(self.return_center) or not torch.isfinite(self.return_scale)
                or self.return_scale <= 0):
            raise ValueError("Return center must be finite and scale must be a finite positive scalar")

        sequences = list(sequences)
        if not sequences:
            raise ValueError("At least one complete sequence is required")
        current_bank, future_bank, mask_bank = [], [], []
        self.sequence_length = None
        for source in sequences:
            symbol = source["symbol"]
            if not isinstance(symbol, str) or not symbol.strip():
                raise ValueError("Each sequence must describe one nonempty symbol")
            periods = source["periods"]
            months = [_month_number(period) for period in periods]
            length = len(months)
            if not length or any(right - left != 1 for left, right in zip(months, months[1:])):
                raise ValueError("Sequence months must be nonempty and contiguous within one symbol")
            if _month_number(source["target_end_period"]) != months[-1] + self.horizon:
                raise ValueError("Target end period must include the complete ten-month horizon")
            if self.sequence_length is None:
                self.sequence_length = length
            elif length != self.sequence_length:
                raise ValueError("All sequences must have the same fixed length")
            current = torch.as_tensor(source["current_state"], dtype=torch.float32, device=self.device).detach()
            future = torch.as_tensor(source["future_reference"], dtype=torch.float32, device=self.device).detach()
            mask = torch.as_tensor(source["future_mask"], device=self.device)
            if current.shape != (length, 16) or future.shape != (length, self.horizon, 15):
                raise ValueError("Sequence states and future descriptors must have shapes [S,16] and [S,10,15]")
            if mask.shape != future.shape or mask.dtype != torch.bool or not mask.all():
                raise ValueError("Environment requires complete boolean future masks with shape [S,10,15]")
            if not torch.isfinite(current).all() or not torch.isfinite(future).all():
                raise ValueError("Sequence states and future descriptors must be finite")
            current_bank.append(current)
            future_bank.append(future)
            mask_bank.append(mask)
        self._current_bank = torch.stack(current_bank)
        self._future_bank = torch.stack(future_bank)
        self._mask_bank = torch.stack(mask_bank)
        self._generator = torch.Generator(device=self.device).manual_seed(seed)
        self._sequence_ids = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self._step_ids = torch.zeros_like(self._sequence_ids)
        self._state_history = torch.zeros(num_envs, history_length, 16, device=self.device)
        self._action_history = torch.zeros(num_envs, history_length, self.horizon, device=self.device)
        self._episode_returns = torch.zeros(num_envs, device=self.device)
        self._episode_lengths = torch.zeros_like(self._sequence_ids)
        if self.encoder_denoising:
            self._encoder_generator = torch.Generator(device=self.device).manual_seed(
                self._generator.initial_seed() ^ 0x5DEECE66D,
            )
            self._encoder_mask_weights = torch.tensor(ENCODER_MASK_WEIGHTS, device=self.device)
            self._encoder_mask_type = torch.zeros(num_envs, 1, dtype=torch.long, device=self.device)
            self._encoder_noise = torch.zeros(num_envs, self.horizon, 15, device=self.device)
            self._encoder_mask_steps_remaining = torch.zeros_like(self._sequence_ids)
        self.reset()

    @torch.no_grad()
    def _refresh_encoder_inputs(self, slot_ids: torch.Tensor, *, reset: bool = False) -> None:
        if not self.encoder_denoising or not slot_ids.numel():
            return
        if reset:
            resample_ids = slot_ids
        else:
            self._encoder_mask_steps_remaining[slot_ids] -= 1
            resample_ids = slot_ids[self._encoder_mask_steps_remaining[slot_ids] == 0]
        if resample_ids.numel():
            self._encoder_mask_type[resample_ids, 0] = torch.multinomial(
                self._encoder_mask_weights, resample_ids.numel(), replacement=True,
                generator=self._encoder_generator,
            )
            minimum, maximum = ENCODER_MASK_DURATION_STEPS
            self._encoder_mask_steps_remaining[resample_ids] = torch.randint(
                minimum, maximum + 1, (resample_ids.numel(),),
                generator=self._encoder_generator, device=self.device,
            )
        self._encoder_noise[slot_ids] = torch.rand(
            slot_ids.numel(), self.horizon, 15, generator=self._encoder_generator, device=self.device,
        ) * (2 * ENCODER_NOISE_BOUND) - ENCODER_NOISE_BOUND

    @torch.no_grad()
    def _reset_slots(self, slot_ids: torch.Tensor) -> None:
        self._sequence_ids[slot_ids] = torch.randint(
            len(self._current_bank), (slot_ids.numel(),), generator=self._generator, device=self.device,
        )
        self._step_ids[slot_ids] = 0
        self._state_history[slot_ids] = 0
        self._state_history[slot_ids, -1] = self._current_bank[self._sequence_ids[slot_ids], 0]
        self._action_history[slot_ids] = 0
        self._episode_returns[slot_ids] = 0
        self._episode_lengths[slot_ids] = 0
        self._refresh_encoder_inputs(slot_ids, reset=True)

    @torch.no_grad()
    def reset(self) -> dict[str, torch.Tensor]:
        """Sample a fresh clip independently for every slot and clear histories."""
        self._reset_slots(torch.arange(self.num_envs, device=self.device))
        return self.observe()

    @torch.no_grad()
    def observe(self) -> dict[str, torch.Tensor]:
        """Return detached snapshots with chronological state, action, future critic order."""
        current = self._current_bank[self._sequence_ids, self._step_ids]
        future = self._future_bank[self._sequence_ids, self._step_ids]
        result = {
            "actor_obs": current,
            "future_reference": future,
            "future_mask": self._mask_bank[self._sequence_ids, self._step_ids],
            "critic_obs": torch.cat((
                self._state_history.flatten(1), self._action_history.flatten(1), future.flatten(1),
            ), dim=-1),
        }
        if self.encoder_denoising:
            result["encoder_mask_type"] = self._encoder_mask_type.clone()
            result["encoder_noise"] = self._encoder_noise.clone()
        return result

    @torch.no_grad()
    def step(self, actions: torch.Tensor) -> tuple[dict, torch.Tensor, torch.Tensor, dict]:
        """Score normalized monthly return paths, advance anchors, and reset terminals."""
        actions = torch.as_tensor(actions, dtype=torch.float32, device=self.device).detach()
        if actions.shape != (self.num_envs, self.horizon) or not torch.isfinite(actions).all():
            raise ValueError("Actions must be finite with shape [num_envs,10]")
        future = self._future_bank[self._sequence_ids, self._step_ids]
        scores = tracking_metrics(
            actions, future[..., 0], self.return_center, self.return_scale,
            previous_actions=self._action_history[:, -1],
            rolling_valid=self._episode_lengths > 0,
            reward_contract=self.reward_contract,
        )
        rewards = scores.pop("reward")
        self._episode_returns += rewards
        self._episode_lengths += 1
        self._step_ids += 1
        dones = self._step_ids == self.sequence_length
        info = scores
        terminal_ids = dones.nonzero(as_tuple=False).flatten()
        if terminal_ids.numel():
            info["episode"] = {
                "env_ids": terminal_ids,
                "return": self._episode_returns[terminal_ids],
                "length": self._episode_lengths[terminal_ids],
                "sequence_ids": self._sequence_ids[terminal_ids],
            }

        self._action_history = self._action_history.roll(-1, dims=1)
        self._action_history[:, -1] = actions
        self._state_history = self._state_history.roll(-1, dims=1)
        continuing_ids = (~dones).nonzero(as_tuple=False).flatten()
        self._state_history[continuing_ids, -1] = self._current_bank[
            self._sequence_ids[continuing_ids], self._step_ids[continuing_ids],
        ]
        self._refresh_encoder_inputs(continuing_ids)
        if terminal_ids.numel():
            self._reset_slots(terminal_ids)
        return self.observe(), rewards, dones, info
