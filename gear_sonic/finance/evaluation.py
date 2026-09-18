"""Deterministic privileged playback of complete checkpoint reference clips."""

from __future__ import annotations

from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import asdict
import math
from pathlib import Path
import sys

import torch

from gear_sonic.finance.denoising import encoder_denoising_contract
from gear_sonic.finance.environment import _month_number
from gear_sonic.finance.model import FinancialSonicConfig
from gear_sonic.finance.policy import make_actor_critic
from gear_sonic.finance.reference import (
    read_reference_config, resolve_reference_pool, validate_reference_config,
)
from gear_sonic.finance.rewards import (
    LEGACY_TRACKING_REWARD_CONTRACT, TRACKING_REWARD_CONTRACT,
    summarize_tracking_metrics, tracking_metrics, validate_reward_contract,
)


_ROLLING_METRICS = ("rolling_mse", "rolling_penalty", "previous_overlap_mse")

# Keep this order stable: the command-line evaluator uses it as the CSV schema.
# Every row describes one predicted month in one ten-month action vector.
TRAJECTORY_FIELDNAMES = (
    "sequence_id", "symbol", "anchor_index", "anchor_period", "target_period", "horizon",
    "predicted_normalized_return", "target_normalized_return",
    "predicted_log_return", "target_log_return",
    "predicted_cumulative_log_return", "target_cumulative_log_return",
    "reference_cumulative_log_return",
    "latent_source", "encoder_input_mode", "action_mode", "cache_reset",
    "cache_reset_reason", "rollout_mode",
)

_ROLLOUT_MODE = "privileged_train_playback"
_LATENT_SOURCE = "oracle_future_encoder"
_ENCODER_INPUT_MODE = "clean"
_ACTION_MODE = "deterministic_mean"


def _positive_integer(name, value):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _period_after(period: str, offset: int) -> str:
    """Return the calendar month ``offset`` months after ``period``."""
    if type(offset) is not int:
        raise ValueError("Month offset must be an integer")
    number = _month_number(period) + offset
    year, month_zero = divmod(number, 12)
    return f"{year:04d}-{month_zero + 1:02d}"


def _trajectory_rows(
    source, sequence_id: int, actions: torch.Tensor, future: torch.Tensor, normalizer,
):
    """Build JSON/CSV-safe per-month rows from one deterministic playback batch item."""
    if actions.ndim != 2 or actions.shape[-1] != 10:
        raise ValueError("Playback actions must have shape [S,10]")
    if future.ndim != 3 or future.shape[:2] != actions.shape or future.shape[-1] != 15:
        raise ValueError("Playback future descriptors must have shape [S,10,15]")
    # Match tracking_metrics' widened arithmetic so finite extreme inputs stay
    # representable in the JSON/CSV ledger instead of overflowing float32.
    actions = actions.detach().double().cpu()
    center = normalizer.center[0].detach().double().cpu()
    scale = normalizer.scale[0].detach().double().cpu()
    predicted_log = actions * scale + center
    future = future.detach().double().cpu()
    target_log = future[..., 0]
    target_normalized = (target_log - center) / scale
    predicted_cumulative = predicted_log.cumsum(dim=-1)
    target_cumulative = target_log.cumsum(dim=-1)
    reference_cumulative = future[..., 1]
    if any(not torch.isfinite(value).all() for value in (
        predicted_log, target_log, target_normalized,
        predicted_cumulative, target_cumulative, reference_cumulative,
    )):
        raise ValueError("Playback trajectory values must remain finite")

    periods = source["periods"]
    rows = []
    for anchor_index, anchor_period in enumerate(periods):
        reset = anchor_index == 0
        reset_reason = "sequence_start" if reset else ""
        for horizon_index in range(10):
            row = {
                "sequence_id": int(sequence_id),
                "symbol": source["symbol"],
                "anchor_index": int(anchor_index),
                "anchor_period": anchor_period,
                "target_period": _period_after(anchor_period, horizon_index + 1),
                "horizon": int(horizon_index + 1),
                "predicted_normalized_return": float(actions[anchor_index, horizon_index].item()),
                "target_normalized_return": float(target_normalized[anchor_index, horizon_index].item()),
                "predicted_log_return": float(predicted_log[anchor_index, horizon_index].item()),
                "target_log_return": float(target_log[anchor_index, horizon_index].item()),
                "predicted_cumulative_log_return": float(predicted_cumulative[anchor_index, horizon_index].item()),
                "target_cumulative_log_return": float(target_cumulative[anchor_index, horizon_index].item()),
                "reference_cumulative_log_return": float(reference_cumulative[anchor_index, horizon_index].item()),
                "latent_source": _LATENT_SOURCE,
                "encoder_input_mode": _ENCODER_INPUT_MODE,
                "action_mode": _ACTION_MODE,
                "cache_reset": bool(reset),
                "cache_reset_reason": reset_reason,
                "rollout_mode": _ROLLOUT_MODE,
            }
            # Guard against accidental schema drift before a CSV writer sees it.
            if tuple(row) != TRAJECTORY_FIELDNAMES:
                raise RuntimeError("Trajectory row schema does not match TRAJECTORY_FIELDNAMES")
            rows.append(row)
    return rows


def _validate_actor(actor):
    model = actor.actor_module.model
    if model.config.horizon != 10 or actor.num_actions != 10:
        raise ValueError("Evaluation requires a ten-month model horizon")
    for normalizer in (model.current_normalizer, model.future_normalizer):
        if (not bool(normalizer.fitted) or not torch.isfinite(normalizer.center).all()
                or not torch.isfinite(normalizer.scale).all() or not (normalizer.scale > 0).all()):
            raise ValueError("Checkpoint normalization must contain fitted finite centers and positive scales")
    if any(not torch.isfinite(value).all() for value in actor.state_dict().values()):
        raise ValueError("Checkpoint actor weights and statistics must be finite")
    return model


def _sequence_tensors(source, sequence_length, device):
    if not isinstance(source["symbol"], str) or not source["symbol"].strip():
        raise ValueError("Each sequence must describe one nonempty symbol")
    months = [_month_number(period) for period in source["periods"]]
    if (len(months) != sequence_length or not months
            or any(right - left != 1 for left, right in zip(months, months[1:]))):
        raise ValueError("All sequences must have the same positive length and contiguous months")
    if _month_number(source["target_end_period"]) != months[-1] + 10:
        raise ValueError("Sequence target end must include the complete ten-month horizon")
    current = torch.as_tensor(source["current_state"], dtype=torch.float32, device=device).detach()
    future = torch.as_tensor(source["future_reference"], dtype=torch.float32, device=device).detach()
    mask = torch.as_tensor(source["future_mask"], device=device)
    if current.shape != (sequence_length, 16) or future.shape != (sequence_length, 10, 15):
        raise ValueError("Sequence states and future descriptors must have shapes [S,16] and [S,10,15]")
    if mask.shape != future.shape or mask.dtype != torch.bool or not mask.all():
        raise ValueError("Evaluation requires complete boolean future masks with shape [S,10,15]")
    if not torch.isfinite(current).all() or not torch.isfinite(future).all():
        raise ValueError("Sequence states and future descriptors must be finite")
    return current, future, mask


def _aggregate(rows, metric_names):
    anchor_count = sum(row["anchor_count"] for row in rows)
    result = {"sequence_count": len(rows), "anchor_count": anchor_count}
    for name in metric_names:
        if name.startswith("cumulative_rmse_"):
            continue
        if name in ("sample_count", "rolling_valid_count") or name.startswith("direction_") and "_count_" in name:
            result[name] = sum(row[name] for row in rows)
        elif name in _ROLLING_METRICS:
            valid_count = sum(row["rolling_valid_count"] for row in rows)
            result[name] = (math.fsum(row[name] * row["rolling_valid_count"] for row in rows
                                     if row["rolling_valid_count"]) / valid_count if valid_count else None)
        else:
            result[name] = math.fsum(row[name] * row["anchor_count"] for row in rows) / anchor_count
    for horizon in (1, 3, 6, 10):
        squared_error = f"cumulative_squared_error_{horizon}m"
        if squared_error in result:
            result[f"cumulative_rmse_{horizon}m"] = math.sqrt(result[squared_error])
    return result


@torch.no_grad()
def evaluate_sequences(
    actor, sequences, *, num_envs=16, reward_contract=None, include_trajectories=False,
    max_sequences=None, trajectory_callback=None,
):
    """Visit every supplied clip once, preserving weights and normalization buffers.

    Consecutive clips share a batch only when they have the same full length.
    Actor history is cleared between batches and when playback finishes. Actual
    future descriptors remain privileged inputs; metrics are reference tracking.

    When ``include_trajectories`` is true, a deterministic per-anchor/per-horizon
    ledger is returned under ``trajectories``.  A ``trajectory_callback`` receives
    each row as soon as it is produced, which permits streaming large reference
    pools without retaining the ledger in memory.  Supplying a callback alone
    enables ledger generation but does not allocate a trajectory list.
    """
    _positive_integer("num_envs", num_envs)
    if type(include_trajectories) is not bool:
        raise ValueError("include_trajectories must be boolean")
    if max_sequences is not None:
        _positive_integer("max_sequences", max_sequences)
    if trajectory_callback is not None and not callable(trajectory_callback):
        raise TypeError("trajectory_callback must be callable")
    reward_contract = TRACKING_REWARD_CONTRACT if reward_contract is None else reward_contract
    validate_reward_contract(reward_contract, allow_legacy=True)
    reference_sequence_count = len(sequences) if hasattr(sequences, "__len__") else None
    sequences = list(sequences)
    if not sequences:
        raise ValueError("At least one complete reference sequence is required")
    if reference_sequence_count is None:
        reference_sequence_count = len(sequences)
    if max_sequences is not None:
        sequences = sequences[:max_sequences]
        if not sequences:
            raise ValueError("max_sequences leaves no reference sequences to evaluate")
    sequence_length = len(sequences[0]["periods"])
    _positive_integer("sequence_length", sequence_length)
    model = _validate_actor(actor)
    device = next(actor.parameters()).device
    normalizer = model.future_normalizer
    actor.eval()
    rows = []
    trajectories = [] if include_trajectories else None
    trajectory_count = 0

    def emit_trajectory(row):
        nonlocal trajectory_count
        trajectory_count += 1
        if trajectory_callback is not None:
            trajectory_callback(row)
        if trajectories is not None:
            trajectories.append(row)

    try:
        for offset in range(0, len(sequences), num_envs):
            batch = sequences[offset:offset + num_envs]
            tensors = [_sequence_tensors(source, sequence_length, device) for source in batch]
            current, future, mask = [torch.stack(values) for values in zip(*tensors)]
            actor.init_rollout()
            dones = torch.zeros(len(batch), dtype=torch.bool, device=device)
            step_metrics = []
            step_actions = []
            previous_actions = None
            for step in range(sequence_length):
                obs = {"actor_obs": current[:, step], "future_reference": future[:, step],
                       "future_mask": mask[:, step]}
                actions = actor.act_inference(obs, cur_dones=dones)
                if actions.shape != (len(batch), 10) or not torch.isfinite(actions).all():
                    raise ValueError("Evaluation produced nonfinite actions with shape [batch,10]")
                metrics = tracking_metrics(
                    actions, future[:, step, :, 0], normalizer.center[0], normalizer.scale[0],
                    previous_actions=previous_actions, reward_contract=reward_contract,
                )
                for name, values in metrics.items():
                    if not torch.isfinite(values).all():
                        raise ValueError(f"Evaluation produced nonfinite {name}")
                step_metrics.append(metrics)
                step_actions.append(actions.detach().clone())
                previous_actions = actions.detach().clone()
            stacked = {name: torch.stack([metrics[name] for metrics in step_metrics], dim=1)
                       for name in step_metrics[0]}
            actions_stacked = torch.stack(step_actions, dim=1)
            for index, source in enumerate(batch):
                summary = summarize_tracking_metrics(
                    {name: values[index] for name, values in stacked.items()}, include_quantiles=False,
                )
                if trajectories is not None or trajectory_callback is not None:
                    for trajectory_row in _trajectory_rows(
                        source, offset + index, actions_stacked[index], future[index], normalizer,
                    ):
                        emit_trajectory(trajectory_row)
                rows.append({
                    "sequence_id": offset + index, "symbol": source["symbol"],
                    "anchor_start": source["periods"][0], "anchor_end": source["periods"][-1],
                    "target_end": source["target_end_period"], "anchor_count": sequence_length,
                    **summary,
                })
    finally:
        actor.init_rollout()
    symbol_rows = {}
    for row in rows:
        symbol_rows.setdefault(row["symbol"], []).append(row)
    overall = _aggregate(rows, summary)
    result = {
        "reward_contract": deepcopy(reward_contract), "action_mode": _ACTION_MODE,
        "encoder_input_mode": _ENCODER_INPUT_MODE,
        "sequence_count": len(rows), "sequence_length": sequence_length,
        "anchor_count": overall["anchor_count"], "overall": overall, "sequences": rows,
        "symbols": [{"symbol": symbol, **_aggregate(symbol_rows[symbol], summary)}
                    for symbol in sorted(symbol_rows)],
        "sequence_limit": max_sequences, "reference_sequence_count": reference_sequence_count,
    }
    if trajectories is not None or trajectory_callback is not None:
        result.update({
            "rollout_mode": _ROLLOUT_MODE, "latent_source": _LATENT_SOURCE,
            "trajectory_count": trajectory_count,
        })
    if trajectories is not None:
        result["trajectories"] = trajectories
    return result


@torch.no_grad()
def rollout_sequences(
    actor, sequences, *, num_envs=16, reward_contract=None, max_sequences=None,
    trajectory_callback=None, include_trajectories=None,
):
    """Run the deterministic Sonic-style privileged train-set playback loop.

    This is a convenience API for callers that want the detailed ledger.  The
    Encoder receives each anchor's clean ten-month reference window (the oracle
    latent source), while the Decoder receives the actual current state.  Every
    supplied clip is an independent stream and starts with an empty KV cache.
    The next current state is always taken from the reference sequence; model
    actions are never fed back into market state.

    By default rows are returned in memory. Supplying a callback streams rows
    without retaining them unless ``include_trajectories=True`` is explicit.
    """
    if include_trajectories is None:
        include_trajectories = trajectory_callback is None
    if type(include_trajectories) is not bool:
        raise ValueError("include_trajectories must be boolean")
    result = evaluate_sequences(
        actor, sequences, num_envs=num_envs, reward_contract=reward_contract,
        include_trajectories=include_trajectories, max_sequences=max_sequences,
        trajectory_callback=trajectory_callback,
    )
    if include_trajectories:
        trajectory_count = len(result["trajectories"])
    else:
        trajectory_count = result.get("trajectory_count", 0)
    result.update({
        "mode": _ROLLOUT_MODE, "rollout_mode": _ROLLOUT_MODE,
        "latent_source": _LATENT_SOURCE, "trajectory_count": trajectory_count,
    })
    return result


def evaluate_checkpoint(checkpoint_path, *, device="cpu", num_envs=16, threads=2,
                        canonical=None, symbols=None, start=None, end=None,
                        custom_reference=False, reference_config=None,
                        include_trajectories=False, max_sequences=None,
                        trajectory_callback=None):
    """Restore a frozen actor and evaluate its saved reference pool by default.

    A legacy checkpoint needs an explicit reference JSON recipe, and cannot
    establish historical training provenance. Custom reference selection is
    explicit and still uses checkpoint architecture, statistics and contracts.
    """
    _positive_integer("num_envs", num_envs)
    _positive_integer("threads", threads)
    if type(include_trajectories) is not bool:
        raise ValueError("include_trajectories must be boolean")
    if max_sequences is not None:
        _positive_integer("max_sequences", max_sequences)
    if trajectory_callback is not None and not callable(trajectory_callback):
        raise TypeError("trajectory_callback must be callable")
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(threads)
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        schema = checkpoint.get("schema_version")
        if type(schema) is not int or schema not in (1, 2, 3, 4):
            raise ValueError("Unsupported checkpoint schema")
        if schema == 4 and (
            checkpoint.get("encoder_denoising_contract") != encoder_denoising_contract()
            or checkpoint.get("env_config", {}).get("encoder_denoising") is not True
        ):
            raise ValueError("Checkpoint encoder denoising contract is incompatible")
        if schema in (3, 4):
            reward_contract = checkpoint.get("reward_contract")
            validate_reward_contract(reward_contract)
        else:
            reward_contract = checkpoint.get("reward_contract", LEGACY_TRACKING_REWARD_CONTRACT)
            validate_reward_contract(reward_contract, allow_legacy=True)
            if reward_contract != LEGACY_TRACKING_REWARD_CONTRACT:
                raise ValueError("Legacy checkpoint schema requires the legacy reward contract")
        saved_reference = checkpoint.get("reference_config")
        legacy = (saved_reference is None
                  or checkpoint.get("reference_provenance") == "legacy_reference_unverified")
        if saved_reference is None:
            if reference_config is None:
                raise ValueError("Legacy checkpoint requires explicit --reference-config JSON")
            saved_reference = read_reference_config(reference_config)
        elif reference_config is not None:
            raise ValueError("Checkpoint reference metadata is authoritative; --reference-config cannot override it")
        validate_reference_config(saved_reference)
        if saved_reference["reward_contract"] != reward_contract:
            raise ValueError("Checkpoint reward contract conflicts with the reference reward contract")
        model_values = dict(checkpoint["model_config"])
        model_values["mlp_hidden_dims"] = tuple(model_values["mlp_hidden_dims"])
        model_config = FinancialSonicConfig(**model_values)
        critic_config, env_config = checkpoint["critic_config"], checkpoint["env_config"]
        _positive_integer("history_length", env_config["history_length"])
        _positive_integer("sequence_length", env_config["sequence_length"])
        if (env_config["horizon"] != 10 or model_config.horizon != 10
                or critic_config["input_dim"] != env_config["history_length"] * 26 + 150):
            raise ValueError("Incompatible checkpoint environment, model horizon or critic dimensions")
        if (env_config["sequence_length"] != saved_reference["sequence_length"]
                or env_config["horizon"] != saved_reference["horizon"]):
            raise ValueError("Checkpoint environment config conflicts with the reference sequence configuration")
        with redirect_stdout(sys.stderr):
            actor, critic = make_actor_critic(
                model_config, critic_config["input_dim"], critic_hidden_dims=critic_config["hidden_dims"],
            )
        del critic
        actor.load_state_dict(checkpoint["actor"])
        model = _validate_actor(actor)
        for name, value in (("return_center", model.future_normalizer.center[0]),
                            ("return_scale", model.future_normalizer.scale[0])):
            if env_config[name] != float(value):
                raise ValueError(f"Checkpoint environment config {name} conflicts with actor normalization")
        sequences, resolved_reference = resolve_reference_pool(
            saved_reference, canonical=canonical, symbols=symbols, start=start, end=end,
            custom_reference=custom_reference,
        )
        reference_sequence_count = len(sequences)
        actor.to(device)
        result = evaluate_sequences(
            actor, sequences, num_envs=num_envs, reward_contract=reward_contract,
            include_trajectories=include_trajectories, max_sequences=max_sequences,
            trajectory_callback=trajectory_callback,
        )
        scope = ("custom_reference_pool" if custom_reference else
                 "legacy_reference_unverified" if legacy else "training_reference_pool")
        mode = (
            _ROLLOUT_MODE if (include_trajectories or trajectory_callback is not None)
            else "privileged_reference_tracking"
        )
        return {
            "schema_version": 1, "mode": mode, "dataset_partition": "none",
            "reference_scope": scope, "legacy_reference_unverified": legacy,
            "checkpoint": str(Path(checkpoint_path).resolve()),
            "checkpoint_iteration": checkpoint["iteration"],
            "training_migrations": deepcopy(checkpoint.get("training_migrations", [])),
            "encoder_denoising_contract": deepcopy(checkpoint.get("encoder_denoising_contract")) if schema == 4 else None,
            "reference_config": saved_reference, "resolved_reference_config": resolved_reference,
            "model_config": asdict(model.config), "critic_config": critic_config,
            "env_config": env_config, "device": str(torch.device(device)),
            "num_envs": num_envs, "threads": threads,
            "sequence_limit": max_sequences, "reference_sequence_count": reference_sequence_count,
            **result,
        }
    finally:
        torch.set_num_threads(previous_threads)
