"""Bounded privileged reference-tracking PPO, without world heads or train/validation splits.

Run with python -m scripts.train_finance_sonic. Real future descriptors are
privileged inputs; these rewards do not measure forecast accuracy or PnL.
Resume restores learning state but restarts reference episodes and model caches.
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys

import torch

from gear_sonic.finance.denoising import encoder_denoising_contract
from gear_sonic.finance.distributed import DistributedContext
from gear_sonic.finance.environment import MonthlyTrackingEnv
from gear_sonic.finance.model import FinancialSonicConfig
from gear_sonic.finance.monitoring import FinanceTensorBoardMonitor, create_tensorboard_monitor
from gear_sonic.finance.policy import make_actor_critic
from gear_sonic.finance.ppo import FinancialPPOTrainer, PPOConfig
from gear_sonic.finance.resume import legacy_run_recipe_matches, prepare_resume_checkpoint
from gear_sonic.finance.reference import (
    load_reference_pool, load_reference_pool_from_trajectories, read_reference_config,
    reference_identity, resolve_reference_pool,
)


def _parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--canonical", type=Path,
        help="Canonical source, or an optional full-market context override for a trajectory archive",
    )
    parser.add_argument("--trajectory-root", type=Path, help="Variable-length monthly trajectory archive")
    parser.add_argument("--trajectory-index", type=Path)
    parser.add_argument("--trajectory-metadata", type=Path)
    parser.add_argument("--trajectory-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--iterations", type=int, required=True, help="Additional PPO iterations to run")
    parser.add_argument("--symbols", nargs="+")
    parser.add_argument("--device", default="cpu")
    for flag, default in (
        ("--num-envs", 16), ("--threads", 2), ("--seed", 7), ("--save-interval", 1),
        ("--sequence-length", None), ("--rollout-steps", None), ("--epochs", None),
        ("--num-minibatches", None),
    ):
        parser.add_argument(flag, type=int, default=default)
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--distributed", action="store_true", help="Use torchrun rank sharding and gradient all-reduce")
    parser.add_argument("--logger", choices=("tensorboard", "none"), default="none")
    parser.add_argument("--start", help="Optional inclusive first anchor month, YYYY-MM")
    parser.add_argument("--end", help="Optional inclusive final label month, YYYY-MM")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--normalization", type=Path)
    source.add_argument("--resume", type=Path)
    parser.add_argument("--reference-config", type=Path, help="Original reference recipe for a checkpoint missing reference metadata")
    return parser.parse_args(argv)


def _runtime_device(requested, distributed_requested):
    device = torch.device(requested)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if device.type != "cuda":
        return device
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA training requested but CUDA is unavailable")
    local_rank = int(os.environ.get("LOCAL_RANK", "0")) if distributed_requested or world_size > 1 else device.index
    local_rank = 0 if local_rank is None else local_rank
    if local_rank >= torch.cuda.device_count():
        raise ValueError(f"LOCAL_RANK={local_rank} exceeds visible CUDA device count")
    torch.cuda.set_device(local_rank)
    return torch.device("cuda", local_rank)


def _parameter_checksum(actor, critic):
    """Cheap deterministic post-update equality signal for rank logs."""
    with torch.no_grad():
        total = sum(
            parameter.detach().double().sum()
            for module in (actor, critic) for parameter in module.parameters()
        )
    return f"{float(total):.17g}"


def _archive_context_for_resume(reference_config, explicit_canonical):
    """Preserve an archive's recorded market-context policy on resume."""
    if explicit_canonical is not None:
        return explicit_canonical
    if reference_config.get("context_source") == "explicit_canonical":
        return Path(reference_config["canonical"])
    return None


def _write_run_recipe(output_dir, recipe, migrations):
    """Keep original recipes immutable when a known training protocol migrates."""
    def matches(existing):
        left = {**existing, "reference_config": reference_identity(existing.get("reference_config", {}))}
        right = {**recipe, "reference_config": reference_identity(recipe["reference_config"])}
        return left == right

    recipe_path = output_dir / "run_config.json"
    if recipe_path.is_symlink():
        raise ValueError("Existing run_config.json conflicts with the resolved training recipe")
    if recipe_path.exists():
        existing = json.loads(recipe_path.read_text(encoding="utf-8"))
        if matches(existing):
            return recipe_path
        if not legacy_run_recipe_matches(existing, recipe, migrations):
            raise ValueError("Existing run_config.json conflicts with the resolved training recipe")
        recipe_path = output_dir / "run_config.schema4.json"
        if recipe_path.is_symlink():
            raise ValueError("Existing run_config.schema4.json conflicts with the resolved training recipe")
        if recipe_path.exists():
            if not matches(json.loads(recipe_path.read_text(encoding="utf-8"))):
                raise ValueError("Existing run_config.schema4.json conflicts with the resolved training recipe")
            return recipe_path
    output_dir.mkdir(parents=True, exist_ok=True)
    with recipe_path.open("x", encoding="utf-8") as handle:
        json.dump(recipe, handle, indent=2, allow_nan=False)
        handle.write("\n")
    return recipe_path


def main(argv=None) -> int:
    """Run one Sonic-style PPO job, optionally as a real torchrun world.

    The actor has a stateful KV cache, so wrapping it in stock DDP would be
    needlessly intrusive.  Each rank therefore owns its own actor/critic and
    environment shard while ``FinancialPPOTrainer`` performs explicit gradient
    all-reduce and rank-zero checkpoint publication.
    """
    args = _parse_args(argv)
    for name in ("iterations", "num_envs", "threads", "save_interval", "sequence_length",
                 "rollout_steps", "epochs", "num_minibatches"):
        value = getattr(args, name)
        if value is not None and value < 1:
            raise ValueError(f"{name} must be positive")

    requested_device = torch.device(args.device)
    world_from_env = int(os.environ.get("WORLD_SIZE", "1"))
    distributed_requested = args.distributed or world_from_env > 1
    runtime_device = _runtime_device(requested_device, distributed_requested)
    distributed = DistributedContext.initialize(
        requested=distributed_requested, device=runtime_device,
        backend=os.environ.get("FINANCE_DDP_BACKEND"),
    )
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(args.threads)
    monitor = FinanceTensorBoardMonitor()
    try:
        if args.trajectory_root is None and any(
            value is not None for value in (
                args.trajectory_index, args.trajectory_metadata, args.trajectory_manifest,
            )
        ):
            raise ValueError("Trajectory companion overrides require --trajectory-root")

        tiny_config = FinancialSonicConfig(
            mlp_hidden_dims=(32, 32), d_model=32, num_heads=2, num_layers=2, ffn_dim=64,
        )
        checkpoint = None
        reference_config = None
        original_reference = None
        reference_provenance = "training_reference_pool"
        if args.resume is not None:
            checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
            if args.reference_config is not None:
                if checkpoint.get("reference_config") is not None:
                    raise ValueError("--reference-config cannot override checkpoint reference metadata")
                original_reference = read_reference_config(args.reference_config)
            checkpoint = prepare_resume_checkpoint(checkpoint, reference_config=original_reference)
            saved_model_config = dict(checkpoint["model_config"])
            saved_model_config["mlp_hidden_dims"] = tuple(saved_model_config["mlp_hidden_dims"])
            model_config = FinancialSonicConfig(**saved_model_config)
            critic_config = checkpoint["critic_config"]
            env_config = checkpoint["env_config"]
            ppo_values = checkpoint["config"]
            if args.tiny and (model_config != tiny_config
                              or tuple(critic_config["hidden_dims"]) != (32, 32)):
                raise ValueError("Explicit --tiny conflicts with checkpoint model configuration")
            if args.sequence_length is not None and args.sequence_length != env_config["sequence_length"]:
                raise ValueError("Explicit sequence_length conflicts with checkpoint")
            for name in ("rollout_steps", "epochs", "num_minibatches"):
                value = getattr(args, name)
                if value is not None and value != ppo_values[name]:
                    raise ValueError(f"Explicit {name} conflicts with checkpoint")
            start_iteration = checkpoint["iteration"]
            reference_config = checkpoint.get("reference_config")
            if reference_config is None:
                raise ValueError("Checkpoint needs an explicit --reference-config; its pool cannot be inferred")
            reference_provenance = checkpoint.get("reference_provenance", "training_reference_pool")
            if (reference_config["sequence_length"] != env_config["sequence_length"]
                    or reference_config["horizon"] != env_config["horizon"]):
                raise ValueError("Reference sequence/horizon conflicts with checkpoint environment")
        else:
            model_config = tiny_config if args.tiny else FinancialSonicConfig()
            critic_config = {"input_dim": 410, "hidden_dims": (32, 32) if args.tiny else
                             (2048, 2048, 1024, 1024, 512, 512)}
            env_config = {"sequence_length": args.sequence_length or 64, "history_length": 10,
                          "horizon": 10, "encoder_denoising": True}
            ppo_values = asdict(PPOConfig())
            ppo_values.update({name: getattr(args, name) for name in
                               ("rollout_steps", "epochs", "num_minibatches") if getattr(args, name) is not None})
            start_iteration = 0
            if args.reference_config is not None:
                raise ValueError("--reference-config is only for checkpoint resume without reference metadata")
            if args.canonical is None and args.trajectory_root is None:
                raise ValueError("A new run requires --canonical or --trajectory-root")

        ppo_config = PPOConfig(**ppo_values)
        if ppo_config.num_minibatches > args.num_envs:
            raise ValueError("num_minibatches must not exceed num_envs")
        if (env_config["horizon"] != 10 or model_config.horizon != 10
                or critic_config["input_dim"] != env_config["history_length"] * 26 + 150):
            raise ValueError("Incompatible checkpoint environment and critic dimensions")
        final_iteration = start_iteration + args.iterations
        save_paths = {
            iteration: args.output_dir / f"checkpoint_{iteration:06d}.pt"
            for iteration in range(start_iteration + 1, final_iteration + 1)
            if iteration % args.save_interval == 0 or iteration == final_iteration
        }
        for path in save_paths.values():
            if path.exists() or path.is_symlink():
                raise FileExistsError(f"Refusing to overwrite checkpoint: {path}")

        # All ranks construct identical initial weights and fit statistics from
        # the global pool before their reference/environment shards diverge.
        torch.manual_seed(args.seed)
        if runtime_device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed)
        archive_progress = None
        if (args.trajectory_root is not None
                or (reference_config is not None
                    and reference_config.get("source_type", "canonical") == "trajectory_archive")):
            print(json.dumps({
                "event": "archive_load_start",
                "rank": distributed.rank,
                "local_rank": distributed.local_rank,
                "world_size": distributed.world_size,
                "trajectory_root": str(
                    args.trajectory_root
                    if args.trajectory_root is not None
                    else reference_config["trajectory_root"]
                ),
                "symbols": args.symbols,
            }), file=sys.stderr, flush=True)

            def archive_progress(processed, total):
                print(json.dumps({
                    "event": "archive_load_progress",
                    "rank": distributed.rank,
                    "local_rank": distributed.local_rank,
                    "world_size": distributed.world_size,
                    "processed_segments": processed,
                    "total_segments": total,
                }), file=sys.stderr, flush=True)

        if reference_config is None:
            if args.trajectory_root is not None:
                sequences, reference_config = load_reference_pool_from_trajectories(
                    args.trajectory_root, index_path=args.trajectory_index,
                    metadata_path=args.trajectory_metadata, manifest_path=args.trajectory_manifest,
                    context_canonical=args.canonical, symbols=args.symbols, start=args.start, end=args.end,
                    sequence_length=env_config["sequence_length"], horizon=env_config["horizon"],
                    progress_callback=archive_progress,
                )
            else:
                sequences, reference_config = load_reference_pool(
                    args.canonical, symbols=args.symbols, start=args.start, end=args.end,
                    sequence_length=env_config["sequence_length"], horizon=env_config["horizon"],
                )
        else:
            archive_reference = reference_config.get("source_type", "canonical") == "trajectory_archive"
            sequences, reference_config = resolve_reference_pool(
                reference_config,
                canonical=(_archive_context_for_resume(reference_config, args.canonical)
                           if archive_reference else args.canonical),
                symbols=args.symbols, start=args.start, end=args.end,
                trajectory_root=args.trajectory_root, trajectory_index=args.trajectory_index,
                trajectory_metadata=args.trajectory_metadata, trajectory_manifest=args.trajectory_manifest,
                progress_callback=archive_progress,
            )
        global_sequences = list(sequences)
        if distributed.enabled:
            local_sequences = global_sequences[distributed.rank::distributed.world_size]
            if not local_sequences:
                raise ValueError(
                    f"Rank {distributed.rank} received no reference sequences; "
                    f"global pool has {len(global_sequences)} entries for {distributed.world_size} ranks"
                )
        else:
            local_sequences = global_sequences
        global_reference_count = len(global_sequences)
        local_reference_count = len(local_sequences)

        with redirect_stdout(sys.stderr):
            actor, critic = make_actor_critic(
                model_config, critic_config["input_dim"], critic_hidden_dims=critic_config["hidden_dims"],
            )
        model = actor.actor_module.model
        if checkpoint is not None:
            actor.load_state_dict(checkpoint["actor"])
        elif args.normalization is not None:
            model.load_normalizers(args.normalization)
        else:
            model.fit_normalizers(
                torch.tensor([row["current_state"] for row in global_sequences], dtype=torch.float32),
                torch.tensor([row["future_reference"] for row in global_sequences], dtype=torch.float32),
                torch.tensor([row["future_mask"] for row in global_sequences], dtype=torch.bool),
            )
        actor.to(runtime_device)
        critic.to(runtime_device)
        # Keep model initialization synchronized, but give each environment
        # stream an independent random sequence and denoising mask.
        torch.manual_seed(args.seed + distributed.rank)
        if runtime_device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed + distributed.rank)
        env = MonthlyTrackingEnv(
            local_sequences, args.num_envs, model.future_normalizer.center[0], model.future_normalizer.scale[0],
            history_length=env_config["history_length"], seed=args.seed + distributed.rank,
            device=runtime_device, encoder_denoising=env_config["encoder_denoising"],
        )
        del sequences, global_sequences, local_sequences
        trainer = FinancialPPOTrainer(
            actor, critic, env, ppo_config, reference_config=reference_config,
            reference_provenance=reference_provenance, distributed=distributed,
            local_reference_count=local_reference_count if distributed.enabled else None,
        )
        if args.resume is not None:
            trainer.load_checkpoint(args.resume, reference_config=original_reference)
        distributed.broadcast_module(actor)
        distributed.broadcast_module(critic)
        recipe = json.loads(json.dumps({
            "schema_version": 1, "dataset_partition": "none",
            "reference_config": reference_config, "reference_provenance": reference_provenance,
            "model_config": asdict(model.config), "critic_config": critic.finance_config,
            "ppo_config": asdict(ppo_config), "env_config": trainer._environment_config(),
            "reward_contract": env.reward_contract,
            "encoder_denoising_contract": encoder_denoising_contract(),
        }))
        if distributed.is_main_process:
            recipe_path = _write_run_recipe(args.output_dir, recipe, trainer.training_migrations)
        else:
            recipe_path = args.output_dir / "run_config.json"
        distributed.barrier()
        if not distributed.is_main_process and not recipe_path.exists():
            migrated_path = args.output_dir / "run_config.schema4.json"
            if migrated_path.exists():
                recipe_path = migrated_path
        monitor = create_tensorboard_monitor(
            args.output_dir,
            enabled=args.logger == "tensorboard",
            is_main_process=distributed.is_main_process,
            resume_iteration=trainer.iteration,
        )

        metadata = distributed.metadata(
            reference_count_global=global_reference_count,
            reference_count_local=local_reference_count,
            env_count_local=args.num_envs,
        )
        print(json.dumps({
            "event": "start", "mode": "privileged_reference_tracking", "dataset_partition": "none",
            "runtime_sharding": "round_robin" if distributed.enabled else "none",
            "environment_sharding": "per_rank" if distributed.enabled else "single_process",
            "excluded_heads": ["s_pred", "z_gmm"],
            "resume_mode": "restart_reference_episodes" if checkpoint is not None else "new_run",
            "training_migrations": trainer.training_migrations, "run_config": str(recipe_path),
            "canonical": reference_config["canonical"], "symbols": reference_config["symbols"],
            "period_bounds": [reference_config["start"], reference_config["end"]],
            "reference_config": reference_config, "reference_provenance": reference_provenance,
            "reward_contract": env.reward_contract,
            "encoder_denoising_contract": encoder_denoising_contract(),
            "sequence_count": global_reference_count, "sequence_length": env.sequence_length,
            "reference_count_global": global_reference_count,
            "reference_count_local": local_reference_count,
            "env_count_global": args.num_envs * distributed.world_size, "env_count_local": args.num_envs,
            "rank": distributed.rank, "local_rank": distributed.local_rank,
            "world_size": distributed.world_size, "distributed_backend": distributed.backend,
            "distributed": metadata,
            "num_envs": args.num_envs, "device": str(env.device), "seed": args.seed + distributed.rank,
            "model_config": asdict(model.config), "critic_config": critic.finance_config,
            "ppo_config": asdict(ppo_config), "history_length": env.history_length,
            "normalization": str(args.normalization) if args.normalization is not None else None,
            "resume": str(args.resume) if args.resume is not None else None,
            "logger": args.logger,
            "tensorboard_dir": str(args.output_dir / "tensorboard")
            if args.logger == "tensorboard" else None,
            "start_iteration": trainer.iteration, "additional_iterations": args.iterations,
        }), flush=True)
        for _ in range(args.iterations):
            metrics = trainer.train_iteration()
            print(json.dumps({
                "event": "iteration", **metrics, "rank": distributed.rank,
                "local_rank": distributed.local_rank, "world_size": distributed.world_size,
                "post_update_checksum": _parameter_checksum(actor, critic),
            }, allow_nan=False), flush=True)
            monitor.log(metrics)
            if trainer.iteration in save_paths:
                distributed.barrier()
                if distributed.is_main_process:
                    trainer.save_checkpoint(save_paths[trainer.iteration])
                    monitor.flush()
                distributed.barrier()
        distributed.barrier()
        print(json.dumps({
            "event": "end", "iteration": trainer.iteration,
            "checkpoint": str(save_paths[trainer.iteration]) if distributed.is_main_process else None,
            "rank": distributed.rank, "local_rank": distributed.local_rank,
            "world_size": distributed.world_size,
        }), flush=True)
        return 0
    finally:
        try:
            monitor.close()
        finally:
            torch.set_num_threads(previous_threads)
            distributed.close()


if __name__ == "__main__":
    raise SystemExit(main())
