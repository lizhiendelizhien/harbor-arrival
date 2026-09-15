"""Bounded privileged reference-tracking PPO, without world heads or data partitions.

Run with python -m scripts.train_finance_sonic. Real future descriptors are
privileged inputs; these rewards do not measure forecast accuracy or PnL.
Resume restores learning state but restarts reference episodes and model caches.
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from dataclasses import asdict
import json
from pathlib import Path
import sys

import torch

from gear_sonic.finance.denoising import encoder_denoising_contract
from gear_sonic.finance.environment import MonthlyTrackingEnv
from gear_sonic.finance.model import FinancialSonicConfig
from gear_sonic.finance.policy import make_actor_critic
from gear_sonic.finance.ppo import FinancialPPOTrainer, PPOConfig
from gear_sonic.finance.resume import legacy_run_recipe_matches, prepare_resume_checkpoint
from gear_sonic.finance.reference import (
    load_reference_pool, read_reference_config, reference_identity, resolve_reference_pool,
)


def _parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical", type=Path, help="New-run data, or identical relocated data on resume")
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
    parser.add_argument("--start", help="Optional inclusive first anchor month, YYYY-MM")
    parser.add_argument("--end", help="Optional inclusive final label month, YYYY-MM")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--normalization", type=Path)
    source.add_argument("--resume", type=Path)
    parser.add_argument("--reference-config", type=Path, help="Original reference recipe for a checkpoint missing reference metadata")
    return parser.parse_args(argv)


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
    args = _parse_args(argv)
    for name in ("iterations", "num_envs", "threads", "save_interval", "sequence_length",
                 "rollout_steps", "epochs", "num_minibatches"):
        value = getattr(args, name)
        if value is not None and value < 1:
            raise ValueError(f"{name} must be positive")
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
        if reference_config.get("reward_contract") != checkpoint["reward_contract"]:
            raise ValueError("Reference reward contract conflicts with checkpoint reward contract")
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
        if args.canonical is None:
            raise ValueError("--canonical is required for a new training run")
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

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(args.threads)
    try:
        torch.manual_seed(args.seed)
        if reference_config is None:
            sequences, reference_config = load_reference_pool(
                args.canonical, symbols=args.symbols, start=args.start, end=args.end,
                sequence_length=env_config["sequence_length"], horizon=env_config["horizon"],
            )
        else:
            sequences, reference_config = resolve_reference_pool(
                reference_config, canonical=args.canonical, symbols=args.symbols, start=args.start, end=args.end,
            )
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
                torch.tensor([row["current_state"] for row in sequences], dtype=torch.float32),
                torch.tensor([row["future_reference"] for row in sequences], dtype=torch.float32),
                torch.tensor([row["future_mask"] for row in sequences], dtype=torch.bool),
            )
        actor.to(args.device)
        critic.to(args.device)
        env = MonthlyTrackingEnv(
            sequences, args.num_envs, model.future_normalizer.center[0], model.future_normalizer.scale[0],
            history_length=env_config["history_length"], seed=args.seed, device=args.device,
            encoder_denoising=env_config["encoder_denoising"],
        )
        trainer = FinancialPPOTrainer(actor, critic, env, ppo_config,
                                      reference_config=reference_config,
                                      reference_provenance=reference_provenance)
        if args.resume is not None:
            trainer.load_checkpoint(args.resume, reference_config=original_reference)
        recipe = json.loads(json.dumps({
            "schema_version": 1, "dataset_partition": "none",
            "reference_config": reference_config, "reference_provenance": reference_provenance,
            "model_config": asdict(model.config), "critic_config": critic.finance_config,
            "ppo_config": asdict(ppo_config), "env_config": trainer._environment_config(),
            "reward_contract": env.reward_contract,
            "encoder_denoising_contract": encoder_denoising_contract(),
        }))
        recipe_path = _write_run_recipe(args.output_dir, recipe, trainer.training_migrations)
        print(json.dumps({
            "event": "start", "mode": "privileged_reference_tracking", "dataset_partition": "none",
            "excluded_heads": ["s_pred", "z_gmm"],
            "resume_mode": "restart_reference_episodes" if checkpoint is not None else "new_run",
            "training_migrations": trainer.training_migrations, "run_config": str(recipe_path),
            "canonical": reference_config["canonical"], "symbols": reference_config["symbols"],
            "period_bounds": [reference_config["start"], reference_config["end"]],
            "reference_config": reference_config, "reference_provenance": reference_provenance,
            "reward_contract": env.reward_contract,
            "encoder_denoising_contract": encoder_denoising_contract(),
            "sequence_count": len(sequences), "sequence_length": env.sequence_length,
            "num_envs": args.num_envs, "device": str(env.device), "seed": args.seed,
            "model_config": asdict(model.config), "critic_config": critic.finance_config,
            "ppo_config": asdict(ppo_config), "history_length": env.history_length,
            "normalization": str(args.normalization) if args.normalization is not None else None,
            "resume": str(args.resume) if args.resume is not None else None,
            "start_iteration": trainer.iteration, "additional_iterations": args.iterations,
        }), flush=True)
        for _ in range(args.iterations):
            metrics = trainer.train_iteration()
            print(json.dumps({"event": "iteration", **metrics}, allow_nan=False), flush=True)
            if trainer.iteration in save_paths:
                trainer.save_checkpoint(save_paths[trainer.iteration])
        print(json.dumps({
            "event": "end", "iteration": trainer.iteration,
            "checkpoint": str(save_paths[trainer.iteration]),
        }), flush=True)
        return 0
    finally:
        torch.set_num_threads(previous_threads)


if __name__ == "__main__":
    raise SystemExit(main())
