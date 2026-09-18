"""Specification tests for the trajectory archive and real DDP launcher.

These tests intentionally exercise the public command line boundary.  The
archive fixture is written in the same format as the production reconstruction
job, while the distributed test uses two CPU/gloo ranks so it is useful on
machines without CUDA.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import sys

import numpy as np
import pytest
import torch


CANONICAL_FIELDS = [
    "symbol", "period", "raw_date", "raw_open", "raw_close", "raw_high", "raw_low",
    "raw_volume", "raw_amount", "raw_turnover", "qfq_date", "qfq_open", "qfq_close",
    "qfq_high", "qfq_low", "hfq_date", "hfq_open", "hfq_close", "hfq_high", "hfq_low",
    "raw_valid", "qfq_valid", "hfq_valid", "quality_flags",
]


def _canonical_row(symbol: str, index: int) -> dict[str, object]:
    year = 2010 + index // 12
    month = index % 12 + 1
    period = f"{year:04d}-{month:02d}"
    # Different slopes make the market-context pass observable while keeping
    # all bars valid and deterministic.
    close = 20.0 + index * (0.15 if symbol == "AAA" else 0.11)
    opened = close * 0.99
    high = close * 1.02
    low = opened * 0.98
    volume = 1000.0 + index * (3.0 if symbol == "AAA" else 5.0)
    qfq = {"open": opened * 2.0, "close": close * 2.0,
           "high": high * 2.0, "low": low * 2.0}
    raw = {"open": opened, "close": close, "high": high, "low": low}
    row: dict[str, object] = {
        "symbol": symbol,
        "period": period,
        "raw_date": f"{period}-28",
        "raw_open": raw["open"], "raw_close": raw["close"],
        "raw_high": raw["high"], "raw_low": raw["low"],
        "raw_volume": volume, "raw_amount": close * volume, "raw_turnover": 1.0,
        "qfq_date": f"{period}-28",
        "qfq_open": qfq["open"], "qfq_close": qfq["close"],
        "qfq_high": qfq["high"], "qfq_low": qfq["low"],
        "hfq_date": f"{period}-28",
        "hfq_open": raw["open"], "hfq_close": raw["close"],
        "hfq_high": raw["high"], "hfq_low": raw["low"],
        "raw_valid": 1, "qfq_valid": 1, "hfq_valid": 1, "quality_flags": "",
    }
    return row


def _write_canonical(path: Path, months: int = 60) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CANONICAL_FIELDS)
        writer.writeheader()
        for symbol in ("AAA", "BBB"):
            for index in range(months):
                writer.writerow(_canonical_row(symbol, index))


@pytest.fixture
def trajectory_archive(tmp_path: Path):
    """A complete companion-file archive plus its source canonical table."""
    from gear_sonic.finance.trajectory import reconstruct_monthly_trajectories

    canonical = tmp_path / "monthly_canonical.csv"
    _write_canonical(canonical)
    archive = tmp_path / "trajectories"
    reconstruct_monthly_trajectories(canonical, archive, min_training_length=17)
    assert (archive / "trajectory_index.csv").is_file()
    assert (archive / "metadata.pkl").is_file()
    assert (archive / "manifest.json").is_file()
    assert list((archive / "trajectories").glob("*.pkl"))
    return canonical, archive


def _archive_loader_args(canonical: Path, archive: Path) -> dict[str, object]:
    return {
        "trajectory_root": archive,
        "index_path": archive / "trajectory_index.csv",
        "metadata_path": archive / "metadata.pkl",
        "manifest_path": archive / "manifest.json",
        "context_canonical": canonical,
        "symbols": ["AAA"],
        "sequence_length": 2,
        "horizon": 10,
    }


def test_trajectory_archive_loader_reconstructs_fixed_sequences_and_config(trajectory_archive):
    """Archive loading must preserve canonical sequence semantics and identity."""
    from gear_sonic.finance.reference import (
        load_reference_pool,
        load_reference_pool_from_trajectories,
    )

    canonical, archive = trajectory_archive
    expected_sequences, expected_canonical_config = load_reference_pool(
        canonical, symbols=["AAA"], sequence_length=2, horizon=10,
    )
    sequences, config = load_reference_pool_from_trajectories(**_archive_loader_args(canonical, archive))

    assert len(sequences) == len(expected_sequences) > 0
    assert config["dataset_partition"] == "none"
    assert config["sequence_count"] == len(sequences)
    assert config["horizon"] == 10
    assert config["sequence_length"] == 2
    assert len(config["sequence_index_sha256"]) == 64
    assert config["reward_contract"]["name"] == "financial_tracking_v1"
    assert config["feature_schema"] == expected_canonical_config["feature_schema"]

    for actual, expected in zip(sequences, expected_sequences):
        assert actual["symbol"] == expected["symbol"]
        assert actual["periods"] == expected["periods"]
        assert actual["target_end_period"] == expected["target_end_period"]
        np.testing.assert_allclose(actual["current_state"], expected["current_state"], rtol=0, atol=1e-12)
        np.testing.assert_allclose(actual["future_reference"], expected["future_reference"], rtol=0, atol=1e-12)
        assert actual["future_mask"] == expected["future_mask"]

    repeated, repeated_config = load_reference_pool_from_trajectories(
        **_archive_loader_args(canonical, archive)
    )
    assert repeated == sequences
    assert repeated_config == config


def test_trajectory_archive_loader_uses_manifest_source_when_csv_override_is_omitted(trajectory_archive):
    """The manifest source path is the default context; an override remains optional."""
    from gear_sonic.finance.reference import load_reference_pool_from_trajectories

    canonical, archive = trajectory_archive
    args = _archive_loader_args(canonical, archive)
    with_csv, _ = load_reference_pool_from_trajectories(**args)
    archive_only, config = load_reference_pool_from_trajectories(
        **{**args, "context_canonical": None}
    )
    assert archive_only == with_csv
    assert config["canonical"] == str(canonical.resolve())


def test_trajectory_archive_loader_reports_segment_progress(trajectory_archive):
    """Large archive startup must expose progress without changing its result."""
    canonical, archive = trajectory_archive
    progress = []
    from gear_sonic.finance.reference import load_reference_pool_from_trajectories

    sequences, _ = load_reference_pool_from_trajectories(
        **_archive_loader_args(canonical, archive),
        progress_callback=lambda processed, total: progress.append((processed, total)),
    )

    assert sequences
    assert progress[0] == (0, 1)
    assert progress[-1] == (1, 1)


@pytest.mark.parametrize("companion", ["trajectory_index.csv", "metadata.pkl", "manifest.json"])
def test_trajectory_archive_loader_validates_each_companion_file(trajectory_archive, companion):
    """A stale or malformed index/metadata/manifest must fail closed."""
    from gear_sonic.finance.reference import load_reference_pool_from_trajectories

    canonical, archive = trajectory_archive
    broken = archive.parent / f"broken_{Path(companion).stem}"
    shutil.copytree(archive, broken)
    path = broken / companion
    if companion == "trajectory_index.csv":
        text = path.read_text(encoding="utf-8")
        path.write_text(text.replace("trajectories/", "missing/", 1), encoding="utf-8")
    elif companion == "metadata.pkl":
        payload = pickle.loads(path.read_bytes())
        key = next(iter(payload))
        payload[key]["length"] += 1
        path.write_bytes(pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))
    else:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest["schema_version"] = 999
        path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises((ValueError, FileNotFoundError, KeyError), match="(?i)archive|index|metadata|manifest|schema|trajectory"):
        load_reference_pool_from_trajectories(
            **{**_archive_loader_args(canonical, broken),
               "index_path": broken / "trajectory_index.csv",
               "metadata_path": broken / "metadata.pkl",
               "manifest_path": broken / "manifest.json"}
        )


def _json_events(output: str) -> list[dict]:
    decoder = json.JSONDecoder()
    events = []
    for line in output.splitlines():
        offset = 0
        # torchrun can interleave two flushed JSON lines without preserving
        # their newline, so decode every object instead of assuming one/line.
        while True:
            start = line.find("{", offset)
            if start < 0:
                break
            try:
                value, consumed = decoder.raw_decode(line[start:])
            except json.JSONDecodeError:
                offset = start + 1
                continue
            if isinstance(value, dict) and "event" in value:
                events.append(value)
            offset = start + consumed
    return events


def _assert_finite_tensors(value) -> None:
    if isinstance(value, torch.Tensor):
        if value.is_floating_point() or value.is_complex():
            assert bool(torch.isfinite(value).all()), "non-finite tensor in checkpoint"
    elif isinstance(value, dict):
        for item in value.values():
            _assert_finite_tensors(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _assert_finite_tensors(item)


def test_distributed_context_averages_parameter_gradients(monkeypatch):
    """Manual synchronization must be an all-reduce sum followed by world-size division."""
    import torch.distributed as dist
    from gear_sonic.finance.distributed import DistributedContext

    calls = []

    def fake_all_reduce(value, op=dist.ReduceOp.SUM):
        calls.append(value.detach().clone())
        # A two-rank test where both local gradients equal the input gradient.
        value.mul_(2)

    monkeypatch.setattr(dist, "all_reduce", fake_all_reduce)
    context = DistributedContext(enabled=True, rank=0, local_rank=0, world_size=2, backend="gloo")
    first = torch.nn.Parameter(torch.tensor([3.0]))
    second = torch.nn.Parameter(torch.tensor([4.0]))
    second.grad = None
    first.grad = torch.tensor([7.0])

    context.sync_gradients([first, second])

    torch.testing.assert_close(first.grad, torch.tensor([7.0]))
    assert second.grad is not None
    torch.testing.assert_close(second.grad, torch.zeros_like(second))
    # Gradients are flattened into one collective instead of synchronizing
    # every parameter tensor independently.
    assert len(calls) == 1


def test_distributed_advantage_standardization_matches_sample_std():
    """The pooled path keeps the original PPO sample-standard-deviation convention."""
    from gear_sonic.finance.distributed import DistributedContext

    values = torch.tensor([[1.0, 2.0], [4.0, 8.0]])
    actual = DistributedContext().standardize(values)
    expected = (values - values.mean()) / (values.std() + 1e-8)
    torch.testing.assert_close(actual, expected)


def test_checkpoint_metadata_preserves_distributed_contract(tmp_path):
    """Checkpoint metadata must make rank ownership and reduction auditable."""
    from gear_sonic.finance.distributed import DistributedContext
    from tests.test_finance_ppo import make_trainer

    trainer = make_trainer(rollout_steps=1, num_envs=2, num_minibatches=1)
    trainer.distributed = DistributedContext(
        enabled=True, rank=0, local_rank=0, world_size=2, backend="gloo",
    )
    trainer.global_reference_count = 4
    trainer.local_reference_count = 2
    path = tmp_path / "distributed.pt"
    trainer.save_checkpoint(path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    assert payload["distributed"] == {
        "enabled": True, "backend": "gloo", "rank": 0, "local_rank": 0,
        "world_size": 2, "gradient_reduction": "all_reduce",
        "reference_sharding": "round_robin", "environment_sharding": "per_rank",
        "reference_count_global": 4, "reference_count_local": 2,
        "env_count_global": 4, "env_count_local": 2,
    }


def test_non_main_rank_cannot_publish_checkpoint(tmp_path):
    """The trainer API itself enforces rank-zero-only checkpoint publication."""
    from gear_sonic.finance.distributed import DistributedContext
    from tests.test_finance_ppo import make_trainer

    trainer = make_trainer(rollout_steps=1, num_envs=2, num_minibatches=1)
    trainer.distributed = DistributedContext(
        enabled=True, rank=1, local_rank=1, world_size=2, backend="gloo",
    )
    with pytest.raises(RuntimeError, match="rank 0"):
        trainer.save_checkpoint(tmp_path / "should-not-exist.pt")
    assert not (tmp_path / "should-not-exist.pt").exists()


def test_two_rank_cpu_ddp_shards_pool_reduces_gradients_and_saves_rank0_only(trajectory_archive, tmp_path):
    """Two gloo ranks must train one synchronized model and publish one checkpoint."""
    canonical, archive = trajectory_archive
    output = tmp_path / "ddp_run"
    command = [
        sys.executable, "-m", "torch.distributed.run", "--rdzv-backend=static",
        "--master-addr=127.0.0.1", "--master-port=29643", "--nproc_per_node=2",
        "-m", "scripts.train_finance_sonic", "--distributed",
        # The manifest records the canonical source, so the archive path alone
        # is sufficient for the normal one-click data contract.
        "--trajectory-root", str(archive),
        "--symbols", "AAA", "BBB", "--output-dir", str(output), "--tiny",
        "--iterations", "1", "--sequence-length", "2", "--rollout-steps", "2",
        "--epochs", "1", "--num-minibatches", "1", "--num-envs", "4",
        "--threads", "1", "--device", "cpu", "--seed", "123", "--logger", "tensorboard",
    ]
    environment = dict(os.environ)
    environment.update({
        "PYTHONPATH": str(Path(__file__).resolve().parents[1])
        + os.pathsep + environment.get("PYTHONPATH", ""),
        "CUDA_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "TORCH_DISTRIBUTED_DEBUG": "DETAIL",
    })
    result = subprocess.run(
        command, cwd=Path(__file__).resolve().parents[1], env=environment,
        capture_output=True, text=True, timeout=180,
    )
    if result.returncode and "operation not permitted" in result.stderr.lower() \
            and "server socket" in result.stderr.lower():
        pytest.skip("this runner forbids local TCP rendezvous; run DDP smoke on a network-enabled worker")
    assert result.returncode == 0, f"torchrun failed\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"

    events = _json_events(result.stdout + "\n" + result.stderr)
    starts = [event for event in events if event["event"] == "start"]
    assert {event.get("rank") for event in starts} == {0, 1}
    assert all(event.get("world_size") == 2 for event in starts)
    assert {event.get("reference_count_local") for event in starts} == {22}
    assert {event.get("reference_count_global") for event in starts} == {44}
    # Match the Sonic launcher convention: --num-envs is the per-rank count;
    # the effective global batch therefore scales with world size.
    assert {event.get("env_count_local") for event in starts} == {4}
    assert {event.get("env_count_global") for event in starts} == {8}

    updates = [event for event in events if event["event"] == "iteration"]
    assert len(updates) == 2
    assert {event.get("iteration") for event in updates} == {1}
    assert {event.get("world_size") for event in updates} == {2}
    checksums = {event.get("post_update_checksum") for event in updates}
    assert len(checksums) == 1 and None not in checksums

    checkpoints = sorted(output.glob("checkpoint_*.pt"))
    assert [path.name for path in checkpoints] == ["checkpoint_000001.pt"]
    assert not list(output.glob("checkpoint_*.tmp"))
    assert not list(output.glob("rank_*/checkpoint_*.pt"))
    assert sorted(path.name for path in output.glob("run_config*.json")) == ["run_config.json"]

    checkpoint = torch.load(checkpoints[0], map_location="cpu", weights_only=True)
    _assert_finite_tensors(checkpoint["actor"])
    _assert_finite_tensors(checkpoint["critic"])
    _assert_finite_tensors(checkpoint["optimizer"])
    assert checkpoint["iteration"] == 1
    assert checkpoint["distributed"]["rank"] == 0
    assert checkpoint["distributed"]["world_size"] == 2
    assert checkpoint["distributed"]["gradient_reduction"] == "all_reduce"
    assert checkpoint["distributed"]["reference_count_global"] == 44
    assert checkpoint["distributed"]["reference_count_local"] == 22
    # One shared prior count plus 2 ranks * 4 envs * 2 rollout steps.
    assert checkpoint["critic"]["running_mean_std.count"].item() == pytest.approx(17.0)

    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    event_dir = output / "tensorboard"
    assert len(list(event_dir.glob("events.out.tfevents.*"))) == 1
    accumulator = EventAccumulator(str(event_dir), size_guidance={"scalars": 0})
    accumulator.Reload()
    assert [item.step for item in accumulator.Scalars("Loss/ppo_loss")] == [1]


def test_finance_multi_gpu_launcher_dry_run_maps_data_resume_and_torchrun(tmp_path):
    """The one-click launcher must expose an inspectable torchrun command."""
    launcher = Path("scripts/train_finance_sonic_4gpu.sh")
    assert launcher.is_file()
    source = launcher.read_text(encoding="utf-8")
    for token in ("set -euo pipefail", "torch.distributed.run", "--nproc_per_node",
                  "--distributed", "--trajectory-root", "FINANCE_DRY_RUN",
                  "PYTHON_CANDIDATES", "importlib.util", "missing_modules",
                  ".[finance]", "NPROC_PER_NODE", "NCCL_DEBUG", "NCCL_P2P_DISABLE",
                  "NCCL_SHM_DISABLE", "NCCL_IB_DISABLE", "NCCL_ALGO", "NCCL_PROTO",
                  "TORCH_NCCL_ASYNC_ERROR_HANDLING", "PYTORCH_CUDA_ALLOC_CONF",
                  "AUTO_TMUX", "TMUX_SESSION", "TMUX_ATTACH",
                  "FINANCE_PYTHONPATH", "LOCAL_PYTHON_DEPS", "LOGGER",
                  "tensorboard --logdir", "--logger"):
        assert token in source

    output = tmp_path / "launcher-output"
    checkpoint = tmp_path / "checkpoint.pt"
    environment = dict(os.environ)
    environment.update({
        "NUM_GPUS": "2", "CUDA_VISIBLE_DEVICES": "0,1", "NUM_ENVS": "4", "SEED": "42",
        "TRAJECTORY_ROOT": str(tmp_path / "archive"), "OUTPUT_DIR": str(output),
        "FINANCE_MULTI_GPU_LAUNCHER": "torchrun", "FINANCE_DRY_RUN": "1",
        "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": "29642",
    })
    result = subprocess.run(
        ["bash", str(launcher), "--resume", str(checkpoint)],
        cwd=Path(__file__).resolve().parents[1], env=environment,
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    command_text = result.stdout + "\n" + result.stderr
    assert "Preparing Finance Sonic launcher" in command_text
    assert "torchrun" in command_text or "torch.distributed.run" in command_text
    assert "--nproc_per_node=2" in command_text or "--nproc_per_node 2" in command_text
    assert "scripts.train_finance_sonic" in command_text
    assert "--distributed" in command_text
    assert "--logger tensorboard" in command_text or "--logger=tensorboard" in command_text
    assert "tensorboard --logdir" in command_text
    assert f"--trajectory-root {environment['TRAJECTORY_ROOT']}" in command_text
    assert f"--output-dir {environment['OUTPUT_DIR']}" in command_text
    assert "--num-envs 4" in command_text or "--num-envs=4" in command_text
    assert "--seed 42" in command_text or "--seed=42" in command_text
    assert f"--resume {checkpoint}" in command_text or f"--resume={checkpoint}" in command_text


def test_finance_launcher_can_disable_tensorboard(tmp_path):
    launcher = Path("scripts/train_finance_sonic_4gpu.sh")
    environment = dict(os.environ)
    environment.update({
        "LOGGER": "none", "NUM_GPUS": "1", "CUDA_VISIBLE_DEVICES": "0",
        "NUM_ENVS": "1", "OUTPUT_DIR": str(tmp_path / "output"),
        "FINANCE_MULTI_GPU_LAUNCHER": "torchrun", "FINANCE_DRY_RUN": "1",
        "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": "29649",
    })
    result = subprocess.run(
        ["bash", str(launcher), "--dry-run"],
        cwd=Path(__file__).resolve().parents[1], env=environment,
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    output = result.stdout + "\n" + result.stderr
    assert "--logger none" in output or "--logger=none" in output
    assert "tensorboard --logdir" not in output


def test_finance_launcher_reports_incomplete_tensorboard_before_torchrun(tmp_path):
    launcher = Path("scripts/train_finance_sonic_4gpu.sh")
    fake_python = tmp_path / "fake-python"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        "if [[ \"$1\" == \"-c\" && \"$2\" == *\"torch.utils.tensorboard\"* ]]; then exit 1; fi\n"
        "exit 0\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    environment = dict(os.environ)
    environment.update({
        "PYTHON_BIN": str(fake_python), "LOGGER": "tensorboard", "DEVICE": "cpu",
        "CUDA_VISIBLE_DEVICES": "", "NUM_GPUS": "1", "NUM_ENVS": "1",
        "FINANCE_MULTI_GPU_LAUNCHER": "torchrun", "MASTER_PORT": "29650",
        "OUTPUT_DIR": str(tmp_path / "output"), "LAUNCH_LOG_DIR": str(tmp_path / "logs"),
    })
    result = subprocess.run(
        ["bash", str(launcher)], cwd=Path(__file__).resolve().parents[1],
        env=environment, capture_output=True, text=True, check=False,
    )
    output = result.stdout + "\n" + result.stderr
    assert result.returncode == 2
    assert "LOGGER=tensorboard requires torch.utils.tensorboard" in output
    assert "LOGGER=none" in output


def test_finance_launcher_reports_nproc_and_nccl_contract_in_dry_run(tmp_path):
    """The launcher exposes the same generic process contract as table tennis."""
    launcher = Path("scripts/train_finance_sonic_4gpu.sh")
    environment = dict(os.environ)
    environment.update({
        "NUM_GPUS": "4", "NUM_ENVS": "1024", "CUDA_VISIBLE_DEVICES": "0,1,2,3",
        "OUTPUT_DIR": str(tmp_path / "output"), "LAUNCH_LOG_DIR": str(tmp_path / "logs"),
        "FINANCE_MULTI_GPU_LAUNCHER": "torchrun", "FINANCE_DRY_RUN": "1",
        "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": "29647",
        "NCCL_DEBUG": "INFO", "NCCL_P2P_DISABLE": "0", "NCCL_ALGO": "Tree",
    })
    result = subprocess.run(
        ["bash", str(launcher), "--dry-run"],
        cwd=Path(__file__).resolve().parents[1], env=environment,
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    output = result.stdout + "\n" + result.stderr
    assert "NPROC_PER_NODE=4" in output
    assert "NUM_ENVS=1024 per rank (global=4096)" in output
    assert "NCCL_DEBUG=INFO" in output
    assert "NCCL_P2P_DISABLE=0" in output
    assert "NCCL_ALGO=Tree" in output
    assert "--nproc_per_node=4" in output or "--nproc_per_node 4" in output


def test_finance_launcher_tmux_wrapper_preserves_finance_invocation(tmp_path):
    """AUTO_TMUX creates a non-recursive wrapper without launching training."""
    launcher = Path("scripts/train_finance_sonic_4gpu.sh")
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    capture = tmp_path / "tmux-args.txt"
    fake_tmux = fake_bin / "tmux"
    fake_tmux.write_text(
        "#!/usr/bin/env bash\n"
        "if [[ \"$1\" == has-session ]]; then exit 1; fi\n"
        "printf '%s\\n' \"$@\" > \"$TMUX_CAPTURE\"\n"
        "exit 0\n",
        encoding="utf-8",
    )
    fake_tmux.chmod(0o755)
    env_file = tmp_path / "finance.env"
    run_file = tmp_path / "finance.run.sh"
    log_file = tmp_path / "finance.tmux.log"
    explicit_python = tmp_path / "finance-python"
    explicit_python.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    explicit_python.chmod(0o755)
    environment = dict(os.environ)
    environment.update({
        "PATH": f"{fake_bin}:{environment['PATH']}", "TMUX_CAPTURE": str(capture),
        "AUTO_TMUX": "1", "TMUX_ATTACH": "0", "TMUX_SESSION": "finance-test-session",
        "TMUX_ENV_FILE": str(env_file), "TMUX_RUN_FILE": str(run_file),
        "TMUX_LOG_FILE": str(log_file), "PYTHON_BIN": str(explicit_python),
        "NUM_GPUS": "2", "NUM_ENVS": "4", "CUDA_VISIBLE_DEVICES": "0,1",
        "FINANCE_DRY_RUN": "1", "FINANCE_MULTI_GPU_LAUNCHER": "torchrun",
        "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": "29648",
        "OUTPUT_DIR": str(tmp_path / "output"), "LAUNCH_LOG_DIR": str(tmp_path / "logs"),
    })
    environment.pop("TMUX", None)
    result = subprocess.run(
        ["bash", str(launcher), "--dry-run", "--", "--custom-finance-arg"],
        cwd=Path(__file__).resolve().parents[1], env=environment,
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert capture.is_file()
    assert "new-session" in capture.read_text(encoding="utf-8")
    assert run_file.is_file()
    wrapper = run_file.read_text(encoding="utf-8")
    assert "export AUTO_TMUX=0" in wrapper
    env_snapshot = env_file.read_text(encoding="utf-8")
    assert f"PYTHON_BIN={explicit_python}" in env_snapshot
    assert "LOGGER=tensorboard" in env_snapshot
    assert "SEQUENCE_LENGTH_EXPLICIT=0" in env_snapshot
    assert "TRAJECTORY_ROOT_EXPLICIT=0" in env_snapshot
    assert "export SEQUENCE_LENGTH_EXPLICIT" in wrapper
    assert "export TRAJECTORY_ROOT_EXPLICIT" in wrapper
    assert "--custom-finance-arg" in wrapper


def test_finance_launcher_resume_inherits_saved_source_when_archive_is_not_explicit(tmp_path):
    """A generic resume must not force archive arguments onto a canonical checkpoint."""
    launcher = Path("scripts/train_finance_sonic_4gpu.sh")
    output = tmp_path / "launcher-output"
    checkpoint = tmp_path / "canonical-checkpoint.pt"
    environment = dict(os.environ)
    for name in (
        "TRAJECTORY_ROOT", "TRAJECTORY_INDEX", "TRAJECTORY_METADATA",
        "TRAJECTORY_MANIFEST", "CONTEXT_CANONICAL", "FINANCE_CONTEXT_CANONICAL",
        "CANONICAL",
    ):
        environment.pop(name, None)
    environment.update({
        "NUM_GPUS": "1", "CUDA_VISIBLE_DEVICES": "0", "NUM_ENVS": "2",
        "OUTPUT_DIR": str(output), "FINANCE_MULTI_GPU_LAUNCHER": "torchrun",
        "FINANCE_DRY_RUN": "1", "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": "29644",
    })
    result = subprocess.run(
        ["bash", str(launcher), "--resume", str(checkpoint)],
        cwd=Path(__file__).resolve().parents[1], env=environment,
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    command_line = next(
        line for line in (result.stdout + "\n" + result.stderr).splitlines()
        if line.startswith("  command=")
    )
    assert "--trajectory-root" not in command_line
    assert f"--resume {checkpoint}" in command_line or f"--resume={checkpoint}" in command_line


def test_finance_launcher_respects_explicit_python_bin_in_dry_run(tmp_path):
    """An explicit interpreter must take precedence over environment probing."""
    launcher = Path("scripts/train_finance_sonic_4gpu.sh")
    explicit_python = tmp_path / "finance-python"
    # Dry-run mode never invokes torchrun, so a tiny executable is sufficient
    # to verify that the path is preserved verbatim in the resolved command.
    explicit_python.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    explicit_python.chmod(0o755)
    environment = dict(os.environ)
    environment.update({
        "PYTHON_BIN": str(explicit_python),
        "NUM_GPUS": "1",
        "NUM_ENVS": "2",
        "CUDA_VISIBLE_DEVICES": "0",
        "OUTPUT_DIR": str(tmp_path / "output"),
        "FINANCE_DRY_RUN": "1",
        "MASTER_ADDR": "127.0.0.1",
        "MASTER_PORT": "29645",
    })
    result = subprocess.run(
        ["bash", str(launcher), "--dry-run"],
        cwd=Path(__file__).resolve().parents[1], env=environment,
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    command_text = result.stdout + "\n" + result.stderr
    assert f"python={explicit_python}" in command_text
    assert f"{explicit_python} -m torch.distributed.run" in command_text


def test_finance_launcher_auto_selects_virtualenv_candidate_in_dry_run(tmp_path):
    """Without PYTHON_BIN, the first executable torch-capable candidate wins."""
    launcher = Path("scripts/train_finance_sonic_4gpu.sh")
    virtualenv = tmp_path / "finance-venv"
    candidate = virtualenv / "bin" / "python"
    candidate.parent.mkdir(parents=True)
    # The launcher probes candidates with ``-c 'import torch'``.  Returning
    # success models a candidate with torch installed; dry-run then avoids any
    # attempt to execute the training command through this stub.
    candidate.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    candidate.chmod(0o755)
    environment = dict(os.environ)
    environment.pop("PYTHON_BIN", None)
    environment.update({
        "VIRTUAL_ENV": str(virtualenv),
        "NUM_GPUS": "1",
        "NUM_ENVS": "2",
        "CUDA_VISIBLE_DEVICES": "0",
        "OUTPUT_DIR": str(tmp_path / "output"),
        "FINANCE_DRY_RUN": "1",
        "MASTER_ADDR": "127.0.0.1",
        "MASTER_PORT": "29646",
    })
    result = subprocess.run(
        ["bash", str(launcher), "--dry-run"],
        cwd=Path(__file__).resolve().parents[1], env=environment,
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    command_text = result.stdout + "\n" + result.stderr
    assert f"python={candidate}" in command_text
    assert f"{candidate} -m torch.distributed.run" in command_text
