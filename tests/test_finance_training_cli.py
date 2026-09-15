import csv
import json

import pytest
import torch


def canonical_file(tmp_path):
    path = tmp_path / "monthly.csv"
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "symbol", "period", "raw_valid", "qfq_valid", "qfq_open", "qfq_high",
            "qfq_low", "qfq_close", "raw_close", "raw_volume",
        ])
        writer.writeheader()
        for symbol in ("AAA", "BBB"):
            for index in range(60):
                price = 20 + index * 0.1 + (index % 4) * 0.05
                writer.writerow({
                    "symbol": symbol, "period": f"{2010 + index // 12:04d}-{index % 12 + 1:02d}",
                    "raw_valid": 1, "qfq_valid": 1, "qfq_open": price - 0.02,
                    "qfq_high": price + 0.1, "qfq_low": price - 0.1,
                    "qfq_close": price, "raw_close": price, "raw_volume": 1000 + index,
                })
    return path


def test_tiny_cli_trains_saves_and_restarts_without_world_heads(tmp_path, capsys):
    from scripts.train_finance_sonic import main

    canonical = canonical_file(tmp_path)
    output_dir = tmp_path / "run"
    common = ["--canonical", str(canonical), "--symbols", "AAA", "BBB",
              "--output-dir", str(output_dir), "--num-envs", "4", "--threads", "1"]
    assert main(common + ["--tiny", "--iterations", "2", "--sequence-length", "8",
                          "--rollout-steps", "6", "--epochs", "1", "--num-minibatches", "2"]) == 0
    checkpoint = output_dir / "checkpoint_000002.pt"
    assert checkpoint.is_file()
    first = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert first["schema_version"] == 4
    assert first["reward_contract"]["name"] == "financial_tracking_v1"
    assert first["iteration"] == 2
    assert not any("s_pred" in key or "z_gmm" in key for key in first["actor"])
    assert first["model_config"]["window_size"] == 32
    assert main(common + ["--resume", str(checkpoint), "--iterations", "1"]) == 0
    second = torch.load(output_dir / "checkpoint_000003.pt", map_location="cpu", weights_only=True)
    assert second["iteration"] == 3
    assert second["config"] == first["config"]
    assert second["model_config"] == first["model_config"]
    assert second["critic_config"] == first["critic_config"]
    logs = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert any(row.get("event") == "iteration" for row in logs)
    starts = [row for row in logs if row.get("event") == "start"]
    assert len(starts) == 2 and all(row["dataset_partition"] == "none" for row in starts)
    assert starts[0]["excluded_heads"] == ["s_pred", "z_gmm"]
    assert starts[1]["resume_mode"] == "restart_reference_episodes"
    assert starts[0]["mode"] == "privileged_reference_tracking"
    assert starts[0]["reward_contract"] == first["reward_contract"]
    assert starts[0]["encoder_denoising_contract"] == first["encoder_denoising_contract"]
    assert starts[1]["encoder_denoising_contract"] == first["encoder_denoising_contract"]
    assert second["encoder_denoising_contract"] == first["encoder_denoising_contract"]
    assert json.loads((output_dir / "run_config.json").read_text())["encoder_denoising_contract"] == first["encoder_denoising_contract"]
    assert json.loads((output_dir / "run_config.json").read_text())["reward_contract"] == first["reward_contract"]
    assert all("reward_change" in row for row in logs if row["event"] == "iteration")
    with pytest.raises(FileExistsError):
        main(common + ["--resume", str(checkpoint), "--iterations", "1"])
    for flag, value in (("--sequence-length", "9"), ("--rollout-steps", "7"),
                        ("--epochs", "2"), ("--num-minibatches", "1")):
        with pytest.raises(ValueError, match="conflict"):
            main(common + ["--resume", str(checkpoint), "--iterations", "1", flag, value])
    changed_window = tmp_path / "window16.pt"
    torch.save({**first, "model_config": {**first["model_config"], "window_size": 16}}, changed_window)
    with pytest.raises(ValueError, match="conflict"):
        main(common + ["--resume", str(changed_window), "--iterations", "1", "--tiny",
                       "--output-dir", str(tmp_path / "new-output"),
                       "--canonical", str(tmp_path / "missing.csv")])


def test_cli_requires_explicit_iteration_budget_and_rejects_invalid_counts(tmp_path):
    from scripts.train_finance_sonic import main

    with pytest.raises(SystemExit):
        main(["--canonical", str(tmp_path / "missing.csv"), "--output-dir", str(tmp_path)])
    with pytest.raises(ValueError, match="iterations"):
        main(["--canonical", str(tmp_path / "missing.csv"), "--output-dir", str(tmp_path),
              "--iterations", "0"])


@pytest.mark.parametrize("flag", [
    "--num-envs", "--threads", "--sequence-length", "--rollout-steps", "--epochs",
    "--num-minibatches", "--save-interval",
])
def test_cli_rejects_invalid_counts_before_file_access(tmp_path, flag):
    from scripts.train_finance_sonic import main

    with pytest.raises(ValueError):
        main(["--canonical", str(tmp_path / "missing.csv"), "--output-dir", str(tmp_path),
              "--iterations", "1", flag, "0"])


def test_cli_preflights_checkpoint_paths_before_loading_data(tmp_path):
    from scripts.train_finance_sonic import main

    existing = tmp_path / "checkpoint_000002.pt"
    existing.write_bytes(b"existing checkpoint")
    with pytest.raises(FileExistsError):
        main(["--canonical", str(tmp_path / "missing.csv"), "--output-dir", str(tmp_path),
              "--iterations", "3", "--save-interval", "2"])
    assert existing.read_bytes() == b"existing checkpoint"
    assert not (tmp_path / "checkpoint_000001.pt").exists()


def test_cli_restores_torch_threads_after_loading_error(tmp_path):
    from scripts.train_finance_sonic import main

    previous = torch.get_num_threads()
    with pytest.raises(FileNotFoundError):
        main(["--canonical", str(tmp_path / "missing.csv"), "--output-dir", str(tmp_path),
              "--iterations", "1", "--threads", str(previous + 1)])
    assert torch.get_num_threads() == previous


def test_cli_normalization_and_resume_are_mutually_exclusive(tmp_path):
    from scripts.train_finance_sonic import main

    with pytest.raises(SystemExit):
        main(["--canonical", str(tmp_path / "missing.csv"), "--output-dir", str(tmp_path),
              "--iterations", "1", "--normalization", "normalization.json", "--resume", "run.pt"])


def test_cli_loads_normalization_uses_all_symbols_and_saves_final_between_intervals(tmp_path, capsys):
    from gear_sonic.finance.observations import CURRENT_FIELDS, FUTURE_FIELDS
    from scripts.train_finance_sonic import main

    canonical = canonical_file(tmp_path)
    normalization = tmp_path / "normalization.json"
    normalization.write_text(json.dumps({
        "schema_version": 1, "horizon": 10,
        "current": {"fields": list(CURRENT_FIELDS), "center": [0.0] * 16,
                    "scale": [1.0] * 16, "clip": 10.0},
        "future": {"fields": list(FUTURE_FIELDS), "center": [0.25] + [0.0] * 14,
                   "scale": [0.5] + [1.0] * 14, "clip": 10.0},
    }))
    previous_threads = torch.get_num_threads()
    output = tmp_path / "bounded"
    assert main([
        "--canonical", str(canonical), "--output-dir", str(output), "--tiny",
        "--iterations", "2", "--sequence-length", "4", "--rollout-steps", "2",
        "--epochs", "1", "--num-minibatches", "1", "--num-envs", "2",
        "--normalization", str(normalization), "--start", "2011-01", "--end", "2013-12",
        "--threads", str(previous_threads + 1), "--save-interval", "3",
    ]) == 0
    assert torch.get_num_threads() == previous_threads
    assert not (output / "checkpoint_000001.pt").exists()
    checkpoint = torch.load(output / "checkpoint_000002.pt", map_location="cpu", weights_only=True)
    assert checkpoint["env_config"]["return_center"] == 0.25
    assert checkpoint["env_config"]["return_scale"] == 0.5
    logs = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert logs[0]["symbols"] is None
    assert logs[0]["period_bounds"] == ["2011-01", "2013-12"]
    assert logs[0]["sequence_count"] == 12
    assert logs[-1]["iteration"] == 2


def test_checkpoint_inherits_reference_without_repeating_data_arguments(tmp_path, capsys):
    from scripts.train_finance_sonic import main

    canonical = canonical_file(tmp_path)
    output = tmp_path / "bound-run"
    assert main([
        "--canonical", str(canonical), "--symbols", "AAA", "--start", "2011-01",
        "--end", "2013-12", "--output-dir", str(output), "--tiny", "--iterations", "1",
        "--sequence-length", "4", "--rollout-steps", "2", "--epochs", "1",
        "--num-minibatches", "1", "--num-envs", "2",
    ]) == 0
    first = torch.load(output / "checkpoint_000001.pt", weights_only=True)
    assert first["schema_version"] == 4
    assert first["reference_config"]["symbols"] == ["AAA"]
    assert first["reference_config"]["start"] == "2011-01"
    assert first["reference_config"]["dataset_partition"] == "none"
    recipe = json.loads((output / "run_config.json").read_text())
    assert recipe["reference_config"] == first["reference_config"]
    assert main(["--resume", str(output / "checkpoint_000001.pt"), "--output-dir", str(output),
                 "--iterations", "1", "--num-envs", "2"]) == 0
    second = torch.load(output / "checkpoint_000002.pt", weights_only=True)
    assert second["reference_config"] == first["reference_config"]
    for name, value in first["actor"].items():
        if "normalizer" in name:
            torch.testing.assert_close(value, second["actor"][name], atol=0, rtol=0)
    capsys.readouterr()
    with pytest.raises(ValueError, match="conflict"):
        main(["--resume", str(output / "checkpoint_000002.pt"), "--output-dir", str(tmp_path / "bad"),
              "--iterations", "1", "--symbols", "BBB", "--num-envs", "2"])
    assert not (tmp_path / "bad").exists()
    moved = tmp_path / "relocated.csv"
    moved.write_bytes(canonical.read_bytes())
    recipe_before = (output / "run_config.json").read_bytes()
    main(["--resume", str(output / "checkpoint_000002.pt"), "--canonical", str(moved),
          "--output-dir", str(output), "--iterations", "1", "--num-envs", "2"])
    relocated = torch.load(output / "checkpoint_000003.pt", weights_only=True)
    assert relocated["reference_config"]["canonical"] == str(moved.resolve())
    assert (output / "run_config.json").read_bytes() == recipe_before


def test_relabeled_modern_checkpoint_cannot_bypass_source_protocol_checks(tmp_path, capsys):
    from scripts.train_finance_sonic import main

    canonical = canonical_file(tmp_path)
    output = tmp_path / "source-run"
    main(["--canonical", str(canonical), "--output-dir", str(output), "--tiny", "--iterations", "1",
          "--sequence-length", "4", "--rollout-steps", "2", "--epochs", "1",
          "--num-minibatches", "1", "--num-envs", "2"])
    saved = torch.load(output / "checkpoint_000001.pt", weights_only=True)
    saved.pop("reference_config", None)
    saved["schema_version"] = 1
    legacy = tmp_path / "legacy.pt"
    torch.save(saved, legacy)
    arguments = ["--resume", str(legacy), "--output-dir", str(tmp_path / "legacy-run"),
                 "--iterations", "1", "--num-envs", "2"]
    with pytest.raises(ValueError, match="reward|schema"):
        main(arguments)
    with pytest.raises(ValueError, match="reward|schema"):
        main(arguments + ["--reference-config", str(output / "run_config.json")])
    assert not (tmp_path / "legacy-run").exists()
    capsys.readouterr()


@pytest.mark.parametrize("case", ["schema3", "missing", "different", "disabled", "nonboolean_env"])
def test_cli_rejects_denoising_protocol_change_before_data_or_state_load(tmp_path, monkeypatch, case):
    from scripts.train_finance_sonic import main
    from test_finance_ppo import make_trainer
    from gear_sonic.trl.modules.actor_critic_modules import Actor

    source = tmp_path / "source.pt"
    make_trainer().save_checkpoint(source)
    payload = torch.load(source, weights_only=True)
    if case == "schema3":
        payload["schema_version"] = 3
    elif case == "missing":
        payload.pop("encoder_denoising_contract", None)
    elif case == "different":
        payload["encoder_denoising_contract"] = {"version": 99}
    else:
        payload["env_config"]["encoder_denoising"] = 1 if case == "nonboolean_env" else False
    bad = tmp_path / "bad.pt"
    torch.save(payload, bad)

    def reject_state_load(*args, **kwargs):
        pytest.fail("A mismatched training protocol must fail before state loading")

    monkeypatch.setattr(Actor, "load_state_dict", reject_state_load)
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="schema|denoising"):
        main(["--resume", str(bad), "--output-dir", str(output), "--iterations", "1"])
    assert not output.exists()


@pytest.mark.parametrize("schema", [2, 3])
def test_cli_migrates_legacy_same_directory_and_keeps_original_recipe(tmp_path, capsys, schema):
    from scripts.train_finance_sonic import main
    from test_finance_ppo import legacy_checkpoint_payload
    from gear_sonic.finance.evaluation import evaluate_checkpoint

    canonical = canonical_file(tmp_path)
    output = tmp_path / "source"
    common = ["--output-dir", str(output), "--num-envs", "2", "--threads", "1"]
    main(common + ["--canonical", str(canonical), "--tiny", "--iterations", "1",
                   "--sequence-length", "4", "--rollout-steps", "2", "--epochs", "1",
                   "--num-minibatches", "1"])
    source = torch.load(output / "checkpoint_000001.pt", weights_only=True)
    old = legacy_checkpoint_payload(source, schema)
    legacy = output / "legacy.pt"
    torch.save(old, legacy)
    recipe_path = output / "run_config.json"
    old_recipe = json.loads(recipe_path.read_text())
    old_recipe.pop("encoder_denoising_contract")
    old_recipe["env_config"].pop("encoder_denoising")
    old_recipe["reference_config"] = old["reference_config"]
    if schema == 2:
        old_recipe.pop("reward_contract")
    recipe_path.write_text(json.dumps(old_recipe))
    original_recipe = recipe_path.read_bytes()
    original_checkpoint = legacy.read_bytes()
    capsys.readouterr()
    with pytest.warns(UserWarning, match=f"schema {schema}"):
        assert main(common + ["--resume", str(legacy), "--iterations", "1"]) == 0
    migrated = torch.load(output / "checkpoint_000002.pt", weights_only=True)
    assert migrated["schema_version"] == 4 and migrated["iteration"] == 2
    assert migrated["training_migrations"][0]["source_schema"] == schema
    assert migrated["reference_config"]["reward_contract"] == migrated["reward_contract"]
    assert migrated["reference_config"]["canonical_sha256"] == old["reference_config"]["canonical_sha256"]
    active_recipe = output / "run_config.schema4.json"
    assert json.loads(active_recipe.read_text())["encoder_denoising_contract"] == migrated["encoder_denoising_contract"]
    logs = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert logs[0]["training_migrations"] == migrated["training_migrations"]
    assert logs[0]["run_config"] == str(active_recipe)
    assert main(common + ["--resume", str(output / "checkpoint_000002.pt"), "--iterations", "1"]) == 0
    again = torch.load(output / "checkpoint_000003.pt", weights_only=True)
    assert again["iteration"] == 3
    assert again["training_migrations"] == migrated["training_migrations"]
    assert recipe_path.read_bytes() == original_recipe
    assert legacy.read_bytes() == original_checkpoint
    report = evaluate_checkpoint(output / "checkpoint_000003.pt", num_envs=2, threads=1)
    assert report["reward_contract"] == migrated["reward_contract"]
    assert report["training_migrations"] == migrated["training_migrations"]


def test_schema1_cli_resume_needs_original_reference_recipe_and_restores_provenance(tmp_path, capsys):
    from scripts.train_finance_sonic import main
    from test_finance_ppo import legacy_checkpoint_payload
    from gear_sonic.finance.rewards import LEGACY_TRACKING_REWARD_CONTRACT

    canonical = canonical_file(tmp_path)
    source_dir = tmp_path / "source"
    main(["--canonical", str(canonical), "--output-dir", str(source_dir), "--tiny", "--iterations", "1",
          "--sequence-length", "4", "--rollout-steps", "2", "--epochs", "1",
          "--num-minibatches", "1", "--num-envs", "2", "--threads", "1"])
    source = torch.load(source_dir / "checkpoint_000001.pt", weights_only=True)
    legacy = tmp_path / "schema1.pt"
    torch.save(legacy_checkpoint_payload(source, 1), legacy)
    output = tmp_path / "resumed"
    args = ["--resume", str(legacy), "--output-dir", str(output), "--iterations", "1",
            "--num-envs", "2", "--threads", "1"]
    with pytest.raises(ValueError, match="reference-config"):
        main(args)
    assert not output.exists()
    reference = source["reference_config"]
    reference["reward_contract"] = LEGACY_TRACKING_REWARD_CONTRACT
    recipe = tmp_path / "original-reference.json"
    recipe.write_text(json.dumps(reference))
    with pytest.warns(UserWarning, match="schema 1"):
        assert main(args + ["--reference-config", str(recipe)]) == 0
    migrated = torch.load(output / "checkpoint_000002.pt", weights_only=True)
    assert migrated["iteration"] == 2
    assert migrated["reference_provenance"] == "legacy_reference_unverified"
    assert migrated["reference_config"]["reward_contract"] == migrated["reward_contract"]
    assert migrated["training_migrations"][0]["source_schema"] == 1
    capsys.readouterr()
