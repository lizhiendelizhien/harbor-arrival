import csv
import copy
from dataclasses import asdict
import json

import pytest
import torch


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def synthetic_sequences(count=5, length=4):
    generator = torch.Generator().manual_seed(21)

    def period(index):
        return f"{2020 + index // 12:04d}-{index % 12 + 1:02d}"

    return [{
        "symbol": "AAA" if index < 3 else "BBB",
        "periods": [period(index * length + step) for step in range(length)],
        "target_end_period": period(index * length + length + 9),
        "current_state": torch.randn(length, 16, generator=generator),
        "future_reference": torch.randn(length, 10, 15, generator=generator) * 0.1,
        "future_mask": torch.ones(length, 10, 15, dtype=torch.bool),
    } for index in range(count)]


def fitted_actor(sequences):
    from gear_sonic.finance.model import FinancialSonicConfig
    from gear_sonic.finance.policy import make_actor_critic

    torch.manual_seed(19)
    config = FinancialSonicConfig(mlp_hidden_dims=(32, 32), d_model=32,
                                  num_heads=2, num_layers=1, ffn_dim=64)
    actor, critic = make_actor_critic(config, 410, critic_hidden_dims=(32, 32))
    actor.actor_module.model.fit_normalizers(
        torch.stack([torch.as_tensor(item["current_state"]) for item in sequences]),
        torch.stack([torch.as_tensor(item["future_reference"]) for item in sequences]),
    )
    return actor, critic


def checkpoint_fixture(tmp_path, *, schema=3):
    from gear_sonic.finance.reference import load_reference_pool

    canonical = tmp_path / "monthly.csv"
    with canonical.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "symbol", "period", "raw_valid", "qfq_valid", "qfq_open", "qfq_high",
            "qfq_low", "qfq_close", "raw_close", "raw_volume",
        ])
        writer.writeheader()
        for symbol in ("AAA", "BBB"):
            for index in range(40):
                price = 20 + index * 0.1 + (index % 4) * 0.05
                writer.writerow({
                    "symbol": symbol, "period": f"{2010 + index // 12:04d}-{index % 12 + 1:02d}",
                    "raw_valid": 1, "qfq_valid": 1, "qfq_open": price - 0.02,
                    "qfq_high": price + 0.1, "qfq_low": price - 0.1,
                    "qfq_close": price, "raw_close": price, "raw_volume": 1000 + index,
                })
    sequences, reference = load_reference_pool(
        canonical, symbols=["AAA"], start="2011-01", end="2013-04", sequence_length=4,
    )
    if schema in (1, 2):
        reference["reward_contract"] = {
            "version": 1, "name": "monthly_and_cumulative_exponential_tracking",
            "monthly_weight": 0.5, "cumulative_weight": 0.5,
            "target": "unclipped_monthly_log_return_normalized_by_checkpoint_center_scale",
            "cumulative_normalization": "sqrt_horizon",
        }
    actor, critic = fitted_actor(sequences)
    normalizer = actor.actor_module.model.future_normalizer
    payload = {
        "schema_version": schema, "iteration": 7,
        "model_config": asdict(actor.actor_module.model.config),
        "critic_config": critic.finance_config,
        "env_config": {"sequence_length": 4, "history_length": 10, "horizon": 10,
                       "return_center": float(normalizer.center[0]),
                       "return_scale": float(normalizer.scale[0])},
        "actor": actor.state_dict(), "critic": critic.state_dict(),
        "reference_config": reference,
    }
    if schema in (3, 4):
        payload["reward_contract"] = copy.deepcopy(reference["reward_contract"])
    if schema == 4:
        from gear_sonic.finance.denoising import encoder_denoising_contract
        payload["encoder_denoising_contract"] = encoder_denoising_contract()
        payload["env_config"]["encoder_denoising"] = True
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(payload, checkpoint)
    return checkpoint, payload, canonical, sequences


def test_evaluate_sequences_visits_each_complete_clip_once_with_partial_batch(monkeypatch):
    from gear_sonic.finance.evaluation import evaluate_sequences

    sequences = synthetic_sequences()
    actor, _ = fitted_actor(sequences)
    calls = []
    original = actor.act_inference

    def record(obs, *, cur_dones):
        assert not actor.training and not torch.is_grad_enabled()
        assert not cur_dones.any()
        assert "encoder_noise" not in obs and "encoder_mask_type" not in obs
        calls.append(obs["actor_obs"].shape[0])
        return original(obs, cur_dones=cur_dones)

    monkeypatch.setattr(actor, "act_inference", record)
    report = evaluate_sequences(actor, sequences, num_envs=2)
    assert calls == [2] * 8 + [1] * 4
    assert report["sequence_count"] == 5
    assert report["anchor_count"] == 20
    assert [row["sequence_id"] for row in report["sequences"]] == list(range(5))
    assert [row["anchor_start"] for row in report["sequences"]] == [s["periods"][0] for s in sequences]
    assert [row["target_end"] for row in report["sequences"]] == [s["target_end_period"] for s in sequences]
    assert [(row["symbol"], row["sequence_count"]) for row in report["symbols"]] == [("AAA", 3), ("BBB", 2)]
    assert report["encoder_input_mode"] == "clean"


@pytest.mark.parametrize("schema", [1, 2, 3, 4])
def test_checkpoint_reports_training_denoising_but_evaluates_clean(tmp_path, schema):
    from gear_sonic.finance.evaluation import evaluate_checkpoint, evaluate_sequences

    checkpoint, payload, _, sequences = checkpoint_fixture(tmp_path, schema=schema)
    report = evaluate_checkpoint(checkpoint, threads=1)
    assert report["encoder_input_mode"] == "clean"
    assert report["encoder_denoising_contract"] == payload.get("encoder_denoising_contract")
    actor, _ = fitted_actor(sequences)
    actor.load_state_dict(payload["actor"])
    clean = evaluate_sequences(actor, sequences, reward_contract=report["reward_contract"])
    assert clean["overall"] == report["overall"]


@pytest.mark.parametrize("case", ["missing", "different", "disabled", "missing_env", "nonboolean_env"])
def test_schema4_evaluation_rejects_invalid_denoising_contract(tmp_path, case):
    from gear_sonic.finance.evaluation import evaluate_checkpoint

    checkpoint, payload, _, _ = checkpoint_fixture(tmp_path, schema=4)
    if case == "missing":
        payload.pop("encoder_denoising_contract")
    elif case == "different":
        payload["encoder_denoising_contract"] = {"version": 99}
    elif case == "missing_env":
        payload["env_config"].pop("encoder_denoising")
    elif case == "nonboolean_env":
        payload["env_config"]["encoder_denoising"] = 1
    else:
        payload["env_config"]["encoder_denoising"] = False
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match="denoising"):
        evaluate_checkpoint(checkpoint)


def test_repeated_inference_is_deterministic_and_preserves_parameters_and_statistics():
    from gear_sonic.finance.evaluation import evaluate_sequences

    sequences = synthetic_sequences()
    actor, _ = fitted_actor(sequences)
    before = {key: value.clone() for key, value in actor.state_dict().items()}
    first = evaluate_sequences(actor, sequences, num_envs=2)
    second = evaluate_sequences(actor, sequences, num_envs=2)
    assert first == second
    for key, value in actor.state_dict().items():
        torch.testing.assert_close(value, before[key], atol=0, rtol=0)
    assert actor.steps == 0
    assert actor.get_rollout_cache_state() is None
    third = evaluate_sequences(actor, sequences, num_envs=3)
    for left, right in zip(first["sequences"], third["sequences"]):
        for metric in ("reward", "monthly_mse", "cumulative_mse"):
            assert left[metric] == pytest.approx(right[metric], rel=3e-5, abs=3e-6)


def test_long_clip_playback_is_independent_of_preexisting_actor_cache():
    from gear_sonic.finance.evaluation import evaluate_sequences

    sequences = synthetic_sequences(count=2, length=35)
    actor, _ = fitted_actor(sequences)
    expected = evaluate_sequences(actor, sequences, num_envs=2)
    obs = {"actor_obs": sequences[0]["current_state"][0:1],
           "future_reference": sequences[0]["future_reference"][0:1],
           "future_mask": sequences[0]["future_mask"][0:1]}
    with torch.no_grad():
        for _ in range(35):
            actor.act_inference(obs, cur_dones=torch.zeros(1, dtype=torch.bool))
    assert actor.get_rollout_cache_state() is not None
    assert evaluate_sequences(actor, sequences, num_envs=2) == expected


def test_metrics_match_environment_and_do_not_clip_targets(monkeypatch):
    from gear_sonic.finance.environment import MonthlyTrackingEnv
    from gear_sonic.finance.evaluation import evaluate_sequences

    sequences = synthetic_sequences(count=1)
    sequences[0]["future_reference"][..., 0] = 50.0
    actor, _ = fitted_actor(sequences)
    normalizer = actor.actor_module.model.future_normalizer
    normalizer.center.zero_()
    normalizer.scale.fill_(1)
    monkeypatch.setattr(actor, "act_inference", lambda obs, **kwargs: torch.zeros(len(obs["actor_obs"]), 10))
    result = evaluate_sequences(actor, sequences, num_envs=1)["overall"]
    env = MonthlyTrackingEnv(sequences, 1, return_center=0, return_scale=1)
    _, reward, _, info = env.step(torch.zeros(1, 10))
    assert result["reward"] == float(reward[0])
    assert result["monthly_mse"] == float(info["monthly_mse"][0]) == 2500.0
    assert result["cumulative_mse"] == float(info["cumulative_mse"][0])
    for name in ("reward_tracking", "reward_total", "reward_month", "reward_path",
                 "reward_change", "reward_vol", "change_mse", "volatility_mse",
                 "contribution_month", "contribution_path", "contribution_change", "contribution_vol"):
        assert result[name] == float(info[name][0])


def test_v1_reports_raw_unit_rmse_direction_and_valid_only_overlap(monkeypatch):
    from gear_sonic.finance.evaluation import evaluate_sequences

    sequences = synthetic_sequences(count=4, length=2)
    for sequence, target in zip(sequences, (-0.1, 0.0, 0.2, 0.2)):
        sequence["future_reference"][..., 0] = target
    actor, _ = fitted_actor(sequences)
    normalizer = actor.actor_module.model.future_normalizer
    normalizer.center.fill_(0.1)
    normalizer.scale.fill_(0.1)
    monkeypatch.setattr(actor, "act_inference", lambda obs, **kwargs: torch.full((len(obs["actor_obs"]), 10), -0.5))
    report = evaluate_sequences(actor, sequences, num_envs=3)
    assert report["reward_contract"]["name"] == "financial_tracking_v1"
    assert report["action_mode"] == "deterministic_mean"
    overall = report["overall"]
    assert overall["sample_count"] == 8
    assert overall["rolling_valid_count"] == 4
    assert overall["rolling_mse"] == 0.0
    for horizon in (1, 3, 6, 10):
        mse = sum((0.05 - target) ** 2 for target in (-0.1, 0.0, 0.2, 0.2)) / 4 * horizon ** 2
        assert overall[f"cumulative_rmse_{horizon}m"] == pytest.approx(mse ** 0.5)
        assert overall[f"cumulative_squared_error_{horizon}m"] == pytest.approx(mse)
        assert overall[f"direction_accuracy_{horizon}m"] == 0.5
        assert overall[f"direction_negative_count_{horizon}m"] == 2
        assert overall[f"direction_neutral_count_{horizon}m"] == 2
        assert overall[f"direction_positive_count_{horizon}m"] == 4
        naive_rmse = sum(row[f"cumulative_rmse_{horizon}m"] for row in report["sequences"]) / 4
        assert overall[f"cumulative_rmse_{horizon}m"] != pytest.approx(naive_rmse)
        aaa = report["symbols"][0]
        assert aaa[f"direction_accuracy_{horizon}m"] == pytest.approx(1 / 3)
        assert aaa[f"direction_positive_count_{horizon}m"] == 2


def test_one_anchor_evaluation_has_null_overlap_metrics():
    from gear_sonic.finance.evaluation import evaluate_sequences

    sequences = synthetic_sequences(count=2, length=1)
    actor, _ = fitted_actor(sequences)
    report = evaluate_sequences(actor, sequences)
    for row in [report["overall"], *report["sequences"], *report["symbols"]]:
        assert row["rolling_valid_count"] == 0
        assert row["rolling_mse"] is None
        assert row["rolling_penalty"] is None
        assert row["previous_overlap_mse"] is None
    json.dumps(report, allow_nan=False)


def test_evaluation_keeps_extreme_finite_diagnostics_json_safe(monkeypatch):
    from gear_sonic.finance.evaluation import evaluate_sequences

    sequences = synthetic_sequences(count=2, length=2)
    actor, _ = fitted_actor(sequences)
    normalizer = actor.actor_module.model.future_normalizer
    normalizer.center.zero_()
    normalizer.scale.fill_(1e-30)
    for sequence in sequences:
        sequence["future_reference"][..., 0] = -1e30
    monkeypatch.setattr(actor, "act_inference", lambda obs, **kwargs: torch.full((len(obs["actor_obs"]), 10), 1e30))
    report = evaluate_sequences(actor, sequences)
    assert report["overall"]["monthly_mse"] > torch.finfo(torch.float32).max
    assert report["overall"]["cumulative_squared_error_10m"] > torch.finfo(torch.float32).max
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("invalid", ["mask", "length", "periods", "nonfinite"])
def test_rejects_incomplete_or_invalid_sequences(invalid):
    from gear_sonic.finance.evaluation import evaluate_sequences

    sequences = synthetic_sequences()
    actor, _ = fitted_actor(sequences)
    if invalid == "mask":
        sequences[0]["future_mask"][0, 0, 0] = False
    elif invalid == "length":
        sequences[0]["current_state"] = sequences[0]["current_state"][:-1]
    elif invalid == "periods":
        sequences[0]["periods"][1] = "2022-01"
    else:
        sequences[0]["future_reference"][0, 0, 0] = float("nan")
    with pytest.raises(ValueError):
        evaluate_sequences(actor, sequences)


def test_checkpoint_defaults_inherit_pool_and_frozen_model(tmp_path):
    from gear_sonic.finance.evaluation import evaluate_checkpoint

    checkpoint, payload, _, sequences = checkpoint_fixture(tmp_path)
    first = evaluate_checkpoint(checkpoint, num_envs=3, threads=1)
    second = evaluate_checkpoint(checkpoint, num_envs=3, threads=1)
    assert first == second
    assert first["reference_scope"] == "training_reference_pool"
    assert first["mode"] == "privileged_reference_tracking"
    assert first["dataset_partition"] == "none"
    assert first["sequence_count"] == len(sequences)
    assert first["reference_config"] == payload["reference_config"]
    assert first["resolved_reference_config"] == payload["reference_config"]
    assert first["checkpoint_iteration"] == 7
    assert first["reward_contract"] == payload["reward_contract"]
    assert not first["legacy_reference_unverified"]


def test_checkpoint_rejects_unapproved_selector_changes_and_hash_mismatch(tmp_path):
    from gear_sonic.finance.evaluation import evaluate_checkpoint

    checkpoint, _, canonical, _ = checkpoint_fixture(tmp_path)
    with pytest.raises(ValueError, match="conflict"):
        evaluate_checkpoint(checkpoint, symbols=["BBB"])
    canonical.write_bytes(canonical.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="hash|fingerprint|source|sha256|changed"):
        evaluate_checkpoint(checkpoint)


def test_identical_relocated_canonical_file_preserves_verified_pool(tmp_path):
    from gear_sonic.finance.evaluation import evaluate_checkpoint

    checkpoint, payload, canonical, _ = checkpoint_fixture(tmp_path)
    relocated = tmp_path / "relocated.csv"
    relocated.write_bytes(canonical.read_bytes())
    result = evaluate_checkpoint(checkpoint, canonical=relocated)
    assert result["reference_scope"] == "training_reference_pool"
    assert result["reference_config"] == payload["reference_config"]
    assert result["resolved_reference_config"]["canonical"] == str(relocated)
    assert result["resolved_reference_config"]["canonical_sha256"] == payload["reference_config"]["canonical_sha256"]


def test_evaluation_never_fits_normalizers_or_creates_an_optimizer(tmp_path, monkeypatch):
    from gear_sonic.finance.evaluation import evaluate_checkpoint
    from gear_sonic.finance.model import RobustNormalizer

    checkpoint, _, _, _ = checkpoint_fixture(tmp_path)

    def forbidden(*args, **kwargs):
        raise AssertionError("Evaluation must reuse frozen checkpoint state")

    monkeypatch.setattr(RobustNormalizer, "fit", forbidden)
    monkeypatch.setattr(torch.optim, "AdamW", forbidden)
    assert evaluate_checkpoint(checkpoint)["sequence_count"] > 0


def test_nonfinite_inference_is_rejected_and_clears_rollout(monkeypatch):
    from gear_sonic.finance.evaluation import evaluate_sequences

    sequences = synthetic_sequences()
    actor, _ = fitted_actor(sequences)
    monkeypatch.setattr(actor, "act_inference", lambda obs, **kwargs: torch.full((len(obs["actor_obs"]), 10), float("nan")))
    with pytest.raises(ValueError, match="finite"):
        evaluate_sequences(actor, sequences)
    assert actor.steps == 0
    assert actor.get_rollout_cache_state() is None


def test_custom_pool_keeps_saved_contract_and_reports_both_recipes(tmp_path):
    from gear_sonic.finance.evaluation import evaluate_checkpoint

    checkpoint, payload, _, _ = checkpoint_fixture(tmp_path)
    result = evaluate_checkpoint(checkpoint, symbols=["BBB"], custom_reference=True, threads=1)
    assert result["reference_scope"] == "custom_reference_pool"
    assert result["reference_config"] == payload["reference_config"]
    assert result["resolved_reference_config"]["symbols"] == ["BBB"]
    assert [row["symbol"] for row in result["symbols"]] == ["BBB"]
    for contract in ("feature_schema", "reward_contract"):
        bad = {**payload, "reference_config": {**payload["reference_config"], contract: {"invalid": True}}}
        torch.save(bad, checkpoint)
        with pytest.raises(ValueError, match="schema|contract"):
            evaluate_checkpoint(checkpoint, custom_reference=True)


@pytest.mark.parametrize("schema", [1, 2, 3])
def test_missing_reference_requires_explicit_recipe_and_never_claims_verified_provenance(tmp_path, schema):
    from gear_sonic.finance.evaluation import evaluate_checkpoint

    checkpoint, payload, _, _ = checkpoint_fixture(tmp_path, schema=schema)
    reference = payload.pop("reference_config")
    payload["schema_version"] = schema
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match="reference-config"):
        evaluate_checkpoint(checkpoint)
    recipe = tmp_path / "run_config.json"
    recipe.write_text(json.dumps({"reference_config": reference}))
    report = evaluate_checkpoint(checkpoint, reference_config=recipe)
    assert report["reference_scope"] == "legacy_reference_unverified"
    assert report["legacy_reference_unverified"]
    custom = evaluate_checkpoint(checkpoint, reference_config=recipe, custom_reference=True, symbols=["BBB"])
    assert custom["reference_scope"] == "custom_reference_pool"
    assert custom["legacy_reference_unverified"]
    assert custom["resolved_reference_config"]["reward_contract"] == reference["reward_contract"]


@pytest.mark.parametrize("schema", [1, 2])
def test_legacy_evaluation_keeps_exact_two_term_scoring(tmp_path, monkeypatch, schema):
    from gear_sonic.finance.evaluation import evaluate_checkpoint
    from gear_sonic.trl.modules.actor_critic_modules import Actor

    checkpoint, payload, _, sequences = checkpoint_fixture(tmp_path, schema=schema)
    monkeypatch.setattr(Actor, "act_inference", lambda self, obs, **kwargs: torch.zeros(len(obs["actor_obs"]), 10))
    result = evaluate_checkpoint(checkpoint)
    center = payload["env_config"]["return_center"]
    scale = payload["env_config"]["return_scale"]
    raw = torch.stack([torch.as_tensor(sequence["future_reference"]) for sequence in sequences])[..., 0].double()
    errors = -(raw - center) / scale
    monthly_mse = errors.square().mean(-1)
    cumulative_mse = (errors.cumsum(-1) / torch.arange(1, 11).double().sqrt()).square().mean(-1)
    expected = (0.5 * ((-monthly_mse).exp() + (-cumulative_mse).exp())).float().double().mean()
    assert result["overall"]["reward"] == pytest.approx(float(expected))
    assert "reward_change" not in result["overall"]
    assert result["reward_contract"] == payload["reference_config"]["reward_contract"]


@pytest.mark.parametrize("tamper", ["missing", "weights", "legacy", "reference_legacy"])
def test_schema3_rejects_missing_or_mismatched_reward_contract(tmp_path, tamper):
    from gear_sonic.finance.evaluation import evaluate_checkpoint

    checkpoint, payload, _, _ = checkpoint_fixture(tmp_path)
    legacy_contract = {
        "version": 1, "name": "monthly_and_cumulative_exponential_tracking",
        "monthly_weight": 0.5, "cumulative_weight": 0.5,
        "target": "unclipped_monthly_log_return_normalized_by_checkpoint_center_scale",
        "cumulative_normalization": "sqrt_horizon",
    }
    if tamper == "missing":
        payload.pop("reward_contract")
    elif tamper == "weights":
        payload["reward_contract"]["monthly_weight"] = 0.5
    elif tamper == "legacy":
        payload["reward_contract"] = legacy_contract
    else:
        payload["reference_config"]["reward_contract"] = legacy_contract
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match="reward|contract"):
        evaluate_checkpoint(checkpoint)


@pytest.mark.parametrize("schema", [1, 2])
@pytest.mark.parametrize("metadata", ["checkpoint", "reference", "recipe"])
def test_legacy_schema_cannot_be_relabeled_as_v1(tmp_path, schema, metadata):
    from gear_sonic.finance.evaluation import evaluate_checkpoint

    checkpoint, payload, _, _ = checkpoint_fixture(tmp_path)
    payload["schema_version"] = schema
    arguments = {}
    if metadata != "checkpoint":
        payload.pop("reward_contract")
    if metadata == "recipe":
        recipe = tmp_path / "reference.json"
        recipe.write_text(json.dumps(payload.pop("reference_config")))
        arguments["reference_config"] = recipe
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match="legacy|reward|contract"):
        evaluate_checkpoint(checkpoint, **arguments)


def test_metadata_is_authoritative_and_legacy_provenance_survives_rebinding(tmp_path):
    from gear_sonic.finance.evaluation import evaluate_checkpoint

    checkpoint, payload, _, _ = checkpoint_fixture(tmp_path)
    with pytest.raises(ValueError, match="authoritative|override|reference-config"):
        evaluate_checkpoint(checkpoint, reference_config=tmp_path / "missing.json")
    payload["reference_provenance"] = "legacy_reference_unverified"
    torch.save(payload, checkpoint)
    report = evaluate_checkpoint(checkpoint)
    assert report["legacy_reference_unverified"]
    assert report["reference_scope"] == "legacy_reference_unverified"


@pytest.mark.parametrize("field,value", [("return_center", 9.0), ("return_scale", 2.0),
                                          ("horizon", 9), ("history_length", 9),
                                          ("sequence_length", 5)])
def test_checkpoint_environment_must_match_loaded_model_and_reference(tmp_path, field, value):
    from gear_sonic.finance.evaluation import evaluate_checkpoint

    checkpoint, payload, _, _ = checkpoint_fixture(tmp_path)
    payload["env_config"][field] = value
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match="config|normalization|horizon|dimension|sequence"):
        evaluate_checkpoint(checkpoint)


def test_evaluation_restores_thread_count_after_failure(tmp_path):
    from gear_sonic.finance.evaluation import evaluate_checkpoint

    previous = torch.get_num_threads()
    with pytest.raises(FileNotFoundError):
        evaluate_checkpoint(tmp_path / "missing.pt", threads=previous + 1)
    assert torch.get_num_threads() == previous


def test_cli_writes_summary_and_complete_csvs_exclusively(tmp_path, capsys):
    from scripts.eval_finance_sonic import main

    checkpoint, _, _, sequences = checkpoint_fixture(tmp_path)
    capsys.readouterr()
    output = tmp_path / "evaluation"
    args = ["--checkpoint", str(checkpoint), "--output-dir", str(output), "--num-envs", "3", "--threads", "1"]
    assert main(args) == 0
    summary = json.loads((output / "summary.json").read_text())
    assert json.loads(capsys.readouterr().out) == summary
    with (output / "sequences.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == len(sequences)
    assert [int(row["sequence_id"]) for row in rows] == list(range(len(sequences)))
    assert (output / "symbols.csv").is_file()
    with pytest.raises(FileExistsError):
        main(["--checkpoint", str(tmp_path / "missing.pt"), "--output-dir", str(output)])
    assert json.loads((output / "summary.json").read_text()) == summary


def test_cli_without_dump_does_not_claim_trajectory_outputs(tmp_path, capsys):
    from scripts.eval_finance_sonic import main

    checkpoint, _, _, _ = checkpoint_fixture(tmp_path)
    output = tmp_path / "ordinary-evaluation"
    output.mkdir()
    sentinel_csv = output / "trajectories.csv"
    sentinel_png = output / "trajectory_overview.png"
    sentinel_csv.write_text("keep-me\n")
    sentinel_png.write_bytes(b"keep-me")
    assert main([
        "--checkpoint", str(checkpoint), "--output-dir", str(output),
        "--threads", "1", "--num-envs", "2",
    ]) == 0
    capsys.readouterr()
    assert sentinel_csv.read_text() == "keep-me\n"
    assert sentinel_png.read_bytes() == b"keep-me"


def test_tiny_training_checkpoint_round_trip_inherits_exact_pool(tmp_path, capsys):
    from gear_sonic.finance.evaluation import evaluate_checkpoint
    from scripts.train_finance_sonic import main

    _, _, canonical, _ = checkpoint_fixture(tmp_path)
    output = tmp_path / "training"
    assert main([
        "--canonical", str(canonical), "--symbols", "AAA", "--start", "2011-01",
        "--end", "2013-04", "--output-dir", str(output), "--tiny", "--iterations", "1",
        "--sequence-length", "4", "--rollout-steps", "2", "--epochs", "1",
        "--num-minibatches", "1", "--num-envs", "2", "--threads", "1",
    ]) == 0
    checkpoint = output / "checkpoint_000001.pt"
    payload = torch.load(checkpoint, weights_only=True)
    report = evaluate_checkpoint(checkpoint, num_envs=3, threads=1)
    assert report["reference_config"] == payload["reference_config"]
    assert report["resolved_reference_config"] == payload["reference_config"]
    assert report["sequence_count"] == payload["reference_config"]["sequence_count"]
    assert report["reference_scope"] == "training_reference_pool"
    assert [row["symbol"] for row in report["symbols"]] == ["AAA"]
    capsys.readouterr()


@pytest.mark.parametrize("flag", ["--num-envs", "--threads"])
def test_cli_rejects_invalid_runtime_counts_before_loading_checkpoint(tmp_path, flag):
    from scripts.eval_finance_sonic import main

    with pytest.raises(ValueError, match="positive"):
        main(["--checkpoint", str(tmp_path / "missing.pt"), flag, "0"])


def test_privileged_rollout_emits_deterministic_monthly_trajectory_ledger(monkeypatch):
    from gear_sonic.finance.evaluation import rollout_sequences

    sequences = synthetic_sequences(count=2, length=2)
    for sequence in sequences:
        sequence["future_reference"][..., 0] = torch.arange(1, 11, dtype=torch.float32) * 0.1
        sequence["future_reference"][..., 1] = torch.arange(1, 11, dtype=torch.float32) * 0.2
    actor, _ = fitted_actor(sequences)
    model = actor.actor_module.model
    model.future_normalizer.center[0] = 0.2
    model.future_normalizer.scale[0] = 0.4

    def fixed_mean(obs, **kwargs):
        return torch.full((obs["actor_obs"].shape[0], 10), -0.5,
                          dtype=obs["actor_obs"].dtype, device=obs["actor_obs"].device)

    monkeypatch.setattr(actor, "act_inference", fixed_mean)
    first = rollout_sequences(actor, sequences, num_envs=2, max_sequences=1)
    second = rollout_sequences(actor, sequences, num_envs=2, max_sequences=1)

    assert first == second
    assert first["mode"] == "privileged_train_playback"
    assert first["latent_source"] == "oracle_future_encoder"
    assert first["encoder_input_mode"] == "clean"
    assert first["action_mode"] == "deterministic_mean"
    assert first["sequence_count"] == 1
    assert first["trajectory_count"] == 20
    assert len(first["trajectories"]) == 20
    row = first["trajectories"][0]
    assert row["sequence_id"] == 0
    assert row["symbol"] == sequences[0]["symbol"]
    assert row["anchor_index"] == 0
    assert row["anchor_period"] == "2020-01"
    assert row["target_period"] == "2020-02"
    assert row["horizon"] == 1
    assert row["cache_reset"] is True
    assert row["predicted_normalized_return"] == pytest.approx(-0.5)
    assert row["target_normalized_return"] == pytest.approx(-0.25)
    assert row["predicted_log_return"] == pytest.approx(0.0)
    assert row["target_log_return"] == pytest.approx(0.1)
    assert row["predicted_cumulative_log_return"] == pytest.approx(0.0)
    assert row["target_cumulative_log_return"] == pytest.approx(0.1)
    assert row["reference_cumulative_log_return"] == pytest.approx(0.2)
    assert actor.steps == 0
    assert actor.get_rollout_cache_state() is None


def test_playback_callback_streams_rows_without_retaining_ledger():
    from gear_sonic.finance.evaluation import evaluate_sequences

    sequences = synthetic_sequences(count=2, length=2)
    actor, _ = fitted_actor(sequences)
    streamed = []
    report = evaluate_sequences(
        actor, sequences, num_envs=2, trajectory_callback=streamed.append,
    )
    assert "trajectories" not in report
    assert report["rollout_mode"] == "privileged_train_playback"
    assert report["trajectory_count"] == len(streamed) == 2 * 2 * 10
    assert streamed[0]["sequence_id"] == 0
    assert streamed[-1]["sequence_id"] == 1
    assert actor.steps == 0
    assert actor.get_rollout_cache_state() is None


def test_rollout_wrapper_uses_callback_without_retaining_rows():
    from gear_sonic.finance.evaluation import rollout_sequences

    sequences = synthetic_sequences(count=1, length=1)
    actor, _ = fitted_actor(sequences)
    streamed = []
    report = rollout_sequences(actor, sequences, trajectory_callback=streamed.append)
    assert "trajectories" not in report
    assert report["mode"] == "privileged_train_playback"
    assert report["trajectory_count"] == len(streamed) == 10

    without_ledger = rollout_sequences(actor, sequences, include_trajectories=False)
    assert "trajectories" not in without_ledger
    assert without_ledger["trajectory_count"] == 0


def test_cli_dumps_privileged_trajectory_csv_and_limits_sequences(tmp_path, capsys):
    from scripts.eval_finance_sonic import main

    checkpoint, _, _, sequences = checkpoint_fixture(tmp_path)
    capsys.readouterr()
    output = tmp_path / "rollout"
    assert main([
        "--checkpoint", str(checkpoint), "--output-dir", str(output), "--threads", "1",
        "--num-envs", "2", "--dump-trajectories", "--max-sequences", "1",
    ]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["mode"] == "privileged_train_playback"
    assert summary["sequence_limit"] == 1
    assert summary["sequence_count"] == 1
    assert summary["reference_sequence_count"] == len(sequences)
    with (output / "trajectories.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1 * 4 * 10
    assert rows[0]["latent_source"] == "oracle_future_encoder"
    assert rows[0]["action_mode"] == "deterministic_mean"
    assert rows[0]["cache_reset"] == "True"
    with pytest.raises(FileExistsError):
        main([
            "--checkpoint", str(checkpoint), "--output-dir", str(output),
            "--dump-trajectories", "--max-sequences", "1",
        ])


@pytest.mark.parametrize("value", [0, -1])
def test_cli_rejects_invalid_max_sequences_before_loading_checkpoint(tmp_path, value):
    from scripts.eval_finance_sonic import main

    with pytest.raises(ValueError, match="positive"):
        main(["--checkpoint", str(tmp_path / "missing.pt"), "--max-sequences", str(value)])


def test_cli_requires_output_dir_for_streamed_trajectory_dump(tmp_path):
    from scripts.eval_finance_sonic import main

    with pytest.raises(ValueError, match="output-dir|stream"):
        main(["--checkpoint", str(tmp_path / "missing.pt"), "--dump-trajectories"])


def test_cli_removes_partial_trajectory_output_when_evaluation_fails(tmp_path):
    from scripts.eval_finance_sonic import main

    output = tmp_path / "failed-rollout"
    with pytest.raises(FileNotFoundError):
        main([
            "--checkpoint", str(tmp_path / "missing.pt"),
            "--output-dir", str(output), "--dump-trajectories",
        ])
    assert output.is_dir()
    assert not (output / "trajectories.csv").exists()


def test_cli_writes_png_and_limits_visualized_clips(tmp_path, capsys):
    from scripts.eval_finance_sonic import main

    checkpoint, _, _, _ = checkpoint_fixture(tmp_path)
    capsys.readouterr()
    output = tmp_path / "png-rollout"
    assert main([
        "--checkpoint", str(checkpoint), "--output-dir", str(output),
        "--dump-trajectories", "--max-sequences", "2", "--plot-sequences", "1",
        "--num-envs", "2", "--threads", "1",
    ]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["plot_sequence_limit"] == 1
    assert summary["plot_path"].endswith("trajectory_overview.png")
    assert summary["plot_sequence_count"] == 1
    assert summary["plot_sequence_total"] == 2
    assert summary["plot_visualized_sequence_count"] == 1
    assert summary["plot_anchor_count"] == 4
    assert summary["plot_anchor_total"] == 8
    assert summary["plot_visualized_anchor_count"] == 4
    assert summary["plot_heatmap_rows"] == [4, 4]
    assert summary["plot_direction_rows"] == [4, 4]
    assert (output / "trajectory_overview.png").read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    with (output / "trajectories.csv").open(newline="") as stream:
        assert len(list(csv.DictReader(stream))) == 2 * 4 * 10


@pytest.mark.parametrize("value", [0, -1])
def test_cli_rejects_invalid_plot_sequences_before_loading_checkpoint(tmp_path, value):
    from scripts.eval_finance_sonic import main

    with pytest.raises(ValueError, match="positive"):
        main([
            "--checkpoint", str(tmp_path / "missing.pt"),
            "--plot-sequences", str(value),
        ])
