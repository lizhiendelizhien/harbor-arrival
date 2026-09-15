"""Versioned reference-pool identity shared by training and playback evaluation."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

from gear_sonic.finance import observations
from gear_sonic.finance.rewards import TRACKING_REWARD_CONTRACT, validate_reward_contract


def _hash_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def feature_schema():
    return {
        "version": 1,
        "observation_code_sha256": _hash_file(observations.__file__),
        "current_fields": list(observations.CURRENT_FIELDS),
        "future_fields": list(observations.FUTURE_FIELDS),
        "units": {
            "default": "fraction_or_ratio_not_percentage_points",
            "log_fields": ["momentum_acceleration", "volatility_ratio", "volume_surprise_3m",
                           "forward_log_return", "cum_log_return", "volatility_3m_ratio",
                           "volatility_6m_ratio", "volume_surprise", "dollar_volume_surprise"],
            "liquidity_rank": "fraction_in_0_1",
        },
        "source_policy": "QFQ OHLC; raw volume and raw_close*volume proxy",
        "market_context": "full_canonical_universe_at_each_month",
        "normalization": "checkpoint_median_IQR_buffers_and_model_clip; raw_inputs_only",
        "sampling": "six_month_warmup; contiguous_anchors_and_future; nonoverlapping_anchor_blocks; drop_short_tails",
    }


def validate_reference_config(config):
    if not isinstance(config, dict) or config.get("schema_version") != 1:
        raise ValueError("Unsupported reference configuration schema")
    if config.get("dataset_partition") != "none":
        raise ValueError("Reference configuration must have dataset_partition none")
    if config.get("feature_schema") != feature_schema():
        raise ValueError("Incompatible reference feature schema or observation code version")
    try:
        validate_reward_contract(config.get("reward_contract"), allow_legacy=True)
    except ValueError as error:
        raise ValueError("Incompatible reference reward contract") from error
    required = ("canonical", "canonical_sha256", "canonical_size", "symbols", "start", "end",
                "sequence_length", "horizon", "sequence_count", "sequence_index_sha256")
    if any(key not in config for key in required):
        raise ValueError("Reference configuration is incomplete")
    if not isinstance(config["canonical"], str) or not config["canonical"]:
        raise ValueError("Reference canonical path must be nonempty")
    for key in ("canonical_sha256", "sequence_index_sha256"):
        value = config[key]
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError(f"Invalid reference {key} fingerprint")
    for key in ("canonical_size", "sequence_length", "horizon", "sequence_count"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"Reference {key} must be positive")
    if config["horizon"] != 10:
        raise ValueError("Reference horizon must be ten months")
    if config["symbols"] is not None and config["symbols"] != _symbols(config["symbols"]):
        raise ValueError("Reference symbols must be a sorted unique list")
    for key in ("start", "end"):
        if config[key] is not None:
            observations.month_number(config[key])
    if config["start"] and config["end"] and config["start"] > config["end"]:
        raise ValueError("Reference start must not exceed end")


def _symbols(symbols):
    if symbols is None:
        return None
    if isinstance(symbols, str) or not symbols or any(not isinstance(s, str) or not s.strip() for s in symbols):
        raise ValueError("Symbols must be a nonempty collection of stock identifiers")
    return sorted(set(symbols))


def read_reference_config(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    config = payload.get("reference_config", payload)
    validate_reference_config(config)
    return config


def load_reference_pool(canonical, *, symbols=None, start=None, end=None, sequence_length=64,
                        horizon=10, expected=None, reward_contract=None):
    """Build the actual training clips; fingerprint the entire market-context source."""
    if expected is not None:
        validate_reference_config(expected)
    if reward_contract is None:
        reward_contract = expected["reward_contract"] if expected is not None else TRACKING_REWARD_CONTRACT
    validate_reward_contract(reward_contract, allow_legacy=True)
    if type(sequence_length) is not int or sequence_length < 1 or horizon != 10:
        raise ValueError("Positive sequence length and ten-month horizon are required")
    symbols = _symbols(symbols)
    path = Path(canonical).resolve(strict=True)
    before = path.stat()
    digest = _hash_file(path)
    if expected is not None and (digest != expected["canonical_sha256"]
                                 or before.st_size != expected["canonical_size"]):
        raise ValueError("Canonical SHA256 fingerprint differs from the checkpoint reference")
    rows = observations.load_observations(path, None if symbols is None else set(symbols))
    sequences = list(observations.iter_sequences(rows, sequence_length=sequence_length, horizon=horizon,
                                                start_period=start, end_period=end))
    after = path.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
        raise ValueError("Canonical source changed while building reference pool")
    if _hash_file(path) != digest:
        raise ValueError("Canonical source changed while building reference pool")
    if not sequences:
        raise ValueError("No complete reference sequences remain in the requested pool")
    index_digest = hashlib.sha256()
    for row in sequences:
        identity = [row["symbol"], row["periods"], row["target_end_period"]]
        index_digest.update((json.dumps(identity, separators=(",", ":")) + "\n").encode("utf-8"))
    config = {
        "schema_version": 1, "dataset_partition": "none", "canonical": str(path),
        "canonical_sha256": digest, "canonical_size": before.st_size,
        "symbols": symbols, "start": start, "end": end,
        "sequence_length": sequence_length, "horizon": horizon,
        "feature_schema": feature_schema(), "reward_contract": deepcopy(reward_contract),
        "sequence_count": len(sequences), "sequence_index_sha256": index_digest.hexdigest(),
    }
    if expected is not None and reference_identity(config) != reference_identity(expected):
        raise ValueError("Rebuilt reference pool identity differs from the checkpoint")
    return sequences, config


def reference_identity(config):
    """An identical source may be relocated without changing its reference identity."""
    return {key: value for key, value in config.items() if key != "canonical"}


def resolve_reference_pool(reference_config, *, canonical=None, symbols=None, start=None,
                           end=None, custom_reference=False):
    validate_reference_config(reference_config)
    selectors = {"symbols": _symbols(symbols), "start": start, "end": end}
    for name, value in selectors.items():
        if value is None:
            selectors[name] = reference_config[name]
        elif value != reference_config[name] and not custom_reference:
            raise ValueError(f"Explicit {name} conflicts with saved reference; use custom-reference for evaluation")
    return load_reference_pool(
        canonical if canonical is not None else reference_config["canonical"], **selectors,
        sequence_length=reference_config["sequence_length"], horizon=reference_config["horizon"],
        expected=None if custom_reference else reference_config,
        reward_contract=reference_config["reward_contract"],
    )
