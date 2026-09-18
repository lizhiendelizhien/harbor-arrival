"""Versioned reference-pool identity shared by training and playback evaluation."""

from __future__ import annotations

from copy import deepcopy
import csv
import hashlib
import json
import math
from pathlib import Path
import pickle
import statistics

from gear_sonic.finance import observations
from gear_sonic.finance.rewards import TRACKING_REWARD_CONTRACT, validate_reward_contract
from gear_sonic.finance.trajectory import TRAJECTORY_KIND, TRAJECTORY_SCHEMA_VERSION, load_trajectory


_ARCHIVE_PATH_KEYS = {
    "trajectory_root", "trajectory_index", "trajectory_metadata", "trajectory_manifest",
}


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
    source_type = config.get("source_type", "canonical")
    if source_type not in ("canonical", "trajectory_archive"):
        raise ValueError("Unsupported reference source type")
    required = ("canonical", "canonical_sha256", "canonical_size", "symbols", "start", "end",
                "sequence_length", "horizon", "sequence_count", "sequence_index_sha256")
    if source_type == "trajectory_archive":
        required += (
            "trajectory_root", "trajectory_index", "trajectory_index_sha256",
            "trajectory_metadata", "trajectory_metadata_sha256",
            "trajectory_manifest", "trajectory_manifest_sha256", "trajectory_file_count",
        )
    if any(key not in config for key in required):
        raise ValueError("Reference configuration is incomplete")
    if not isinstance(config["canonical"], str) or not config["canonical"]:
        raise ValueError("Reference canonical path must be nonempty")
    fingerprint_keys = ["canonical_sha256", "sequence_index_sha256"]
    if source_type == "trajectory_archive":
        fingerprint_keys.extend((
            "trajectory_index_sha256", "trajectory_metadata_sha256", "trajectory_manifest_sha256",
        ))
        for key in _ARCHIVE_PATH_KEYS:
            if not isinstance(config[key], str) or not config[key]:
                raise ValueError(f"Reference {key} path must be nonempty")
    for key in fingerprint_keys:
        value = config[key]
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError(f"Invalid reference {key} fingerprint")
    positive_keys = ["canonical_size", "sequence_length", "horizon", "sequence_count"]
    if source_type == "trajectory_archive":
        positive_keys.append("trajectory_file_count")
    for key in positive_keys:
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


def _archive_paths(trajectory_root, index_path, metadata_path, manifest_path):
    root = Path(trajectory_root).resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(root)
    paths = {
        "index": Path(index_path) if index_path is not None else root / "trajectory_index.csv",
        "metadata": Path(metadata_path) if metadata_path is not None else root / "metadata.pkl",
        "manifest": Path(manifest_path) if manifest_path is not None else root / "manifest.json",
    }
    return root, {name: path.resolve(strict=True) for name, path in paths.items()}


def _read_archive_catalog(root, paths):
    try:
        manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError("Trajectory archive manifest is not valid JSON") from error
    if (not isinstance(manifest, dict) or manifest.get("status") != "complete"
            or manifest.get("schema_version") != TRAJECTORY_SCHEMA_VERSION
            or manifest.get("schema") != TRAJECTORY_KIND):
        raise ValueError("Trajectory archive manifest has an unsupported schema")

    try:
        with paths["metadata"].open("rb") as handle:
            metadata = pickle.load(handle)
    except Exception as error:
        raise ValueError("Trajectory archive metadata is unreadable") from error
    if not isinstance(metadata, dict) or not metadata:
        raise ValueError("Trajectory archive metadata must be a nonempty mapping")

    with paths["index"].open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "key", "path", "symbol", "segment_id", "start_period", "end_period", "length",
            "complete_anchor_count", "eligible_anchor_count",
        }
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError("Trajectory archive index is missing required columns")
        rows = list(reader)
    if not rows or len(rows) != len(metadata):
        raise ValueError("Trajectory archive index and metadata counts differ")

    seen_keys = set()
    expected_files = set()
    for row in rows:
        key = row["key"]
        if not key or key in seen_keys:
            raise ValueError("Trajectory archive index contains an empty or duplicate key")
        seen_keys.add(key)
        entry = metadata.get(key)
        if not isinstance(entry, dict):
            raise ValueError(f"Trajectory archive metadata is missing index key {key!r}")
        try:
            indexed = {
                "symbol": row["symbol"], "segment_id": int(row["segment_id"]),
                "start_period": row["start_period"], "end_period": row["end_period"],
                "length": int(row["length"]),
                "complete_anchor_count": int(row["complete_anchor_count"]),
                "eligible_anchor_count": int(row["eligible_anchor_count"]),
            }
        except (TypeError, ValueError) as error:
            raise ValueError(f"Trajectory archive index has invalid numeric metadata for {key!r}") from error
        if any(entry.get(name) != value for name, value in indexed.items()):
            raise ValueError(f"Trajectory archive index and metadata differ for {key!r}")
        if (entry.get("schema_version") != TRAJECTORY_SCHEMA_VERSION
                or entry.get("kind") != TRAJECTORY_KIND):
            raise ValueError(f"Trajectory archive metadata schema is invalid for {key!r}")
        relative = Path(row["path"])
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise ValueError(f"Trajectory archive index path escapes its root: {relative}") from error
        if not path.is_file():
            raise FileNotFoundError(f"Trajectory archive index file is missing: {path}")
        source_relpath = entry.get("source_relpath")
        if source_relpath is not None and Path(source_relpath) != relative:
            raise ValueError(f"Trajectory archive metadata path differs for {key!r}")
        row["_resolved_path"] = path
        expected_files.add(path)

    actual_files = {path.resolve() for path in (root / "trajectories").glob("*.pkl")}
    if actual_files != expected_files:
        raise ValueError("Trajectory archive index, metadata, and PKL file set differ")
    counts = manifest.get("counts", {})
    if counts.get("segment_count") != len(rows):
        raise ValueError("Trajectory archive manifest segment count differs from its index")
    return manifest, metadata, rows


def _trajectory_bars(row, metadata):
    key, entry = load_trajectory(row["_resolved_path"])
    if key != row["key"]:
        raise ValueError("Trajectory archive PKL key differs from its index")
    expected = metadata[key]
    for name in ("symbol", "segment_id", "length", "start_period", "end_period"):
        if entry.get(name) != expected.get(name):
            raise ValueError(f"Trajectory archive PKL metadata differs for {key!r}")
    raw, qfq = entry["raw"], entry["qfq"]
    return [
        observations.MonthlyBar(
            entry["symbol"], str(period), float(adjusted[0]), float(adjusted[1]),
            float(adjusted[2]), float(adjusted[3]), float(unadjusted[1]), float(unadjusted[4]),
        )
        for period, unadjusted, adjusted in zip(entry["periods"], raw, qfq)
    ]


def _sequence_digest(sequences):
    digest = hashlib.sha256()
    for row in sequences:
        identity = [row["symbol"], row["periods"], row["target_end_period"]]
        digest.update((json.dumps(identity, separators=(",", ":")) + "\n").encode("utf-8"))
    return digest.hexdigest()


def _archive_market_context(rows, metadata):
    """Build the same month-level context as ``build_market_context`` in-memory.

    Returns are reset at every archive segment boundary, while liquidity ranks
    still use every valid symbol present in a month.  This makes the archive a
    self-contained source when the canonical CSV recorded by the manifest is
    unavailable.
    """
    returns = {}
    liquidity = {}
    for row in rows:
        bars = _trajectory_bars(row, metadata)
        previous = None
        for bar in bars:
            liquidity.setdefault(bar.period, []).append((bar.symbol, bar.dollar_volume))
            if previous is not None and observations.month_number(bar.period) - observations.month_number(previous.period) == 1:
                returns.setdefault(bar.period, []).append(bar.close / previous.close - 1)
            previous = bar
    median_returns = {
        observations.month_number(period): statistics.median(values)
        for period, values in returns.items()
    }
    result = {}
    for period, values in liquidity.items():
        ordered = sorted(values, key=lambda item: item[1])
        ranks = {}
        index = 0
        while index < len(ordered):
            end = index + 1
            while end < len(ordered) and ordered[end][1] == ordered[index][1]:
                end += 1
            rank = ((index + end - 1) / 2) / (len(ordered) - 1) if len(ordered) > 1 else 0.5
            for symbol, _ in ordered[index:end]:
                ranks[symbol] = rank
            index = end
        number = observations.month_number(period)
        recent = [median_returns.get(number - lag) for lag in range(6)]
        momentum = math.prod(1 + value for value in recent) - 1 if all(v is not None for v in recent) else None
        result[period] = observations.MarketMonth(median_returns.get(number), momentum, ranks)
    return result


def load_reference_pool_from_trajectories(
    trajectory_root, *, index_path=None, metadata_path=None, manifest_path=None,
    context_canonical=None, symbols=None, start=None, end=None, sequence_length=64,
    horizon=10, expected=None, reward_contract=None, progress_callback=None,
):
    """Build fixed Sonic clips from the validated variable-length trajectory archive."""
    if expected is not None:
        validate_reference_config(expected)
        if expected.get("source_type", "canonical") != "trajectory_archive":
            raise ValueError("Expected reference configuration is not a trajectory archive")
    if type(sequence_length) is not int or sequence_length < 1 or horizon != 10:
        raise ValueError("Positive sequence length and ten-month horizon are required")
    if reward_contract is None:
        reward_contract = expected["reward_contract"] if expected is not None else TRACKING_REWARD_CONTRACT
    validate_reward_contract(reward_contract, allow_legacy=True)
    symbols = _symbols(symbols)
    root, paths = _archive_paths(trajectory_root, index_path, metadata_path, manifest_path)
    before = {name: (path.stat(), _hash_file(path)) for name, path in paths.items()}
    manifest, metadata, rows = _read_archive_catalog(root, paths)
    selected = set(symbols) if symbols is not None else None
    candidate_rows = [row for row in rows if selected is None or row["symbol"] in selected]
    if progress_callback is not None:
        progress_callback(0, len(candidate_rows))

    source = manifest.get("source")
    if not isinstance(source, dict):
        raise ValueError("Trajectory archive manifest is missing its canonical source")
    manifest_canonical = Path(str(source.get("path", ""))) if source.get("path") else None
    # The archive is the requested primary data source.  A canonical CSV is
    # only used when explicitly supplied as a context override; otherwise all
    # market statistics are derived from the companion PKLs themselves.
    canonical = Path(context_canonical) if context_canonical is not None else None
    context_source = "explicit_canonical" if canonical is not None else "trajectory_archive"
    canonical_stat = None
    canonical_digest = source.get("sha256")
    if canonical is not None:
        try:
            canonical = canonical.resolve(strict=True)
        except (FileNotFoundError, RuntimeError):
            raise FileNotFoundError(
                "Explicit --canonical context source for the trajectory archive does not exist"
            )
        if canonical is not None:
            canonical_stat = canonical.stat()
            canonical_digest = _hash_file(canonical)
            if (canonical_digest != source.get("sha256")
                    or canonical_stat.st_size != source.get("size_bytes")):
                raise ValueError("Trajectory archive canonical source fingerprint differs from its manifest")
    if canonical is not None:
        market = observations.build_market_context(observations.read_canonical(canonical))
    else:
        context_source = "trajectory_archive"
        market = _archive_market_context(rows, metadata)

    sequences = []
    for processed, row in enumerate(candidate_rows, 1):
        if int(row["eligible_anchor_count"]) >= sequence_length:
            bars = _trajectory_bars(row, metadata)
            segment_observations = observations.build_symbol_observations(bars, market)
            sequences.extend(observations.iter_sequences(
                {row["symbol"]: segment_observations}, sequence_length=sequence_length, horizon=horizon,
                start_period=start, end_period=end,
            ))
        if progress_callback is not None and (processed % 1000 == 0 or processed == len(candidate_rows)):
            progress_callback(processed, len(candidate_rows))
    if not sequences:
        raise ValueError("No complete reference sequences remain in the requested trajectory archive")

    for name, (stat, digest) in before.items():
        path = paths[name]
        current = path.stat()
        if ((stat.st_ino, stat.st_size, stat.st_mtime_ns)
                != (current.st_ino, current.st_size, current.st_mtime_ns)
                or _hash_file(path) != digest):
            raise ValueError("Trajectory archive companion files changed while loading")
    config = {
        "schema_version": 1, "dataset_partition": "none", "source_type": "trajectory_archive",
        "context_source": context_source,
        "canonical": str(canonical if canonical is not None else manifest_canonical or "archive"),
        "canonical_sha256": canonical_digest,
        "canonical_size": (canonical_stat.st_size if canonical_stat is not None
                            else int(source.get("size_bytes", 1))),
        "trajectory_root": str(root),
        "trajectory_index": str(paths["index"]),
        "trajectory_index_sha256": before["index"][1],
        "trajectory_metadata": str(paths["metadata"]),
        "trajectory_metadata_sha256": before["metadata"][1],
        "trajectory_manifest": str(paths["manifest"]),
        "trajectory_manifest_sha256": before["manifest"][1],
        "trajectory_file_count": len(rows),
        "symbols": symbols, "start": start, "end": end,
        "sequence_length": sequence_length, "horizon": horizon,
        "feature_schema": feature_schema(), "reward_contract": deepcopy(reward_contract),
        "sequence_count": len(sequences), "sequence_index_sha256": _sequence_digest(sequences),
    }
    validate_reference_config(config)
    if expected is not None and reference_identity(config) != reference_identity(expected):
        raise ValueError("Rebuilt trajectory reference pool identity differs from the checkpoint")
    return sequences, config


def reference_identity(config):
    """An identical source may be relocated without changing its reference identity."""
    return {
        key: value for key, value in config.items()
        if key != "canonical" and key not in _ARCHIVE_PATH_KEYS
    }


def resolve_reference_pool(reference_config, *, canonical=None, symbols=None, start=None,
                           end=None, custom_reference=False, trajectory_root=None,
                           trajectory_index=None, trajectory_metadata=None,
                           trajectory_manifest=None, progress_callback=None):
    validate_reference_config(reference_config)
    selectors = {"symbols": _symbols(symbols), "start": start, "end": end}
    for name, value in selectors.items():
        if value is None:
            selectors[name] = reference_config[name]
        elif value != reference_config[name] and not custom_reference:
            raise ValueError(f"Explicit {name} conflicts with saved reference; use custom-reference for evaluation")
    expected = None if custom_reference else reference_config
    if reference_config.get("source_type", "canonical") == "trajectory_archive":
        context_canonical = canonical
        if context_canonical is None and reference_config.get("context_source") == "explicit_canonical":
            context_canonical = reference_config["canonical"]
        return load_reference_pool_from_trajectories(
            trajectory_root if trajectory_root is not None else reference_config["trajectory_root"],
            index_path=(trajectory_index if trajectory_index is not None
                        else reference_config["trajectory_index"]),
            metadata_path=(trajectory_metadata if trajectory_metadata is not None
                           else reference_config["trajectory_metadata"]),
            manifest_path=(trajectory_manifest if trajectory_manifest is not None
                           else reference_config["trajectory_manifest"]),
            context_canonical=context_canonical,
            **selectors, sequence_length=reference_config["sequence_length"],
            horizon=reference_config["horizon"], expected=expected,
            reward_contract=reference_config["reward_contract"], progress_callback=progress_callback,
        )
    if any(value is not None for value in (
        trajectory_root, trajectory_index, trajectory_metadata, trajectory_manifest,
    )):
        raise ValueError("Trajectory archive arguments conflict with a canonical reference")
    return load_reference_pool(
        canonical if canonical is not None else reference_config["canonical"], **selectors,
        sequence_length=reference_config["sequence_length"], horizon=reference_config["horizon"],
        expected=expected, reward_contract=reference_config["reward_contract"],
    )
