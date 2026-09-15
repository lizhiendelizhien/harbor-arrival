"""Bound legacy playback features using one unsplit pool of unique months.

Normalized reference trajectories are tracking targets, not financial returns.
The unchanged input JSONL is the source for raw reference values. Only a fresh
output directory is accepted; manifest.json is written after successful finish.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from datetime import datetime, timezone
from itertools import islice
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gear_sonic.finance.preprocessing import fit_feature_statistics, transform_features
from scripts.clean_reconstruct_monthly import FRAME_FIELDS


CONTINUOUS_FIELDS = FRAME_FIELDS[:-1]


def _source_stat(path: Path) -> dict:
    stat = path.stat()
    return {"path": str(path.resolve()), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _line_count(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for _ in handle)


def _identity(symbol, period) -> tuple[str, str]:
    if not isinstance(symbol, str) or not symbol or any(char.isspace() for char in symbol):
        raise ValueError(f"invalid symbol: {symbol!r}")
    if not isinstance(period, str) or not re.fullmatch(r"\d{4}-(0[1-9]|1[0-2])", period):
        raise ValueError(f"invalid period: {period!r}")
    return symbol, period


def _write_json(path: Path, value: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def _read_monthly(path: Path, index_path: Path) -> np.ndarray:
    expected_count = _line_count(path) - 1
    if expected_count <= 0:
        raise ValueError("monthly features are empty")
    values = np.empty((expected_count, len(FRAME_FIELDS)), dtype=np.float64)
    seen = set()
    count = 0
    with path.open(encoding="utf-8-sig", newline="") as handle, index_path.open("w", newline="") as index:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["symbol", "period", *FRAME_FIELDS]:
            raise ValueError("monthly feature schema does not match legacy FRAME_FIELDS")
        writer = csv.writer(index)
        writer.writerow(["row_index", "symbol", "period", "source_line"])
        for row in reader:
            identity = _identity(row["symbol"], row["period"])
            if identity in seen:
                raise ValueError(f"duplicate symbol-month: {identity}")
            seen.add(identity)
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"wrong column count at {identity}")
            frame = [float(row[field]) if row[field].strip() else np.nan for field in FRAME_FIELDS]
            if frame[-1] not in (0.0, 1.0):
                raise ValueError(f"invalid data_valid at {identity}")
            values[count] = frame
            writer.writerow([count, *identity, reader.line_num])
            count += 1
    if count != expected_count:
        raise ValueError("monthly row count changed or multiline/blank CSV rows found")
    return values


def _normalize_frames(values: np.ndarray, stats: dict, *, allow_missing: bool):
    missing = ~np.isfinite(values)
    if missing.any() and not allow_missing:
        raise ValueError("playback frames must contain only finite numeric values")
    if not np.isin(values[..., -1], [0, 1]).all():
        raise ValueError("data_valid must be 0 or 1")
    continuous = np.where(missing[..., :-1], np.asarray(stats["center"]), values[..., :-1])
    normalized_continuous, clipped_continuous = transform_features(continuous, stats)
    normalized = np.empty(values.shape, dtype=np.float32)
    normalized[..., :-1] = normalized_continuous
    normalized[..., -1] = values[..., -1]
    normalized[missing] = 0
    clipped = np.zeros(values.shape, dtype=np.bool_)
    clipped[..., :-1] = clipped_continuous
    clipped[missing] = False
    return normalized, clipped, missing


def _new_array(output: Path, name: str, shape: tuple, dtype):
    return np.lib.format.open_memmap(output / name, mode="w+", dtype=dtype, shape=shape)


def prepare_legacy_features(
    monthly_features: Path,
    playback_samples: Path,
    output_dir: Path,
    *,
    clip: float = 10.0,
    batch_size: int = 2048,
) -> dict:
    """Preserve all source rows and windows while bounding model-only features."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    monthly_features, playback_samples, output_dir = map(Path, (monthly_features, playback_samples, output_dir))
    sources = {"monthly_features": _source_stat(monthly_features), "playback_samples": _source_stat(playback_samples)}
    output_dir.mkdir(parents=True, exist_ok=False)
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    values = _read_monthly(monthly_features, output_dir / "monthly_index.csv")
    stats = fit_feature_statistics(values[:, :-1], CONTINUOUS_FIELDS, clip=clip)
    stats["fit_scope"] = "all unique symbol-month rows; no train/validation/test split"
    stats["source"] = sources["monthly_features"]
    stats["untouched_fields"] = ["data_valid"]
    _write_json(output_dir / "normalization_stats.json", stats)
    monthly_count = len(values)
    monthly = _new_array(output_dir, "monthly_features.npy", values.shape, np.float32)
    monthly_clipped = _new_array(output_dir, "monthly_clipped.npy", values.shape, np.bool_)
    monthly_missing = _new_array(output_dir, "monthly_missing.npy", values.shape, np.bool_)
    clipped_counts = np.zeros(len(FRAME_FIELDS), dtype=np.int64)
    missing_counts = np.zeros(len(FRAME_FIELDS), dtype=np.int64)
    for start in range(0, monthly_count, batch_size):
        end = min(start + batch_size, monthly_count)
        normalized, clipped, missing = _normalize_frames(values[start:end], stats, allow_missing=True)
        monthly[start:end], monthly_clipped[start:end], monthly_missing[start:end] = normalized, clipped, missing
        clipped_counts += clipped.sum(axis=0)
        missing_counts += missing.sum(axis=0)
    for array in (monthly, monthly_clipped, monthly_missing):
        array.flush()
    del values, monthly, monthly_clipped, monthly_missing
    print(f"Prepared {monthly_count:,} unique monthly rows; processing playback windows", flush=True)

    sample_count = _line_count(playback_samples)
    if sample_count == 0:
        raise ValueError("playback samples are empty")
    current = _new_array(output_dir, "current_state.npy", (sample_count, 6, 16), np.float32)
    future = _new_array(output_dir, "future_reference.npy", (sample_count, 10, 16), np.float32)
    current_clipped = _new_array(output_dir, "current_clipped.npy", current.shape, np.bool_)
    future_clipped = _new_array(output_dir, "future_clipped.npy", future.shape, np.bool_)
    current_counts = np.zeros(16, dtype=np.int64)
    future_counts = np.zeros(16, dtype=np.int64)
    seen_samples = set()
    offset = 0
    samples_with_clipping = 0
    with playback_samples.open(encoding="utf-8") as handle, (output_dir / "playback_index.csv").open("w", newline="") as index:
        writer = csv.writer(index)
        writer.writerow(["sample_index", "symbol", "anchor_period", "source_line"])
        while lines := list(islice(handle, batch_size)):
            batch_current = np.empty((len(lines), 6, 16), dtype=np.float64)
            batch_future = np.empty((len(lines), 10, 16), dtype=np.float64)
            for row_number, line in enumerate(lines):
                sample = json.loads(line)
                identity = _identity(sample.get("symbol"), sample.get("anchor_period"))
                if identity in seen_samples:
                    raise ValueError(f"duplicate playback identity: {identity}")
                seen_samples.add(identity)
                for key, target, shape in (
                    ("current_state", batch_current, (6, 16)),
                    ("future_reference", batch_future, (10, 16)),
                ):
                    frame = np.asarray(sample.get(key), dtype=np.float64)
                    if frame.shape != shape:
                        raise ValueError(f"{key} at {identity} must have shape {shape}")
                    if not np.isfinite(frame).all():
                        raise ValueError(f"{key} at {identity} must contain finite values")
                    if not (frame[:, -1] == 1).all():
                        raise ValueError(f"strict playback data_valid must be 1 at {identity}")
                    target[row_number] = frame
                writer.writerow([offset + row_number, *identity, offset + row_number + 1])
            stop = offset + len(lines)
            norm_current, clip_current, _ = _normalize_frames(batch_current, stats, allow_missing=False)
            norm_future, clip_future, _ = _normalize_frames(batch_future, stats, allow_missing=False)
            current[offset:stop], current_clipped[offset:stop] = norm_current, clip_current
            future[offset:stop], future_clipped[offset:stop] = norm_future, clip_future
            current_counts += clip_current.sum(axis=(0, 1))
            future_counts += clip_future.sum(axis=(0, 1))
            samples_with_clipping += int((clip_current.any(axis=(1, 2)) | clip_future.any(axis=(1, 2))).sum())
            offset = stop
            if offset % (batch_size * 16) == 0:
                print(f"Prepared {offset:,}/{sample_count:,} playback windows", flush=True)
    if offset != sample_count:
        raise ValueError("playback row count changed during preparation")
    for array in (current, future, current_clipped, future_clipped):
        array.flush()
    for name, path in (("monthly_features", monthly_features), ("playback_samples", playback_samples)):
        if _source_stat(path) != sources[name]:
            raise ValueError(f"source changed during preparation: {path}")

    report = {
        "policy": "median/IQR scaling with bounded normalized features; original observations unchanged",
        "rows_dropped": 0,
        "samples_dropped": 0,
        "samples_with_any_clipping": samples_with_clipping,
        "features": {
            field: {
                **stats.get("features", {}).get(field, {}),
                "monthly_clipped_count": int(clipped_counts[column]),
                "monthly_missing_count": int(missing_counts[column]),
                "current_clipped_count": int(current_counts[column]),
                "future_clipped_count": int(future_counts[column]),
            }
            for column, field in enumerate(FRAME_FIELDS)
        },
        "clipped_reference_notice": "Clipped future_reference.npy is a bounded tracking target. Use original playback_samples.jsonl for raw trajectories and economic calculations.",
    }
    _write_json(output_dir / "feature_outlier_report.json", report)
    manifest = {
        "status": "complete",
        "schema": "legacy_monthly_playback_6x16_10x16",
        "split": "none",
        "quality_weighting": "none",
        "monthly_row_count": monthly_count,
        "sample_count": sample_count,
        "fields": FRAME_FIELDS,
        "current_state_shape": [sample_count, 6, 16],
        "future_reference_shape": [sample_count, 10, 16],
        "dtype": "float32",
        "masks_dtype": "bool",
        "missing_policy": "monthly missing values stored as zero plus monthly_missing mask; playback must be fully finite",
        "clip": float(clip),
        "raw_reference_source": sources["playback_samples"],
        "sources": sources,
        "outputs": sorted(path.name for path in output_dir.iterdir()) + ["manifest.json"],
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
    }
    _write_json(output_dir / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--monthly-features", required=True, type=Path)
    parser.add_argument("--playback-samples", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--clip", type=float, default=10.0)
    parser.add_argument("--batch-size", type=int, default=2048)
    args = parser.parse_args()
    manifest = prepare_legacy_features(args.monthly_features, args.playback_samples, args.output_dir, clip=args.clip, batch_size=args.batch_size)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
