"""Prepare one full financial Sonic reference pool with audited feature clipping.

Run with python -m scripts.prepare_finance_features. No date split or quality
sampling weights are introduced. Raw reference targets are stored alongside
bounded model inputs; FinancialSonic expects raw inputs and applies the saved
statistics internally, so normalized arrays must not be normalized again.
"""

from __future__ import annotations

import argparse
import csv
from datetime import date, datetime, timezone
import hashlib
from itertools import groupby
import json
from pathlib import Path
import time

import numpy as np

from gear_sonic.finance.observations import (
    CURRENT_FIELDS, FUTURE_FIELDS, build_market_context, build_symbol_observations,
    make_sample, month_number, read_canonical,
)
from gear_sonic.finance.preprocessing import fit_feature_statistics, transform_features


def _anchors(rows, horizon):
    run_length = 0
    for i, row in enumerate(rows):
        consecutive = i > 0 and month_number(row.period) - month_number(rows[i - 1].period) == 1
        run_length = run_length + 1 if consecutive else 1
        if run_length > horizon:
            yield i - horizon


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _hash_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_features(
    canonical: str | Path, output_dir: str | Path, *, as_of: str,
    horizon: int = 10, clip: float = 10.0, symbols: set[str] | None = None,
) -> dict:
    started = time.perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()
    canonical, output_dir = Path(canonical), Path(output_dir)
    as_of_date = date.fromisoformat(as_of)
    cutoff = as_of_date.strftime("%Y-%m")
    if horizon < 1 or not np.isfinite(clip) or clip <= 0:
        raise ValueError("horizon and clip must be positive and finite")
    if output_dir.exists():
        raise FileExistsError(f"Use a fresh output directory: {output_dir}")
    source_before = canonical.stat()

    def completed_bars(selected=None):
        return (bar for bar in read_canonical(canonical, selected) if bar.period < cutoff)

    print("Building full-universe context from completed months", flush=True)
    market = build_market_context(completed_bars())
    observations = {}
    for symbol, group in groupby(completed_bars(symbols), key=lambda bar: bar.symbol):
        if symbol in observations:
            raise ValueError("Canonical rows must be grouped by symbol")
        observations[symbol] = build_symbol_observations(list(group), market)
    del market
    counts = {symbol: sum(1 for _ in _anchors(rows, horizon))
              for symbol, rows in observations.items()}
    count = sum(counts.values())
    if not count:
        raise ValueError("No complete reference windows available")
    print(f"Exporting {count} anchors from {sum(v > 0 for v in counts.values())} symbols", flush=True)
    output_dir.mkdir(parents=True, exist_ok=False)
    shapes = {"current": (count, len(CURRENT_FIELDS)),
              "future": (count, horizon, len(FUTURE_FIELDS))}
    raw = {name: np.lib.format.open_memmap(output_dir / f"{name}_raw.npy", mode="w+",
                                         dtype=np.float32, shape=shape)
           for name, shape in shapes.items()}
    with (output_dir / "sample_index.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["row_id", "symbol", "anchor_period", "target_end_period"])
        batch, offset = [], 0

        def flush():
            nonlocal offset
            if not batch:
                return
            for name, key in (("current", "current_state"), ("future", "future_reference")):
                values = np.asarray([sample[key] for sample in batch], dtype=np.float32)
                if not np.isfinite(values).all():
                    raise ValueError("Nonfinite feature in a complete reference window")
                raw[name][offset:offset + len(batch)] = values
            for i, sample in enumerate(batch, start=offset):
                writer.writerow([i, sample["symbol"], sample["anchor_period"], sample["target_end_period"]])
            offset += len(batch)
            batch.clear()

        for symbol, rows in observations.items():
            for anchor in _anchors(rows, horizon):
                batch.append(make_sample(rows, anchor, horizon))
                if len(batch) == 4096:
                    flush()
        flush()
        if offset != count:
            raise RuntimeError("Export count disagrees with eligible anchors")
    del observations
    for array in raw.values():
        array.flush()

    statistics = {"schema_version": 1, "fit_scope": "all_reference_trajectories",
                  "split": "none", "horizon": horizon, "as_of": as_of,
                  "method": "median_iqr_then_clip", "sample_count": count,
                  "fit_weighting": "one current per anchor; all future horizon occurrences"}
    report_groups = {}
    for name, fields in (("current", CURRENT_FIELDS), ("future", FUTURE_FIELDS)):
        print(f"Fitting exact {name} feature quantiles", flush=True)
        stats = fit_feature_statistics(raw[name], fields, clip=clip)
        statistics[name] = stats
        normalized = np.lib.format.open_memmap(output_dir / f"{name}_normalized.npy", mode="w+",
                                              dtype=np.float32, shape=shapes[name])
        masks = np.lib.format.open_memmap(output_dir / f"{name}_clipped.npy", mode="w+",
                                         dtype=np.bool_, shape=shapes[name])
        clipped_values, clipped_samples = 0, 0
        for start in range(0, count, 4096):
            stop = min(start + 4096, count)
            values, clipped = transform_features(raw[name][start:stop], stats)
            normalized[start:stop], masks[start:stop] = values, clipped
            clipped_values += int(clipped.sum())
            clipped_samples += int(clipped.reshape(stop - start, -1).any(axis=1).sum())
        normalized.flush()
        masks.flush()
        report_groups[name] = {
            "clipped_values": clipped_values, "samples_with_clipping": clipped_samples,
            "total_values": int(raw[name].size), "features": stats["features"],
        }
    _write_json(output_dir / "normalization_stats.json", statistics)
    _write_json(output_dir / "feature_outlier_report.json", report_groups)
    with (output_dir / "symbol_coverage.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["symbol", "samples", "uniform_window_probability"])
        writer.writerows((symbol, number, number / count) for symbol, number in counts.items())
    _write_json(output_dir / "feature_schema.json", {
        "schema_version": 1, "current_fields": list(CURRENT_FIELDS), "future_fields": list(FUTURE_FIELDS),
        "horizon": horizon, "shapes": shapes, "dtype": "float32",
        "mask_semantics": "clipped arrays mark altered inputs, not missing data",
        "validity": "all exported cells valid; no imputation or padded periods",
        "raw_target": "future_raw.npy[..., 0] is the unmodified monthly log-return target",
        "model_input": "load_normalizers then pass current_raw and future_raw to FinancialSonic",
        "normalization_warning": "Do not feed normalized arrays through model normalizers again",
        "ordering": "grouped by symbol and ascending anchor month; reset cache at time gaps/symbol changes",
        "source_policy": "QFQ prices; raw volume and raw_close*volume proxy; no reported turnover/amount",
    })
    source_after = canonical.stat()
    if (source_before.st_size, source_before.st_mtime_ns) != (source_after.st_size, source_after.st_mtime_ns):
        raise RuntimeError("Canonical input changed during export")
    report = {
        "status": "complete", "schema_version": 1, "split": "none",
        "quality_weighting": "disabled", "sampling": "uniform windows available; coverage report only",
        "fit_scope": "all_reference_trajectories", "as_of": as_of,
        "completed_months_before": cutoff, "horizon": horizon,
        "samples": count, "symbols_with_samples": sum(v > 0 for v in counts.values()),
        "canonical": str(canonical.resolve()), "canonical_sha256": _hash_file(canonical),
        "canonical_size": source_after.st_size,
        "shapes": shapes, "clip": clip, "target_clipping": False,
        "clipped_values": {name: value["clipped_values"] for name, value in report_groups.items()},
        "started_at": started_at, "finished_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    _write_json(output_dir / "manifest.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--as-of", default=date.today().isoformat())
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--clip", type=float, default=10.0)
    parser.add_argument("--symbols", nargs="+")
    args = parser.parse_args()
    report = prepare_features(args.canonical, args.output_dir, as_of=args.as_of, horizon=args.horizon,
                              clip=args.clip, symbols=set(args.symbols) if args.symbols else None)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
