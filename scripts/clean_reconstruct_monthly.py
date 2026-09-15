"""Clean and reconstruct the three monthly US equity price series.

The raw files are kept untouched.  The unadjusted directory is treated as the
source of volume, amount, turnover, and execution-quality fields; forward
adjusted prices are used for continuous price features; backward adjusted
prices are retained for audit only.

The command writes a canonical monthly table, feature rows, quality reports,
and strict playback windows (six months of context followed by ten reference
months).  It uses the standard library and processes one symbol at a time so
the full universe does not need to fit in memory.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Iterable


HEADER = [
    "日期",
    "代码",
    "开盘价",
    "收盘价",
    "最高价",
    "最低价",
    "成交量",
    "成交额",
    "振幅",
    "涨跌幅",
    "涨跌额",
    "换手率",
]

NUMERIC_COLUMNS = {
    "open": 2,
    "close": 3,
    "high": 4,
    "low": 5,
    "volume": 6,
    "amount": 7,
    "amplitude": 8,
    "pct_change": 9,
    "change": 10,
    "turnover": 11,
}

FRAME_FIELDS = [
    "return_1m",
    "gap_return",
    "intramonth_return",
    "amplitude",
    "close_location",
    "momentum_3m",
    "momentum_6m",
    "volatility_3m",
    "volatility_6m",
    "log_volume",
    "volume_ratio_3m",
    "log_amount",
    "amount_ratio_3m",
    "turnover",
    "turnover_ratio_3m",
    "data_valid",
]

FEATURE_FIELDS = ["symbol", "period", *FRAME_FIELDS]

RAW_FIELDS = [
    "date",
    "open",
    "close",
    "high",
    "low",
    "volume",
    "amount",
    "turnover",
]

CANONICAL_FIELDS = [
    "symbol",
    "period",
    "raw_date",
    "raw_open",
    "raw_close",
    "raw_high",
    "raw_low",
    "raw_volume",
    "raw_amount",
    "raw_turnover",
    "qfq_date",
    "qfq_open",
    "qfq_close",
    "qfq_high",
    "qfq_low",
    "hfq_date",
    "hfq_open",
    "hfq_close",
    "hfq_high",
    "hfq_low",
    "raw_valid",
    "qfq_valid",
    "hfq_valid",
    "quality_flags",
]


@dataclass
class ParsedSeries:
    rows: dict[str, dict[str, Any]]
    stats: dict[str, Any]
    issues: list[dict[str, Any]]


def _month_key(parsed: date) -> str:
    return f"{parsed.year:04d}-{parsed.month:02d}"


def _month_number(period: str) -> int:
    year, month = period.split("-", 1)
    return int(year) * 12 + int(month)


def _consecutive(left: str, right: str) -> bool:
    return _month_number(right) - _month_number(left) == 1


def _symbol_from_path(path: Path, suffix: str) -> str:
    if not path.name.endswith(suffix):
        raise ValueError(f"unexpected file suffix: {path}")
    return path.name[: -len(suffix)]


def _empty_stats() -> dict[str, Any]:
    return {
        "rows_seen": 0,
        "valid_rows": 0,
        "invalid_rows": 0,
        "reasons": Counter(),
    }


def _issue(
    issues: list[dict[str, Any]],
    stats: dict[str, Any],
    *,
    source: str,
    symbol: str,
    line_number: int,
    period: str | None,
    reason: str,
) -> None:
    stats["invalid_rows"] += 1
    stats["reasons"][reason] += 1
    issues.append(
        {
            "source": source,
            "symbol": symbol,
            "period": period or "",
            "line_number": line_number,
            "reason": reason,
        }
    )


def _validate_values(values: dict[str, float]) -> str | None:
    if any(not math.isfinite(value) for value in values.values()):
        return "nonfinite_numeric"
    if any(values[name] <= 0 for name in ("open", "close", "high", "low")):
        return "nonpositive_ohlc"
    if values["high"] < values["low"]:
        return "high_below_low"
    if values["high"] < max(values["open"], values["close"]):
        return "high_below_open_close"
    if values["low"] > min(values["open"], values["close"]):
        return "low_above_open_close"
    if any(values[name] < 0 for name in ("volume", "amount", "amplitude", "turnover")):
        return "negative_market_field"
    return None


def parse_series_file(path: Path, source: str) -> ParsedSeries:
    """Parse and hard-filter one monthly CSV without changing the raw file."""
    suffix_map = {"raw": "_monthly_bfq.csv", "qfq": "_monthly_qfq.csv", "hfq": "_monthly_hfq.csv"}
    symbol = _symbol_from_path(path, suffix_map[source])
    stats = _empty_stats()
    issues: list[dict[str, Any]] = []
    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration:
            _issue(issues, stats, source=source, symbol=symbol, line_number=1, period=None, reason="empty_file")
            return ParsedSeries(rows, stats, issues)
        if header != HEADER:
            _issue(issues, stats, source=source, symbol=symbol, line_number=1, period=None, reason="unexpected_header")
            return ParsedSeries(rows, stats, issues)
        for line_number, row in enumerate(reader, start=2):
            stats["rows_seen"] += 1
            if len(row) != len(HEADER):
                _issue(issues, stats, source=source, symbol=symbol, line_number=line_number, period=None, reason="wrong_column_count")
                continue
            raw_date = row[0].strip()
            raw_code = row[1].strip()
            try:
                parsed_date = date.fromisoformat(raw_date)
            except ValueError:
                _issue(issues, stats, source=source, symbol=symbol, line_number=line_number, period=None, reason="invalid_date")
                continue
            period = _month_key(parsed_date)
            if raw_code != symbol:
                _issue(issues, stats, source=source, symbol=symbol, line_number=line_number, period=period, reason="code_filename_mismatch")
                continue
            try:
                values = {name: float(row[index]) for name, index in NUMERIC_COLUMNS.items()}
            except (TypeError, ValueError):
                _issue(issues, stats, source=source, symbol=symbol, line_number=line_number, period=period, reason="invalid_numeric")
                continue
            reason = _validate_values(values)
            if reason:
                _issue(issues, stats, source=source, symbol=symbol, line_number=line_number, period=period, reason=reason)
                continue
            if period in rows:
                _issue(issues, stats, source=source, symbol=symbol, line_number=line_number, period=period, reason="duplicate_period")
                # Keep the later valid trading date for a duplicate month.
                if raw_date <= rows[period]["date"]:
                    continue
            item = {name: values[name] for name in RAW_FIELDS if name != "date"}
            item["date"] = raw_date
            item["period"] = period
            rows[period] = item
            stats["valid_rows"] += 1
    stats["reasons"] = dict(sorted(stats["reasons"].items()))
    return ParsedSeries(rows, stats, issues)


def _mean(values: Iterable[float]) -> float | None:
    values = list(values)
    return statistics.fmean(values) if values else None


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator - 1 if denominator > 0 else None


def _contiguous(periods: list[str], start: int, end: int) -> bool:
    return all(_consecutive(periods[index - 1], periods[index]) for index in range(start + 1, end + 1))


def build_features(symbol: str, raw_rows: dict[str, dict[str, Any]], qfq_rows: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Build one feature row per raw/qfq paired month.

    Price features come from QFQ; volume, amount, and turnover come from raw.
    Rows with insufficient history keep ``None`` for the relevant feature and
    can be excluded by the strict playback-window builder.
    """
    periods = sorted(set(raw_rows) & set(qfq_rows), key=_month_number)
    rows: list[dict[str, Any]] = []
    returns: list[float | None] = []
    for index, period in enumerate(periods):
        raw = raw_rows[period]
        qfq = qfq_rows[period]
        previous = qfq_rows.get(periods[index - 1]) if index > 0 and _consecutive(periods[index - 1], period) else None
        qfq_open, qfq_close = qfq["open"], qfq["close"]
        qfq_high, qfq_low = qfq["high"], qfq["low"]
        return_1m = _safe_ratio(qfq_close, previous["close"]) if previous else None
        returns.append(return_1m)
        feature: dict[str, Any] = {
            "symbol": symbol,
            "period": period,
            "return_1m": return_1m,
            "gap_return": _safe_ratio(qfq_open, previous["close"]) if previous else None,
            "intramonth_return": _safe_ratio(qfq_close, qfq_open),
            "amplitude": (qfq_high - qfq_low) / qfq_open,
            "close_location": (qfq_close - qfq_low) / (qfq_high - qfq_low) if qfq_high > qfq_low else 0.5,
            "momentum_3m": None,
            "momentum_6m": None,
            "volatility_3m": None,
            "volatility_6m": None,
            "log_volume": math.log1p(raw["volume"]),
            "volume_ratio_3m": None,
            "log_amount": math.log1p(raw["amount"]),
            "amount_ratio_3m": None,
            "turnover": raw["turnover"],
            "turnover_ratio_3m": None,
            "data_valid": 1,
        }
        for window, name in ((3, "momentum_3m"), (6, "momentum_6m")):
            if index >= window and _contiguous(periods, index - window, index):
                feature[name] = _safe_ratio(qfq_close, qfq_rows[periods[index - window]]["close"])
        for window, name in ((3, "volatility_3m"), (6, "volatility_6m")):
            if index >= window and _contiguous(periods, index - window, index) and all(
                value is not None for value in returns[index - window + 1 : index + 1]
            ):
                feature[name] = statistics.pstdev(returns[index - window + 1 : index + 1])
        if index >= 3 and _contiguous(periods, index - 3, index):
            previous_raw = [raw_rows[periods[item]] for item in range(index - 3, index)]
            volume_mean = _mean(item["volume"] for item in previous_raw)
            amount_mean = _mean(item["amount"] for item in previous_raw)
            turnover_mean = _mean(item["turnover"] for item in previous_raw)
            feature["volume_ratio_3m"] = raw["volume"] / volume_mean if volume_mean and volume_mean > 0 else None
            feature["amount_ratio_3m"] = raw["amount"] / amount_mean if amount_mean and amount_mean > 0 else None
            feature["turnover_ratio_3m"] = raw["turnover"] / turnover_mean if turnover_mean and turnover_mean > 0 else None
        feature["frame"] = [feature[field] for field in FRAME_FIELDS]
        rows.append(feature)
    return rows


def build_playback_samples(
    features: list[dict[str, Any]], *, context_length: int = 6, future_length: int = 10
) -> list[dict[str, Any]]:
    """Build strict, contiguous playback windows from feature rows."""
    samples: list[dict[str, Any]] = []
    total = context_length + future_length
    for anchor in range(context_length - 1, len(features) - future_length):
        start = anchor - context_length + 1
        end = anchor + future_length + 1
        window = features[start:end]
        if len(window) != total:
            continue
        if any(not item.get("data_valid") or any(value is None for value in item["frame"][:-1]) for item in window):
            continue
        if not all(_consecutive(window[index - 1]["period"], window[index]["period"]) for index in range(1, len(window))):
            continue
        samples.append(
            {
                "symbol": features[0]["symbol"],
                "anchor_period": features[anchor]["period"],
                "current_state": [item["frame"] for item in features[start : anchor + 1]],
                "future_reference": [item["frame"] for item in features[anchor + 1 : end]],
            }
        )
    return samples


def _collect_paths(directory: Path, suffix: str) -> dict[str, Path]:
    return {_symbol_from_path(path, suffix): path for path in directory.glob(f"*{suffix}")}


def _csv_writer(path: Path, fields: list[str]) -> tuple[Any, csv.DictWriter]:
    handle = path.open("w", encoding="utf-8", newline="")
    return handle, csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")


def _canonical_row(symbol: str, period: str, raw: dict[str, Any] | None, qfq: dict[str, Any] | None, hfq: dict[str, Any] | None) -> dict[str, Any]:
    row: dict[str, Any] = {"symbol": symbol, "period": period}
    for prefix, source in (("raw", raw), ("qfq", qfq), ("hfq", hfq)):
        row[f"{prefix}_date"] = source.get("date") if source else ""
        for field in ("open", "close", "high", "low"):
            row[f"{prefix}_{field}"] = source.get(field) if source else ""
        if prefix == "raw":
            for field in ("volume", "amount", "turnover"):
                row[f"raw_{field}"] = source.get(field) if source else ""
        row[f"{prefix}_valid"] = int(source is not None)
    flags = []
    if raw is None:
        flags.append("missing_raw")
    if qfq is None:
        flags.append("missing_qfq")
    if hfq is None:
        flags.append("missing_hfq")
    row["quality_flags"] = ";".join(flags)
    return row


def clean_monthly(
    raw_dir: Path,
    qfq_dir: Path,
    hfq_dir: Path,
    output_dir: Path,
    *,
    max_symbols: int | None = None,
    context_length: int = 6,
    future_length: int = 10,
) -> dict[str, Any]:
    raw_paths = _collect_paths(raw_dir, "_monthly_bfq.csv")
    qfq_paths = _collect_paths(qfq_dir, "_monthly_qfq.csv")
    hfq_paths = _collect_paths(hfq_dir, "_monthly_hfq.csv")
    symbols = sorted(set(raw_paths) | set(qfq_paths) | set(hfq_paths))
    if max_symbols is not None:
        symbols = symbols[:max_symbols]
    output_dir.mkdir(parents=True, exist_ok=True)
    canonical_handle, canonical_writer = _csv_writer(output_dir / "monthly_canonical.csv", CANONICAL_FIELDS)
    feature_handle, feature_writer = _csv_writer(output_dir / "monthly_features.csv", FEATURE_FIELDS)
    issue_handle, issue_writer = _csv_writer(output_dir / "row_issues.csv", ["source", "symbol", "period", "line_number", "reason"])
    sample_handle = (output_dir / "playback_samples.jsonl").open("w", encoding="utf-8")
    canonical_writer.writeheader()
    feature_writer.writeheader()
    issue_writer.writeheader()
    quality_rows: list[dict[str, Any]] = []
    total_canonical = 0
    total_features = 0
    total_samples = 0
    try:
        for symbol in symbols:
            parsed: dict[str, ParsedSeries] = {}
            for source, paths in (("raw", raw_paths), ("qfq", qfq_paths), ("hfq", hfq_paths)):
                parsed[source] = parse_series_file(paths[symbol], source) if symbol in paths else ParsedSeries({}, _empty_stats(), [])
                issue_writer.writerows(parsed[source].issues)
            raw_rows = parsed["raw"].rows
            qfq_rows = parsed["qfq"].rows
            hfq_rows = parsed["hfq"].rows
            for period in sorted(set(raw_rows) | set(qfq_rows) | set(hfq_rows), key=_month_number):
                canonical_writer.writerow(_canonical_row(symbol, period, raw_rows.get(period), qfq_rows.get(period), hfq_rows.get(period)))
                total_canonical += 1
            features = build_features(symbol, raw_rows, qfq_rows)
            for feature in features:
                feature_writer.writerow(feature)
                total_features += 1
            samples = build_playback_samples(features, context_length=context_length, future_length=future_length)
            for sample in samples:
                sample_handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
            total_samples += len(samples)
            paired_periods = set(raw_rows) & set(qfq_rows)
            quality_rows.append(
                {
                    "symbol": symbol,
                    "raw_rows": len(raw_rows),
                    "qfq_rows": len(qfq_rows),
                    "hfq_rows": len(hfq_rows),
                    "paired_rows": len(paired_periods),
                    "feature_rows": len(features),
                    "playback_samples": len(samples),
                    "missing_qfq_periods": len(set(raw_rows) - set(qfq_rows)),
                    "missing_raw_periods": len(set(qfq_rows) - set(raw_rows)),
                    "raw_invalid_rows": parsed["raw"].stats["invalid_rows"],
                    "qfq_invalid_rows": parsed["qfq"].stats["invalid_rows"],
                    "hfq_invalid_rows": parsed["hfq"].stats["invalid_rows"],
                    "raw_reasons": json.dumps(parsed["raw"].stats["reasons"], ensure_ascii=False, sort_keys=True),
                    "qfq_reasons": json.dumps(parsed["qfq"].stats["reasons"], ensure_ascii=False, sort_keys=True),
                    "hfq_reasons": json.dumps(parsed["hfq"].stats["reasons"], ensure_ascii=False, sort_keys=True),
                }
            )
    finally:
        canonical_handle.close()
        feature_handle.close()
        sample_handle.close()
        issue_handle.close()
    quality_fields = list(quality_rows[0]) if quality_rows else ["symbol"]
    quality_handle, quality_writer = _csv_writer(output_dir / "symbol_quality.csv", quality_fields)
    try:
        quality_writer.writeheader()
        quality_writer.writerows(quality_rows)
    finally:
        quality_handle.close()
    report = {
        "symbols": len(symbols),
        "canonical_rows": total_canonical,
        "feature_rows": total_features,
        "playback_samples": total_samples,
        "context_length": context_length,
        "future_length": future_length,
        "source_policy": {
            "price_features": "qfq",
            "volume_amount_turnover": "raw_bfq_directory",
            "hfq": "audit_only",
        },
        "output_files": [
            "monthly_canonical.csv",
            "monthly_features.csv",
            "playback_samples.jsonl",
            "row_issues.csv",
            "symbol_quality.csv",
        ],
    }
    (output_dir / "clean_manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--qfq-dir", type=Path, required=True)
    parser.add_argument("--hfq-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-symbols", type=int)
    parser.add_argument("--context-length", type=int, default=6)
    parser.add_argument("--future-length", type=int, default=10)
    args = parser.parse_args(argv)
    for directory in (args.raw_dir, args.qfq_dir, args.hfq_dir):
        if not directory.is_dir():
            parser.error(f"not a directory: {directory}")
    report = clean_monthly(
        args.raw_dir,
        args.qfq_dir,
        args.hfq_dir,
        args.output_dir,
        max_symbols=args.max_symbols,
        context_length=args.context_length,
        future_length=args.future_length,
    )
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
