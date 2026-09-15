"""Monthly US equity selection baseline.

The module keeps raw files untouched and builds a small, auditable panel from
unadjusted and forward-adjusted monthly CSV files. It intentionally uses the
standard library so it can run in the repository's minimal environment.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
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


def _symbol_from_path(path: Path, suffix: str) -> str:
    if not path.name.endswith(suffix):
        raise ValueError(f"unexpected filename: {path.name}")
    return path.name[: -len(suffix)]


def _quality_record() -> dict[str, Any]:
    return {
        "bfq_rows": 0,
        "qfq_rows": 0,
        "invalid_rows": 0,
        "reasons": set(),
    }


def _record_issue(record: dict[str, Any], reason: str) -> None:
    record["invalid_rows"] += 1
    record["reasons"].add(reason)


def _parse_file(path: Path, adjustment: str, quality: dict[str, Any]) -> dict[str, dict[str, Any]]:
    suffix = f"_monthly_{adjustment}.csv"
    symbol = _symbol_from_path(path, suffix)
    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration:
            _record_issue(quality, "empty_file")
            return rows
        if header != HEADER:
            _record_issue(quality, "unexpected_header")
            return rows
        for line_number, row in enumerate(reader, start=2):
            if len(row) != len(HEADER):
                _record_issue(quality, "wrong_column_count")
                continue
            raw_date, raw_code = row[0].strip(), row[1].strip()
            try:
                parsed_date = date.fromisoformat(raw_date)
            except ValueError:
                _record_issue(quality, "invalid_date")
                continue
            if raw_code != symbol:
                _record_issue(quality, "code_filename_mismatch")
                continue
            values: dict[str, float] = {}
            try:
                for name, index in NUMERIC_COLUMNS.items():
                    value = float(row[index])
                    if not math.isfinite(value):
                        raise ValueError
                    values[name] = value
            except (TypeError, ValueError):
                _record_issue(quality, "invalid_numeric")
                continue
            if values["open"] <= 0 or values["close"] <= 0:
                _record_issue(quality, "nonpositive_open_close")
                continue
            if values["high"] <= 0 or values["low"] <= 0:
                _record_issue(quality, "nonpositive_high_low")
                continue
            if values["high"] < values["low"]:
                _record_issue(quality, "high_below_low")
                continue
            if values["high"] < max(values["open"], values["close"]):
                _record_issue(quality, "high_below_open_close")
                continue
            if values["low"] > min(values["open"], values["close"]):
                _record_issue(quality, "low_above_open_close")
                continue
            if any(values[name] < 0 for name in ("volume", "amount", "amplitude", "turnover")):
                _record_issue(quality, "negative_market_field")
                continue
            if raw_date in rows:
                _record_issue(quality, "duplicate_date")
                continue
            values.update({"date": raw_date, "parsed_date": parsed_date, "symbol": symbol})
            rows[raw_date] = values
            quality[f"{adjustment}_rows"] += 1
    return rows


def _read_directory(
    directory: Path, adjustment: str, selected_symbols: set[str] | None = None
):
    suffix = f"_monthly_{adjustment}.csv"
    paths = sorted(directory.glob(f"*{suffix}"))
    if selected_symbols is not None:
        paths = [path for path in paths if _symbol_from_path(path, suffix) in selected_symbols]
    result: dict[str, dict[str, dict[str, Any]]] = {}
    quality: dict[str, dict[str, Any]] = {}
    for path in paths:
        symbol = _symbol_from_path(path, suffix)
        record = _quality_record()
        result[symbol] = _parse_file(path, adjustment, record)
        quality[symbol] = record
    return result, quality


def build_monthly_panel(
    bfq_dir: Path | str,
    qfq_dir: Path | str,
    *,
    min_history: int = 12,
    max_symbols: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Build valid BFQ/QFQ rows keyed by symbol and date.

    ``min_history`` is used only to report feature eligibility later; rows are
    retained here so the quality report remains auditable.
    """
    del min_history
    bfq_path = Path(bfq_dir)
    qfq_path = Path(qfq_dir)
    bfq_suffix = "_monthly_bfq.csv"
    qfq_suffix = "_monthly_qfq.csv"
    available_symbols = sorted(
        {
            _symbol_from_path(path, bfq_suffix)
            for path in bfq_path.glob(f"*{bfq_suffix}")
        }
        | {
            _symbol_from_path(path, qfq_suffix)
            for path in qfq_path.glob(f"*{qfq_suffix}")
        }
    )
    selected_symbols = set(available_symbols[:max_symbols]) if max_symbols is not None else None
    bfq, bfq_quality = _read_directory(bfq_path, "bfq", selected_symbols)
    qfq, qfq_quality = _read_directory(qfq_path, "qfq", selected_symbols)
    symbols = sorted(set(bfq) | set(qfq))
    quality: dict[str, dict[str, Any]] = {}
    panel: list[dict[str, Any]] = []
    for symbol in symbols:
        merged_quality = _quality_record()
        for source in (bfq_quality.get(symbol, _quality_record()), qfq_quality.get(symbol, _quality_record())):
            merged_quality["bfq_rows"] += source.get("bfq_rows", 0)
            merged_quality["qfq_rows"] += source.get("qfq_rows", 0)
            merged_quality["invalid_rows"] += source.get("invalid_rows", 0)
            merged_quality["reasons"].update(source.get("reasons", set()))
        bfq_rows = bfq.get(symbol, {})
        qfq_rows = qfq.get(symbol, {})
        for raw_date in sorted(set(bfq_rows) & set(qfq_rows)):
            b, q = bfq_rows[raw_date], qfq_rows[raw_date]
            panel.append(
                {
                    "date": raw_date,
                    "symbol": symbol,
                    "parsed_date": b["parsed_date"],
                    "open_bfq": b["open"],
                    "close_bfq": b["close"],
                    "high_bfq": b["high"],
                    "low_bfq": b["low"],
                    "volume": b["volume"],
                    "amount": b["amount"],
                    "turnover": b["turnover"],
                    "close_qfq": q["close"],
                }
            )
        missing_qfq = set(bfq_rows) - set(qfq_rows)
        missing_bfq = set(qfq_rows) - set(bfq_rows)
        if not bfq_rows:
            merged_quality["reasons"].add("missing_bfq_file")
        if not qfq_rows:
            merged_quality["reasons"].add("missing_qfq_file")
        if missing_qfq:
            merged_quality["reasons"].add("missing_qfq_dates")
        if missing_bfq:
            merged_quality["reasons"].add("missing_bfq_dates")
        merged_quality["missing_qfq_date_count"] = len(missing_qfq)
        merged_quality["missing_bfq_date_count"] = len(missing_bfq)
        merged_quality["paired_rows"] = len(set(bfq_rows) & set(qfq_rows))
        merged_quality["reasons"] = sorted(merged_quality["reasons"])
        quality[symbol] = merged_quality
    panel.sort(key=lambda row: (row["date"], row["symbol"]))
    return panel, quality


def _month_distance(left: date, right: date) -> int:
    return (right.year - left.year) * 12 + right.month - left.month


def _rank(values: Iterable[tuple[int, float]]) -> dict[int, float]:
    ordered = sorted(values, key=lambda item: (-item[1], item[0]))
    if not ordered:
        return {}
    denominator = max(len(ordered) - 1, 1)
    return {index: (len(ordered) - 1 - rank) / denominator for rank, (index, _) in enumerate(ordered)}


def _features(panel: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_symbol: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in panel:
        by_symbol[row["symbol"]].append(row)
    result: list[dict[str, Any]] = []
    for rows in by_symbol.values():
        rows.sort(key=lambda row: row["parsed_date"])
        returns: list[float | None] = [None]
        for index, row in enumerate(rows):
            if index == 0:
                returns[0] = None
            else:
                previous = rows[index - 1]["close_qfq"]
                returns.append(row["close_qfq"] / previous - 1 if previous > 0 else None)
        for index, row in enumerate(rows):
            feature = dict(row)
            feature["history_count"] = index + 1
            for window in (3, 6, 12):
                feature[f"momentum_{window}m"] = (
                    row["close_qfq"] / rows[index - window]["close_qfq"] - 1
                    if index >= window and rows[index - window]["close_qfq"] > 0
                    else None
                )
            recent = rows[max(0, index - 2) : index + 1]
            feature["dollar_volume_3m"] = statistics.fmean(
                max(item["amount"], item["close_bfq"] * item["volume"], 0) for item in recent
            )
            feature["volume_mean_3m"] = statistics.fmean(item["volume"] for item in recent)
            feature["turnover_mean_3m"] = statistics.fmean(item["turnover"] for item in recent)
            recent_returns = [value for value in returns[max(0, index - 2) : index + 1] if value is not None]
            feature["volatility_3m"] = statistics.pstdev(recent_returns) if len(recent_returns) >= 2 else None
            next_row = rows[index + 1] if index + 1 < len(rows) else None
            if next_row and _month_distance(row["parsed_date"], next_row["parsed_date"]) == 1:
                feature["next_date"] = next_row["date"]
                feature["next_return"] = next_row["close_qfq"] / row["close_qfq"] - 1
            else:
                feature["next_date"] = None
                feature["next_return"] = None
            result.append(feature)
    return result


def run_backtest(
    panel: list[dict[str, Any]],
    *,
    top_n: int = 20,
    min_history: int = 12,
    min_price: float = 5.0,
    min_dollar_volume: float = 1_000_000.0,
) -> dict[str, list[dict[str, Any]]]:
    """Run a transparent monthly momentum/liquidity baseline."""
    rows = _features(panel)
    by_date: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_date[row["date"]].append(row)
    selections: list[dict[str, Any]] = []
    equity_curve: list[dict[str, Any]] = []
    equity = 1.0
    for signal_date in sorted(by_date):
        candidates = [
            row
            for row in by_date[signal_date]
            if row["history_count"] >= min_history
            and row["close_bfq"] >= min_price
            and row["dollar_volume_3m"] >= min_dollar_volume
            and row["momentum_6m"] is not None
        ]
        momentum_rank = _rank((id(row), row["momentum_6m"]) for row in candidates)
        liquidity_rank = _rank((id(row), math.log1p(row["dollar_volume_3m"])) for row in candidates)
        turnover_rank = _rank((id(row), row["turnover_mean_3m"]) for row in candidates)
        for row in candidates:
            row["score"] = (
                0.6 * momentum_rank[id(row)]
                + 0.2 * liquidity_rank[id(row)]
                + 0.2 * turnover_rank[id(row)]
            )
        chosen = sorted(candidates, key=lambda row: (-row["score"], row["symbol"]))[:top_n]
        for row in chosen:
            selections.append(
                {
                    "date": row["date"],
                    "symbol": row["symbol"],
                    "score": row["score"],
                    "momentum_6m": row["momentum_6m"],
                    "dollar_volume_3m": row["dollar_volume_3m"],
                    "turnover_mean_3m": row["turnover_mean_3m"],
                    "next_date": row["next_date"],
                    "next_return": row["next_return"],
                }
            )
        returns = [row["next_return"] for row in chosen if row["next_return"] is not None]
        if returns:
            portfolio_return = statistics.fmean(returns)
            equity *= 1 + portfolio_return
            equity_curve.append(
                {
                    "date": signal_date,
                    "portfolio_return": portfolio_return,
                    "equity": equity,
                    "selected_count": len(chosen),
                    "return_count": len(returns),
                }
            )
    return {"selections": selections, "equity_curve": equity_curve}


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fields} for row in rows)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _gap_report(quality: dict[str, dict[str, Any]]) -> dict[str, Any]:
    reason_counts: dict[str, int] = defaultdict(int)
    for item in quality.values():
        for reason in item["reasons"]:
            reason_counts[reason] += 1
    return {
        "symbol_count": len(quality),
        "reason_symbol_counts": dict(sorted(reason_counts.items())),
        "missing_bfq_file_symbols": sorted(
            symbol for symbol, item in quality.items() if "missing_bfq_file" in item["reasons"]
        ),
        "missing_qfq_file_symbols": sorted(
            symbol for symbol, item in quality.items() if "missing_qfq_file" in item["reasons"]
        ),
        "missing_bfq_date_rows": sum(item.get("missing_bfq_date_count", 0) for item in quality.values()),
        "missing_qfq_date_rows": sum(item.get("missing_qfq_date_count", 0) for item in quality.values()),
        "invalid_rows": sum(item["invalid_rows"] for item in quality.values()),
        "production_fields_not_present": [
            "point_in_time_corporate_actions",
            "dividends_and_splits_cashflow",
            "historical_shares_outstanding_and_float",
            "exchange_listing_delisting_status",
            "halt_and_resume_history",
            "commission_fees_and_borrow_costs",
            "spread_and_market_impact_slippage",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bfq-dir", type=Path, required=True)
    parser.add_argument("--qfq-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--min-history", type=int, default=12)
    parser.add_argument("--min-price", type=float, default=5.0)
    parser.add_argument("--min-dollar-volume", type=float, default=1_000_000.0)
    parser.add_argument("--max-symbols", type=int)
    args = parser.parse_args(argv)
    if not args.bfq_dir.is_dir() or not args.qfq_dir.is_dir():
        parser.error("both --bfq-dir and --qfq-dir must be directories")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    panel, quality = build_monthly_panel(
        args.bfq_dir, args.qfq_dir, min_history=args.min_history, max_symbols=args.max_symbols
    )
    result = run_backtest(
        panel,
        top_n=args.top_n,
        min_history=args.min_history,
        min_price=args.min_price,
        min_dollar_volume=args.min_dollar_volume,
    )
    selection_fields = [
        "date",
        "symbol",
        "score",
        "momentum_6m",
        "dollar_volume_3m",
        "turnover_mean_3m",
        "next_date",
        "next_return",
    ]
    curve_fields = ["date", "portfolio_return", "equity", "selected_count", "return_count"]
    _write_csv(args.output_dir / "selections.csv", result["selections"], selection_fields)
    _write_csv(args.output_dir / "equity_curve.csv", result["equity_curve"], curve_fields)
    _write_json(args.output_dir / "quality_report.json", quality)
    _write_json(args.output_dir / "data_gap_report.json", _gap_report(quality))
    parameters = {
        "top_n": args.top_n,
        "min_history": args.min_history,
        "min_price": args.min_price,
        "min_dollar_volume": args.min_dollar_volume,
        "max_symbols": args.max_symbols,
    }
    _write_json(
        args.output_dir / "run_manifest.json",
        {
            "bfq_dir": str(args.bfq_dir),
            "qfq_dir": str(args.qfq_dir),
            "panel_rows": len(panel),
            "selection_rows": len(result["selections"]),
            "equity_rows": len(result["equity_curve"]),
            "parameters": parameters,
            "warning": "QFQ data is a research convenience; build point-in-time adjustments for production backtests.",
        },
    )
    print(
        f"panel_rows={len(panel)} selections={len(result['selections'])} "
        f"equity_points={len(result['equity_curve'])} output={args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
