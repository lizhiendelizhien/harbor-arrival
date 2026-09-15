"""Download and clean the small Databento market-data supplement.

The script intentionally uses the HTTP API and the Python standard library so
the repository does not need the Databento client package just to run a small
monthly execution-cost sample.
"""

from __future__ import annotations

import argparse
import base64
import csv
import gzip
import json
import os
import time
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo


API_ROOT = "https://hist.databento.com/v0"
PRICE_SCALE = 1_000_000_000
EASTERN = ZoneInfo("America/New_York")
UTC = timezone.utc


def _json_lines(path: Path) -> Iterator[dict[str, Any]]:
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def load_definition_map(path: Path) -> dict[int, str]:
    mapping: dict[int, str] = {}
    for record in _json_lines(path):
        header = record.get("hd", {})
        instrument_id = header.get("instrument_id")
        symbol = record.get("raw_symbol")
        if instrument_id is not None and symbol:
            mapping[int(instrument_id)] = str(symbol)
    return mapping


def _quality() -> dict[str, int]:
    return {
        "rows": 0,
        "missing_levels": 0,
        "crossed_quotes": 0,
        "unmapped_instrument": 0,
        "invalid_price": 0,
        "invalid_json": 0,
    }


def normalize_bbo_file(
    path: Path, symbol_map: dict[int, str]
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows: list[dict[str, Any]] = []
    quality = _quality()
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                header = record.get("hd", {})
                instrument_id = int(header["instrument_id"])
                levels = record.get("levels") or []
                if not levels:
                    quality["missing_levels"] += 1
                    continue
                level = levels[0]
                raw_bid = int(level["bid_px"])
                raw_ask = int(level["ask_px"])
                if raw_bid >= 9_000_000_000_000_000_000 or raw_ask >= 9_000_000_000_000_000_000:
                    quality["invalid_price"] += 1
                    continue
                bid = raw_bid / PRICE_SCALE
                ask = raw_ask / PRICE_SCALE
                if bid <= 0 or ask <= 0:
                    quality["invalid_price"] += 1
                    continue
                if bid > ask:
                    quality["crossed_quotes"] += 1
                    continue
                symbol = symbol_map.get(instrument_id)
                if symbol is None:
                    quality["unmapped_instrument"] += 1
                    continue
                rows.append(
                    {
                        "ts_event": header.get("ts_event"),
                        "ts_recv": record.get("ts_recv"),
                        "instrument_id": instrument_id,
                        "symbol": symbol,
                        "sequence": record.get("sequence"),
                        "bid_price": bid,
                        "ask_price": ask,
                        "bid_size": level.get("bid_sz"),
                        "ask_size": level.get("ask_sz"),
                    }
                )
                quality["rows"] += 1
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                quality["invalid_json"] += 1
    return rows, quality


def normalize_status_file(path: Path, symbol_map: dict[int, str]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows: list[dict[str, Any]] = []
    quality: Counter[str] = Counter()
    for record in _json_lines(path):
        header = record.get("hd", {})
        instrument_id = header.get("instrument_id")
        if instrument_id is None:
            quality["invalid_instrument"] += 1
            continue
        symbol = symbol_map.get(int(instrument_id))
        if symbol is None:
            quality["unmapped_instrument"] += 1
            continue
        rows.append(
            {
                "ts_event": header.get("ts_event"),
                "ts_recv": record.get("ts_recv"),
                "instrument_id": int(instrument_id),
                "symbol": symbol,
                "action": record.get("action"),
                "reason": record.get("reason"),
                "trading_event": record.get("trading_event"),
                "is_trading": record.get("is_trading"),
                "is_quoting": record.get("is_quoting"),
                "is_short_sell_restricted": record.get("is_short_sell_restricted"),
            }
        )
    quality["rows"] = len(rows)
    return rows, dict(quality)


def _request_url(endpoint: str, params: dict[str, str]) -> str:
    return f"{API_ROOT}/{endpoint}?{urlencode(params)}"


def download_jsonl(
    *,
    api_key: str,
    endpoint: str,
    params: dict[str, str],
    output: Path,
    force: bool = False,
) -> None:
    if output.exists() and output.stat().st_size > 0 and not force:
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    token = base64.b64encode(f"{api_key}:".encode()).decode()
    request = Request(
        _request_url(endpoint, params),
        headers={"Authorization": f"Basic {token}", "Accept": "application/json"},
    )
    temporary = output.with_suffix(output.suffix + ".part")
    for attempt in range(3):
        try:
            with urlopen(request, timeout=900) as response, gzip.open(temporary, "wb") as handle:
                while chunk := response.read(1024 * 1024):
                    handle.write(chunk)
            break
        except (HTTPError, URLError, TimeoutError, OSError):
            temporary.unlink(missing_ok=True)
            if attempt == 2:
                raise
            time.sleep(2**attempt)
    temporary.replace(output)


def _first_weekday(year: int, month: int) -> date:
    current = date(year, month, 1)
    while current.weekday() >= 5:
        current += timedelta(days=1)
    if current.month == 1 and current.day == 1:
        current += timedelta(days=1)
        while current.weekday() >= 5:
            current += timedelta(days=1)
    return current


def monthly_windows(start: date, end: date) -> list[tuple[date, datetime, datetime]]:
    current = date(start.year, start.month, 1)
    result: list[tuple[date, datetime, datetime]] = []
    while current < end:
        session_date = _first_weekday(current.year, current.month)
        if start <= session_date < end:
            local_open = datetime.combine(session_date, datetime.min.time(), tzinfo=EASTERN).replace(
                hour=9, minute=30
            )
            result.append((session_date, local_open.astimezone(UTC), (local_open + timedelta(minutes=30)).astimezone(UTC)))
        current = date(current.year + (current.month == 12), 1 if current.month == 12 else current.month + 1, 1)
    return result


def download_monthly_windows(output_dir: Path, start: date, end: date, force: bool = False) -> list[str]:
    api_key = os.environ.get("DATABENTO_API_KEY")
    if not api_key:
        raise RuntimeError("DATABENTO_API_KEY is not set")
    completed: list[str] = []
    for session_date, start_utc, end_utc in monthly_windows(start, end):
        day = session_date.isoformat()
        window = {"start": start_utc.isoformat(), "end": end_utc.isoformat()}
        base = {"dataset": "EQUS.MINI", "stype_in": "raw_symbol", "symbols": "ALL_SYMBOLS"}
        download_jsonl(
            api_key=api_key,
            endpoint="timeseries.get_range",
            params={**base, "schema": "bbo-1m", "encoding": "json", **window},
            output=output_dir / "raw" / "bbo" / f"{day}.jsonl.gz",
            force=force,
        )
        full_day = {"start": f"{day}T00:00:00Z", "end": f"{(session_date + timedelta(days=1)).isoformat()}T00:00:00Z"}
        download_jsonl(
            api_key=api_key,
            endpoint="timeseries.get_range",
            params={**base, "schema": "definition", "encoding": "json", **full_day},
            output=output_dir / "raw" / "definition" / f"{day}.jsonl.gz",
            force=force,
        )
        status_base = {"dataset": "DBEQ.BASIC", "schema": "status", "stype_in": "raw_symbol", "symbols": "ALL_SYMBOLS"}
        download_jsonl(
            api_key=api_key,
            endpoint="timeseries.get_range",
            params={**status_base, "encoding": "json", **full_day},
            output=output_dir / "raw" / "status" / f"{day}.jsonl.gz",
            force=force,
        )
        completed.append(day)
    return completed


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fields} for row in rows)


def normalize_download(output_dir: Path) -> dict[str, Any]:
    bbo_rows: list[dict[str, Any]] = []
    status_rows: list[dict[str, Any]] = []
    quality: dict[str, Any] = {"bbo": {}, "status": {}}
    for definition_path in sorted((output_dir / "raw" / "definition").glob("*.jsonl.gz")):
        day = definition_path.stem.split(".", 1)[0]
        symbol_map = load_definition_map(definition_path)
        bbo_path = output_dir / "raw" / "bbo" / f"{day}.jsonl.gz"
        status_path = output_dir / "raw" / "status" / f"{day}.jsonl.gz"
        if bbo_path.exists():
            rows, report = normalize_bbo_file(bbo_path, symbol_map)
            bbo_rows.extend(rows)
            quality["bbo"][day] = report
        if status_path.exists():
            rows, report = normalize_status_file(status_path, symbol_map)
            status_rows.extend(rows)
            quality["status"][day] = report
    _write_csv(
        output_dir / "normalized" / "bbo.csv",
        bbo_rows,
        ["ts_event", "ts_recv", "instrument_id", "symbol", "sequence", "bid_price", "ask_price", "bid_size", "ask_size"],
    )
    _write_csv(
        output_dir / "normalized" / "status.csv",
        status_rows,
        ["ts_event", "ts_recv", "instrument_id", "symbol", "action", "reason", "trading_event", "is_trading", "is_quoting", "is_short_sell_restricted"],
    )
    quality["bbo_total_rows"] = len(bbo_rows)
    quality["status_total_rows"] = len(status_rows)
    (output_dir / "normalized" / "quality_report.json").write_text(
        json.dumps(quality, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return quality


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    download = subparsers.add_parser("download")
    download.add_argument("--output-dir", type=Path, required=True)
    download.add_argument("--start", type=date.fromisoformat, default=date(2023, 4, 1))
    download.add_argument("--end", type=date.fromisoformat, default=date.today())
    download.add_argument("--force", action="store_true")
    normalize = subparsers.add_parser("normalize")
    normalize.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "download":
        days = download_monthly_windows(args.output_dir, args.start, args.end, args.force)
        print(f"downloaded_windows={len(days)} output={args.output_dir}")
    else:
        report = normalize_download(args.output_dir)
        print(f"bbo_rows={report['bbo_total_rows']} status_rows={report['status_total_rows']} output={args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
