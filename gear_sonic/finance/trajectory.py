"""Variable-length monthly finance trajectories and their on-disk index.

The trajectory format mirrors the file-per-motion organization used by Sonic,
while keeping finance-specific monthly fields and provenance.  The canonical
CSV remains the source of truth; this module only creates a new output tree.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import re
import shutil
import sys
import time
from typing import Callable, Iterable, Mapping

import numpy as np


TRAJECTORY_SCHEMA_VERSION = 1
TRAJECTORY_KIND = "finance_monthly_trajectory"
CADENCE = "1M"
DEFAULT_MIN_TRAINING_LENGTH = 17
WARMUP_MONTHS = 6
HORIZON_MONTHS = 10

RAW_FIELDS = ("open", "close", "high", "low", "volume", "amount", "turnover")
ADJUSTED_FIELDS = ("open", "close", "high", "low")
VALIDITY_FIELDS = ("raw", "qfq", "hfq")
DATE_FIELDS = ("raw", "qfq", "hfq")
EVENT_STATUSES = frozenset({"confirmed", "review", "rejected"})
EVENT_ACTIONS = frozenset({"split_before", "exclude"})

CANONICAL_FIELDS = (
    "symbol", "period", "raw_date", "raw_open", "raw_close", "raw_high", "raw_low",
    "raw_volume", "raw_amount", "raw_turnover", "qfq_date", "qfq_open", "qfq_close",
    "qfq_high", "qfq_low", "hfq_date", "hfq_open", "hfq_close", "hfq_high", "hfq_low",
    "raw_valid", "qfq_valid", "hfq_valid", "quality_flags",
)


@dataclass(frozen=True, slots=True)
class EventBreak:
    """A reviewed boundary in a symbol's chronological history."""

    symbol: str
    period: str
    reason: str
    status: str = "confirmed"
    action: str = "split_before"
    reference: str = ""

    def __post_init__(self) -> None:
        if not self.symbol.strip():
            raise ValueError("Event symbol must be nonempty")
        _month_number(self.period)
        if not self.reason.strip():
            raise ValueError("Event reason must be nonempty")
        if self.status not in EVENT_STATUSES:
            raise ValueError(f"Unsupported event status: {self.status}")
        if self.action not in EVENT_ACTIONS:
            raise ValueError(f"Unsupported event action: {self.action}")


def _normalize_statuses(statuses: Iterable[str] | str) -> tuple[str, ...]:
    """Normalize status arguments without treating one string as characters."""
    if isinstance(statuses, str):
        normalized = (statuses.strip(),) if statuses.strip() else ()
    else:
        normalized = tuple(str(status).strip() for status in statuses if str(status).strip())
    if not set(normalized).issubset(EVENT_STATUSES):
        raise ValueError(f"Unsupported applied event status: {normalized}")
    if "rejected" in normalized:
        raise ValueError("Rejected events cannot be applied")
    return normalized


@dataclass
class TrajectorySegment:
    """One continuous, retained monthly run."""

    symbol: str
    segment_id: int
    rows: list[dict]
    break_before_reason: str | None = None
    break_after_reason: str | None = None


def _month_number(period: str) -> int:
    match = re.fullmatch(r"(\d{4})-(\d{2})", str(period).strip())
    if match is None or not 1 <= int(match.group(2)) <= 12:
        raise ValueError(f"Invalid monthly period: {period!r}")
    return int(match.group(1)) * 12 + int(match.group(2))


def _as_float(value: object) -> float:
    text = "" if value is None else str(value).strip()
    if not text:
        return math.nan
    try:
        value = float(text)
        return value if math.isfinite(value) else math.nan
    except (TypeError, ValueError):
        return math.nan


def _valid_ohlc(values: Mapping[str, float]) -> bool:
    if not all(math.isfinite(values[name]) and values[name] > 0 for name in ADJUSTED_FIELDS):
        return False
    return (
        values["high"] >= values["low"]
        and values["high"] >= max(values["open"], values["close"])
        and values["low"] <= min(values["open"], values["close"])
    )


def _parse_validity(value: object) -> bool:
    text = str(value).strip().casefold()
    if text in {"1", "true"}:
        return True
    if text in {"", "0", "false"}:
        return False
    try:
        return float(text) == 1.0
    except ValueError:
        return False


def parse_canonical_row(row: Mapping[str, object]) -> tuple[dict | None, str | None]:
    """Validate and convert one canonical row.

    ``raw_amount`` and ``raw_turnover`` are optional provenance fields.  Their
    non-finite values are retained as NaN rather than invalidating a price bar.
    """
    symbol = str(row.get("symbol", "")).strip()
    period = str(row.get("period", "")).strip()
    if not symbol:
        return None, "empty_symbol"
    try:
        _month_number(period)
    except ValueError:
        return None, "invalid_period"
    if not (_parse_validity(row.get("raw_valid")) and _parse_validity(row.get("qfq_valid"))):
        return None, "raw_or_qfq_invalid"

    raw = {name: _as_float(row.get(f"raw_{name}")) for name in ADJUSTED_FIELDS}
    qfq = {name: _as_float(row.get(f"qfq_{name}")) for name in ADJUSTED_FIELDS}
    if not _valid_ohlc(raw) or not _valid_ohlc(qfq):
        return None, "raw_or_qfq_ohlc_invalid"
    raw_volume = _as_float(row.get("raw_volume"))
    if not math.isfinite(raw_volume) or raw_volume <= 0:
        return None, "nonpositive_volume"

    hfq = {name: _as_float(row.get(f"hfq_{name}")) for name in ADJUSTED_FIELDS}
    hfq_valid = _parse_validity(row.get("hfq_valid")) and _valid_ohlc(hfq)
    if not hfq_valid:
        hfq = {name: math.nan for name in ADJUSTED_FIELDS}

    source_line = row.get("_source_line")
    try:
        source_line = int(source_line) if source_line is not None else None
    except (TypeError, ValueError):
        source_line = None
    parsed = {
        "symbol": symbol,
        "period": period,
        "dates": {name: str(row.get(f"{name}_date", "") or "").strip() for name in DATE_FIELDS},
        "raw": [raw[name] for name in ADJUSTED_FIELDS]
        + [raw_volume, _as_float(row.get("raw_amount")), _as_float(row.get("raw_turnover"))],
        "qfq": [qfq[name] for name in ADJUSTED_FIELDS],
        "hfq": [hfq[name] for name in ADJUSTED_FIELDS],
        "validity": [True, True, hfq_valid],
        "quality_flags": str(row.get("quality_flags", "") or "").strip(),
        "source_row_number": source_line,
    }
    return parsed, None


def read_event_breaks(path: str | Path | None) -> dict[tuple[str, str], EventBreak]:
    """Read a CSV event list with ``symbol,period,reason`` columns."""
    if path is None:
        return {}
    path = Path(path)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"symbol", "period", "reason"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError("Event CSV must contain symbol, period, and reason columns")
        result: dict[tuple[str, str], EventBreak] = {}
        for line_number, row in enumerate(reader, start=2):
            event = EventBreak(
                str(row.get("symbol", "")).strip(),
                str(row.get("period", "")).strip(),
                str(row.get("reason", "")).strip(),
                str(row.get("status", "confirmed") or "confirmed").strip(),
                str(row.get("action", "split_before") or "split_before").strip(),
                str(row.get("reference", "") or "").strip(),
            )
            key = (event.symbol, event.period)
            if key in result:
                raise ValueError(f"Duplicate event at {key} (line {line_number})")
            result[key] = event
    return result


def iter_trajectory_segments(
    rows: Iterable[Mapping[str, object]],
    *,
    event_breaks: Mapping[tuple[str, str], EventBreak] | None = None,
    apply_statuses: Iterable[str] | str = ("confirmed",),
    on_exclusion: Callable[[dict], None] | None = None,
    on_event: Callable[[EventBreak], None] | None = None,
):
    """Yield continuous segments from canonical-style rows.

    Rows must be grouped by symbol.  A bad row closes the current segment and
    is reported through ``on_exclusion``.  Event boundaries are opt-in by
    status, so provisional audit findings cannot silently alter the dataset.
    """
    event_breaks = dict(event_breaks or {})
    for key, event in event_breaks.items():
        if not isinstance(event, EventBreak):
            raise TypeError(f"Event mapping value must be EventBreak: {key!r}")
        expected_key = (event.symbol, event.period)
        if key != expected_key:
            raise ValueError(f"Event mapping key {key!r} does not match {expected_key!r}")
    statuses = _normalize_statuses(apply_statuses)

    current_symbol: str | None = None
    current_rows: list[dict] = []
    current_segment_id: int | None = None
    current_break_before: str | None = None
    last_period_number: int | None = None
    seen_symbols: set[str] = set()
    segment_counts: dict[str, int] = {}

    def flush(after_reason: str | None):
        nonlocal current_rows, current_segment_id, current_break_before, last_period_number
        if not current_rows:
            return None
        segment = TrajectorySegment(
            symbol=current_symbol,  # type: ignore[arg-type]
            segment_id=current_segment_id,  # type: ignore[arg-type]
            rows=current_rows,
            break_before_reason=current_break_before,
            break_after_reason=after_reason,
        )
        current_rows = []
        current_segment_id = None
        current_break_before = None
        last_period_number = None
        return segment

    def exclude(row: Mapping[str, object], reason: str) -> None:
        if on_exclusion is not None:
            on_exclusion({
                "source_row_number": row.get("_source_line"),
                "symbol": str(row.get("symbol", "")).strip(),
                "period": str(row.get("period", "")).strip(),
                "reason": reason,
                "quality_flags": str(row.get("quality_flags", "") or "").strip(),
            })

    for row in rows:
        raw_symbol = str(row.get("symbol", "")).strip()
        if not raw_symbol:
            previous = flush("excluded:empty_symbol")
            if previous is not None:
                yield previous
            exclude(row, "empty_symbol")
            # Keep the current group identity: an empty malformed row should
            # split the run, but must not make a later row for the same symbol
            # look like a non-contiguous symbol reappearance.
            last_period_number = None
            current_break_before = "excluded:empty_symbol"
            continue
        if current_symbol != raw_symbol:
            # A symbol transition is ordinary CSV grouping, not an anomaly boundary.
            previous = flush(None)
            if previous is not None:
                yield previous
            if raw_symbol in seen_symbols:
                raise ValueError(f"Canonical rows must be grouped by symbol: {raw_symbol}")
            if current_symbol is not None:
                current_break_before = None
            current_symbol = raw_symbol
            seen_symbols.add(raw_symbol)
            last_period_number = None

        parsed, error = parse_canonical_row(row)
        if parsed is None:
            previous = flush(f"excluded:{error}")
            if previous is not None:
                yield previous
            exclude(row, error or "invalid_row")
            last_period_number = None
            current_break_before = f"excluded:{error}"
            continue

        period = parsed["period"]
        event = event_breaks.get((parsed["symbol"], period))
        if event is not None and event.status in statuses:
            if on_event is not None:
                on_event(event)
            boundary_reason = f"event:{event.reason}"
            previous = flush(boundary_reason)
            if previous is not None:
                yield previous
            current_break_before = boundary_reason
            if event.action == "exclude":
                exclude(row, f"event_excluded:{event.reason}")
                last_period_number = None
                continue

        period_number = _month_number(period)
        if last_period_number is not None:
            delta = period_number - last_period_number
            if delta == 0:
                boundary_reason = "duplicate_period"
                previous = flush(boundary_reason)
                if previous is not None:
                    yield previous
                exclude(row, boundary_reason)
                current_break_before = f"excluded:{boundary_reason}"
                last_period_number = None
                continue
            if delta < 0:
                boundary_reason = "out_of_order_period"
                previous = flush(boundary_reason)
                if previous is not None:
                    yield previous
                exclude(row, boundary_reason)
                current_break_before = f"excluded:{boundary_reason}"
                last_period_number = None
                continue
            if delta > 1:
                boundary_reason = f"month_gap:{delta}"
                previous = flush(boundary_reason)
                if previous is not None:
                    yield previous
                current_break_before = boundary_reason

        if not current_rows:
            current_segment_id = segment_counts.get(parsed["symbol"], 0)
            segment_counts[parsed["symbol"]] = current_segment_id + 1
        current_rows.append(parsed)
        last_period_number = period_number

    previous = flush(None)
    if previous is not None:
        yield previous


def _string_array(values: Iterable[str]) -> np.ndarray:
    values = list(values)
    width = max([len(value) for value in values] + [1])
    return np.asarray(values, dtype=f"U{width}")


def trajectory_payload(segment: TrajectorySegment, key: str) -> dict[str, dict]:
    """Convert a segment to a mapping-wrapped, pickleable payload."""
    rows = segment.rows
    entry = {
        "schema_version": TRAJECTORY_SCHEMA_VERSION,
        "kind": TRAJECTORY_KIND,
        "symbol": segment.symbol,
        "segment_id": segment.segment_id,
        "cadence": CADENCE,
        "length": len(rows),
        "start_period": rows[0]["period"],
        "end_period": rows[-1]["period"],
        "break_before_reason": segment.break_before_reason,
        "break_after_reason": segment.break_after_reason,
        "periods": _string_array(row["period"] for row in rows),
        "dates": {
            name: _string_array(row["dates"][name] for row in rows) for name in DATE_FIELDS
        },
        "raw": np.asarray([row["raw"] for row in rows], dtype=np.float64),
        "qfq": np.asarray([row["qfq"] for row in rows], dtype=np.float64),
        "hfq": np.asarray([row["hfq"] for row in rows], dtype=np.float64),
        "validity": np.asarray([row["validity"] for row in rows], dtype=np.bool_),
        "quality_flags": tuple(row["quality_flags"] for row in rows),
        # Keep an explicit per-row name for downstream audit code; quality_flags
        # remains as the backwards-compatible field used by early consumers.
        "row_quality_flags": tuple(row["quality_flags"] for row in rows),
        "source_row_numbers": np.asarray(
            [row["source_row_number"] if row["source_row_number"] is not None else -1 for row in rows],
            dtype=np.int64,
        ),
        "field_names": {
            "raw": RAW_FIELDS,
            "qfq": ADJUSTED_FIELDS,
            "hfq": ADJUSTED_FIELDS,
            "validity": VALIDITY_FIELDS,
        },
        "feature_policy": "derive_current_and_future_features_at_load_time",
    }
    return {key: entry}


def load_trajectory(path: str | Path) -> tuple[str, dict]:
    """Load and minimally validate one mapping-wrapped trajectory PKL."""
    with Path(path).open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict) or len(payload) != 1:
        raise ValueError(f"Trajectory PKL must contain one mapping entry: {path}")
    key, entry = next(iter(payload.items()))
    if not isinstance(key, str) or not isinstance(entry, dict):
        raise ValueError(f"Trajectory PKL has invalid mapping types: {path}")
    required = {
        "schema_version", "kind", "cadence", "symbol", "segment_id", "length",
        "start_period", "end_period", "periods", "dates", "raw", "qfq", "hfq",
        "validity", "row_quality_flags", "source_row_numbers", "field_names",
    }
    if not required.issubset(entry):
        raise ValueError(f"Trajectory PKL is missing fields: {path}")
    if entry["schema_version"] != TRAJECTORY_SCHEMA_VERSION:
        raise ValueError(f"Unsupported trajectory schema version: {path}")
    if entry["kind"] != TRAJECTORY_KIND or entry["cadence"] != CADENCE:
        raise ValueError(f"Trajectory kind or cadence is invalid: {path}")
    symbol = entry["symbol"]
    if not isinstance(symbol, str) or not symbol.strip():
        raise ValueError(f"Trajectory symbol must be a nonempty string: {path}")
    segment_id = entry["segment_id"]
    if isinstance(segment_id, bool) or not isinstance(segment_id, (int, np.integer)) or segment_id < 0:
        raise ValueError(f"Trajectory segment_id must be a nonnegative integer: {path}")
    try:
        periods = np.asarray(entry["periods"])
        raw = np.asarray(entry["raw"])
        qfq = np.asarray(entry["qfq"])
        hfq = np.asarray(entry["hfq"])
        validity = np.asarray(entry["validity"])
    except Exception as exc:  # pragma: no cover - defensive boundary for malformed pickle input
        raise ValueError(f"Trajectory arrays are not readable: {path}") from exc
    if periods.ndim != 1 or not len(periods):
        raise ValueError(f"Periods must be a nonempty vector: {path}")
    if periods.dtype.kind != "U":
        raise ValueError(f"Periods must use a fixed-width Unicode dtype: {path}")
    period_values = [str(value) for value in periods.tolist()]
    try:
        period_numbers = [_month_number(value) for value in period_values]
    except ValueError as exc:
        raise ValueError(f"Trajectory contains an invalid period: {path}") from exc
    if any(right - left != 1 for left, right in zip(period_numbers, period_numbers[1:])):
        raise ValueError(f"Trajectory periods are not contiguous: {path}")
    length = len(periods)
    try:
        declared_length = int(entry["length"])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Trajectory length metadata is invalid: {path}") from exc
    if declared_length != length:
        raise ValueError(f"Trajectory length metadata does not match periods: {path}")
    if entry["start_period"] != period_values[0] or entry["end_period"] != period_values[-1]:
        raise ValueError(f"Trajectory start/end metadata does not match periods: {path}")
    if raw.shape != (length, len(RAW_FIELDS)):
        raise ValueError(f"Raw shape does not match periods: {path}")
    if qfq.shape != (length, len(ADJUSTED_FIELDS)):
        raise ValueError(f"QFQ shape does not match periods: {path}")
    if hfq.shape != (length, len(ADJUSTED_FIELDS)):
        raise ValueError(f"HFQ shape does not match periods: {path}")
    if validity.shape != (length, len(VALIDITY_FIELDS)):
        raise ValueError(f"Validity shape does not match periods: {path}")
    if raw.dtype != np.float64 or qfq.dtype != np.float64 or hfq.dtype != np.float64:
        raise ValueError(f"Price arrays must use float64: {path}")
    if validity.dtype != np.bool_:
        raise ValueError(f"Validity array must use bool: {path}")
    if not np.all(validity[:, :2]):
        raise ValueError(f"Raw/QFQ validity cannot be false in a trajectory: {path}")
    if not np.isfinite(raw[:, :5]).all() or (raw[:, :5] <= 0).any():
        raise ValueError(f"Raw OHLCV contains non-finite or nonpositive values: {path}")
    if np.isinf(raw[:, 5:]).any():
        raise ValueError(f"Raw amount/turnover contains infinite values: {path}")
    if not np.isfinite(qfq).all() or (qfq <= 0).any():
        raise ValueError(f"QFQ OHLC contains non-finite or nonpositive values: {path}")
    for name, values in (("raw", raw[:, :4]), ("qfq", qfq)):
        if (values[:, 2] < values[:, 3]).any():
            raise ValueError(f"{name.upper()} OHLC ordering is invalid: {path}")
        if (values[:, 2] < values[:, :2].max(axis=1)).any():
            raise ValueError(f"{name.upper()} OHLC ordering is invalid: {path}")
        if (values[:, 3] > values[:, :2].min(axis=1)).any():
            raise ValueError(f"{name.upper()} OHLC ordering is invalid: {path}")
    hfq_valid = validity[:, 2]
    if hfq_valid.any():
        if not np.isfinite(hfq[hfq_valid]).all() or (hfq[hfq_valid] <= 0).any():
            raise ValueError(f"HFQ valid rows contain invalid values: {path}")
    if (~hfq_valid).any() and not np.isnan(hfq[~hfq_valid]).all():
        raise ValueError(f"HFQ invalid rows must contain NaN: {path}")
    if len(entry["row_quality_flags"]) != length:
        raise ValueError(f"Row quality flags do not match periods: {path}")
    source_row_numbers = np.asarray(entry["source_row_numbers"])
    if source_row_numbers.shape != (length,) or source_row_numbers.dtype.kind not in {"i", "u"}:
        raise ValueError(f"Source row numbers do not match periods: {path}")
    dates = entry["dates"]
    if not isinstance(dates, Mapping) or set(dates) != set(DATE_FIELDS):
        raise ValueError(f"Trajectory dates mapping is invalid: {path}")
    for date_name in DATE_FIELDS:
        date_values = np.asarray(dates[date_name])
        if date_values.shape != (length,) or date_values.dtype.kind != "U":
            raise ValueError(f"Trajectory {date_name} dates do not match periods: {path}")
    field_names = entry["field_names"]
    expected_fields = {
        "raw": RAW_FIELDS,
        "qfq": ADJUSTED_FIELDS,
        "hfq": ADJUSTED_FIELDS,
        "validity": VALIDITY_FIELDS,
    }
    if not isinstance(field_names, Mapping) or any(
        tuple(field_names.get(name, ())) != fields for name, fields in expected_fields.items()
    ):
        raise ValueError(f"Trajectory field names do not match the schema: {path}")
    expected_key = f"{_safe_symbol(symbol)}__segment_{int(segment_id):03d}"
    if key != expected_key:
        raise ValueError(f"Trajectory key does not match symbol and segment_id: {path}")
    return key, entry


def _safe_symbol(symbol: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", symbol).strip("._") or "symbol"
    if slug != symbol:
        slug += "__" + hashlib.sha1(symbol.encode("utf-8")).hexdigest()[:8]
    return slug


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_fingerprint(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "inode": stat.st_ino,
        "sha256": _hash_file(path),
    }


def _csv_rows(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not set(CANONICAL_FIELDS).issubset(reader.fieldnames):
            raise ValueError("Canonical CSV header is missing required fields")
        for row in reader:
            row["_source_line"] = reader.line_num
            yield row


def _dump_pickle(value: object, path: Path) -> None:
    with path.open("wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)


def reconstruct_monthly_trajectories(
    canonical: str | Path,
    output_dir: str | Path,
    *,
    event_breaks: str | Path | Mapping[tuple[str, str], EventBreak] | None = None,
    apply_statuses: Iterable[str] | str = ("confirmed",),
    min_training_length: int = DEFAULT_MIN_TRAINING_LENGTH,
    dry_run: bool = False,
) -> dict:
    """Create a fresh variable-length trajectory tree from canonical CSV."""
    canonical = Path(canonical).resolve(strict=True)
    output_dir = Path(output_dir).resolve()
    minimum_complete_length = WARMUP_MONTHS + HORIZON_MONTHS + 1
    if min_training_length < minimum_complete_length:
        raise ValueError(
            f"min_training_length must be at least {minimum_complete_length} "
            f"for a complete {WARMUP_MONTHS}-month warmup and {HORIZON_MONTHS}-month horizon"
        )
    if not dry_run and output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_dir}")
    event_breaks_path: Path | None = None
    event_source_before: dict[str, object] | None = None
    if isinstance(event_breaks, Mapping):
        events = dict(event_breaks)
    else:
        if event_breaks is not None:
            event_breaks_path = Path(event_breaks).resolve(strict=True)
            event_source_before = _source_fingerprint(event_breaks_path)
        events = read_event_breaks(event_breaks)
        if event_source_before is not None and event_source_before != _source_fingerprint(event_breaks_path):
            raise RuntimeError("Event-break source changed while it was being read")
    for key, event in events.items():
        if not isinstance(event, EventBreak):
            raise TypeError(f"Event mapping value must be EventBreak: {key!r}")
        expected_key = (event.symbol, event.period)
        if key != expected_key:
            raise ValueError(f"Event mapping key {key!r} does not match {expected_key!r}")
    statuses = _normalize_statuses(apply_statuses)
    source_before = _source_fingerprint(canonical)
    started = time.perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()
    counts = {
        "source_rows": 0,
        "retained_rows": 0,
        "excluded_rows": 0,
        "segment_count": 0,
        "eligible_segment_count": 0,
        "complete_anchor_count": 0,
        "eligible_anchor_count": 0,
    }
    source_symbols: set[str] = set()
    symbols: set[str] = set()
    event_applied: set[tuple[str, str]] = set()
    used_keys: dict[str, str] = {}
    temp_dir: Path | None = None
    metadata: dict[str, dict] = {}
    index_handle = exclusion_handle = None
    index_writer = exclusion_writer = None
    try:
        if not dry_run:
            output_dir.parent.mkdir(parents=True, exist_ok=True)
            temp_dir = output_dir.parent / f".{output_dir.name}.tmp-{os.getpid()}"
            if temp_dir.exists():
                raise FileExistsError(f"Temporary output already exists: {temp_dir}")
            (temp_dir / "trajectories").mkdir(parents=True)
            index_handle = (temp_dir / "trajectory_index.csv").open("w", encoding="utf-8", newline="")
            exclusion_handle = (temp_dir / "excluded_rows.csv").open("w", encoding="utf-8", newline="")
            index_writer = csv.DictWriter(index_handle, fieldnames=[
                "key", "path", "symbol", "segment_id", "start_period", "end_period", "length",
                "complete_anchor_count", "eligible_anchor_count",
                "break_before_reason", "break_after_reason",
            ])
            exclusion_writer = csv.DictWriter(
                exclusion_handle,
                fieldnames=["source_row_number", "symbol", "period", "reason", "quality_flags"],
            )
            index_writer.writeheader()
            exclusion_writer.writeheader()

        def close_output_handles() -> None:
            nonlocal index_handle, exclusion_handle
            for handle in (index_handle, exclusion_handle):
                if handle is not None and not handle.closed:
                    handle.flush()
                    os.fsync(handle.fileno())
                    handle.close()
            index_handle = None
            exclusion_handle = None

        def on_exclusion(item: dict) -> None:
            counts["excluded_rows"] += 1
            if exclusion_writer is not None:
                exclusion_writer.writerow(item)

        def on_event(event: EventBreak) -> None:
            event_applied.add((event.symbol, event.period))

        def rows_with_count():
            for row in _csv_rows(canonical):
                counts["source_rows"] += 1
                symbol = str(row.get("symbol", "")).strip()
                if symbol:
                    source_symbols.add(symbol)
                yield row

        for segment in iter_trajectory_segments(
            rows_with_count(), event_breaks=events, apply_statuses=statuses,
            on_exclusion=on_exclusion, on_event=on_event,
        ):
            symbols.add(segment.symbol)
            length = len(segment.rows)
            counts["retained_rows"] += length
            counts["segment_count"] += 1
            complete_anchors = max(0, length - (WARMUP_MONTHS + HORIZON_MONTHS))
            training_eligible = length >= min_training_length
            eligible_anchors = complete_anchors if training_eligible else 0
            counts["complete_anchor_count"] += complete_anchors
            if training_eligible:
                counts["eligible_segment_count"] += 1
            counts["eligible_anchor_count"] += eligible_anchors
            if index_writer is None:
                continue
            key = f"{_safe_symbol(segment.symbol)}__segment_{segment.segment_id:03d}"
            if key in used_keys:
                previous_symbol = used_keys[key]
                raise RuntimeError(
                    f"Duplicate trajectory key: {key!r} already represents {previous_symbol!r}"
                )
            used_keys[key] = segment.symbol
            relative_path = Path("trajectories") / f"{key}.pkl"
            _dump_pickle(trajectory_payload(segment, key), temp_dir / relative_path)  # type: ignore[arg-type]
            metadata[key] = {
                "schema_version": TRAJECTORY_SCHEMA_VERSION,
                "kind": TRAJECTORY_KIND,
                "symbol": segment.symbol,
                "segment_id": segment.segment_id,
                "length": length,
                "cadence": CADENCE,
                "start_period": segment.rows[0]["period"],
                "end_period": segment.rows[-1]["period"],
                "duration_months": length - 1,
                "complete_anchor_count": complete_anchors,
                "eligible_anchor_count": eligible_anchors,
                "training_eligible": training_eligible,
                "source_relpath": str(relative_path),
                "source_row_start": segment.rows[0]["source_row_number"],
                "source_row_end": segment.rows[-1]["source_row_number"],
                "break_before_reason": segment.break_before_reason,
                "break_after_reason": segment.break_after_reason,
            }
            index_writer.writerow({
                "key": key,
                "path": str(relative_path),
                "symbol": segment.symbol,
                "segment_id": segment.segment_id,
                "start_period": segment.rows[0]["period"],
                "end_period": segment.rows[-1]["period"],
                "length": length,
                "complete_anchor_count": complete_anchors,
                "eligible_anchor_count": eligible_anchors,
                "break_before_reason": segment.break_before_reason or "",
                "break_after_reason": segment.break_after_reason or "",
            })

        if not dry_run:
            # Flush and fsync indexes before checking the source and publishing.
            close_output_handles()
        source_after = _source_fingerprint(canonical)
        if source_before != source_after:
            raise RuntimeError("Canonical source changed during trajectory reconstruction")
        if event_source_before is not None and event_source_before != _source_fingerprint(event_breaks_path):
            raise RuntimeError("Event-break source changed during trajectory reconstruction")
        selected_event_keys = {key for key, event in events.items() if event.status in statuses}
        event_records = []
        for key, event in sorted(events.items()):
            selected = key in selected_event_keys
            applied = key in event_applied
            event_records.append({
                "symbol": event.symbol,
                "period": event.period,
                "reason": event.reason,
                "status": event.status,
                "action": event.action,
                "reference": event.reference,
                "selected": selected,
                "outcome": "applied" if applied else ("unmatched" if selected else "ignored_status"),
            })
        manifest = {
            "status": "complete",
            "schema_version": TRAJECTORY_SCHEMA_VERSION,
            "schema": TRAJECTORY_KIND,
            "serialization": {
                "python_version": sys.version.split()[0],
                "numpy_version": np.__version__,
                "pickle_protocol": pickle.HIGHEST_PROTOCOL,
            },
            "payload_schema": {
                "periods": {"dtype": "unicode", "format": "YYYY-MM"},
                "raw": {"dtype": "float64", "fields": list(RAW_FIELDS)},
                "qfq": {"dtype": "float64", "fields": list(ADJUSTED_FIELDS)},
                "hfq": {
                    "dtype": "float64", "fields": list(ADJUSTED_FIELDS),
                    "invalid_row_value": "NaN",
                },
                "validity": {"dtype": "bool", "fields": list(VALIDITY_FIELDS)},
            },
            "source": source_before,
            "segmentation": {
                "grouping": "symbol_then_contiguous_month",
                "validity": "raw_valid_and_qfq_valid_with_positive_ohlcv",
                "gap_policy": "close_segment_and_start_new_segment",
                "hfq_policy": "audit_only_optional",
                "min_training_length": min_training_length,
                "minimum_complete_length": minimum_complete_length,
                "warmup_months": WARMUP_MONTHS,
                "horizon_months": HORIZON_MONTHS,
            },
            "event_breaks": {
                "supplied": len(events),
                "applied_statuses": list(statuses),
                "source": (
                    {"path": str(event_breaks_path), "sha256": event_source_before["sha256"]}
                    if event_source_before is not None else None
                ),
                "records": event_records,
                "applied": [record for record in event_records if record["outcome"] == "applied"],
                "selected_unseen": [
                    {"symbol": symbol, "period": period}
                    for symbol, period in sorted(selected_event_keys - event_applied)
                ],
            },
            "counts": {
                **counts,
                "source_symbol_count": len(source_symbols),
                "retained_symbol_count": len(symbols),
                "excluded_only_symbol_count": len(source_symbols - symbols),
            },
            "source_rows": counts["source_rows"],
            "retained_rows": counts["retained_rows"],
            "excluded_rows": counts["excluded_rows"],
            "source_symbol_count": len(source_symbols),
            "retained_symbol_count": len(symbols),
            "excluded_only_symbol_count": len(source_symbols - symbols),
            "symbol_count": len(symbols),
            "segment_count": counts["segment_count"],
            "eligible_segment_count": counts["eligible_segment_count"],
            "complete_anchor_count": counts["complete_anchor_count"],
            "eligible_anchor_count": counts["eligible_anchor_count"],
            "dry_run": dry_run,
            "started_at": started_at,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": time.perf_counter() - started,
        }
        if dry_run:
            return manifest

        assert temp_dir is not None
        _dump_pickle(metadata, temp_dir / "metadata.pkl")
        manifest["outputs"] = [
            "trajectories/*.pkl", "metadata.pkl", "trajectory_index.csv",
            "excluded_rows.csv", "manifest.json",
        ]
        (temp_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        manifest_handle = (temp_dir / "manifest.json").open("rb")
        try:
            os.fsync(manifest_handle.fileno())
        finally:
            manifest_handle.close()
        # Cover the interval in which metadata and manifest are written.
        if source_before != _source_fingerprint(canonical):
            raise RuntimeError("Canonical source changed during trajectory reconstruction")
        if event_source_before is not None and event_source_before != _source_fingerprint(event_breaks_path):
            raise RuntimeError("Event-break source changed during trajectory reconstruction")
        if output_dir.exists():
            raise FileExistsError(f"Refusing to overwrite output created concurrently: {output_dir}")
        temp_dir.rename(output_dir)
        temp_dir = None
        return manifest
    except Exception:
        if temp_dir is not None and temp_dir.exists():
            shutil.rmtree(temp_dir)
        raise
    finally:
        if index_handle is not None:
            index_handle.close()
        if exclusion_handle is not None:
            exclusion_handle.close()
