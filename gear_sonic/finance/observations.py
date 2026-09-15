"""Causal monthly states and privileged future trajectories from canonical OHLCV."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from itertools import groupby
import math
from pathlib import Path
import statistics
from typing import Iterable, Iterator, Sequence


CURRENT_FIELDS = (
    "return_1m", "gap_return", "intramonth_return", "amplitude", "close_location",
    "momentum_3m", "momentum_6m", "momentum_acceleration", "volatility_3m",
    "volatility_6m", "volatility_ratio", "drawdown_6m", "liquidity_rank",
    "volume_surprise_3m", "market_return_1m", "relative_strength_6m",
)
FUTURE_FIELDS = (
    "forward_log_return", "cum_log_return", "gap_return", "intramonth_return",
    "amplitude", "close_location", "momentum_3m_delta", "momentum_6m_delta",
    "volatility_3m_ratio", "volatility_6m_ratio", "running_drawdown",
    "volume_surprise", "dollar_volume_surprise", "relative_strength_1m",
    "relative_strength_6m",
)
EPS = 1e-6


def month_number(period: str) -> int:
    year, month = period.split("-")
    if len(period) != 7 or not 1 <= int(month) <= 12:
        raise ValueError(f"Invalid monthly period: {period}")
    return int(year) * 12 + int(month)


@dataclass(frozen=True, slots=True)
class MonthlyBar:
    symbol: str
    period: str
    open: float
    close: float
    high: float
    low: float
    raw_close: float
    volume: float

    @property
    def dollar_volume(self) -> float:
        # One consistent proxy avoids mixing reported and imputed amounts.
        return self.raw_close * self.volume


@dataclass(frozen=True, slots=True)
class MarketMonth:
    return_1m: float | None
    momentum_6m: float | None
    liquidity_ranks: dict[str, float]


@dataclass(frozen=True, slots=True)
class MonthlyObservation:
    symbol: str
    period: str
    current_state: tuple[float, ...]
    close: float
    dollar_volume_surprise: float


def read_canonical(path: str | Path, symbols: set[str] | None = None) -> Iterator[MonthlyBar]:
    """Read valid paired rows; prices are QFQ, volume and liquidity are raw.

    The canonical file is grouped by symbol and sorted by month. Zero-volume
    rows cannot supply log-volume surprises and are excluded, leaving a gap.
    Reported amount and turnover are deliberately not required.
    """
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if symbols is not None and row["symbol"] not in symbols:
                continue
            if row["raw_valid"] != "1" or row["qfq_valid"] != "1":
                continue
            values = [float(row[key]) for key in (
                "qfq_open", "qfq_close", "qfq_high", "qfq_low", "raw_close", "raw_volume",
            )]
            if not all(math.isfinite(value) and value > 0 for value in values):
                continue
            opened, closed, high, low, raw_close, volume = values
            if low > min(opened, closed) or high < max(opened, closed):
                continue
            month_number(row["period"])
            yield MonthlyBar(row["symbol"], row["period"], opened, closed, high, low, raw_close, volume)


def build_market_context(bars: Iterable[MonthlyBar]) -> dict[str, MarketMonth]:
    """Compute same-month median returns and tie-aware liquidity percentiles.

    Each symbol must arrive in chronological order. Always pass the full
    available universe here, even when loading observations for one symbol.
    """
    previous: dict[str, MonthlyBar] = {}
    returns: dict[str, list[float]] = {}
    liquidity: dict[str, list[tuple[str, float]]] = {}
    for bar in bars:
        prior = previous.get(bar.symbol)
        if prior is not None:
            distance = month_number(bar.period) - month_number(prior.period)
            if distance <= 0:
                raise ValueError(f"duplicate or unsorted month: {bar.symbol} {bar.period}")
            if distance == 1:
                returns.setdefault(bar.period, []).append(bar.close / prior.close - 1)
        liquidity.setdefault(bar.period, []).append((bar.symbol, bar.dollar_volume))
        previous[bar.symbol] = bar

    median_returns = {month_number(period): statistics.median(values) for period, values in returns.items()}
    result = {}
    for period, values in liquidity.items():
        ordered = sorted(values, key=lambda item: item[1])
        ranks = {}
        index = 0
        for _, tied in groupby(ordered, key=lambda item: item[1]):
            group = list(tied)
            rank = (index + (len(group) - 1) / 2) / (len(ordered) - 1) if len(ordered) > 1 else 0.5
            ranks.update((symbol, rank) for symbol, _ in group)
            index += len(group)
        number = month_number(period)
        recent = [median_returns.get(number - lag) for lag in range(6)]
        momentum = math.prod(1 + value for value in recent) - 1 if all(v is not None for v in recent) else None
        result[period] = MarketMonth(median_returns.get(number), momentum, ranks)
    return result


def _contiguous(rows: Sequence[MonthlyBar | MonthlyObservation]) -> bool:
    return all(
        left.symbol == right.symbol and month_number(right.period) - month_number(left.period) == 1
        for left, right in zip(rows, rows[1:])
    )


def build_symbol_observations(
    bars: Sequence[MonthlyBar], market: dict[str, MarketMonth],
) -> list[MonthlyObservation]:
    """Build one 16-dimensional state per month after six months of warm-up."""
    observations = []
    for index in range(6, len(bars)):
        history = bars[index - 6:index + 1]
        if not _contiguous(history):
            continue
        current, previous = history[-1], history[-2]
        context = market.get(current.period)
        if context is None or context.return_1m is None or context.momentum_6m is None:
            continue
        returns = [right.close / left.close - 1 for left, right in zip(history, history[1:])]
        mom3 = current.close / history[-4].close - 1
        mom6 = current.close / history[0].close - 1
        vol3, vol6 = statistics.pstdev(returns[-3:]), statistics.pstdev(returns)
        volume_mean = statistics.fmean(row.volume for row in history[-4:-1])
        dollar_mean = statistics.fmean(row.dollar_volume for row in history[-4:-1])
        if min(volume_mean, dollar_mean, current.volume, current.dollar_volume) <= 0:
            continue
        state = (
            returns[-1], current.open / previous.close - 1, current.close / current.open - 1,
            (current.high - current.low) / current.open,
            (current.close - current.low) / (current.high - current.low) if current.high > current.low else 0.5,
            mom3, mom6, math.log1p(mom3) / 3 - math.log1p(mom6) / 6,
            vol3, vol6, math.log((vol3 + EPS) / (vol6 + EPS)),
            current.close / max(row.close for row in history[-6:]) - 1,
            context.liquidity_ranks[current.symbol], math.log(current.volume / volume_mean),
            context.return_1m, mom6 - context.momentum_6m,
        )
        if not all(math.isfinite(value) for value in state):
            continue
        observations.append(MonthlyObservation(
            current.symbol, current.period, state, current.close, math.log(current.dollar_volume / dollar_mean),
        ))
    return observations


def load_observations(path: str | Path, symbols: set[str] | None = None) -> dict[str, list[MonthlyObservation]]:
    """Two-pass loading keeps all-universe market context but only selected histories."""
    market = build_market_context(read_canonical(path))
    result = {}
    for symbol, group in groupby(read_canonical(path, symbols), key=lambda bar: bar.symbol):
        if symbol in result:
            raise ValueError("Canonical rows must be grouped by symbol")
        result[symbol] = build_symbol_observations(list(group), market)
    return result


def make_sample(rows: Sequence[MonthlyObservation], anchor: int, horizon: int = 10) -> dict:
    if horizon < 1 or anchor < 0 or anchor + horizon >= len(rows):
        raise ValueError("A complete future horizon is required")
    window = rows[anchor:anchor + horizon + 1]
    if not _contiguous(window):
        raise ValueError("Reference months must be contiguous within one symbol")
    current = window[0]
    base = current.current_state
    peak = current.close
    future = []
    for row in window[1:]:
        state = row.current_state
        peak = max(peak, row.close)
        future.append((
            math.log1p(state[0]), math.log(row.close / current.close),
            state[1], state[2], state[3], state[4], state[5] - base[5], state[6] - base[6],
            math.log((state[8] + EPS) / (base[8] + EPS)),
            math.log((state[9] + EPS) / (base[9] + EPS)), row.close / peak - 1,
            state[13], row.dollar_volume_surprise, state[0] - state[14], state[15],
        ))
    return {
        "symbol": current.symbol, "anchor_period": current.period,
        "target_end_period": window[-1].period, "current_state": current.current_state,
        "future_reference": future, "future_mask": [[True] * len(FUTURE_FIELDS) for _ in future],
    }


def iter_sequences(
    observations: dict[str, list[MonthlyObservation]], *, sequence_length: int = 32,
    horizon: int = 10, start_period: str | None = None, end_period: str | None = None,
) -> Iterator[dict]:
    """Yield independent, non-overlapping sequences of monthly anchor steps.

    Split bounds include anchors AND the entire future label window. Short
    tails are dropped. Each emitted sequence starts with an empty cache.
    """
    if sequence_length < 1 or horizon < 1:
        raise ValueError("sequence_length and horizon must be positive")
    if start_period is not None:
        month_number(start_period)
    if end_period is not None:
        month_number(end_period)
    if start_period and end_period and start_period > end_period:
        raise ValueError("start_period must not exceed end_period")
    for symbol, rows in observations.items():
        pending = []
        for anchor in range(len(rows) - horizon):
            if not _contiguous(rows[anchor:anchor + horizon + 1]):
                pending = []
                continue
            sample = make_sample(rows, anchor, horizon)
            if ((start_period and sample["anchor_period"] < start_period)
                    or (end_period and sample["target_end_period"] > end_period)):
                pending = []
                continue
            pending.append(sample)
            if len(pending) == sequence_length:
                yield {
                    "symbol": symbol, "periods": [item["anchor_period"] for item in pending],
                    "target_end_period": pending[-1]["target_end_period"],
                    "current_state": [item["current_state"] for item in pending],
                    "future_reference": [item["future_reference"] for item in pending],
                    "future_mask": [item["future_mask"] for item in pending],
                }
                pending = []
