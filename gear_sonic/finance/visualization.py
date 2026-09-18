"""Small, deterministic PNG visualizations for privileged finance rollouts.

The evaluator deliberately does not depend on a plotting stack.  This module
uses NumPy for the raster buffer and the Python standard library for the PNG
container, which keeps command-line evaluation usable on the training nodes
where optional GUI/plotting packages are not installed.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import math
import numbers
from pathlib import Path
import struct
from typing import Any
import zlib

import numpy as np

from gear_sonic.finance.evaluation import TRAJECTORY_FIELDNAMES


PNG_WIDTH = 1600
PNG_HEIGHT = 1000
HORIZON_COUNT = 10

_REQUIRED_FIELDS = frozenset(TRAJECTORY_FIELDNAMES)
_NUMERIC_FIELDS = (
    "predicted_normalized_return",
    "target_normalized_return",
    "predicted_log_return",
    "target_log_return",
    "predicted_cumulative_log_return",
    "target_cumulative_log_return",
    "reference_cumulative_log_return",
)
_TEXT_FIELDS = (
    "symbol",
    "anchor_period",
    "target_period",
    "latent_source",
    "encoder_input_mode",
    "action_mode",
    "cache_reset_reason",
    "rollout_mode",
)


def _integer(value: Any, name: str, *, minimum: int | None = None) -> int:
    """Parse an integer while rejecting booleans and lossy float values."""
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    if isinstance(value, numbers.Integral):
        result = int(value)
    elif isinstance(value, str):
        try:
            result = int(value.strip())
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be an integer") from exc
    else:
        raise ValueError(f"{name} must be an integer")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _text(value: Any, name: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    if not allow_empty and not value.strip():
        raise ValueError(f"{name} must not be empty")
    return value


def _period_number(value: str, name: str) -> int:
    """Convert a canonical ``YYYY-MM`` label to a monotonic month number."""
    pieces = value.split("-")
    if len(pieces) != 2:
        raise ValueError(f"{name} must use YYYY-MM format")
    try:
        year, month = int(pieces[0]), int(pieces[1])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must use YYYY-MM format") from exc
    if year < 1 or not 1 <= month <= 12:
        raise ValueError(f"{name} must use YYYY-MM format")
    return year * 12 + month - 1


def _period_after(period: str, offset: int) -> str:
    month_number = _period_number(period, "anchor_period") + offset
    year, month_zero = divmod(month_number, 12)
    return f"{year:04d}-{month_zero + 1:02d}"


def _boolean(value: Any, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no"}:
            return False
    raise ValueError(f"{name} must be boolean")


def _normalise_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Validate and normalize the CSV-compatible trajectory ledger.

    A complete ten-horizon block is required for each ``(sequence, anchor)``.
    This catches both truncated CSV streams and accidental duplicate rows before
    either can produce a misleading picture.
    """
    if isinstance(rows, (str, bytes)):
        raise ValueError("Trajectory rows must be an iterable of mappings")
    try:
        raw_rows = list(rows)
    except TypeError as exc:
        raise ValueError("Trajectory rows must be an iterable of mappings") from exc
    if not raw_rows:
        raise ValueError("At least one trajectory row is required")

    normalized: list[dict[str, Any]] = []
    for row_number, row in enumerate(raw_rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"Trajectory row {row_number} must be a mapping")
        keys = set(row)
        missing = _REQUIRED_FIELDS - keys
        extra = keys - _REQUIRED_FIELDS
        if missing:
            raise ValueError(
                f"Trajectory row {row_number} is missing fields: {', '.join(sorted(missing))}"
            )
        if extra:
            raise ValueError(
                f"Trajectory row {row_number} has unknown fields: "
                f"{', '.join(sorted(map(str, extra)))}"
            )

        item: dict[str, Any] = {}
        item["sequence_id"] = _integer(row["sequence_id"], "sequence_id", minimum=0)
        item["symbol"] = _text(row["symbol"], "symbol")
        item["anchor_index"] = _integer(row["anchor_index"], "anchor_index", minimum=0)
        item["anchor_period"] = _text(row["anchor_period"], "anchor_period")
        item["target_period"] = _text(row["target_period"], "target_period")
        item["horizon"] = _integer(row["horizon"], "horizon", minimum=1)
        if item["horizon"] > HORIZON_COUNT:
            raise ValueError(f"horizon must be in [1, {HORIZON_COUNT}]")
        for name in _NUMERIC_FIELDS:
            item[name] = _finite(row[name], name)
        for name in _TEXT_FIELDS[3:]:
            item[name] = _text(row[name], name, allow_empty=name == "cache_reset_reason")
        item["cache_reset"] = _boolean(row["cache_reset"], "cache_reset")

        # Keep the chart from silently presenting another evaluation mode as a
        # causal forecast.  These values are part of the playback contract.
        if item["latent_source"] != "oracle_future_encoder":
            raise ValueError("trajectory latent_source must be oracle_future_encoder")
        if item["encoder_input_mode"] != "clean":
            raise ValueError("trajectory encoder_input_mode must be clean")
        if item["action_mode"] != "deterministic_mean":
            raise ValueError("trajectory action_mode must be deterministic_mean")
        if item["rollout_mode"] != "privileged_train_playback":
            raise ValueError("trajectory rollout_mode must be privileged_train_playback")
        normalized.append(item)

    # Index rows by semantic identity and enforce one complete action vector per
    # anchor.  Sorting below makes rendering independent of input/CSV order.
    blocks: dict[tuple[int, int], dict[int, dict[str, Any]]] = {}
    symbols: dict[int, str] = {}
    for item in normalized:
        sequence_id = item["sequence_id"]
        anchor_index = item["anchor_index"]
        horizon = item["horizon"]
        if sequence_id in symbols and symbols[sequence_id] != item["symbol"]:
            raise ValueError(f"sequence {sequence_id} changes symbol")
        symbols[sequence_id] = item["symbol"]
        block = blocks.setdefault((sequence_id, anchor_index), {})
        if horizon in block:
            raise ValueError(
                f"duplicate trajectory row for sequence {sequence_id}, anchor {anchor_index}, "
                f"horizon {horizon}"
            )
        block[horizon] = item

    expected_horizons = set(range(1, HORIZON_COUNT + 1))
    for (sequence_id, anchor_index), block in blocks.items():
        if set(block) != expected_horizons:
            missing = expected_horizons - set(block)
            extra = set(block) - expected_horizons
            detail = []
            if missing:
                detail.append(f"missing horizons {sorted(missing)}")
            if extra:
                detail.append(f"invalid horizons {sorted(extra)}")
            raise ValueError(
                f"incomplete trajectory block for sequence {sequence_id}, anchor {anchor_index} "
                f"({'; '.join(detail)})"
            )
        periods = {block[h]["anchor_period"] for h in expected_horizons}
        if len(periods) != 1:
            raise ValueError(f"anchor {anchor_index} has inconsistent anchor_period values")
        anchor_period = next(iter(periods))
        # A horizon row is identified by both its integer horizon and its
        # target month.  Checking the calendar relation prevents a shifted row
        # from producing a plausible-looking but semantically wrong heatmap.
        for horizon in range(1, HORIZON_COUNT + 1):
            expected_period = _period_after(anchor_period, horizon)
            if block[horizon]["target_period"] != expected_period:
                raise ValueError(
                    f"sequence {sequence_id}, anchor {anchor_index}, horizon {horizon} "
                    f"has target_period {block[horizon]['target_period']!r}; "
                    f"expected {expected_period!r}"
                )

    sequence_ids = sorted(symbols)
    return [
        blocks[(sequence_id, anchor_index)][horizon]
        for sequence_id in sequence_ids
        for anchor_index in sorted(
            anchor for sid, anchor in blocks if sid == sequence_id
        )
        for horizon in range(1, HORIZON_COUNT + 1)
    ]


def _select_rows(rows: list[dict[str, Any]], max_sequences: int | None) -> list[dict[str, Any]]:
    if max_sequences is not None:
        max_sequences = _integer(max_sequences, "max_sequences", minimum=1)
    sequence_ids = sorted({row["sequence_id"] for row in rows})
    if max_sequences is not None:
        sequence_ids = sequence_ids[:max_sequences]
    selected = set(sequence_ids)
    return [row for row in rows if row["sequence_id"] in selected]


def _group_blocks(rows: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    blocks: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for row in rows:
        blocks.setdefault((row["sequence_id"], row["anchor_index"]), []).append(row)
    return [
        sorted(blocks[key], key=lambda row: row["horizon"])
        for key in sorted(blocks)
    ]


def _group_sequence_blocks(
    rows: list[dict[str, Any]],
) -> list[tuple[int, str, list[list[dict[str, Any]]]]]:
    """Return ``(sequence_id, symbol, anchor blocks)`` in stable order."""
    grouped: dict[int, dict[int, list[dict[str, Any]]]] = {}
    symbols: dict[int, str] = {}
    for row in rows:
        sequence_id = row["sequence_id"]
        symbols[sequence_id] = row["symbol"]
        grouped.setdefault(sequence_id, {}).setdefault(row["anchor_index"], []).append(row)
    result = []
    for sequence_id in sorted(grouped):
        anchors = [
            sorted(grouped[sequence_id][anchor], key=lambda item: item["horizon"])
            for anchor in sorted(grouped[sequence_id])
        ]
        result.append((sequence_id, symbols[sequence_id], anchors))
    return result


def _sample_indices(total: int, limit: int) -> np.ndarray:
    if total <= 0:
        return np.empty(0, dtype=np.int64)
    if total <= limit:
        return np.arange(total, dtype=np.int64)
    return np.unique(np.rint(np.linspace(0, total - 1, limit)).astype(np.int64))


def _rect(canvas: np.ndarray, x0: int, y0: int, x1: int, y1: int, color: tuple[int, int, int]):
    height, width = canvas.shape[:2]
    x0, x1 = max(0, min(width, x0)), max(0, min(width, x1))
    y0, y1 = max(0, min(height, y0)), max(0, min(height, y1))
    if x1 > x0 and y1 > y0:
        canvas[y0:y1, x0:x1] = color


def _line(
    canvas: np.ndarray,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    color: tuple[int, int, int],
    width: int = 1,
):
    """Draw a clipped anti-alias-free line with deterministic integer samples."""
    steps = max(abs(int(round(x1 - x0))), abs(int(round(y1 - y0))), 1) * 2 + 1
    xs = np.rint(np.linspace(x0, x1, steps)).astype(np.int32)
    ys = np.rint(np.linspace(y0, y1, steps)).astype(np.int32)
    height, canvas_width = canvas.shape[:2]
    radius = max(0, int(width) // 2)
    for x, y in zip(xs, ys):
        if radius:
            xa, xb = max(0, x - radius), min(canvas_width, x + radius + 1)
            ya, yb = max(0, y - radius), min(height, y + radius + 1)
            if xa < xb and ya < yb:
                canvas[ya:yb, xa:xb] = color
        elif 0 <= x < canvas_width and 0 <= y < height:
            canvas[y, x] = color


def _dashed_line(
    canvas: np.ndarray,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    color: tuple[int, int, int],
    *,
    width: int = 1,
    dash: float = 8.0,
    gap: float = 5.0,
):
    """Draw a deterministic dash pattern along a line segment."""
    length = math.hypot(x1 - x0, y1 - y0)
    if length <= 0:
        _line(canvas, x0, y0, x1, y1, color, width)
        return
    period = max(1.0, float(dash) + float(gap))
    start = 0.0
    while start < length:
        end = min(length, start + max(1.0, float(dash)))
        fraction_start, fraction_end = start / length, end / length
        _line(
            canvas,
            x0 + (x1 - x0) * fraction_start,
            y0 + (y1 - y0) * fraction_start,
            x0 + (x1 - x0) * fraction_end,
            y0 + (y1 - y0) * fraction_end,
            color,
            width,
        )
        start += period


# A compact 5x7 bitmap font keeps labels readable without a system font or a
# platform-dependent rasterizer.  Unknown characters are rendered as a box.
_FONT = {
    " ": ("00000",) * 7,
    "-": ("00000", "00000", "00000", "11111", "00000", "00000", "00000"),
    ".": ("00000", "00000", "00000", "00000", "00000", "01100", "01100"),
    ":": ("00000", "01100", "01100", "00000", "01100", "01100", "00000"),
    "/": ("00001", "00010", "00100", "01000", "10000", "00000", "00000"),
    "|": ("00100", "00100", "00100", "00100", "00100", "00100", "00100"),
    "_": ("00000", "00000", "00000", "00000", "00000", "00000", "11111"),
    "(": ("00010", "00100", "01000", "01000", "01000", "00100", "00010"),
    ")": ("01000", "00100", "00010", "00010", "00010", "00100", "01000"),
    "[": ("01110", "01000", "01000", "01000", "01000", "01000", "01110"),
    "]": ("01110", "00010", "00010", "00010", "00010", "00010", "01110"),
    "+": ("00000", "00100", "00100", "11111", "00100", "00100", "00000"),
    "%": ("11001", "11010", "00010", "00100", "01000", "01011", "10011"),
    "?": ("11110", "00001", "00010", "00100", "00100", "00000", "00100"),
    "0": ("01110", "10001", "10011", "10101", "11001", "10001", "01110"),
    "1": ("00100", "01100", "00100", "00100", "00100", "00100", "01110"),
    "2": ("01110", "10001", "00001", "00010", "00100", "01000", "11111"),
    "3": ("11110", "00001", "00001", "01110", "00001", "00001", "11110"),
    "4": ("00010", "00110", "01010", "10010", "11111", "00010", "00010"),
    "5": ("11111", "10000", "10000", "11110", "00001", "00001", "11110"),
    "6": ("01110", "10000", "10000", "11110", "10001", "10001", "01110"),
    "7": ("11111", "00001", "00010", "00100", "01000", "01000", "01000"),
    "8": ("01110", "10001", "10001", "01110", "10001", "10001", "01110"),
    "9": ("01110", "10001", "10001", "01111", "00001", "00001", "01110"),
    "A": ("01110", "10001", "10001", "11111", "10001", "10001", "10001"),
    "B": ("11110", "10001", "10001", "11110", "10001", "10001", "11110"),
    "C": ("01111", "10000", "10000", "10000", "10000", "10000", "01111"),
    "D": ("11110", "10001", "10001", "10001", "10001", "10001", "11110"),
    "E": ("11111", "10000", "10000", "11110", "10000", "10000", "11111"),
    "F": ("11111", "10000", "10000", "11110", "10000", "10000", "10000"),
    "G": ("01111", "10000", "10000", "10111", "10001", "10001", "01111"),
    "H": ("10001", "10001", "10001", "11111", "10001", "10001", "10001"),
    "I": ("11111", "00100", "00100", "00100", "00100", "00100", "11111"),
    "J": ("00111", "00010", "00010", "00010", "00010", "10010", "01100"),
    "K": ("10001", "10010", "10100", "11000", "10100", "10010", "10001"),
    "L": ("10000", "10000", "10000", "10000", "10000", "10000", "11111"),
    "M": ("10001", "11011", "10101", "10101", "10001", "10001", "10001"),
    "N": ("10001", "11001", "10101", "10011", "10001", "10001", "10001"),
    "O": ("01110", "10001", "10001", "10001", "10001", "10001", "01110"),
    "P": ("11110", "10001", "10001", "11110", "10000", "10000", "10000"),
    "Q": ("01110", "10001", "10001", "10001", "10101", "10010", "01101"),
    "R": ("11110", "10001", "10001", "11110", "10100", "10010", "10001"),
    "S": ("01111", "10000", "10000", "01110", "00001", "00001", "11110"),
    "T": ("11111", "00100", "00100", "00100", "00100", "00100", "00100"),
    "U": ("10001", "10001", "10001", "10001", "10001", "10001", "01110"),
    "V": ("10001", "10001", "10001", "10001", "10001", "01010", "00100"),
    "W": ("10001", "10001", "10001", "10101", "10101", "11011", "10001"),
    "X": ("10001", "10001", "01010", "00100", "01010", "10001", "10001"),
    "Y": ("10001", "10001", "01010", "00100", "00100", "00100", "00100"),
    "Z": ("11111", "00001", "00010", "00100", "01000", "10000", "11111"),
}


def _draw_text(
    canvas: np.ndarray,
    x: int,
    y: int,
    text: str,
    color: tuple[int, int, int],
    *,
    scale: int = 1,
):
    scale = max(1, int(scale))
    cursor = int(x)
    for character in str(text):
        glyph = _FONT.get(character.upper(), _FONT["?"])
        for row_index, bits in enumerate(glyph):
            for column_index, bit in enumerate(bits):
                if bit == "1":
                    _rect(
                        canvas,
                        cursor + column_index * scale,
                        y + row_index * scale,
                        cursor + (column_index + 1) * scale,
                        y + (row_index + 1) * scale,
                        color,
                    )
        cursor += 6 * scale


def _panel(
    canvas: np.ndarray,
    bounds: tuple[int, int, int, int],
    heading: str,
    *,
    heading_scale: int = 2,
):
    x0, y0, x1, y1 = bounds
    _rect(canvas, x0, y0, x1, y1, (255, 255, 255))
    border = (179, 188, 200)
    _rect(canvas, x0, y0, x1, y0 + 2, border)
    _rect(canvas, x0, y1 - 2, x1, y1, border)
    _rect(canvas, x0, y0, x0 + 2, y1, border)
    _rect(canvas, x1 - 2, y0, x1, y1, border)
    _draw_text(canvas, x0 + 14, y0 + 13, heading, (35, 48, 64), scale=heading_scale)


def _scale_y(value: float, low: float, high: float, y0: int, y1: int) -> float:
    if high <= low:
        return (y0 + y1) / 2
    return y1 - (value - low) / (high - low) * (y1 - y0)


def _tick_label(value: float) -> str:
    if abs(value) >= 100 or (abs(value) > 0 and abs(value) < 0.01):
        return f"{value:.2g}"
    return f"{value:.3f}".rstrip("0").rstrip(".")


def _draw_paths(
    canvas: np.ndarray,
    sequence_groups: list[tuple[int, str, list[list[dict[str, Any]]]]],
) -> dict[str, Any]:
    """Draw one small multiple per clip and report any visual downsampling."""
    bounds = (70, 105, 1020, 480)
    visible_groups = sequence_groups[:4]
    visible_count = len(visible_groups)
    _panel(canvas, bounds, f"CUMULATIVE PATHS ({visible_count}/{len(sequence_groups)} CLIPS)")
    x0, y0, x1, y1 = bounds
    prediction_color = (37, 103, 182)  # blue solid
    target_color = (218, 105, 45)  # orange dashed
    zero_color = (145, 155, 168)

    # The four fixed tiles keep the chart readable even when a reference pool
    # contains thousands of anchors.  Empty slots are intentionally outlined so
    # the selected clip count is visually obvious.
    grid_x0, grid_x1 = x0 + 14, x1 - 14
    grid_y0, grid_y1 = y0 + 52, y1 - 11
    gap_x, gap_y = 14, 10
    tile_width = (grid_x1 - grid_x0 - gap_x) // 2
    tile_height = (grid_y1 - grid_y0 - gap_y) // 2
    anchor_limit = 12
    rendered_anchor_count = 0

    for tile_index in range(4):
        tile_column, tile_row = tile_index % 2, tile_index // 2
        tx = grid_x0 + tile_column * (tile_width + gap_x)
        ty = grid_y0 + tile_row * (tile_height + gap_y)
        _rect(canvas, tx, ty, tx + tile_width, ty + tile_height, (250, 252, 255))
        _rect(canvas, tx, ty, tx + tile_width, ty + 1, (190, 199, 210))
        _rect(canvas, tx, ty + tile_height - 1, tx + tile_width, ty + tile_height, (190, 199, 210))
        _rect(canvas, tx, ty, tx + 1, ty + tile_height, (190, 199, 210))
        _rect(canvas, tx + tile_width - 1, ty, tx + tile_width, ty + tile_height, (190, 199, 210))
        if tile_index >= visible_count:
            _draw_text(canvas, tx + 12, ty + tile_height // 2 - 4, "NO CLIP", (145, 155, 168), scale=1)
            continue

        sequence_id, symbol, anchors = visible_groups[tile_index]
        indices = _sample_indices(len(anchors), anchor_limit)
        rendered_anchor_count += len(indices)
        heading = f"{symbol} / SEQ {sequence_id}  {len(indices)}/{len(anchors)} A"
        _draw_text(canvas, tx + 8, ty + 6, heading, (45, 59, 75), scale=1)
        plot_x0, plot_x1 = tx + 34, tx + tile_width - 9
        plot_y0, plot_y1 = ty + 24, ty + tile_height - 17
        predicted = np.asarray(
            [[anchors[index][h]["predicted_cumulative_log_return"] for h in range(HORIZON_COUNT)]
             for index in indices],
            dtype=np.float64,
        )
        target = np.asarray(
            [[anchors[index][h]["target_cumulative_log_return"] for h in range(HORIZON_COUNT)]
             for index in indices],
            dtype=np.float64,
        )
        value_scale = max(1.0, float(np.max(np.abs(predicted))), float(np.max(np.abs(target))))
        predicted_plot, target_plot = predicted / value_scale, target / value_scale
        low = float(min(predicted_plot.min(), target_plot.min(), 0.0))
        high = float(max(predicted_plot.max(), target_plot.max(), 0.0))
        if math.isclose(low, high):
            delta = max(abs(low) * 0.1, 0.01)
            low, high = low - delta, high + delta
        for tick in np.linspace(low, high, 3):
            y = int(round(_scale_y(float(tick), low, high, plot_y0, plot_y1)))
            _line(canvas, plot_x0, y, plot_x1, y, (225, 231, 238))
            if tile_index % 2 == 0:
                _draw_text(
                    canvas,
                    tx + 3,
                    y - 3,
                    _tick_label(float(tick) * value_scale),
                    (106, 117, 130),
                    scale=1,
                )
        zero_y = int(round(_scale_y(0.0, low, high, plot_y0, plot_y1)))
        _line(canvas, plot_x0, zero_y, plot_x1, zero_y, zero_color, 1)
        xs = np.linspace(plot_x0, plot_x1, HORIZON_COUNT)
        for pred_path, target_path in zip(predicted_plot, target_plot):
            for left in range(HORIZON_COUNT - 1):
                _dashed_line(
                    canvas,
                    xs[left], _scale_y(float(target_path[left]), low, high, plot_y0, plot_y1),
                    xs[left + 1], _scale_y(float(target_path[left + 1]), low, high, plot_y0, plot_y1),
                    target_color, width=2, dash=7, gap=4,
                )
                _line(
                    canvas,
                    xs[left], _scale_y(float(pred_path[left]), low, high, plot_y0, plot_y1),
                    xs[left + 1], _scale_y(float(pred_path[left + 1]), low, high, plot_y0, plot_y1),
                    prediction_color, 2,
                )
        for horizon in (1, 5, 10):
            x = xs[horizon - 1]
            _draw_text(canvas, int(x) - (3 if horizon < 10 else 6), plot_y1 + 5, str(horizon), (100, 111, 125))

    # A compact global legend sits in the panel header and applies to every tile.
    legend_x = x1 - 216
    legend_y = y0 + 27
    _line(canvas, legend_x, legend_y, legend_x + 18, legend_y, prediction_color, 2)
    _draw_text(canvas, legend_x + 23, legend_y - 5, "PRED SOLID", (65, 76, 90), scale=1)
    _dashed_line(canvas, legend_x + 112, legend_y, legend_x + 130, legend_y, target_color, width=2, dash=5, gap=3)
    _draw_text(canvas, legend_x + 135, legend_y - 5, "TARGET DASH", (65, 76, 90), scale=1)
    return {
        "sequence_count": visible_count,
        "sequence_total": len(sequence_groups),
        "sequence_downsampled": visible_count < len(sequence_groups),
        "anchor_count": sum(len(anchors) for _, _, anchors in sequence_groups),
        "anchor_count_rendered": rendered_anchor_count,
        "anchor_downsampled": rendered_anchor_count < sum(len(anchors) for _, _, anchors in visible_groups),
    }


def _heat_color(value: float, maximum: float) -> tuple[int, int, int]:
    if maximum <= 0:
        return (225, 239, 233)
    ratio = min(1.0, max(0.0, value / maximum))
    # Pale green at zero error, warm red at the largest error.
    return (
        int(round(224 + 31 * ratio)),
        int(round(243 - 145 * ratio)),
        int(round(232 - 145 * ratio)),
    )


def _draw_heatmap(canvas: np.ndarray, blocks: list[list[dict[str, Any]]]) -> dict[str, Any]:
    bounds = (1060, 105, 1530, 480)
    row_count = min(len(blocks), 25)
    indices = _sample_indices(len(blocks), row_count)
    _panel(
        canvas,
        bounds,
        f"ABS ERROR HEATMAP ({len(indices)}/{len(blocks)} ANCHORS)",
        heading_scale=1,
    )
    x0, y0, x1, y1 = bounds
    plot_x0, plot_x1 = x0 + 28, x1 - 16
    # Leave a dedicated strip below the cells for both horizon labels and the
    # color scale. Keeping these regions separate avoids labels being painted
    # over one another at native resolution.
    plot_y0, plot_y1 = y0 + 58, y1 - 58
    predicted = np.asarray(
        [[row["predicted_cumulative_log_return"] for row in block] for block in blocks],
        dtype=np.float64,
    )
    target = np.asarray(
        [[row["target_cumulative_log_return"] for row in block] for block in blocks],
        dtype=np.float64,
    )
    value_scale = max(1.0, float(np.max(np.abs(predicted))), float(np.max(np.abs(target))))
    errors = np.abs(predicted / value_scale - target / value_scale)
    maximum = float(errors.max()) if errors.size else 0.0
    raw_maximum = maximum * value_scale
    raw_maximum_label = _tick_label(raw_maximum) if math.isfinite(raw_maximum) else "inf"
    cell_width = (plot_x1 - plot_x0) / HORIZON_COUNT
    cell_height = (plot_y1 - plot_y0) / max(len(indices), 1)
    for display_row, source_row in enumerate(indices):
        top = int(round(plot_y0 + display_row * cell_height))
        bottom = int(round(plot_y0 + (display_row + 1) * cell_height))
        for horizon in range(HORIZON_COUNT):
            left = int(round(plot_x0 + horizon * cell_width))
            right = int(round(plot_x0 + (horizon + 1) * cell_width))
            _rect(
                canvas,
                left + 1,
                top + 1,
                right,
                bottom,
                _heat_color(float(errors[source_row, horizon]), maximum),
            )
    for horizon in range(1, HORIZON_COUNT + 1):
        left = int(round(plot_x0 + (horizon - 0.5) * cell_width))
        _draw_text(canvas, left - 3, plot_y1 + 8, str(horizon), (100, 111, 125))
    _draw_text(canvas, x0 + 18, y1 - 20, "MIN 0", (80, 120, 98))
    _rect(canvas, x0 + 66, y1 - 22, x0 + 80, y1 - 8, (224, 239, 231))
    _draw_text(canvas, x0 + 92, y1 - 20, "MAX", (145, 70, 58))
    _rect(canvas, x0 + 122, y1 - 22, x0 + 136, y1 - 8, (255, 98, 87))
    _draw_text(canvas, x0 + 148, y1 - 20, raw_maximum_label, (145, 70, 58))
    return {
        "rows_rendered": len(indices),
        "rows_total": len(blocks),
        "downsampled": len(indices) < len(blocks),
    }


def _direction_color(predicted: float, target: float) -> tuple[int, int, int]:
    pred_sign = 1 if predicted > 0 else -1 if predicted < 0 else 0
    target_sign = 1 if target > 0 else -1 if target < 0 else 0
    if pred_sign == target_sign and pred_sign != 0:
        return (65, 166, 111)
    if pred_sign == 0 or target_sign == 0:
        return (190, 198, 207)
    return (213, 82, 76)


def _draw_directions(canvas: np.ndarray, blocks: list[list[dict[str, Any]]]) -> dict[str, Any]:
    bounds = (70, 560, 1530, 925)
    row_count = min(len(blocks), 36)
    indices = _sample_indices(len(blocks), row_count)
    _panel(
        canvas,
        bounds,
        f"DIRECTION AGREEMENT (MONTHLY {len(indices)}/{len(blocks)} ANCHORS)",
        heading_scale=1,
    )
    x0, y0, x1, y1 = bounds
    plot_x0, plot_x1 = x0 + 48, x1 - 22
    # Reserve the lower strip for horizon labels and the legend. The matrix
    # remains tall enough to show dozens of anchors while keeping the two text
    # rows visually independent.
    plot_y0, plot_y1 = y0 + 58, y1 - 72
    cell_width = (plot_x1 - plot_x0) / HORIZON_COUNT
    cell_height = (plot_y1 - plot_y0) / max(len(indices), 1)
    for display_row, source_row in enumerate(indices):
        top = int(round(plot_y0 + display_row * cell_height))
        bottom = int(round(plot_y0 + (display_row + 1) * cell_height))
        block = blocks[source_row]
        for horizon, row in enumerate(block):
            left = int(round(plot_x0 + horizon * cell_width))
            right = int(round(plot_x0 + (horizon + 1) * cell_width))
            _rect(
                canvas, left + 1, top + 1, right, bottom,
                _direction_color(row["predicted_log_return"], row["target_log_return"]),
            )
    for horizon in range(1, HORIZON_COUNT + 1):
        left = int(round(plot_x0 + (horizon - 0.5) * cell_width))
        _draw_text(canvas, left - 3, plot_y1 + 8, str(horizon), (100, 111, 125))
    legend_y = y1 - 22
    legend = (("CORRECT", (65, 166, 111)), ("WRONG", (213, 82, 76)), ("NEUTRAL", (190, 198, 207)))
    cursor = x0 + 25
    for label, color in legend:
        _rect(canvas, cursor, legend_y - 1, cursor + 14, legend_y + 13, color)
        _draw_text(canvas, cursor + 20, legend_y, label, (80, 91, 105))
        cursor += 115
    return {
        "rows_rendered": len(indices),
        "rows_total": len(blocks),
        "downsampled": len(indices) < len(blocks),
    }


def _chunk(kind: bytes, payload: bytes) -> bytes:
    checksum = zlib.crc32(kind + payload) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", checksum)


def _encode_png(canvas: np.ndarray) -> bytes:
    if canvas.shape != (PNG_HEIGHT, PNG_WIDTH, 3) or canvas.dtype != np.uint8:
        raise ValueError("PNG canvas must be an uint8 RGB raster with fixed dimensions")
    scanlines = b"".join(b"\x00" + row.tobytes() for row in canvas)
    header = struct.pack(">IIBBBBB", PNG_WIDTH, PNG_HEIGHT, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", header)
        + _chunk(b"IDAT", zlib.compress(scanlines, level=9))
        + _chunk(b"IEND", b"")
    )


def render_trajectory_png(
    rows: Iterable[Mapping[str, Any]],
    path: str | Path,
    *,
    title: str = "FINANCE SONIC PRIVILEGED TRAIN PLAYBACK",
    max_sequences: int | None = None,
) -> dict[str, Any]:
    """Render cumulative paths, errors, and direction agreement to one PNG.

    The output is intentionally an exclusive-create artifact.  Existing files
    are never overwritten, and a failed write removes only a file created by
    this invocation.  Returned metadata contains no path or timestamp, making
    repeated renders byte-for-byte and metadata deterministic.
    """
    normalized = _normalise_rows(rows)
    selected = _select_rows(normalized, max_sequences)
    if not selected:
        raise ValueError("max_sequences leaves no trajectory rows")
    sequence_groups = _group_sequence_blocks(selected)
    blocks = _group_blocks(selected)
    selected_sequence_count = len(sequence_groups)
    anchor_count = len(blocks)
    visualized_sequence_count = min(selected_sequence_count, 4)
    visualized_sequence_ids = {
        sequence_id
        for sequence_id, _, _ in sequence_groups[:visualized_sequence_count]
    }
    panel_blocks = [
        block for block in blocks
        if block[0]["sequence_id"] in visualized_sequence_ids
    ]

    canvas = np.full((PNG_HEIGHT, PNG_WIDTH, 3), (248, 250, 253), dtype=np.uint8)
    display_title = f"{title} | {visualized_sequence_count} CLIPS"
    _draw_text(canvas, 70, 28, display_title, (27, 40, 55), scale=3)
    _draw_text(
        canvas,
        70,
        71,
        "CLEAN FUTURE ENCODER / DETERMINISTIC MEAN / ACTUAL MONTHLY STATE",
        (94, 107, 123),
        scale=1,
    )
    path_metadata = _draw_paths(canvas, sequence_groups)
    heatmap_metadata = _draw_heatmap(canvas, panel_blocks)
    direction_metadata = _draw_directions(canvas, panel_blocks)
    payload = _encode_png(canvas)

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    created = False
    try:
        with output.open("xb") as stream:
            created = True
            stream.write(payload)
    except Exception:
        if created:
            output.unlink(missing_ok=True)
        raise

    return {
        "width": PNG_WIDTH,
        "height": PNG_HEIGHT,
        "sequence_count": selected_sequence_count,
        "anchor_count": anchor_count,
        "visualized_sequence_count": visualized_sequence_count,
        "visualized_anchor_count": path_metadata["anchor_count_rendered"],
        "path_sequence_downsampled": path_metadata["sequence_downsampled"],
        "path_anchor_downsampled": path_metadata["anchor_downsampled"],
        "heatmap_rows_rendered": heatmap_metadata["rows_rendered"],
        "heatmap_rows_total": heatmap_metadata["rows_total"],
        "heatmap_downsampled": heatmap_metadata["downsampled"],
        "direction_rows_rendered": direction_metadata["rows_rendered"],
        "direction_rows_total": direction_metadata["rows_total"],
        "direction_downsampled": direction_metadata["downsampled"],
    }


__all__ = ["PNG_WIDTH", "PNG_HEIGHT", "render_trajectory_png"]
