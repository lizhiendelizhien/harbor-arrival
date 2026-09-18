"""Reconstruct variable-length monthly finance trajectories from the canonical table.

The command writes one mapping-wrapped PKL per continuous valid symbol segment.
Use ``--dry-run`` to inspect counts without creating an output directory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gear_sonic.finance.trajectory import (
    DEFAULT_MIN_TRAINING_LENGTH,
    reconstruct_monthly_trajectories,
)


def _statuses(values: list[str] | None) -> tuple[str, ...]:
    if not values:
        return ("confirmed",)
    result: list[str] = []
    for value in values:
        result.extend(part.strip() for part in value.split(",") if part.strip())
    return tuple(result)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--canonical", type=Path, required=True,
        help="cleaned monthly_canonical.csv",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="new output directory (must not already exist)",
    )
    parser.add_argument(
        "--event-breaks", type=Path, default=None,
        help="optional CSV with symbol,period,reason,status,action,reference",
    )
    parser.add_argument(
        "--apply-status", action="append", default=None,
        help="event status to apply; repeat or use comma-separated values (default: confirmed)",
    )
    parser.add_argument(
        "--min-training-length", type=int, default=DEFAULT_MIN_TRAINING_LENGTH,
        help="minimum segment length considered training-eligible (default: 17)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="scan and report counts without writing output files",
    )
    args = parser.parse_args(argv)
    report = reconstruct_monthly_trajectories(
        args.canonical,
        args.output_dir,
        event_breaks=args.event_breaks,
        apply_statuses=_statuses(args.apply_status),
        min_training_length=args.min_training_length,
        dry_run=args.dry_run,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
