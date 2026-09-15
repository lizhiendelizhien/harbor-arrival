"""Evaluate complete privileged reference clips with checkpoint-inherited settings.

Run with python -m scripts.eval_finance_sonic. Tracking metrics use real future
descriptors and do not measure deployable forecast accuracy or financial PnL.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from gear_sonic.finance.evaluation import evaluate_checkpoint


def _parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--canonical", type=Path)
    parser.add_argument("--symbols", nargs="+")
    parser.add_argument("--start", help="Inclusive first anchor month, YYYY-MM")
    parser.add_argument("--end", help="Inclusive final label month, YYYY-MM")
    parser.add_argument("--custom-reference", action="store_true")
    parser.add_argument("--reference-config", type=Path, help="Explicit reference JSON for legacy checkpoints")
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    for name in ("num_envs", "threads"):
        if getattr(args, name) < 1:
            raise ValueError(f"{name} must be positive")
    if args.output_dir is not None:
        if args.output_dir.exists() and not args.output_dir.is_dir():
            raise NotADirectoryError(args.output_dir)
        for name in ("summary.json", "sequences.csv", "symbols.csv"):
            path = args.output_dir / name
            if path.exists() or path.is_symlink():
                raise FileExistsError(f"Refusing to overwrite evaluation output: {path}")
    report = evaluate_checkpoint(
        args.checkpoint, device=args.device, num_envs=args.num_envs, threads=args.threads,
        canonical=args.canonical, symbols=args.symbols, start=args.start, end=args.end,
        custom_reference=args.custom_reference, reference_config=args.reference_config,
    )
    summary = {key: value for key, value in report.items() if key not in ("sequences", "symbols")}
    serialized = json.dumps(summary, allow_nan=False, indent=2)
    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for name in ("sequences", "symbols"):
            with (args.output_dir / f"{name}.csv").open("x", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(report[name][0]))
                writer.writeheader()
                writer.writerows(report[name])
        with (args.output_dir / "summary.json").open("x", encoding="utf-8") as stream:
            stream.write(serialized + "\n")
    print(serialized)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
