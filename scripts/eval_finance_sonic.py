"""Evaluate complete privileged reference clips with checkpoint-inherited settings.

Run with python -m scripts.eval_finance_sonic. Tracking metrics use real future
descriptors and do not measure deployable forecast accuracy or financial PnL.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from gear_sonic.finance.evaluation import TRAJECTORY_FIELDNAMES, evaluate_checkpoint
from gear_sonic.finance.visualization import render_trajectory_png


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
    parser.add_argument(
        "--dump-trajectories", action="store_true",
        help="Write one deterministic prediction/target row per anchor and horizon",
    )
    parser.add_argument(
        "--max-sequences", type=int,
        help="Limit playback to the first N reference clips (useful for inspection)",
    )
    parser.add_argument(
        "--plot-sequences", type=int, default=4,
        help="Number of initial clips to include in trajectory_overview.png",
    )
    return parser.parse_args(argv)


def _write_csv(path, rows, fieldnames, created_outputs):
    with path.open("x", encoding="utf-8", newline="") as stream:
        created_outputs.append(path)
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    args = _parse_args(argv)
    for name in ("num_envs", "threads"):
        if getattr(args, name) < 1:
            raise ValueError(f"{name} must be positive")
    if args.max_sequences is not None and args.max_sequences < 1:
        raise ValueError("max_sequences must be positive")
    if args.plot_sequences < 1:
        raise ValueError("plot_sequences must be positive")
    if args.dump_trajectories and args.output_dir is None:
        raise ValueError("--dump-trajectories requires --output-dir for streaming output")
    if args.output_dir is not None:
        if args.output_dir.exists() and not args.output_dir.is_dir():
            raise NotADirectoryError(args.output_dir)
        output_names = ["summary.json", "sequences.csv", "symbols.csv"]
        if args.dump_trajectories:
            output_names.extend(("trajectories.csv", "trajectory_overview.png"))
        for name in output_names:
            path = args.output_dir / name
            if path.exists() or path.is_symlink():
                raise FileExistsError(f"Refusing to overwrite evaluation output: {path}")

    evaluate_kwargs = {
        "device": args.device, "num_envs": args.num_envs, "threads": args.threads,
        "canonical": args.canonical, "symbols": args.symbols, "start": args.start,
        "end": args.end, "custom_reference": args.custom_reference,
        "reference_config": args.reference_config, "max_sequences": args.max_sequences,
    }
    created_outputs = []
    try:
        if args.output_dir is not None:
            args.output_dir.mkdir(parents=True, exist_ok=True)
        if args.dump_trajectories:
            trajectory_path = args.output_dir / "trajectories.csv"
            plot_rows = []
            plot_sequence_ids = set()

            def write_and_capture(row):
                writer.writerow(row)
                sequence_id = int(row["sequence_id"])
                # Evaluation normally emits zero-based IDs, but selecting by
                # encounter order also handles custom callers with sparse IDs.
                if (sequence_id not in plot_sequence_ids
                        and len(plot_sequence_ids) < args.plot_sequences):
                    plot_sequence_ids.add(sequence_id)
                if sequence_id in plot_sequence_ids:
                    plot_rows.append(row)

            with trajectory_path.open("x", encoding="utf-8", newline="") as stream:
                created_outputs.append(trajectory_path)
                writer = csv.DictWriter(stream, fieldnames=TRAJECTORY_FIELDNAMES)
                writer.writeheader()
                report = evaluate_checkpoint(
                    args.checkpoint, include_trajectories=False,
                    trajectory_callback=write_and_capture, **evaluate_kwargs,
                )
            plot_path = args.output_dir / "trajectory_overview.png"
            plot_metadata = render_trajectory_png(
                plot_rows, plot_path,
                title=(f"{report['mode']} | checkpoint {report['checkpoint_iteration']}"),
                max_sequences=args.plot_sequences,
            )
            created_outputs.append(plot_path)
        else:
            report = evaluate_checkpoint(
                args.checkpoint, include_trajectories=False, **evaluate_kwargs,
            )
        summary = {key: value for key, value in report.items()
                   if key not in ("sequences", "symbols", "trajectories")}
        if args.dump_trajectories:
            summary.update({
                "plot_path": str(plot_path),
                "plot_sequence_limit": args.plot_sequences,
                "plot_sequence_count": plot_metadata["sequence_count"],
                "plot_sequence_total": report["sequence_count"],
                "plot_visualized_sequence_count": plot_metadata["visualized_sequence_count"],
                "plot_anchor_count": plot_metadata["anchor_count"],
                "plot_anchor_total": report["anchor_count"],
                "plot_visualized_anchor_count": plot_metadata["visualized_anchor_count"],
                "plot_heatmap_rows": [
                    plot_metadata["heatmap_rows_rendered"],
                    plot_metadata["heatmap_rows_total"],
                ],
                "plot_direction_rows": [
                    plot_metadata["direction_rows_rendered"],
                    plot_metadata["direction_rows_total"],
                ],
                "plot_dimensions": [plot_metadata["width"], plot_metadata["height"]],
            })
        serialized = json.dumps(summary, allow_nan=False, indent=2)
        if args.output_dir is not None:
            for name in ("sequences", "symbols"):
                _write_csv(
                    args.output_dir / f"{name}.csv", report[name],
                    list(report[name][0]), created_outputs,
                )
            summary_path = args.output_dir / "summary.json"
            with summary_path.open("x", encoding="utf-8") as stream:
                created_outputs.append(summary_path)
                stream.write(serialized + "\n")
    except Exception:
        for path in reversed(created_outputs):
            path.unlink(missing_ok=True)
        raise
    print(serialized)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
