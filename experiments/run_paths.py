"""Shared CLI helpers for analysis scripts (--run-dir)."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from experiments.run_output import latest_run_dir


@dataclass
class RunPaths:
    run_dir: Path
    aggregated_csv: Path
    summary_dir: Path
    figures_dir: Path


def add_run_dir_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Experiment run directory (e.g. results/20250625_143052_comparison)",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Alias of --run-dir (backward compatible with aggregate_results.py)",
    )


def resolve_run_paths(
    args: argparse.Namespace,
    *,
    default_csv: str | None = None,
    require_run_dir: bool = False,
    experiment_name: str | None = None,
) -> RunPaths:
    run_dir_arg = getattr(args, "run_dir", None) or getattr(args, "results_dir", None)
    if run_dir_arg:
        run_dir = Path(run_dir_arg)
    else:
        latest = latest_run_dir(Path("results"), experiment_name)
        if latest is not None:
            run_dir = latest
            print(f"Using latest run directory: {run_dir}")
        elif require_run_dir:
            raise SystemExit(
                "No --run-dir specified and no timestamped run found under results/. "
                "Run an experiment first or pass --run-dir explicitly."
            )
        else:
            run_dir = Path("results")

    csv_arg = getattr(args, "csv", None) or default_csv
    aggregated = Path(csv_arg) if csv_arg else run_dir / "aggregated_comparison.csv"
    return RunPaths(
        run_dir=run_dir,
        aggregated_csv=aggregated,
        summary_dir=run_dir / "summary",
        figures_dir=run_dir / "figures",
    )
