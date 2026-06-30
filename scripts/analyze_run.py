"""Aggregate, summarize, and plot results for one timestamped experiment run."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SECTION_PLOTS: dict[str, list[str]] = {
    "comparison": [
        "scripts/plot_comparison_bar.py",
        "scripts/plot_comparison_radar.py",
        "scripts/plot_comparison_escape_sector_structure.py",
        "scripts/plot_comparison_capture_structure.py",
    ],
    "speed_pressure": [
        "scripts/plot_speed_pressure.py",
    ],
    "layer_validation": [
        "scripts/plot_layer_drop.py",
    ],
    "ablation": [
        "scripts/plot_sensitivity.py",
    ],
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze one experiment run directory")
    parser.add_argument("--run-dir", type=str, required=True, help="e.g. results/20250625_143052_comparison")
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument(
        "--skip-per-step",
        action="store_true",
        help="Skip expensive per-step diagnostics export; still writes trial-level aggregated CSV",
    )
    parser.add_argument("--section", type=str, default=None, help="Override manifest experiment_section for plots")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        raise SystemExit(f"Run directory not found: {run_dir}")

    from experiments.run_output import read_run_manifest

    manifest = read_run_manifest(run_dir) or {}
    section = args.section or manifest.get("experiment_section") or manifest.get("experiment_name") or "comparison"
    run_arg = ["--run-dir", str(run_dir)]

    steps = [
        (
            [
                sys.executable,
                str(ROOT / "scripts/aggregate_results.py"),
                *run_arg,
                *(["--skip-per-step"] if args.skip_per_step else []),
            ],
            "aggregate",
        ),
        (
            [sys.executable, str(ROOT / "scripts/summarize_experiments.py"), *run_arg],
            "summarize",
        ),
        (
            [sys.executable, str(ROOT / "scripts/compare_methods.py"), *run_arg, "--section", section],
            "compare",
        ),
    ]

    for cmd, label in steps:
        print(f"\n>>> [{label}] {' '.join(cmd)}")
        subprocess.run(cmd, check=True, cwd=str(ROOT))

    if not args.skip_plots:
        for script in SECTION_PLOTS.get(section, SECTION_PLOTS.get("comparison", [])):
            cmd = [sys.executable, str(ROOT / script), *run_arg, "--section", section]
            print(f"\n>>> [plot] {' '.join(cmd)}")
            subprocess.run(cmd, check=True, cwd=str(ROOT))

    print(f"\n=== Done ===")
    print(f"Run directory: {run_dir}")
    print(f"Aggregated:    {run_dir / 'aggregated_comparison.csv'}")
    if args.skip_per_step:
        print("Per-step:      skipped")
    else:
        print(f"Per-step:      {run_dir / 'per_step_metrics.csv'}")
    print(f"Summary:       {run_dir / 'summary'}")
    print(f"Figures:       {run_dir / 'figures'}")


if __name__ == "__main__":
    main()
