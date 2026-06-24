"""Run all FCEM experiment sections and aggregate results."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Default dynamics for the full suite: v_e / v_p = 2.5
DEFAULT_PURSUER_VMAX = 4.0
DEFAULT_EVADER_VMAX = 10.0
DEFAULT_PURSUER_AMAX = 3.2
DEFAULT_EVADER_AMAX = 4.0

SECTION_RUNNERS = {
    "layer_validation": "experiments/run_layer_validation_2d.py",
    "comparison": "experiments/run_comparison_2d.py",
    "combination": "experiments/run_combination_ablation_2d.py",
    "sensitivity": "experiments/run_sensitivity_2d.py",
    "components": "experiments/run_ablation_2d.py",
}

PLOT_SCRIPTS = [
    ("scripts/generate_layer_table.py", []),
    ("scripts/plot_comparison_bar.py", []),
    ("scripts/plot_comparison_radar.py", []),
    ("scripts/plot_layer_drop.py", []),
    ("scripts/plot_sensitivity.py", []),
]


def _run_script(rel_path: str, extra_args: list[str]) -> None:
    cmd = [sys.executable, str(ROOT / rel_path), *extra_args]
    print(f"\n>>> {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run FCEM experiment suite")
    parser.add_argument(
        "--sections",
        type=str,
        default="layer_validation,comparison,combination,sensitivity",
        help="Comma-separated: layer_validation,comparison,combination,sensitivity,components",
    )
    parser.add_argument("--trials", type=int, default=None)
    parser.add_argument("--scenarios", nargs="+", default=None)
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--pursuer-vmax",
        type=float,
        default=DEFAULT_PURSUER_VMAX,
        help="pursuer max speed (default: 4.0, ratio v_e/v_p=2.5 with default evader)",
    )
    parser.add_argument(
        "--evader-vmax",
        type=float,
        default=DEFAULT_EVADER_VMAX,
        help="evader max speed (default: 10.0)",
    )
    parser.add_argument("--pursuer-amax", type=float, default=DEFAULT_PURSUER_AMAX)
    parser.add_argument("--evader-amax", type=float, default=DEFAULT_EVADER_AMAX)
    args = parser.parse_args()

    sections = [s.strip() for s in args.sections.split(",") if s.strip()]
    trial_args = ["--trials", str(args.trials)] if args.trials is not None else []
    scenario_args = ["--scenarios", *args.scenarios] if args.scenarios else []
    dynamics_args = [
        "--pursuer-vmax", str(args.pursuer_vmax),
        "--evader-vmax", str(args.evader_vmax),
        "--pursuer-amax", str(args.pursuer_amax),
        "--evader-amax", str(args.evader_amax),
    ]
    ratio = args.evader_vmax / args.pursuer_vmax
    print(f"Suite dynamics: v_p={args.pursuer_vmax}, v_e={args.evader_vmax} (ratio={ratio:.2f})")

    for section in sections:
        runner = SECTION_RUNNERS.get(section)
        if not runner:
            raise ValueError(f"Unknown section: {section}. Choose from {list(SECTION_RUNNERS)}")
        cmd = [sys.executable, str(ROOT / runner), *trial_args, *scenario_args, *dynamics_args]
        print(f"\n>>> {' '.join(cmd)}")
        if args.dry_run:
            continue
        subprocess.run(cmd, check=True, cwd=str(ROOT))

    if args.dry_run:
        print("\n[dry-run] Skipping aggregate and plots")
        return

    _run_script("scripts/aggregate_results.py", ["--results-dir", "results", "--out", "results/aggregated.csv"])
    _run_script("scripts/summarize_experiments.py", ["--csv", "results/aggregated.csv", "--out-dir", "results/summary"])

    if not args.skip_plots:
        for script, extra in PLOT_SCRIPTS:
            _run_script(script, extra)

    print("\n=== Done ===")
    print("Aggregated: results/aggregated.csv")
    print("Summaries:  results/summary/")
    print("Figures:    results/figures/")


if __name__ == "__main__":
    main()
