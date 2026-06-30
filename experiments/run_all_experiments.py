"""Run all FCEM experiment sections and aggregate results."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Default dynamics for the full suite: v_e / v_p = 2.0
DEFAULT_PURSUER_VMAX = 4.0
DEFAULT_EVADER_VMAX = 8.0
DEFAULT_PURSUER_AMAX = 3.2
DEFAULT_EVADER_AMAX = 3.2

SECTION_RUNNERS = {
    "layer_validation": ("experiments/run_layer_validation_2d.py", "layer_validation"),
    "comparison": ("experiments/run_comparison_2d.py", "comparison"),
    "speed_pressure": ("experiments/run_speed_pressure_2d.py", "speed_pressure"),
    "combination": ("experiments/run_combination_ablation_2d.py", "ablation_combination"),
    "sensitivity": ("experiments/run_sensitivity_2d.py", "ablation_sensitivity"),
    "components": ("experiments/run_ablation_2d.py", "ablation_components"),
}


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
        help="Comma-separated: layer_validation,comparison,speed_pressure,combination,sensitivity,components",
    )
    parser.add_argument("--trials", type=int, default=None)
    parser.add_argument("--scenarios", nargs="+", default=None)
    parser.add_argument("--skip-analyze", action="store_true", help="Skip per-run analyze step at the end")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--pursuer-vmax",
        type=float,
        default=DEFAULT_PURSUER_VMAX,
        help="pursuer max speed (default: 4.0, ratio v_e/v_p=2.0 with default evader)",
    )
    parser.add_argument(
        "--evader-vmax",
        type=float,
        default=DEFAULT_EVADER_VMAX,
        help="evader max speed (default: 8.0)",
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

    from experiments.run_output import list_run_dirs

    results_root = ROOT / "results"
    before_runs = {name: {p.name for p in list_run_dirs(results_root, name)} for _, name in SECTION_RUNNERS.values()}

    for section in sections:
        entry = SECTION_RUNNERS.get(section)
        if not entry:
            raise ValueError(f"Unknown section: {section}. Choose from {list(SECTION_RUNNERS)}")
        runner, _ = entry
        cmd = [sys.executable, str(ROOT / runner), *trial_args, *scenario_args, *dynamics_args]
        print(f"\n>>> {' '.join(cmd)}")
        if args.dry_run:
            continue
        subprocess.run(cmd, check=True, cwd=str(ROOT))

    if args.dry_run:
        print("\n[dry-run] Skipping analyze")
        return

    print("\n=== Run directories (this session) ===")
    new_run_dirs: list[Path] = []
    for section in sections:
        _, exp_name = SECTION_RUNNERS[section]
        after = {p.name for p in list_run_dirs(results_root, exp_name)}
        added = sorted(after - before_runs.get(exp_name, set()))
        if added:
            run_path = results_root / added[-1]
            new_run_dirs.append(run_path)
            print(f"  {section}: {run_path}")
        else:
            print(f"  {section}: (no new timestamped run detected)")

    if args.skip_analyze:
        print("\nSkipped analyze. For each run directory:")
        print("  python scripts/analyze_run.py --run-dir results/<timestamp>_<experiment>")
        return

    for run_dir in new_run_dirs:
        _run_script("scripts/analyze_run.py", ["--run-dir", str(run_dir.relative_to(ROOT))])

    print("\n=== Done ===")
    print("Each experiment run is under results/<YYYYMMDD_HHMMSS>_<experiment>/")
    print("Analyze manually: python scripts/analyze_run.py --run-dir results/<timestamp>_<experiment>")


if __name__ == "__main__":
    main()
