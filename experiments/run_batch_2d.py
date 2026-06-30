"""Batch 2D experiment runner."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch 2D experiments")
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--methods", nargs="+", default=["fcem", "pure_pursuit", "liao_mpc", "ac_baseline"])
    parser.add_argument("--scenarios", nargs="+", default=["free", "random_obstacles", "single_exit"])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cmd = [
        sys.executable,
        str(ROOT / "experiments" / "run_comparison_2d.py"),
        "--trials", str(args.trials),
        "--methods", *args.methods,
        "--scenarios", *args.scenarios,
    ]
    if args.dry_run:
        print("Would run:", " ".join(cmd))
        return
    subprocess.run(cmd, check=True, cwd=str(ROOT))


if __name__ == "__main__":
    main()
