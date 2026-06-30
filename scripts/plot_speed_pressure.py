"""Line plots: capture rate vs evader speed ratio (speed-pressure sweep)."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt

from experiments.run_paths import add_run_dir_arg, resolve_run_paths

METHOD_COLORS = {
    "fcem": "#2563eb",
    "liao_mpc": "#16a34a",
    "ac_baseline": "#ca8a04",
    "pure_pursuit": "#dc2626",
    "open_marl": "#9333ea",
}


def _to_bool(val: object) -> bool:
    return str(val).lower() in ("true", "1", "yes")


def plot_speed_pressure(csv_path: Path, out_path: Path, section: str = "speed_pressure") -> None:
    stats: dict[tuple[str, str, float], list[bool]] = defaultdict(list)
    with csv_path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if section and row.get("experiment_section") != section:
                continue
            param = row.get("sweep_param", "")
            value = row.get("sweep_value", "")
            if param != "speed_ratio" or value == "":
                continue
            try:
                ratio = float(value)
            except ValueError:
                continue
            key = (row["method"], row["scenario"], ratio)
            stats[key].append(_to_bool(row.get("success", row.get("captured"))))

    if not stats:
        print(f"No speed_ratio sweep rows in {csv_path} (section={section})")
        return

    scenarios = sorted({s for _, s, _ in stats})
    fig, axes = plt.subplots(1, len(scenarios), figsize=(5 * len(scenarios), 4), squeeze=False)

    for ax, scenario in zip(axes[0], scenarios):
        methods = sorted({m for m, s, _ in stats if s == scenario})
        for method in methods:
            points = sorted(
                (r, stats[(method, scenario, r)])
                for m, s, r in stats
                if m == method and s == scenario
            )
            xs = [r for r, _ in points]
            rates = [sum(ok) / len(ok) if ok else 0.0 for _, ok in points]
            ax.plot(
                xs,
                rates,
                marker="o",
                label=method,
                color=METHOD_COLORS.get(method),
                linewidth=2,
            )
        ax.set_title(scenario)
        ax.set_xlabel(r"Speed ratio $v_e / v_p$")
        ax.set_ylabel("Capture rate")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.suptitle("Speed pressure: capture rate vs evader speed advantage", y=1.02)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    add_run_dir_arg(parser)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--section", type=str, default="speed_pressure")
    args = parser.parse_args()
    paths = resolve_run_paths(args, experiment_name="speed_pressure")
    out = Path(args.out) if args.out else paths.figures_dir / "speed_pressure.png"
    plot_speed_pressure(paths.aggregated_csv, out, section=args.section)


if __name__ == "__main__":
    main()
