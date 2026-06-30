"""Bar charts for comparative evaluation (success rate + timeout-adjusted TTC)."""

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
import numpy as np

from experiments.run_paths import add_run_dir_arg, resolve_run_paths


def _to_bool(val: object) -> bool:
    return str(val).lower() in ("true", "1", "yes")


def _to_float(val: object) -> float | None:
    if val in ("", None):
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def load_comparison_stats(csv_path: Path, section: str = "comparison") -> dict:
    stats: dict[tuple[str, str], dict] = defaultdict(
        lambda: {"success": [], "ttc_cond": [], "ttc_adj": []}
    )
    with csv_path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if section and row.get("experiment_section") != section:
                continue
            key = (row["method"], row["scenario"])
            captured = _to_bool(row.get("success", row.get("captured")))
            stats[key]["success"].append(captured)
            adj = _to_float(row.get("time_to_capture_adj_s"))
            if adj is not None:
                stats[key]["ttc_adj"].append(adj)
            if captured:
                ttc = _to_float(row.get("time_to_capture_s"))
                if ttc is not None:
                    stats[key]["ttc_cond"].append(ttc)
    return stats


def plot_bars(csv_path: Path, out_path: Path, section: str = "comparison") -> None:
    stats = load_comparison_stats(csv_path, section)
    methods = sorted({m for m, _ in stats})
    scenarios = sorted({s for _, s in stats})

    x = np.arange(len(scenarios))
    width = 0.8 / max(len(methods), 1)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for i, method in enumerate(methods):
        rates = []
        ttcs_cond = []
        ttcs_adj = []
        for scenario in scenarios:
            data = stats.get((method, scenario)) or {"success": [], "ttc_cond": [], "ttc_adj": []}
            n = len(data["success"])
            rates.append(sum(data["success"]) / n if n else 0.0)
            ttcs_cond.append(sum(data["ttc_cond"]) / len(data["ttc_cond"]) if data["ttc_cond"] else 0.0)
            ttcs_adj.append(sum(data["ttc_adj"]) / len(data["ttc_adj"]) if data["ttc_adj"] else 0.0)
        offset = (i - (len(methods) - 1) / 2) * width
        axes[0].bar(x + offset, rates, width, label=method)
        axes[1].bar(x + offset, ttcs_cond, width, label=method)
        axes[2].bar(x + offset, ttcs_adj, width, label=method)

    for ax, title, ylabel in zip(
        axes,
        (
            "Success Rate",
            "Conditional TTC (successful trials)",
            "Adjusted TTC (all trials, fail=T_max)",
        ),
        ("success rate", "time (s)", "time (s)"),
    ):
        ax.set_xticks(x)
        ax.set_xticklabels(scenarios, rotation=15, ha="right")
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Comparative Evaluation")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    add_run_dir_arg(parser)
    parser.add_argument("--csv", type=str, default=None)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--section", type=str, default="comparison")
    args = parser.parse_args()
    paths = resolve_run_paths(args, experiment_name="comparison")
    out = Path(args.out) if args.out else paths.figures_dir / "comparison_performance.png"
    plot_bars(paths.aggregated_csv, out, section=args.section)


if __name__ == "__main__":
    main()
