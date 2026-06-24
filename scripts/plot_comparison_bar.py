"""Bar charts for comparative evaluation (success rate + time-to-capture)."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


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
        lambda: {"success": [], "ttc": []}
    )
    with csv_path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if section and row.get("experiment_section") != section:
                continue
            key = (row["method"], row["scenario"])
            stats[key]["success"].append(_to_bool(row.get("success", row.get("captured"))))
            if _to_bool(row.get("success", row.get("captured"))):
                ttc = _to_float(row.get("time_to_capture_s"))
                if ttc is not None:
                    stats[key]["ttc"].append(ttc)
    return stats


def plot_bars(csv_path: Path, out_path: Path, section: str = "comparison") -> None:
    stats = load_comparison_stats(csv_path, section)
    methods = sorted({m for m, _ in stats})
    scenarios = sorted({s for _, s in stats})

    x = np.arange(len(scenarios))
    width = 0.8 / max(len(methods), 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for i, method in enumerate(methods):
        rates = []
        ttcs = []
        for scenario in scenarios:
            data = stats.get((method, scenario)) or {"success": [], "ttc": []}
            n = len(data["success"])
            rates.append(sum(data["success"]) / n if n else 0.0)
            ttcs.append(sum(data["ttc"]) / len(data["ttc"]) if data["ttc"] else 0.0)
        offset = (i - (len(methods) - 1) / 2) * width
        axes[0].bar(x + offset, rates, width, label=method)
        axes[1].bar(x + offset, ttcs, width, label=method)

    for ax, title, ylabel in zip(
        axes,
        ("Success Rate", "Mean Time-to-Capture (s)"),
        ("success rate", "time (s)"),
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
    parser.add_argument("--csv", type=str, default="results/aggregated.csv")
    parser.add_argument("--out", type=str, default="results/figures/comparison_bar.png")
    parser.add_argument("--section", type=str, default="comparison")
    args = parser.parse_args()
    plot_bars(Path(args.csv), Path(args.out), section=args.section)


if __name__ == "__main__":
    main()
