"""Waterfall chart: performance drop when removing each layer."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _to_bool(val: object) -> bool:
    return str(val).lower() in ("true", "1", "yes")


def plot_layer_drop(csv_path: Path, out_path: Path, section: str = "layer_validation") -> None:
    stats: dict[tuple[str, str], list[bool]] = defaultdict(list)
    with csv_path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if section and row.get("experiment_section") != section:
                continue
            variant = row.get("variant") or row["method"].replace("fcem_", "")
            stats[(variant, row["scenario"])].append(_to_bool(row.get("success", row.get("captured"))))

    scenarios = sorted({s for _, s in stats})
    variants = ["full", "w_o_L1", "w_o_L2", "w_o_L3", "w_o_L4"]
    x = np.arange(len(variants))

    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.8 / max(len(scenarios), 1)

    for i, scenario in enumerate(scenarios):
        rates = []
        full_rate = 0.0
        for v in variants:
            data = stats.get((v, scenario), [])
            rate = sum(data) / len(data) if data else 0.0
            rates.append(rate)
            if v == "full":
                full_rate = rate
        drops = [full_rate - r for r in rates]
        offset = (i - (len(scenarios) - 1) / 2) * width
        ax.bar(x + offset, drops, width, label=scenario)

    ax.set_xticks(x)
    ax.set_xticklabels(variants, rotation=15, ha="right")
    ax.set_ylabel("Success rate drop vs full")
    ax.set_title("Layer-wise Validation: performance drop")
    ax.axhline(0, color="k", linewidth=0.8)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="results/aggregated.csv")
    parser.add_argument("--out", type=str, default="results/figures/layer_drop_waterfall.png")
    parser.add_argument("--section", type=str, default="layer_validation")
    args = parser.parse_args()
    plot_layer_drop(Path(args.csv), Path(args.out), section=args.section)


if __name__ == "__main__":
    main()
