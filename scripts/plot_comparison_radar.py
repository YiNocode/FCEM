"""Radar chart for multi-scenario comparative evaluation."""

from __future__ import annotations

import argparse
import csv
import math
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


def _empty_stats() -> dict:
    return {"success": [], "ttc": [], "d_ang": [], "c_cov": [], "g_max": []}


def load_method_scenario_stats(csv_path: Path, section: str = "comparison") -> dict:
    stats: dict[tuple[str, str], dict] = defaultdict(
        lambda: {"success": [], "ttc": [], "d_ang": [], "c_cov": [], "g_max": []}
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
            for field, dest in (
                ("mean_D_ang", "d_ang"),
                ("mean_C_cov", "c_cov"),
                ("mean_G_max_deg", "g_max"),
            ):
                v = _to_float(row.get(field))
                if v is not None:
                    stats[key][dest].append(v)
    return stats


def _normalize(values: list[float], higher_better: bool) -> list[float]:
    if not values:
        return [0.0] * 5
    vmin, vmax = min(values), max(values)
    if math.isclose(vmin, vmax):
        return [1.0 if higher_better else 0.0 for _ in values]
    out = []
    for v in values:
        norm = (v - vmin) / (vmax - vmin)
        out.append(norm if higher_better else 1.0 - norm)
    return out


def plot_radar(csv_path: Path, out_path: Path, section: str = "comparison") -> None:
    stats = load_method_scenario_stats(csv_path, section)
    methods = sorted({m for m, _ in stats})
    scenarios = sorted({s for _, s in stats})

    metric_names = ["success", "1/T_capture", "D_ang", "C_cov", "1/G_max"]
    angles = np.linspace(0, 2 * np.pi, len(metric_names), endpoint=False).tolist()
    angles += angles[:1]

    fig, axes = plt.subplots(1, len(scenarios), figsize=(5 * len(scenarios), 5), subplot_kw=dict(polar=True))
    if len(scenarios) == 1:
        axes = [axes]

    for ax, scenario in zip(axes, scenarios):
        raw_by_method: dict[str, list[float]] = {}
        for method in methods:
            data = stats.get((method, scenario)) or _empty_stats()
            n = len(data["success"])
            success = sum(data["success"]) / n if n else 0.0
            ttc = sum(data["ttc"]) / len(data["ttc"]) if data["ttc"] else float("inf")
            inv_ttc = 1.0 / ttc if ttc and ttc < float("inf") else 0.0
            d_ang = sum(data["d_ang"]) / len(data["d_ang"]) if data["d_ang"] else 0.0
            c_cov = sum(data["c_cov"]) / len(data["c_cov"]) if data["c_cov"] else 0.0
            g_max = sum(data["g_max"]) / len(data["g_max"]) if data["g_max"] else 180.0
            inv_g = 1.0 / g_max if g_max > 0 else 0.0
            raw_by_method[method] = [success, inv_ttc, d_ang, c_cov, inv_g]

        for metric_idx in range(5):
            col = [raw_by_method[m][metric_idx] for m in methods]
            higher = metric_idx != 4  # 1/G_max: lower G_max is better
            normed_col = _normalize(col, higher_better=higher)
            for method_idx, method in enumerate(methods):
                raw_by_method[method][metric_idx] = normed_col[method_idx]

        for method in methods:
            values = raw_by_method[method] + raw_by_method[method][:1]
            ax.plot(angles, values, label=method)
            ax.fill(angles, values, alpha=0.1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metric_names)
        ax.set_ylim(0, 1)
        ax.set_title(scenario)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

    fig.suptitle("Comparative Evaluation (normalized radar)")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="results/aggregated.csv")
    parser.add_argument("--out", type=str, default="results/figures/comparison_radar.png")
    parser.add_argument("--section", type=str, default="comparison")
    args = parser.parse_args()
    plot_radar(Path(args.csv), Path(args.out), section=args.section)


if __name__ == "__main__":
    main()
