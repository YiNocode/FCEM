"""Radar chart for multi-scenario comparative evaluation."""

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
from metrics.comparison_metrics import RADAR_G_MAX_WORST_DEG, radar_fixed_scale


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
    return {
        "success": [],
        "ttc": [],
        "d_ang": [],
        "c_cov": [],
        "g_max": [],
        "c_sync": [],
    }


def load_method_scenario_stats(csv_path: Path, section: str = "comparison") -> dict:
    """
    Load per-trial stats for radar plotting.

    Structural axes use pre-capture canonical metrics (successful trials only).
    Speed axis uses timeout-adjusted TTC averaged over all trials.
    """
    stats: dict[tuple[str, str], dict] = defaultdict(_empty_stats)
    with csv_path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if section and row.get("experiment_section") != section:
                continue
            key = (row["method"], row["scenario"])
            captured = _to_bool(row.get("success", row.get("captured")))
            stats[key]["success"].append(captured)
            ttc_adj = _to_float(row.get("time_to_capture_adj_s"))
            if ttc_adj is not None:
                stats[key]["ttc"].append(ttc_adj)
            if not captured:
                continue
            for field, dest in (
                ("pre_capture_canonical_D_ang", "d_ang"),
                ("pre_capture_canonical_C_cov", "c_cov"),
                ("pre_capture_canonical_G_max_deg", "g_max"),
                ("pre_capture_canonical_C_sync", "c_sync"),
            ):
                v = _to_float(row.get(field))
                if v is not None:
                    stats[key][dest].append(v)
    return stats


def plot_radar(csv_path: Path, out_path: Path, section: str = "comparison") -> None:
    stats = load_method_scenario_stats(csv_path, section)
    methods = sorted({m for m, _ in stats})
    scenarios = sorted({s for _, s in stats})

    metric_names = ["success", "1/T_adj", "D_ang", "C_cov", "ang_closure"]
    metric_keys = ["success", "inv_ttc", "d_ang", "c_cov", "inv_g"]
    angles = np.linspace(0, 2 * np.pi, len(metric_names), endpoint=False).tolist()
    angles += angles[:1]

    fig, axes = plt.subplots(1, len(scenarios), figsize=(5 * len(scenarios), 5), subplot_kw=dict(polar=True))
    if len(scenarios) == 1:
        axes = [axes]

    for ax, scenario in zip(axes, scenarios):
        for method in methods:
            data = stats.get((method, scenario)) or _empty_stats()
            n = len(data["success"])
            success = sum(data["success"]) / n if n else 0.0
            ttc = sum(data["ttc"]) / len(data["ttc"]) if data["ttc"] else 0.0
            d_ang = sum(data["d_ang"]) / len(data["d_ang"]) if data["d_ang"] else 0.0
            c_cov = sum(data["c_cov"]) / len(data["c_cov"]) if data["c_cov"] else 0.0
            g_max = sum(data["g_max"]) / len(data["g_max"]) if data["g_max"] else RADAR_G_MAX_WORST_DEG
            raw = {
                "success": success,
                "inv_ttc": ttc,
                "d_ang": d_ang,
                "c_cov": c_cov,
                "inv_g": g_max,
            }
            values = [radar_fixed_scale(key, raw[key]) for key in metric_keys]
            values += values[:1]
            ax.plot(angles, values, label=method)
            ax.fill(angles, values, alpha=0.1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metric_names)
        ax.set_ylim(0, 1)
        ax.set_title(scenario)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

    fig.suptitle("Comparative Evaluation (adj TTC + pre-capture canonical, fixed scale)")
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
    out = Path(args.out) if args.out else paths.figures_dir / "comparison_radar.png"
    plot_radar(paths.aggregated_csv, out, section=args.section)


if __name__ == "__main__":
    main()
