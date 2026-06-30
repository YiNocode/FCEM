"""Bar charts for full-circle structural diagnostics at capture windows."""

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

METHOD_COLORS = {
    "fcem": "#2563eb",
    "liao_mpc": "#16a34a",
    "ac_baseline": "#ca8a04",
    "pure_pursuit": "#dc2626",
    "open_marl": "#9333ea",
}

CAPTURE_METRIC_PANELS: tuple[tuple[str, str], ...] = (
    ("D_ang_full_at_capture", r"$D_{\mathrm{ang}}^{\mathrm{full}}$ at capture"),
    ("C_cov_full_at_capture", r"$C_{\mathrm{cov}}^{\mathrm{full}}$ at capture"),
    ("G_max_full_at_capture_deg", r"$G_{\max}^{\mathrm{full}}$ at capture (deg)"),
    ("D_ang_full_final5_mean", r"$D_{\mathrm{ang}}^{\mathrm{full}}$ mean (final 5 steps)"),
    ("C_cov_full_final5_mean", r"$C_{\mathrm{cov}}^{\mathrm{full}}$ mean (final 5 steps)"),
    ("G_max_full_final5_mean_deg", r"$G_{\max}^{\mathrm{full}}$ mean (final 5 steps, deg)"),
    ("D_ang_full_final10_mean", r"$D_{\mathrm{ang}}^{\mathrm{full}}$ mean (final 10 steps)"),
    ("C_cov_full_final10_mean", r"$C_{\mathrm{cov}}^{\mathrm{full}}$ mean (final 10 steps)"),
    ("G_max_full_final10_mean_deg", r"$G_{\max}^{\mathrm{full}}$ mean (final 10 steps, deg)"),
)


def _to_bool(val: object) -> bool:
    return str(val).lower() in ("true", "1", "yes")


def _to_float(val: object) -> float | None:
    if val in ("", None):
        return None
    try:
        v = float(val)
        if v != v:
            return None
        return v
    except (TypeError, ValueError):
        return None


def load_capture_structure_stats(csv_path: Path, section: str = "comparison") -> dict:
    """Per (method, scenario) lists of full-circle diagnostic metrics (successful trials only)."""
    fields = [col for col, _ in CAPTURE_METRIC_PANELS]
    stats: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(
        lambda: {f: [] for f in fields}
    )
    with csv_path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if section and row.get("experiment_section") != section:
                continue
            if not _to_bool(row.get("success", row.get("captured"))):
                continue
            key = (row["method"], row["scenario"])
            for field in fields:
                v = _to_float(row.get(field))
                if v is None:
                    legacy = {
                        "D_ang_full_at_capture": "capture_D_ang",
                        "C_cov_full_at_capture": "capture_C_cov",
                        "G_max_full_at_capture_deg": "capture_G_max_deg",
                        "D_ang_full_final5_mean": "pre_capture_5_D_ang",
                        "C_cov_full_final5_mean": "pre_capture_5_C_cov",
                        "G_max_full_final5_mean_deg": "pre_capture_5_G_max_deg",
                        "D_ang_full_final10_mean": "pre_capture_10_D_ang",
                        "C_cov_full_final10_mean": "pre_capture_10_C_cov",
                        "G_max_full_final10_mean_deg": "pre_capture_10_G_max_deg",
                    }.get(field)
                    if legacy:
                        v = _to_float(row.get(legacy))
                if v is not None:
                    stats[key][field].append(v)
    return stats


def plot_capture_structure_bars(
    csv_path: Path,
    out_path: Path,
    section: str = "comparison",
) -> None:
    stats = load_capture_structure_stats(csv_path, section)
    if not stats:
        print(f"No full-circle diagnostic rows in {csv_path} (section={section})")
        return

    methods = sorted({m for m, _ in stats})
    scenarios = sorted({s for _, s in stats})
    x = np.arange(len(scenarios))
    width = 0.8 / max(len(methods), 1)

    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    axes_flat = axes.ravel()

    for ax, (field, title) in zip(axes_flat, CAPTURE_METRIC_PANELS):
        for i, method in enumerate(methods):
            means = []
            for scenario in scenarios:
                vals = stats.get((method, scenario), {}).get(field, [])
                means.append(sum(vals) / len(vals) if vals else float("nan"))
            offset = (i - (len(methods) - 1) / 2) * width
            color = METHOD_COLORS.get(method)
            ax.bar(
                x + offset,
                means,
                width,
                label=method,
                color=color,
                edgecolor="white",
                linewidth=0.5,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(scenarios, rotation=15, ha="right")
        ax.set_title(title, fontsize=10)
        ax.grid(True, alpha=0.3, axis="y")

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(len(methods), 5), bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(
        "Full-circle structural diagnostics, not task capture criterion (successful trials)",
        y=1.06,
        fontsize=12,
    )
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
    out = Path(args.out) if args.out else paths.figures_dir / "comparison_full_circle_diagnostics.png"
    plot_capture_structure_bars(paths.aggregated_csv, out, section=args.section)


if __name__ == "__main__":
    main()
