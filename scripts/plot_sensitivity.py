"""Line plots for hyperparameter sensitivity sweeps."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


def _to_bool(val: object) -> bool:
    return str(val).lower() in ("true", "1", "yes")


def _to_float(val: object) -> float | None:
    if val in ("", None):
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def plot_sensitivity(csv_path: Path, out_dir: Path, section: str = "ablation") -> None:
    stats: dict[tuple[str, str, float], dict] = defaultdict(
        lambda: {"success": [], "ttc": []}
    )
    with csv_path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            param = row.get("sweep_param", "")
            value = row.get("sweep_value", "")
            if not param or value == "":
                continue
            try:
                v = float(value)
            except ValueError:
                continue
            key = (param, row["scenario"], v)
            stats[key]["success"].append(_to_bool(row.get("success", row.get("captured"))))
            if _to_bool(row.get("success", row.get("captured"))):
                ttc = _to_float(row.get("time_to_capture_s"))
                if ttc is not None:
                    stats[key]["ttc"].append(ttc)

    params = sorted({p for p, _, _ in stats})
    out_dir.mkdir(parents=True, exist_ok=True)

    for param in params:
        scenarios = sorted({s for p, s, _ in stats if p == param})
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for scenario in scenarios:
            points = sorted(
                (v, stats[(param, scenario, v)])
                for p, s, v in stats
                if p == param and s == scenario
            )
            xs = [v for v, _ in points]
            rates = [sum(d["success"]) / len(d["success"]) if d["success"] else 0.0 for _, d in points]
            ttcs = [sum(d["ttc"]) / len(d["ttc"]) if d["ttc"] else 0.0 for _, d in points]
            axes[0].plot(xs, rates, marker="o", label=scenario)
            axes[1].plot(xs, ttcs, marker="o", label=scenario)

        axes[0].set_title(f"{param}: success rate")
        axes[1].set_title(f"{param}: mean time-to-capture (s)")
        for ax in axes:
            ax.set_xlabel(param)
            ax.legend()
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
        safe = param.replace(".", "_")
        out_path = out_dir / f"sensitivity_{safe}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="results/aggregated.csv")
    parser.add_argument("--out-dir", type=str, default="results/figures")
    parser.add_argument("--section", type=str, default="ablation")
    args = parser.parse_args()
    plot_sensitivity(Path(args.csv), Path(args.out_dir), section=args.section)


if __name__ == "__main__":
    main()
