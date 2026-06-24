"""Plot structure metrics over time from a single run JSON."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt


def _get_steps(data: dict) -> list:
    return data.get("records") or data.get("steps") or []


def plot_structure(run_json: Path, out_path: Path, dt: float = 0.1) -> None:
    data = json.loads(run_json.read_text(encoding="utf-8"))
    steps = _get_steps(data)
    t = [s["step"] * dt for s in steps]
    d_ang = [s.get("D_ang") or s["metrics"]["D_ang"] for s in steps]
    c_cov = [s.get("C_cov") or s["metrics"]["C_cov"] for s in steps]
    g_max = [math.degrees(s.get("G_max") or s["metrics"]["G_max"]) for s in steps]
    c_col = [s.get("C_col") or s["metrics"]["C_col"] for s in steps]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes[0, 0].plot(t, d_ang)
    axes[0, 0].set_title("D_ang")
    axes[0, 1].plot(t, c_cov)
    axes[0, 1].set_title("C_cov")
    axes[1, 0].plot(t, g_max)
    axes[1, 0].set_title("G_max (deg)")
    axes[1, 1].plot(t, c_col)
    axes[1, 1].set_title("C_col")
    for ax in axes.ravel():
        ax.grid(True, alpha=0.4)
        ax.set_xlabel("time (s)")
    trial = data.get("trial", data.get("trial_id", "?"))
    fig.suptitle(f"{data.get('method')} / {data.get('scenario')} trial {trial}")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_json", type=str)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()
    run_path = Path(args.run_json)
    out = Path(args.out) if args.out else run_path.with_suffix(".structure.png")
    plot_structure(run_path, out)


if __name__ == "__main__":
    main()
