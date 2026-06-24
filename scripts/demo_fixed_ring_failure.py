#!/usr/bin/env python3
"""Demonstrate fixed_ring failure on the asymmetric fast-breakaway scenario."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.fixed_ring_apf import make_fixed_ring_controller
from baselines.registry import METHODS
from envs.sim2d import Sim2D, make_fcem_controller
from experiments.config_loader import load_config, obstacles_from_scenario


def run_method(method: str, seed: int) -> dict:
    cfg = load_config("fixed_ring_failure")
    cfg["seed"] = seed
    obstacles = obstacles_from_scenario(cfg["scenario"])
    if method == "fcem":
        controller = make_fcem_controller()
    elif method == "fixed_ring":
        controller = make_fixed_ring_controller()
    else:
        controller = METHODS[method]()
    sim = Sim2D(cfg, obstacles, controller, np.random.default_rng(seed))
    result = sim.run()
    last_metrics = result["frames"][-1]["metrics"] if result["frames"] else {}
    g_max_deg = math.degrees(float(last_metrics.get("G_max", 0.0)))
    return {
        "method": method,
        "captured": bool(result["captured"]),
        "failed": bool(result.get("failed", False)),
        "failure_reason": result.get("failure_reason"),
        "num_steps": result["num_steps"],
        "final_G_max_deg": g_max_deg,
        "frames": result["frames"],
    }


def plot_comparison(frames_fr: list, frames_fcem: list, cfg: dict, out_path: Path) -> None:
    w = cfg["world"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    titles = ("fixed_ring (fails)", "fcem (succeeds)")
    all_frames = (frames_fr, frames_fcem)

    for ax, title, frames in zip(axes, titles, all_frames):
        idx = len(frames) - 1
        frame = frames[idx]
        ax.set_xlim(w["xmin"], w["xmax"])
        ax.set_ylim(w["ymin"], w["ymax"])
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, linewidth=0.4, alpha=0.6)
        ax.set_title(title)

        target = frame["evader"]
        pursuers = frame["pursuers"]
        slots = frame.get("slots")
        capture_r = cfg["fcem"]["capture_radius"]
        ax.add_patch(
            plt.Circle(target, capture_r, fill=False, linestyle="-.", linewidth=1.2, color="gray")
        )
        ax.scatter([target[0]], [target[1]], marker="*", s=140, color="C3", label="evader")
        ax.scatter(pursuers[:, 0], pursuers[:, 1], marker="^", s=80, color="C0", label="pursuers")
        if slots is not None:
            ax.scatter(slots[:, 0], slots[:, 1], marker="o", s=55, color="C2", alpha=0.8, label="slots")

        trail = np.array([f["evader"] for f in frames[::8]])
        ax.plot(trail[:, 0], trail[:, 1], linestyle="--", linewidth=1.0, alpha=0.5, color="C3")

        m = frame["metrics"]
        g_deg = math.degrees(m["G_max"])
        status = "CAPTURED" if frame.get("captured") else "NOT CAPTURED"
        ax.text(
            0.02,
            0.98,
            f"step={frame['step']}\nG_max={g_deg:.0f}°\n{status}",
            transform=ax.transAxes,
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round", alpha=0.85),
        )
        ax.legend(loc="upper right", fontsize=8)

    fig.suptitle("fixed_ring_failure: asymmetric fast breakaway (40×40 m)", fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--plot", type=Path, default=Path("results/fixed_ring_failure_comparison.png"))
    parser.add_argument("--json", type=Path, default=Path("results/fixed_ring_failure_summary.json"))
    args = parser.parse_args()

    cfg = load_config("fixed_ring_failure")
    fr = run_method("fixed_ring", args.seed)
    fcem = run_method("fcem", args.seed)

    summary = {
        "scenario": "fixed_ring_failure",
        "seed": args.seed,
        "fixed_ring": {k: v for k, v in fr.items() if k != "frames"},
        "fcem": {k: v for k, v in fcem.items() if k != "frames"},
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.plot.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    plot_comparison(fr["frames"], fcem["frames"], cfg, args.plot)

    print("=== fixed_ring_failure demo ===")
    print(json.dumps(summary, indent=2))
    print(f"plot -> {args.plot}")


if __name__ == "__main__":
    main()
