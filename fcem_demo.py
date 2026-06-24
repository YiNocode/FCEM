#!/usr/bin/env python3
"""
FCEM Guarded Contraction Animation Demo

Uses unified 2D simulation modules. Run:
    python fcem_demo.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from envs.sim2d import Sim2D, make_fcem_controller
from experiments.config_loader import load_config, obstacles_from_scenario


def draw_frame(ax, sim_frames, obstacles, cfg, idx, show_legend=True):
    frame = sim_frames[idx]
    w = cfg["world"]
    ax.clear()
    ax.set_xlim(w["xmin"], w["xmax"])
    ax.set_ylim(w["ymin"], w["ymax"])
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linewidth=0.4, alpha=0.6)

    for obs in obstacles:
        ax.add_patch(plt.Circle(obs.center, obs.radius, fill=False, linestyle="--", linewidth=1.8))

    target = frame["evader"]
    pursuers = frame["pursuers"]
    center = frame.get("center", target)
    slots = frame.get("slots")
    curve = frame.get("curve")
    assignment = frame.get("assignment", tuple(range(len(pursuers))))
    escape_dir = frame.get("escape_dir", np.array([1.0, 0.0]))
    capture_r = cfg["fcem"]["capture_radius"]

    if curve is not None:
        closed_curve = np.vstack([curve, curve[0]])
        ax.plot(closed_curve[:, 0], closed_curve[:, 1], linewidth=1.8, label="FCEM manifold")

    ax.add_patch(plt.Circle(target, capture_r, fill=False, linestyle="-.", linewidth=1.5, label="capture radius"))
    ax.scatter([target[0]], [target[1]], marker="*", s=170, label="evader")
    ax.scatter([center[0]], [center[1]], marker="+", s=110, label="manifold center")
    ax.arrow(
        target[0], target[1],
        1.25 * escape_dir[0], 1.25 * escape_dir[1],
        head_width=0.22, length_includes_head=True, linewidth=1.3,
    )
    ax.scatter(pursuers[:, 0], pursuers[:, 1], marker="^", s=90, label="pursuers")
    if slots is not None:
        ax.scatter(slots[:, 0], slots[:, 1], marker="o", s=75, label="slots")
        for i, j in enumerate(assignment):
            p, s = pursuers[i], slots[j]
            ax.plot([p[0], s[0]], [p[1], s[1]], linestyle=":", linewidth=1.2)

    m = frame["metrics"]
    trap_mode = frame.get("trap_mode", "open_space")
    g_show = m.get("G_free", m["G_max"]) if trap_mode != "open_space" else m["G_max"]
    g_label = "G_free" if trap_mode != "open_space" else "G_max"
    info = (
        f"step={frame['step']} | mode={trap_mode}\n"
        f"R={frame.get('R', 0):.2f}, q={frame.get('q', 0):.2f}\n"
        f"D={m.get('D_free', m['D_ang']):.2f}, C={m.get('C_free', m['C_cov']):.2f}, "
        f"{g_label}={math.degrees(g_show):.0f}°\n"
        f"recovery={frame.get('recovery_mode', False)}"
    )
    ax.text(0.02, 0.98, info, transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(boxstyle="round", alpha=0.8))
    if frame.get("captured"):
        ax.text(0.50, 0.06, "CAPTURED", transform=ax.transAxes, ha="center", fontsize=12, weight="bold")
    if show_legend:
        ax.legend(loc="upper right", fontsize=8)


def main():
    cfg = load_config("random_obstacles")
    obstacles = obstacles_from_scenario(cfg["scenario"])
    sim = Sim2D(cfg, obstacles, make_fcem_controller(), np.random.default_rng(cfg["seed"]))
    result = sim.run()
    frames = result["frames"]

    summary = {
        "captured": result["captured"],
        "capture_step": result["capture_step"],
        "num_steps": result["num_steps"],
    }
    Path("fcem_guarded_contraction_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    idxs = [0, len(frames) // 3, 2 * len(frames) // 3, len(frames) - 1]
    for ax, idx in zip(axes.ravel(), idxs):
        draw_frame(ax, frames, obstacles, cfg, idx, show_legend=False)
    plt.tight_layout()
    plt.savefig("fcem_guarded_contraction_keyframe.png", dpi=220)
    plt.close()

    fig, ax = plt.subplots(figsize=(7.4, 7.4))
    indices = list(range(0, len(frames), 4))
    if indices[-1] != len(frames) - 1:
        indices.append(len(frames) - 1)

    def update(i):
        draw_frame(ax, frames, obstacles, cfg, indices[i], show_legend=(i == 0))
        return []

    ani = FuncAnimation(fig, update, frames=len(indices), interval=100, blit=False)
    ani.save("fcem_guarded_contraction_animation.gif", writer=PillowWriter(fps=10))
    plt.close()

    print("=== FCEM guarded contraction animation demo ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
