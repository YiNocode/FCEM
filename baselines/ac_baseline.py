"""Apollonius-circle baseline: shrink evader safe region S_E via directional search.

Each pursuer moves along the direction that most reduces the area of
S_E = intersection of evader advantage disks D_i.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from baselines.common.geometry import best_area_shrink_direction
from baselines.common.low_level import baseline_cfg, empty_timing
from common.dynamics import clip_norm, norm, unit
from common.obstacles import Obstacle
from fcem.low_level.hermite_planner import (
    enforce_position_clearance,
    plan_velocity_command,
    planner_kwargs_from_config,
)
from metrics.experiment_logger import TimingBlock


def make_ac_baseline_controller() -> Any:
    def controller(
        step: int,
        evader: np.ndarray,
        evader_v: np.ndarray,
        pursuers: np.ndarray,
        pursuer_v: np.ndarray,
        obstacles: list[Obstacle],
        bounds: tuple[float, float, float, float],
        R: float,
        prev_slots: np.ndarray | None,
        prev_assignment: tuple[int, ...] | None,
        config: dict[str, Any],
    ) -> dict[str, Any]:
        bcfg = baseline_cfg(config, "ac_baseline")
        grid_n = int(bcfg.get("grid_n", 48))
        direction_samples = int(bcfg.get("direction_samples", 16))
        preview_delta = float(bcfg.get("preview_delta", 0.35))
        timing = empty_timing()
        pkw = planner_kwargs_from_config(config)
        vmax = config["pursuer_vmax"]
        evader_vmax = config.get("evader_vmax", vmax)
        mu = evader_vmax / max(vmax, 1e-6)

        with TimingBlock() as tb:
            pursuers = pursuers.copy()
            pursuer_v = pursuer_v.copy()
            for i in range(len(pursuers)):
                shrink_dir = best_area_shrink_direction(
                    i,
                    pursuers,
                    evader,
                    mu,
                    bounds,
                    preview_delta=preview_delta,
                    direction_samples=direction_samples,
                    grid_n=grid_n,
                )
                if norm(shrink_dir) < 1e-9:
                    shrink_dir = unit(evader - pursuers[i])
                goal_vel = vmax * shrink_dir
                cmd = plan_velocity_command(
                    pursuers[i],
                    pursuer_v[i],
                    evader,
                    goal_vel,
                    obstacles,
                    bounds,
                    config["dt"],
                    vmax,
                    config["pursuer_amax"],
                    **pkw,
                )
                pursuer_v[i] = clip_norm(cmd, vmax)
                pursuers[i] = pursuers[i] + pursuer_v[i] * config["dt"]
                pursuers[i] = enforce_position_clearance(
                    pursuers[i],
                    obstacles,
                    float(pkw.get("body_radius", 0.25)),
                    float(pkw.get("clearance", 0.55)),
                )
                xmin, xmax, ymin, ymax = bounds
                pursuers[i, 0] = np.clip(pursuers[i, 0], xmin + 0.15, xmax - 0.15)
                pursuers[i, 1] = np.clip(pursuers[i, 1], ymin + 0.15, ymax - 0.15)
        timing["manifold_gen_ms"] = tb.ms
        timing["low_level_ms"] = tb.ms
        timing["total_ms"] = tb.ms

        return {
            "pursuers": pursuers,
            "pursuer_v": pursuer_v,
            "R": R,
            "timing_ms": timing,
        }

    return controller
