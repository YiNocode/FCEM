"""Pure pursuit baseline with MIGHTY-style Hermite local planner."""

from __future__ import annotations

from typing import Any

import numpy as np

from common.dynamics import clip_norm, unit
from common.obstacles import Obstacle
from fcem.low_level.hermite_planner import (
    enforce_position_clearance,
    plan_velocity_command,
    planner_kwargs_from_config,
)
from metrics.experiment_logger import TimingBlock


def make_pure_pursuit_controller() -> Any:
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
        timing: dict[str, float] = {}
        pkw = planner_kwargs_from_config(config)

        with TimingBlock() as tb:
            pursuers = pursuers.copy()
            pursuer_v = pursuer_v.copy()
            for i in range(len(pursuers)):
                goal_vel = config["pursuer_vmax"] * unit(evader - pursuers[i])
                cmd = plan_velocity_command(
                    pursuers[i],
                    pursuer_v[i],
                    evader,
                    goal_vel,
                    obstacles,
                    bounds,
                    config["dt"],
                    config["pursuer_vmax"],
                    config["pursuer_amax"],
                    **pkw,
                )
                pursuer_v[i] = clip_norm(cmd, config["pursuer_vmax"])
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
        timing["low_level_ms"] = tb.ms
        timing["prediction_ms"] = 0.0
        timing["manifold_gen_ms"] = 0.0
        timing["assignment_ms"] = 0.0
        timing["total_ms"] = tb.ms
        return {
            "pursuers": pursuers,
            "pursuer_v": pursuer_v,
            "R": R,
            "timing_ms": timing,
        }

    return controller
