"""Fixed-ring APF baseline: pursuers track static evenly spaced ring slots."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from common.obstacles import Obstacle
from fcem.low_level.pd_tracker import pd_planner_kwargs_from_config, pd_track_step
from metrics.experiment_logger import TimingBlock


def _fixed_ring_slots(evader: np.ndarray, R: float, n: int) -> np.ndarray:
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return np.array([evader + R * np.array([math.cos(a), math.sin(a)]) for a in angles])


def make_fixed_ring_controller() -> Any:
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
        fcem_cfg = config["fcem"]
        timing: dict[str, float] = {}

        with TimingBlock() as tb:
            slots = _fixed_ring_slots(evader, R, len(pursuers))
            assignment = tuple(range(len(pursuers)))
            if prev_slots is None:
                slot_vel = np.zeros_like(slots)
            else:
                slot_vel = (slots - prev_slots) / config["dt"]
        timing["manifold_gen_ms"] = tb.ms
        timing["prediction_ms"] = 0.0
        timing["assignment_ms"] = 0.0

        with TimingBlock() as tb:
            pursuers, pursuer_v = pd_track_step(
                pursuers.copy(),
                pursuer_v.copy(),
                slots,
                slot_vel,
                assignment,
                obstacles,
                bounds,
                config["dt"],
                config["pursuer_kp"],
                config["pursuer_kd"],
                0.0,
                config["pursuer_vmax"],
                config["pursuer_amax"],
                config.get("obstacle_influence", 2.20),
                config.get("pursuer_obs_gain", 1.25),
                config.get("boundary_margin", 1.0),
                config.get("boundary_gain", 1.45),
                planner_kwargs=pd_planner_kwargs_from_config(config),
            )
        timing["low_level_ms"] = tb.ms
        timing["total_ms"] = sum(timing.values())

        # Simple contraction without guarded gate.
        R = max(fcem_cfg["R_terminal"], R - fcem_cfg["contraction_rate"] * 0.5)

        return {
            "pursuers": pursuers,
            "pursuer_v": pursuer_v,
            "R": R,
            "slots": slots,
            "assignment": assignment,
            "timing_ms": timing,
        }

    return controller
