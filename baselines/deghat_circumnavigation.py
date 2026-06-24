"""Deghat et al. bearing-only circumnavigation (point-mass adaptation).

Reference: Deghat et al., IROS 2012 — localization and circumnavigation
using bearing-only measurements. Adapted to 2D point-mass with PD slot tracking.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from baselines.common.assignment import assign_slots
from baselines.common.geometry import bearing_circumnavigation_slot, shrink_radius
from baselines.common.low_level import baseline_cfg, empty_timing, track_slots
from common.obstacles import Obstacle
from metrics.experiment_logger import TimingBlock


def make_deghat_circumnavigation_controller() -> Any:
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
        bcfg = baseline_cfg(config, "deghat_circumnavigation")
        omega_des = float(bcfg.get("omega_des", 0.35))
        timing = empty_timing()
        n = len(pursuers)

        with TimingBlock() as tb:
            escape_angle = math.atan2(evader_v[1], evader_v[0]) if np.linalg.norm(evader_v) > 0.1 else 0.0
            desired_bearings = escape_angle + np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
            slots = np.zeros((n, 2))
            slot_vel = np.zeros((n, 2))
            for i in range(n):
                slot, sv = bearing_circumnavigation_slot(
                    pursuers[i],
                    evader,
                    R,
                    desired_bearings[i],
                    omega_des,
                    config["dt"],
                )
                slot += evader_v * config["dt"] * 0.5
                slots[i] = slot
                slot_vel[i] = sv + evader_v
            assignment = assign_slots(
                pursuers, slots, method="nearest", prev_assignment=prev_assignment
            )
        timing["manifold_gen_ms"] = tb.ms
        timing["assignment_ms"] = tb.ms

        pursuers, pursuer_v, ll_ms = track_slots(
            pursuers, pursuer_v, slots, slot_vel, assignment, obstacles, bounds, config
        )
        timing["low_level_ms"] = ll_ms
        R = shrink_radius(R, fcem_cfg)
        timing["total_ms"] = sum(timing.values())

        return {
            "pursuers": pursuers,
            "pursuer_v": pursuer_v,
            "R": R,
            "slots": slots,
            "assignment": assignment,
            "timing_ms": timing,
        }

    return controller
