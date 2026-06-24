"""Kou & Xiang target fencing control (point-mass adaptation).

Reference: Kou & Xiang, Acta Automatica Sinica 2022 — multi-robot target fencing
via output feedback linearization. Simplified as fence polygon slots + Hungarian
assignment + PD/Hermite tracking.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from baselines.common.assignment import assign_slots
from baselines.common.geometry import fence_polygon_slots, shrink_radius
from baselines.common.low_level import baseline_cfg, empty_timing, track_slots
from common.dynamics import unit
from common.obstacles import Obstacle
from metrics.experiment_logger import TimingBlock


def make_kou_xiang_fencing_controller() -> Any:
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
        bcfg = baseline_cfg(config, "kou_xiang_fencing")
        fence_gain = float(bcfg.get("fence_gain", 2.0))
        tangential_gain = float(bcfg.get("tangential_gain", 0.6))
        timing = empty_timing()
        n = len(pursuers)

        with TimingBlock() as tb:
            phase = math.atan2(evader_v[1], evader_v[0]) if np.linalg.norm(evader_v) > 0.1 else 0.0
            base_slots = fence_polygon_slots(evader, R, n, phase)
            slots = base_slots.copy()
            slot_vel = np.zeros((n, 2))
            dt = config["dt"]
            for i in range(n):
                rel = pursuers[i] - evader
                r_hat = unit(rel) if np.linalg.norm(rel) > 1e-6 else unit(base_slots[i] - evader)
                t_hat = np.array([-r_hat[1], r_hat[0]])
                radial_err = R - float(np.linalg.norm(rel))
                fence_acc = fence_gain * radial_err * r_hat + tangential_gain * t_hat
                slots[i] = base_slots[i] + fence_acc * dt * dt * 0.5
                slot_vel[i] = (slots[i] - base_slots[i]) / dt + evader_v
            assignment = assign_slots(pursuers, slots, method="hungarian")
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
