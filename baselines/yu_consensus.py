"""Yu et al. circular formation consensus (point-mass adaptation).

Reference: Yu et al., Automatica 2018 — circular formation of networked
dynamic unicycles. Adapted to point-mass with angular consensus on a ring.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from baselines.common.assignment import assign_slots
from baselines.common.geometry import shrink_radius
from baselines.common.low_level import baseline_cfg, empty_timing, track_slots
from common.obstacles import Obstacle
from metrics.experiment_logger import TimingBlock


def _adjacency(n: int, mode: str) -> np.ndarray:
    a = np.zeros((n, n))
    if mode == "complete":
        for i in range(n):
            for j in range(n):
                if i != j:
                    a[i, j] = 1.0
        return a
    for i in range(n):
        j = (i + 1) % n
        a[i, j] = 1.0
        a[j, i] = 1.0
    return a


def make_yu_consensus_controller() -> Any:
    theta: np.ndarray | None = None

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
        nonlocal theta
        fcem_cfg = config["fcem"]
        bcfg = baseline_cfg(config, "yu_consensus")
        consensus_gain = float(bcfg.get("consensus_gain", 0.8))
        adjacency_mode = str(bcfg.get("adjacency", "ring"))
        timing = empty_timing()
        n = len(pursuers)
        dt = config["dt"]

        with TimingBlock() as tb:
            if theta is None or len(theta) != n:
                rel0 = pursuers[0] - evader
                phase0 = math.atan2(rel0[1], rel0[0]) if np.linalg.norm(rel0) > 1e-6 else 0.0
                theta = phase0 + np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
            escape_phase = (
                math.atan2(evader_v[1], evader_v[0])
                if np.linalg.norm(evader_v) > 0.1
                else float(theta[0])
            )
            desired = escape_phase + np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
            a_mat = _adjacency(n, adjacency_mode)
            d_theta = np.zeros(n)
            for i in range(n):
                consensus = 0.0
                for j in range(n):
                    if a_mat[i, j] > 0:
                        diff = (theta[j] - theta[i] + np.pi) % (2.0 * np.pi) - np.pi
                        consensus += a_mat[i, j] * diff
                des_err = (desired[i] - theta[i] + np.pi) % (2.0 * np.pi) - np.pi
                d_theta[i] = consensus_gain * consensus + 1.5 * consensus_gain * des_err
            theta = (theta + d_theta * dt) % (2.0 * np.pi)
            slots = np.array(
                [evader + R * np.array([math.cos(t), math.sin(t)]) for t in theta]
            )
            slot_vel = np.zeros((n, 2))
            for i in range(n):
                slot_vel[i] = (
                    R * d_theta[i] * np.array([-math.sin(theta[i]), math.cos(theta[i])])
                    + evader_v
                )
            assignment = assign_slots(pursuers, slots, method="nearest")
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
