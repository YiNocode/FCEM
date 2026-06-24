"""Liao et al. MPC cooperative hunting (point-mass adaptation).

Reference: Liao et al., ICRA 2021 — cooperative hunting in obstacle-rich
environments. Distributed short-horizon MPC per pursuer with scipy optimizer.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import minimize

from baselines.common.assignment import assign_slots
from baselines.common.geometry import ring_slots, shrink_radius
from baselines.common.low_level import baseline_cfg, empty_timing
from common.dynamics import clip_norm, norm
from common.obstacles import Obstacle
from metrics.experiment_logger import TimingBlock


def _obstacle_penalty(pos: np.ndarray, obstacles: list[Obstacle], clearance: float) -> float:
    penalty = 0.0
    for obs in obstacles:
        d = norm(pos - obs.center) - obs.radius
        if d < clearance:
            penalty += (clearance - d) ** 2
    return penalty


def _mpc_step(
    pos: np.ndarray,
    vel: np.ndarray,
    target_slot: np.ndarray,
    obstacles: list[Obstacle],
    dt: float,
    vmax: float,
    amax: float,
    horizon: int,
    w_pos: float,
    w_u: float,
    w_obs: float,
    warm_u: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_vars = horizon * 2
    x0 = np.zeros(n_vars)
    if warm_u is not None and len(warm_u) == n_vars:
        x0 = warm_u.copy()

    bounds = [(-amax, amax), (-amax, amax)] * horizon

    def objective(u_flat: np.ndarray) -> float:
        p = pos.copy()
        v = vel.copy()
        cost = 0.0
        for k in range(horizon):
            uk = u_flat[2 * k : 2 * k + 2]
            cost += w_u * float(np.dot(uk, uk))
            v = clip_norm(v + uk * dt, vmax)
            p = p + v * dt
            cost += w_pos * norm(p - target_slot) ** 2
            cost += w_obs * _obstacle_penalty(p, obstacles, 0.55)
        return cost

    res = minimize(objective, x0, method="L-BFGS-B", bounds=bounds, options={"maxiter": 25})
    u_opt = res.x.reshape(horizon, 2)
    u0 = u_opt[0]
    v_new = clip_norm(vel + u0 * dt, vmax)
    p_new = pos + v_new * dt
    return p_new, v_new, res.x


def make_liao_mpc_controller() -> Any:
    warm_starts: dict[int, np.ndarray] = {}

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
        bcfg = baseline_cfg(config, "liao_mpc")
        horizon = int(bcfg.get("horizon", 6))
        w_pos = float(bcfg.get("w_pos", 1.0))
        w_u = float(bcfg.get("w_u", 0.05))
        w_obs = float(bcfg.get("w_obs", 8.0))
        timing = empty_timing()
        n = len(pursuers)
        dt = config["dt"]
        vmax = config["pursuer_vmax"]
        amax = config["pursuer_amax"]

        with TimingBlock() as tb:
            slots = ring_slots(evader, R, n)
            if prev_slots is not None:
                slot_vel = (slots - prev_slots) / dt
            else:
                slot_vel = np.zeros_like(slots)
            assignment = assign_slots(pursuers, slots, method="nearest")
        timing["manifold_gen_ms"] = tb.ms
        timing["assignment_ms"] = tb.ms

        pursuers = pursuers.copy()
        pursuer_v = pursuer_v.copy()
        mpc_ms = 0.0
        xmin, xmax, ymin, ymax = bounds
        for i in range(n):
            j = assignment[i]
            target = slots[j] + slot_vel[j] * dt
            with TimingBlock() as tb_mpc:
                warm = warm_starts.get(i)
                p_new, v_new, u_seq = _mpc_step(
                    pursuers[i],
                    pursuer_v[i],
                    target,
                    obstacles,
                    dt,
                    vmax,
                    amax,
                    horizon,
                    w_pos,
                    w_u,
                    w_obs,
                    warm,
                )
                warm_starts[i] = np.roll(u_seq.reshape(-1), -2)
            mpc_ms += tb_mpc.ms
            pursuers[i] = p_new
            pursuer_v[i] = v_new
            pursuers[i, 0] = np.clip(pursuers[i, 0], xmin + 0.15, xmax - 0.15)
            pursuers[i, 1] = np.clip(pursuers[i, 1], ymin + 0.15, ymax - 0.15)

        timing["mpc_ms"] = mpc_ms
        timing["low_level_ms"] = mpc_ms
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
