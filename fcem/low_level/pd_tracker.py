"""PD slot tracker for 2D pursuers with Hermite local planner."""

from __future__ import annotations

from typing import Any

import numpy as np

from common.dynamics import clip_norm
from common.obstacles import Obstacle
from fcem.low_level.hermite_planner import enforce_position_clearance, plan_velocity_command, planner_kwargs_from_config


def _planner_params(
    obstacle_influence: float,
    pursuer_obs_gain: float,
    boundary_margin: float,
    boundary_gain: float,
    planner_kwargs: dict[str, Any] | None,
) -> dict[str, Any]:
    """Legacy APF args are ignored; kept for call-site compatibility."""
    _ = (obstacle_influence, pursuer_obs_gain)
    out = dict(planner_kwargs or {})
    out.setdefault("boundary_margin", boundary_margin)
    out.setdefault("boundary_gain", boundary_gain)
    return out


def pd_track_step(
    pursuers: np.ndarray,
    pursuer_vel: np.ndarray,
    slots: np.ndarray,
    slot_vel: np.ndarray,
    assignment: tuple[int, ...],
    obstacles: list[Obstacle],
    bounds: tuple[float, float, float, float],
    dt: float,
    kp: float,
    kd: float,
    slot_v_ff_gain: float,
    vmax: float,
    amax: float,
    obstacle_influence: float,
    pursuer_obs_gain: float,
    boundary_margin: float,
    boundary_gain: float,
    ablate_no_slot_vel_ff: bool = False,
    planner_kwargs: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    ff_gain = 0.0 if ablate_no_slot_vel_ff else slot_v_ff_gain
    pkw = _planner_params(
        obstacle_influence, pursuer_obs_gain, boundary_margin, boundary_gain, planner_kwargs
    )
    pkw.setdefault("kp_fallback", kp)
    pkw.setdefault("kd_fallback", kd)

    n = len(pursuers)
    for i in range(n):
        j = assignment[i]
        cmd = plan_velocity_command(
            pursuers[i],
            pursuer_vel[i],
            slots[j],
            ff_gain * slot_vel[j],
            obstacles,
            bounds,
            dt,
            vmax,
            amax,
            **pkw,
        )
        pursuer_vel[i] = cmd
        pursuers[i] = pursuers[i] + pursuer_vel[i] * dt
        pursuers[i] = enforce_position_clearance(
            pursuers[i],
            obstacles,
            float(pkw.get("body_radius", 0.25)),
            float(pkw.get("clearance", 0.55)),
        )
        xmin, xmax, ymin, ymax = bounds
        pursuers[i, 0] = np.clip(pursuers[i, 0], xmin + 0.15, xmax - 0.15)
        pursuers[i, 1] = np.clip(pursuers[i, 1], ymin + 0.15, ymax - 0.15)
    return pursuers, pursuer_vel


def pd_velocity_command(
    pursuers: np.ndarray,
    pursuer_vel: np.ndarray,
    slots: np.ndarray,
    slot_vel: np.ndarray,
    assignment: tuple[int, ...],
    obstacles: list[Obstacle],
    bounds: tuple[float, float, float, float],
    dt: float,
    kp: float,
    kd: float,
    slot_v_ff_gain: float,
    vmax: float,
    amax: float,
    obstacle_influence: float,
    pursuer_obs_gain: float,
    boundary_margin: float,
    boundary_gain: float,
    ablate_no_slot_vel_ff: bool = False,
    planner_kwargs: dict[str, Any] | None = None,
) -> np.ndarray:
    """World-frame XY velocity command (onboard-style), without integrating position."""
    ff_gain = 0.0 if ablate_no_slot_vel_ff else slot_v_ff_gain
    pkw = _planner_params(
        obstacle_influence, pursuer_obs_gain, boundary_margin, boundary_gain, planner_kwargs
    )
    pkw.setdefault("kp_fallback", kp)
    pkw.setdefault("kd_fallback", kd)

    n = len(pursuers)
    cmd = np.zeros((n, 2), dtype=float)
    for i in range(n):
        j = assignment[i]
        cmd[i] = plan_velocity_command(
            pursuers[i],
            pursuer_vel[i],
            slots[j],
            ff_gain * slot_vel[j],
            obstacles,
            bounds,
            dt,
            vmax,
            amax,
            **pkw,
        )
    return cmd


def pd_planner_kwargs_from_config(config: dict[str, Any]) -> dict[str, Any]:
    return planner_kwargs_from_config(config)
