"""Map FCEM slot targets to PyFlyt onboard velocity setpoints (mode 6: vx, vy, vr, vz)."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from common.dynamics import wrap_angle
from common.obstacles import Obstacle
from fcem.low_level.pd_tracker import pd_planner_kwargs_from_config, pd_velocity_command


def slot_velocities(
    slots: np.ndarray,
    prev_slots: np.ndarray | None,
    dt: float,
) -> np.ndarray:
    if prev_slots is None:
        return np.zeros_like(slots)
    return (slots - prev_slots) / dt


def velocity_setpoints_mode6(
    pursuers_xy: np.ndarray,
    pursuer_v_xy: np.ndarray,
    slots: np.ndarray,
    slot_vel: np.ndarray,
    assignment: tuple[int, ...],
    yaw_rad: np.ndarray,
    altitude: float,
    heights: np.ndarray,
    obstacles: list[Obstacle],
    bounds: tuple[float, float, float, float],
    config: dict[str, Any],
    ablation: dict[str, bool] | None = None,
) -> np.ndarray:
    """Build (n, 4) PyFlyt mode-6 setpoints: ground vx, vy, yaw-rate, vz."""
    ablation = ablation or {}
    pf = config.get("pyflyt", {})
    cmd_xy = pd_velocity_command(
        pursuers_xy,
        pursuer_v_xy,
        slots,
        slot_vel,
        assignment,
        obstacles,
        bounds,
        config["dt"],
        config["pursuer_kp"],
        config["pursuer_kd"],
        config["fcem"].get("slot_v_ff_gain", 0.85),
        config["pursuer_vmax"],
        config["pursuer_amax"],
        config.get("obstacle_influence", 2.20),
        config.get("pursuer_obs_gain", 1.25),
        config.get("boundary_margin", 1.0),
        config.get("boundary_gain", 1.45),
        ablate_no_slot_vel_ff=ablation.get("ablate_no_slot_vel_ff", False),
        planner_kwargs=pd_planner_kwargs_from_config(config),
    )

    kp_yaw = float(pf.get("yaw_rate_gain", 2.0))
    max_yaw_rate = float(pf.get("max_yaw_rate", 1.2))
    kp_alt = float(pf.get("altitude_rate_gain", 1.5))
    max_vz = float(pf.get("max_vz", 0.8))

    setpoints = np.zeros((len(pursuers_xy), 4), dtype=float)
    for i, slot_idx in enumerate(assignment):
        slot = slots[slot_idx]
        v = cmd_xy[i]
        speed = float(np.linalg.norm(v))
        if speed > 0.15:
            yaw_target = math.atan2(v[1], v[0])
        else:
            delta = slot - pursuers_xy[i]
            yaw_target = math.atan2(delta[1], delta[0]) if float(np.linalg.norm(delta)) > 1e-6 else yaw_rad[i]
        yaw_err = wrap_angle(yaw_target - yaw_rad[i])
        vr = float(np.clip(kp_yaw * yaw_err, -max_yaw_rate, max_yaw_rate))
        vz = float(np.clip(kp_alt * (altitude - heights[i]), -max_vz, max_vz))
        setpoints[i] = [v[0], v[1], vr, vz]
    return setpoints


def direct_velocity_setpoints_mode6(
    pursuer_v_xy: np.ndarray,
    yaw_rad: np.ndarray,
    heights: np.ndarray,
    altitude: float,
    config: dict[str, Any],
    heading_targets: np.ndarray | None = None,
) -> np.ndarray:
    """Map planner XY velocity to mode-6 setpoints (pure pursuit / direct APF)."""
    pf = config.get("pyflyt", {})
    kp_yaw = float(pf.get("yaw_rate_gain", 2.0))
    max_yaw_rate = float(pf.get("max_yaw_rate", 1.2))
    kp_alt = float(pf.get("altitude_rate_gain", 1.5))
    max_vz = float(pf.get("max_vz", 0.8))

    setpoints = np.zeros((len(pursuer_v_xy), 4), dtype=float)
    for i in range(len(pursuer_v_xy)):
        v = pursuer_v_xy[i]
        speed = float(np.linalg.norm(v))
        if speed > 0.15:
            yaw_target = math.atan2(v[1], v[0])
        elif heading_targets is not None:
            delta = heading_targets[i]
            yaw_target = (
                math.atan2(delta[1], delta[0])
                if float(np.linalg.norm(delta)) > 1e-6
                else yaw_rad[i]
            )
        else:
            yaw_target = yaw_rad[i]
        yaw_err = wrap_angle(yaw_target - yaw_rad[i])
        vr = float(np.clip(kp_yaw * yaw_err, -max_yaw_rate, max_yaw_rate))
        vz = float(np.clip(kp_alt * (altitude - heights[i]), -max_vz, max_vz))
        setpoints[i] = [v[0], v[1], vr, vz]
    return setpoints
