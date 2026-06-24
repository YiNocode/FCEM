"""Unified low-level tracking for literature baselines."""

from __future__ import annotations

from typing import Any

import numpy as np

from common.obstacles import Obstacle
from fcem.low_level.pd_tracker import pd_planner_kwargs_from_config, pd_track_step
from metrics.experiment_logger import TimingBlock


def baseline_cfg(config: dict[str, Any], name: str) -> dict[str, Any]:
    return dict(config.get("baselines", {}).get(name, {}))


def track_slots(
    pursuers: np.ndarray,
    pursuer_v: np.ndarray,
    slots: np.ndarray,
    slot_vel: np.ndarray,
    assignment: tuple[int, ...],
    obstacles: list[Obstacle],
    bounds: tuple[float, float, float, float],
    config: dict[str, Any],
    slot_v_ff_gain: float | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Run PD + Hermite tracking; returns updated state and low_level_ms."""
    ff = slot_v_ff_gain
    if ff is None:
        ff = config.get("fcem", {}).get("slot_v_ff_gain", 0.85)

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
            ff,
            config["pursuer_vmax"],
            config["pursuer_amax"],
            config.get("obstacle_influence", 2.20),
            config.get("pursuer_obs_gain", 1.25),
            config.get("boundary_margin", 5.0),
            config.get("boundary_gain", 2.20),
            planner_kwargs=pd_planner_kwargs_from_config(config),
        )
    return pursuers, pursuer_v, tb.ms


def slot_velocities(
    slots: np.ndarray,
    prev_slots: np.ndarray | None,
    dt: float,
) -> np.ndarray:
    if prev_slots is None:
        return np.zeros_like(slots)
    return (slots - prev_slots) / dt


def empty_timing() -> dict[str, float]:
    return {
        "prediction_ms": 0.0,
        "manifold_gen_ms": 0.0,
        "assignment_ms": 0.0,
        "low_level_ms": 0.0,
        "mpc_ms": 0.0,
        "total_ms": 0.0,
    }
