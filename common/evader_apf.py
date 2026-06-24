"""Evader (target) APF policy with robust boundary handling."""

from __future__ import annotations

from typing import Any

import numpy as np

from common.dynamics import clip_norm, norm, unit, wall_clearances
from common.obstacles import Obstacle, obstacle_repulsion


def pursuer_escape_direction(
    evader: np.ndarray,
    pursuers: np.ndarray | None,
    pursuer_centroid: np.ndarray,
    centroid_gain: float = 1.0,
    nearest_gain: float = 1.0,
    nearest_range: float = 14.0,
) -> np.ndarray:
    """Blend repulsion from pursuer centroid and nearest pursuer."""
    away_centroid = unit(evader - pursuer_centroid)
    if pursuers is None or len(pursuers) == 0:
        return away_centroid

    diffs = pursuers - evader[None, :]
    dists = np.linalg.norm(diffs, axis=1)
    nearest_idx = int(np.argmin(dists))
    d_near = float(dists[nearest_idx])
    away_nearest = unit(evader - pursuers[nearest_idx])

    w_cent = centroid_gain
    w_near = nearest_gain * max(0.0, 1.0 - d_near / max(nearest_range, 1e-6))
    combo = w_cent * away_centroid + w_near * away_nearest
    if norm(combo) < 1e-9:
        return away_centroid
    return unit(combo)


def continuous_boundary_barrier(
    pos: np.ndarray,
    bounds: tuple[float, float, float, float],
    barrier_range: float,
    gain: float,
) -> np.ndarray:
    """Smooth inward barrier force; active within ``barrier_range`` of each wall."""
    if barrier_range <= 1e-9:
        return np.zeros(2)

    walls = wall_clearances(pos, bounds)
    acc = np.zeros(2)

    def _push(inward_axis: int, clearance: float, sign: float) -> None:
        if clearance >= barrier_range:
            return
        t = 1.0 - clearance / barrier_range
        acc[inward_axis] += sign * gain * t * t

    _push(0, walls["left"], +1.0)
    _push(0, walls["right"], -1.0)
    _push(1, walls["bottom"], +1.0)
    _push(1, walls["top"], -1.0)
    return acc


def effective_boundary_vmax(
    vmax: float,
    wall_min: float,
    margin: float,
    vmin_frac: float = 0.35,
) -> float:
    """Reduce speed cap only when very close to walls."""
    if wall_min >= margin:
        return vmax
    t = max(0.0, wall_min / max(margin, 1e-6))
    return vmax * (vmin_frac + (1.0 - vmin_frac) * t)


def project_velocity_inward(
    vel: np.ndarray,
    pos: np.ndarray,
    bounds: tuple[float, float, float, float],
    margin: float,
) -> np.ndarray:
    """Remove velocity components that point toward nearby walls."""
    if margin <= 1e-9:
        return vel.copy()

    walls = wall_clearances(pos, bounds)
    v = vel.copy()

    def _damp(axis: int, clearance: float, outward_positive: bool) -> None:
        if clearance >= margin:
            return
        blend = clearance / margin
        if outward_positive and v[axis] > 0.0:
            v[axis] *= blend
        elif not outward_positive and v[axis] < 0.0:
            v[axis] *= blend

    _damp(0, walls["left"], outward_positive=False)
    _damp(0, walls["right"], outward_positive=True)
    _damp(1, walls["bottom"], outward_positive=False)
    _damp(1, walls["top"], outward_positive=True)
    return v


def boundary_brake_accel(
    vel: np.ndarray,
    pos: np.ndarray,
    bounds: tuple[float, float, float, float],
    margin: float,
    brake_gain: float,
) -> np.ndarray:
    """Damp only velocity components that point toward nearby walls."""
    if margin <= 1e-9 or brake_gain <= 0.0:
        return np.zeros(2)

    walls = wall_clearances(pos, bounds)
    if walls["min"] >= margin:
        return np.zeros(2)

    acc = np.zeros(2)

    def _brake(axis: int, clearance: float, outward_positive: bool) -> None:
        if clearance >= margin:
            return
        strength = (1.0 - clearance / margin) ** 2
        v_axis = vel[axis]
        if outward_positive and v_axis > 0.0:
            acc[axis] -= brake_gain * strength * v_axis
        elif not outward_positive and v_axis < 0.0:
            acc[axis] -= brake_gain * strength * v_axis

    _brake(0, walls["left"], outward_positive=False)
    _brake(0, walls["right"], outward_positive=True)
    _brake(1, walls["bottom"], outward_positive=False)
    _brake(1, walls["top"], outward_positive=True)
    return acc


def boundary_corner_slide(
    pos: np.ndarray,
    bounds: tuple[float, float, float, float],
    margin: float,
    slide_gain: float,
) -> np.ndarray:
    """Corner-only inward nudge; avoids long-range wall-running escape."""
    walls = wall_clearances(pos, bounds)
    if walls["min"] >= margin or slide_gain <= 0.0:
        return np.zeros(2)

    xmin, xmax, ymin, ymax = bounds
    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)
    near_lr = walls["left"] < margin or walls["right"] < margin
    near_bt = walls["bottom"] < margin or walls["top"] < margin
    if not (near_lr and near_bt):
        return np.zeros(2)

    strength = (1.0 - walls["min"] / margin) ** 2
    inward = unit(np.array([cx - pos[0], cy - pos[1]]))
    return slide_gain * strength * inward


def evader_apf_step(
    evader: np.ndarray,
    evader_v: np.ndarray,
    pursuer_centroid: np.ndarray,
    obstacles: list[Obstacle],
    bounds: tuple[float, float, float, float],
    dt: float,
    vmax: float,
    amax: float,
    pursuers: np.ndarray | None = None,
    centroid_gain: float = 1.0,
    nearest_gain: float = 1.0,
    nearest_range: float = 14.0,
    obs_gain: float = 0.60,
    obstacle_influence: float = 2.20,
    boundary_margin: float = 5.0,
    boundary_gain: float = 2.20,
    boundary_barrier_range: float | None = None,
    boundary_slide_gain: float = 1.35,
    boundary_brake_gain: float = 2.40,
    boundary_vmin_frac: float = 0.30,
    boundary_tight_margin: float = 2.50,
    inner_margin: float = 0.30,
) -> tuple[np.ndarray, np.ndarray]:
    xmin, xmax, ymin, ymax = bounds
    barrier_range = boundary_barrier_range if boundary_barrier_range is not None else boundary_margin
    tight_margin = min(boundary_tight_margin, boundary_margin)

    esc_dir = pursuer_escape_direction(
        evader,
        pursuers,
        pursuer_centroid,
        centroid_gain=centroid_gain,
        nearest_gain=nearest_gain,
        nearest_range=nearest_range,
    )

    walls = wall_clearances(evader, bounds)
    vmax_eff = effective_boundary_vmax(
        vmax, walls["min"], tight_margin, boundary_vmin_frac
    )

    desired_v = vmax_eff * esc_dir
    desired_v = project_velocity_inward(desired_v, evader, bounds, boundary_margin)

    acc = centroid_gain * (desired_v - evader_v)
    acc += obstacle_repulsion(evader, obstacles, obstacle_influence, obs_gain)
    acc += continuous_boundary_barrier(evader, bounds, barrier_range, boundary_gain)
    acc += boundary_corner_slide(evader, bounds, tight_margin, boundary_slide_gain)
    acc += boundary_brake_accel(
        evader_v, evader, bounds, tight_margin, boundary_brake_gain
    )

    acc = clip_norm(acc, amax)
    evader_v = clip_norm(evader_v + acc * dt, vmax_eff)
    evader = evader + evader_v * dt
    evader[0] = np.clip(evader[0], xmin + inner_margin, xmax - inner_margin)
    evader[1] = np.clip(evader[1], ymin + inner_margin, ymax - inner_margin)
    return evader, evader_v


def evader_apf_kwargs_from_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Shared evader policy parameters from experiment config."""
    return {
        "centroid_gain": cfg.get("evader_centroid_gain", 1.0),
        "nearest_gain": cfg.get("evader_nearest_gain", 1.0),
        "nearest_range": cfg.get("evader_nearest_range", 14.0),
        "obs_gain": cfg.get("evader_obs_gain", 0.60),
        "obstacle_influence": cfg.get("obstacle_influence", 2.20),
        "boundary_margin": cfg.get("boundary_margin", 5.0),
        "boundary_gain": cfg.get("boundary_gain", 2.20),
        "boundary_barrier_range": cfg.get("evader_boundary_barrier_range"),
        "boundary_slide_gain": cfg.get("evader_boundary_slide_gain", 1.35),
        "boundary_brake_gain": cfg.get("evader_boundary_brake_gain", 2.40),
        "boundary_vmin_frac": cfg.get("evader_boundary_vmin_frac", 0.30),
        "boundary_tight_margin": cfg.get("evader_boundary_tight_margin", 2.50),
    }


def estimate_escape_direction(
    target: np.ndarray,
    target_v: np.ndarray,
    pursuer_centroid: np.ndarray,
    pursuers: np.ndarray | None = None,
) -> np.ndarray:
    esc = pursuer_escape_direction(
        target,
        pursuers,
        pursuer_centroid,
        centroid_gain=0.55,
        nearest_gain=0.75,
        nearest_range=14.0,
    )
    esc = unit(0.75 * target_v + 0.55 * esc)
    if norm(esc) < 1e-9:
        esc = np.array([1.0, 0.0])
    return esc
