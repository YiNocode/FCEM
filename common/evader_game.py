"""Differential-game evader: maximin clearance + largest-gap breakthrough."""

from __future__ import annotations

from typing import Any

import numpy as np

from common.dynamics import clip_norm, norm, unit
from common.evader_apf import (
    boundary_brake_accel,
    boundary_corner_slide,
    continuous_boundary_barrier,
    effective_boundary_vmax,
    project_velocity_inward,
)
from common.obstacles import Obstacle, obstacle_repulsion


def largest_angular_gap_direction(evader: np.ndarray, pursuers: np.ndarray) -> np.ndarray:
    """Unit direction bisecting the largest empty angular sector around the evader."""
    if pursuers is None or len(pursuers) == 0:
        return np.array([1.0, 0.0])

    rel = pursuers - evader[None, :]
    angles = np.sort(np.arctan2(rel[:, 1], rel[:, 0]))
    extended = np.r_[angles, angles[0] + 2.0 * np.pi]
    gaps = np.diff(extended)
    gap_idx = int(np.argmax(gaps))
    mid = angles[gap_idx] + 0.5 * gaps[gap_idx]
    return np.array([np.cos(mid), np.sin(mid)], dtype=float)


def minimax_evasion_direction(
    evader: np.ndarray,
    pursuers: np.ndarray,
    pursuer_vmax: float,
    horizon: float,
    n_directions: int = 48,
    evader_vmax: float = 1.0,
) -> np.ndarray:
    """
    Discrete Isaacs-style evasion: maximize worst-case clearance after horizon.

    Pursuers are predicted with pure pursuit at ``pursuer_vmax``; the evader
    candidate moves at ``evader_vmax`` along each trial heading.
    """
    if pursuers is None or len(pursuers) == 0:
        return np.array([1.0, 0.0])

    pursuers = np.asarray(pursuers, dtype=float)
    ev_speed = max(float(evader_vmax), 1e-6)
    best_score = -np.inf
    best_dir = np.array([1.0, 0.0])
    thetas = np.linspace(0.0, 2.0 * np.pi, n_directions, endpoint=False)

    for theta in thetas:
        u = np.array([np.cos(theta), np.sin(theta)], dtype=float)
        ev_future = evader + ev_speed * horizon * u
        worst = np.inf
        for p in pursuers:
            chase = unit(evader - p)
            p_future = p + pursuer_vmax * horizon * chase
            worst = min(worst, norm(ev_future - p_future))
        if worst > best_score:
            best_score = worst
            best_dir = u

    return best_dir


def game_escape_direction(
    evader: np.ndarray,
    evader_v: np.ndarray,
    pursuers: np.ndarray | None,
    pursuer_vmax: float,
    horizon: float = 0.85,
    n_directions: int = 48,
    minimax_weight: float = 1.0,
    gap_gain: float = 1.15,
    velocity_inertia: float = 0.35,
    evader_vmax: float = 1.0,
) -> np.ndarray:
    """Blend maximin heading, largest-gap breakout, and velocity persistence."""
    if pursuers is None or len(pursuers) == 0:
        if norm(evader_v) > 1e-9:
            return unit(evader_v)
        return np.array([1.0, 0.0])

    u_minimax = minimax_evasion_direction(
        evader,
        pursuers,
        pursuer_vmax,
        horizon,
        n_directions,
        evader_vmax=evader_vmax,
    )
    u_gap = largest_angular_gap_direction(evader, pursuers)
    combo = minimax_weight * u_minimax + gap_gain * u_gap + velocity_inertia * evader_v
    if norm(combo) < 1e-9:
        return u_minimax
    return unit(combo)


def evader_game_step(
    evader: np.ndarray,
    evader_v: np.ndarray,
    pursuer_centroid: np.ndarray,
    obstacles: list[Obstacle],
    bounds: tuple[float, float, float, float],
    dt: float,
    vmax: float,
    amax: float,
    pursuers: np.ndarray | None = None,
    pursuer_vmax: float = 5.0,
    horizon: float = 0.85,
    n_directions: int = 48,
    minimax_weight: float = 1.0,
    gap_gain: float = 1.15,
    velocity_inertia: float = 0.35,
    obs_gain: float = 0.60,
    obstacle_influence: float = 2.20,
    boundary_margin: float = 5.0,
    boundary_gain: float = 2.20,
    boundary_barrier_range: float | None = None,
    boundary_slide_gain: float = 0.85,
    boundary_brake_gain: float = 2.40,
    boundary_vmin_frac: float = 0.30,
    boundary_tight_margin: float = 2.50,
    inner_margin: float = 0.30,
    **_: Any,
) -> tuple[np.ndarray, np.ndarray]:
    xmin, xmax, ymin, ymax = bounds
    barrier_range = boundary_barrier_range if boundary_barrier_range is not None else boundary_margin
    tight_margin = min(boundary_tight_margin, boundary_margin)

    esc_dir = game_escape_direction(
        evader,
        evader_v,
        pursuers,
        pursuer_vmax,
        horizon=horizon,
        n_directions=n_directions,
        minimax_weight=minimax_weight,
        gap_gain=gap_gain,
        velocity_inertia=velocity_inertia,
        evader_vmax=vmax,
    )

    from common.dynamics import wall_clearances

    walls = wall_clearances(evader, bounds)
    vmax_eff = effective_boundary_vmax(
        vmax, walls["min"], tight_margin, boundary_vmin_frac
    )

    desired_v = vmax_eff * esc_dir
    desired_v = project_velocity_inward(desired_v, evader, bounds, boundary_margin)

    acc = 2.2 * (desired_v - evader_v)
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


def evader_game_kwargs_from_config(cfg: dict[str, Any]) -> dict[str, Any]:
    game = cfg.get("evader_game", {})
    from common.evader_apf import evader_apf_kwargs_from_config

    shared = evader_apf_kwargs_from_config(cfg)
    return {
        **shared,
        "pursuer_vmax": cfg.get("pursuer_vmax", 5.0),
        "horizon": game.get("horizon", 0.85),
        "n_directions": game.get("n_directions", 48),
        "minimax_weight": game.get("minimax_weight", 1.0),
        "gap_gain": game.get("gap_gain", 1.15),
        "velocity_inertia": game.get("velocity_inertia", 0.35),
    }
