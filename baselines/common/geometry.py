"""Geometric helpers for literature baselines (point-mass adaptations)."""

from __future__ import annotations

import math

import numpy as np

from common.dynamics import norm, unit, wrap_angle


def ring_slots(
    center: np.ndarray,
    R: float,
    n: int,
    phase: float = 0.0,
) -> np.ndarray:
    """Evenly spaced slots on a circle around center."""
    angles = phase + np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return np.array([center + R * np.array([math.cos(a), math.sin(a)]) for a in angles])


def apollonius_circle(
    pursuer: np.ndarray,
    evader: np.ndarray,
    speed_ratio: float,
) -> tuple[np.ndarray, float]:
    """
    Apollonius circle for pursuer-evader with speed ratio lambda = v_p / v_e.

    Returns (center, radius). When lambda < 1 (slower pursuer), radius may be
    negative meaning no finite capture region — caller should treat as infeasible.
    """
    lam = max(speed_ratio, 1e-6)
    pe = pursuer - evader
    d = norm(pe)
    if d < 1e-9:
        return evader.copy(), 0.0
    if lam >= 1.0 - 1e-9:
        # Pursuer not slower: entire plane is effectively reachable.
        return evader.copy(), max(d * 2.0, 1.0)

    # Standard formula: center on line PE, radius = lam * d / (1 - lam^2)
    center = evader + pe / (1.0 - lam * lam)
    radius = lam * d / (1.0 - lam * lam)
    return center, radius


def point_in_apollonius_region(
    point: np.ndarray,
    pursuer: np.ndarray,
    evader: np.ndarray,
    speed_ratio: float,
) -> bool:
    """True if pursuer can reach point before evader (equal-speed straight line)."""
    lam = max(speed_ratio, 1e-6)
    if lam >= 1.0 - 1e-9:
        return True
    dp = norm(point - pursuer)
    de = norm(point - evader)
    return dp <= lam * de + 1e-6


def voronoi_nearest_point(
    pursuer: np.ndarray,
    evader: np.ndarray,
    all_pursuers: np.ndarray,
) -> np.ndarray:
    """
    Nearest point on the pursuer's Voronoi cell boundary toward evader.
    Simplified 2D bisector construction among pursuers + evader as generator set.
    """
    n = len(all_pursuers)
    if n <= 1:
        return evader.copy()

    best = evader.copy()
    best_dist = norm(evader - pursuer)

    # Sample directions toward evader and bisector midpoints with neighbors.
    for j in range(n):
        if j == _pursuer_index(pursuer, all_pursuers):
            continue
        other = all_pursuers[j]
        mid = 0.5 * (pursuer + other)
        bis_dir = unit(other - pursuer)
        if norm(bis_dir) < 1e-9:
            continue
        normal = np.array([-bis_dir[1], bis_dir[0]])
        # Two candidate points along normal through midpoint.
        for sign in (-1.0, 1.0):
            candidate = mid + sign * normal * 0.5
            if _voronoi_owner(candidate, pursuer, all_pursuers) == _pursuer_index(
                pursuer, all_pursuers
            ):
                d = norm(candidate - pursuer)
                if d < best_dist:
                    best = candidate
                    best_dist = d

    # Fallback: step from pursuer toward evader to bisector with nearest neighbor.
    to_e = evader - pursuer
    if norm(to_e) > 1e-6:
        candidate = pursuer + 0.5 * to_e
        if _voronoi_owner(candidate, pursuer, all_pursuers) == _pursuer_index(
            pursuer, all_pursuers
        ):
            d = norm(candidate - pursuer)
            if d < best_dist:
                best = candidate
    return best


def _pursuer_index(pursuer: np.ndarray, all_pursuers: np.ndarray) -> int:
    dists = np.linalg.norm(all_pursuers - pursuer[None, :], axis=1)
    return int(np.argmin(dists))


def _voronoi_owner(point: np.ndarray, ref: np.ndarray, all_pursuers: np.ndarray) -> int:
    dists = np.linalg.norm(all_pursuers - point[None, :], axis=1)
    return int(np.argmin(dists))


def bearing_circumnavigation_slot(
    pursuer: np.ndarray,
    evader: np.ndarray,
    R: float,
    desired_bearing: float,
    omega_des: float,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Deghat-style bearing circumnavigation slot (point-mass adaptation).

    Advances desired bearing by omega_des and places slot on circle at radius R.
    Returns (slot_position, slot_velocity).
    """
    rel = pursuer - evader
    current_bearing = math.atan2(rel[1], rel[0])
    bearing_err = wrap_angle(desired_bearing - current_bearing)
    omega = omega_des + 0.8 * bearing_err
    next_bearing = desired_bearing + omega * dt

    slot = evader + R * np.array([math.cos(next_bearing), math.sin(next_bearing)])
    evader_vel_est = np.zeros(2)  # filled by caller if available
    slot_vel = R * omega * np.array([-math.sin(next_bearing), math.cos(next_bearing)])
    _ = evader_vel_est
    return slot, slot_vel


def predict_evader(
    pos: np.ndarray,
    vel: np.ndarray,
    horizon: float,
    dt: float,
) -> np.ndarray:
    """Constant-velocity evader prediction."""
    steps = max(1, int(round(horizon / max(dt, 1e-6))))
    return pos + vel * (steps * dt)


def fence_polygon_slots(
    evader: np.ndarray,
    R: float,
    n: int,
    phase: float = 0.0,
) -> np.ndarray:
    """Convex fence vertices (regular n-gon) around evader."""
    return ring_slots(evader, R, n, phase)


def shrink_radius(R: float, fcem_cfg: dict, rate_scale: float = 0.5) -> float:
    """Linear contraction matching fixed_ring baseline."""
    return max(fcem_cfg["R_terminal"], R - fcem_cfg["contraction_rate"] * rate_scale)


def time_to_intercept(pursuer: np.ndarray, target: np.ndarray, vmax: float) -> float:
    """Straight-line intercept time at max speed."""
    return norm(target - pursuer) / max(vmax, 1e-6)
