"""Synchronization arrival-time metrics and assignment cost components."""

from __future__ import annotations

import numpy as np

from common.dynamics import norm, unit, wrap_angle
from common.obstacles import Obstacle, point_segment_distance


def estimate_arrival_times(
    pursuers: np.ndarray,
    pursuer_v: np.ndarray,
    slots: np.ndarray,
    assignment: tuple[int, ...],
    vmax: float,
    v_min_frac: float = 0.15,
) -> np.ndarray:
    """Speed-projected ETA per pursuer: T_hat_i = ||p_i - s_j|| / max(v_proj, v_min_frac * vmax)."""
    v_floor = v_min_frac * vmax
    T_hats = np.zeros(len(pursuers), dtype=float)
    for i, j in enumerate(assignment):
        delta = slots[j] - pursuers[i]
        dist = norm(delta)
        direction = unit(delta)
        v_proj = max(float(np.dot(pursuer_v[i], direction)), v_floor)
        T_hats[i] = dist / v_proj
    return T_hats


def sync_coverage(T_hats: np.ndarray, tau_T: float) -> float:
    """C_sync = exp(-(max(T_hat) - min(T_hat)) / tau_T)."""
    spread = float(np.max(T_hats) - np.min(T_hats))
    return float(np.exp(-spread / max(tau_T, 1e-9)))


def angular_mismatch_cost(
    pursuer: np.ndarray,
    slot: np.ndarray,
    target: np.ndarray,
) -> float:
    """J_ang: normalized bearing mismatch between pursuer-target and slot-target."""
    vec_p = pursuer - target
    vec_s = slot - target
    if norm(vec_p) < 1e-9 or norm(vec_s) < 1e-9:
        return 0.0
    ang_p = np.arctan2(vec_p[1], vec_p[0])
    ang_s = np.arctan2(vec_s[1], vec_s[0])
    diff = abs(wrap_angle(ang_p - ang_s))
    return float(np.clip(diff / np.pi, 0.0, 1.0))


def segment_obstacle_cost(
    pursuer: np.ndarray,
    slot: np.ndarray,
    obstacles: list[Obstacle],
    clearance: float,
) -> float:
    """J_safe: penalty when pursuer-slot segment violates obstacle clearance."""
    if not obstacles:
        return 0.0
    penalty = 0.0
    for obs in obstacles:
        d = point_segment_distance(obs.center, pursuer, slot) - obs.radius
        if d < clearance:
            penalty += (clearance - d) / max(clearance, 1e-9)
    return float(penalty)
