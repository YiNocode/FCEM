"""Candidate manifold generation for FCEM."""

from __future__ import annotations

import math

import numpy as np

from common.dynamics import norm, wrap_angle
from common.obstacles import Obstacle, point_segment_distance


def radius_at_angle(
    center: np.ndarray,
    target: np.ndarray,
    theta: float,
    R: float,
    obstacles: list[Obstacle],
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    enable_obstacle_deform: bool = True,
) -> float:
    r = R
    u = np.array([math.cos(theta), math.sin(theta)])
    nominal = target + R * u

    if enable_obstacle_deform:
        for obs in obstacles:
            rel = obs.center - target
            obs_angle = math.atan2(rel[1], rel[0])
            angle_err = wrap_angle(theta - obs_angle)
            ray_dist = point_segment_distance(obs.center, target, nominal)
            penetration = max(0.0, obs.radius + 0.45 - ray_dist)
            bump = 1.55 * penetration * math.exp(-0.5 * (angle_err / 0.42) ** 2)
            r += bump

    p = center + r * u
    p[0] = np.clip(p[0], xmin + 0.25, xmax - 0.25)
    p[1] = np.clip(p[1], ymin + 0.25, ymax - 0.25)
    return norm(p - center)


def build_manifold(
    center: np.ndarray,
    target: np.ndarray,
    obstacles: list[Obstacle],
    R: float,
    escape_dir: np.ndarray,
    n_slots: int,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    phase_offset: float = 0.0,
    enable_obstacle_deform: bool = True,
    enable_escape_lock: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if enable_escape_lock:
        theta_esc = math.atan2(escape_dir[1], escape_dir[0])
    else:
        theta_esc = 0.0
    theta_esc += phase_offset

    slot_angles = theta_esc + np.linspace(0.0, 2.0 * np.pi, n_slots, endpoint=False)
    dense_angles = theta_esc + np.linspace(0.0, 2.0 * np.pi, 181, endpoint=False)

    dense_points = []
    for theta in dense_angles:
        r = radius_at_angle(
            center, target, theta, R, obstacles,
            xmin, xmax, ymin, ymax, enable_obstacle_deform,
        )
        dense_points.append(center + r * np.array([math.cos(theta), math.sin(theta)]))

    slots = []
    for theta in slot_angles:
        r = radius_at_angle(
            center, target, theta, R, obstacles,
            xmin, xmax, ymin, ymax, enable_obstacle_deform,
        )
        slots.append(center + r * np.array([math.cos(theta), math.sin(theta)]))

    return np.array(slot_angles), np.array(slots), np.array(dense_points)


def generate_candidate_manifolds(
    center: np.ndarray,
    target: np.ndarray,
    obstacles: list[Obstacle],
    R: float,
    escape_dir: np.ndarray,
    n_slots: int,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    n_candidates: int = 3,
    enable_obstacle_deform: bool = True,
    enable_escape_lock: bool = True,
) -> list[tuple[float, np.ndarray, np.ndarray, np.ndarray]]:
    """Return list of (phase_offset, slot_angles, slots, curve)."""
    if not enable_escape_lock:
        offsets = [0.0]
    else:
        offsets = [0.0]
        if n_candidates >= 2:
            offsets.append(math.pi / n_slots)
        if n_candidates >= 3:
            offsets.append(-math.pi / n_slots)

    candidates = []
    for off in offsets[:n_candidates]:
        angles, slots, curve = build_manifold(
            center, target, obstacles, R, escape_dir, n_slots,
            xmin, xmax, ymin, ymax, off,
            enable_obstacle_deform, enable_escape_lock,
        )
        candidates.append((off, angles, slots, curve))
    return candidates
