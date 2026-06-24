"""Obstacle definitions and scenario builders."""

from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np

from common.dynamics import norm


@dataclass
class Obstacle:
    center: np.ndarray
    radius: float


def point_segment_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom < 1e-12:
        return norm(p - a)
    t = float(np.clip(np.dot(p - a, ab) / denom, 0.0, 1.0))
    q = a + t * ab
    return norm(p - q)


def boundary_repulsion(
    pos: np.ndarray,
    bounds_or_xmin: tuple[float, float, float, float] | float,
    margin_or_xmax: float | tuple[float, float, float, float] | None = None,
    gain_or_ymin: float | None = None,
    ymax: float | None = None,
    margin: float | None = None,
    gain: float | None = None,
) -> np.ndarray:
    if isinstance(bounds_or_xmin, tuple):
        xmin, xmax, ymin, ymax_v = bounds_or_xmin
        assert margin_or_xmax is not None and gain_or_ymin is not None
        m, g = margin_or_xmax, gain_or_ymin
    else:
        xmin = float(bounds_or_xmin)
        xmax = float(margin_or_xmax)  # type: ignore[arg-type]
        ymin = float(gain_or_ymin)  # type: ignore[arg-type]
        ymax_v = float(ymax)  # type: ignore[arg-type]
        m = float(margin)  # type: ignore[arg-type]
        g = float(gain)  # type: ignore[arg-type]

    acc = np.zeros(2)
    if pos[0] - xmin < m:
        acc[0] += g * (m - (pos[0] - xmin)) / m
    if xmax - pos[0] < m:
        acc[0] -= g * (m - (xmax - pos[0])) / m
    if pos[1] - ymin < m:
        acc[1] += g * (m - (pos[1] - ymin)) / m
    if ymax_v - pos[1] < m:
        acc[1] -= g * (m - (ymax_v - pos[1])) / m
    return acc


def obstacle_repulsion(
    pos: np.ndarray,
    obstacles: list[Obstacle],
    influence: float,
    gain: float,
) -> np.ndarray:
    acc = np.zeros(2)
    for obs in obstacles:
        delta = pos - obs.center
        d_center = norm(delta)
        d_surface = d_center - obs.radius
        if d_center < 1e-9:
            continue
        if d_surface < influence:
            strength = gain * (1.0 / max(d_surface, 0.08) - 1.0 / influence)
            acc += max(0.0, strength) * (delta / d_center)
    return acc


def pursuer_obstacle_collision(
    pos: np.ndarray,
    obstacles: list[Obstacle],
    body_radius: float = 0.0,
) -> tuple[bool, int | None]:
    """Return whether a pursuer intersects any cylindrical obstacle."""
    for idx, obs in enumerate(obstacles):
        if norm(pos - obs.center) <= obs.radius + body_radius:
            return True, idx
    return False, None


def any_pursuer_obstacle_collision(
    pursuers: np.ndarray,
    obstacles: list[Obstacle],
    body_radius: float = 0.0,
) -> tuple[bool, list[int]]:
    """Return collision flag and indices of pursuers that hit obstacles."""
    if not obstacles:
        return False, []
    hit: list[int] = []
    for i, pos in enumerate(pursuers):
        collided, _ = pursuer_obstacle_collision(pos, obstacles, body_radius)
        if collided:
            hit.append(i)
    return bool(hit), hit


def scenario_free() -> list[Obstacle]:
    return []


def scenario_random_obstacles(
    seed: int = 0,
    n: int = 8,
    bounds: tuple[float, float, float, float] = (0.0, 40.0, 0.0, 40.0),
) -> list[Obstacle]:
    rng = random.Random(seed)
    xmin, xmax, ymin, ymax = bounds
    margin = 0.20 * min(xmax - xmin, ymax - ymin)
    obstacles: list[Obstacle] = []
    for _ in range(n):
        cx = rng.uniform(xmin + margin, xmax - margin)
        cy = rng.uniform(ymin + margin, ymax - margin)
        r = rng.uniform(0.45, 0.85)
        obstacles.append(Obstacle(np.array([cx, cy]), r))
    return obstacles


def scenario_single_exit(
    bounds: tuple[float, float, float, float] = (0.0, 40.0, 0.0, 40.0),
) -> list[Obstacle]:
    """U-shaped enclosure with 5 m gap at the top."""
    xmin, xmax, ymin, ymax = bounds
    cx = 0.5 * (xmin + xmax)
    gap_half = 2.5
    walls: list[Obstacle] = []
    for y in np.linspace(ymin + 0.10 * (ymax - ymin), ymax - 0.10 * (ymax - ymin), 9):
        walls.append(Obstacle(np.array([xmin + 0.15 * (xmax - xmin), y]), 0.55))
    for y in np.linspace(ymin + 0.10 * (ymax - ymin), ymax - 0.175 * (ymax - ymin), 8):
        walls.append(Obstacle(np.array([xmax - 0.15 * (xmax - xmin), y]), 0.55))
    for x in np.linspace(xmin + 0.20 * (xmax - xmin), xmax - 0.20 * (xmax - xmin), 7):
        walls.append(Obstacle(np.array([x, ymin + 0.10 * (ymax - ymin)]), 0.55))
    top_y = ymax - 0.125 * (ymax - ymin)
    for x in np.linspace(xmin + 0.1125 * (xmax - xmin), cx - gap_half - 0.5, 4):
        walls.append(Obstacle(np.array([x, top_y]), 0.55))
    for x in np.linspace(cx + gap_half + 0.5, xmax - 0.1125 * (xmax - xmin), 4):
        walls.append(Obstacle(np.array([x, top_y]), 0.55))
    return walls


def build_scenario(name: str, seed: int = 0) -> list[Obstacle]:
    if name == "free":
        return scenario_free()
    if name == "random_obstacles":
        return scenario_random_obstacles(seed=seed)
    if name == "single_exit":
        return scenario_single_exit()
    raise ValueError(f"Unknown scenario: {name}")
