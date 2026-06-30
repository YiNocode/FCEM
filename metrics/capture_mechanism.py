"""Boundary-aware capture mechanism attribution."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from metrics.escape_sector_metrics import normalize_obstacle, normalize_world_bounds

CAPTURE_MECHANISMS = (
    "open_field_capture",
    "boundary_assisted_capture",
    "corner_assisted_capture",
    "obstacle_assisted_capture",
    "mixed_assisted_capture",
    "invalid_or_ambiguous",
)


def wall_distances(
    evader_pos: np.ndarray | list[float],
    bounds: tuple[float, float, float, float] | dict[str, float],
) -> dict[str, float]:
    """Distances from the evader to each rectangular workspace wall."""
    evader = np.asarray(evader_pos, dtype=float).reshape(2)
    xmin, xmax, ymin, ymax = normalize_world_bounds(bounds)
    return {
        "left": float(evader[0] - xmin),
        "right": float(xmax - evader[0]),
        "bottom": float(evader[1] - ymin),
        "top": float(ymax - evader[1]),
    }


def distance_to_nearest_wall(
    evader_pos: np.ndarray | list[float],
    bounds: tuple[float, float, float, float] | dict[str, float],
) -> float:
    return float(min(wall_distances(evader_pos, bounds).values()))


def distance_to_nearest_corner(
    evader_pos: np.ndarray | list[float],
    bounds: tuple[float, float, float, float] | dict[str, float],
) -> float:
    evader = np.asarray(evader_pos, dtype=float).reshape(2)
    xmin, xmax, ymin, ymax = normalize_world_bounds(bounds)
    corners = np.array(
        [[xmin, ymin], [xmin, ymax], [xmax, ymin], [xmax, ymax]],
        dtype=float,
    )
    return float(np.min(np.linalg.norm(corners - evader[None, :], axis=1)))


def environment_blockage_angles(
    evader_pos: np.ndarray | list[float],
    obstacles: list[Any],
    bounds: tuple[float, float, float, float] | dict[str, float],
    *,
    ray_length: float = 6.0,
    num_angles: int = 720,
    num_ray_samples: int = 40,
    obstacle_margin: float = 0.0,
    boundary_margin: float = 0.0,
) -> dict[str, float]:
    """
    Estimate which passive environment feature first makes each finite ray
    infeasible.
    """
    evader = np.asarray(evader_pos, dtype=float).reshape(2)
    xmin, xmax, ymin, ymax = normalize_world_bounds(bounds)
    norm_obs = [normalize_obstacle(obs) for obs in obstacles]
    n = max(int(num_angles), 1)
    angle_step_deg = 360.0 / n
    sample_count = max(int(num_ray_samples), 1)

    boundary_hits = np.zeros(n, dtype=bool)
    obstacle_hits = np.zeros(n, dtype=bool)

    for i, angle in enumerate(np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)):
        direction = np.array([math.cos(float(angle)), math.sin(float(angle))])
        for j in range(sample_count + 1):
            s = float(ray_length) * j / sample_count
            point = evader + s * direction
            outside = (
                point[0] < xmin + boundary_margin
                or point[0] > xmax - boundary_margin
                or point[1] < ymin + boundary_margin
                or point[1] > ymax - boundary_margin
            )
            if outside:
                boundary_hits[i] = True
                break

            hit_obstacle = any(
                np.linalg.norm(point - center) <= radius + obstacle_margin
                for center, radius in norm_obs
            )
            if hit_obstacle:
                obstacle_hits[i] = True
                break

    return {
        "boundary_blocked_angle_deg": float(boundary_hits.sum()) * angle_step_deg,
        "obstacle_blocked_angle_deg": float(obstacle_hits.sum()) * angle_step_deg,
    }


def classify_capture_mechanism(
    evader_pos: np.ndarray | list[float] | None,
    obstacles: list[Any],
    bounds: tuple[float, float, float, float] | dict[str, float],
    escape_metrics: dict[str, Any],
    *,
    valid_distance_capture: bool,
    valid_escape_sector_capture: bool,
    ray_length: float = 6.0,
    num_angles: int = 720,
    num_ray_samples: int = 40,
    obstacle_margin: float = 0.0,
    boundary_margin: float = 0.0,
) -> tuple[str, dict[str, float]]:
    """Return a capture mechanism label and supporting passive-blockage values."""
    if evader_pos is None or not valid_distance_capture or not valid_escape_sector_capture:
        return "invalid_or_ambiguous", {
            "boundary_blocked_angle_deg": float("nan"),
            "obstacle_blocked_angle_deg": float("nan"),
        }

    env = environment_blockage_angles(
        evader_pos,
        obstacles,
        bounds,
        ray_length=ray_length,
        num_angles=num_angles,
        num_ray_samples=num_ray_samples,
        obstacle_margin=obstacle_margin,
        boundary_margin=boundary_margin,
    )
    angle_tol = 360.0 / max(int(num_angles), 1) + 1e-9
    boundary_active = env["boundary_blocked_angle_deg"] > angle_tol
    obstacle_active = env["obstacle_blocked_angle_deg"] > angle_tol

    walls = sorted(wall_distances(evader_pos, bounds).values())
    near_one_wall = walls[0] <= ray_length + boundary_margin + 1e-9
    near_corner = len(walls) >= 2 and walls[1] <= ray_length + boundary_margin + 1e-9

    free_angle = float(escape_metrics.get("free_escape_angle_deg", float("nan")))
    if not math.isfinite(free_angle):
        return "invalid_or_ambiguous", env

    if free_angle >= 360.0 - angle_tol and not boundary_active and not obstacle_active:
        return "open_field_capture", env

    sources: list[str] = []
    if boundary_active and near_corner:
        sources.append("corner")
    elif boundary_active and near_one_wall:
        sources.append("boundary")
    elif boundary_active:
        sources.append("boundary")

    if obstacle_active:
        sources.append("obstacle")

    if len(sources) > 1:
        return "mixed_assisted_capture", env
    if sources == ["corner"]:
        return "corner_assisted_capture", env
    if sources == ["boundary"]:
        return "boundary_assisted_capture", env
    if sources == ["obstacle"]:
        return "obstacle_assisted_capture", env
    return "open_field_capture", env
