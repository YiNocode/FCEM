"""Boundary-aware feasible escape-sector metrics."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from common.obstacles import Obstacle


def normalize_obstacle(obs: Any) -> tuple[np.ndarray, float]:
    if isinstance(obs, Obstacle):
        return np.asarray(obs.center, dtype=float), float(obs.radius)
    if isinstance(obs, dict):
        return np.asarray(obs["center"], dtype=float), float(obs["radius"])
    if isinstance(obs, (tuple, list)) and len(obs) >= 3:
        return np.asarray(obs[:2], dtype=float), float(obs[2])
    if hasattr(obs, "center") and hasattr(obs, "radius"):
        return np.asarray(obs.center, dtype=float), float(obs.radius)
    raise TypeError(f"Unsupported obstacle format: {type(obs)!r}")


def normalize_world_bounds(
    world_bounds: tuple[float, float, float, float] | dict[str, float],
) -> tuple[float, float, float, float]:
    if isinstance(world_bounds, dict):
        return (
            float(world_bounds["xmin"]),
            float(world_bounds["xmax"]),
            float(world_bounds["ymin"]),
            float(world_bounds["ymax"]),
        )
    xmin, xmax, ymin, ymax = world_bounds
    return float(xmin), float(xmax), float(ymin), float(ymax)


def escape_metrics_config_from_config(config: dict[str, Any]) -> dict[str, Any]:
    """Read escape-sector parameters from experiment config."""
    metrics_cfg = config.get("metrics", {})
    if not isinstance(metrics_cfg, dict):
        metrics_cfg = {}

    exit_config = None
    scenario = config.get("scenario", {})
    if isinstance(scenario, dict):
        exit_config = scenario.get("exit_config")
    if exit_config is None:
        exit_config = config.get("exit_config")

    c_esc_min = metrics_cfg.get("C_esc_min", 0.85)
    if c_esc_min is not None:
        c_esc_min = float(c_esc_min)

    return {
        "ray_length": float(metrics_cfg.get("escape_ray_length", 6.0)),
        "num_angles": int(metrics_cfg.get("escape_num_angles", 720)),
        "num_ray_samples": int(metrics_cfg.get("escape_num_ray_samples", 40)),
        "pursuer_block_radius": float(metrics_cfg.get("pursuer_block_radius", 1.0)),
        "min_forward_block_dist": float(metrics_cfg.get("min_forward_block_dist", 0.0)),
        "obstacle_margin": float(metrics_cfg.get("obstacle_margin", 0.0)),
        "boundary_margin": float(metrics_cfg.get("boundary_margin", 0.0)),
        "exit_config": exit_config,
        "G_esc_allow_deg": float(metrics_cfg.get("G_esc_allow_deg", 30.0)),
        "C_esc_min": c_esc_min,
    }


def max_contiguous_true_arc_deg(mask: np.ndarray, angle_step_deg: float) -> float:
    """Given a circular boolean mask, return the maximum contiguous True arc in degrees."""
    n = len(mask)
    if n == 0:
        return 0.0
    if not np.any(mask):
        return 0.0
    if np.all(mask):
        return n * angle_step_deg

    doubled = np.concatenate([mask, mask])
    best = 0
    run = 0
    for val in doubled:
        if val:
            run += 1
            best = max(best, run)
        else:
            run = 0
    best = min(best, n)
    return best * angle_step_deg


def _ray_feasible(
    evader_pos: np.ndarray,
    angle: float,
    ray_length: float,
    num_ray_samples: int,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    boundary_margin: float,
    obstacles: list[tuple[np.ndarray, float]],
    obstacle_margin: float,
) -> bool:
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    for i in range(num_ray_samples + 1):
        s = ray_length * i / max(num_ray_samples, 1)
        px = evader_pos[0] + s * cos_a
        py = evader_pos[1] + s * sin_a
        if (
            px < xmin + boundary_margin
            or px > xmax - boundary_margin
            or py < ymin + boundary_margin
            or py > ymax - boundary_margin
        ):
            return False
        for center, radius in obstacles:
            if np.hypot(px - center[0], py - center[1]) <= radius + obstacle_margin:
                return False
    return True


def _ray_blocked_by_pursuer(
    evader_pos: np.ndarray,
    angle: float,
    ray_length: float,
    pursuer: np.ndarray,
    pursuer_block_radius: float,
    min_forward: float = 0.05,
) -> bool:
    """
    Block a ray only if the pursuer lies within ``pursuer_block_radius`` of a
    point on the finite ray segment. The projection is not clamped: a pursuer
    blocks only when ``0 <= s <= ray_length``.
    """
    _ = min_forward
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    direction = np.array([cos_a, sin_a])
    rel = pursuer - evader_pos
    s = float(np.dot(rel, direction))
    if s < 0.0 or s > ray_length:
        return False
    closest = evader_pos + s * direction
    return float(np.linalg.norm(pursuer - closest)) <= pursuer_block_radius


def escape_status_from_values(
    free_escape_angle_deg: float,
    blocked_escape_angle_deg: float,
    unblocked_escape_angle_deg: float,
    C_esc: float,
    G_esc_deg: float,
    *,
    G_esc_allow_deg: float = 30.0,
    C_esc_min: float | None = 0.85,
    angle_eps_deg: float = 1e-9,
) -> str:
    """Classify the current feasible escape-sector state."""
    vals = (
        free_escape_angle_deg,
        blocked_escape_angle_deg,
        unblocked_escape_angle_deg,
        C_esc,
        G_esc_deg,
    )
    if any(not math.isfinite(float(v)) for v in vals):
        return "invalid"
    if free_escape_angle_deg <= angle_eps_deg:
        return "no_feasible_escape"
    c_ok = True if C_esc_min is None else C_esc >= float(C_esc_min)
    if unblocked_escape_angle_deg <= angle_eps_deg or (
        c_ok and G_esc_deg <= G_esc_allow_deg
    ):
        return "closed"
    if blocked_escape_angle_deg > angle_eps_deg:
        return "partially_blocked"
    return "open"


def escape_status_from_metrics(
    metrics: dict[str, Any],
    *,
    G_esc_allow_deg: float = 30.0,
    C_esc_min: float | None = 0.85,
    angle_eps_deg: float = 1e-9,
) -> str:
    """Classify escape status from a metric dictionary."""
    return escape_status_from_values(
        float(metrics.get("free_escape_angle_deg", float("nan"))),
        float(metrics.get("blocked_escape_angle_deg", float("nan"))),
        float(metrics.get("unblocked_escape_angle_deg", float("nan"))),
        float(metrics.get("C_esc", float("nan"))),
        float(metrics.get("G_esc_deg", float("nan"))),
        G_esc_allow_deg=G_esc_allow_deg,
        C_esc_min=C_esc_min,
        angle_eps_deg=angle_eps_deg,
    )


def compute_exit_blockage(
    evader_pos: np.ndarray,
    feasible_mask: np.ndarray,
    blocked_mask: np.ndarray,
    angles_rad: np.ndarray,
    exit_config: dict[str, Any] | None,
) -> float:
    if not exit_config or not exit_config.get("enabled", True):
        return float("nan")

    center = np.asarray(exit_config["center"], dtype=float)
    half_width = float(exit_config.get("angular_half_width_deg", 20.0))
    direction = center - evader_pos
    if np.linalg.norm(direction) < 1e-9:
        return float("nan")

    exit_angle = math.atan2(direction[1], direction[0]) % (2.0 * math.pi)
    half_width_rad = math.radians(half_width)
    n = len(angles_rad)
    angle_step_deg = 360.0 / n

    sector_feasible = 0
    sector_blocked = 0
    for i, theta in enumerate(angles_rad):
        delta = (theta - exit_angle + math.pi) % (2.0 * math.pi) - math.pi
        if abs(delta) > half_width_rad:
            continue
        if feasible_mask[i]:
            sector_feasible += 1
            if blocked_mask[i]:
                sector_blocked += 1

    if sector_feasible == 0:
        return 1.0
    return sector_blocked / sector_feasible


def _validate_full_circle_gaps(pursuers: np.ndarray, evader_pos: np.ndarray) -> None:
    if len(pursuers) == 0:
        return
    theta = np.arctan2(pursuers[:, 1] - evader_pos[1], pursuers[:, 0] - evader_pos[0])
    theta = np.mod(theta, 2.0 * np.pi)
    theta = np.sort(theta)
    gaps = np.diff(np.r_[theta, theta[0] + 2.0 * np.pi])
    assert abs(float(np.sum(gaps)) - 2.0 * np.pi) < 1e-6
    assert float(np.max(gaps)) * 180.0 / np.pi >= 360.0 / len(pursuers) - 1e-6


def compute_escape_sector_metrics(
    evader_pos: np.ndarray,
    pursuer_positions: np.ndarray,
    obstacles: list[Any],
    world_bounds: tuple[float, float, float, float] | dict[str, float],
    ray_length: float = 6.0,
    num_angles: int = 720,
    num_ray_samples: int = 40,
    pursuer_block_radius: float = 1.0,
    obstacle_margin: float = 0.0,
    boundary_margin: float = 0.0,
    exit_config: dict[str, Any] | None = None,
    min_forward_block_dist: float = 0.0,
    G_esc_allow_deg: float = 30.0,
    C_esc_min: float | None = 0.85,
    validate: bool = False,
) -> dict[str, Any]:
    """
    Compute boundary-aware feasible escape-sector metrics.

    Returns dict with scalar metrics and boolean masks (for tests / debugging).
    """
    evader_pos = np.asarray(evader_pos, dtype=float).reshape(2)
    pursuer_positions = np.asarray(pursuer_positions, dtype=float).reshape(-1, 2)
    xmin, xmax, ymin, ymax = normalize_world_bounds(world_bounds)
    norm_obs = [normalize_obstacle(o) for o in obstacles]

    angles_rad = np.linspace(0.0, 2.0 * np.pi, num_angles, endpoint=False)
    angle_step_deg = 360.0 / num_angles

    feasible_mask = np.zeros(num_angles, dtype=bool)
    blocked_mask = np.zeros(num_angles, dtype=bool)

    for i, angle in enumerate(angles_rad):
        if _ray_feasible(
            evader_pos,
            float(angle),
            ray_length,
            num_ray_samples,
            xmin,
            xmax,
            ymin,
            ymax,
            boundary_margin,
            norm_obs,
            obstacle_margin,
        ):
            feasible_mask[i] = True
            for pursuer in pursuer_positions:
                if _ray_blocked_by_pursuer(
                    evader_pos,
                    float(angle),
                    ray_length,
                    pursuer,
                    pursuer_block_radius,
                    min_forward=min_forward_block_dist,
                ):
                    blocked_mask[i] = True
                    break

    unblocked_mask = feasible_mask & (~blocked_mask)

    free_escape_angle_deg = float(feasible_mask.sum()) * angle_step_deg
    blocked_escape_angle_deg = float((feasible_mask & blocked_mask).sum()) * angle_step_deg
    unblocked_escape_angle_deg = float(unblocked_mask.sum()) * angle_step_deg

    if free_escape_angle_deg <= 0.0:
        C_esc = 1.0
        G_esc_deg = 0.0
        free_escape_angle_deg = 0.0
        unblocked_escape_angle_deg = 0.0
        blocked_escape_angle_deg = 0.0
    elif unblocked_escape_angle_deg <= 0.0:
        C_esc = 1.0
        G_esc_deg = 0.0
    elif blocked_escape_angle_deg <= 0.0:
        C_esc = 0.0
        G_esc_deg = max_contiguous_true_arc_deg(feasible_mask, angle_step_deg)
    else:
        C_esc = 1.0 - unblocked_escape_angle_deg / free_escape_angle_deg
        G_esc_deg = max_contiguous_true_arc_deg(unblocked_mask, angle_step_deg)

    exit_blockage = compute_exit_blockage(
        evader_pos, feasible_mask, blocked_mask, angles_rad, exit_config
    )
    escape_status = escape_status_from_values(
        free_escape_angle_deg,
        blocked_escape_angle_deg,
        unblocked_escape_angle_deg,
        C_esc,
        G_esc_deg,
        G_esc_allow_deg=G_esc_allow_deg,
        C_esc_min=C_esc_min,
        angle_eps_deg=angle_step_deg + 1e-9,
    )

    if validate:
        tolerance = angle_step_deg + 1e-6
        _validate_full_circle_gaps(pursuer_positions, evader_pos)
        assert 0.0 <= C_esc <= 1.0 + 1e-9
        assert 0.0 <= G_esc_deg <= 360.0 + 1e-6
        assert abs(
            free_escape_angle_deg - blocked_escape_angle_deg - unblocked_escape_angle_deg
        ) < tolerance

    return {
        "C_esc": float(C_esc),
        "G_esc_deg": float(G_esc_deg),
        "free_escape_angle_deg": float(free_escape_angle_deg),
        "blocked_escape_angle_deg": float(blocked_escape_angle_deg),
        "unblocked_escape_angle_deg": float(unblocked_escape_angle_deg),
        "feasible_mask": feasible_mask,
        "blocked_mask": blocked_mask,
        "unblocked_mask": unblocked_mask,
        "angles_rad": angles_rad,
        "exit_blockage": exit_blockage,
        "escape_status": escape_status,
    }


def escape_sector_from_step(
    step: dict[str, Any],
    obstacles: list[Any],
    bounds: tuple[float, float, float, float],
    config: dict[str, Any],
) -> dict[str, float] | None:
    """Recompute escape-sector metrics from a logged step dict."""
    evader = step.get("evader")
    pursuers = step.get("pursuers")
    if evader is None or pursuers is None:
        return None
    esc_cfg = escape_metrics_config_from_config(config)
    result = compute_escape_sector_metrics(
        np.asarray(evader, dtype=float),
        np.asarray(pursuers, dtype=float),
        obstacles,
        bounds,
        ray_length=esc_cfg["ray_length"],
        num_angles=esc_cfg["num_angles"],
        num_ray_samples=esc_cfg["num_ray_samples"],
        pursuer_block_radius=esc_cfg["pursuer_block_radius"],
        obstacle_margin=esc_cfg["obstacle_margin"],
        boundary_margin=esc_cfg["boundary_margin"],
        exit_config=esc_cfg["exit_config"],
        min_forward_block_dist=esc_cfg["min_forward_block_dist"],
        G_esc_allow_deg=esc_cfg["G_esc_allow_deg"],
        C_esc_min=esc_cfg["C_esc_min"],
    )
    return {
        "C_esc": result["C_esc"],
        "G_esc_deg": result["G_esc_deg"],
        "free_escape_angle_deg": result["free_escape_angle_deg"],
        "blocked_escape_angle_deg": result["blocked_escape_angle_deg"],
        "unblocked_escape_angle_deg": result["unblocked_escape_angle_deg"],
        "exit_blockage": result["exit_blockage"],
        "escape_status": result["escape_status"],
    }
