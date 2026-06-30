"""Geometric helpers for literature baselines (point-mass adaptations)."""

from __future__ import annotations

import math

import numpy as np

from common.dynamics import norm, unit


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
        return evader.copy(), max(d * 2.0, 1.0)

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


def in_evader_safe_disk(
    point: np.ndarray,
    evader: np.ndarray,
    pursuer: np.ndarray,
    mu: float,
) -> bool:
    """True if point lies in evader advantage disk D_i: ||x-e|| <= mu ||x-p||."""
    de = norm(point - evader)
    dp = norm(point - pursuer)
    return de <= mu * dp + 1e-9


def evader_safe_region_area(
    pursuers: np.ndarray,
    evader: np.ndarray,
    mu: float,
    bounds: tuple[float, float, float, float],
    grid_n: int = 48,
) -> float:
    """Grid estimate of Area(S_E) where S_E = intersection of evader advantage disks."""
    xmin, xmax, ymin, ymax = bounds
    n = max(int(grid_n), 8)
    xs = np.linspace(xmin, xmax, n)
    ys = np.linspace(ymin, ymax, n)
    cell_area = ((xmax - xmin) / max(n - 1, 1)) * ((ymax - ymin) / max(n - 1, 1))

    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([xx.ravel(), yy.ravel()], axis=1)
    de = np.linalg.norm(points - evader[None, :], axis=1)
    mask = np.ones(len(points), dtype=bool)
    for i in range(len(pursuers)):
        dp = np.linalg.norm(points - pursuers[i][None, :], axis=1)
        mask &= de <= mu * dp + 1e-9
    return float(mask.sum()) * cell_area


def best_area_shrink_direction(
    pursuer_idx: int,
    pursuers: np.ndarray,
    evader: np.ndarray,
    mu: float,
    bounds: tuple[float, float, float, float],
    preview_delta: float = 0.35,
    direction_samples: int = 16,
    grid_n: int = 48,
) -> np.ndarray:
    """Unit direction that most reduces evader safe-region area when pursuer moves."""
    base_area = evader_safe_region_area(pursuers, evader, mu, bounds, grid_n)
    fallback = unit(evader - pursuers[pursuer_idx])
    if norm(fallback) < 1e-9:
        fallback = np.array([1.0, 0.0])

    best_dir = fallback
    best_reduction = -np.inf
    samples = max(int(direction_samples), 4)
    delta = max(float(preview_delta), 1e-3)

    for k in range(samples):
        angle = 2.0 * math.pi * k / samples
        direction = np.array([math.cos(angle), math.sin(angle)])
        perturbed = pursuers.copy()
        perturbed[pursuer_idx] = pursuers[pursuer_idx] + delta * direction
        new_area = evader_safe_region_area(perturbed, evader, mu, bounds, grid_n)
        reduction = base_area - new_area
        if reduction > best_reduction:
            best_reduction = reduction
            best_dir = direction

    return unit(best_dir)


def shrink_radius(R: float, fcem_cfg: dict, rate_scale: float = 0.5) -> float:
    """Linear contraction for slot-based baselines."""
    return max(fcem_cfg["R_terminal"], R - fcem_cfg["contraction_rate"] * rate_scale)
