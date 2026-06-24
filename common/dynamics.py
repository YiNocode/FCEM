"""2D second-order point-mass dynamics utilities."""

from __future__ import annotations

import numpy as np


def norm(x: np.ndarray) -> float:
    return float(np.linalg.norm(x))


def unit(x: np.ndarray) -> np.ndarray:
    n = norm(x)
    if n < 1e-9:
        return np.zeros_like(x)
    return x / n


def clip_norm(x: np.ndarray, max_norm: float) -> np.ndarray:
    n = norm(x)
    if n > max_norm:
        return x / n * max_norm
    return x


def wrap_angle(a: float) -> float:
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def integrate_point_mass(
    pos: np.ndarray,
    vel: np.ndarray,
    acc: np.ndarray,
    dt: float,
    vmax: float,
    amax: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Semi-implicit Euler with acceleration and velocity limits."""
    acc = clip_norm(acc, amax)
    vel = clip_norm(vel + acc * dt, vmax)
    pos = pos + vel * dt
    return pos, vel


def clip_to_bounds(
    pos: np.ndarray,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    margin: float = 0.15,
) -> np.ndarray:
    pos = pos.copy()
    pos[0] = np.clip(pos[0], xmin + margin, xmax - margin)
    pos[1] = np.clip(pos[1], ymin + margin, ymax - margin)
    return pos


def in_bounds(
    pos: np.ndarray,
    bounds: tuple[float, float, float, float],
    margin: float = 0.15,
    eps: float = 1e-4,
) -> bool:
    xmin, xmax, ymin, ymax = bounds
    return bool(
        xmin + margin - eps <= pos[0] <= xmax - margin + eps
        and ymin + margin - eps <= pos[1] <= ymax - margin + eps
    )


def wall_clearances(
    pos: np.ndarray,
    bounds: tuple[float, float, float, float],
) -> dict[str, float]:
    xmin, xmax, ymin, ymax = bounds
    left = float(pos[0] - xmin)
    right = float(xmax - pos[0])
    bottom = float(pos[1] - ymin)
    top = float(ymax - pos[1])
    return {
        "left": left,
        "right": right,
        "bottom": bottom,
        "top": top,
        "min": min(left, right, bottom, top),
    }
