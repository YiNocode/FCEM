"""Evader escape direction and manifold center prediction."""

from __future__ import annotations

import numpy as np

from common.dynamics import clip_norm, norm, unit


def predict_escape_direction(
    evader_pos: np.ndarray,
    evader_vel: np.ndarray,
    pursuer_centroid: np.ndarray,
    ablate_no_esc_dir: bool = False,
) -> np.ndarray:
    if ablate_no_esc_dir:
        return np.array([1.0, 0.0])
    esc = unit(0.75 * evader_vel + 0.55 * (evader_pos - pursuer_centroid))
    if norm(esc) < 1e-9:
        esc = np.array([1.0, 0.0])
    return esc


def predict_manifold_center(
    evader_pos: np.ndarray,
    evader_vel: np.ndarray,
    pursuer_centroid: np.ndarray,
    R: float,
    bounds: tuple[float, float, float, float],
    lookahead_time: float = 0.80,
    center_shift_frac: float = 0.28,
    ablate_no_center_shift: bool = False,
) -> np.ndarray:
    xmin, xmax, ymin, ymax = bounds
    if ablate_no_center_shift:
        center = evader_pos.copy()
    else:
        center_shift = lookahead_time * evader_vel + 0.10 * (evader_pos - pursuer_centroid)
        center_shift = clip_norm(center_shift, center_shift_frac * R)
        center = evader_pos + center_shift
    center[0] = np.clip(center[0], xmin + R * 0.15, xmax - R * 0.15)
    center[1] = np.clip(center[1], ymin + R * 0.15, ymax - R * 0.15)
    return center
