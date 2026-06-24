"""Capture condition checking."""

from __future__ import annotations

import numpy as np


def check_capture(
    pursuers: np.ndarray,
    evader: np.ndarray,
    capture_radius: float,
    g_max: float,
    g_max_allowed: float,
    trap_mode: str = "open_space",
    g_free: float | None = None,
    g_free_allowed: float | None = None,
) -> bool:
    dists = np.linalg.norm(pursuers - evader[None, :], axis=1)
    all_inside = bool(np.all(dists <= capture_radius))
    if trap_mode in ("boundary", "corner") and g_free is not None and g_free_allowed is not None:
        structure_closed = g_free <= g_free_allowed
    else:
        structure_closed = g_max <= g_max_allowed
    return all_inside and structure_closed


def check_capture_with_dists(
    target: np.ndarray,
    pursuers: np.ndarray,
    capture_radius: float,
    g_max: float,
    g_max_allowed: float,
) -> tuple[bool, np.ndarray]:
    dists = np.linalg.norm(pursuers - target[None, :], axis=1)
    captured = bool(np.all(dists <= capture_radius)) and g_max <= g_max_allowed
    return captured, dists
