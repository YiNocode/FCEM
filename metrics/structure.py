"""Structural encirclement metrics."""

from __future__ import annotations

import numpy as np


def structural_metrics_from_positions(
    target: np.ndarray, pursuers: np.ndarray
) -> dict[str, float]:
    rel = pursuers - target
    angles = np.sort(np.arctan2(rel[:, 1], rel[:, 0]) % (2.0 * np.pi))
    gaps = np.diff(np.r_[angles, angles[0] + 2.0 * np.pi])
    ideal_gap = 2.0 * np.pi / len(pursuers)

    D_ang = 1.0 - np.mean(np.abs(gaps - ideal_gap)) / ideal_gap
    D_ang = float(np.clip(D_ang, 0.0, 1.0))

    C_cov = float(np.clip(np.min(gaps) / ideal_gap, 0.0, 1.0))
    G_max = float(np.max(gaps))

    centroid = np.mean(pursuers, axis=0)
    mean_radius = np.mean(np.linalg.norm(rel, axis=1))
    C_col = float(np.linalg.norm(centroid - target) / (mean_radius + 1e-6))

    return {
        "D_ang": D_ang,
        "C_cov": C_cov,
        "G_max": G_max,
        "C_col": C_col,
    }


def contraction_gate(
    metrics: dict[str, float],
    slot_error: float,
    R: float,
    D_min: float,
    C_min: float,
    G_max_allowed: float,
    slot_error_frac: float,
    slot_error_abs: float,
    trap_mode: str = "open_space",
    g_free_allowed: float | None = None,
    g_max_gate_allowed: float | None = None,
    gate_slot_floor: float = 0.0,
    T_min: float = 0.35,
    C_sync: float = 1.0,
    ablate_no_sync_gate: bool = False,
) -> tuple[float, dict[str, float]]:
    """Guarded contraction gate; uses G_free in boundary/corner trap modes."""
    use_free = trap_mode in ("boundary", "corner") and "G_free" in metrics
    d_key = "D_free" if use_free else "D_ang"
    c_key = "C_free" if use_free else "C_cov"
    g_key = "G_free" if use_free else "G_max"
    g_allow = g_free_allowed if (use_free and g_free_allowed is not None) else G_max_allowed
    g_gate = g_max_gate_allowed if g_max_gate_allowed is not None else g_allow

    q_D = np.clip((metrics.get(d_key, metrics["D_ang"]) - D_min) / (1.0 - D_min + 1e-9), 0.0, 1.0)
    q_C = np.clip((metrics.get(c_key, metrics["C_cov"]) - C_min) / (1.0 - C_min + 1e-9), 0.0, 1.0)
    g_val = metrics.get(g_key, metrics["G_max"])
    q_G_gate = np.clip((g_gate - g_val) / (g_gate + 1e-9), 0.0, 1.0)
    q_G_capture = np.clip((g_allow - g_val) / (g_allow + 1e-9), 0.0, 1.0)
    err_allow = max(slot_error_abs, slot_error_frac * R)
    q_slot = np.clip((err_allow - slot_error) / err_allow, 0.0, 1.0)

    if ablate_no_sync_gate:
        q_T = 1.0
    else:
        q_T = float(np.clip((C_sync - T_min) / (1.0 - T_min + 1e-9), 0.0, 1.0))

    if trap_mode == "open_space":
        q_struct = float(min(q_D, q_C, q_T))
        q = float(min(q_struct, q_G_gate, q_slot))
    else:
        q = float(min(q_D, q_C, q_G_gate, q_slot, q_T))

    if gate_slot_floor > 0.0:
        q = float(max(q, gate_slot_floor * q_slot))

    parts = {
        "q_D": float(q_D),
        "q_C": float(q_C),
        "q_G": float(q_G_gate),
        "q_G_capture": float(q_G_capture),
        "q_slot": float(q_slot),
        "q_T": float(q_T),
        "C_sync": float(C_sync),
        "trap_mode": trap_mode,
        "g_threshold": float(g_allow),
        "g_gate_threshold": float(g_gate),
    }
    return q, parts
