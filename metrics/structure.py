"""Structural encirclement metrics."""

from __future__ import annotations

import numpy as np


def angular_gaps_rad(bearings: np.ndarray) -> np.ndarray:
    """Return sorted full-circle angular gaps, including the wraparound gap."""
    angles = np.sort(np.asarray(bearings, dtype=float) % (2.0 * np.pi))
    if angles.size == 0:
        return np.array([], dtype=float)
    return np.diff(np.r_[angles, angles[0] + 2.0 * np.pi])


def max_angular_gap_rad(bearings: np.ndarray) -> float:
    gaps = angular_gaps_rad(bearings)
    if gaps.size == 0:
        return 0.0
    return float(np.max(gaps))


def structural_metrics_from_positions(
    target: np.ndarray, pursuers: np.ndarray
) -> dict[str, float]:
    rel = pursuers - target
    angles = np.arctan2(rel[:, 1], rel[:, 0])
    gaps = angular_gaps_rad(angles)
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


def contraction_allowed(
    metrics: dict[str, float],
    fcem_cfg: dict,
    config: dict,
    R: float,
    formation_expanded_latched: bool,
) -> tuple[bool, dict[str, float | bool | str], bool]:
    """Boolean contraction latch used by the high-level radius controller.

    The latch separates the "formation has opened enough to start closure" check
    from the stricter capture-time angular-gap condition. This lets FCEM begin
    contraction after coverage is established without requiring final closure yet.
    """
    c_expand_min = float(fcem_cfg.get("C_expand_min", 0.25))
    min_R_for_closure = float(fcem_cfg.get("min_R_for_closure", 0.0))
    g_capture_allowed = np.deg2rad(float(config.get("G_max_allowed_deg", 140.0)))
    g_contract_threshold = np.deg2rad(
        float(config.get("G_contract_threshold_deg", np.rad2deg(g_capture_allowed)))
    )

    c_cov = float(metrics.get("C_cov", 0.0))
    g_max = float(metrics.get("G_max", np.inf))
    expanded_now = c_cov >= c_expand_min
    latched = bool(formation_expanded_latched or expanded_now)
    g_capture_ok = bool(g_max <= g_capture_allowed)
    g_contract_ok = bool(g_max <= g_contract_threshold)

    parts: dict[str, float | bool | str] = {
        "formation_expanded": latched,
        "expanded_now": expanded_now,
        "C_expand_min": c_expand_min,
        "g_capture_ok": g_capture_ok,
        "g_contract_ok": g_contract_ok,
        "G_capture_allowed": float(g_capture_allowed),
        "G_contract_threshold": float(g_contract_threshold),
        "R": float(R),
        "min_R_for_closure": min_R_for_closure,
    }

    if not latched:
        parts["block_reason"] = "not_expanded"
        return False, parts, latched

    parts["block_reason"] = ""
    return True, parts, latched


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
