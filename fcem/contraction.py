"""Guarded contraction gate for FCEM radius."""

from __future__ import annotations

import numpy as np


def contraction_gate(
    metrics: dict[str, float],
    slot_error: float,
    R: float,
    D_min: float,
    C_min: float,
    G_max_allowed: float,
    slot_error_frac: float,
    slot_error_abs: float,
    T_min: float = 0.35,
    C_sync: float = 1.0,
    ablate_no_sync_gate: bool = False,
) -> tuple[float, dict[str, float]]:
    q_D = np.clip((metrics["D_ang"] - D_min) / (1.0 - D_min + 1e-9), 0.0, 1.0)
    q_C = np.clip((metrics["C_cov"] - C_min) / (1.0 - C_min + 1e-9), 0.0, 1.0)
    q_G = np.clip((G_max_allowed - metrics["G_max"]) / G_max_allowed, 0.0, 1.0)

    err_allow = max(slot_error_abs, slot_error_frac * R)
    q_slot = np.clip((err_allow - slot_error) / err_allow, 0.0, 1.0)

    if ablate_no_sync_gate:
        q_T = 1.0
    else:
        q_T = float(np.clip((C_sync - T_min) / (1.0 - T_min + 1e-9), 0.0, 1.0))

    q = float(min(q_D, q_C, q_G, q_slot, q_T))
    parts = {
        "q_D": float(q_D),
        "q_C": float(q_C),
        "q_G": float(q_G),
        "q_slot": float(q_slot),
        "q_T": float(q_T),
        "C_sync": float(C_sync),
    }
    return q, parts


def update_radius(
    R: float,
    q: float,
    R_init: float,
    R_terminal: float,
    contraction_rate: float,
    expansion_rate: float,
    enable_guarded_contraction: bool = True,
    fixed_shrink: bool = False,
    can_contract: bool = True,
) -> float:
    if fixed_shrink:
        if can_contract:
            return max(R_terminal, R - contraction_rate)
        return R

    if not enable_guarded_contraction:
        if can_contract:
            return max(R_terminal, R - contraction_rate * 0.5)
        return R

    if q > 0.12 and can_contract:
        return max(R_terminal, R - contraction_rate * q)
    if q <= 0.12:
        return min(R_init, R + expansion_rate)
    return R


def phase_label(R: float, R_init: float, R_terminal: float, min_R_for_closure: float, captured: bool) -> str:
    if captured:
        return "Captured"
    if R > min_R_for_closure:
        return "Closure"
    if R > R_terminal + 0.15:
        return "Guarded contraction"
    return "Terminal capture"
