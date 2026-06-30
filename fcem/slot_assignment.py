"""Slot assignment via permutation search with full J_ij cost."""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

from common.dynamics import norm
from common.obstacles import Obstacle
from metrics.sync import (
    angular_mismatch_cost,
    estimate_arrival_times,
    ordered_coverage_cost,
    segment_obstacle_cost,
    sector_violation_cost,
    sync_coverage,
)


@dataclass
class AssignmentWeights:
    w_reach: float = 1.0
    w_ang: float = 0.30
    w_cov: float = 0.0
    w_sector: float = 0.0
    w_sync: float = 0.50
    w_switch: float = 1.0
    w_safe: float = 0.40


def assignment_weights_from_config(
    fcem_cfg: dict,
    ablation: dict | None = None,
) -> AssignmentWeights:
    raw = fcem_cfg.get("assignment_weights", {})
    weights = AssignmentWeights(
        w_reach=float(raw.get("w_reach", 1.0)),
        w_ang=float(raw.get("w_ang", 0.30)),
        w_cov=float(raw.get("w_cov", 0.0)),
        w_sector=float(raw.get("w_sector", 0.0)),
        w_sync=float(raw.get("w_sync", 0.50)),
        w_switch=float(raw.get("w_switch", 1.0)),
        w_safe=float(raw.get("w_safe", 0.40)),
    )
    if ablation and ablation.get("ablate_no_sync_assign", False):
        weights.w_sync = 0.0
    return weights


def assignment_cost_jij(
    pursuers: np.ndarray,
    pursuer_v: np.ndarray,
    slots: np.ndarray,
    assignment: tuple[int, ...],
    prev_assignment: tuple[int, ...] | None,
    target: np.ndarray,
    obstacles: list[Obstacle],
    weights: AssignmentWeights,
    inertia: float,
    tau_T: float,
    vmax: float,
    v_min_frac: float = 0.15,
    clearance: float = 0.55,
    ablate_nearest_assign: bool = False,
) -> tuple[float, dict[str, float]]:
    J_reach = 0.0
    J_ang = 0.0
    J_sector = 0.0
    J_switch = 0.0
    J_safe = 0.0

    for i, j in enumerate(assignment):
        slot = slots[j]
        J_reach += norm(pursuers[i] - slot)
        J_ang += angular_mismatch_cost(pursuers[i], slot, target)
        J_sector += sector_violation_cost(pursuers[i], slot, target, len(assignment))
        J_safe += segment_obstacle_cost(pursuers[i], slot, obstacles, clearance)
        if not ablate_nearest_assign and prev_assignment is not None and j != prev_assignment[i]:
            J_switch += inertia

    J_cov, C_cov_ordered = ordered_coverage_cost(pursuers, slots, assignment, target)
    T_hats = estimate_arrival_times(
        pursuers, pursuer_v, slots, assignment, vmax, v_min_frac
    )
    C_sync = sync_coverage(T_hats, tau_T)
    J_sync = 1.0 - C_sync

    total = (
        weights.w_reach * J_reach
        + weights.w_ang * J_ang
        + weights.w_cov * J_cov
        + weights.w_sector * J_sector
        + weights.w_sync * J_sync
        + weights.w_switch * J_switch
        + weights.w_safe * J_safe
    )
    components = {
        "J_reach": float(J_reach),
        "J_ang": float(J_ang),
        "J_cov": float(J_cov),
        "C_cov_ordered": float(C_cov_ordered),
        "J_sector": float(J_sector),
        "J_sync": float(J_sync),
        "J_switch": float(J_switch),
        "J_safe": float(J_safe),
        "C_sync": float(C_sync),
        "T_hat_max": float(np.max(T_hats)),
        "T_hat_min": float(np.min(T_hats)),
        "T_hat_spread": float(np.max(T_hats) - np.min(T_hats)),
    }
    return float(total), components


def assign_slots(
    pursuers: np.ndarray,
    slots: np.ndarray,
    prev_assignment: tuple[int, ...] | None = None,
    inertia: float = 0.12,
    ablate_nearest_assign: bool = False,
    pursuer_v: np.ndarray | None = None,
    target: np.ndarray | None = None,
    obstacles: list[Obstacle] | None = None,
    weights: AssignmentWeights | None = None,
    tau_T: float = 2.5,
    vmax: float = 5.0,
    v_min_frac: float = 0.15,
    clearance: float = 0.55,
) -> tuple[tuple[int, ...], float, dict[str, float]]:
    n = len(pursuers)
    weights = weights or AssignmentWeights()
    obstacles = obstacles or []
    pursuer_v = pursuer_v if pursuer_v is not None else np.zeros_like(pursuers)
    target = target if target is not None else np.mean(slots, axis=0)

    if ablate_nearest_assign:
        remaining = set(range(n))
        greedy: list[int] = []
        for i in range(n):
            j = min(remaining, key=lambda idx: norm(pursuers[i] - slots[idx]))
            greedy.append(j)
            remaining.remove(j)
        assignment = tuple(greedy)
        cost, components = assignment_cost_jij(
            pursuers,
            pursuer_v,
            slots,
            assignment,
            None,
            target,
            obstacles,
            weights,
            0.0,
            tau_T,
            vmax,
            v_min_frac,
            clearance,
            True,
        )
        return assignment, cost, components

    best_cost = None
    best_perm = None
    best_components: dict[str, float] = {}

    for perm in itertools.permutations(range(n), n):
        cost, components = assignment_cost_jij(
            pursuers,
            pursuer_v,
            slots,
            perm,
            prev_assignment,
            target,
            obstacles,
            weights,
            inertia,
            tau_T,
            vmax,
            v_min_frac,
            clearance,
            ablate_nearest_assign,
        )
        if best_cost is None or cost < best_cost:
            best_cost = cost
            best_perm = perm
            best_components = components

    assert best_perm is not None
    return best_perm, float(best_cost), best_components


def score_assignment(
    manifold_structure_score: float,
    manifold_blocker_score: float,
    assignment_cost: float,
    infeasible: bool,
    infeasible_penalty: float = 8.0,
    assignment_score_scale: float = 0.15,
) -> float:
    return (
        manifold_structure_score
        + manifold_blocker_score
        - assignment_cost * assignment_score_scale
        - (infeasible_penalty if infeasible else 0.0)
    )
