"""Slot assignment and executability scoring."""

from __future__ import annotations

import itertools

import numpy as np

from common.dynamics import norm
from common.obstacles import Obstacle
from fcem.slot_assignment import AssignmentWeights, assignment_cost_jij
from metrics.structure import structural_metrics_from_positions


def assign_slots_nearest_with_inertia(
    pursuers: np.ndarray,
    slots: np.ndarray,
    prev_assignment: tuple[int, ...] | None,
    inertia: float = 0.12,
    pursuer_v: np.ndarray | None = None,
    target: np.ndarray | None = None,
    obstacles: list[Obstacle] | None = None,
    weights: AssignmentWeights | None = None,
    tau_T: float = 2.5,
    vmax: float = 5.0,
    v_min_frac: float = 0.15,
    clearance: float = 0.55,
) -> tuple[int, ...]:
    n = len(pursuers)
    weights = weights or AssignmentWeights()
    obstacles = obstacles or []
    pursuer_v = pursuer_v if pursuer_v is not None else np.zeros_like(pursuers)
    target = target if target is not None else np.mean(slots, axis=0)

    best = None
    for perm in itertools.permutations(range(n), n):
        cost, _ = assignment_cost_jij(
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
        )
        if best is None or cost < best[0]:
            best = (cost, perm)
    return best[1]


def assign_slots_nearest(
    pursuers: np.ndarray,
    slots: np.ndarray,
    **kwargs: object,
) -> tuple[int, ...]:
    return assign_slots_nearest_with_inertia(pursuers, slots, None, 0.0, **kwargs)


def assignment_cost(
    pursuers: np.ndarray,
    slots: np.ndarray,
    assignment: tuple[int, ...],
    prev_assignment: tuple[int, ...] | None,
    inertia: float,
    pursuer_v: np.ndarray | None = None,
    target: np.ndarray | None = None,
    obstacles: list[Obstacle] | None = None,
    weights: AssignmentWeights | None = None,
    tau_T: float = 2.5,
    vmax: float = 5.0,
    v_min_frac: float = 0.15,
    clearance: float = 0.55,
) -> tuple[float, dict[str, float]]:
    weights = weights or AssignmentWeights()
    obstacles = obstacles or []
    pursuer_v = pursuer_v if pursuer_v is not None else np.zeros_like(pursuers)
    target = target if target is not None else np.mean(slots, axis=0)
    return assignment_cost_jij(
        pursuers,
        pursuer_v,
        slots,
        assignment,
        prev_assignment,
        target,
        obstacles,
        weights,
        inertia,
        tau_T,
        vmax,
        v_min_frac,
        clearance,
    )


def executability_rollout_score(
    pursuers: np.ndarray,
    pursuer_v: np.ndarray,
    target: np.ndarray,
    slots: np.ndarray,
    assignment: tuple[int, ...],
    dt: float,
    kp: float,
    kd: float,
    vmax: float,
    amax: float,
    horizon: int = 3,
) -> float:
    """
    Short open-loop rollout: lower score is better.
    Combines mean slot tracking error and projected structural metrics.
    """
    pos = pursuers.copy()
    vel = pursuer_v.copy()
    total_err = 0.0

    for _ in range(horizon):
        for i, j in enumerate(assignment):
            slot = slots[j]
            pos_err = slot - pos[i]
            vel_err = -vel[i]
            acc = kp * pos_err + kd * vel_err
            n = np.linalg.norm(acc)
            if n > amax:
                acc = acc / n * amax
            vel[i] = vel[i] + acc * dt
            vn = np.linalg.norm(vel[i])
            if vn > vmax:
                vel[i] = vel[i] / vn * vmax
            pos[i] = pos[i] + vel[i] * dt
            total_err += norm(pos[i] - slot)

    metrics = structural_metrics_from_positions(target, pos)
    mean_err = total_err / (horizon * len(pursuers))
    gap_penalty = max(0.0, metrics["G_max"] - 2.5)
    return mean_err + 0.35 * gap_penalty


def score_candidate(
    pursuers: np.ndarray,
    pursuer_v: np.ndarray,
    target: np.ndarray,
    slots: np.ndarray,
    prev_assignment: tuple[int, ...] | None,
    dt: float,
    kp: float,
    kd: float,
    vmax: float,
    amax: float,
    inertia: float = 0.12,
    enable_rollout: bool = True,
    enable_inertia: bool = True,
    obstacles: list[Obstacle] | None = None,
    weights: AssignmentWeights | None = None,
    tau_T: float = 2.5,
    v_min_frac: float = 0.15,
    clearance: float = 0.55,
) -> tuple[tuple[int, ...], float]:
    assign_kwargs = {
        "pursuer_v": pursuer_v,
        "target": target,
        "obstacles": obstacles,
        "weights": weights,
        "tau_T": tau_T,
        "vmax": vmax,
        "v_min_frac": v_min_frac,
        "clearance": clearance,
    }
    if enable_inertia:
        assignment = assign_slots_nearest_with_inertia(
            pursuers, slots, prev_assignment, inertia, **assign_kwargs
        )
        assign_cost, _ = assignment_cost(
            pursuers, slots, assignment, prev_assignment, inertia, **assign_kwargs
        )
    else:
        assignment = assign_slots_nearest(pursuers, slots, **assign_kwargs)
        assign_cost, _ = assignment_cost(
            pursuers, slots, assignment, None, 0.0, **assign_kwargs
        )

    if enable_rollout:
        rollout = executability_rollout_score(
            pursuers, pursuer_v, target, slots, assignment,
            dt, kp, kd, vmax, amax,
        )
        total = assign_cost + rollout
    else:
        total = assign_cost

    return assignment, total


def select_best_candidate(
    candidates: list[tuple[float, np.ndarray, np.ndarray, np.ndarray]],
    pursuers: np.ndarray,
    pursuer_v: np.ndarray,
    target: np.ndarray,
    prev_assignment: tuple[int, ...] | None,
    dt: float,
    kp: float,
    kd: float,
    vmax: float,
    amax: float,
    inertia: float = 0.12,
    enable_rollout: bool = True,
    enable_inertia: bool = True,
    obstacles: list[Obstacle] | None = None,
    weights: AssignmentWeights | None = None,
    tau_T: float = 2.5,
    v_min_frac: float = 0.15,
    clearance: float = 0.55,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, tuple[int, ...], float]:
    best = None
    for phase_off, angles, slots, curve in candidates:
        assignment, score = score_candidate(
            pursuers, pursuer_v, target, slots, prev_assignment,
            dt, kp, kd, vmax, amax, inertia, enable_rollout, enable_inertia,
            obstacles, weights, tau_T, v_min_frac, clearance,
        )
        if best is None or score < best[0]:
            best = (score, phase_off, angles, slots, curve, assignment)

    assert best is not None
    score, phase_off, angles, slots, curve, assignment = best
    return phase_off, angles, slots, curve, assignment, score
