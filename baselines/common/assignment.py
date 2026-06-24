"""Assignment helpers for literature baselines."""

from __future__ import annotations

import numpy as np

from fcem.assignment import assign_slots_nearest, assign_slots_nearest_with_inertia


def hungarian_assign(cost_matrix: np.ndarray) -> tuple[int, ...]:
    """Minimum-cost assignment; uses scipy if available."""
    from scipy.optimize import linear_sum_assignment

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    n = cost_matrix.shape[0]
    assignment = [0] * n
    for r, c in zip(row_ind, col_ind):
        assignment[r] = int(c)
    return tuple(assignment)


def cost_matrix_from_positions(
    pursuers: np.ndarray,
    slots: np.ndarray,
) -> np.ndarray:
    n = len(pursuers)
    cost = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            cost[i, j] = float(np.linalg.norm(pursuers[i] - slots[j]))
    return cost


def assign_slots(
    pursuers: np.ndarray,
    slots: np.ndarray,
    method: str = "nearest",
    prev_assignment: tuple[int, ...] | None = None,
    inertia: float = 0.0,
) -> tuple[int, ...]:
    if method == "hungarian":
        return hungarian_assign(cost_matrix_from_positions(pursuers, slots))
    if inertia > 0.0 and prev_assignment is not None:
        return assign_slots_nearest_with_inertia(pursuers, slots, prev_assignment, inertia)
    return assign_slots_nearest(pursuers, slots)
