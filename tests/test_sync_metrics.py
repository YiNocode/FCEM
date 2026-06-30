"""Unit tests for C_sync synchronization metrics and J_ij assignment."""

from __future__ import annotations

import numpy as np
import pytest

from fcem.slot_assignment import AssignmentWeights, assign_slots, assignment_cost_jij
from metrics.structure import contraction_gate
from metrics.sync import (
    estimate_arrival_times,
    ordered_coverage_cost,
    sector_violation_cost,
    sync_coverage,
)


def test_equal_distance_speed_gives_c_sync_near_one() -> None:
    pursuers = np.array([[0.0, 0.0], [0.0, 10.0], [-10.0, 0.0]], dtype=float)
    pursuer_v = np.array([[5.0, 0.0], [0.0, -5.0], [5.0, 0.0]], dtype=float)
    slots = np.array([[10.0, 0.0], [0.0, 0.0], [0.0, 0.0]], dtype=float)
    assignment = (0, 1, 2)

    T_hats = estimate_arrival_times(pursuers, pursuer_v, slots, assignment, vmax=5.0)
    assert np.allclose(T_hats, T_hats[0], rtol=1e-6)
    C_sync = sync_coverage(T_hats, tau_T=2.5)
    assert C_sync == pytest.approx(1.0, abs=1e-9)


def test_one_agent_far_slow_reduces_c_sync() -> None:
  pursuers = np.array([[0.0, 0.0], [0.0, 0.0]], dtype=float)
  pursuer_v = np.array([[5.0, 0.0], [0.5, 0.0]], dtype=float)
  slots = np.array([[10.0, 0.0], [10.0, 0.0]], dtype=float)
  assignment = (0, 1)

  T_hats = estimate_arrival_times(pursuers, pursuer_v, slots, assignment, vmax=5.0)
  C_sync = sync_coverage(T_hats, tau_T=2.5)

  assert C_sync < 1.0
  assert T_hats[1] > T_hats[0]


def test_larger_tau_T_increases_c_sync_for_same_spread() -> None:
    T_hats = np.array([1.0, 4.0])
    c_small = sync_coverage(T_hats, tau_T=1.0)
    c_large = sync_coverage(T_hats, tau_T=4.0)
    assert c_large > c_small


def test_permutation_search_prefers_sync_favorable_assignment() -> None:
    pursuers = np.array([[0.0, 0.0], [8.0, 0.0]], dtype=float)
    pursuer_v = np.array([[5.0, 0.0], [0.5, 0.0]], dtype=float)
    slots = np.array([[10.0, 0.0], [2.0, 0.0]], dtype=float)
    target = np.array([5.0, 0.0])
    weights = AssignmentWeights(w_reach=0.0, w_ang=0.0, w_sync=1.0, w_switch=0.0, w_safe=0.0)

    assignment, _, components = assign_slots(
        pursuers,
        slots,
        prev_assignment=None,
        pursuer_v=pursuer_v,
        target=target,
        weights=weights,
        tau_T=2.5,
        vmax=5.0,
    )

    assert assignment == (1, 0)
    assert components["C_sync"] > sync_coverage(
        estimate_arrival_times(pursuers, pursuer_v, slots, (0, 1), 5.0),
        2.5,
    )


def test_contraction_gate_blocks_when_c_sync_below_t_min() -> None:
    metrics = {"D_ang": 0.9, "C_cov": 0.9, "G_max": 1.0}
    q_good, parts_good = contraction_gate(
        metrics,
        slot_error=0.1,
        R=10.0,
        D_min=0.18,
        C_min=0.08,
        G_max_allowed=2.5,
        slot_error_frac=0.95,
        slot_error_abs=1.5,
        T_min=0.35,
        C_sync=0.9,
    )
    q_bad, parts_bad = contraction_gate(
        metrics,
        slot_error=0.1,
        R=10.0,
        D_min=0.18,
        C_min=0.08,
        G_max_allowed=2.5,
        slot_error_frac=0.95,
        slot_error_abs=1.5,
        T_min=0.35,
        C_sync=0.1,
    )

    assert parts_bad["q_T"] == pytest.approx(0.0)
    assert q_bad < q_good
    assert q_bad == pytest.approx(0.0)


def test_ablate_no_sync_gate_sets_q_t_to_one() -> None:
    metrics = {"D_ang": 0.9, "C_cov": 0.9, "G_max": 1.0}
    q, parts = contraction_gate(
        metrics,
        slot_error=0.1,
        R=10.0,
        D_min=0.18,
        C_min=0.08,
        G_max_allowed=2.5,
        slot_error_frac=0.95,
        slot_error_abs=1.5,
        T_min=0.35,
        C_sync=0.0,
        ablate_no_sync_gate=True,
    )
    assert parts["q_T"] == pytest.approx(1.0)
    assert q > 0.0


def test_assignment_cost_jij_reports_components() -> None:
    pursuers = np.array([[0.0, 0.0], [5.0, 0.0]], dtype=float)
    pursuer_v = np.array([[5.0, 0.0], [5.0, 0.0]], dtype=float)
    slots = np.array([[5.0, 0.0], [0.0, 0.0]], dtype=float)
    target = np.array([2.5, 0.0])
    weights = AssignmentWeights()

    total, components = assignment_cost_jij(
        pursuers,
        pursuer_v,
        slots,
        (0, 1),
        None,
        target,
        [],
        weights,
        inertia=0.12,
        tau_T=2.5,
        vmax=5.0,
    )

    assert total > 0.0
    assert "J_reach" in components
    assert "J_sync" in components
    assert "C_sync" in components
    assert "J_cov" in components
    assert "J_sector" in components
    assert components["J_sync"] == pytest.approx(1.0 - components["C_sync"])


def test_ordered_coverage_prefers_spread_assignment() -> None:
    target = np.array([0.0, 0.0])
    pursuers = np.array(
        [
            [1.0, 0.0],
            [0.95, 0.2],
            [-1.0, 0.0],
        ]
    )
    slots = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
        ]
    )
    good = (0, 1, 2)
    bad = (0, 2, 1)

    j_good, c_good = ordered_coverage_cost(pursuers, slots, good, target)
    j_bad, c_bad = ordered_coverage_cost(pursuers, slots, bad, target)

    assert c_good > c_bad
    assert j_good < j_bad


def test_sector_violation_penalizes_wrong_side_assignment() -> None:
    target = np.array([0.0, 0.0])
    pursuer_same_side = np.array([1.0, 0.0])
    slot_opposite = np.array([-1.0, 0.0])
    slot_same = np.array([1.0, 0.1])

    wrong = sector_violation_cost(pursuer_same_side, slot_opposite, target, n=3)
    right = sector_violation_cost(pursuer_same_side, slot_same, target, n=3)

    assert wrong > right
    assert right == pytest.approx(0.0, abs=1e-9)


def test_assign_slots_weights_cov_and_sector() -> None:
    target = np.array([0.0, 0.0])
    pursuers = np.array(
        [
            [1.0, 0.0],
            [0.9, 0.2],
            [-1.0, 0.0],
        ]
    )
    slots = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
        ]
    )
    weights = AssignmentWeights(
        w_reach=0.0,
        w_ang=0.0,
        w_cov=1.0,
        w_sector=1.0,
        w_sync=0.0,
        w_switch=0.0,
        w_safe=0.0,
    )

    assignment, _, components = assign_slots(
        pursuers,
        slots,
        target=target,
        weights=weights,
    )

    assert assignment == (0, 1, 2)
    assert components["J_cov"] + components["C_cov_ordered"] == pytest.approx(1.0)
