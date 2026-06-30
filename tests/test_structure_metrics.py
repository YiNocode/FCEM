"""Tests for full-circle angular gap metrics."""

from __future__ import annotations

import math

import numpy as np
import pytest

from fcem.boundary_trap import TrapState, structural_metrics_free_cone
from metrics.structure import (
    angular_gaps_rad,
    max_angular_gap_rad,
    structural_metrics_from_positions,
)


def test_angular_gaps_include_wraparound():
    bearings = np.array([0.0, math.radians(10.0), math.radians(350.0)])
    gaps = angular_gaps_rad(bearings)
    assert len(gaps) == 3
    assert max(gaps) == pytest.approx(math.radians(340.0), rel=1e-6)


def test_three_pursuers_g_max_at_least_120_deg():
    target = np.array([0.0, 0.0])
    pursuers = np.array(
        [
            [1.0, 0.0],
            [math.cos(math.radians(130.0)), math.sin(math.radians(130.0))],
            [math.cos(math.radians(210.0)), math.sin(math.radians(210.0))],
        ]
    )
    m = structural_metrics_from_positions(target, pursuers)
    assert math.degrees(m["G_max"]) >= 120.0 - 1e-6


def test_trap_mode_keeps_canonical_g_max_not_g_free():
    target = np.array([1.0, 1.0])
    pursuers = np.array(
        [
            [1.5, 1.2],
            [0.8, 1.4],
            [1.2, 0.6],
        ]
    )
    trap = TrapState(
        mode="corner",
        corner="bottom_left",
        theta_min=0.0,
        theta_max=math.pi / 2,
        phi_free=math.pi / 2,
        d_left=1.0,
        d_right=39.0,
        d_bottom=1.0,
        d_top=39.0,
        near_boundary=True,
        near_corner=True,
    )
    m = structural_metrics_free_cone(target, pursuers, trap)
    assert math.degrees(m["G_max"]) >= 120.0 - 1e-6
    assert m["G_free"] <= trap.phi_free + 1e-9


def test_max_angular_gap_matches_structural_metrics():
    target = np.zeros(2)
    pursuers = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    bearings = np.arctan2(pursuers[:, 1] - target[1], pursuers[:, 0] - target[0])
    m = structural_metrics_from_positions(target, pursuers)
    assert m["G_max"] == pytest.approx(max_angular_gap_rad(bearings))
