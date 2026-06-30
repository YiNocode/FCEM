"""Tests for boundary-aware escape-sector metrics."""

from __future__ import annotations

import math

import numpy as np
import pytest

from common.obstacles import Obstacle
from metrics.escape_sector_metrics import (
    compute_escape_sector_metrics,
    max_contiguous_true_arc_deg,
)


def test_max_contiguous_true_arc_wraparound():
    mask = np.array([True, False, False, True, True])
    assert max_contiguous_true_arc_deg(mask, 360.0 / len(mask)) == pytest.approx(216.0, abs=1e-6)


def test_max_contiguous_all_true():
    mask = np.array([True, True, True, True])
    assert max_contiguous_true_arc_deg(mask, 90.0) == pytest.approx(360.0)


def test_max_contiguous_no_true():
    mask = np.array([False, False, False])
    assert max_contiguous_true_arc_deg(mask, 120.0) == 0.0


def test_corner_evader_has_feasible_sector_less_than_360():
    evader = np.array([1.0, 1.0])
    pursuers = np.array([[2.0, 2.0], [2.0, 1.5], [1.5, 2.0]])
    bounds = (0.0, 40.0, 0.0, 40.0)
    result = compute_escape_sector_metrics(
        evader,
        pursuers,
        [],
        bounds,
        ray_length=6.0,
        num_angles=360,
        num_ray_samples=20,
        validate=True,
    )
    assert result["free_escape_angle_deg"] < 360.0
    assert result["free_escape_angle_deg"] > 0.0


def test_obstacle_blocks_feasible_direction():
    evader = np.array([20.0, 20.0])
    pursuers = np.array([[25.0, 20.0], [15.0, 20.0], [20.0, 25.0]])
    obstacles = [Obstacle(np.array([26.0, 20.0]), 1.0)]
    bounds = (0.0, 40.0, 0.0, 40.0)
    result = compute_escape_sector_metrics(
        evader,
        pursuers,
        obstacles,
        bounds,
        ray_length=8.0,
        num_angles=720,
        validate=True,
    )
    assert result["free_escape_angle_deg"] < 360.0


def test_no_feasible_directions_sets_c_esc_one():
    evader = np.array([0.5, 0.5])
    pursuers = np.array([[1.0, 1.0], [1.0, 0.5], [0.5, 1.0]])
    obstacles = [
        Obstacle(np.array([0.5, 0.5]), 5.0),
    ]
    bounds = (0.0, 40.0, 0.0, 40.0)
    result = compute_escape_sector_metrics(
        evader,
        pursuers,
        obstacles,
        bounds,
        ray_length=6.0,
        num_angles=180,
        boundary_margin=0.0,
    )
    assert result["C_esc"] == pytest.approx(1.0)
    assert result["G_esc_deg"] == pytest.approx(0.0)
    assert result["free_escape_angle_deg"] == pytest.approx(0.0)


def test_interior_open_field_free_is_near_360():
    evader = np.array([20.0, 20.0])
    pursuers = np.array([[35.0, 20.0], [20.0, 35.0], [5.0, 20.0]])
    bounds = (0.0, 40.0, 0.0, 40.0)
    result = compute_escape_sector_metrics(
        evader,
        pursuers,
        [],
        bounds,
        ray_length=6.0,
        num_angles=720,
        validate=True,
    )
    assert result["free_escape_angle_deg"] == pytest.approx(360.0, abs=1.0)
    assert result["blocked_escape_angle_deg"] == pytest.approx(0.0, abs=1.0)


def test_pursuer_does_not_block_opposite_ray():
    evader = np.array([20.0, 20.0])
    pursuer = np.array([21.0, 20.0])
    bounds = (0.0, 40.0, 0.0, 40.0)
    result = compute_escape_sector_metrics(
        evader,
        np.array([pursuer]),
        [],
        bounds,
        ray_length=6.0,
        num_angles=720,
        pursuer_block_radius=1.0,
    )
    west_blocked = 0
    for i, theta in enumerate(result["angles_rad"]):
        if not result["feasible_mask"][i]:
            continue
        delta = (theta - math.pi) % (2.0 * math.pi)
        if delta < math.radians(30.0) or delta > 2.0 * math.pi - math.radians(30.0):
            if result["blocked_mask"][i]:
                west_blocked += 1
    assert west_blocked == 0


def test_pursuer_beyond_finite_ray_endpoint_does_not_block():
    evader = np.array([20.0, 20.0])
    pursuer = np.array([26.5, 20.0])
    bounds = (0.0, 40.0, 0.0, 40.0)
    result = compute_escape_sector_metrics(
        evader,
        np.array([pursuer]),
        [],
        bounds,
        ray_length=6.0,
        num_angles=360,
        pursuer_block_radius=1.0,
    )
    east_idx = 0
    assert bool(result["feasible_mask"][east_idx]) is True
    assert bool(result["blocked_mask"][east_idx]) is False
    assert result["escape_status"] in {"open", "partially_blocked", "closed", "no_feasible_escape"}


def test_corner_evader_free_sector_less_than_360_without_pursuers():
    evader = np.array([38.5, 1.4])
    bounds = (0.0, 40.0, 0.0, 40.0)
    result = compute_escape_sector_metrics(
        evader,
        np.zeros((0, 2)),
        [],
        bounds,
        ray_length=6.0,
        num_angles=720,
    )
    assert 90.0 < result["free_escape_angle_deg"] < 180.0
    assert result["blocked_escape_angle_deg"] == pytest.approx(0.0, abs=1.0)


def test_unblocked_feasible_gives_zero_c_esc():
    evader = np.array([20.0, 20.0])
    pursuers = np.array([[100.0, 100.0], [110.0, 100.0], [105.0, 110.0]])
    bounds = (0.0, 40.0, 0.0, 40.0)
    result = compute_escape_sector_metrics(
        evader,
        pursuers,
        [],
        bounds,
        ray_length=4.0,
        num_angles=360,
        pursuer_block_radius=0.5,
        validate=True,
    )
    assert result["C_esc"] == pytest.approx(0.0)
    assert result["G_esc_deg"] > 0.0


def test_angle_conservation():
    evader = np.array([10.0, 10.0])
    pursuers = np.array([[11.0, 10.0], [10.0, 11.0], [9.0, 10.0]])
    bounds = (0.0, 40.0, 0.0, 40.0)
    num_angles = 720
    result = compute_escape_sector_metrics(
        evader,
        pursuers,
        [],
        bounds,
        num_angles=num_angles,
        validate=True,
    )
    tol = 360.0 / num_angles + 1e-6
    assert abs(
        result["free_escape_angle_deg"]
        - result["blocked_escape_angle_deg"]
        - result["unblocked_escape_angle_deg"]
    ) < tol
