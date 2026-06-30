"""Tests for boundary-aware capture mechanism labels."""

from __future__ import annotations

import numpy as np

from metrics.capture_mechanism import classify_capture_mechanism


BOUNDS = (0.0, 40.0, 0.0, 40.0)


def _closed_metrics(free_angle: float = 180.0) -> dict:
    return {
        "free_escape_angle_deg": free_angle,
        "blocked_escape_angle_deg": free_angle,
        "unblocked_escape_angle_deg": 0.0,
        "C_esc": 1.0,
        "G_esc_deg": 0.0,
    }


def test_open_field_capture_label():
    label, details = classify_capture_mechanism(
        np.array([20.0, 20.0]),
        [],
        BOUNDS,
        _closed_metrics(360.0),
        valid_distance_capture=True,
        valid_escape_sector_capture=True,
    )
    assert label == "open_field_capture"
    assert details["boundary_blocked_angle_deg"] == 0.0
    assert details["obstacle_blocked_angle_deg"] == 0.0


def test_boundary_assisted_capture_label():
    label, _ = classify_capture_mechanism(
        np.array([1.0, 20.0]),
        [],
        BOUNDS,
        _closed_metrics(),
        valid_distance_capture=True,
        valid_escape_sector_capture=True,
    )
    assert label == "boundary_assisted_capture"


def test_corner_assisted_capture_label():
    label, _ = classify_capture_mechanism(
        np.array([1.0, 1.0]),
        [],
        BOUNDS,
        _closed_metrics(),
        valid_distance_capture=True,
        valid_escape_sector_capture=True,
    )
    assert label == "corner_assisted_capture"


def test_obstacle_assisted_capture_label():
    label, _ = classify_capture_mechanism(
        np.array([20.0, 20.0]),
        [{"center": [23.0, 20.0], "radius": 1.0}],
        BOUNDS,
        _closed_metrics(),
        valid_distance_capture=True,
        valid_escape_sector_capture=True,
    )
    assert label == "obstacle_assisted_capture"


def test_invalid_capture_label_when_escape_not_valid():
    label, _ = classify_capture_mechanism(
        np.array([20.0, 20.0]),
        [],
        BOUNDS,
        _closed_metrics(),
        valid_distance_capture=True,
        valid_escape_sector_capture=False,
    )
    assert label == "invalid_or_ambiguous"
