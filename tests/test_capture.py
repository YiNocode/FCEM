"""Tests for strict full-circle capture condition (legacy check_capture)."""

from __future__ import annotations

import math

import numpy as np

from common.capture import check_capture, evaluate_capture_conditions


def test_trap_mode_uses_g_free_when_provided() -> None:
    evader = np.array([0.0, 0.0])
    pursuers = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
        ]
    )
    g_max = math.radians(120.0)
    g_free = math.radians(30.0)
    g_max_allowed = math.radians(140.0)
    g_free_allowed = math.radians(50.0)

    captured = check_capture(
        pursuers,
        evader,
        capture_radius=2.0,
        g_max=g_max,
        g_max_allowed=g_max_allowed,
        trap_mode="corner",
        g_free=g_free,
        g_free_allowed=g_free_allowed,
    )

    assert captured is True


def test_trap_mode_accepts_when_g_free_ok_even_if_g_max_large() -> None:
    evader = np.array([0.0, 0.0])
    pursuers = np.array(
        [
            [1.0, 0.0],
            [0.5, 0.1],
            [0.5, -0.1],
        ]
    )
    g_max = math.radians(160.0)
    g_free = math.radians(30.0)

    captured = check_capture(
        pursuers,
        evader,
        capture_radius=2.0,
        g_max=g_max,
        g_max_allowed=math.radians(140.0),
        trap_mode="boundary",
        g_free=g_free,
        g_free_allowed=math.radians(50.0),
    )

    assert captured is True


def test_full_circle_mode_rejects_large_g_max() -> None:
    evader = np.array([0.0, 0.0])
    pursuers = np.array(
        [
            [1.0, 0.0],
            [0.5, 0.1],
            [0.5, -0.1],
        ]
    )
    flags = evaluate_capture_conditions(
        pursuers,
        evader,
        capture_radius=2.0,
        g_max=math.radians(160.0),
        g_max_allowed=math.radians(140.0),
        escape_metrics={"G_esc_deg": 10.0, "C_esc": 0.95},
        capture_mode="full_circle",
    )
    assert flags["capture_condition_valid_full_circle"] is False
    assert flags["captured"] is False
