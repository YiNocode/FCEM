"""Tests for selectable capture modes and validity flags."""

from __future__ import annotations

import math

import numpy as np

from common.capture import evaluate_capture_conditions


def _escape(g_esc_deg: float = 10.0, c_esc: float = 0.9) -> dict[str, float]:
    return {"G_esc_deg": g_esc_deg, "C_esc": c_esc}


def test_distance_only_mode():
    evader = np.array([0.0, 0.0])
    pursuers = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    flags = evaluate_capture_conditions(
        pursuers,
        evader,
        capture_radius=2.0,
        g_max=math.radians(200.0),
        g_max_allowed=math.radians(140.0),
        escape_metrics=_escape(g_esc_deg=90.0, c_esc=0.1),
        capture_mode="distance_only",
    )
    assert flags["capture_condition_valid_distance"] is True
    assert flags["capture_condition_valid_full_circle"] is False
    assert flags["capture_condition_valid_escape_sector"] is False
    assert flags["captured"] is True


def test_full_circle_mode():
    evader = np.array([0.0, 0.0])
    pursuers = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    flags = evaluate_capture_conditions(
        pursuers,
        evader,
        capture_radius=2.0,
        g_max=math.radians(120.0),
        g_max_allowed=math.radians(140.0),
        escape_metrics=_escape(g_esc_deg=90.0, c_esc=0.1),
        capture_mode="full_circle",
    )
    assert flags["capture_condition_valid_full_circle"] is True
    assert flags["captured"] is True


def test_escape_sector_mode():
    evader = np.array([0.0, 0.0])
    pursuers = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    flags = evaluate_capture_conditions(
        pursuers,
        evader,
        capture_radius=2.0,
        g_max=math.radians(200.0),
        g_max_allowed=math.radians(140.0),
        escape_metrics=_escape(g_esc_deg=20.0, c_esc=0.9),
        g_esc_allow_deg=30.0,
        c_esc_min=0.85,
        capture_mode="escape_sector",
    )
    assert flags["capture_condition_valid_escape_sector"] is True
    assert flags["captured"] is True


def test_escape_sector_rejects_large_gap():
    evader = np.array([0.0, 0.0])
    pursuers = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    flags = evaluate_capture_conditions(
        pursuers,
        evader,
        capture_radius=2.0,
        g_max=math.radians(120.0),
        g_max_allowed=math.radians(140.0),
        escape_metrics=_escape(g_esc_deg=60.0, c_esc=0.9),
        g_esc_allow_deg=30.0,
        capture_mode="escape_sector",
    )
    assert flags["capture_condition_valid_escape_sector"] is False
    assert flags["captured"] is False


def test_all_validity_flags_logged_independently():
    evader = np.array([0.0, 0.0])
    pursuers = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    flags = evaluate_capture_conditions(
        pursuers,
        evader,
        capture_radius=2.0,
        g_max=math.radians(120.0),
        g_max_allowed=math.radians(140.0),
        escape_metrics=_escape(g_esc_deg=20.0, c_esc=0.9),
        g_esc_allow_deg=30.0,
        c_esc_min=0.85,
        capture_mode="escape_sector",
    )
    assert flags["capture_condition_valid_distance"] is True
    assert flags["capture_condition_valid_full_circle"] is True
    assert flags["capture_condition_valid_escape_sector"] is True
