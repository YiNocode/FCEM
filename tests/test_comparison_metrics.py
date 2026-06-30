"""Tests for cross-method comparison metrics."""

from __future__ import annotations

import math

import numpy as np

from metrics.comparison_metrics import (
    adjusted_time_to_capture_s,
    canonical_structure_from_step,
    episode_structure_stats,
    episode_t_max_s,
    pre_capture_canonical_metrics,
    pre_capture_trap_metrics,
    radar_fixed_scale,
    trap_aware_structure_from_step,
)


def _step(step_id: int, evader: list[float], pursuers: list[list[float]]) -> dict:
    return {"step": step_id, "evader": evader, "pursuers": pursuers}


def test_canonical_structure_from_positions():
    step = _step(0, [0.0, 0.0], [[2.0, 0.0], [-1.0, 1.732], [-1.0, -1.732]])
    m = canonical_structure_from_step(step)
    assert m is not None
    assert 0.0 <= m["D_ang"] <= 1.0
    assert 0.0 <= m["C_cov"] <= 1.0
    assert m["G_max"] > 0.0


def test_pre_capture_canonical_successful_only():
    steps = [
        _step(i, [0.0, 0.0], [[2.0, 0.0], [-1.0, 1.732], [-1.0, -1.732]])
        for i in range(5)
    ]
    m = pre_capture_canonical_metrics(steps, capture_step=4, captured=True, k=2)
    assert m["pre_capture_canonical_D_ang"] != ""
    empty = pre_capture_canonical_metrics(steps, capture_step=4, captured=False, k=2)
    assert empty["pre_capture_canonical_D_ang"] == ""


def test_trap_aware_matches_free_cone_in_open_space():
    config = {
        "world": {"xmin": 0.0, "xmax": 40.0, "ymin": 0.0, "ymax": 40.0},
        "fcem": {"trap": {"boundary_trap_threshold": 5.0, "corner_trap_threshold": 6.0}},
    }
    bounds = (0.0, 40.0, 0.0, 40.0)
    step = _step(0, [20.0, 20.0], [[22.0, 20.0], [19.0, 22.0], [19.0, 18.0]])
    trap_m = trap_aware_structure_from_step(step, bounds, config)
    can_m = canonical_structure_from_step(step)
    assert trap_m is not None and can_m is not None
    assert abs(trap_m["D_ang"] - can_m["D_ang"]) < 1e-9


def test_episode_structure_stats_open_space_fraction():
    config = {
        "world": {"xmin": 0.0, "xmax": 40.0, "ymin": 0.0, "ymax": 40.0},
        "fcem": {"trap": {"boundary_trap_threshold": 5.0, "corner_trap_threshold": 6.0}},
    }
    bounds = (0.0, 40.0, 0.0, 40.0)
    steps = [
        _step(0, [20.0, 20.0], [[22.0, 20.0], [19.0, 22.0], [19.0, 18.0]]),
        _step(1, [1.0, 1.0], [[3.0, 1.0], [0.0, 2.5], [0.0, -0.5]]),
    ]
    stats = episode_structure_stats(steps, bounds, config)
    assert stats["mean_canonical_D_ang"] != ""
    assert stats["open_space_step_frac"] == 0.5


def test_adjusted_time_to_capture():
    cfg = {"dt": 0.1, "max_steps": 1200}
    t_max = episode_t_max_s(cfg)
    assert t_max == 120.0
    assert adjusted_time_to_capture_s(True, 76.5, t_max) == 76.5
    assert adjusted_time_to_capture_s(False, None, t_max) == 120.0


def test_radar_fixed_scale_g_max():
    assert radar_fixed_scale("inv_g", 120.0) == 1.0
    assert radar_fixed_scale("inv_g", 360.0) == 0.0
    assert abs(radar_fixed_scale("inv_g", 312.6) - 0.1975) < 0.02
    assert abs(radar_fixed_scale("inv_g", 217.1) - 0.5954) < 0.02


def test_radar_fixed_scale_unit_metrics():
    assert radar_fixed_scale("d_ang", 1.2) == 1.0
    assert radar_fixed_scale("inv_ttc", 120.0) == 1.0
    assert radar_fixed_scale("inv_ttc", 60.0) == 1.0


def test_pre_capture_trap_metrics():
    config = {
        "world": {"xmin": 0.0, "xmax": 40.0, "ymin": 0.0, "ymax": 40.0},
        "fcem": {"trap": {"boundary_trap_threshold": 5.0, "corner_trap_threshold": 6.0}},
    }
    bounds = (0.0, 40.0, 0.0, 40.0)
    steps = [_step(i, [1.0, 1.0], [[3.0, 1.0], [0.0, 2.5], [0.0, -0.5]]) for i in range(3)]
    m = pre_capture_trap_metrics(steps, capture_step=2, captured=True, k=2, bounds=bounds, config=config)
    assert m["pre_capture_D_free"] != ""
