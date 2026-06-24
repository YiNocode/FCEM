"""Tests for pre-capture structural metrics."""

from __future__ import annotations

from metrics.pre_capture import pre_capture_step_window, pre_capture_structure_metrics


def _step(step_id: int, d: float, c: float, g_deg: float, sync: float) -> dict:
    import math

    return {
        "step": step_id,
        "D_ang": d,
        "C_cov": c,
        "G_max": math.radians(g_deg),
        "C_sync": sync,
    }


def test_pre_capture_window_last_k_steps():
    steps = [_step(i, 0.1 * i, 0.2, 90.0 + i, 0.5) for i in range(20)]
    window = pre_capture_step_window(steps, capture_step=15, k=3)
    assert [s["step"] for s in window] == [13, 14, 15]


def test_pre_capture_means_over_window():
    steps = [
        _step(0, 0.2, 0.3, 100.0, 0.4),
        _step(1, 0.4, 0.5, 110.0, 0.6),
        _step(2, 0.6, 0.7, 120.0, 0.8),
        _step(3, 0.8, 0.9, 130.0, 1.0),
    ]
    m = pre_capture_structure_metrics(steps, capture_step=3, captured=True, k=2)
    assert m["pre_capture_n_steps"] == 2
    assert abs(m["pre_capture_D_ang"] - 0.7) < 1e-9
    assert abs(m["pre_capture_C_cov"] - 0.8) < 1e-9
    assert abs(m["pre_capture_G_max_deg"] - 125.0) < 1e-9
    assert abs(m["pre_capture_C_sync"] - 0.9) < 1e-9


def test_pre_capture_empty_when_not_captured():
    steps = [_step(0, 0.5, 0.5, 90.0, 0.5)]
    m = pre_capture_structure_metrics(steps, capture_step=None, captured=False, k=10)
    assert m["pre_capture_D_ang"] == ""
