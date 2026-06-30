"""Tests for experiment runner helpers."""

from __future__ import annotations

from experiments.runner_common import format_trial_progress_line


def test_format_trial_progress_line_captured():
    summary = {
        "captured": True,
        "num_steps": 751,
        "capture_D_ang": 0.546,
        "capture_C_cov": 0.465,
        "capture_G_max_deg": 74.9,
        "pre_capture_5_D_ang": 0.563,
        "pre_capture_10_D_ang": 0.571,
    }
    line = format_trial_progress_line(summary, prefix="fcem/free trial 0")
    assert "captured=True" in line
    assert "steps=751" in line
    assert "capD=0.546" in line
    assert "capC=0.465" in line
    assert "capG=74.9" in line
    assert "p5D=0.563" in line
    assert "p10D=0.571" in line
    assert line.endswith("OK")


def test_format_trial_progress_line_failed():
    summary = {
        "captured": False,
        "num_steps": 1200,
        "failure_reason": "timeout",
        "final_D_ang": 0.234,
        "final_C_cov": 0.112,
        "final_G_max_deg": 95.3,
    }
    line = format_trial_progress_line(summary, prefix="fcem/free trial 0")
    assert "captured=False" in line
    assert "finD=0.234" in line
    assert "finC=0.112" in line
    assert "finG=95.3" in line
    assert line.endswith("timeout")
