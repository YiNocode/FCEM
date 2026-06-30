"""Tests for contraction gates (formation expansion latch)."""

from __future__ import annotations

import math

from metrics.structure import contraction_allowed


def test_allows_contraction_when_expanded_regardless_of_g_max() -> None:
    metrics = {"G_max": math.radians(270.0), "C_cov": 0.40}
    fcem_cfg = {"C_expand_min": 0.25}
    config = {"G_max_allowed_deg": 140.0}

    allowed, parts, latched = contraction_allowed(
        metrics, fcem_cfg, config, R=10.0, formation_expanded_latched=False
    )

    assert allowed is True
    assert latched is True
    assert parts["g_capture_ok"] is False


def test_contraction_blocked_when_not_expanded() -> None:
    metrics = {"G_max": math.radians(100.0), "C_cov": 0.10}
    fcem_cfg = {"C_expand_min": 0.25}
    config = {"G_max_allowed_deg": 140.0}

    allowed, parts, latched = contraction_allowed(
        metrics, fcem_cfg, config, R=10.0, formation_expanded_latched=False
    )

    assert allowed is False
    assert latched is False
    assert parts["block_reason"] == "not_expanded"


def test_formation_expansion_latches_after_threshold() -> None:
    metrics = {"G_max": math.radians(200.0), "C_cov": 0.10}
    fcem_cfg = {"C_expand_min": 0.25}
    config = {"G_max_allowed_deg": 140.0}

    _, _, latched = contraction_allowed(
        metrics, fcem_cfg, config, R=10.0, formation_expanded_latched=True
    )
    assert latched is True

    allowed, parts, _ = contraction_allowed(
        metrics, fcem_cfg, config, R=2.0, formation_expanded_latched=True
    )
    assert allowed is True
    assert parts["formation_expanded"] is True
