"""Smoke test for the fixed_ring_failure scenario."""

from __future__ import annotations

import math

import numpy as np

from baselines.fixed_ring_apf import make_fixed_ring_controller
from envs.sim2d import Sim2D, make_fcem_controller
from experiments.config_loader import load_config, obstacles_from_scenario


def test_fixed_ring_fails_fcem_succeeds():
    cfg = load_config("fixed_ring_failure")
    cfg["seed"] = 42
    obstacles = obstacles_from_scenario(cfg["scenario"])

    sim_fr = Sim2D(cfg, obstacles, make_fixed_ring_controller(), np.random.default_rng(42))
    result_fr = sim_fr.run()
    assert not result_fr["captured"]
    g_fr = result_fr["frames"][-1]["metrics"]["G_max"]
    assert math.degrees(g_fr) > 200.0

    sim_fcem = Sim2D(cfg, obstacles, make_fcem_controller(), np.random.default_rng(42))
    result_fcem = sim_fcem.run()
    assert result_fcem["captured"]
    g_fcem = result_fcem["frames"][-1]["metrics"]["G_max"]
    assert math.degrees(g_fcem) <= 140.0
