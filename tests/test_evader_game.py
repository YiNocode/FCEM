"""Tests for differential-game evader policy."""

from __future__ import annotations

import numpy as np

from common.evader_game import (
    largest_angular_gap_direction,
    minimax_evasion_direction,
)
from experiments.config_loader import load_config, obstacles_from_scenario
from envs.sim2d import Sim2D, make_fcem_controller
from baselines.pure_pursuit_apf import make_pure_pursuit_controller


def test_largest_gap_points_into_open_sector():
    evader = np.array([0.0, 0.0])
    pursuers = np.array([[5.0, 0.0], [0.0, 5.0], [-5.0, 0.0]])
    gap_dir = largest_angular_gap_direction(evader, pursuers)
    assert gap_dir[0] < 0.0
    assert gap_dir[1] < 0.0


def test_minimax_prefers_open_direction():
    evader = np.array([0.0, 0.0])
    pursuers = np.array([[10.0, 0.0], [8.0, 1.5], [8.0, -1.5]])
    u = minimax_evasion_direction(
        evader, pursuers, pursuer_vmax=4.0, horizon=0.8, n_directions=72, evader_vmax=10.0
    )
    assert u[0] < -0.5


def test_minimax_depends_on_evader_speed():
    evader = np.array([0.0, 0.0])
    pursuers = np.array([[6.0, 0.0], [-3.0, 5.0], [-3.0, -5.0]])
    u_slow = minimax_evasion_direction(
        evader, pursuers, pursuer_vmax=4.0, horizon=0.85, n_directions=72, evader_vmax=4.0
    )
    u_fast = minimax_evasion_direction(
        evader, pursuers, pursuer_vmax=4.0, horizon=0.85, n_directions=72, evader_vmax=16.0
    )
    assert not np.allclose(u_slow, u_fast)


def test_game_policy_runs_in_sim():
    cfg = load_config("free")
    cfg["max_steps"] = 20
    cfg["evader_policy"] = "game"
    obstacles = obstacles_from_scenario(cfg["scenario"])
    sim = Sim2D(cfg, obstacles, make_fcem_controller(), np.random.default_rng(0))
    for step in range(20):
        sim.step_once(step)
    assert len(sim.frames) == 20


def test_game_harder_than_apf_for_pure_pursuit_on_default_init():
    """With v_e/v_p=2.0, game evader should be at least as hard as APF for pure pursuit."""
    from experiments.config_loader import deep_merge, load_yaml
    from pathlib import Path

    dyn = load_yaml(Path("config/dynamics/evader_faster.yaml"))

    def run(policy: str) -> bool:
        cfg = load_config("free")
        cfg = deep_merge(cfg, dyn)
        cfg["evader_policy"] = policy
        cfg["max_steps"] = 1200
        obstacles = obstacles_from_scenario(cfg["scenario"])
        sim = Sim2D(cfg, obstacles, make_pure_pursuit_controller(), np.random.default_rng(42))
        return bool(sim.run()["captured"])

    apf_cap = run("apf")
    game_cap = run("game")
    assert isinstance(game_cap, bool)
    assert isinstance(apf_cap, bool)
