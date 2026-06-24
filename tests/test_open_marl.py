"""Tests for OPEN MARL baseline (no trained checkpoint required)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.open_marl.observation import (
    MASK_VALUE,
    ObservationBuilder,
    OpenMarlConfig,
    has_line_of_sight,
    team_evader_detected,
)
from baselines.open_marl.reward import RewardConfig, compute_team_reward
from common.obstacles import Obstacle

torch = pytest.importorskip("torch")

from baselines.open_marl.networks import OpenMARLPolicy


def test_los_blocked_by_obstacle() -> None:
    p = np.array([0.0, 0.0])
    e = np.array([10.0, 0.0])
    obs = [Obstacle(np.array([5.0, 0.0]), 1.0)]
    assert not has_line_of_sight(p, e, obs)
    assert not team_evader_detected(np.array([p]), e, obs)


def test_los_clear() -> None:
    p = np.array([0.0, 0.0])
    e = np.array([10.0, 0.0])
    obs = [Obstacle(np.array([5.0, 5.0]), 1.0)]
    assert has_line_of_sight(p, e, obs)
    assert team_evader_detected(np.array([p]), e, obs)


def test_observation_mask_when_occluded() -> None:
    cfg = OpenMarlConfig()
    builder = ObservationBuilder(cfg)
    pursuers = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]])
    pursuer_v = np.zeros((3, 2))
    evader = np.array([10.0, 0.0])
    evader_v = np.array([0.5, 0.0])
    obstacles = [Obstacle(np.array([5.0, 0.0]), 1.5)]
    obs, detected = builder.build_observations(
        pursuers, pursuer_v, evader, evader_v, obstacles, step=0
    )
    assert not detected
    assert obs.shape == (3, cfg.obs_dim)
    assert np.all(obs[:, 2:4] == MASK_VALUE)


def test_network_forward_shapes() -> None:
    cfg = OpenMarlConfig()
    policy = OpenMARLPolicy(cfg)
    b = 4
    obs = torch.randn(b, cfg.obs_dim)
    hist = torch.randn(b, cfg.history_step, cfg.epn_input_dim)
    action, log_prob, entropy, _ = policy.actor_forward(obs, hist, deterministic=True)
    assert action.shape == (b, 2)
    assert log_prob.shape == (b,)
    obs_all = torch.randn(b, cfg.n_pursuers, cfg.obs_dim)
    value = policy.critic_forward(obs_all, hist)
    assert value.shape == (b,)


def test_checkpoint_roundtrip(tmp_path: Path) -> None:
    cfg = OpenMarlConfig()
    policy = OpenMARLPolicy(cfg)
    path = tmp_path / "test.pt"
    policy.save_checkpoint(str(path))
    loaded, meta = OpenMARLPolicy.load_checkpoint(str(path))
    assert meta["obs_dim"] == cfg.obs_dim
    assert loaded.cfg.n_pursuers == cfg.n_pursuers


def test_team_reward_components() -> None:
    pursuers = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    evader = np.array([0.0, 0.0])
    r, comp = compute_team_reward(
        pursuers, evader, [], np.zeros((3, 2)), None, capture_radius=2.0, cfg=RewardConfig()
    )
    assert "distance" in comp
    assert isinstance(r, float)


def test_open_marl_controller_requires_checkpoint() -> None:
    from baselines.open_marl import make_open_marl_controller
    from experiments.config_loader import load_config

    cfg = load_config("free")
    cfg["max_steps"] = 1
    cfg["baselines"]["open_marl"]["checkpoint_path"] = "checkpoints/open_marl/nonexistent.pt"
    controller = make_open_marl_controller()
    with pytest.raises(FileNotFoundError, match="checkpoint not found"):
        controller(
            step=0,
            evader=np.zeros(2),
            evader_v=np.zeros(2),
            pursuers=np.zeros((3, 2)),
            pursuer_v=np.zeros((3, 2)),
            obstacles=[],
            bounds=(0, 40, 0, 40),
            R=12.0,
            prev_slots=None,
            prev_assignment=None,
            config=cfg,
        )
