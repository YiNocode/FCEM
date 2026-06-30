"""Tests for OPEN MARL baseline (no trained checkpoint required)."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.open_marl.aeg import AEGConfig, AdaptiveEnvironmentGenerator, TaskSeed, sample_center_init
from baselines.open_marl.mappo import RolloutBuffer, fill_epn_targets
from baselines.open_marl.observation import (
    MASK_VALUE,
    ObservationBuilder,
    OpenMarlConfig,
    has_line_of_sight,
    split_observation,
    team_evader_detected,
)
from baselines.open_marl.reward import RewardConfig, compute_team_reward
from common.capture import check_capture
from common.obstacles import Obstacle
from metrics.structure import structural_metrics_from_positions

torch = pytest.importorskip("torch")

from baselines.open_marl.env import OpenMarlVecEnv
from baselines.open_marl.networks import OpenMARLPolicy
from baselines.open_marl.reward import RewardConfig as RC


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


def test_split_observation() -> None:
    cfg = OpenMarlConfig()
    obs = np.random.randn(cfg.obs_dim).astype(np.float32)
    o_self, o_other, o_ob = split_observation(obs, cfg)
    assert o_self.shape == (cfg.self_obs_dim,)
    assert o_other.shape == (cfg.other_obs_dim,)
    assert o_ob.shape == (cfg.obstacle_obs_dim,)


def test_network_forward_shapes() -> None:
    cfg = OpenMarlConfig()
    policy = OpenMARLPolicy(cfg)
    b = 4
    obs_all = torch.randn(b, cfg.n_pursuers, cfg.obs_dim)
    hist = torch.randn(b, cfg.history_step, cfg.epn_input_dim)
    actions, log_prob, entropy, pred = policy.actor_forward_batch(obs_all, hist, deterministic=True)
    assert actions.shape == (b, cfg.n_pursuers, 2)
    assert log_prob.shape == (b, cfg.n_pursuers)
    assert pred.shape == (b, cfg.epn_target_dim)
    value = policy.critic_forward(obs_all, hist)
    assert value.shape == (b,)


def test_checkpoint_roundtrip(tmp_path: Path) -> None:
    cfg = OpenMarlConfig()
    policy = OpenMARLPolicy(cfg)
    path = tmp_path / "test.pt"
    policy.save_checkpoint(str(path))
    loaded, meta = OpenMARLPolicy.load_checkpoint(str(path))
    assert meta["version"] == 2
    assert meta["obs_dim"] == cfg.obs_dim
    assert loaded.cfg.n_pursuers == cfg.n_pursuers


def test_legacy_checkpoint_rejected(tmp_path: Path) -> None:
    cfg = OpenMarlConfig()
    policy = OpenMARLPolicy(cfg)
    path = tmp_path / "legacy.pt"
    torch.save(
        {
            "state_dict": policy.state_dict(),
            "meta": {"cfg": {"n_pursuers": 3}, "obs_dim": cfg.obs_dim, "version": 1},
        },
        str(path),
    )
    with pytest.raises(ValueError, match="legacy"):
        OpenMARLPolicy.load_checkpoint(str(path))


def test_team_reward_components() -> None:
    pursuers = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    evader = np.array([0.0, 0.0])
    metrics = structural_metrics_from_positions(evader, pursuers)
    r, comp = compute_team_reward(
        pursuers,
        evader,
        [],
        np.zeros((3, 2)),
        None,
        None,
        capture_radius=2.0,
        metrics=metrics,
        g_max_allowed=math.radians(140.0),
        strict_captured=True,
        loose_captured=True,
        cfg=RewardConfig(),
    )
    assert comp["capture"] == 6.0
    assert isinstance(r, float)
    assert abs(r) <= 10.0


def test_loose_capture_bonus() -> None:
    pursuers = np.array([[1.0, 0.0], [5.0, 0.0], [5.0, 1.0]])
    evader = np.array([0.0, 0.0])
    metrics = structural_metrics_from_positions(evader, pursuers)
    _, comp = compute_team_reward(
        pursuers,
        evader,
        [],
        np.zeros((3, 2)),
        None,
        None,
        capture_radius=1.5,
        metrics=metrics,
        g_max_allowed=math.radians(140.0),
        strict_captured=False,
        loose_captured=True,
        cfg=RewardConfig(capture_mode="loose"),
    )
    assert comp["capture"] == 3.0


def test_reward_not_catastrophic() -> None:
    pursuers = np.array([[20.0, 0.0], [21.0, 1.0], [19.0, -1.0]])
    evader = np.array([0.0, 0.0])
    metrics = structural_metrics_from_positions(evader, pursuers)
    total = 0.0
    for _ in range(100):
        r, _ = compute_team_reward(
            pursuers,
            evader,
            [],
            np.zeros((3, 2)),
            None,
            25.0,
            capture_radius=1.8,
            metrics=metrics,
            g_max_allowed=math.radians(140.0),
            strict_captured=False,
            loose_captured=False,
            cfg=RewardConfig(),
        )
        total += r
    assert total > -1000.0


def test_smoothness_reward_stage2() -> None:
    cfg = RewardConfig(use_smoothness=True, smoothness_coef=4.0)
    pursuers = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    evader = np.array([5.0, 0.0])
    metrics = structural_metrics_from_positions(evader, pursuers)
    _, comp = compute_team_reward(
        pursuers,
        evader,
        [],
        np.zeros((3, 2)),
        np.zeros((3, 2)),
        None,
        capture_radius=0.5,
        metrics=metrics,
        g_max_allowed=math.radians(140.0),
        strict_captured=False,
        loose_captured=False,
        cfg=cfg,
    )
    assert comp["smoothness"] > 0


def test_fill_epn_targets_from_buffer() -> None:
    cfg = OpenMarlConfig()
    n_envs = 2
    buf = RolloutBuffer(8, n_envs, 3, cfg.obs_dim, cfg.history_step, cfg.epn_input_dim, cfg.epn_target_dim, 2)
    for t in range(6):
        e = t % n_envs
        buf.add(
            np.zeros((3, cfg.obs_dim), dtype=np.float32),
            np.zeros((cfg.history_step, cfg.epn_input_dim), dtype=np.float32),
            np.zeros((3, 2), dtype=np.float32),
            np.zeros(3, dtype=np.float32),
            0.0,
            False,
            0.0,
            np.array([0.0, 0.0], dtype=np.float32),
            np.array([float(t), float(e)], dtype=np.float32),
            40.0,
        )
    fill_epn_targets(buf, cfg)
    assert buf.epn_targets[0, 0] == pytest.approx(2 / 40.0, abs=1e-4)
    assert buf.epn_targets[0, 1] == pytest.approx(0 / 40.0, abs=1e-4)


def test_aeg_archive_update() -> None:
    rng = np.random.default_rng(0)
    aeg = AdaptiveEnvironmentGenerator(AEGConfig(), rng)
    seed = TaskSeed(
        scenario_name="free",
        pursuers=np.zeros((3, 2)),
        evader=np.zeros(2),
        pursuer_v=np.zeros((3, 2)),
        evader_v=np.zeros(2),
    )
    aeg.update_archive(seed, 0.7)
    assert len(aeg.archive) == 1
    aeg.update_archive(seed, 0.2)
    assert len(aeg.archive) == 1


def test_strict_capture_condition() -> None:
    evader = np.array([0.0, 0.0])
    pursuers = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [-0.9, 0.1],
        ]
    )
    metrics = structural_metrics_from_positions(evader, pursuers)
    loose = bool(np.any(np.linalg.norm(pursuers - evader, axis=1) <= 1.8))
    strict = check_capture(pursuers, evader, 1.8, metrics["G_max"], math.radians(140.0))
    assert loose
    assert isinstance(strict, bool)


def test_center_init_distances() -> None:
    bounds = (0.0, 40.0, 0.0, 40.0)
    rng = np.random.default_rng(1)
    pursuers, _, evader, _ = sample_center_init(bounds, rng)
    dists = np.linalg.norm(pursuers - evader[None, :], axis=1)
    assert np.all(dists >= 4.5)
    assert np.all(dists <= 10.0)


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


def test_vec_env_step() -> None:
    from experiments.config_loader import build_experiment_base_config, load_experiment_config

    exp_cfg = load_experiment_config(ROOT / "config/experiments/setup.yaml")
    base = build_experiment_base_config(exp_cfg, timestamped=False, write_manifest=False)
    base["max_steps"] = 10
    cfg = OpenMarlConfig()
    reward = RC()
    env = OpenMarlVecEnv(2, base, cfg, reward, ["free"], seed=0, global_step=0, total_steps=1000)
    obs, hist = env.reset()
    assert obs.shape == (2, 3, cfg.obs_dim)
    assert hist.shape == (2, cfg.history_step, cfg.epn_input_dim)
    actions = np.zeros((2, 3, 2), dtype=np.float32)
    obs2, hist2, rewards, dones, info = env.step(actions)
    assert obs2.shape == obs.shape
    assert "centroids" in info
    assert "g_max" in info
