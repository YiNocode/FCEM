"""Vectorized training environment for OPEN MARL."""

from __future__ import annotations

from typing import Any

import numpy as np

from baselines.open_marl.observation import ObservationBuilder, OpenMarlConfig
from baselines.open_marl.reward import RewardConfig, compute_team_reward
from common.dynamics import integrate_point_mass
from common.evader_policy import evader_kwargs_from_config, evader_step
from common.obstacles import Obstacle, any_pursuer_obstacle_collision
from envs.sim2d import init_from_scenario
from experiments.config_loader import deep_merge, load_config, obstacles_from_scenario


class OpenMarlVecEnv:
    """Batch of independent 2D pursuit-evasion episodes for MAPPO training."""

    def __init__(
        self,
        n_envs: int,
        base_config: dict[str, Any],
        open_cfg: OpenMarlConfig,
        reward_cfg: RewardConfig,
        scenario_names: list[str],
        seed: int = 0,
    ) -> None:
        self.n_envs = n_envs
        self.base_config = base_config
        self.open_cfg = open_cfg
        self.reward_cfg = reward_cfg
        self.scenario_names = scenario_names
        self.rng = np.random.default_rng(seed)

        self.configs: list[dict[str, Any]] = []
        self.obstacles: list[list[Obstacle]] = []
        self.bounds: list[tuple[float, float, float, float]] = []
        self.obs_builders: list[ObservationBuilder] = []
        self._reset_all()

        n = open_cfg.n_pursuers
        self.pursuers = np.zeros((n_envs, n, 2), dtype=np.float32)
        self.pursuer_v = np.zeros((n_envs, n, 2), dtype=np.float32)
        self.evader = np.zeros((n_envs, 2), dtype=np.float32)
        self.evader_v = np.zeros((n_envs, 2), dtype=np.float32)
        self.steps = np.zeros(n_envs, dtype=np.int32)
        self.done = np.zeros(n_envs, dtype=bool)
        self.prev_actions = np.zeros((n_envs, n, 2), dtype=np.float32)
        self.epn_targets = np.zeros((n_envs, open_cfg.epn_target_dim), dtype=np.float32)

    def _sample_scenario(self) -> tuple[dict[str, Any], list[Obstacle], tuple[float, float, float, float]]:
        name = self.scenario_names[int(self.rng.integers(0, len(self.scenario_names)))]
        cfg = deep_merge(self.base_config, load_config(name))
        obstacles = obstacles_from_scenario(cfg["scenario"])
        w = cfg["world"]
        bounds = (w["xmin"], w["xmax"], w["ymin"], w["ymax"])
        return cfg, obstacles, bounds

    def _reset_all(self) -> None:
        self.configs = []
        self.obstacles = []
        self.bounds = []
        self.obs_builders = []
        for _ in range(self.n_envs):
            cfg, obs, bounds = self._sample_scenario()
            self.configs.append(cfg)
            self.obstacles.append(obs)
            self.bounds.append(bounds)
            builder = ObservationBuilder(self.open_cfg)
            w = cfg["world"]
            scale = max(w["xmax"] - w["xmin"], w["ymax"] - w["ymin"])
            builder.reset(arena_scale=scale)
            self.obs_builders.append(builder)

    def _reset_env(self, i: int) -> None:
        cfg, obs, bounds = self._sample_scenario()
        self.configs[i] = cfg
        self.obstacles[i] = obs
        self.bounds[i] = bounds
        rng = np.random.default_rng(int(self.rng.integers(0, 2**31 - 1)))
        pursuers, pursuer_v, evader, evader_v = init_from_scenario(cfg, bounds, rng)
        self.pursuers[i] = pursuers
        self.pursuer_v[i] = pursuer_v
        self.evader[i] = evader
        self.evader_v[i] = evader_v
        self.steps[i] = 0
        self.done[i] = False
        self.prev_actions[i] = 0.0
        builder = ObservationBuilder(self.open_cfg)
        w = cfg["world"]
        scale = max(w["xmax"] - w["xmin"], w["ymax"] - w["ymin"])
        builder.reset(arena_scale=scale)
        self.obs_builders[i] = builder

    def reset(self) -> tuple[np.ndarray, np.ndarray]:
        for i in range(self.n_envs):
            self._reset_env(i)
        return self._get_obs(), self._get_history()

    def _get_obs(self) -> np.ndarray:
        n = self.open_cfg.n_pursuers
        obs = np.zeros((self.n_envs, n, self.open_cfg.obs_dim), dtype=np.float32)
        for i in range(self.n_envs):
            if self.done[i]:
                continue
            o, _ = self.obs_builders[i].build_observations(
                self.pursuers[i],
                self.pursuer_v[i],
                self.evader[i],
                self.evader_v[i],
                self.obstacles[i],
                int(self.steps[i]),
            )
            obs[i] = o
        return obs

    def _get_history(self) -> np.ndarray:
        h = self.open_cfg.history_step
        d = self.open_cfg.epn_input_dim
        out = np.zeros((self.n_envs, h, d), dtype=np.float32)
        for i in range(self.n_envs):
            out[i] = self.obs_builders[i].history_tensor()
        return out

    def _compute_epn_targets(self) -> np.ndarray:
        """Supervised labels: future evader positions relative to centroid."""
        k = self.open_cfg.future_prediction_step
        dt = float(self.base_config["dt"])
        targets = np.zeros((self.n_envs, self.open_cfg.epn_target_dim), dtype=np.float32)
        for i in range(self.n_envs):
            if self.done[i]:
                continue
            cfg = self.configs[i]
            ev = self.evader[i].copy()
            ev_v = self.evader_v[i].copy()
            future = []
            centroid = np.mean(self.pursuers[i], axis=0)
            for _ in range(k):
                ev = ev + ev_v * dt
                future.append(ev.copy())
            scale = self.obs_builders[i]._arena_scale
            for t, pos in enumerate(future):
                rel = (pos - centroid) / max(scale, 1e-6)
                targets[i, 2 * t : 2 * t + 2] = rel.astype(np.float32)
        return targets

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        """
        actions: (n_envs, n_agents, 2) acceleration commands
        returns obs, history, rewards, dones, info
        """
        n = self.open_cfg.n_pursuers
        rewards = np.zeros(self.n_envs, dtype=np.float32)
        epn_targets = self._compute_epn_targets()
        captured_flags = np.zeros(self.n_envs, dtype=bool)
        episode_done = np.zeros(self.n_envs, dtype=bool)

        for i in range(self.n_envs):
            if self.done[i]:
                continue
            cfg = self.configs[i]
            dt = cfg["dt"]
            bounds = self.bounds[i]
            fcem_cfg = cfg["fcem"]

            centroid = np.mean(self.pursuers[i], axis=0)
            self.evader[i], self.evader_v[i] = evader_step(
                self.evader[i],
                self.evader_v[i],
                centroid,
                self.obstacles[i],
                bounds,
                dt,
                cfg["evader_vmax"],
                cfg["evader_amax"],
                pursuers=self.pursuers[i],
                **evader_kwargs_from_config(cfg),
            )

            for j in range(n):
                p_new, v_new = integrate_point_mass(
                    self.pursuers[i, j],
                    self.pursuer_v[i, j],
                    actions[i, j],
                    dt,
                    cfg["pursuer_vmax"],
                    cfg["pursuer_amax"],
                )
                self.pursuers[i, j] = p_new
                self.pursuer_v[i, j] = v_new
                self.pursuers[i, j, 0] = np.clip(self.pursuers[i, j, 0], bounds[0] + 0.15, bounds[1] - 0.15)
                self.pursuers[i, j, 1] = np.clip(self.pursuers[i, j, 1], bounds[2] + 0.15, bounds[3] - 0.15)

            r, _ = compute_team_reward(
                self.pursuers[i],
                self.evader[i],
                self.obstacles[i],
                actions[i],
                self.prev_actions[i],
                fcem_cfg["capture_radius"],
                self.reward_cfg,
            )
            rewards[i] = r
            self.prev_actions[i] = actions[i].copy()
            self.steps[i] += 1

            dists = np.linalg.norm(self.pursuers[i] - self.evader[i][None, :], axis=1)
            captured = bool(np.any(dists <= fcem_cfg["capture_radius"]))
            body_r = float(cfg.get("pursuer_collision_radius", 0.25))
            hit, _ = any_pursuer_obstacle_collision(self.pursuers[i], self.obstacles[i], body_r)
            timeout = self.steps[i] >= cfg["max_steps"]

            if captured or hit or timeout:
                self.done[i] = True
                episode_done[i] = True
                captured_flags[i] = captured

        for i in range(self.n_envs):
            if self.done[i]:
                self._reset_env(i)

        obs = self._get_obs()
        history = self._get_history()
        return obs, history, rewards, episode_done.copy(), {
            "epn_targets": epn_targets,
            "captured": captured_flags,
            "episode_done": episode_done,
        }
