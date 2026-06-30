"""Vectorized training environment for OPEN MARL."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from baselines.open_marl.aeg import (
    AdaptiveEnvironmentGenerator,
    AEGConfig,
    CurriculumConfig,
    CurriculumPhase,
    TaskSeed,
    sample_center_init,
)
from baselines.open_marl.observation import ObservationBuilder, OpenMarlConfig
from baselines.open_marl.reward import RewardConfig, compute_team_reward, evaluate_capture
from common.dynamics import integrate_point_mass
from common.evader_policy import evader_kwargs_from_config, evader_step
from common.obstacles import Obstacle, any_pursuer_obstacle_collision
from envs.sim2d import init_from_scenario
from experiments.config_loader import deep_merge, load_config, obstacles_from_scenario
from metrics.structure import structural_metrics_from_positions


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
        aeg_cfg: AEGConfig | None = None,
        curriculum_cfg: CurriculumConfig | None = None,
        global_step: int = 0,
        total_steps: int = 1,
    ) -> None:
        self.n_envs = n_envs
        self.base_config = base_config
        self.open_cfg = open_cfg
        self.reward_cfg = reward_cfg
        self.all_scenario_names = list(scenario_names)
        self.rng = np.random.default_rng(seed)
        self.aeg_cfg = aeg_cfg or AEGConfig()
        self.curriculum_cfg = curriculum_cfg or CurriculumConfig.from_dict({})
        self.aeg = AdaptiveEnvironmentGenerator(self.aeg_cfg, self.rng)
        self.global_step = global_step
        self.total_steps = total_steps

        self.configs: list[dict[str, Any]] = []
        self.obstacles: list[list[Obstacle]] = []
        self.bounds: list[tuple[float, float, float, float]] = []
        self.obs_builders: list[ObservationBuilder] = []
        n = open_cfg.n_pursuers
        self.pursuers = np.zeros((n_envs, n, 2), dtype=np.float32)
        self.pursuer_v = np.zeros((n_envs, n, 2), dtype=np.float32)
        self.evader = np.zeros((n_envs, 2), dtype=np.float32)
        self.evader_v = np.zeros((n_envs, 2), dtype=np.float32)
        self.steps = np.zeros(n_envs, dtype=np.int32)
        self.done = np.zeros(n_envs, dtype=bool)
        self.prev_actions = np.zeros((n_envs, n, 2), dtype=np.float32)
        self.prev_min_dist = np.full(n_envs, np.inf, dtype=np.float32)
        self.scenario_names: list[str] = [""] * n_envs
        self._reset_all()

    def set_training_progress(self, global_step: int, total_steps: int) -> None:
        self.global_step = global_step
        self.total_steps = total_steps

    def _current_phase(self) -> CurriculumPhase:
        return self.curriculum_cfg.phase_at(self.global_step, self.total_steps)

    def _phase_scenarios(self) -> list[str]:
        phase = self._current_phase()
        allowed = [s for s in phase.scenarios if s in self.all_scenario_names]
        return allowed or list(self.all_scenario_names)

    def _phase_reward_cfg(self) -> RewardConfig:
        phase = self._current_phase()
        overrides = {"capture_mode": phase.capture_mode, **phase.reward_overrides}
        return self.reward_cfg.with_overrides(overrides)

    def _episode_limit(self, cfg: dict[str, Any]) -> int:
        phase = self._current_phase()
        if phase.episode_max_steps is not None:
            return int(phase.episode_max_steps)
        return int(cfg["max_steps"])

    def _apply_phase_dynamics(self, cfg: dict[str, Any]) -> dict[str, Any]:
        phase = self._current_phase()
        out = dict(cfg)
        for k, v in phase.dynamics.items():
            out[k] = v
        return out

    def _sample_scenario(self) -> tuple[str, dict[str, Any], list[Obstacle], tuple[float, float, float, float]]:
        names = self._phase_scenarios()
        name = names[int(self.rng.integers(0, len(names)))]
        cfg = deep_merge(self.base_config, load_config(name))
        cfg = self._apply_phase_dynamics(cfg)
        obstacles = obstacles_from_scenario(cfg["scenario"])
        w = cfg["world"]
        bounds = (w["xmin"], w["xmax"], w["ymin"], w["ymax"])
        return name, cfg, obstacles, bounds

    def _reset_all(self) -> None:
        self.configs = []
        self.obstacles = []
        self.bounds = []
        self.obs_builders = []
        self.scenario_names = []
        for i in range(self.n_envs):
            self._reset_env(i, use_aeg=False)

    def _reset_env(
        self,
        i: int,
        use_aeg: bool | None = None,
        task_override: tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None,
    ) -> None:
        phase = self._current_phase()
        if use_aeg is None:
            use_aeg = phase.use_aeg

        if task_override is not None:
            scen_name, pursuers, pursuer_v, evader, evader_v = task_override
            cfg = deep_merge(self.base_config, load_config(scen_name))
            cfg = self._apply_phase_dynamics(cfg)
            obstacles = obstacles_from_scenario(cfg["scenario"])
            w = cfg["world"]
            bounds = (w["xmin"], w["xmax"], w["ymin"], w["ymax"])
        else:
            scen_name, cfg, obstacles, bounds = self._sample_scenario()
            rng = np.random.default_rng(int(self.rng.integers(0, 2**31 - 1)))

            aeg_task = None
            if use_aeg:
                aeg_task = self.aeg.sample_task(self._phase_scenarios(), bounds, global_explore=False)

            if aeg_task is not None:
                scen_name, pursuers, pursuer_v, evader, evader_v = aeg_task
                cfg = deep_merge(self.base_config, load_config(scen_name))
                cfg = self._apply_phase_dynamics(cfg)
                obstacles = obstacles_from_scenario(cfg["scenario"])
            elif phase.center_init:
                pursuers, pursuer_v, evader, evader_v = sample_center_init(bounds, rng, self.open_cfg.n_pursuers)
            else:
                pursuers, pursuer_v, evader, evader_v = init_from_scenario(cfg, bounds, rng)

        builder = ObservationBuilder(self.open_cfg)
        w = cfg["world"]
        scale = max(w["xmax"] - w["xmin"], w["ymax"] - w["ymin"])
        builder.reset(arena_scale=scale)

        if i < len(self.configs):
            self.configs[i] = cfg
            self.obstacles[i] = obstacles
            self.bounds[i] = bounds
            self.scenario_names[i] = scen_name
            self.pursuers[i] = pursuers
            self.pursuer_v[i] = pursuer_v
            self.evader[i] = evader
            self.evader_v[i] = evader_v
            self.steps[i] = 0
            self.done[i] = False
            self.prev_actions[i] = 0.0
            self.prev_min_dist[i] = np.inf
            self.obs_builders[i] = builder
        else:
            self.configs.append(cfg)
            self.obstacles.append(obstacles)
            self.bounds.append(bounds)
            self.scenario_names.append(scen_name)
            self.pursuers[i] = pursuers
            self.pursuer_v[i] = pursuer_v
            self.evader[i] = evader
            self.evader_v[i] = evader_v
            self.steps[i] = 0
            self.done[i] = False
            self.prev_actions[i] = 0.0
            self.prev_min_dist[i] = np.inf
            self.obs_builders.append(builder)

    def reset(self) -> tuple[np.ndarray, np.ndarray]:
        for i in range(self.n_envs):
            self._reset_env(i)
        return self._get_obs(), self._get_history()

    def _get_obs(self) -> np.ndarray:
        n = self.open_cfg.n_pursuers
        obs = np.zeros((self.n_envs, n, self.open_cfg.obs_dim), dtype=np.float32)
        for i in range(self.n_envs):
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

    def _capture_state(self, i: int) -> tuple[bool, bool, dict[str, float], float]:
        cfg = self.configs[i]
        fcem_cfg = cfg["fcem"]
        metrics = structural_metrics_from_positions(self.evader[i], self.pursuers[i])
        g_max_allowed = math.radians(float(cfg.get("G_max_allowed_deg", 140.0)))
        reward_cfg = self._phase_reward_cfg()
        episode_success, strict_success = evaluate_capture(
            self.pursuers[i],
            self.evader[i],
            fcem_cfg["capture_radius"],
            metrics["G_max"],
            g_max_allowed,
            reward_cfg.capture_mode,
        )
        return episode_success, strict_success, metrics, g_max_allowed

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        n = self.open_cfg.n_pursuers
        rewards = np.zeros(self.n_envs, dtype=np.float32)
        centroids = np.zeros((self.n_envs, 2), dtype=np.float32)
        evader_positions = np.zeros((self.n_envs, 2), dtype=np.float32)
        arena_scales = np.zeros(self.n_envs, dtype=np.float32)
        captured_flags = np.zeros(self.n_envs, dtype=bool)
        strict_flags = np.zeros(self.n_envs, dtype=bool)
        loose_flags = np.zeros(self.n_envs, dtype=bool)
        episode_done = np.zeros(self.n_envs, dtype=bool)
        g_max_vals = np.zeros(self.n_envs, dtype=np.float32)
        reward_cfg = self._phase_reward_cfg()

        for i in range(self.n_envs):
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

            metrics = structural_metrics_from_positions(self.evader[i], self.pursuers[i])
            episode_success, strict_success, metrics, g_max_allowed = self._capture_state(i)
            dists = np.linalg.norm(self.pursuers[i] - self.evader[i][None, :], axis=1)
            min_dist = float(np.min(dists))
            loose = bool(np.any(dists <= fcem_cfg["capture_radius"]))
            g_max_vals[i] = metrics["G_max"]
            prev_min = None if not np.isfinite(self.prev_min_dist[i]) else float(self.prev_min_dist[i])

            r, _ = compute_team_reward(
                self.pursuers[i],
                self.evader[i],
                self.obstacles[i],
                actions[i],
                self.prev_actions[i],
                prev_min,
                fcem_cfg["capture_radius"],
                metrics,
                g_max_allowed,
                strict_success,
                loose,
                reward_cfg,
            )
            rewards[i] = r
            self.prev_actions[i] = actions[i].copy()
            self.prev_min_dist[i] = min_dist
            self.steps[i] += 1

            centroids[i] = centroid
            evader_positions[i] = self.evader[i].copy()
            arena_scales[i] = self.obs_builders[i]._arena_scale

            body_r = float(cfg.get("pursuer_collision_radius", 0.25))
            hit, _ = any_pursuer_obstacle_collision(self.pursuers[i], self.obstacles[i], body_r)
            timeout = self.steps[i] >= self._episode_limit(cfg)

            if episode_success or hit or timeout:
                self.done[i] = True
                episode_done[i] = True
                captured_flags[i] = episode_success
                strict_flags[i] = strict_success
                loose_flags[i] = loose

        for i in range(self.n_envs):
            if self.done[i]:
                self._reset_env(i)

        obs = self._get_obs()
        history = self._get_history()
        return obs, history, rewards, episode_done.copy(), {
            "centroids": centroids,
            "evader_positions": evader_positions,
            "arena_scales": arena_scales,
            "captured": captured_flags,
            "strict_captured": strict_flags,
            "loose_captured": loose_flags,
            "episode_done": episode_done,
            "g_max": g_max_vals,
            "capture_mode": reward_cfg.capture_mode,
        }

    def task_seed(self, env_idx: int) -> TaskSeed:
        return TaskSeed(
            scenario_name=self.scenario_names[env_idx],
            pursuers=self.pursuers[env_idx].copy(),
            evader=self.evader[env_idx].copy(),
            pursuer_v=self.pursuer_v[env_idx].copy(),
            evader_v=self.evader_v[env_idx].copy(),
        )

    @property
    def aeg_sampler(self) -> AdaptiveEnvironmentGenerator:
        return self.aeg
