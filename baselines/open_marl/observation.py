"""Partial-observability observations and LOS for OPEN MARL."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from common.obstacles import Obstacle, point_segment_distance


MASK_VALUE = -5.0


@dataclass
class OpenMarlConfig:
    n_pursuers: int = 3
    history_step: int = 10
    future_prediction_step: int = 5
    obs_max_obstacles: int = 3
    mask_value: float = MASK_VALUE
    arena_scale: float = 40.0

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> OpenMarlConfig:
        return cls(
            n_pursuers=int(d.get("n_pursuers", 3)),
            history_step=int(d.get("history_step", 10)),
            future_prediction_step=int(d.get("future_prediction_step", 5)),
            obs_max_obstacles=int(d.get("obs_max_obstacles", 3)),
            mask_value=float(d.get("mask_value", MASK_VALUE)),
            arena_scale=float(d.get("arena_scale", 40.0)),
        )

    @property
    def obs_dim(self) -> int:
        # self_vel(2) + rel_evader(2) + others((n-1)*2) + obstacles(k*2)
        n = self.n_pursuers
        k = self.obs_max_obstacles
        return 2 + 2 + (n - 1) * 2 + k * 2

    @property
    def epn_input_dim(self) -> int:
        # pursuer positions (n*2) + evader pos(2) + evader vel(2) + timestep(1)
        return self.n_pursuers * 2 + 2 + 2 + 1

    @property
    def epn_target_dim(self) -> int:
        return self.future_prediction_step * 2


def _scale_vec(v: np.ndarray, scale: float) -> np.ndarray:
    return v / max(scale, 1e-6)


def has_line_of_sight(
    observer: np.ndarray,
    target: np.ndarray,
    obstacles: list[Obstacle],
    clearance: float = 0.05,
) -> bool:
    """Return True if segment observer→target is not blocked by obstacles."""
    for obs in obstacles:
        d = point_segment_distance(obs.center, observer, target)
        if d < obs.radius + clearance:
            return False
    return True


def team_evader_detected(
    pursuers: np.ndarray,
    evader: np.ndarray,
    obstacles: list[Obstacle],
) -> bool:
    """Any pursuer with LOS to evader → team shares detection (paper III-A)."""
    for p in pursuers:
        if has_line_of_sight(p, evader, obstacles):
            return True
    return False


def k_nearest_obstacle_offsets(
    pos: np.ndarray,
    obstacles: list[Obstacle],
    k: int,
    scale: float,
) -> np.ndarray:
    out = np.zeros(k * 2, dtype=np.float32)
    if not obstacles:
        return out
    dists = [(norm := float(np.linalg.norm(pos - o.center)), o) for o in obstacles]
    dists.sort(key=lambda x: x[0])
    for i, (_, obs) in enumerate(dists[:k]):
        rel = _scale_vec(obs.center - pos, scale)
        out[2 * i : 2 * i + 2] = rel
    return out


class ObservationBuilder:
    """Build per-agent observations and EPN history for one env instance."""

    def __init__(self, cfg: OpenMarlConfig) -> None:
        self.cfg = cfg
        self.history: list[np.ndarray] = []
        self._arena_scale = cfg.arena_scale

    def reset(self, arena_scale: float | None = None) -> None:
        self.history = []
        if arena_scale is not None:
            self._arena_scale = arena_scale

    def _epn_frame(
        self,
        pursuers: np.ndarray,
        evader: np.ndarray,
        evader_v: np.ndarray,
        step: int,
        detected: bool,
    ) -> np.ndarray:
        scale = self._arena_scale
        n = self.cfg.n_pursuers
        frame = np.zeros(self.cfg.epn_input_dim, dtype=np.float32)
        idx = 0
        for i in range(n):
            frame[idx : idx + 2] = _scale_vec(pursuers[i], scale)
            idx += 2
        if detected:
            frame[idx : idx + 2] = _scale_vec(evader, scale)
            frame[idx + 2 : idx + 4] = _scale_vec(evader_v, scale)
        else:
            frame[idx : idx + 4] = self.cfg.mask_value
        frame[-1] = step / 1000.0
        return frame

    def _push_history(
        self,
        pursuers: np.ndarray,
        evader: np.ndarray,
        evader_v: np.ndarray,
        step: int,
        detected: bool,
    ) -> None:
        frame = self._epn_frame(pursuers, evader, evader_v, step, detected)
        self.history.append(frame)
        if len(self.history) > self.cfg.history_step:
            self.history.pop(0)

    def history_tensor(self) -> np.ndarray:
        """Shape (history_step, epn_input_dim), zero-padded at start."""
        h = self.cfg.history_step
        d = self.cfg.epn_input_dim
        out = np.zeros((h, d), dtype=np.float32)
        n = len(self.history)
        if n == 0:
            return out
        out[-n:] = np.stack(self.history[-n:], axis=0)
        return out

    def future_evader_targets(
        self,
        pursuers: np.ndarray,
        evader_future: np.ndarray,
    ) -> np.ndarray:
        """Relative evader positions for K future steps from pursuer centroid."""
        centroid = np.mean(pursuers, axis=0)
        k = self.cfg.future_prediction_step
        scale = self._arena_scale
        targets = np.zeros(k * 2, dtype=np.float32)
        for t in range(min(k, len(evader_future))):
            rel = _scale_vec(evader_future[t] - centroid, scale)
            targets[2 * t : 2 * t + 2] = rel
        return targets

    def build_observations(
        self,
        pursuers: np.ndarray,
        pursuer_v: np.ndarray,
        evader: np.ndarray,
        evader_v: np.ndarray,
        obstacles: list[Obstacle],
        step: int,
        update_history: bool = True,
    ) -> tuple[np.ndarray, bool]:
        """Return (n_agents, obs_dim) and team detection flag."""
        cfg = self.cfg
        scale = self._arena_scale
        n = cfg.n_pursuers
        detected = team_evader_detected(pursuers, evader, obstacles)

        if update_history:
            self._push_history(pursuers, evader, evader_v, step, detected)

        obs = np.zeros((n, cfg.obs_dim), dtype=np.float32)
        for i in range(n):
            idx = 0
            obs[i, idx : idx + 2] = _scale_vec(pursuer_v[i], scale)
            idx += 2
            if detected:
                obs[i, idx : idx + 2] = _scale_vec(evader - pursuers[i], scale)
            else:
                obs[i, idx : idx + 2] = cfg.mask_value
            idx += 2
            others = [j for j in range(n) if j != i]
            for j in others:
                obs[i, idx : idx + 2] = _scale_vec(pursuers[j] - pursuers[i], scale)
                idx += 2
            k_obs = k_nearest_obstacle_offsets(pursuers[i], obstacles, cfg.obs_max_obstacles, scale)
            obs[i, idx : idx + len(k_obs)] = k_obs
        return obs, detected
