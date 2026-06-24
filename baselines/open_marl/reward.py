"""OPEN-style team reward for MARL training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from common.obstacles import Obstacle
from common.dynamics import norm


@dataclass
class RewardConfig:
    capture_reward: float = 6.0
    distance_coef: float = 0.1
    collision_penalty: float = 10.0
    smoothness_coef: float = 0.01
    safety_margin: float = 0.15

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RewardConfig:
        return cls(
            capture_reward=float(d.get("capture_reward", 6.0)),
            distance_coef=float(d.get("distance_coef", 0.1)),
            collision_penalty=float(d.get("collision_penalty", 10.0)),
            smoothness_coef=float(d.get("smoothness_coef", 0.01)),
            safety_margin=float(d.get("safety_margin", 0.15)),
        )


def compute_team_reward(
    pursuers: np.ndarray,
    evader: np.ndarray,
    obstacles: list[Obstacle],
    actions: np.ndarray,
    prev_actions: np.ndarray | None,
    capture_radius: float,
    cfg: RewardConfig,
) -> tuple[float, dict[str, float]]:
    """Scalar team reward and component breakdown."""
    dists = np.linalg.norm(pursuers - evader[None, :], axis=1)
    mean_dist = float(np.mean(dists))

    components = {
        "distance": -cfg.distance_coef * mean_dist,
        "capture": 0.0,
        "collision": 0.0,
        "smoothness": 0.0,
    }

    if bool(np.any(dists <= capture_radius)):
        components["capture"] = cfg.capture_reward

    for p in pursuers:
        for obs in obstacles:
            d_surface = norm(p - obs.center) - obs.radius
            if d_surface < cfg.safety_margin:
                components["collision"] -= cfg.collision_penalty

    if prev_actions is not None and cfg.smoothness_coef > 0:
        delta = actions - prev_actions
        components["smoothness"] = -cfg.smoothness_coef * float(np.sum(delta * delta))

    total = sum(components.values())
    return total, components
