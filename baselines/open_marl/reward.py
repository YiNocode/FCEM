"""OPEN-style team reward for MARL training (curriculum capture modes)."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Literal

import numpy as np

from common.obstacles import Obstacle
from common.dynamics import norm

CaptureMode = Literal["loose", "strict"]


@dataclass
class RewardConfig:
    capture_reward: float = 6.0
    loose_capture_bonus: float = 3.0
    distance_coef: float = 0.05
    proximity_coef: float = 0.25
    spread_coef: float = 3.0
    structure_penalty_coef: float = 0.02
    collision_penalty: float = 10.0
    smoothness_coef: float = 4.0
    safety_margin: float = 0.15
    reward_scale: float = 1.0
    reward_clip: float = 10.0
    use_smoothness: bool = False
    capture_mode: CaptureMode = "strict"

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RewardConfig:
        return cls(
            capture_reward=float(d.get("capture_reward", 6.0)),
            loose_capture_bonus=float(d.get("loose_capture_bonus", 3.0)),
            distance_coef=float(d.get("distance_coef", 0.05)),
            proximity_coef=float(d.get("proximity_coef", 0.25)),
            spread_coef=float(d.get("spread_coef", 3.0)),
            structure_penalty_coef=float(d.get("structure_penalty_coef", 0.02)),
            collision_penalty=float(d.get("collision_penalty", 10.0)),
            smoothness_coef=float(d.get("smoothness_coef", 4.0)),
            safety_margin=float(d.get("safety_margin", 0.15)),
            reward_scale=float(d.get("reward_scale", 1.0)),
            reward_clip=float(d.get("reward_clip", 10.0)),
            use_smoothness=bool(d.get("use_smoothness", False)),
            capture_mode=str(d.get("capture_mode", "strict")),  # type: ignore[arg-type]
        )

    def with_overrides(self, overrides: dict[str, Any]) -> RewardConfig:
        if not overrides:
            return self
        data = {f.name: getattr(self, f.name) for f in self.__dataclass_fields__.values()}
        data.update(overrides)
        return RewardConfig(**data)


def evaluate_capture(
    pursuers: np.ndarray,
    evader: np.ndarray,
    capture_radius: float,
    g_max: float,
    g_max_allowed: float,
    mode: CaptureMode,
) -> tuple[bool, bool]:
    """Return (episode_success, strict_success)."""
    dists = np.linalg.norm(pursuers - evader[None, :], axis=1)
    loose = bool(np.any(dists <= capture_radius))
    strict = loose and bool(np.all(dists <= capture_radius)) and g_max <= g_max_allowed
    if mode == "loose":
        return loose, strict
    return strict, strict


def compute_team_reward(
    pursuers: np.ndarray,
    evader: np.ndarray,
    obstacles: list[Obstacle],
    actions: np.ndarray,
    prev_actions: np.ndarray | None,
    prev_min_dist: float | None,
    capture_radius: float,
    metrics: dict[str, float],
    g_max_allowed: float,
    strict_captured: bool,
    loose_captured: bool,
    cfg: RewardConfig,
) -> tuple[float, dict[str, float]]:
    """Scalar team reward; clipped to keep critic stable."""
    dists = np.linalg.norm(pursuers - evader[None, :], axis=1)
    mean_dist = float(np.mean(dists))
    min_dist = float(np.min(dists))
    g_max_deg = math.degrees(metrics["G_max"])
    g_allow_deg = math.degrees(g_max_allowed)
    d_ang = float(metrics.get("D_ang", 0.0))
    c_cov = float(metrics.get("C_cov", 0.0))

    components = {
        "distance": -cfg.distance_coef * mean_dist,
        "proximity": -cfg.proximity_coef * min_dist,
        "spread": cfg.spread_coef * (d_ang + c_cov),
        "capture": 0.0,
        "structure": 0.0,
        "approach": 0.0,
        "collision": 0.0,
        "smoothness": 0.0,
    }

    if strict_captured:
        components["capture"] = cfg.capture_reward
    elif loose_captured and cfg.capture_mode == "loose":
        components["capture"] = cfg.loose_capture_bonus

    all_near = bool(np.all(dists <= 2.0 * capture_radius))
    if all_near and g_max_deg > g_allow_deg:
        components["structure"] = -cfg.structure_penalty_coef * (g_max_deg - g_allow_deg)

    if prev_min_dist is not None:
        components["approach"] = 0.5 * (prev_min_dist - min_dist)

    for p in pursuers:
        for obs in obstacles:
            d_surface = norm(p - obs.center) - obs.radius
            if d_surface < cfg.safety_margin:
                components["collision"] -= cfg.collision_penalty

    if prev_actions is not None and cfg.use_smoothness and cfg.smoothness_coef > 0:
        delta = actions - prev_actions
        delta_norm = float(np.linalg.norm(delta))
        components["smoothness"] = cfg.smoothness_coef * math.exp(-delta_norm)

    total = sum(components.values()) * cfg.reward_scale
    if cfg.reward_clip > 0:
        total = float(np.clip(total, -cfg.reward_clip, cfg.reward_clip))
    return total, components
