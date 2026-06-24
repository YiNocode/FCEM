"""Unified evader policy dispatch (APF legacy vs differential-game)."""

from __future__ import annotations

from typing import Any

import numpy as np

from common.evader_apf import evader_apf_kwargs_from_config, evader_apf_step
from common.evader_game import evader_game_kwargs_from_config, evader_game_step
from common.dynamics import norm, unit

POLICIES = {
    "apf": evader_apf_step,
    "game": evader_game_step,
}


def evader_step(
    evader: np.ndarray,
    evader_v: np.ndarray,
    pursuer_centroid: np.ndarray,
    obstacles: list,
    bounds: tuple[float, float, float, float],
    dt: float,
    vmax: float,
    amax: float,
    pursuers: np.ndarray | None = None,
    policy: str = "game",
    **kwargs: Any,
) -> tuple[np.ndarray, np.ndarray]:
    step_fn = POLICIES.get(policy, evader_game_step)
    return step_fn(
        evader,
        evader_v,
        pursuer_centroid,
        obstacles,
        bounds,
        dt,
        vmax,
        amax,
        pursuers=pursuers,
        **kwargs,
    )


def evader_kwargs_from_config(cfg: dict[str, Any]) -> dict[str, Any]:
    policy = str(cfg.get("evader_policy", "game")).lower()
    if policy == "apf":
        return {"policy": "apf", **evader_apf_kwargs_from_config(cfg)}
    return {"policy": "game", **evader_game_kwargs_from_config(cfg)}


def estimate_escape_direction(
    target: np.ndarray,
    target_v: np.ndarray,
    pursuer_centroid: np.ndarray,
    pursuers: np.ndarray | None = None,
    policy: str = "game",
    pursuer_vmax: float = 5.0,
    horizon: float = 0.85,
) -> np.ndarray:
    """FCEM-compatible escape direction estimate under active evader policy."""
    if policy == "apf":
        from common.evader_apf import pursuer_escape_direction

        esc = pursuer_escape_direction(
            target, pursuers, pursuer_centroid, centroid_gain=0.55, nearest_gain=0.75
        )
        esc = unit(0.75 * target_v + 0.55 * esc)
    else:
        from common.evader_game import game_escape_direction

        esc = game_escape_direction(
            target,
            target_v,
            pursuers,
            pursuer_vmax,
            horizon=horizon,
        )
    if norm(esc) < 1e-9:
        esc = np.array([1.0, 0.0])
    return esc
