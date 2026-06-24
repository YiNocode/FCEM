"""Simulation session builder for the Vue visualization server."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from baselines.registry import METHODS, normalize_method
from common.obstacles import Obstacle, scenario_random_obstacles, scenario_single_exit
from envs.sim2d import Sim2D, make_fcem_controller
from experiments.config_loader import load_config
from experiments.layer_registry import ALL_LAYER_IDS, flags_for_remove_layer, layer_definitions
from server.serialize import frame_to_client, session_meta


@dataclass
class VizConfig:
    method: str = "fcem"
    scenario: str = "free"
    seed: int = 42
    world_size: float = 40.0
    obstacle_count: int = 8
    pursuer_vmax: float = 4.0
    evader_vmax: float = 10.0
    pursuer_amax: float = 3.2
    evader_amax: float = 4.0
    evader_policy: str = "game"
    max_steps: int = 1200
    remove_layers: list[str] = field(default_factory=list)
    ablation: dict[str, bool] = field(default_factory=dict)


def _obstacles_for_scenario(
    scenario: str,
    *,
    seed: int,
    obstacle_count: int,
    bounds: tuple[float, float, float, float],
) -> list[Obstacle]:
    if scenario == "free":
        return []
    if scenario == "random_obstacles":
        return scenario_random_obstacles(seed=seed, n=max(0, obstacle_count), bounds=bounds)
    if scenario == "single_exit":
        return scenario_single_exit(bounds=bounds)
    return []


def _obstacles_to_yaml(obstacles: list[Obstacle]) -> list[dict[str, Any]]:
    return [{"center": obs.center.tolist(), "radius": float(obs.radius)} for obs in obstacles]


def _resolve_ablation(remove_layers: list[str], overrides: dict[str, bool]) -> dict[str, bool]:
    flags: dict[str, bool] = {}
    for layer_id in remove_layers:
        lid = layer_id.upper()
        if lid in ALL_LAYER_IDS:
            flags.update(flags_for_remove_layer(lid))
    flags.update(overrides)
    return flags


def build_sim_config(params: VizConfig) -> tuple[dict[str, Any], list[Obstacle]]:
    """Build merged runtime config and obstacle list from UI parameters."""
    scenario = params.scenario
    base_scenario = scenario if scenario in ("free", "random_obstacles", "single_exit") else "free"

    ablation_flags = _resolve_ablation(params.remove_layers, params.ablation)
    cfg = load_config(base_scenario, ablation_flags=ablation_flags or None)

    size = float(params.world_size)
    bounds = (0.0, size, 0.0, size)
    cfg["world"] = {"xmin": 0.0, "xmax": size, "ymin": 0.0, "ymax": size}
    cfg["seed"] = int(params.seed)
    cfg["max_steps"] = int(params.max_steps)
    cfg["pursuer_vmax"] = float(params.pursuer_vmax)
    cfg["evader_vmax"] = float(params.evader_vmax)
    cfg["pursuer_amax"] = float(params.pursuer_amax)
    cfg["evader_amax"] = float(params.evader_amax)
    cfg["evader_policy"] = params.evader_policy

    obstacles = _obstacles_for_scenario(
        scenario,
        seed=int(params.seed),
        obstacle_count=int(params.obstacle_count),
        bounds=bounds,
    )
    scenario_block = dict(cfg.get("scenario", {}))
    scenario_block["name"] = scenario
    scenario_block["obstacles"] = _obstacles_to_yaml(obstacles)
    cfg["scenario"] = scenario_block

    return cfg, obstacles


def make_controller(method: str, ablation: dict[str, bool]) -> Any:
    method = normalize_method(method)
    if method == "fcem":
        return make_fcem_controller(ablation)
    factory = METHODS.get(method)
    if factory is None:
        raise ValueError(f"Unknown method: {method}")
    return factory(None)


class SimSession:
    """One interactive simulation instance."""

    def __init__(self, params: VizConfig) -> None:
        self.session_id = uuid.uuid4().hex[:12]
        self.params = params
        self.config, self.obstacles = build_sim_config(params)
        self.method = normalize_method(params.method)
        self.controller = make_controller(self.method, self.config.get("ablation", {}))
        self.rng = np.random.default_rng(self.config.get("seed", 0))
        self.sim = Sim2D(self.config, self.obstacles, self.controller, self.rng)
        self.step_idx = 0
        self.history: list[dict[str, Any]] = []
        self.done = False

    def meta(self) -> dict[str, Any]:
        return session_meta(
            method=self.method,
            scenario=self.params.scenario,
            config=self.config,
            obstacles=_obstacles_to_yaml(self.obstacles),
        )

    def reset(self, params: VizConfig | None = None) -> dict[str, Any]:
        if params is not None:
            self.params = params
            self.config, self.obstacles = build_sim_config(params)
            self.method = normalize_method(params.method)
            self.controller = make_controller(self.method, self.config.get("ablation", {}))
            self.rng = np.random.default_rng(self.config.get("seed", 0))
            self.sim = Sim2D(self.config, self.obstacles, self.controller, self.rng)
        else:
            self.sim.reset()
        self.step_idx = 0
        self.history = []
        self.done = False
        return {"type": "meta", "meta": self.meta(), "summary": self._summary()}

    def step(self) -> dict[str, Any]:
        if self.done:
            return {"type": "done", "summary": self._summary(), "frame": self._last_frame()}

        frame = self.sim.step_once(self.step_idx)
        self.step_idx += 1
        client_frame = frame_to_client(frame)
        self.history.append(client_frame)

        if self.sim.captured or self.sim.failed or self.step_idx >= self.config["max_steps"]:
            self.done = True
            return {
                "type": "done",
                "frame": client_frame,
                "summary": self._summary(),
            }
        return {"type": "frame", "frame": client_frame, "summary": self._summary()}

    def _last_frame(self) -> dict[str, Any] | None:
        return self.history[-1] if self.history else None

    def _summary(self) -> dict[str, Any]:
        return {
            "step": self.step_idx,
            "captured": self.sim.captured,
            "capture_step": self.sim.capture_step,
            "failed": self.sim.failed,
            "failure_step": self.sim.failure_step,
            "failure_reason": self.sim.failure_reason,
            "done": self.done,
        }


class SessionStore:
    def __init__(self) -> None:
        self._sessions: dict[str, SimSession] = {}

    def create(self, params: VizConfig) -> SimSession:
        session = SimSession(params)
        self._sessions[session.session_id] = session
        return session

    def get(self, session_id: str) -> SimSession:
        session = self._sessions.get(session_id)
        if session is None:
            raise KeyError(session_id)
        return session

    def delete(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)


def options_payload() -> dict[str, Any]:
    methods = sorted({normalize_method(k) for k in METHODS})
    layers = [
        {
            "id": layer["id"],
            "name": layer.get("name", ""),
            "description": layer.get("description", ""),
            "experiment": layer.get("experiment", ""),
        }
        for layer in layer_definitions()
    ]
    return {
        "methods": methods,
        "scenarios": [
            {"id": "free", "label": "空旷场地"},
            {"id": "random_obstacles", "label": "随机圆柱障碍"},
            {"id": "single_exit", "label": "单出口 U 形"},
        ],
        "evader_policies": [
            {"id": "game", "label": "微分博弈 (game)"},
            {"id": "apf", "label": "APF 势场"},
        ],
        "layers": layers,
        "defaults": {
            "method": "fcem",
            "scenario": "free",
            "world_size": 40.0,
            "obstacle_count": 8,
            "pursuer_vmax": 4.0,
            "evader_vmax": 10.0,
            "pursuer_amax": 3.2,
            "evader_amax": 4.0,
            "evader_policy": "game",
            "seed": 42,
            "max_steps": 1200,
            "remove_layers": [],
        },
    }


def viz_config_from_dict(data: dict[str, Any]) -> VizConfig:
    return VizConfig(
        method=str(data.get("method", "fcem")),
        scenario=str(data.get("scenario", "free")),
        seed=int(data.get("seed", 42)),
        world_size=float(data.get("world_size", 40.0)),
        obstacle_count=int(data.get("obstacle_count", 8)),
        pursuer_vmax=float(data.get("pursuer_vmax", 4.0)),
        evader_vmax=float(data.get("evader_vmax", 10.0)),
        pursuer_amax=float(data.get("pursuer_amax", 3.2)),
        evader_amax=float(data.get("evader_amax", 4.0)),
        evader_policy=str(data.get("evader_policy", "game")),
        max_steps=int(data.get("max_steps", 1200)),
        remove_layers=[str(x).upper() for x in data.get("remove_layers", [])],
        ablation={str(k): bool(v) for k, v in (data.get("ablation") or {}).items()},
    )
