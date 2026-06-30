"""Adaptive Environment Generator (AEG) — 2D curriculum for OPEN MARL."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from baselines.open_marl.reward import CaptureMode


@dataclass
class TaskSeed:
    scenario_name: str
    pursuers: np.ndarray
    evader: np.ndarray
    pursuer_v: np.ndarray
    evader_v: np.ndarray
    success_rate: float = 0.0


@dataclass
class AEGConfig:
    local_prob: float = 0.7
    sigma_min: float = 0.5
    sigma_max: float = 0.9
    position_noise: float = 3.0
    archive_max: int = 256

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> AEGConfig:
        return cls(
            local_prob=float(d.get("local_prob", 0.7)),
            sigma_min=float(d.get("sigma_min", 0.5)),
            sigma_max=float(d.get("sigma_max", 0.9)),
            position_noise=float(d.get("position_noise", 3.0)),
            archive_max=int(d.get("archive_max", 256)),
        )


@dataclass
class CurriculumPhase:
    name: str
    step_fraction_end: float
    scenarios: list[str]
    dynamics: dict[str, float]
    center_init: bool = False
    use_aeg: bool = False
    capture_mode: CaptureMode = "strict"
    episode_max_steps: int | None = None
    reward_overrides: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CurriculumPhase:
        return cls(
            name=str(d.get("name", "phase")),
            step_fraction_end=float(d.get("step_fraction_end", 1.0)),
            scenarios=list(d.get("scenarios", ["free"])),
            dynamics=dict(d.get("dynamics", {})),
            center_init=bool(d.get("center_init", False)),
            use_aeg=bool(d.get("use_aeg", False)),
            capture_mode=str(d.get("capture_mode", "strict")),  # type: ignore[arg-type]
            episode_max_steps=int(d["episode_max_steps"]) if d.get("episode_max_steps") is not None else None,
            reward_overrides=dict(d.get("reward_overrides", {})),
        )


@dataclass
class CurriculumConfig:
    phases: list[CurriculumPhase] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CurriculumConfig:
        raw = d.get("phases", [])
        if not raw:
            raw = [
                {
                    "name": "easy",
                    "step_fraction_end": 0.35,
                    "scenarios": ["free"],
                    "dynamics": {"pursuer_vmax": 5.0, "evader_vmax": 3.0, "evader_amax": 2.0},
                    "center_init": True,
                    "capture_mode": "loose",
                    "episode_max_steps": 600,
                    "reward_overrides": {"capture_mode": "loose", "spread_coef": 4.0},
                },
                {
                    "name": "medium",
                    "step_fraction_end": 0.75,
                    "scenarios": ["free"],
                    "dynamics": {"pursuer_vmax": 4.5, "evader_vmax": 6.0},
                    "capture_mode": "loose",
                    "episode_max_steps": 900,
                },
                {
                    "name": "hard",
                    "step_fraction_end": 1.0,
                    "scenarios": ["free", "random_obstacles", "single_exit"],
                    "dynamics": {"pursuer_vmax": 4.0, "evader_vmax": 10.0},
                    "capture_mode": "strict",
                    "use_aeg": True,
                },
            ]
        return cls(phases=[CurriculumPhase.from_dict(p) for p in raw])

    def phase_at(self, global_step: int, total_steps: int) -> CurriculumPhase:
        frac = global_step / max(total_steps, 1)
        for phase in self.phases:
            if frac <= phase.step_fraction_end:
                return phase
        return self.phases[-1]


class AdaptiveEnvironmentGenerator:
    """Local expansion archive + global exploration (paper Algo.1, 2D)."""

    def __init__(self, cfg: AEGConfig, rng: np.random.Generator) -> None:
        self.cfg = cfg
        self.rng = rng
        self.archive: list[TaskSeed] = []

    def update_archive(self, seed: TaskSeed, success_rate: float) -> None:
        seed = copy.deepcopy(seed)
        seed.success_rate = success_rate
        if self.cfg.sigma_min <= success_rate <= self.cfg.sigma_max:
            self.archive.append(seed)
            if len(self.archive) > self.cfg.archive_max:
                self.archive.pop(0)

    def sample_task(
        self,
        scenario_names: list[str],
        bounds: tuple[float, float, float, float],
        global_explore: bool,
    ) -> tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
        if global_explore or not self.archive or self.rng.random() > self.cfg.local_prob:
            return None

        seed = self.archive[int(self.rng.integers(0, len(self.archive)))]
        if seed.scenario_name not in scenario_names:
            return None

        noise = self.rng.uniform(-self.cfg.position_noise, self.cfg.position_noise, (4, 2))
        pursuers = seed.pursuers + noise[: seed.pursuers.shape[0]]
        evader = seed.evader + noise[-1]

        xmin, xmax, ymin, ymax = bounds
        margin = 0.15
        pursuers[:, 0] = np.clip(pursuers[:, 0], xmin + margin, xmax - margin)
        pursuers[:, 1] = np.clip(pursuers[:, 1], ymin + margin, ymax - margin)
        evader[0] = np.clip(evader[0], xmin + margin, xmax - margin)
        evader[1] = np.clip(evader[1], ymin + margin, ymax - margin)

        return (
            seed.scenario_name,
            pursuers.astype(np.float32),
            seed.pursuer_v.copy(),
            evader.astype(np.float32),
            seed.evader_v.copy(),
        )


def sample_center_init(
    bounds: tuple[float, float, float, float],
    rng: np.random.Generator,
    n_pursuers: int = 3,
    min_dist: float = 5.0,
    max_dist: float = 9.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Evader near center; pursuers on a ring with 120° spacing."""
    xmin, xmax, ymin, ymax = bounds
    cx, cy = 0.5 * (xmin + xmax), 0.5 * (ymin + ymax)
    evader = np.array([cx, cy], dtype=np.float32) + rng.uniform(-1.0, 1.0, 2).astype(np.float32)

    angles = np.linspace(0, 2 * np.pi, n_pursuers, endpoint=False) + float(rng.uniform(0, 2 * np.pi))
    radius = float(rng.uniform(min_dist, max_dist))
    pursuers = np.zeros((n_pursuers, 2), dtype=np.float32)
    for i, ang in enumerate(angles):
        pursuers[i] = evader + radius * np.array([np.cos(ang), np.sin(ang)], dtype=np.float32)

    pursuer_v = np.zeros((n_pursuers, 2), dtype=np.float32)
    evader_v = np.zeros(2, dtype=np.float32)
    return pursuers, pursuer_v, evader, evader_v
