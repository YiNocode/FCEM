"""OPEN MARL baseline controller (inference from trained checkpoint)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from baselines.common.low_level import baseline_cfg, empty_timing
from baselines.open_marl.observation import ObservationBuilder
from common.dynamics import integrate_point_mass
from common.obstacles import Obstacle
from metrics.experiment_logger import TimingBlock

ROOT = Path(__file__).resolve().parent.parent.parent


def _resolve_checkpoint(path: str) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    if not p.exists():
        raise FileNotFoundError(
            f"OPEN MARL checkpoint not found: {p}\n"
            "Train with: python experiments/train_open_marl.py --stage 1 --device cuda"
        )
    return p


def make_open_marl_controller() -> Any:
    try:
        import torch
        from baselines.open_marl.networks import OpenMARLPolicy
    except ImportError as e:
        raise ImportError(
            "OPEN MARL baseline requires PyTorch. Install: pip install -r requirements-rl.txt"
        ) from e

    policy: OpenMARLPolicy | None = None
    obs_builder: ObservationBuilder | None = None
    device = torch.device("cpu")

    def _ensure_loaded(config: dict[str, Any]) -> tuple[OpenMARLPolicy, ObservationBuilder]:
        nonlocal policy, obs_builder, device
        if policy is not None and obs_builder is not None:
            return policy, obs_builder

        bcfg = baseline_cfg(config, "open_marl")
        ckpt_path = _resolve_checkpoint(str(bcfg.get("checkpoint_path", "checkpoints/open_marl/default.pt")))
        device_name = str(bcfg.get("device", "cpu"))
        device = torch.device(device_name)

        policy, _meta = OpenMARLPolicy.load_checkpoint(str(ckpt_path), device=device)
        open_cfg = policy.cfg
        obs_builder = ObservationBuilder(open_cfg)
        w = config["world"]
        scale = max(w["xmax"] - w["xmin"], w["ymax"] - w["ymin"])
        obs_builder.reset(arena_scale=scale)
        return policy, obs_builder

    def controller(
        step: int,
        evader: np.ndarray,
        evader_v: np.ndarray,
        pursuers: np.ndarray,
        pursuer_v: np.ndarray,
        obstacles: list[Obstacle],
        bounds: tuple[float, float, float, float],
        R: float,
        prev_slots: np.ndarray | None,
        prev_assignment: tuple[int, ...] | None,
        config: dict[str, Any],
    ) -> dict[str, Any]:
        timing = empty_timing()
        bcfg = baseline_cfg(config, "open_marl")
        deterministic = bool(bcfg.get("deterministic", True))

        with TimingBlock() as tb:
            import torch

            pol, builder = _ensure_loaded(config)
            w = config["world"]
            scale = max(w["xmax"] - w["xmin"], w["ymax"] - w["ymin"])
            if abs(builder._arena_scale - scale) > 1e-3:
                builder.reset(arena_scale=scale)

            obs, _ = builder.build_observations(
                pursuers, pursuer_v, evader, evader_v, obstacles, step
            )
            history = builder.history_tensor()

            pursuers = pursuers.copy()
            pursuer_v = pursuer_v.copy()
            n = len(pursuers)
            dt = config["dt"]

            obs_t = torch.tensor(obs.tolist(), dtype=torch.float32, device=device).unsqueeze(0)
            hist_t = torch.tensor(history.tolist(), dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                actions, _, _, _ = pol.actor_forward_batch(obs_t, hist_t, deterministic=deterministic)
            actions_np = np.array(actions.cpu().tolist()[0], dtype=np.float32)

            for i in range(n):
                acc = actions_np[i]
                p_new, v_new = integrate_point_mass(
                    pursuers[i],
                    pursuer_v[i],
                    acc,
                    dt,
                    config["pursuer_vmax"],
                    config["pursuer_amax"],
                )
                pursuers[i] = p_new
                pursuer_v[i] = v_new
                xmin, xmax, ymin, ymax = bounds
                pursuers[i, 0] = np.clip(pursuers[i, 0], xmin + 0.15, xmax - 0.15)
                pursuers[i, 1] = np.clip(pursuers[i, 1], ymin + 0.15, ymax - 0.15)

        timing["prediction_ms"] = tb.ms
        timing["low_level_ms"] = tb.ms
        timing["total_ms"] = tb.ms

        return {
            "pursuers": pursuers,
            "pursuer_v": pursuer_v,
            "timing_ms": timing,
        }

    return controller
