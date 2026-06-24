"""Smoke tests for all registered pursuit methods."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.registry import METHODS
from envs.sim2d import Sim2D
from experiments.config_loader import load_config, obstacles_from_scenario


def test_all_methods_five_steps() -> None:
    cfg = load_config("free")
    cfg["max_steps"] = 5
    obstacles = obstacles_from_scenario(cfg["scenario"])
    rng = np.random.default_rng(0)

    for name, factory in METHODS.items():
        if name.endswith("_apf"):
            continue
        if name == "open_marl":
            ckpt = ROOT / "checkpoints" / "open_marl" / "default.pt"
            if not ckpt.exists():
                continue
        controller = factory(None)
        sim = Sim2D(cfg, obstacles, controller, rng)
        sim.reset()
        for step in range(5):
            frame = sim.step_once(step)
            assert "pursuers" in frame
            assert frame["pursuers"].shape[0] == 3
        result = {"captured": sim.captured, "frames": sim.frames}
        assert len(result["frames"]) == 5


if __name__ == "__main__":
    test_all_methods_five_steps()
    print("OK: all methods completed 5 steps")
