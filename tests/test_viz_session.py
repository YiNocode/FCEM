"""Tests for Vue visualization session builder."""

from __future__ import annotations

from server.sim_session import SessionStore, VizConfig, build_sim_config, options_payload


def test_options_payload_has_methods_and_layers() -> None:
    opts = options_payload()
    assert "fcem" in opts["methods"]
    assert len(opts["layers"]) == 4
    assert opts["defaults"]["pursuer_vmax"] == 4.0


def test_build_random_obstacle_scenario() -> None:
    cfg, obstacles = build_sim_config(
        VizConfig(scenario="random_obstacles", obstacle_count=5, world_size=50.0, seed=1)
    )
    assert len(obstacles) == 5
    assert cfg["world"]["xmax"] == 50.0
    assert len(cfg["scenario"]["obstacles"]) == 5


def test_session_step_and_capture() -> None:
    store = SessionStore()
    session = store.create(VizConfig(method="pure_pursuit", scenario="free", max_steps=500))
    session.reset()
    done = False
    for _ in range(500):
        msg = session.step()
        if msg["type"] == "done":
            done = True
            break
    assert done
    assert session.step_idx > 0
