"""FastAPI server for Vue real-time FCEM visualization."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from server.sim_session import SessionStore, VizConfig, options_payload, viz_config_from_dict

app = FastAPI(title="FCEM Viz Server", version="1.0.0")
store = SessionStore()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SessionCreateBody(BaseModel):
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
    remove_layers: list[str] = Field(default_factory=list)
    ablation: dict[str, bool] = Field(default_factory=dict)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/options")
def get_options() -> dict:
    return options_payload()


@app.post("/api/sessions")
def create_session(body: SessionCreateBody) -> dict:
    params = viz_config_from_dict(body.model_dump())
    session = store.create(params)
    reset_msg = session.reset()
    return {
        "session_id": session.session_id,
        "meta": reset_msg["meta"],
        "summary": reset_msg["summary"],
    }


@app.get("/api/sessions/{session_id}")
def get_session(session_id: str) -> dict:
    try:
        session = store.get(session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="session not found") from exc
    return {
        "session_id": session_id,
        "meta": session.meta(),
        "summary": session._summary(),
        "history_len": len(session.history),
    }


@app.post("/api/sessions/{session_id}/step")
def step_session(session_id: str) -> dict:
    try:
        session = store.get(session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="session not found") from exc
    return session.step()


@app.post("/api/sessions/{session_id}/reset")
def reset_session(session_id: str, body: SessionCreateBody | None = None) -> dict:
    try:
        session = store.get(session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="session not found") from exc
    params = viz_config_from_dict(body.model_dump()) if body is not None else None
    return session.reset(params)


@app.delete("/api/sessions/{session_id}")
def delete_session(session_id: str) -> dict:
    store.delete(session_id)
    return {"deleted": session_id}


@app.websocket("/api/ws/{session_id}")
async def ws_session(websocket: WebSocket, session_id: str) -> None:
    try:
        session = store.get(session_id)
    except KeyError:
        await websocket.close(code=4404)
        return

    await websocket.accept()
    await websocket.send_json({"type": "meta", "meta": session.meta(), "summary": session._summary()})

    playing = False
    speed = 1.0

    async def _recv_loop() -> None:
        nonlocal playing, speed
        while True:
            raw = await websocket.receive_text()
            data = json.loads(raw)
            action = data.get("action")

            if action == "play":
                playing = True
                speed = float(data.get("speed", 1.0))
            elif action == "pause":
                playing = False
            elif action == "step":
                playing = False
                await websocket.send_json(session.step())
            elif action == "reset":
                playing = False
                params = data.get("config")
                reset_params = viz_config_from_dict(params) if params else None
                await websocket.send_json(session.reset(reset_params))
            elif action == "configure":
                playing = False
                params = viz_config_from_dict(data.get("config", {}))
                await websocket.send_json(session.reset(params))
            elif action == "set_speed":
                speed = float(data.get("speed", 1.0))
            else:
                await websocket.send_json({"type": "error", "message": f"unknown action: {action}"})

    recv_task = asyncio.create_task(_recv_loop())

    try:
        while True:
            if playing and not session.done:
                msg = session.step()
                await websocket.send_json(msg)
                if msg.get("type") == "done":
                    playing = False
                dt = float(session.config.get("dt", 0.1))
                await asyncio.sleep(max(0.01, dt / max(speed, 0.1)))
            else:
                await asyncio.sleep(0.05)
    except WebSocketDisconnect:
        recv_task.cancel()
        return


# Serve built Vue app when viz/dist exists.
_dist = ROOT / "viz" / "dist"
if _dist.is_dir():
    app.mount("/", StaticFiles(directory=str(_dist), html=True), name="static")
