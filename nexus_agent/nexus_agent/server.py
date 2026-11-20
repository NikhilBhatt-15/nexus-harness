"""HTTP API so other apps can drive agents.

    POST /v1/runs                 {"task", "session_id"?, "model"?, "tools"? (spec), "mode"?} -> run
    GET  /v1/runs/{id}            status, result, pending approval
    GET  /v1/runs/{id}/events     ?after=N  (poll)   |   /v1/runs/{id}/stream  (Server-Sent Events)
    POST /v1/runs/{id}/approve    {"allow": true|false}  answer a pending tool approval
    GET  /v1/sessions             saved sessions
    GET  /v1/tools                toolsets of the default config

Agents are cached per session id, so follow-up runs with the same ``session_id`` continue the
conversation. Set ``NXA_SERVER_TOKEN`` to require ``Authorization: Bearer <token>``.
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse

from nexus_agent.config import build_agent, load_config
from nexus_agent.runtime import Session


def create_app(config_path: str | None = None) -> FastAPI:
    app = FastAPI(title="nexus-agent")
    cfg = load_config(config_path)
    agents: dict[str, Any] = {}
    runs: dict[str, dict] = {}
    lock = threading.Lock()

    def auth(authorization: str | None = Header(default=None)) -> None:
        token = os.environ.get("NXA_SERVER_TOKEN")
        if token and authorization != f"Bearer {token}":
            raise HTTPException(401, "bad token")

    def approver_for(run: dict):
        def approve(t, args) -> bool:
            ev = threading.Event()
            risk, reasons = t.risk_for(args)
            run["pending_approval"] = {"tool": t.name, "risk": risk, "reasons": reasons, "arguments": args, "_ev": ev}
            ok = ev.wait(timeout=float(os.environ.get("NXA_APPROVAL_TIMEOUT", "300")))
            decision = bool(ok and run["pending_approval"].get("allow"))
            run["pending_approval"] = None
            return decision
        return approve

    def worker(run: dict, agent, task: str) -> None:
        run["status"] = "running"
        # collect via on_event so events emitted during a blocking model call (retries) show up live
        agent.on_event = run["events"].append
        try:
            for _ in agent.stream(task):
                pass
        finally:
            agent.on_event = None
        done = run["events"][-1]
        run.update(status="completed", result={k: done.get(k) for k in
                                               ("text", "stop_reason", "steps", "tool_calls", "cost_usd", "error")})

    @app.post("/v1/runs", dependencies=[Depends(auth)])
    def create_run(body: dict):
        task = body.get("task")
        if not task:
            raise HTTPException(400, "task is required")
        sid = body.get("session_id")
        run = {"id": f"run_{uuid.uuid4().hex[:10]}", "status": "queued", "events": [], "result": None,
               "pending_approval": None, "created": time.time()}
        with lock:
            agent = agents.get(sid) if sid else None
            if agent is None:
                agent = build_agent(cfg, model=body.get("model"), tool_spec=body.get("tools"),
                                    mode=body.get("mode"), session=Session(sid) if sid else True)
                agents[agent.session.id] = agent
        agent.permissions.approver = approver_for(run)
        run["session_id"] = agent.session.id
        runs[run["id"]] = run
        threading.Thread(target=worker, args=(run, agent, task), daemon=True).start()
        return _public(run)

    def _public(run: dict) -> dict:
        pa = run.get("pending_approval")
        return {**{k: v for k, v in run.items() if k not in ("events", "pending_approval")},
                "events": len(run["events"]),
                "pending_approval": {k: v for k, v in pa.items() if k != "_ev"} if pa else None}

    def _get(run_id: str) -> dict:
        if run_id not in runs:
            raise HTTPException(404, "run not found")
        return runs[run_id]

    @app.get("/v1/runs/{run_id}", dependencies=[Depends(auth)])
    def get_run(run_id: str):
        return _public(_get(run_id))

    @app.get("/v1/runs/{run_id}/events", dependencies=[Depends(auth)])
    def get_events(run_id: str, after: int = 0):
        run = _get(run_id)
        return {"status": run["status"], "events": run["events"][after:], "next": len(run["events"])}

    @app.get("/v1/runs/{run_id}/stream", dependencies=[Depends(auth)])
    def stream(run_id: str):
        run = _get(run_id)

        def gen():
            i = 0
            while True:
                while i < len(run["events"]):
                    yield f"data: {json.dumps(run['events'][i], default=str)}\n\n"
                    i += 1
                if run["status"] == "completed" and i >= len(run["events"]):
                    break
                time.sleep(0.2)

        return StreamingResponse(gen(), media_type="text/event-stream")

    @app.post("/v1/runs/{run_id}/approve", dependencies=[Depends(auth)])
    def approve(run_id: str, body: dict):
        run = _get(run_id)
        pa = run.get("pending_approval")
        if not pa:
            raise HTTPException(409, "no pending approval")
        pa["allow"] = bool(body.get("allow"))
        pa["_ev"].set()
        return {"ok": True}

    @app.get("/v1/sessions", dependencies=[Depends(auth)])
    def sessions():
        return Session.list()

    @app.get("/v1/sessions/{session_id}/messages", dependencies=[Depends(auth)])
    def session_messages(session_id: str):
        from nexus_agent.runtime import sessions_dir

        if not (sessions_dir() / f"{session_id}.jsonl").exists():
            raise HTTPException(404, "session not found")
        return Session(session_id).messages()

    @app.get("/v1/info", dependencies=[Depends(auth)])
    def info():
        from nexus_agent.config import toolset_from_entry

        sets = []
        for e in cfg.get("toolsets", []) or []:
            try:
                sets.append(toolset_from_entry(*next(iter(e.items())), cfg.get("_dir")).describe())
            except Exception as ex:  # show misconfigured toolsets instead of failing the page
                sets.append(f"{next(iter(e))}: (error: {ex})")
        from pathlib import Path

        from nexus_agent.runtime import load_skills

        skills = [s.name for s in load_skills([Path(cfg.get('_dir', '.')) / d for d in cfg.get('skills_dirs') or []])]
        return {"model": cfg.get("model"), "mode": cfg.get("permissions", "auto"), "toolsets": sets,
                "skills": skills}

    @app.get("/", include_in_schema=False)
    def index():
        from pathlib import Path

        from fastapi.responses import HTMLResponse

        return HTMLResponse((Path(__file__).parent / "web" / "index.html").read_text(encoding="utf-8"))

    @app.get("/v1/tools", dependencies=[Depends(auth)])
    def tools():
        from nexus_agent.config import toolset_from_entry

        return [toolset_from_entry(*next(iter(e.items())), cfg.get("_dir")).describe() for e in cfg.get("toolsets", [])]

    return app


def serve(config_path: str | None, host: str = "127.0.0.1", port: int = 8787) -> None:
    import uvicorn

    uvicorn.run(create_app(config_path), host=host, port=port)
