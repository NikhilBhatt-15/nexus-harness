"""HTTP API + web UI for nexus-agent.

Projects (one per connected dataset / database, each with its own agent.yaml):
    GET    /v1/projects                         list projects
    POST   /v1/projects/connect                 {"source", "name", "dictionary"?, "describe_model"?} -> job
    GET    /v1/jobs/{id}                        connect job status + log (poll)
    POST   /v1/uploads?name=FILE&batch=ID        raw file body -> stored under NXA_HOME/uploads/<batch>/ (use the
                                                 returned folder as a connect source)
    DELETE /v1/projects/{name}                  unregister (files are kept)
Runs:
    POST /v1/runs                 {"project", "task", "session_id"?} -> run
    GET  /v1/runs/{id}            status, result, pending approval
    GET  /v1/runs/{id}/events     ?after=N (poll)  |  /v1/runs/{id}/stream (Server-Sent Events)
    POST /v1/runs/{id}/approve    {"allow": bool}  answer a pending tool approval
    POST /v1/runs/{id}/cancel     stop after the current step
Sessions / info:
    GET  /v1/sessions?project=    GET /v1/sessions/{id}/messages     GET /v1/info?project=
    GET  /                        web UI

Set NXA_SERVER_TOKEN to require ``Authorization: Bearer <token>`` (do this before exposing the server).
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse

from nexus_agent.config import build_agent, load_config, toolset_from_entry
from nexus_agent.runtime import Session, load_skills, sessions_dir


def nxa_home() -> Path:
    return Path(os.environ.get("NXA_HOME", Path.home() / ".nexus-agent"))


class Projects:
    """Registry of projects: name -> agent.yaml path (stored in NXA_HOME/projects.json)."""

    def __init__(self) -> None:
        self.path = nxa_home() / "projects.json"
        self.lock = threading.Lock()

    def all(self) -> dict[str, str]:
        if not self.path.exists():
            return {}
        data = json.loads(self.path.read_text(encoding="utf-8"))
        return {k: v for k, v in data.items() if Path(v).exists()}

    def add(self, name: str, config: str | Path) -> None:
        with self.lock:
            data = self.all()
            data[name] = str(Path(config).resolve())
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(data, indent=1), encoding="utf-8")

    def remove(self, name: str) -> None:
        with self.lock:
            data = self.all()
            data.pop(name, None)
            self.path.write_text(json.dumps(data, indent=1), encoding="utf-8")


def _project_name_for(config_path: str | Path) -> str:
    cfg = load_config(config_path)
    return str(cfg.get("name") or Path(config_path).resolve().parent.name)


def create_app(config_path: str | None = None) -> FastAPI:
    app = FastAPI(title="nexus-agent")
    projects = Projects()
    if config_path:
        projects.add(_project_name_for(config_path), config_path)
    agents: dict[str, Any] = {}      # session id -> Agent
    runs: dict[str, dict] = {}
    jobs: dict[str, dict] = {}
    lock = threading.Lock()

    def auth(request: Request, authorization: str | None = Header(default=None)) -> None:
        token = os.environ.get("NXA_SERVER_TOKEN")
        # header for API clients; ?token= for the browser's EventSource, which cannot send headers
        if token and authorization != f"Bearer {token}" and request.query_params.get("token") != token:
            raise HTTPException(401, "bad token")

    def cfg_for(project: str | None) -> tuple[str, dict]:
        all_p = projects.all()
        if not all_p:
            raise HTTPException(409, "no projects yet: connect a dataset first")
        name = project if project in all_p else next(iter(all_p))
        if project and project not in all_p:
            raise HTTPException(404, f"project {project!r} not found")
        return name, load_config(all_p[name])

    # ------------------------------------------------------------------ projects
    @app.get("/v1/projects", dependencies=[Depends(auth)])
    def list_projects():
        out = []
        for name, path in projects.all().items():
            cfg = load_config(path)
            prof = Path(path).parent / "profile.json"
            summary = json.loads(prof.read_text(encoding="utf-8")).get("summary", {}) if prof.exists() else {}
            out.append({"name": name, "config": path, "model": cfg.get("model"),
                        "tables": summary.get("tables"), "relations": len(summary.get("relations", []) or []),
                        "self_check": summary.get("self_check")})
        return out

    @app.get("/v1/projects/{name}/profile", dependencies=[Depends(auth)])
    def project_profile(name: str):
        path = projects.all().get(name)
        prof = Path(path).parent / "profile.json" if path else None
        if not prof or not prof.exists():
            raise HTTPException(404, "no profile for this project")
        data = json.loads(prof.read_text(encoding="utf-8"))
        return {"summary": data.get("summary"),
                "tables": [{"name": t["name"], "rows": t["rows"], "columns": len(t["columns"])} for t in data.get("tables", [])]}

    @app.delete("/v1/projects/{name}", dependencies=[Depends(auth)])
    def delete_project(name: str):
        projects.remove(name)
        return {"ok": True}

    @app.post("/v1/uploads", dependencies=[Depends(auth)])
    async def upload(request: Request, name: str, batch: str | None = None):
        safe = re.sub(r"[^\w.\-]", "_", Path(name).name)
        if not safe or safe.startswith("."):
            raise HTTPException(400, "bad file name")
        batch = re.sub(r"[^\w-]", "", batch or uuid.uuid4().hex[:10])
        folder = nxa_home() / "uploads" / batch
        folder.mkdir(parents=True, exist_ok=True)
        size = 0
        with open(folder / safe, "wb") as f:
            async for chunk in request.stream():
                f.write(chunk)
                size += len(chunk)
        return {"batch": batch, "folder": str(folder), "file": safe, "bytes": size}

    @app.post("/v1/projects/connect", dependencies=[Depends(auth)])
    def connect_project(body: dict):
        source, name = (body.get("source") or "").strip(), (body.get("name") or "").strip()
        if not source or not name:
            raise HTTPException(400, "source and name are required")
        from nexus_agent.connect import slugify

        slug = slugify(name)
        job = {"id": f"job_{uuid.uuid4().hex[:10]}", "name": slug, "status": "running", "log": [], "step": 0,
               "summary": None, "error": None, "started": time.time()}
        jobs[job["id"]] = job

        def log(msg: str) -> None:
            job["log"].append(msg)
            m = re.match(r"(\d)/7 ", msg)
            if m:
                job["step"] = int(m.group(1))

        def work() -> None:
            from nexus_agent.connect import connect

            try:
                out = Path(body.get("out") or (nxa_home() / "projects" / slug))
                summary = connect(source, slug, out=str(out), dictionary=body.get("dictionary") or None,
                                  describe_model=body.get("describe_model") or None,
                                  embedder=body.get("embedder") or os.environ.get("NXA_EMBEDDER") or None, log=log)
                projects.add(slug, out / "agent.yaml")
                job.update(status="done", summary=summary, step=8)
            except BaseException as e:  # surface every failure in the UI
                job.update(status="failed", error=f"{type(e).__name__}: {e}")
                log(f"FAILED: {type(e).__name__}: {e}")

        threading.Thread(target=work, daemon=True).start()
        return job

    @app.get("/v1/jobs/{job_id}", dependencies=[Depends(auth)])
    def get_job(job_id: str):
        if job_id not in jobs:
            raise HTTPException(404, "job not found")
        return jobs[job_id]

    # ------------------------------------------------------------------ runs
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
        agent.on_event = run["events"].append  # events emitted during blocking model calls show up live
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
        project, cfg = cfg_for(body.get("project"))
        sid = body.get("session_id")
        run = {"id": f"run_{uuid.uuid4().hex[:10]}", "status": "queued", "events": [], "result": None,
               "pending_approval": None, "created": time.time(), "project": project}
        with lock:
            agent = agents.get(sid) if sid else None
            if agent is None:
                session = Session(sid) if sid else Session(meta={"project": project, "model": cfg.get("model")})
                agent = build_agent(cfg, model=body.get("model"), tool_spec=body.get("tools"), mode=body.get("mode"),
                                    session=session)
                agents[agent.session.id] = agent
        agent.permissions.approver = approver_for(run)
        agent.cancel_requested = False
        run["session_id"] = agent.session.id
        runs[run["id"]] = run
        run["_agent"] = agent
        threading.Thread(target=worker, args=(run, agent, task), daemon=True).start()
        return _public(run)

    def _public(run: dict) -> dict:
        pa = run.get("pending_approval")
        return {**{k: v for k, v in run.items() if k not in ("events", "pending_approval", "_agent")},
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

    @app.post("/v1/runs/{run_id}/cancel", dependencies=[Depends(auth)])
    def cancel(run_id: str):
        run = _get(run_id)
        run["_agent"].cancel_requested = True
        pa = run.get("pending_approval")
        if pa:  # a run waiting for approval: deny so it can stop
            pa["allow"] = False
            pa["_ev"].set()
        return {"ok": True}

    # ------------------------------------------------------------------ sessions / info
    @app.get("/v1/sessions", dependencies=[Depends(auth)])
    def sessions(project: str | None = None):
        items = Session.list()
        if not project:
            return items
        # chats created before projects existed carry no tag: match them by the folder they ran in
        pdir = str(Path(projects.all().get(project, ".")).resolve().parent).lower()
        return [s for s in items if s.get("project") == project or
                (not s.get("project") and str(s.get("workdir") or "").lower().startswith(pdir))]

    @app.get("/v1/sessions/{session_id}/messages", dependencies=[Depends(auth)])
    def session_messages(session_id: str):
        if not (sessions_dir() / f"{session_id}.jsonl").exists():
            raise HTTPException(404, "session not found")
        return Session(session_id).messages()

    @app.get("/v1/info", dependencies=[Depends(auth)])
    def info(project: str | None = None):
        name, cfg = cfg_for(project)
        sets = []
        for e in cfg.get("toolsets", []) or []:
            try:
                ts = toolset_from_entry(*next(iter(e.items())), cfg.get("_dir"))
                sets.append({"name": ts.name, "tools": [{"name": t.name, "risk": t.risk,
                                                         "description": (t.description or "").split("\n")[0][:160]}
                                                        for t in ts.tools]})
                if ts.teardown:
                    ts.teardown()
            except Exception as ex:  # show misconfigured toolsets instead of failing the page
                sets.append({"name": next(iter(e)), "error": str(ex), "tools": []})
        skills = [s.name for s in load_skills([Path(cfg.get("_dir", ".")) / d for d in cfg.get("skills_dirs") or []])]
        from nexus_agent.llm import KeyPool

        pool = KeyPool.for_model(cfg.get("model") or "")
        return {"project": name, "model": cfg.get("model"), "mode": cfg.get("permissions", "auto"), "toolsets": sets,
                "skills": skills, "keys": pool.stats() if pool else None}

    @app.get("/", include_in_schema=False)
    def index():
        return HTMLResponse((Path(__file__).parent / "web" / "index.html").read_text(encoding="utf-8"))

    return app


def serve(config_path: str | None, host: str = "127.0.0.1", port: int = 8787) -> None:
    import uvicorn

    uvicorn.run(create_app(config_path), host=host, port=port)
