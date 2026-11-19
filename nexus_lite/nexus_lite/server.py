"""HTTP API mirroring the Nexus v0 query surface used by the tau2 client.

    POST /api/v0/auth/login        {"api_key"} -> {"token"}   (local: any key)
    POST /api/v0/query             {scope, ask, background?, model?, shape?, budget?} -> query doc
    GET  /api/v0/queries/{id}      -> query doc (status: queued|running|completed|failed)
    GET  /api/v0/contexts          -> compiled contexts + meta
    GET  /api/v0/contexts/{slug}/manifest
"""

from __future__ import annotations

import json
import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import FastAPI, HTTPException

from nexus_lite.query import Engine


def create_app(root: str, workers: int = 8) -> FastAPI:
    app = FastAPI(title="nexus-lite")
    engine = Engine(root)
    pool = ThreadPoolExecutor(max_workers=workers)
    queries: dict[str, dict] = {}

    def run(qid: str, req: dict) -> None:
        queries[qid]["status"] = "running"
        try:
            queries[qid] = engine.query({**req, "id": qid})
        except Exception as e:  # surfaced to the agent as a tool-result string
            queries[qid] = {"id": qid, "status": "failed", "error": f"{type(e).__name__}: {e}"}

    @app.post("/api/v0/auth/login")
    def login(body: dict):
        return {"token": "local-" + uuid.uuid4().hex}

    @app.get("/api/v0/contexts")
    def contexts():
        out = []
        for meta in Path(root).glob("*/meta.json"):
            m = json.loads(meta.read_text())
            out.append({k: m.get(k) for k in ("slug", "embedder", "last_curated_at", "counts")})
        return out

    @app.get("/api/v0/contexts/{slug}/manifest")
    def manifest(slug: str):
        p = Path(root) / slug / "meta.json"
        if not p.exists():
            raise HTTPException(404, "context not found")
        return json.loads(p.read_text()).get("manifest", {})

    @app.post("/api/v0/query", status_code=202)
    def query(req: dict):
        for slug in req.get("scope") or []:
            if not (Path(root) / slug / "context.sqlite").exists():
                raise HTTPException(404, f"context {slug!r} not found")
        qid = f"q_{uuid.uuid4().hex[:12]}"
        if not req.get("background", False):
            return engine.query({**req, "id": qid})
        queries[qid] = {"id": qid, "status": "queued"}
        pool.submit(run, qid, req)
        return queries[qid]

    @app.get("/api/v0/queries/{qid}")
    def get_query(qid: str):
        if qid not in queries:
            raise HTTPException(404, "query not found")
        return queries[qid]

    return app


def serve(root: str, host: str = "127.0.0.1", port: int = 8765) -> None:
    import uvicorn

    uvicorn.run(create_app(root, int(os.environ.get("NEXUS_LITE_WORKERS", "8"))), host=host, port=port)
