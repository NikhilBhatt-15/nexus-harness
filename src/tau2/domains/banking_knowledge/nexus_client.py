"""Client for a Nexus-style knowledge layer (nexus-lite, or the Pinecone Nexus v0 API).

Two transports, chosen by ``NEXUS_URL``:

- ``inproc`` or ``inproc:<contexts_root>``: load the nexus-lite engine in this
  process (no server; default root ``contexts``).
- ``http(s)://...``: the v0 HTTP API — ``POST /api/v0/query`` with
  ``background: true``, then poll ``GET /api/v0/queries/{id}`` until terminal.
  Works against ``nexus-lite serve`` and against Pinecone Nexus (auth via
  ``NEXUS_TOKEN`` or ``PINECONE_API_KEY`` -> ``/api/v0/auth/login``).

Other env:
  NEXUS_CONTEXT          context slug (required)
  NEXUS_QUERY_DIRECTIVE  appended to every ask as "[Answer requirements: ...]"
  NEXUS_QUERY_MODEL      model for server-side synthesis (unset = evidence mode on nexus-lite)
  NEXUS_QUERY_DEPTH      quick | standard | deep
  NEXUS_USAGE_LOG        jsonl path; one row per query (task_id, latency, tokens, cost)

Failures are returned as strings so the agent sees them as tool results.
"""

from __future__ import annotations

import json
import os
import threading
import time
from typing import Any, Optional

import httpx

TERMINAL = {"completed", "failed", "cancelled"}
_ENGINES: dict[str, Any] = {}
_ENGINE_LOCK = threading.Lock()
_LOG_LOCK = threading.Lock()


def load_local_context(context: Optional[str] = None):
    """Load (and cache) a local nexus-lite Context for router mode."""
    slug = (context or os.environ.get("NEXUS_CONTEXT", "")).strip()
    if not slug:
        raise NexusClientError("Set NEXUS_CONTEXT (context slug) or pass nexus_context.")
    url = os.environ.get("NEXUS_URL", "inproc")
    root = (url.partition(":")[2] if url.startswith("inproc") else "") or os.environ.get("NEXUS_LITE_ROOT", "contexts")
    key = f"ctx::{root}::{slug}"
    with _ENGINE_LOCK:
        if key not in _ENGINES:
            from nexus_lite.context import Context

            _ENGINES[key] = Context(root, slug)
        return _ENGINES[key]


class NexusClientError(Exception):
    pass


class NexusClient:
    def __init__(self, context: Optional[str] = None, url: Optional[str] = None, timeout_seconds: int = 600):
        self.context = (context or os.environ.get("NEXUS_CONTEXT", "")).strip()
        if not self.context:
            raise NexusClientError("Set NEXUS_CONTEXT (context slug) or pass nexus_context.")
        self.url = (url or os.environ.get("NEXUS_URL") or "inproc").rstrip("/")
        self.timeout_seconds = timeout_seconds
        self._token: Optional[str] = None

    # ---------------------------------------------------------------- transport
    def _engine(self):
        root = self.url.partition(":")[2] or os.environ.get("NEXUS_LITE_ROOT", "contexts")
        with _ENGINE_LOCK:
            if root not in _ENGINES:
                from nexus_lite.query import Engine

                _ENGINES[root] = Engine(root)
            return _ENGINES[root]

    def _login(self) -> str:
        tok = os.environ.get("NEXUS_TOKEN", "").strip()
        if tok:
            return tok
        r = httpx.post(f"{self.url}/api/v0/auth/login",
                       json={"api_key": os.environ.get("PINECONE_API_KEY", "local")}, timeout=30)
        r.raise_for_status()
        return r.json()["token"]

    def _req(self, method: str, path: str, retry: bool = True, **kw) -> httpx.Response:
        if self._token is None:
            self._token = self._login()
        r = httpx.request(method, f"{self.url}{path}", headers={"Authorization": f"Bearer {self._token}"},
                          timeout=60, **kw)
        if r.status_code == 401 and retry:
            self._token = None
            return self._req(method, path, retry=False, **kw)
        return r

    def _run_http(self, payload: dict) -> dict | str:
        r = self._req("POST", "/api/v0/query", json={**payload, "background": True})
        if r.status_code == 404:
            return f"Knowledge service error: context '{self.context}' not found."
        if r.status_code not in (200, 201, 202):
            return f"Knowledge query failed (HTTP {r.status_code}): {r.text[:300]}"
        row = r.json()
        deadline = time.monotonic() + self.timeout_seconds
        while row.get("status") not in TERMINAL:
            if time.monotonic() > deadline:
                return f"Knowledge query timed out after {self.timeout_seconds}s."
            time.sleep(1.0 if self.url.startswith("http://127.") or "localhost" in self.url else 3.0)
            p = self._req("GET", f"/api/v0/queries/{row['id']}")
            if p.status_code != 200:
                return f"Knowledge poll failed (HTTP {p.status_code})."
            row = p.json()
        return row

    # ---------------------------------------------------------------- query
    def query(self, question: str, task_id: Optional[str] = None) -> str:
        directive = os.environ.get("NEXUS_QUERY_DIRECTIVE", "").strip()
        ask = f"{question}\n\n[Answer requirements: {directive}]" if directive else question
        payload: dict[str, Any] = {"scope": [self.context], "ask": ask,
                                   "budget": {"depth": os.environ.get("NEXUS_QUERY_DEPTH", "standard")}}
        model = os.environ.get("NEXUS_QUERY_MODEL", "").strip()
        if model:
            payload["model"] = model
        t0 = time.monotonic()
        try:
            row = self._engine().query(payload) if self.url.startswith("inproc") else self._run_http(payload)
        except Exception as e:  # never crash the simulation
            return f"Knowledge query error: {type(e).__name__}: {e}"
        if isinstance(row, str):
            return row
        self._log(row, question, int((time.monotonic() - t0) * 1000), task_id, model)
        if row.get("status") != "completed":
            return f"Knowledge query {row.get('status')}: {row.get('error') or 'no answer'}"
        return format_result(row)

    def _log(self, row: dict, question: str, wall_ms: int, task_id: Optional[str], model: str) -> None:
        path = os.environ.get("NEXUS_USAGE_LOG")
        if not path:
            return
        usage = row.get("usage") or {}
        rec = {"ts": time.time(), "task_id": task_id, "query_id": row.get("id"), "context": self.context,
               "status": row.get("status"), "model": row.get("model") or model or None, "wall_ms": wall_ms,
               "runtime_ms": row.get("runtime_ms"), "usage": usage, "cost_usd": usage.get("cost_usd"),
               "question_preview": question[:240]}
        with _LOG_LOCK, open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")


def format_result(row: dict) -> str:
    text = "".join(p.get("text", "") for item in row.get("output") or [] for p in item.get("content") or []
                   if p.get("type") == "output_text").strip()
    if not text:
        return "Knowledge service returned no answer text."
    cites = row.get("citations") or []
    if not cites:
        return text
    lines = [text, "", "Citations:"]
    for i, c in enumerate(cites, 1):
        if isinstance(c, dict):
            label = c.get("title") or c.get("name") or c.get("id") or json.dumps(c)[:120]
            lines.append(f"[{i}] {label} ({c['id']})" if c.get("id") and c.get("id") != label else f"[{i}] {label}")
        else:
            lines.append(f"[{i}] {str(c)[:200]}")
    return "\n".join(lines)
