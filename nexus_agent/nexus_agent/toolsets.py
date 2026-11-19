"""Built-in toolsets. Each factory returns a ``Toolset`` you can attach/detach.

    files(root)                 read/write/edit/ls/find/grep, jailed to root
    shell(root)                 run commands (bash, powershell or cmd), timeout, truncated output
    web()                       fetch a URL as text
    sqlite(path) / sql(url)     schema, read queries, writes (destructive statements gated separately)
    http_api(base_url, ...)     call a REST API; auth headers come from env vars and are never shown to the model
    knowledge(slug, ...)        nexus-lite compiled knowledge: cited answers, SQL over artifacts, sources
    mcp(command=... | url=...)  every tool of an MCP server
    python_tools(path)          your own @tool functions from a .py file (plugins)
    memory(path)                persistent notes across sessions
    subagent(model, toolsets)   delegate a self-contained task to a fresh agent
"""

from __future__ import annotations

import fnmatch
import html
import importlib.util
import json
import os
import re
import shutil
import sqlite3
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Literal

import httpx

from nexus_agent.tools import Tool, Toolset, tool

# ---------------------------------------------------------------------------
# files
# ---------------------------------------------------------------------------


def files(root: str | Path = ".", *, allow_outside: bool = False, writable: bool = True) -> Toolset:
    base = Path(root).resolve()

    def resolve(p: str) -> Path:
        path = (base / p).resolve() if not Path(p).is_absolute() else Path(p).resolve()
        if not allow_outside and base not in (path, *path.parents):
            raise PermissionError(f"{p} is outside the workspace {base}")
        return path

    @tool
    def read_file(path: str, offset: int = 1, limit: int = 400) -> str:
        """Read a text file with line numbers.

        Args:
            path: File path (relative to the workspace).
            offset: First line to read (1-based).
            limit: Maximum number of lines.
        """
        lines = resolve(path).read_text(encoding="utf-8", errors="replace").splitlines()
        chunk = lines[offset - 1: offset - 1 + limit]
        more = f"\n... ({len(lines) - (offset - 1 + len(chunk))} more lines)" if offset - 1 + limit < len(lines) else ""
        return "\n".join(f"{i:>5}  {line}" for i, line in enumerate(chunk, offset)) + more

    @tool(risk="write")
    def write_file(path: str, content: str) -> str:
        """Create or overwrite a text file.

        Args:
            path: File path (relative to the workspace).
            content: Full file content.
        """
        p = resolve(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"wrote {len(content)} chars to {p.relative_to(base) if base in p.parents else p}"

    @tool(risk="write")
    def edit_file(path: str, old_text: str, new_text: str, replace_all: bool = False) -> str:
        """Replace exact text in a file. old_text must match exactly (including whitespace) and be unique
        unless replace_all is true.

        Args:
            path: File path.
            old_text: Exact text to find.
            new_text: Replacement text.
            replace_all: Replace every occurrence.
        """
        p = resolve(path)
        text = p.read_text(encoding="utf-8")
        n = text.count(old_text)
        if n == 0:
            return "Error: old_text not found. Read the file and copy the exact text."
        if n > 1 and not replace_all:
            return f"Error: old_text occurs {n} times; add surrounding context or set replace_all."
        p.write_text(text.replace(old_text, new_text) if replace_all else text.replace(old_text, new_text, 1),
                     encoding="utf-8")
        return f"replaced {n if replace_all else 1} occurrence(s) in {path}"

    @tool
    def list_dir(path: str = ".") -> str:
        """List a directory (directories end with /).

        Args:
            path: Directory path.
        """
        d = resolve(path)
        items = sorted(d.iterdir(), key=lambda x: (x.is_file(), x.name.lower()))
        return "\n".join(f"{x.name}/" if x.is_dir() else f"{x.name}  ({x.stat().st_size} B)" for x in items[:500]) \
            or "(empty)"

    @tool
    def find_files(pattern: str, path: str = ".") -> str:
        """Find files by glob pattern, e.g. "**/*.py".

        Args:
            pattern: Glob pattern.
            path: Directory to search.
        """
        d = resolve(path)
        skip = {".git", ".venv", "node_modules", "__pycache__"}
        hits = [str(p.relative_to(d)) for p in d.glob(pattern) if not skip & set(p.parts)]
        return "\n".join(sorted(hits)[:500]) or "(no matches)"

    @tool
    def grep(pattern: str, path: str = ".", glob: str = "*", ignore_case: bool = False) -> str:
        """Search file contents with a regular expression. Returns path:line: text.

        Args:
            pattern: Regular expression.
            path: File or directory.
            glob: Only files matching this glob (e.g. "*.py").
            ignore_case: Case-insensitive search.
        """
        rx = re.compile(pattern, re.I if ignore_case else 0)
        d = resolve(path)
        targets = [d] if d.is_file() else [p for p in d.rglob("*") if p.is_file()]
        skip = {".git", ".venv", "node_modules", "__pycache__"}
        out: list[str] = []
        for f in targets:
            if skip & set(f.parts) or not fnmatch.fnmatch(f.name, glob) or f.stat().st_size > 2_000_000:
                continue
            try:
                for i, line in enumerate(f.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
                    if rx.search(line):
                        out.append(f"{f.relative_to(base) if base in f.parents else f}:{i}: {line.strip()[:240]}")
                        if len(out) >= 300:
                            return "\n".join(out) + "\n... (truncated at 300 matches)"
            except OSError:
                continue
        return "\n".join(out) or "(no matches)"

    tools = [read_file, list_dir, find_files, grep] + ([write_file, edit_file] if writable else [])
    return Toolset("files", tools, prompt=f"File tools work inside {base}. Read a file before editing it; "
                                          "edit_file needs the exact existing text.")


# ---------------------------------------------------------------------------
# shell
# ---------------------------------------------------------------------------


def shell(root: str | Path = ".", *, kind: Literal["auto", "bash", "powershell", "cmd"] = "auto",
          timeout_s: int = 120) -> Toolset:
    base = Path(root).resolve()
    if kind == "auto":
        kind = "bash" if shutil.which("bash") else ("powershell" if os.name == "nt" else "bash")
    argv = {"bash": ["bash", "-lc"], "powershell": ["powershell", "-NoProfile", "-Command"],
            "cmd": ["cmd", "/c"]}[kind]

    @tool(risk="exec")
    def run_command(command: str, timeout: int = timeout_s) -> str:
        """Run a shell command in the workspace and return exit code, stdout and stderr.

        Args:
            command: The command line.
            timeout: Seconds before the command is killed.
        """
        try:
            p = subprocess.run(argv + [command], cwd=base, capture_output=True, text=True, timeout=timeout,
                               encoding="utf-8", errors="replace")
        except subprocess.TimeoutExpired:
            return f"Error: timed out after {timeout}s"
        out = f"exit code: {p.returncode}\n"
        if p.stdout:
            out += f"--- stdout ---\n{p.stdout}"
        if p.stderr:
            out += f"\n--- stderr ---\n{p.stderr}"
        return out

    return Toolset("shell", [run_command],
                   prompt=f"run_command uses {kind} in {base}. Prefer non-interactive commands; never start "
                          "long-running servers in the foreground.")


# ---------------------------------------------------------------------------
# web
# ---------------------------------------------------------------------------


def _html_to_text(raw: str) -> str:
    raw = re.sub(r"(?is)<(script|style|noscript|svg|head).*?</\1>", " ", raw)
    raw = re.sub(r"(?i)<br\s*/?>|</(p|div|li|h[1-6]|tr)>", "\n", raw)
    raw = re.sub(r"(?s)<[^>]+>", " ", raw)
    raw = html.unescape(raw)
    raw = re.sub(r"[ \t]+", " ", raw)
    return re.sub(r"\n\s*\n+", "\n\n", raw).strip()


def web(*, timeout_s: int = 30) -> Toolset:
    @tool
    def fetch_url(url: str, max_chars: int = 20000) -> str:
        """Fetch a web page or API URL (GET) and return its text.

        Args:
            url: http(s) URL.
            max_chars: Maximum characters returned.
        """
        r = httpx.get(url, timeout=timeout_s, follow_redirects=True, headers={"User-Agent": "nexus-agent/0.1"})
        ctype = r.headers.get("content-type", "")
        text = _html_to_text(r.text) if "html" in ctype else r.text
        return f"HTTP {r.status_code} {ctype}\n\n{text[:max_chars]}"

    return Toolset("web", [fetch_url])


# ---------------------------------------------------------------------------
# databases
# ---------------------------------------------------------------------------

_READ_SQL = re.compile(r"^\s*(select|with|pragma|explain|show|describe)\b", re.I)
_DESTRUCTIVE_SQL = re.compile(r"^\s*(drop|truncate|alter)\b|^\s*delete\s+from\s+\S+\s*;?\s*$", re.I)


class _DB:
    """Tiny adapter over sqlite3 or SQLAlchemy (for postgres/mysql/...)."""

    def __init__(self, url: str):
        self.url = url
        self.lock = threading.Lock()
        if url.startswith("sqlite:///") or not re.match(r"^\w+(\+\w+)?://", url):
            path = url.removeprefix("sqlite:///")
            self.kind, self.conn = "sqlite", sqlite3.connect(path, check_same_thread=False)
        else:
            from sqlalchemy import create_engine  # optional dependency

            self.kind, self.engine = "sqlalchemy", create_engine(url)

    def run(self, sql: str, params: list | dict | None = None, limit: int = 200) -> tuple[list[str], list[list], int]:
        with self.lock:
            if self.kind == "sqlite":
                cur = self.conn.execute(sql, params or [])
                cols = [d[0] for d in cur.description] if cur.description else []
                rows = [list(r) for r in cur.fetchmany(limit + 1)] if cols else []
                self.conn.commit()
                return cols, rows, cur.rowcount
            from sqlalchemy import text

            with self.engine.begin() as c:
                res = c.execute(text(sql), params or {})
                if res.returns_rows:
                    return list(res.keys()), [list(r) for r in res.fetchmany(limit + 1)], -1
                return [], [], res.rowcount

    def close(self) -> None:
        if self.kind == "sqlite":
            self.conn.close()
        else:
            self.engine.dispose()

    def schema(self) -> str:
        if self.kind == "sqlite":
            rows = self.conn.execute("SELECT name, sql FROM sqlite_master WHERE type IN ('table','view') "
                                     "AND name NOT LIKE 'sqlite_%'").fetchall()
            out = []
            for name, ddl in rows:
                n = self.conn.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()[0]
                out.append(f"{ddl};  -- {n} rows")
            return "\n".join(out) or "(no tables)"
        from sqlalchemy import inspect as sa_inspect

        insp = sa_inspect(self.engine)
        out = []
        for t in insp.get_table_names():
            cols = ", ".join(f"{c['name']} {c['type']}" for c in insp.get_columns(t))
            out.append(f"{t}({cols})")
        return "\n".join(out) or "(no tables)"


def _fmt_rows(cols: list[str], rows: list[list], limit: int) -> str:
    if not cols:
        return "(no rows)"
    more = len(rows) > limit
    rows = rows[:limit]
    lines = [" | ".join(cols)] + [" | ".join("" if v is None else str(v) for v in r) for r in rows]
    return "\n".join(lines) + (f"\n... (more than {limit} rows; refine the query)" if more else f"\n({len(rows)} rows)")


def sql(url: str, *, name: str = "db", read_only: bool = False, row_limit: int = 200) -> Toolset:
    """Any SQL database. ``url``: a sqlite file path, ``sqlite:///path``, or a SQLAlchemy URL."""
    db = _DB(url)

    @tool(name=f"{name}_schema")
    def db_schema() -> str:
        """Show the database tables, columns and row counts."""
        return db.schema()

    @tool(name=f"{name}_query")
    def db_query(query: str, params: list[str | int | float] | None = None) -> str:
        """Run a read-only SQL query (SELECT/WITH/PRAGMA/EXPLAIN) and return rows.

        Args:
            query: SQL query.
            params: Optional positional parameters for ? placeholders.
        """
        if not _READ_SQL.match(query):
            return f"Error: {name}_query is read-only; use {name}_execute for writes."
        cols, rows, _ = db.run(query, params, row_limit)
        return _fmt_rows(cols, rows, row_limit)

    @tool(name=f"{name}_execute", risk="write")
    def db_execute(statement: str, params: list[str | int | float] | None = None) -> str:
        """Run a write statement (INSERT/UPDATE/DELETE with WHERE/CREATE). Schema-destroying statements
        (DROP/TRUNCATE/ALTER, DELETE without WHERE) must use the admin tool.

        Args:
            statement: SQL statement.
            params: Optional positional parameters.
        """
        if _DESTRUCTIVE_SQL.match(statement):
            return f"Error: destructive statement; use {name}_admin (requires approval)."
        _, _, n = db.run(statement, params)
        return f"ok ({n} rows affected)"

    @tool(name=f"{name}_admin", risk="destructive")
    def db_admin(statement: str) -> str:
        """Run a destructive statement (DROP/TRUNCATE/ALTER/DELETE without WHERE). Always needs approval.

        Args:
            statement: SQL statement.
        """
        _, _, n = db.run(statement)
        return f"ok ({n} rows affected)"

    tools = [db_schema, db_query] + ([] if read_only else [db_execute, db_admin])
    return Toolset(name, tools, teardown=db.close, prompt=f"Database `{name}` ({db.kind}{', read-only' if read_only else ''}). "
                                       f"Call {name}_schema first; write SQL against the real column names.",
                   meta={"url": url})


def sqlite(path: str, **kw: Any) -> Toolset:
    return sql(path, **kw)


# ---------------------------------------------------------------------------
# HTTP APIs
# ---------------------------------------------------------------------------


def http_api(base_url: str, *, name: str = "api", headers_from_env: dict[str, str] | None = None,
             headers: dict[str, str] | None = None, description: str = "", allow_write: bool = True,
             timeout_s: int = 60) -> Toolset:
    """A REST API. Secrets: ``headers_from_env={"Authorization": "MY_API_TOKEN"}`` reads the env var at call
    time; the value never enters the model's context."""
    base = base_url.rstrip("/")

    def _headers() -> dict:
        h = dict(headers or {})
        for k, env in (headers_from_env or {}).items():
            if os.environ.get(env):
                h[k] = os.environ[env]
        return h

    def _call(method: str, path: str, query: dict | None, body: Any) -> str:
        if re.match(r"^\w+://", path):
            return "Error: pass a path relative to the API base URL, not a full URL."
        r = httpx.request(method, f"{base}/{path.lstrip('/')}", params=query, json=body, headers=_headers(),
                          timeout=timeout_s)
        try:
            payload = json.dumps(r.json(), indent=1, ensure_ascii=False)
        except ValueError:
            payload = r.text
        return f"HTTP {r.status_code}\n{payload[:20000]}"

    @tool(name=f"{name}_get")
    def api_get(path: str, query: dict | None = None) -> str:
        """GET a resource from the API.

        Args:
            path: Path relative to the base URL, e.g. "/orders/42".
            query: Query parameters.
        """
        return _call("GET", path, query, None)

    @tool(name=f"{name}_call", risk="write")
    def api_call(method: Literal["POST", "PUT", "PATCH", "DELETE"], path: str, body: dict | None = None,
                 query: dict | None = None) -> str:
        """Call the API with a write method.

        Args:
            method: HTTP method.
            path: Path relative to the base URL.
            body: JSON body.
            query: Query parameters.
        """
        return _call(method, path, query, body)

    return Toolset(name, [api_get] + ([api_call] if allow_write else []),
                   prompt=f"API `{name}` at {base}. {description}".strip())


# ---------------------------------------------------------------------------
# knowledge (nexus-lite)
# ---------------------------------------------------------------------------


def knowledge(slug: str, *, root: str = "contexts", source: str | None = None, model: str | None = None,
              name: str = "kb") -> Toolset:
    """Compiled knowledge over a docs folder. If ``source`` is given and the context does not exist yet,
    it is compiled on attach (``nexus-lite build``)."""
    from nexus_lite.query import Engine

    state: dict[str, Any] = {}

    def setup() -> None:
        if source and not (Path(root) / slug / "context.sqlite").exists():
            from nexus_lite.context import compile_context
            from nexus_lite.corpus import load_corpus

            compile_context(root, slug, load_corpus(source), log=lambda *_: None)
        state["engine"] = Engine(root, default_model=model)

    def engine() -> Engine:
        if "engine" not in state:
            setup()
        return state["engine"]

    @tool(name=f"{name}_query")
    def kb_query(question: str, depth: Literal["quick", "standard", "deep"] = "standard") -> str:
        """Ask the knowledge base. Returns compiled, cited knowledge: fact sheets, comparison rosters,
        procedures with exact tool names, and relevant passages. Ask one specific question per call.

        Args:
            question: A specific, self-contained question.
            depth: How much to retrieve.
        """
        r = engine().query({"scope": [slug], "ask": question, "budget": {"depth": depth}})
        if r.get("status") != "completed":
            return f"Knowledge query failed: {r.get('error')}"
        text = r["output"][0]["content"][0]["text"]
        cites = "\n".join(f"[{i}] {c['title']} ({c['id']})" for i, c in enumerate(r.get("citations", []), 1))
        return f"{text}\n\nCitations:\n{cites}" if cites else text

    @tool(name=f"{name}_sql")
    def kb_sql(query: str) -> str:
        """Read-only SQL over the compiled knowledge tables: entities, facts(kind,text), procedures(procedure,
        step_ord,text,tool_name), tools(tool_name,arg_hints), enums, artifacts(name,body), documents.

        Args:
            query: SELECT statement.
        """
        ctx = engine().context(slug)
        rows = ctx.sql(query)
        return "\n".join(json.dumps(r, ensure_ascii=False) for r in rows[:100]) or "(no rows)"

    @tool(name=f"{name}_read_source")
    def kb_read_source(doc_id: str) -> str:
        """Read the full source document behind a citation.

        Args:
            doc_id: Document id from a citation.
        """
        return engine().context(slug).read_source(doc_id) or f"No document {doc_id!r}"

    return Toolset(name, [kb_query, kb_sql, kb_read_source], setup=setup,
                   prompt=f"Knowledge base `{slug}`: use {name}_query for policies, products, procedures and "
                          "facts before answering from memory; cite sources. Use the exact names and values it returns.")


# ---------------------------------------------------------------------------
# MCP servers
# ---------------------------------------------------------------------------


class _MCPClient:
    """Runs an MCP client session on a background asyncio loop so tools can be called synchronously."""

    def __init__(self, command: str | None, args: list[str] | None, env: dict | None, url: str | None,
                 headers: dict | None, timeout_s: float):
        import asyncio

        self.timeout_s = timeout_s
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, daemon=True)
        self.thread.start()
        self._closed = asyncio.Event()
        self._ready: Any = None
        self.session = None
        self.tools: list = []
        self._task = asyncio.run_coroutine_threadsafe(self._main(command, args, env, url, headers), self.loop)
        deadline = time.time() + timeout_s
        while self._ready is None and time.time() < deadline:
            if self._task.done():
                self._task.result()  # raises the connection error
            time.sleep(0.05)
        if self._ready is None:
            raise TimeoutError("MCP server did not initialize in time")

    async def _main(self, command, args, env, url, headers) -> None:
        from contextlib import AsyncExitStack

        from mcp import ClientSession, StdioServerParameters

        async with AsyncExitStack() as stack:
            if url:
                from mcp.client.streamable_http import streamable_http_client

                streams = await stack.enter_async_context(streamable_http_client(url))
            else:
                from mcp.client.stdio import stdio_client

                params = StdioServerParameters(command=command, args=args or [], env={**os.environ, **(env or {})})
                streams = await stack.enter_async_context(stdio_client(params))
            session = await stack.enter_async_context(ClientSession(streams[0], streams[1]))
            await session.initialize()
            self.tools = (await session.list_tools()).tools
            self.session = session
            self._ready = True
            await self._closed.wait()

    def call(self, name: str, arguments: dict) -> str:
        import asyncio

        fut = asyncio.run_coroutine_threadsafe(self.session.call_tool(name, arguments), self.loop)
        res = fut.result(timeout=self.timeout_s)
        parts = []
        for c in getattr(res, "content", []) or []:
            parts.append(getattr(c, "text", None) or json.dumps(getattr(c, "model_dump", lambda: str(c))(),
                                                                 default=str))
        structured = getattr(res, "structured_content", None) or getattr(res, "structuredContent", None)
        if structured and not parts:
            parts.append(json.dumps(structured, indent=1, default=str))
        text = "\n".join(parts) or "(no content)"
        is_error = getattr(res, "is_error", None) or getattr(res, "isError", False)
        return f"Error from MCP tool: {text}" if is_error else text

    def close(self) -> None:
        self.loop.call_soon_threadsafe(self._closed.set)


def mcp(name: str, *, command: str | None = None, args: list[str] | None = None, env: dict | None = None,
        url: str | None = None, headers: dict | None = None, risk: str = "exec", timeout_s: float = 60,
        include: list[str] | None = None) -> Toolset:
    """All tools of an MCP server (stdio ``command``/``args`` or streamable-HTTP ``url``).
    Tools are prefixed ``<name>__`` and default to risk ``exec`` (ask before use in auto mode)."""
    client = _MCPClient(command, args, env, url, headers, timeout_s)
    tools = []
    for t in client.tools:
        if include and t.name not in include:
            continue
        raw = getattr(t, "input_schema", None) or getattr(t, "inputSchema", None) or {}
        schema = dict(raw) if isinstance(raw, dict) else raw.model_dump()
        schema.setdefault("type", "object")
        schema.setdefault("properties", {})

        def make(tool_name: str):
            return lambda **kw: client.call(tool_name, kw)

        tools.append(Tool(name=f"{name}__{t.name}", description=t.description or t.name, parameters=schema,
                          fn=make(t.name), risk=risk))  # type: ignore[arg-type]
    return Toolset(name, tools, teardown=client.close, meta={"mcp": url or command})


# ---------------------------------------------------------------------------
# plugins, memory, sub-agents
# ---------------------------------------------------------------------------


def python_tools(path: str | Path, name: str | None = None) -> Toolset:
    """Load every ``@tool`` (or ``Tool``) defined at module level in a Python file."""
    path = Path(path).resolve()
    spec = importlib.util.spec_from_file_location(f"nxa_plugin_{path.stem}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    found = [v for v in vars(mod).values() if isinstance(v, Tool)]
    prompt = (mod.__doc__ or "").strip()
    return Toolset(name or path.stem, found, prompt=prompt)


def memory(path: str | Path = ".nxa/memory.json") -> Toolset:
    p = Path(path)

    def load() -> dict:
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}

    @tool(risk="write")
    def remember(key: str, value: str) -> str:
        """Save a note that persists across sessions (overwrites the same key).

        Args:
            key: Short identifier.
            value: What to remember.
        """
        d = load()
        d[key] = value
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(d, indent=1, ensure_ascii=False), encoding="utf-8")
        return f"remembered {key}"

    @tool
    def recall(query: str = "") -> str:
        """List saved notes, optionally filtered by a substring.

        Args:
            query: Substring filter on keys and values.
        """
        d = load()
        hits = {k: v for k, v in d.items() if query.lower() in f"{k} {v}".lower()}
        return json.dumps(hits, indent=1, ensure_ascii=False) if hits else "(no notes)"

    mem = load()
    note = ("Saved notes: " + "; ".join(f"{k}: {v}" for k, v in list(mem.items())[:20])) if mem else ""
    return Toolset("memory", [remember, recall], prompt=note)


def subagent(model: str, toolsets: list[Toolset], *, name: str = "delegate", description: str | None = None,
             max_steps: int = 30, permissions: str = "auto") -> Toolset:
    """A tool that hands a self-contained task to a fresh agent with its own context and tools."""
    from nexus_agent.agent import Agent

    def run_sub(task: str) -> str:
        r = Agent(model, list(toolsets), max_steps=max_steps, permissions=permissions, name=name).run(task)  # type: ignore[arg-type]
        return f"[{name}: {r.stop_reason}, {r.tool_calls} tool calls, ${r.cost_usd:.4f}]\n{r.text}"

    t = tool(run_sub, name=name, risk="exec",
             description=description or "Delegate a self-contained sub-task to a helper agent with its own context. "
                                        "Give full instructions; it returns a summary.")
    return Toolset(f"subagent:{name}", [t])


REGISTRY = {"files": files, "shell": shell, "web": web, "sqlite": sqlite, "sql": sql, "http_api": http_api,
            "knowledge": knowledge, "mcp": mcp, "python": python_tools, "memory": memory}
