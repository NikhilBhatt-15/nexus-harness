"""Build an Agent from a YAML config and/or compact CLI tool specs.

agent.yaml::

    model: gemini/gemini-3.1-flash-lite
    llm: {temperature: 0}            # extra litellm params
    permissions: auto                # readonly | ask | auto | yolo
    allow: [db_execute]              # always-allowed tools/toolsets
    deny: [run_command]
    max_steps: 40
    max_cost_usd: 0.50
    system: You are the ops assistant for Acme.
    toolsets:
      - files: {root: .}
      - shell: {}
      - web: {}
      - sqlite: {path: shop.db, name: shop}
      - http_api: {name: crm, base_url: https://api.example.com, headers_from_env: {Authorization: CRM_TOKEN}}
      - knowledge: {slug: handbook, source: ./docs}
      - mcp: {name: github, command: npx, args: ["-y", "@modelcontextprotocol/server-github"]}
      - python: {path: my_tools.py}
      - memory: {}

CLI spec (``-t``), comma separated:
    files | files=DIR | shell | web | memory | sqlite=PATH | sql=URL | kb=SLUG | kb=SLUG@DOCS_DIR |
    python=FILE.py | api=NAME@BASE_URL | mcp=NAME@COMMAND ARGS... | mcp=NAME@http(s)://URL
"""

from __future__ import annotations

import shlex
from pathlib import Path
from typing import Any

import yaml

from nexus_agent import toolsets as T
from nexus_agent.agent import Agent
from nexus_agent.runtime import Permissions
from nexus_agent.tools import Toolset


def toolset_from_entry(kind: str, opts: dict | None) -> Toolset:
    opts = dict(opts or {})
    if kind == "files":
        return T.files(opts.pop("root", "."), **opts)
    if kind == "shell":
        return T.shell(opts.pop("root", "."), **opts)
    if kind == "web":
        return T.web(**opts)
    if kind in ("sqlite", "sql", "db"):
        return T.sql(opts.pop("path", None) or opts.pop("url"), **opts)
    if kind in ("http_api", "api"):
        return T.http_api(opts.pop("base_url"), **opts)
    if kind in ("knowledge", "kb"):
        return T.knowledge(opts.pop("slug"), **opts)
    if kind == "mcp":
        return T.mcp(opts.pop("name"), **opts)
    if kind == "python":
        return T.python_tools(opts.pop("path"), opts.pop("name", None))
    if kind == "memory":
        return T.memory(opts.pop("path", ".nxa/memory.json"))
    raise ValueError(f"unknown toolset kind {kind!r}")


def parse_tool_spec(spec: str, workdir: str = ".") -> list[Toolset]:
    """Paths in specs (files=, sqlite=, python=, kb=SLUG@DOCS) are relative to ``workdir``."""
    out: list[Toolset] = []
    wd = Path(workdir)

    def rel(p: str) -> str:
        return p if not p or Path(p).is_absolute() or "://" in p else str(wd / p)

    for item in [s.strip() for s in spec.split(",") if s.strip()]:
        kind, _, arg = item.partition("=")
        if kind == "files":
            out.append(T.files(rel(arg) if arg else workdir))
        elif kind == "shell":
            out.append(T.shell(workdir))
        elif kind in ("web", "memory"):
            out.append(toolset_from_entry(kind, {}))
        elif kind == "sqlite":
            out.append(T.sql(rel(arg), name=Path(arg).stem.replace("-", "_") or "db"))
        elif kind == "sql":
            out.append(T.sql(arg))
        elif kind == "kb":
            slug, _, src = arg.partition("@")
            out.append(T.knowledge(slug, source=rel(src) if src else None))
        elif kind == "python":
            out.append(T.python_tools(rel(arg)))
        elif kind == "api":
            name, _, url = arg.partition("@")
            out.append(T.http_api(url, name=name))
        elif kind == "mcp":
            name, _, target = arg.partition("@")
            if target.startswith("http"):
                out.append(T.mcp(name, url=target))
            else:
                parts = shlex.split(target, posix=True)
                out.append(T.mcp(name, command=parts[0], args=parts[1:]))
        else:
            raise ValueError(f"unknown tool spec {item!r}")
    return out


def load_config(path: str | Path | None) -> dict:
    if path is None:
        for cand in ("agent.yaml", "agent.yml", ".nxa/agent.yaml"):
            if Path(cand).exists():
                path = cand
                break
    if path is None:
        return {}
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def build_agent(cfg: dict, *, model: str | None = None, tool_spec: str | None = None, workdir: str = ".",
                mode: str | None = None, approver=None, on_event=None, session: Any = None,
                max_steps: int | None = None) -> Agent:
    toolsets: list[Toolset] = []
    for entry in cfg.get("toolsets", []) or []:
        (kind, opts), = entry.items() if isinstance(entry, dict) else ((entry, {}),)
        toolsets.append(toolset_from_entry(kind, opts))
    if tool_spec:
        toolsets += parse_tool_spec(tool_spec, workdir)
    perms = Permissions(mode or cfg.get("permissions", "auto"), approver,
                        allow=set(cfg.get("allow", [])), deny=set(cfg.get("deny", [])))
    model = model or cfg.get("model")
    if not model:
        raise SystemExit("No model: pass -m MODEL or set `model:` in agent.yaml")
    return Agent(model, toolsets, system=cfg.get("system"), workdir=cfg.get("workdir", workdir), permissions=perms,
                 max_steps=max_steps or cfg.get("max_steps", 50), max_cost_usd=cfg.get("max_cost_usd"),
                 llm_params=cfg.get("llm"), on_event=on_event, session=session,
                 skills_dirs=cfg.get("skills_dirs"))
