"""Tools and toolsets.

A ``Tool`` is a Python callable plus a JSON schema. ``@tool`` builds the schema
from type hints and the docstring (Google/NumPy/reST style ``Args:`` sections).

A ``Toolset`` is a named, attachable bundle of tools with optional setup/teardown
and a prompt note that tells the model how to use the bundle. Agents attach and
detach toolsets at runtime::

    agent.attach(toolsets.sqlite("shop.db"))
    agent.detach("sqlite")

Risk levels drive permissions: ``read`` < ``write`` < ``exec`` < ``destructive``.
"""

from __future__ import annotations

import inspect
import json
import types
import typing
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, get_args, get_origin

from docstring_parser import parse as parse_docstring

Risk = Literal["read", "write", "exec", "destructive"]
RISK_ORDER = {"read": 0, "write": 1, "exec": 2, "destructive": 3}

_PY_TO_JSON = {str: "string", int: "integer", float: "number", bool: "boolean", dict: "object", list: "array"}


def _json_type(tp: Any) -> dict:
    origin = get_origin(tp)
    if tp is inspect.Parameter.empty or tp is Any:
        return {}
    if origin is typing.Union or origin is types.UnionType:
        args = [a for a in get_args(tp) if a is not type(None)]
        return _json_type(args[0]) if len(args) == 1 else {"anyOf": [_json_type(a) for a in args]}
    if origin is Literal:
        vals = list(get_args(tp))
        return {"type": _PY_TO_JSON.get(type(vals[0]), "string"), "enum": vals}
    if origin in (list, tuple, set):
        args = get_args(tp)
        return {"type": "array", "items": (_json_type(args[0]) or {"type": "string"}) if args else {"type": "string"}}
    if origin is dict:
        return {"type": "object"}
    if tp is list:
        return {"type": "array", "items": {"type": "string"}}
    return {"type": _PY_TO_JSON.get(tp, "string")}


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    fn: Callable[..., Any]
    risk: Risk = "read"
    toolset: str = ""
    # Output handling: offload=False keeps the whole result in context (up to max_output chars);
    # offload_at overrides the agent's offload threshold for this tool.
    offload: bool = True
    offload_at: int | None = None
    max_output: int | None = None

    def schema(self) -> dict:
        return {"type": "function",
                "function": {"name": self.name, "description": self.description, "parameters": self.parameters}}

    def __call__(self, **kwargs: Any) -> Any:
        return self.fn(**kwargs)


def tool(fn: Callable | None = None, *, name: str | None = None, risk: Risk = "read",
         description: str | None = None, offload: bool = True, offload_at: int | None = None,
         max_output: int | None = None) -> Any:
    """Decorator: turn a typed, documented function into a ``Tool``.

    Usage: ``@tool`` or ``@tool(risk="write", name="save_note")``.
    """

    def wrap(f: Callable) -> Tool:
        doc = parse_docstring(inspect.getdoc(f) or "")
        hints = typing.get_type_hints(f)
        arg_docs = {p.arg_name: p.description for p in doc.params}
        props, required = {}, []
        for pname, p in inspect.signature(f).parameters.items():
            if pname in ("self", "cls") or p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
                continue
            spec = _json_type(hints.get(pname, p.annotation))
            if arg_docs.get(pname):
                spec["description"] = arg_docs[pname]
            if p.default is inspect.Parameter.empty:
                required.append(pname)
            else:
                if p.default is not None and isinstance(p.default, (str, int, float, bool)):
                    spec["default"] = p.default
            props[pname] = spec
        desc = description or "\n\n".join(x for x in (doc.short_description, doc.long_description) if x)
        return Tool(name=name or f.__name__, description=desc or f.__name__,
                    parameters={"type": "object", "properties": props, "required": required}, fn=f, risk=risk,
                    offload=offload, offload_at=offload_at, max_output=max_output)

    return wrap(fn) if fn is not None else wrap


@dataclass
class Toolset:
    """A named, attachable bundle of tools."""

    name: str
    tools: list[Tool]
    prompt: str = ""  # guidance appended to the system prompt while attached
    setup: Callable[[], None] | None = None
    teardown: Callable[[], None] | None = None
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        for t in self.tools:
            t.toolset = self.name

    def describe(self) -> str:
        return f"{self.name}: " + ", ".join(f"{t.name}[{t.risk}]" for t in self.tools)


def render_result(value: Any) -> str:
    if value is None:
        return "(no output)"
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, indent=1, default=str)
    except TypeError:
        return str(value)
