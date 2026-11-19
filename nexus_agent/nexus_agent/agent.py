"""The agent loop.

    model -> (text | tool calls) -> run permitted tools -> feed results back -> repeat

Everything else lives around the loop: toolsets you attach/detach, permissions, budgets
(steps, cost), tool-output truncation, context compaction, sessions, skills, sub-agents,
and an event stream that every interface (CLI, SDK, HTTP, evals) consumes.
"""

from __future__ import annotations

import json
import platform
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator

from nexus_agent.llm import LLM, Completion, QuotaExhausted
from nexus_agent.runtime import (COMPACT_PROMPT, Approver, Mode, Permissions, Session, load_skills,
                                 split_for_compaction, truncate)
from nexus_agent.tools import Tool, Toolset, render_result, tool

BASE_PROMPT = """You are a capable, careful agent. You complete the user's task by calling tools and reasoning about \
their results, then you report back.

How to work:
- Understand the goal, then act. Prefer doing over describing. Use tools to look things up instead of guessing.
- Break big tasks into steps. After each tool result, decide the next step.
- Verify your work when you can (re-read files you wrote, re-run queries, run tests).
- If a tool fails, read the error and try a different approach; do not repeat the same failing call.
- Never invent facts, ids, file contents or query results. If something is not available, say so.
- Ask the user only when you truly cannot proceed (missing credentials, ambiguous intent with real consequences).
- When done, reply with a concise final answer: what you did, the result, and anything left undone.

Environment: {os} · working directory: {cwd} · date: {date}"""


@dataclass
class RunResult:
    text: str
    stop_reason: str  # done | max_steps | max_cost | quota | error | interrupted
    steps: int
    tool_calls: int
    cost_usd: float
    input_tokens: int
    output_tokens: int
    session_id: str | None
    events: list[dict] = field(default_factory=list)
    error: str | None = None


class Agent:
    def __init__(
        self,
        model: str | LLM,
        tools: list[Toolset | Tool] | None = None,
        *,
        system: str | None = None,
        workdir: str | Path = ".",
        permissions: Mode | Permissions = "auto",
        approver: Approver | None = None,
        max_steps: int = 50,
        max_cost_usd: float | None = None,
        max_tool_chars: int = 16_000,
        compact_at: float = 0.75,
        keep_recent: int = 12,
        session: Session | str | bool | None = None,
        skills_dirs: list[str | Path] | None = None,
        instruction_files: tuple[str, ...] = ("AGENTS.md", "CLAUDE.md"),
        on_event: Callable[[dict], None] | None = None,
        llm_params: dict | None = None,
        name: str = "agent",
    ):
        if isinstance(model, LLM):
            self.llm = model
        elif model.startswith("relay:"):
            from nexus_agent.llm import RelayLLM

            self.llm = RelayLLM(model.split(":", 1)[1])
        else:
            self.llm = LLM(model, **(llm_params or {}))
        self.workdir = Path(workdir).resolve()
        self.system_extra = system
        self.permissions = permissions if isinstance(permissions, Permissions) else Permissions(permissions, approver)
        if approver and isinstance(permissions, Permissions) and permissions.approver is None:
            self.permissions.approver = approver
        self.max_steps = max_steps
        self.max_cost_usd = max_cost_usd
        self.max_tool_chars = max_tool_chars
        self.compact_at = compact_at
        self.keep_recent = keep_recent
        self.on_event = on_event
        self.instruction_files = instruction_files
        self.name = name
        self.toolsets: dict[str, Toolset] = {}
        self.skills = load_skills(skills_dirs or [self.workdir / "skills", self.workdir / ".nxa" / "skills"])
        if isinstance(session, Session):
            self.session = session
        elif isinstance(session, str):
            self.session = Session(session)
        elif session:
            self.session = Session(meta={"model": self.llm.model, "workdir": str(self.workdir)})
        else:
            self.session = None
        self.messages: list[dict] = self.session.messages() if self.session else []
        for t in tools or []:
            self.attach(t)
        if self.skills:
            self.attach(self._skills_toolset())

    # ------------------------------------------------------------------ tools
    def attach(self, item: Toolset | Tool) -> "Agent":
        ts = item if isinstance(item, Toolset) else Toolset(name=item.toolset or item.name, tools=[item])
        if ts.name in self.toolsets:
            self.detach(ts.name)
        if ts.setup:
            ts.setup()
        self.toolsets[ts.name] = ts
        self._emit({"type": "toolset_attached", "toolset": ts.name, "tools": [t.name for t in ts.tools]})
        return self

    def detach(self, name: str) -> "Agent":
        ts = self.toolsets.pop(name, None)
        if ts and ts.teardown:
            ts.teardown()
        if ts:
            self._emit({"type": "toolset_detached", "toolset": name})
        return self

    @property
    def tools(self) -> list[Tool]:
        by_name: dict[str, Tool] = {}
        for ts in self.toolsets.values():
            for t in ts.tools:
                by_name[t.name] = t  # later toolsets win on name collisions
        return list(by_name.values())

    def _skills_toolset(self) -> Toolset:
        skills = {s.name: s for s in self.skills}

        @tool
        def load_skill(name: str) -> str:
            """Load the full instructions of a skill listed in the system prompt.

            Args:
                name: Skill name.
            """
            s = skills.get(name)
            return s.body() if s else f"No skill named {name!r}. Available: {', '.join(skills)}"

        return Toolset("skills", [load_skill])

    # ------------------------------------------------------------------ prompt
    def system_prompt(self) -> str:
        parts = [BASE_PROMPT.format(os=f"{platform.system()} {platform.release()}", cwd=self.workdir,
                                    date=time.strftime("%Y-%m-%d"))]
        for ts in self.toolsets.values():
            if ts.prompt:
                parts.append(f"## Toolset: {ts.name}\n{ts.prompt}")
        if self.skills:
            parts.append("## Skills (call load_skill(name) before doing a task a skill covers)\n" +
                         "\n".join(f"- {s.name}: {s.description}" for s in self.skills))
        for fname in self.instruction_files:
            p = self.workdir / fname
            if p.exists():
                parts.append(f"## Project instructions ({fname})\n{p.read_text(encoding='utf-8')[:20000]}")
        if self.system_extra:
            parts.append(self.system_extra)
        return "\n\n".join(parts)

    # ------------------------------------------------------------------ events
    def _emit(self, event: dict) -> dict:
        event.setdefault("ts", time.time())
        if self.on_event:
            self.on_event(event)
        if self.session and event["type"] not in ("text_delta",):
            self.session.add_event({"event": event["type"], **{k: v for k, v in event.items() if k != "type"}})
        return event

    def _add(self, msg: dict) -> None:
        self.messages.append(msg)
        if self.session:
            self.session.add_message(msg)

    # ------------------------------------------------------------------ context
    def _maybe_compact(self) -> dict | None:
        system = [{"role": "system", "content": self.system_prompt()}]
        used = self.llm.count_tokens(system + self.messages)
        if used < self.compact_at * self.llm.context_window:
            return None
        old, recent = split_for_compaction(self.messages, self.keep_recent)
        if not old:
            return None
        summary = self.llm.complete(system + old + [{"role": "user", "content": COMPACT_PROMPT}]).text
        self.messages = [{"role": "user", "content": f"[Summary of earlier conversation]\n{summary}"},
                         {"role": "assistant", "content": "Understood. Continuing from that summary."}] + recent
        ev = self._emit({"type": "compaction", "tokens_before": used, "summarized_messages": len(old)})
        if self.session:
            self.session.add_event({"event": "compaction", "messages": self.messages})
        return ev

    # ------------------------------------------------------------------ run
    def run(self, task: str) -> RunResult:
        events = list(self.stream(task))
        done = events[-1]
        return RunResult(text=done.get("text", ""), stop_reason=done["stop_reason"], steps=done["steps"],
                         tool_calls=done["tool_calls"], cost_usd=self.llm.total_cost,
                         input_tokens=self.llm.total_input_tokens, output_tokens=self.llm.total_output_tokens,
                         session_id=self.session.id if self.session else None, events=events,
                         error=done.get("error"))

    def stream(self, task: str) -> Iterator[dict]:
        """Run one user turn; yields events. Conversation state persists across calls (multi-turn)."""
        self._add({"role": "user", "content": task})
        yield self._emit({"type": "user", "text": task})
        steps = n_calls = 0
        final_text, stop, error = "", "max_steps", None
        cost0 = self.llm.total_cost
        try:
            while steps < self.max_steps:
                steps += 1
                ev = self._maybe_compact()
                if ev:
                    yield ev
                messages = [{"role": "system", "content": self.system_prompt()}] + self.messages
                comp: Completion = self.llm.complete(messages, self.tools)
                self._add(comp.assistant_message())
                yield self._emit({"type": "assistant", "text": comp.text, "step": steps,
                                  "tool_calls": [{"id": c.id, "name": c.name, "arguments": c.arguments}
                                                 for c in comp.tool_calls],
                                  "cost_usd": comp.cost_usd, "tokens": [comp.input_tokens, comp.output_tokens]})
                if not comp.tool_calls:
                    final_text, stop = comp.text, "done"
                    break
                for call in comp.tool_calls:
                    n_calls += 1
                    yield from self._execute(call)
                if self.max_cost_usd is not None and self.llm.total_cost - cost0 > self.max_cost_usd:
                    stop = "max_cost"
                    break
        except QuotaExhausted as e:
            stop, error = "quota", str(e)
        except KeyboardInterrupt:
            stop = "interrupted"
        except Exception as e:  # surface, don't crash the host app
            stop, error = "error", f"{type(e).__name__}: {e}"
        if stop != "done" and not final_text:
            final_text = next((m.get("content") for m in reversed(self.messages)
                               if m["role"] == "assistant" and m.get("content")), "") or ""
        yield self._emit({"type": "done", "stop_reason": stop, "text": final_text, "steps": steps,
                          "tool_calls": n_calls, "cost_usd": round(self.llm.total_cost - cost0, 6), "error": error})

    def _execute(self, call) -> Iterator[dict]:
        t = next((x for x in self.tools if x.name == call.name), None)
        yield self._emit({"type": "tool_call", "id": call.id, "name": call.name, "arguments": call.arguments,
                          "toolset": t.toolset if t else None})
        if t is None:
            out, ok = f"Error: unknown tool '{call.name}'. Available: {', '.join(x.name for x in self.tools)}", False
        else:
            allowed, reason = self.permissions.check(t, call.arguments)
            if not allowed:
                out, ok = f"Permission denied: {reason}. Choose another approach or ask the user.", False
                yield self._emit({"type": "denied", "name": call.name, "reason": reason})
            else:
                t0 = time.time()
                try:
                    out, ok = render_result(t.fn(**call.arguments)), True
                except TypeError as e:
                    out, ok = f"Error: bad arguments for {call.name}: {e}. Schema: {json.dumps(t.parameters)}", False
                except Exception as e:
                    out, ok = f"Error: {type(e).__name__}: {e}", False
                out = truncate(out, self.max_tool_chars)
                yield self._emit({"type": "tool_result", "id": call.id, "name": call.name, "ok": ok,
                                  "chars": len(out), "ms": int((time.time() - t0) * 1000), "preview": out[:300]})
        self._add({"role": "tool", "tool_call_id": call.id, "name": call.name, "content": out})

    # ------------------------------------------------------------------ sub-agents
    def as_tool(self, name: str, description: str) -> Tool:
        """Expose this agent as a tool another agent can delegate to (fresh context per call)."""
        parent_self = self

        def delegate(task: str) -> str:
            child = Agent(parent_self.llm, list(parent_self.toolsets.values()), system=parent_self.system_extra,
                          workdir=parent_self.workdir, permissions=parent_self.permissions,
                          max_steps=parent_self.max_steps, name=name)
            r = child.run(task)
            return f"[{name} finished: {r.stop_reason}, {r.tool_calls} tool calls]\n{r.text}"

        t = tool(delegate, name=name, risk="exec", description=description)
        t.parameters["properties"]["task"]["description"] = "Complete, self-contained task for the sub-agent."
        return t
