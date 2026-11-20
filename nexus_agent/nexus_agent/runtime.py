"""Runtime services around the loop: permissions, sessions, context management, skills."""

from __future__ import annotations

import json
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

from nexus_agent.tools import RISK_ORDER, Tool

# ---------------------------------------------------------------------------
# Permissions
# ---------------------------------------------------------------------------

Mode = Literal["readonly", "ask", "auto", "yolo"]
Approver = Callable[[Tool, dict], bool]

# Highest risk allowed without asking, per mode. Above it: ask (or deny in readonly).
_AUTO_LIMIT = {"readonly": "read", "ask": "read", "auto": "write", "yolo": "destructive"}


@dataclass
class Permissions:
    mode: Mode = "auto"
    approver: Approver | None = None  # asked when a call exceeds the mode's limit; None = deny
    allow: set[str] = field(default_factory=set)  # tool names always allowed
    deny: set[str] = field(default_factory=set)  # tool names always denied

    def check(self, t: Tool, args: dict) -> tuple[bool, str]:
        if t.name in self.deny or t.toolset in self.deny:
            return False, f"tool '{t.name}' is denied by configuration"
        if t.name in self.allow or t.toolset in self.allow:
            return True, ""
        limit = _AUTO_LIMIT[self.mode]
        risk, reasons = t.risk_for(args)
        if RISK_ORDER[risk] <= RISK_ORDER[limit]:
            return True, ""
        why = f" ({'; '.join(reasons)})" if reasons else ""
        if self.mode == "readonly":
            return False, f"read-only mode: '{t.name}' ({risk}{why}) is not allowed"
        if self.approver is None:
            return False, f"'{t.name}' ({risk}{why}) needs approval and no approver is available"
        return (True, "") if self.approver(t, args) else (False, f"user declined '{t.name}'")


# ---------------------------------------------------------------------------
# Sessions (JSONL; one record per line; resumable and forkable)
# ---------------------------------------------------------------------------

def sessions_dir() -> Path:
    d = Path(os.environ.get("NXA_HOME", Path.home() / ".nexus-agent")) / "sessions"
    d.mkdir(parents=True, exist_ok=True)
    return d


class Session:
    def __init__(self, session_id: str | None = None, *, meta: dict | None = None):
        self.id = session_id or time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        self.path = sessions_dir() / f"{self.id}.jsonl"
        if not self.path.exists():
            self._write({"type": "meta", "ts": time.time(), **(meta or {})})

    def _write(self, rec: dict) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")

    def add_message(self, msg: dict) -> None:
        self._write({"type": "message", "ts": time.time(), "message": msg})

    def add_event(self, event: dict) -> None:
        self._write({"type": "event", "ts": time.time(), **event})

    def records(self) -> list[dict]:
        return [json.loads(line) for line in open(self.path, encoding="utf-8") if line.strip()]

    def messages(self) -> list[dict]:
        """Current conversation (compaction records replace everything before them)."""
        msgs: list[dict] = []
        for r in self.records():
            if r["type"] == "message":
                msgs.append(r["message"])
            elif r["type"] == "event" and r.get("event") == "compaction":
                msgs = list(r["messages"])
        return msgs

    def fork(self, upto_message: int | None = None) -> "Session":
        new = Session(meta={"forked_from": self.id})
        for m in self.messages()[:upto_message]:
            new.add_message(m)
        return new

    @staticmethod
    def list() -> list[dict]:
        out = []
        for p in sorted(sessions_dir().glob("*.jsonl"), reverse=True):
            recs = [json.loads(line) for line in open(p, encoding="utf-8") if line.strip()]
            first = recs[0] if recs else {}
            msgs = [r["message"] for r in recs if r.get("type") == "message"]
            title = first.get("title") or next((m["content"][:70] for m in msgs if m.get("role") == "user"), None)
            out.append({"id": p.stem, "messages": len(msgs), "model": first.get("model"), "title": title})
        return out


# ---------------------------------------------------------------------------
# Context management
# ---------------------------------------------------------------------------

def truncate(text: str, limit: int) -> str:
    """Keep head and tail of long tool output so the model sees both ends."""
    if len(text) <= limit:
        return text
    head = int(limit * 0.7)
    tail = limit - head
    return f"{text[:head]}\n\n... [{len(text) - limit} characters omitted] ...\n\n{text[-tail:]}"


COMPACT_PROMPT = (
    "Summarize the conversation so far for your own future reference. Keep: the user's goal and constraints, "
    "decisions made, facts discovered (names, ids, paths, numbers, exact values), what has been done, what "
    "failed and why, and what remains to do. Be specific and concise. Output only the summary."
)


def split_for_compaction(messages: list[dict], keep_last: int) -> tuple[list[dict], list[dict]]:
    """Split into (old, recent) without separating a tool result from its assistant tool call."""
    cut = max(0, len(messages) - keep_last)
    while cut > 0 and messages[cut]["role"] == "tool":
        cut -= 1
    return messages[:cut], messages[cut:]


# ---------------------------------------------------------------------------
# Skills (SKILL.md with optional YAML frontmatter: name, description)
# ---------------------------------------------------------------------------

@dataclass
class Skill:
    name: str
    description: str
    path: Path

    def body(self) -> str:
        text = self.path.read_text(encoding="utf-8")
        return re.sub(r"^---\n.*?\n---\n", "", text, flags=re.S).strip()


def load_skills(dirs: list[str | Path]) -> list[Skill]:
    skills: list[Skill] = []
    for d in dirs:
        root = Path(d).expanduser()
        if not root.exists():
            continue
        for f in sorted(root.rglob("SKILL.md")):
            text = f.read_text(encoding="utf-8")
            meta: dict[str, Any] = {}
            m = re.match(r"^---\n(.*?)\n---\n", text, re.S)
            if m:
                import yaml

                meta = yaml.safe_load(m.group(1)) or {}
            first = next((ln.strip("# ").strip() for ln in text.splitlines() if ln.strip() and ln.strip() != "---"), "")
            skills.append(Skill(name=str(meta.get("name") or f.parent.name), path=f,
                                description=str(meta.get("description") or first)[:300]))
    return skills
