"""Any-LLM client.

- Backed by litellm, so any provider/model string works: ``gpt-5.2``, ``anthropic/claude-...``,
  ``gemini/gemini-3.1-flash-lite``, ``openrouter/...``, ``groq/...``, ``ollama_chat/qwen3:4b``.
- Tool calling: ``native`` (provider function calling) or ``text`` (a JSON-in-text protocol for
  models without function calling). ``auto`` uses native when litellm says the model supports it,
  and falls back to text if the provider rejects tools.
- Retries with backoff on rate limits / overload / timeouts; honours "retry in Ns" hints and gives
  up with ``QuotaExhausted`` when the wait is longer than ``max_wait_s`` (e.g. daily free-tier caps).
- Messages are OpenAI-style chat dicts.
"""

from __future__ import annotations

import json
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

from nexus_agent.tools import Tool

ToolMode = Literal["auto", "native", "text"]


class QuotaExhausted(RuntimeError):
    """Provider quota/rate limit that will not clear within the allowed wait."""


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class Completion:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    model: str = ""

    def assistant_message(self) -> dict:
        msg: dict[str, Any] = {"role": "assistant", "content": self.text or None}
        if self.tool_calls:
            msg["tool_calls"] = [{"id": c.id, "type": "function",
                                  "function": {"name": c.name, "arguments": json.dumps(c.arguments)}}
                                 for c in self.tool_calls]
        return msg


TEXT_PROTOCOL = """
# Calling tools
To call a tool, reply with one or more blocks exactly like this (JSON inside, nothing else in the block):

```tool
{"name": "<tool name>", "arguments": {<arguments as JSON>}}
```

You may write a short sentence before the blocks. After you call tools, wait for TOOL RESULT messages.
When you are completely done, reply with your final answer and NO tool block.

# Available tools
{tools}
""".strip()

_RETRYABLE = ("RateLimitError", "ServiceUnavailableError", "Timeout", "APIConnectionError",
              "InternalServerError", "APIError", "BadGatewayError")


def _retry_after_seconds(err: Exception) -> float | None:
    m = re.search(r"retry in (?:(\d+)h)?(?:(\d+)m)?([\d.]+)s", str(err))
    if m:
        h, mi, s = m.groups()
        return int(h or 0) * 3600 + int(mi or 0) * 60 + float(s)
    m = re.search(r"retry[- ]after[\"': ]+([\d.]+)", str(err), re.I)
    return float(m.group(1)) if m else None


class LLM:
    def __init__(self, model: str, *, tool_mode: ToolMode = "auto", max_retries: int = 6,
                 max_wait_s: float = 120.0, **params: Any):
        self.model = model
        self.params = params  # temperature, reasoning_effort, api_base, ...
        self.max_retries = max_retries
        self.max_wait_s = max_wait_s
        self.tool_mode: ToolMode = tool_mode
        if tool_mode == "auto":
            self.tool_mode = "native" if self._supports_tools() else "text"
        self.total_cost = 0.0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.calls = 0

    def _supports_tools(self) -> bool:
        try:
            import litellm

            return bool(litellm.supports_function_calling(model=self.model))
        except Exception:
            # Unknown to litellm's table: hosted models almost always support tools; local ollama
            # models often do via ollama_chat. Try native first; complete() falls back on error.
            return True

    @property
    def context_window(self) -> int:
        try:
            import litellm

            info = litellm.get_model_info(self.model)
            return int(info.get("max_input_tokens") or info.get("max_tokens") or 128_000)
        except Exception:
            return int(os.environ.get("NXA_CONTEXT_WINDOW", "128000"))

    def count_tokens(self, messages: list[dict]) -> int:
        try:
            import litellm

            return int(litellm.token_counter(model=self.model, messages=messages))
        except Exception:
            return sum(len(json.dumps(m, default=str)) for m in messages) // 4

    # ------------------------------------------------------------------ complete
    def complete(self, messages: list[dict], tools: list[Tool] | None = None) -> Completion:
        tools = tools or []
        if self.tool_mode == "text" or not tools:
            return self._call(self._to_text_protocol(messages, tools) if tools else messages, None, text=bool(tools))
        try:
            return self._call(messages, tools, text=False)
        except QuotaExhausted:
            raise
        except Exception as e:  # provider rejected native tools -> switch to text protocol for good
            if re.search(r"tool|function", str(e), re.I) and not re.search(r"rate|quota|429", str(e), re.I):
                self.tool_mode = "text"
                return self._call(self._to_text_protocol(messages, tools), None, text=True)
            raise

    def _call(self, messages: list[dict], tools: list[Tool] | None, *, text: bool) -> Completion:
        import litellm

        kwargs = dict(self.params)
        if tools:
            kwargs["tools"] = [t.schema() for t in tools]
        delay = 2.0
        for attempt in range(self.max_retries + 1):
            try:
                resp = litellm.completion(model=self.model, messages=messages, **kwargs)
                break
            except Exception as e:
                name = type(e).__name__
                if not any(r in name for r in _RETRYABLE) or attempt == self.max_retries:
                    raise
                wait = _retry_after_seconds(e) or delay
                if wait > self.max_wait_s:
                    raise QuotaExhausted(f"{self.model}: {str(e).splitlines()[0][:300]} (retry after {wait:.0f}s)") from e
                time.sleep(wait)
                delay = min(delay * 2, 60)
        self.calls += 1
        msg = resp.choices[0].message
        u = getattr(resp, "usage", None)
        comp = Completion(model=self.model, input_tokens=getattr(u, "prompt_tokens", 0) or 0,
                          output_tokens=getattr(u, "completion_tokens", 0) or 0)
        try:
            comp.cost_usd = float(litellm.completion_cost(completion_response=resp) or 0.0)
        except Exception:
            comp.cost_usd = 0.0
        content = msg.content or ""
        if text:
            comp.text, comp.tool_calls = parse_text_tool_calls(content)
        else:
            comp.text = content
            for tc in msg.tool_calls or []:
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {"_raw": tc.function.arguments}
                comp.tool_calls.append(ToolCall(id=tc.id or f"call_{uuid.uuid4().hex[:8]}",
                                                name=tc.function.name, arguments=args))
        self.total_cost += comp.cost_usd
        self.total_input_tokens += comp.input_tokens
        self.total_output_tokens += comp.output_tokens
        return comp

    # ------------------------------------------------------------------ text protocol
    @staticmethod
    def _to_text_protocol(messages: list[dict], tools: list[Tool]) -> list[dict]:
        listing = "\n".join(f"- {t.name}: {t.description.splitlines()[0] if t.description else ''}\n"
                            f"  parameters: {json.dumps(t.parameters['properties'])}" for t in tools)
        proto = TEXT_PROTOCOL.replace("{tools}", listing)
        out: list[dict] = []
        for m in messages:
            role = m["role"]
            if role == "system":
                out.append({"role": "system", "content": f"{m['content']}\n\n{proto}"})
            elif role == "assistant" and m.get("tool_calls"):
                blocks = "\n".join("```tool\n" + json.dumps({"name": c["function"]["name"],
                                                             "arguments": json.loads(c["function"]["arguments"])})
                                   + "\n```" for c in m["tool_calls"])
                out.append({"role": "assistant", "content": ((m.get("content") or "") + "\n" + blocks).strip()})
            elif role == "tool":
                out.append({"role": "user", "content": f"TOOL RESULT ({m.get('name', '')}):\n{m['content']}"})
            else:
                out.append({"role": role, "content": m.get("content") or ""})
        if not any(m["role"] == "system" for m in out):
            out.insert(0, {"role": "system", "content": proto})
        return out


_BLOCK = re.compile(r"```(?:tool|json)?\s*\n?(\{.*?\})\s*```", re.S)


def parse_text_tool_calls(content: str) -> tuple[str, list[ToolCall]]:
    calls: list[ToolCall] = []
    for raw in _BLOCK.findall(content):
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "name" in obj:
            calls.append(ToolCall(id=f"call_{uuid.uuid4().hex[:8]}", name=obj["name"],
                                  arguments=obj.get("arguments") or {}))
    text = _BLOCK.sub("", content).strip() if calls else content
    return text, calls


class RelayLLM(LLM):
    """File-relay model: any external process (a person, another agent) plays the LLM.

    Model string ``relay:<dir>``. For each step the harness writes ``<dir>/req_<n>.json``
    ({"step", "messages", "tools": [openai tool schemas]}) and waits for ``<dir>/resp_<n>.json``:
    ``{"text": "...", "tool_calls": [{"name": "...", "arguments": {...}}]}``. Writing ``<dir>/STOP``
    aborts the run. Useful for testing the harness with a model that has no API key.
    """

    def __init__(self, directory: str, timeout_s: float = 900, poll_s: float = 0.5):
        from pathlib import Path

        self.model = f"relay:{directory}"
        self.params = {}
        self.tool_mode = "native"
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.timeout_s, self.poll_s = timeout_s, poll_s
        self.total_cost = self.total_input_tokens = self.total_output_tokens = 0
        self.calls = 0

    @property
    def context_window(self) -> int:
        return 200_000

    def complete(self, messages: list[dict], tools: list[Tool] | None = None) -> Completion:
        self.calls += 1
        n = self.calls
        req, resp = self.dir / f"req_{n:03d}.json", self.dir / f"resp_{n:03d}.json"
        tmp = req.with_suffix(".tmp")
        tmp.write_text(json.dumps({"step": n, "messages": messages, "tools": [t.schema() for t in tools or []]},
                                  indent=1, ensure_ascii=False, default=str), encoding="utf-8")
        tmp.replace(req)
        deadline = time.time() + self.timeout_s
        while not resp.exists():
            if (self.dir / "STOP").exists():
                raise QuotaExhausted("relay stopped")
            if time.time() > deadline:
                raise QuotaExhausted(f"relay: no response to {req.name} within {self.timeout_s}s")
            time.sleep(self.poll_s)
        time.sleep(0.1)
        data = json.loads(resp.read_text(encoding="utf-8"))
        calls = [ToolCall(id=f"call_{n}_{i}", name=c["name"], arguments=c.get("arguments") or {})
                 for i, c in enumerate(data.get("tool_calls") or [])]
        self.total_input_tokens += len(json.dumps(messages)) // 4
        self.total_output_tokens += len(json.dumps(data)) // 4
        return Completion(text=data.get("text") or "", tool_calls=calls, model=self.model)


class FakeLLM(LLM):
    """Scripted model for tests: each step returns the next Completion (or calls a function)."""

    def __init__(self, script: list[Completion] | Callable[[list[dict], list[Tool]], Completion]):
        self.model = "fake"
        self.params = {}
        self.tool_mode = "native"
        self.script = script
        self.total_cost = self.total_input_tokens = self.total_output_tokens = 0
        self.calls = 0
        self.seen: list[list[dict]] = []

    @property
    def context_window(self) -> int:
        return int(os.environ.get("NXA_CONTEXT_WINDOW", "100000"))

    def complete(self, messages: list[dict], tools: list[Tool] | None = None) -> Completion:
        self.seen.append(messages)
        self.calls += 1
        if callable(self.script):
            return self.script(messages, tools or [])
        return self.script.pop(0) if self.script else Completion(text="(script exhausted)")
