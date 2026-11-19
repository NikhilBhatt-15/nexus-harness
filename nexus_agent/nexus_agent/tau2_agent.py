"""Run the nexus-agent LLM layer as the agent inside tau2-bench.

tau2 owns the conversation (user simulator, environment tools, grading); this adapter swaps the
model call for nexus-agent's: any litellm model, native or text-protocol tool calling, retries with
backoff and quota detection, and context trimming of old tool outputs.

    tau2 run --domain banking_knowledge --agent nexus_agent --agent-llm gemini/gemini-3.1-flash-lite ...

Agent llm_args understood here (besides normal litellm params):
    tool_mode: auto|native|text     old_tool_chars: int (trim tool outputs older than the last 6 messages)
"""

from __future__ import annotations

from typing import Any

from tau2.agent.llm_agent import LLMAgent
from tau2.data_model.message import AssistantMessage, MultiToolMessage, ToolCall, UserMessage
from tau2.utils.llm_utils import to_litellm_messages

from nexus_agent.llm import LLM
from nexus_agent.runtime import truncate
from nexus_agent.tools import Tool


class NexusTau2Agent(LLMAgent):
    def __init__(self, tools, domain_policy, llm: str, llm_args: dict | None = None, **kw: Any):
        super().__init__(tools=tools, domain_policy=domain_policy, llm=llm, llm_args=llm_args, **kw)
        args = dict(llm_args or {})
        self.old_tool_chars = int(args.pop("old_tool_chars", 6000))
        tool_mode = args.pop("tool_mode", "auto")
        self.client = LLM(llm, tool_mode=tool_mode, **args)
        self.nxa_tools = []
        for t in tools:
            fn = t.openai_schema["function"]
            self.nxa_tools.append(Tool(name=fn["name"], description=fn.get("description", ""),
                                       parameters=fn.get("parameters", {"type": "object", "properties": {}}),
                                       fn=lambda **_: None))

    def _generate_next_message(self, message, state) -> AssistantMessage:
        if isinstance(message, UserMessage) and message.is_audio:
            raise ValueError("audio not supported")
        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
        else:
            state.messages.append(message)
        msgs = to_litellm_messages(state.system_messages + state.messages)
        # context management: older tool outputs are trimmed; the latest results stay intact
        for m in msgs[:-6]:
            if m.get("role") == "tool" and isinstance(m.get("content"), str):
                m["content"] = truncate(m["content"], self.old_tool_chars)
        comp = self.client.complete(msgs, self.nxa_tools)
        calls = [ToolCall(id=c.id, name=c.name, arguments=c.arguments) for c in comp.tool_calls] or None
        return AssistantMessage(role="assistant", content=comp.text or None if calls else (comp.text or ""),
                                tool_calls=calls, cost=comp.cost_usd,
                                usage={"prompt_tokens": comp.input_tokens, "completion_tokens": comp.output_tokens})


def create_nexus_agent(tools, domain_policy, **kwargs):
    return NexusTau2Agent(tools=tools, domain_policy=domain_policy, llm=kwargs.get("llm"),
                          llm_args=kwargs.get("llm_args"),
                          additional_stop_tool_names=kwargs.get("additional_stop_tool_names", frozenset()))
