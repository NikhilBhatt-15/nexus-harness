"""nexus-agent: a general agent harness.

    from nexus_agent import Agent, tool, toolsets

    agent = Agent(model="gemini/gemini-3.1-flash-lite",
                  tools=[toolsets.files("."), toolsets.shell(), toolsets.sqlite("shop.db")])
    result = agent.run("Which 5 customers spent the most last month? Write them to top5.csv")
    print(result.text)

Pieces:
  llm         any litellm model; native tool calling or a text-protocol fallback; retries/backoff; cost
  tools       ``@tool`` decorator, ``Tool``, ``Toolset`` (attach/detach at runtime), built-in toolsets
  agent       the loop: model -> tool calls -> results -> repeat, with budgets, permissions, compaction
  session     JSONL session log (resume, fork)
  permissions ask / auto / readonly / yolo, with a pluggable approver
  skills      SKILL.md know-how, loaded on demand
  config      agent.yaml: model, toolsets, permissions in one file
  cli/server  `nxa` terminal app and an HTTP API
  evals       task-suite runner + tau2 agent adapter
"""

from nexus_agent.agent import Agent, RunResult
from nexus_agent.tools import Tool, Toolset, tool
from nexus_agent import toolsets

__all__ = ["Agent", "RunResult", "Tool", "Toolset", "tool", "toolsets"]
__version__ = "0.1.0"
