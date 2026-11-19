# A general agent harness: any LLM, any task

Drafted 2026-11-20. Builds on nexus-harness (`nexus_lite`, tau2 integration, scorecard).

## 1. What the good harnesses actually are

A 2026 source-code study of 11 production harnesses (Claude Code, Codex CLI, Gemini CLI, OpenHands, Aider, Pi, OpenCode, Mini-SWE-Agent, …) found:
- The **core loop is small**: call the model → run the tools it asked for → repeat. Most of the code is *around* the loop: permissions, context budgeting, error recovery, sessions, extensions.
- **None of them import a general agent framework.** Every one is a hand-rolled async loop.
- **Retrieval is deterministic** (grep, read, find), not vector embeddings, for code. For *documents and policies*, compiled knowledge (our nexus-lite) beats raw search, which is the point of our benchmark work.
- **Skills (SKILL.md)** are adopted more widely than MCP (9/11 vs 8/11). Behavioral rules are moving from prompt prose into config.

Reference designs:

| Harness | Idea worth copying |
|---|---|
| **Pi** (Mario Zechner, MIT) | Minimal core: 4 tools (`read`, `write`, `edit`, `bash`), a tiny system prompt, everything else as extensions. Unified LLM API (`pi-ai`) across ~25 providers. Sessions are JSONL trees, so you can branch and fork. Modes: interactive, print (`-p`), JSON events, RPC, SDK. |
| **Claude Code** (closed) | Small, tightly scoped tools; a multi-stage context-compaction pipeline; subagents with their own isolated context that return a summary; permission modes; MCP for new *actions*, Skills for new *know-how*. |
| **OpenCode** | Client/server split, so a UI talks to an agent server. 75+ providers. |
| **Codex CLI / Goose** | Sandbox-first command execution; one self-contained binary. |
| **OpenHands** | Event-stream architecture: every action and observation is an event, so runs are replayable. |

## 2. Answer: yes, with one honest limit

We can build a harness where you pick any LLM (cloud or local) and give it a task. It plans, uses tools, reads and writes files, runs commands, consults compiled knowledge, and reports back.

**The limit:** the harness gives the model hands and memory, but how well a task gets done still depends on the model. A frontier model will do multi-step work well. A free flash-lite or small local model will do simple tasks and fail on long ones. The harness should make *every* model as good as it can be: good tools, clean context, retries, verification. Our benchmark then tells us how good each model+harness combination really is.

## 3. What we reuse from nexus-harness

| Already built | Role in the general harness |
|---|---|
| litellm (via tau2) | "Any LLM": 7,000+ models: OpenAI, Anthropic, Gemini, OpenRouter, Groq, Ollama (local, free) |
| `nexus_lite` | The **knowledge tool**: `kb build ./docs` then `KB_query` gives cited answers from your docs |
| tau2 + packs + scorecard | The **eval harness**: score any model+harness config on banking/airline/retail/telecom |
| usage log + cost tracking | Per-call token and $ accounting |
| Tool decorator pattern (`@is_tool`) | Tool registration with auto-generated JSON schemas from docstrings |

## 4. Design

```
                ┌─────────────── interfaces ───────────────┐
                │ CLI chat · -p one-shot · JSON events ·    │
                │ Python SDK · HTTP/RPC server               │
                └───────────────────┬──────────────────────┘
                                    ▼
┌──────────── core (≈ 500 lines) ────────────┐   ┌──── providers ─────┐
│ agent loop: model → tool calls → results   │◀─▶│ litellm + capability │
│ stop rules · retries · step/time/$ budgets │   │ table (tools? ctx?   │
│ event stream (every step is an event)      │   │ cost) + fallback     │
└───────┬───────────────┬────────────────────┘   │ text-protocol tools  │
        ▼               ▼                         └─────────────────────┘
  context manager    tool registry
  · AGENTS.md         · built-in: read, write, edit, bash, grep, find, ls
  · token budget      · web: fetch, search
  · output truncation · knowledge: KB_query / KB_sql (nexus-lite)
  · compaction        · memory: remember / recall
    (summarize)       · subagent: spawn child loop, get summary back
  · skills (SKILL.md) · MCP client: any MCP server's tools
        ▼               ▼
  sessions (JSONL tree: resume, fork)    safety (permission modes, confirm destructive, workdir jail)
```

**Key decisions**
- **Python**, so it reuses litellm, nexus-lite and tau2 directly.
- **Tool calling for weak models:** native function calling where the model supports it. Otherwise a JSON-in-text fallback protocol, so small local models still work.
- **Context:** every tool result is truncated to a budget, and compaction (summarize older turns) runs at about 80% of the context window. The full history stays in the session file.
- **Safety by default:** ask before `bash`, writes outside the workdir, or anything destructive. `--yolo` for trusted runs.
- **Extensions:** drop a Python file in `extensions/` to add tools, commands or hooks. Skills are markdown in `skills/`.

## 5. Build plan

| Phase | Deliverable | Done when |
|---|---|---|
| **v0 core** (1–2 days) | Loop + litellm providers + 7 file/shell tools + `-p` one-shot + interactive chat + JSONL sessions + usage log | Gemini flash-lite and an Ollama model both finish "read this repo, fix this bug, run the test" |
| **v1 context + safety** | Truncation, compaction, AGENTS.md, permission modes, workdir jail, retries and backoff (free-tier 429s) | A 50-step task runs without overflowing context |
| **v2 knowledge + extensibility** | `KB_query` via nexus-lite, skills, MCP client, extension hooks, subagents | "Answer from my docs folder" and "use an MCP server's tools" both work |
| **v3 interfaces** | Python SDK, JSON event stream, HTTP/RPC server, optional textual TUI | Another app can drive it |
| **v4 eval** | tau2 adapter: run our harness as the tau2 agent. Optional Terminal-Bench / SWE-bench-lite subset | Scorecard per model on the same tasks |

## 6. Cost reality

- Building and testing is free: Ollama local models plus Gemini free tier.
- The Gemini free tier is tight: `gemini-3-flash` allows 20 requests/day. Flash-lite models have more headroom. The harness needs rate-limit-aware retries and per-run budgets from day one.

## Sources
- Harness Engineering: a source-code study of eleven systems (arXiv 2609.00006): https://arxiv.org/abs/2609.00006
- Dive into Claude Code (arXiv 2604.14228): https://arxiv.org/html/2604.14228v1
- Pi coding agent README: https://unpkg.com/@mariozechner/pi-coding-agent@0.73.1/README.md
- Harness comparison (Upstash): https://upstash.com/kb/opencode-vs-pi-vs-gemini-cli-vs-crewai-vs-goose-9-agent-harnesses-compared
- What is an agent harness (freeCodeCamp): https://www.freecodecamp.org/news/what-is-an-agent-harness/
