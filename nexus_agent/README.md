# nexus-agent

A general agent harness. Pick **any LLM**, **attach the tools** it should have (files, shell, web, your databases, your APIs, MCP servers, your own Python functions, and a **nexus-lite knowledge base** compiled from your docs), and give it tasks. It plans, calls tools, verifies, and reports back, with permissions, sessions, budgets, and an eval runner to measure how well each model does.

```
          ┌──── you: CLI chat · nxa run · Python SDK · HTTP API ────┐
          ▼                                                          │
   Agent loop ── model (any litellm model, native or text-protocol tools; retries; cost)
      │
      ├─ toolsets you attach/detach at runtime
      │    files · shell · web · sql/sqlite · http_api · knowledge (nexus-lite) · mcp · python plugins · memory · subagent
      ├─ permissions   readonly / ask / auto / yolo + per-tool allow/deny; risky calls go to an approver
      ├─ context       tool-output truncation, auto-compaction (summarize old turns), AGENTS.md, skills
      └─ sessions      JSONL log: resume, fork; every step is an event
```

## Install

From the repo root (Python 3.12):

```bash
pip install -e ./nexus_lite -e "./nexus_agent[mcp]"     # add [sql] for Postgres/MySQL via SQLAlchemy
```

Put API keys in `.env` (for example `GEMINI_API_KEY=...`, `OPENAI_API_KEY=...`, `ANTHROPIC_API_KEY=...`, `OPENROUTER_API_KEY=...`). For free local models, run [Ollama](https://ollama.com) and use `ollama_chat/<model>`.

## Use it

### CLI

```bash
nxa chat -m gemini/gemini-3.1-flash-lite -t files,shell,web          # interactive
nxa run "Summarize this repo's README" -m gpt-5.2 -t files
nxa run "Who are our top 5 customers this month?" -m gemini/gemini-3.1-flash-lite -t sqlite=shop.db
nxa run "What does our policy say about refunds after 30 days?" -t kb=handbook@./docs   # builds the KB on first use
nxa run "..." --json        # JSON-lines events for scripts
nxa chat --resume last      # continue the latest session
nxa sessions | nxa models gemini | nxa tools -t files,sqlite=shop.db
```

Tool spec (`-t`, comma separated): `files`, `files=DIR`, `shell`, `web`, `memory`, `sqlite=PATH`, `sql=URL`, `kb=SLUG`, `kb=SLUG@DOCS_DIR`, `python=FILE.py`, `api=NAME@BASE_URL`, `mcp=NAME@COMMAND ARGS`, `mcp=NAME@https://URL`.

In chat: `/tools`, `/attach sqlite=other.db`, `/detach shell`, `/model ollama_chat/qwen3:4b`, `/mode readonly`, `/cost`, `/fork`, `/clear`, `/exit`.

### Config file (`agent.yaml`)

```yaml
model: gemini/gemini-3.1-flash-lite
permissions: auto            # readonly | ask | auto | yolo
deny: [shop_admin]           # tools or toolsets never allowed
max_steps: 30
max_cost_usd: 0.50
system: You are the operations assistant for Acme. Today is 2026-11-20.
skills_dirs: [skills]
toolsets:
  - sqlite: {path: shop.db, name: shop}
  - knowledge: {slug: policies, source: ./docs}
  - http_api: {name: crm, base_url: https://crm.example.com/api, headers_from_env: {Authorization: CRM_TOKEN}}
  - mcp: {name: github, command: npx, args: ["-y", "@modelcontextprotocol/server-github"]}
  - python: {path: my_tools.py}
  - files: {root: .}
  - memory: {}
```

`nxa chat -c agent.yaml`. A full working example is in [`examples/shop-ops/`](../examples/shop-ops/).

### Python SDK

```python
from nexus_agent import Agent, tool, toolsets

@tool(risk="write")
def send_sms(phone: str, text: str) -> str:
    """Send an SMS to a customer.

    Args:
        phone: E.164 phone number.
        text: Message body.
    """
    return my_sms_provider.send(phone, text)

agent = Agent("gemini/gemini-3.1-flash-lite",
              [toolsets.sqlite("shop.db", name="shop"),
               toolsets.knowledge("policies", source="./docs"),
               send_sms],
              permissions="ask", approver=lambda t, args: input(f"allow {t.name}? ") == "y",
              session=True)
print(agent.run("Refund order 103 per policy and text the customer").text)
agent.detach("shop")                      # tools come and go at runtime
for ev in agent.stream("and the next one?"):   # events: user, assistant, tool_call, tool_result, denied, compaction, done
    ...
```

### HTTP API

`nxa serve -c agent.yaml --port 8787`, optionally with `NXA_SERVER_TOKEN` set.

| Call | What it does |
|---|---|
| `POST /v1/runs {"task", "session_id"?, "model"?, "tools"?, "mode"?}` | Starts a run. Reusing a `session_id` continues the conversation. |
| `GET /v1/runs/{id}` | Status and result. Includes `pending_approval` when a risky tool waits. |
| `POST /v1/runs/{id}/approve {"allow": true}` | Answers a pending approval. |
| `GET /v1/runs/{id}/events?after=N` or `/stream` (SSE) | Live events. |

## Attaching your systems

| You have | Attach | The agent gets |
|---|---|---|
| A database | `toolsets.sql("postgresql://...")` or `sqlite("x.db")` | `*_schema`, `*_query` (read), `*_execute` (write), `*_admin` (DROP/ALTER/mass DELETE, always needs approval) |
| A REST API | `toolsets.http_api(base_url, headers_from_env={...})` | `*_get` (read), `*_call` (write). Secrets come from env vars and never enter the prompt. |
| Docs, manuals, policies | `toolsets.knowledge(slug, source="./docs")` | `kb_query` (cited answers from compiled fact sheets and procedures), `kb_sql`, `kb_read_source` |
| An MCP server | `toolsets.mcp(name, command=..., args=...)` or `url=` | Every server tool, as `name__tool` |
| Your own code | `@tool` functions, or `toolsets.python_tools("file.py")` | Your functions, with schemas built from type hints and docstrings |
| Repeatable know-how | `skills/<name>/SKILL.md` | A skills index in the prompt, loaded on demand with `load_skill` |

Risk levels (`read` < `write` < `exec` < `destructive`) drive permissions. `auto` runs reads and writes and asks before shell commands, MCP tools and destructive actions.

## Measure it

```bash
nxa eval examples/shop-ops/suite.yaml -m gemini/gemini-3.1-flash-lite -m ollama_chat/qwen3:4b --out results.json
```

Each task runs in a fresh copy of its workspace. Available checks: `answer_contains`, `answer_regex`, `file_exists`, `file_contains`, `sql` (query equals value), `command` (exit code), `max_tool_calls`, `judge` (LLM rubric). The output is a per-model pass rate, steps, tool calls, cost and time.

Agent benchmarks: `tau2 run --domain banking_knowledge --agent nexus_agent --agent-llm <model>` runs this harness's model layer inside τ-bench.

**Relay model** (testing without an API key): `-m relay:<dir>`. The harness writes each request to `<dir>/req_N.json` and waits for `<dir>/resp_N.json`, so a person or another agent can play the model.

## Tests

```bash
pytest nexus_agent/tests      # scripted fake model; no API keys needed
```
