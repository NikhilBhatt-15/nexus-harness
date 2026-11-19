# nexus-harness

An open rebuild of the **Pinecone Nexus** benchmark harness on Sierra's **τ-Knowledge** (`banking_knowledge`) benchmark.

Pinecone Nexus is a "knowledge engine" for agents. It compiles a corpus ahead of time into task-shaped artifacts, and the agent asks one question per call instead of running agentic RAG. Pinecone reported 36–47% on τ-Knowledge with far fewer tool calls and lower cost. This repo rebuilds that setup with open parts, so the same experiment can be run, inspected and compared:

- **`nexus_lite/`**: a local knowledge engine (compile → artifacts → cited answers). Runs free on a laptop: SQLite, BM25 and local Ollama embeddings.
- **tau2-bench fork** (Sierra, MIT): new `nexus`, `nexus_multiturn` and `nexus_router` retrieval configs, frozen instruction packs, per-query usage logging, a scorecard, and offline regrading.

Background research, Pinecone's real numbers and the build plan are in [RESEARCH_AND_PLAN.md](RESEARCH_AND_PLAN.md).

**Also here: [`nexus_agent/`](nexus_agent/README.md), a general agent harness** built on the same parts. Pick any LLM, attach tools (files, shell, web, your databases, REST APIs, MCP servers, your own Python functions, and nexus-lite knowledge compiled from your docs), and give it tasks. It ships with a CLI (`nxa`), a Python SDK, an HTTP API with remote tool approval, sessions, permissions, skills, and an eval runner. A working example is in [`examples/shop-ops/`](examples/shop-ops/).

## How it works

```
698 KB docs ──curate()──▶ Context (SQLite artifacts + hybrid index) ──query()──▶ cited answer ──▶ KB_query tool ──▶ tau2 agent
```

**curate()** (`nexus_lite/compiler.py`, incremental and keyed by content hash) builds:

| Artifact | What it holds |
|---|---|
| `entities` | 70 products/programs, grouped from doc ids |
| `facts` | 3,189 fee / APY / limit / bonus / eligibility lines with numbers |
| `procedures` | 754 numbered steps, with the internal tool name pulled out |
| `tools` | 46 internal tools: docs, argument hints, mentions |
| `enums` | quoted literals and codes (account classes, reason codes) |
| `fact_sheet:<entity>` | one document per product: key figures, tools, all its source docs |
| `roster:<category>` | an "all X" comparison view per category |
| `tool:<name>` | tool cards |
| chunks | 768/128-token heading-aware chunks; BM25 + `embeddinggemma-2` (Ollama), fused with RRF |

Optional: `--manifest nexus_lite/manifests/banking.json --llm-model <litellm model>` adds LLM-extracted tables (`x_product`, `x_procedure_step`, …) in the same shape as a Nexus manifest.

**query()** (`nexus_lite/query.py`) takes a KnowQL-lite request (`ask`, `scope`, `shape`, `ground`, `filter`, `budget`):
1. Routes the question to the matching entities, categories and tools, and decides whether it is procedural or comparative.
2. Composes, within the budget: procedure documents with exact tool names, product fact sheets, rosters, tool cards, and hybrid search hits.
3. Returns the evidence with citations (evidence mode, free). If a model is set, it also writes a grounded answer, as Nexus's server-side loop does.

**Retrieval configs added to tau2:**

| Config | Agent gets |
|---|---|
| `nexus` / `nexus_multiturn` | one tool, `KB_query(question)`. Same contract as Pinecone's harness. `nexus_multiturn` adds a process-compliance policy. |
| `nexus_router` | direct tools over the compiled store: `KB_orient`, `KB_sql`, `KB_read_artifact`, `KB_read_source`, `KB_search_passages` |

`NEXUS_URL=inproc` loads the engine inside the tau2 process. An `http(s)://` URL talks to `nexus-lite serve`, or to Pinecone Nexus itself through the same v0 API (`POST /api/v0/query` with background polling).

**Instruction packs** (`configs/packs/`) freeze the policy, query directive and hard rules:
- `clean`: generic rules only. This is the honest number.
- `matched`: Pinecone's published v2 pack, fetched by `scripts/nexus/fetch_pinecone_pack.sh` and not committed. Its rules name specific benchmark tasks, so scores with it are tuned on the test set.

## Install

Needs Python 3.12 and, for free dense retrieval, [Ollama](https://ollama.com) with `ollama pull embeddinggemma-2`.

```bash
python -m venv .venv && . .venv/Scripts/activate   # or .venv/bin/activate
pip install -e ".[knowledge]" -e ./nexus_lite
```

## Build the knowledge context

```bash
nexus-lite build --source data/tau2/domains/banking_knowledge/documents --slug banking
#   about 11 min on CPU the first time (embeddings); later builds only redo changed docs
nexus-lite query --slug banking "What is the monthly fee for Beige business checking and how is it waived?"
nexus-lite sql   --slug banking "SELECT tool_name, arg_hints FROM tools LIMIT 5"
nexus-lite serve --port 8765        # optional HTTP API
```

`--embedder none` builds a BM25-only context with no Ollama.

## Run the benchmark

Running agents needs an LLM. Any litellm model works, including local `ollama_chat/...` models.

```bash
PACK=clean AGENT_LLM=gpt-5.2 AGENT_LLM_ARGS='{"reasoning_effort":"high"}' \
  NUM_TRIALS=4 bash scripts/nexus/run_pack.sh
# Pinecone-matched setup:
bash scripts/nexus/fetch_pinecone_pack.sh && PACK=matched AGENT_LLM=... bash scripts/nexus/run_pack.sh
```

Each run writes `data/simulations/<name>/results.json` plus `<name>.nexus_usage.jsonl` (one row per KB query).

## Score and compare

```bash
# Re-grade published baselines under the current grader, offline (DB-only, no API key):
python scripts/nexus/regrade_db.py data/published/gpt52_alltools.json data/published/regraded/gpt52_alltools.json --fresh-tasks
# Scorecard: pass^k, trial-1, tool/model/KB calls, $ per sim, head-to-head:
python scripts/nexus/scorecard.py OURS.json --baseline data/published/regraded/gpt52_alltools.json \
  --usage OURS.nexus_usage.jsonl --out SCORECARD.md
```

Published trajectories come from Sierra's public bucket, for example `https://sierra-tau-bench-public.s3.amazonaws.com/submissions/gpt-5-2_sierra_2026-02-26/trajectories/gpt-5.2_high_banking_knowledge_gpt-5.2_4trials.json`.

## Baselines (alltools, Sierra's published trajectories)

| Model | As published (raw traj) | Leaderboard (tau2 1.0.1 regrade) | Regraded here (tau2 2.0.0, DB-only) | Pinecone Nexus (their run) |
|---|---:|---:|---:|---:|
| GPT-5.2 high | 24.74% | 32.22% | **38.14%** | 36.08% (trial-1) |
| GPT-5.5 xhigh | 37.37% | 46.39% | **50.26%** | 46.13% pass^1 |

The grader version moves the baseline by up to 13 points. Always compare runs that were graded the same way. Details are in [docs/nexus/](docs/nexus/).

## Status

- Done: engine, compile, query, HTTP API, tau2 integration (3 configs), packs, usage logging, scorecard, offline regrade, and tests (`pytest tests/test_domains/test_banking_knowledge/test_retrieval_e2e.py`).
- Not done yet: full benchmark runs (they need an LLM budget; the Gemini free tier allows 20 requests/day on gemini-3-flash), and the coding-agent Context Compiler loop (Phase 3).

## Credits

τ-bench / τ-Knowledge © Sierra Research (MIT): see [docs/TAU2_README.md](docs/TAU2_README.md). Pinecone Nexus is a Pinecone product; this repo is an independent open reimplementation and is not affiliated with Pinecone.
