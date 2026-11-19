# Rebuilding the Pinecone Nexus harness: research and plan

Researched 2026-11-19. Sources are listed at the bottom.

---

## 1. What Nexus is

Pinecone Nexus is a "knowledge engine" for agents. Instead of the agent doing retrieval at runtime (agentic RAG: search → read chunks → search again), Nexus **compiles** the corpus ahead of time into task-shaped structured artifacts. The agent then asks one declarative question and gets back a typed, cited answer.

| Piece | What it does |
|---|---|
| **Manifest** | A domain expert (or the compiler) declares entities, relationships and answer shapes. In practice it's a JSON config of artifact types (see §3). |
| **Context Compiler** | An autonomous coding agent. Inputs: (1) a per-domain eval set with known answers plus the sources, (2) a library of vetted "skills" (doc processing, entity extraction, chunking, …), (3) a feedback loop that scores each iteration. It edits two functions, `curate()` (build artifacts) and `query()` (retrieve and compose), runs the evals, learns from failures, and repeats until the evals pass. |
| **Artifacts / Context** | Typed outputs: summaries, LLM-extracted SQLite tables, an entity/relationship graph, plus chunks in a hybrid (BM25 + dense) Pinecone index. A *Context* is a curated bundle of artifacts for one job. |
| **Composable retriever / query service** | Takes `{scope, ask, shape?, budget?}`, runs a **server-side agent loop** with a cheap model over the artifacts, and returns an answer plus citations. Queries run async: POST, then poll. |
| **KnowQL** | The declarative query language: intent (ask + shape + scope), filter, provenance (`ground: true`), and control (budget/latency). Spec v0.1.0 is a working draft at spec.knowql.org. |

## 2. The headline numbers and what they actually are

Pinecone's GA claim: on Sierra's **τ-Knowledge** benchmark (`banking_knowledge`, 97 tasks, about 698 docs / 195K tokens), the "GPT-5.5 + Nexus" agent scored **47.4%** vs **46.4%** for GPT-5.5 alone, at roughly 74–77% lower cost.

**Pinecone's own benchmark harness is public**: `github.com/pinecone-io/tau2-bench`, branches `nexus` and `siva/nexus-router-retrieval`. Its scorecards show:

| Run | Nexus | Published `alltools` baseline | Metric |
|---|---:|---:|---|
| GPT-5.2 (high) | **36.08%** (35/97) | 24.74% from traj; board shows 32.2% after regrade | trial-1 |
| GPT-5.5 (xhigh), 4 trials | **46.13%** pass^1 (45.36% trial-1) | **46.39%** board pass^1 | pass^1 |
| Claude Opus 5 | **47.42%** (46/97) | **50.52%** | trial-1 |

Things to know before setting a target:

1. **Where 47.4% comes from.** In the repo, 47.42% is the **Opus 5** single-trial run, and that run scored *below* Opus 5's own alltools baseline (50.52%). The GPT-5.5 Nexus run scored 46.13% pass^1, slightly *below* the regraded board (46.39%). The press release's "GPT-5.5 + Nexus = 47.4%" does not match the repo's labels. The repo also contains a `_scrub_gpt_labels.py` script that renames run labels.
2. **The real win is cost and steps, not accuracy.** Tool calls per task fell from 42.5 to 17.7 (GPT-5.2) and from 28.6 to 16.0 (GPT-5.5). Agent spend fell 4–5x. Accuracy is about equal, except on GPT-5.2.
3. **The prompts were tuned on the test set.** The "hard rules" and the 249-line policy name specific benchmark tasks (`task_027`, `task_099`, `task_102`) and specific task details ("Uline + Grainger supplier disputes", "98.00 not 100"). That is test-set overfitting. Some of the gain comes from those rules, not from the knowledge engine.
4. **Grading version matters.** `tau2` v1.0.1 changed banking grading, and older trajectories under-score by 7–9 pp. Always run on current `tau2` and compare against regraded board numbers.

## 3. How the harness actually works (from their code)

### 3a. Agent side (tau2-bench fork). Small, fully public.

- New retrieval variants `nexus`, `nexus_multiturn` (the champion), `nexus_toolcat`, `nexus_checklist*`. Each one swaps the `alltools` toolset (`KB_search_bm25`, `KB_search_dense`, `shell`) for **one tool**:
  - `KB_query(question: str) -> str`: "ask one specific, self-contained question per call, name the exact product; for procedures ask for exact tool names and arguments."
- `NexusClient`: `POST /api/v0/query {scope:[context], ask, background:true, model?}`, then polls `GET /api/v0/queries/{id}` every 3 s, with a 600 s timeout. It flattens `output[].content[].output_text` and appends a `Citations:` list. Errors come back as strings so the agent can retry.
- **Query directive** (`NEXUS_QUERY_DIRECTIVE`): text appended to every ask as `[Answer requirements: …]`. It pins the bank date (2025-11-14) and requires COPY_LITERALS (exact enum strings, tool names, transfer-reason decision tree, the interest formula).
- **Hard rules** (`TAU2_AGENT_HARD_RULES_FILE`): injected as `<hard_rules>` *before* `<policy>` in the system prompt, plus one line saying they override the policy.
- **Policy** (`TAU2_NEXUS_POLICY_FILE`): a process-compliance protocol (get the full procedure from the KB before any write, verify first, keep a checklist, no extra writes, exact canonical args), product-selection rules, promo rules, and transfer-code decision trees.
- Settings used: user simulator `gpt-5.2` (reasoning low), seed 300, 4 trials, concurrency 8. Query model is `claude-sonnet-5`, `GLM-5.2`, or the server default. Measured cost was about $0.04–0.30 per query.
- Per-query usage log (JSONL): task_id, latency, tokens, cost_usd.

### 3b. Engine side (closed, but the configuration is visible)

The build script `nexus_build_context.py` runs: create context with manifest → upload the 698 docs as a markdown zip → `curate` → `optimize` (one-time; makes the context queryable) → check the manifest didn't drift → smoke query.

The **champion manifest** is in the router branch (`nexus_manifest_champion.json`):
- Chunks: 768 tokens with 128 overlap (hybrid index).
- Artifacts: LLM extraction (`artifact_model: standard`, 4,000 max tokens, 50K-char extraction window, ≤500 per type). Types:
  - `doc_guard`: one-line doc-type label (product, procedure, policy, other)
  - `product_profile` (SQLite, per doc): about 27 columns, including product, family, class, fees, APY, tiers, ATM terms, `enum_card_type`, `account_class_enum`, `referral_account_type`, eligibility
  - `atm_fee_terms`, `savings_card_boost`, `internal_tool` (tool_name, args_json, preconditions), `tool_step` (procedure, step_ord, action, tool_name, trap_or_check, args_hint), `product_select_rule` (wins_when, loses_when, hard_filters, rank_metric), `transfer_protocol` (reason_code, when_to_use, do_not_use_when)
  - `scenario_playbook`: a corpus-level markdown topic artifact
  - Every row carries `source_doc` for citations.
- **Router mode** (experimental branch): instead of one `KB_query`, the agent gets MCP tools directly: `orient`, `query_db` (SQL over those tables), `read_source`, `outline_knowledge`, `list_artifacts`, `read_artifact[_full]`, `read_rosters`, `search_source[_by_keyword]`. The prompt says "orient once → SQL first → read 1–3 winning docs → search only as fallback."

## 4. Can we do this?

- **Agent-side harness: yes, exactly.** It's about 500 lines on top of tau2-bench, and all of it is public.
- **Engine: yes, closely.** It's closed source, but the API shape, manifest schema, artifact tables and query loop are all visible. We'd build an open version (SQLite + a local hybrid index + an LLM query loop).
- **Context Compiler (the coding-agent loop): yes.** We'd build it with the Claude Agent SDK or Claude Code headless, editing `curate()`/`query()` against an eval.
- **Matching the numbers: plausible for GPT-5.2 (~36%) and GPT-5.5 (~46%), but it costs money.** One full GPT-5.5 xhigh 4-trial run cost them about $190 for the agent plus about $57 for Nexus queries. GPT-5.2 single-trial was about $27 + $25. We need an API budget for the agent, user-sim and query models. This conflicts with the free-only constraint, so it's your call.
- **Honesty line:** we *can* hit their number by copying their task-specific hard rules. That's test-set tuning, so we'll report two numbers: **clean** (generic rules, compiler tuned on a dev split) and **matched** (their exact pack).

## 5. Build plan

Repo: `nexus-harness/` (public, per the workspace convention). Python 3.12, `uv`.

### Phase 0: Baseline reproduction (1–2 days)
1. Fork the latest `sierra-research/tau2-bench` (≥ v1.0.1 grading). Run `uv sync --extra knowledge`.
2. Run the `bm25` config (offline, cheap) with a small agent on 10 tasks to prove the pipeline works.
3. Download the published `alltools` trajectories (GPT-5.2 and GPT-5.5) from Sierra's public S3. Re-score them with `tau2 evaluate-trajs --fresh-tasks` so our comparison baseline uses current grading.
4. Write the scorecard script: pass^1 (mean successes/trials), trial-1 rate, head-to-head per task, tool calls, model calls, $ per task.

### Phase 1: Agent-side harness, same as Pinecone's (1 day)
5. Add a `KB_query` retrieval variant, an HTTP client (async POST + poll), and a usage log with task_id.
6. Inject hard rules (`<hard_rules>` before `<policy>`), allow a policy-file override, and append the query directive.
7. Make "packs" (`configs/packs/{clean,matched}/`): policy.md, query_directive.txt, hard_rules.txt, setup.env.
   - `matched` = Pinecone's v2 pack verbatim (for reproduction).
   - `clean` = generic rules only, with every task-specific line removed.

### Phase 2: Open knowledge engine, "nexus-lite" (4–6 days)
8. **Ingest:** 698 JSON docs → markdown, with doc IDs kept for citations.
9. **Chunk index:** 768/128 chunks; BM25 (`bm25s`) plus dense embeddings (local `bge`/`qwen3-embedding` to stay free, or `text-embedding-3-large` for parity); RRF fusion. Storage: LanceDB or sqlite-vec (local, free).
10. **Manifest-driven compiler (`curate()`):** read the manifest JSON (same schema as theirs), and for each artifact type run LLM extraction per document (with an extraction window) → validated rows → SQLite tables with `source_doc`. Also produce `doc_guard` labels and corpus-level `scenario_playbook` markdown. Must be incremental: hash each doc and recompile only what changed.
11. **Query service (`query()`):** FastAPI `POST /api/v0/query`, `GET /api/v0/queries/{id}`. A server-side agent loop with a cheap model (GLM / Qwen / Haiku). Tools: `sql(query)`, `search(hybrid)`, `read_source(doc)`. Output: answer plus a citations array. Budget knobs: max steps and latency.
12. **KnowQL v0 subset:** `ask`, `scope`, `shape` (JSON Schema → structured output), `ground` (citations), `budget`.
13. **Router mode:** expose the same tools over MCP for the "agent queries SQL directly" variant.

### Phase 3: Context Compiler, the coding-agent harness (4–6 days)
14. **Dev eval set** (no test leakage): generate about 200 KB-only QA pairs from the docs (fees, enums, procedure steps, tool args), checked by an LLM judge and spot-checked by hand. Optionally also split the 97 tasks into 48 dev / 49 held-out.
15. **Skills library:** chunkers, a table extractor, enum normalizer, procedure-step extractor, entity linker, summarizer, each a small tested function.
16. **Compiler loop** (Claude Agent SDK): workspace = `curate.py`, `query.py`, `manifest.json`, skills/. Loop: edit → rebuild changed artifacts → run dev evals (accuracy, tokens, latency) → read failures → edit. Stop at the target score or after N iterations. Every iteration is git-committed so it can be replayed.
17. Freeze the best context as `context-v1`, plus a manifest snapshot and a downloadable archive (as their "compiled layer download" does).

### Phase 4: Benchmark campaign (2–4 days, needs budget)
18. Smoke test: 10 tasks × 1 trial for each pack.
19. Full runs: 97 tasks × 4 trials, seed 300, user-sim `gpt-5.2` low, c=8. Run for GPT-5.2 high and GPT-5.5 xhigh, each with the `clean` and `matched` packs.
20. Scorecards against the regraded `alltools` baseline: pass^1, tool calls and model calls per task, $ per task (agent + query + user).

### Targets
| Config | Pinecone reported | Our target |
|---|---:|---|
| GPT-5.2 + engine, matched pack | 36.1% | ≥ 35% |
| GPT-5.5 + engine, matched pack | 46.1% pass^1 | ≥ 45% |
| Tool calls per task (GPT-5.2) | 17.7 | ≤ 20 |
| Cost reduction vs alltools | 74–80% | ≥ 70% |
| Clean pack | n/a | report as-is (honest number) |

## 6. Risks
- **Cost.** This is the main blocker. A rough minimum for a meaningful run: GPT-5.2 × 97 × 1 trial is about $30 agent + $10–25 queries + a few $ user-sim. 4 trials × 2 models × 2 packs is several hundred $.
- **Model availability.** GPT-5.5 xhigh and Opus 5 need paid API access. Open-weight agents (GLM/Qwen via OpenRouter) are cheaper but won't match those numbers.
- **Variance.** pass^1 on 97 tasks moves ±3–4 pp between runs. A 1 pp "win" like theirs is noise.
- **Sandbox runtime** for the `alltools` baseline needs Linux/macOS (bubblewrap/ripgrep). On Windows, use WSL. Or skip running the baseline and use the re-scored published trajectories.

## Sources
- Pinecone blog: introducing Nexus (compiler, KnowQL, KRAFTBench): https://www.pinecone.io/blog/introducing-nexus-knowledge-engine/
- Pinecone GA post (τ-Knowledge numbers): https://www.pinecone.io/blog/pinecone-nexus-generally-available/
- Pinecone: knowledge infrastructure for agents: https://www.pinecone.io/blog/knowledge-infrastructure-for-agents/
- Pinecone EA benchmarks: https://www.pinecone.io/blog/nexus-ea-benchmarks/
- Pinecone harness fork: https://github.com/pinecone-io/tau2-bench (branches `nexus`, `siva/nexus-router-retrieval`)
- Sierra τ-Knowledge: https://sierra.ai/blog/tau-knowledge · paper https://arxiv.org/abs/2603.04370 · code https://github.com/sierra-research/tau2-bench
- KnowQL spec (draft 0.1.0): https://spec.knowql.org
