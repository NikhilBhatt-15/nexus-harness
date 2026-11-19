"""query(): execute a KnowQL-lite request against compiled contexts.

Request (subset of KnowQL v0 primitives)::

    {
      "ask":    "What is the monthly fee for Sky Blue and how is it waived?",
      "scope":  ["banking"],                 # context slugs
      "shape":  {...JSON Schema...},         # optional; typed output (needs a model)
      "ground": true,                        # attach citations
      "filter": {"category": [...], "entity": [...]},
      "budget": {"depth": "standard"|"deep", "max_chars": 16000},
      "model":  "gpt-5-mini"                 # optional litellm model for synthesis
    }

Without a model the engine runs in *evidence mode*: it routes the ask to the
right artifacts (entity fact sheets, category rosters, procedure steps, tool
cards, hybrid chunk hits) and returns them composed under the budget, with
citations. With a model it additionally synthesizes a grounded answer.

Response mirrors the Nexus v0 wire shape::

    {"id", "status", "output": [{"content": [{"type": "output_text", "text"}]}],
     "citations": [{"id", "title"}], "usage": {...}, "runtime_ms", "model", "plan"}
"""

from __future__ import annotations

import json
import os
import re
import time
import uuid

from nexus_lite.compiler import TOOL_RE
from nexus_lite.context import Context

DEPTH_K = {"quick": 5, "standard": 8, "deep": 16}
DEFAULT_MAX_CHARS = int(os.environ.get("NEXUS_LITE_MAX_CHARS", "14000"))

PROCEDURE_WORDS = re.compile(
    r"\b(how (do|to|can|should|does)|procedure|process|steps?\b|tools?\b|arguments?|parameters?|"
    r"what do (i|we|you) need|requirements? (to|for)|dispute|refund|freeze|replac|verif|escalat|"
    r"unlock|reorder|reimburse|waive|(to|i|we|customer wants to|can i) (open|close|apply|file|submit|transfer|"
    r"cancel|upgrade|downgrade|report|order|request|add|remove|change|update))", re.I)
COMPARE_WORDS = re.compile(r"\b(compare|best|which|highest|lowest|all|every|list|rank|options?|eligible)\b", re.I)

SYNTH_SYSTEM = (
    "You are an internal knowledge service. Answer ONLY from the evidence provided. "
    "Copy exact tool names, argument names, enum strings, amounts and dates verbatim. "
    "When a procedure applies, list its steps in order with the exact tools. "
    "If evidence conflicts, say so and cite both. If the evidence does not contain the answer, say "
    "'Not found in knowledge base' and list the closest relevant facts. Cite sources as [doc_id]."
)


def _category_terms(cat: str) -> list[str]:
    words = cat.replace("_", " ")
    terms = {words, words.rstrip("s")}
    if words.endswith("accounts"):
        terms.add(words[: -len(" accounts")])
    return sorted(terms, key=len, reverse=True)


class Engine:
    """Holds loaded contexts and answers queries."""

    def __init__(self, root: str, default_model: str | None = None):
        self.root = root
        self.contexts: dict[str, Context] = {}
        self.default_model = default_model or os.environ.get("NEXUS_LITE_QUERY_MODEL") or None

    def context(self, slug: str) -> Context:
        if slug not in self.contexts:
            self.contexts[slug] = Context(self.root, slug)
        return self.contexts[slug]

    # ------------------------------------------------------------------ routing
    def route(self, ctx: Context, ask: str, filt: dict | None = None) -> dict:
        low = ask.lower()
        ents = ctx.sql("SELECT entity, entity_name, category, aliases FROM entities")
        matched: list[tuple[int, str]] = []
        for e in ents:
            names = [e["entity_name"]] + json.loads(e["aliases"] or "[]")
            best = 0
            for n in names:
                n_low = re.sub(r"\s*\(general\)", "", n.lower()).strip()
                if len(n_low) >= 3 and re.search(r"(?<![a-z])" + re.escape(n_low) + r"(?![a-z])", low):
                    best = max(best, len(n_low))
            if best and "(general)" not in e["entity_name"].lower():
                matched.append((best, e["entity"]))
        matched.sort(reverse=True)
        # drop entities whose name is a substring of a longer matched one ("Gold" inside "Gold Plus")
        names = {e["entity"]: e["entity_name"].lower() for e in ents}
        entities: list[str] = []
        for _, ent in matched:
            if not any(names[ent] in names[o] and ent != o for o in entities):
                entities.append(ent)
        cats = [c for c in {e["category"] for e in ents}
                if any(re.search(r"(?<![a-z])" + re.escape(t) + r"s?(?![a-z])", low) for t in _category_terms(c))]
        # a "checking" hit inside "business checking" should not also select personal checking
        cats = [c for c in cats if not any(o != c and o.endswith(c) for o in cats)]
        if filt:
            if filt.get("entity"):
                entities = [e for e in filt["entity"]]
            if filt.get("category"):
                cats = list(filt["category"])
        tools = sorted(set(TOOL_RE.findall(ask)))
        return {
            "entities": entities[:4],
            "categories": cats,
            "tools": tools,
            "procedural": bool(PROCEDURE_WORDS.search(ask)),
            "comparative": bool(COMPARE_WORDS.search(ask)) or len(entities) > 2,
        }

    # ------------------------------------------------------------------ compose
    def gather(self, ctx: Context, ask: str, plan: dict, depth: str, max_chars: int) -> tuple[str, list[str]]:
        k = DEPTH_K.get(depth, 8)
        sections: list[tuple[str, str, list[str]]] = []  # (label, body, sources)
        hits = ctx.index.search(ask, k=k * 2)
        hit_docs: list[str] = []
        for cid, _ in hits:
            d = ctx.chunk_doc[cid]
            if d not in hit_docs:
                hit_docs.append(d)

        # 1) entity fact sheets (only for 1-2 specific entities; bigger sets use rosters)
        sheet_docs: set[str] = set()
        if plan["entities"] and len(plan["entities"]) <= 2:
            per = max_chars // (2 * len(plan["entities"]))
            for ent in plan["entities"]:
                art = ctx.artifact(f"fact_sheet:{ent}")
                if art:
                    body = art["body"]
                    if len(body) > per:  # keep key figures + the docs the search ranked highest
                        body = self._focus_sheet(ctx, ent, body, hit_docs, per)
                    srcs = [s for s in json.loads(art["sources"]) if f"[{s}]" in body]
                    sheet_docs.update(srcs)
                    sections.append((f"Fact sheet: {ent}", body, srcs))

        # 2) rosters for comparative / category asks
        if plan["comparative"] or (plan["categories"] and not plan["entities"]):
            for cat in plan["categories"][:2]:
                art = ctx.artifact(f"roster:{cat}")
                if art:
                    sections.append((f"Roster: {cat}", art["body"][: max_chars // 3], json.loads(art["sources"])))

        # 3) procedures from top docs (+ explicitly named tools)
        if plan["procedural"] or plan["tools"]:
            # Procedure docs are short and their enum lists / sub-bullets matter, so include the
            # full source of the top-ranked docs that contain numbered steps or internal tools.
            proc_docs: list[str] = []
            for d in hit_docs[:k]:
                if d in sheet_docs:
                    continue
                n = ctx.sql("SELECT COUNT(*) AS n FROM procedures WHERE doc_id=?", (d,))[0]["n"]
                if n or TOOL_RE.search(ctx.read_source(d) or ""):
                    proc_docs.append(d)
                if len(proc_docs) >= 3:
                    break
            for t in plan["tools"]:
                for row in ctx.sql("SELECT doc_ids FROM tools WHERE tool_name=?", (t,)):
                    for d in json.loads(row["doc_ids"])[:2]:
                        if d not in proc_docs and d not in sheet_docs:
                            proc_docs.append(d)
            proc_text = "\n\n".join(ctx.read_source(d) for d in proc_docs)
            if proc_text:
                sections.append(("Procedures", proc_text, proc_docs))
                sheet_docs.update(proc_docs)
            tools = list(plan["tools"])
            tools += [t for t in dict.fromkeys(TOOL_RE.findall(proc_text)) if t not in tools]
            for t in tools[:3]:
                art = ctx.artifact(f"tool:{t}")
                if art:
                    sections.append((f"Tool card: {t}", art["body"][:1200], json.loads(art["sources"])))

        # 4) hybrid chunk hits not already covered by a fact sheet
        chunk_lines = []
        chunk_srcs = []
        for cid, _ in hits[: k + 4]:
            d = ctx.chunk_doc[cid]
            if d in sheet_docs:
                continue
            chunk_lines.append(f"--- [{d}] ---\n{ctx.chunk_text[cid]}")
            chunk_srcs.append(d)
            if len(chunk_lines) >= k:
                break
        if chunk_lines:
            sections.append(("Search results", "\n\n".join(chunk_lines), chunk_srcs))

        # budget: fill in priority order, truncate the last section that fits partially
        out, sources, used = [], [], 0
        prio = ({"Rost": 0, "Fact": 1, "Proc": 2, "Sear": 3, "Tool": 4} if plan["comparative"]
                else {"Proc": 0, "Fact": 1, "Tool": 2, "Sear": 3, "Rost": 4})
        order = sorted(sections, key=lambda s: prio[s[0][:4]])
        for label, body, srcs in order:
            room = max_chars - used
            if room < 400:
                break
            block = f"## {label}\n{body}"
            if len(block) > room:
                block = block[:room] + "\n(... truncated for budget)"
            out.append(block)
            used += len(block)
            sources += [s for s in srcs if s not in sources and f"[{s}]" in block]
        return "\n\n".join(out), sources

    @staticmethod
    def _focus_sheet(ctx: Context, ent: str, body: str, hit_docs: list[str], limit: int) -> str:
        head, _, rest = body.partition("\n## Source documents\n")
        parts = re.split(r"\n(?=### )", rest)
        ranked = sorted(parts, key=lambda p: next((i for i, d in enumerate(hit_docs) if f"[{d}]" in p), 999))
        out = head[: limit // 3] + "\n## Most relevant source documents\n"
        for p in ranked:
            if len(out) + len(p) > limit:
                break
            out += "\n" + p
        return out

    # ------------------------------------------------------------------ answer
    def query(self, req: dict) -> dict:
        t0 = time.time()
        qid = req.get("id") or f"q_{uuid.uuid4().hex[:12]}"
        scope = req.get("scope") or []
        ask = (req.get("ask") or "").strip()
        # An answer-shape directive ("[Answer requirements: ...]") shapes synthesis only;
        # it must not steer routing or search.
        m = re.search(r"\n*\[Answer requirements:(.*)\]\s*$", ask, re.S)
        directive = m.group(1).strip() if m else ""
        if m:
            ask = ask[: m.start()].strip()
        if not scope or not ask:
            return {"id": qid, "status": "failed", "error": "ask and scope are required"}
        budget = req.get("budget") or {}
        depth = budget.get("depth", "standard")
        max_chars = int(budget.get("max_chars", DEFAULT_MAX_CHARS))
        model = req.get("model") or self.default_model

        blocks, sources, plans = [], [], {}
        for slug in scope:
            ctx = self.context(slug)
            plan = self.route(ctx, ask, req.get("filter"))
            plans[slug] = plan
            text, srcs = self.gather(ctx, ask, plan, depth, max_chars // len(scope))
            blocks.append(text)
            sources += [s for s in srcs if s not in sources]
        evidence = "\n\n".join(blocks)
        usage: dict = {"evidence_chars": len(evidence), "evidence_tokens_est": len(evidence) // 4}

        if model:
            answer, u = self._synthesize(model, ask, evidence, req.get("shape"), directive)
            usage.update(u)
        else:
            answer = f"Compiled knowledge for: {ask}\n\n{evidence}" if evidence else "Not found in knowledge base."

        titles = {}
        for slug in scope:
            titles.update(self.context(slug).doc_title)
        citations = [{"id": s, "title": titles.get(s, s)} for s in sources] if req.get("ground", True) else []
        return {
            "id": qid,
            "status": "completed",
            "model": model or "evidence",
            "output": [{"content": [{"type": "output_text", "text": answer}]}],
            "citations": citations,
            "usage": usage,
            "runtime_ms": int((time.time() - t0) * 1000),
            "plan": plans,
        }

    def _synthesize(self, model: str, ask: str, evidence: str, shape: dict | None,
                    directive: str = "") -> tuple[str, dict]:
        import litellm

        user = f"QUESTION:\n{ask}\n\nEVIDENCE:\n{evidence}"
        if directive:
            user += f"\n\nANSWER REQUIREMENTS:\n{directive}"
        kwargs: dict = {}
        if shape:
            user += f"\n\nReturn ONLY JSON matching this JSON Schema:\n{json.dumps(shape)}"
            kwargs["response_format"] = {"type": "json_object"}
        resp = litellm.completion(model=model, messages=[{"role": "system", "content": SYNTH_SYSTEM},
                                                         {"role": "user", "content": user}], **kwargs)
        text = resp.choices[0].message.content or ""
        u = getattr(resp, "usage", None)
        usage = {"input_tokens": getattr(u, "prompt_tokens", None), "output_tokens": getattr(u, "completion_tokens", None)}
        try:
            usage["cost_usd"] = litellm.completion_cost(completion_response=resp)
        except Exception:
            pass
        return text, usage
