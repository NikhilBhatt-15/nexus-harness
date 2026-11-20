#!/usr/bin/env python3
"""Evaluate a compiled nexus-lite context against plain retrieval baselines.

Systems (same evidence budget in characters):
  nexus   nexus-lite query(): routing + compiled artifacts (fact sheets, rosters, procedures) + hybrid search
  hybrid  plain BM25 + dense chunks fused with RRF (the same index, without compiled artifacts or routing)
  bm25    plain BM25 chunks
  dense   plain dense-embedding chunks

Retrieval metrics (no LLM):
  doc_recall    share of gold documents present in the evidence
  facts_in_ctx  share of questions whose evidence contains ALL key facts (the answer is reachable)
  chars         evidence size (≈ tokens × 4), latency

End-to-end (--e2e MODEL): the model answers from each system's evidence; correct = all key facts in the answer.

    python scripts/nexus/kb_eval/run_eval.py --questions data/kb_eval/banking_questions.json --slug banking \
        --budget 8000 --e2e gemini/gemini-3.1-flash-lite --out docs/nexus/KB_EVAL.md
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

from nexus_lite.context import Context
from nexus_lite.query import Engine

ANSWER_PROMPT = ("You answer questions for bank support agents using ONLY the context below. Reply in one or two "
                 "sentences and include the exact numbers, product names and tool names from the context. If the "
                 "context does not contain the answer, reply exactly: NOT FOUND.\n\nCONTEXT:\n{ctx}\n\nQUESTION: {q}")


def norm(s: str) -> str:
    s = s.lower().replace(",", "").replace("`", "").replace("*", "")
    s = re.sub(r"(\d)\.00\b", r"\1", s)  # $25.00 == $25
    return re.sub(r"\s+", " ", s).strip()


def has_all(text: str, facts: list[str]) -> bool:
    t = norm(text)
    return all(norm(f) in t for f in facts)


def chunk_evidence(ctx: Context, hits: list[tuple[str, float]], budget: int) -> tuple[str, list[str]]:
    parts, docs, used = [], [], 0
    for cid, _ in hits:
        block = f"--- [{ctx.chunk_doc[cid]}] ---\n{ctx.chunk_text[cid]}"
        if used + len(block) > budget:
            if not parts:
                parts.append(block[:budget])
                docs.append(ctx.chunk_doc[cid])
            break
        parts.append(block)
        used += len(block)
        if ctx.chunk_doc[cid] not in docs:
            docs.append(ctx.chunk_doc[cid])
    return "\n\n".join(parts), docs


def retrieve(system: str, q: str, engine: Engine, ctx: Context, slug: str, budget: int) -> tuple[str, list[str]]:
    if system == "nexus":
        r = engine.query({"scope": [slug], "ask": q, "budget": {"max_chars": budget}})
        text = r["output"][0]["content"][0]["text"]
        docs = [c["id"] for c in r.get("citations", [])]
        docs += [d for d in re.findall(r"\[(doc_[^\]\s]+)\]", text) if d not in docs]
        return text, docs
    k = 40
    if system == "bm25":
        hits = [(ctx.index.ids[i], s) for i, s in ctx.index.search_bm25(q, k)]
    elif system == "dense":
        hits = [(ctx.index.ids[i], s) for i, s in ctx.index.search_dense(q, k)]
    else:
        hits = ctx.index.search(q, k=k)
    return chunk_evidence(ctx, hits, budget)


def ambiguous(item: dict, titles: dict[str, str]) -> bool:
    """Comparison questions that do not name at least two of the compared products can't be graded against
    only the sampled gold docs (the true answer depends on the whole catalog)."""
    if item["type"] != "compare":
        return False
    q = item["question"].lower()
    names = {titles.get(d, "").split(":")[0].strip().lower() for d in item["gold_docs"]}
    named = sum(1 for n in names if n and (n in q or n.replace(" account", "") in q))
    return named < 2


def answer(model: str, ctx: str, q: str) -> tuple[str, float]:
    from nexus_agent.llm import LLM  # rotates <PROVIDER>_API_KEYS and retries rate limits

    for attempt in range(6):
        try:
            t0 = time.time()
            c = LLM(model, temperature=0).complete([{"role": "user", "content": ANSWER_PROMPT.format(ctx=ctx, q=q)}])
            return c.text or "", time.time() - t0
        except Exception as e:
            if "quota" in str(e).lower() and ("day" in str(e).lower() or "retry in" in str(e).lower() and "h" in str(e)):
                raise
            time.sleep(min(60, 5 * 2 ** attempt))
    return "", 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", required=True)
    ap.add_argument("--slug", default="banking")
    ap.add_argument("--root", default="contexts")
    ap.add_argument("--budget", type=int, default=8000)
    ap.add_argument("--systems", default="nexus,hybrid,bm25,dense")
    ap.add_argument("--e2e", help="litellm model for end-to-end answering")
    ap.add_argument("--e2e-systems", default="nexus,hybrid,bm25,dense")
    ap.add_argument("--include-ambiguous", action="store_true")
    ap.add_argument("--out")
    a = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    from dotenv import load_dotenv

    load_dotenv(".env")
    items = json.loads(Path(a.questions).read_text(encoding="utf-8"))
    engine = Engine(a.root)
    ctx = engine.context(a.slug)
    titles = dict(ctx.conn.execute("SELECT doc_id, title FROM documents").fetchall())
    skipped = [it["id"] for it in items if ambiguous(it, titles)]
    if not a.include_ambiguous:
        items = [it for it in items if it["id"] not in skipped]
    systems = a.systems.split(",")
    rows: list[dict] = []
    for it in items:
        for sysname in systems:
            t0 = time.time()
            text, docs = retrieve(sysname, it["question"], engine, ctx, a.slug, a.budget)
            rows.append({"id": it["id"], "type": it["type"], "system": sysname, "ms": int((time.time() - t0) * 1000),
                         "chars": len(text), "doc_recall": sum(d in docs for d in it["gold_docs"]) / len(it["gold_docs"]),
                         "facts_in_ctx": has_all(text, it["key_facts"]), "_ctx": text})
        print(f"{it['id']} retrieved", flush=True, end="\r")
    if a.e2e:
        e2e_sys = a.e2e_systems.split(",")
        for r in rows:
            if r["system"] not in e2e_sys:
                continue
            it = next(x for x in items if x["id"] == r["id"])
            try:
                ans, sec = answer(a.e2e, r["_ctx"], it["question"])
            except Exception as e:
                print(f"\nstopping end-to-end: {str(e)[:200]}")
                break
            r["answer"], r["correct"], r["answer_s"] = ans.strip(), has_all(ans, it["key_facts"]), round(sec, 2)
            print(f"{r['id']} {r['system']:6s} {'OK ' if r['correct'] else 'BAD'} {ans.strip()[:90]!r}", flush=True)

    def agg(rs: list[dict]) -> dict:
        n = len(rs)
        e = [r for r in rs if "correct" in r]
        return {"n": n, "doc_recall": sum(r["doc_recall"] for r in rs) / n,
                "facts_in_ctx": sum(r["facts_in_ctx"] for r in rs) / n,
                "chars": sum(r["chars"] for r in rs) / n, "ms": sum(r["ms"] for r in rs) / n,
                "e2e": (sum(r["correct"] for r in e) / len(e)) if e else None, "e2e_n": len(e)}

    lines = [f"# Knowledge engine eval: `{a.slug}`", "",
             f"{len(items)} questions ({len(skipped)} ambiguous comparison questions "
             f"{'included' if a.include_ambiguous else 'excluded'}), evidence budget {a.budget} chars"
             + (f", end-to-end model `{a.e2e}`" if a.e2e else "") + ".", "",
             "| system | gold-doc recall | answer in context | end-to-end correct | avg chars | avg ms |",
             "|---|---:|---:|---:|---:|---:|"]
    for s in systems:
        g = agg([r for r in rows if r["system"] == s])
        e2e = f"{100 * g['e2e']:.0f}% ({g['e2e_n']})" if g["e2e"] is not None else "-"
        lines.append(f"| {s} | {100 * g['doc_recall']:.0f}% | {100 * g['facts_in_ctx']:.0f}% | {e2e} | "
                     f"{g['chars']:.0f} | {g['ms']:.0f} |")
    lines += ["", "## By question type", "", "| type | system | n | gold-doc recall | answer in context | end-to-end |",
              "|---|---|---:|---:|---:|---:|"]
    for t in ("single", "compare", "procedure"):
        for s in systems:
            rs = [r for r in rows if r["system"] == s and r["type"] == t]
            if rs:
                g = agg(rs)
                e2e = f"{100 * g['e2e']:.0f}%" if g["e2e"] is not None else "-"
                lines.append(f"| {t} | {s} | {g['n']} | {100 * g['doc_recall']:.0f}% | {100 * g['facts_in_ctx']:.0f}% | {e2e} |")
    misses = [r for r in rows if r["system"] == "nexus" and not r["facts_in_ctx"]]
    if misses:
        lines += ["", "## nexus misses (answer not in context)", ""]
        for r in misses:
            it = next(x for x in items if x["id"] == r["id"])
            lines.append(f"- {r['id']} ({r['type']}): {it['question']}  — needs {it['key_facts']}, "
                         f"gold docs found {r['doc_recall']:.0%}")
    report = "\n".join(lines) + "\n"
    print("\n" + report)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(report, encoding="utf-8")
        Path(a.out).with_suffix(".json").write_text(
            json.dumps([{k: v for k, v in r.items() if k != "_ctx"} for r in rows], indent=1, ensure_ascii=False),
            encoding="utf-8")


if __name__ == "__main__":
    main()
