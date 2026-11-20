#!/usr/bin/env python3
"""Generate a labeled question set for evaluating a knowledge engine on a document corpus.

Each item: {id, type, question, answer, key_facts[], gold_docs[]}
- single:    one fact from one document
- compare:   choose/compare across 2-3 products of the same category (multi-document)
- procedure: steps / exact internal tool names from a procedure document

An LLM writes the question from the given documents; every key fact must appear verbatim in the gold
documents or the item is dropped, so the labels are grounded in the source text.

    python scripts/nexus/kb_eval/gen_questions.py --docs data/tau2/domains/banking_knowledge/documents \
        --out data/kb_eval/banking_questions.json --single 25 --compare 20 --procedure 15 --model gemini/gemini-3.1-flash-lite
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

PROMPTS = {
    "single": "Write ONE question a bank customer-support agent might need answered, whose answer is a specific "
              "fact (an amount, rate, limit, count, eligibility rule or name) stated in the document.",
    "compare": "Write ONE question that requires comparing the products in these documents (for example which has "
               "the lowest fee / highest rate / which qualifies under a condition) and naming the winner with its "
               "value. The answer must need at least two of the documents.",
    "procedure": "Write ONE question about how an agent must handle a situation described in the document: the "
                 "required steps, conditions, or the exact internal tool name(s) to use.",
}

INSTRUCTIONS = """You create evaluation questions for a bank's internal knowledge base.
{task}
Rules:
- The question must be answerable from the documents alone and must NOT mention document titles or ids.
- Name the product explicitly when the question is about a product.
- key_facts: 1-3 SHORT strings copied VERBATIM from the documents that a correct answer must contain
  (e.g. "$200.00", "2.5%", "Business Gold Rewards Card", "open_bank_account_4821"). No paraphrases.
Return JSON: {{"question": "...", "answer": "one or two sentences", "key_facts": ["..."]}}

DOCUMENTS:
{docs}"""


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower()).strip()


def load_docs(d: Path) -> list[dict]:
    out = []
    for f in sorted(d.glob("*.json")):
        doc = json.loads(f.read_text(encoding="utf-8"))
        stem = re.sub(r"_\d+$", "", doc["id"].removeprefix("doc_"))
        doc["group"] = stem
        out.append(doc)
    return out


def ask(model: str, prompt: str) -> dict | None:
    import litellm

    for attempt in range(5):
        try:
            r = litellm.completion(model=model, messages=[{"role": "user", "content": prompt}],
                                   response_format={"type": "json_object"}, temperature=0.4)
            return json.loads(r.choices[0].message.content)
        except json.JSONDecodeError:
            return None
        except Exception as e:  # rate limits / overload
            msg = str(e)
            if "quota" in msg.lower() and "day" in msg.lower():
                sys.exit(f"daily quota exhausted: {msg[:200]}")
            time.sleep(min(60, 5 * 2 ** attempt))
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="gemini/gemini-3.1-flash-lite")
    ap.add_argument("--single", type=int, default=25)
    ap.add_argument("--compare", type=int, default=20)
    ap.add_argument("--procedure", type=int, default=15)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    from dotenv import load_dotenv

    load_dotenv(".env")
    rng = random.Random(a.seed)
    docs = load_docs(Path(a.docs))
    by_cat = defaultdict(lambda: defaultdict(list))  # category -> product group -> docs
    for d in docs:
        cat = d["group"].split("_")[0] + "_" + d["group"].split("_")[1] if "_" in d["group"] else d["group"]
        by_cat[cat][d["group"]].append(d)
    proc_docs = [d for d in docs if re.search(r"\b[a-z]+(?:_[a-z]+)+_\d{4}\b", d["content"])]
    fact_docs = [d for d in docs if re.search(r"\$\d|\d%", d["content"]) and "(general)" not in d["group"]]

    out_path = Path(a.out)
    items = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else []
    have = defaultdict(int)
    for it in items:
        have[it["type"]] += 1
    targets = {"single": a.single, "compare": a.compare, "procedure": a.procedure}
    used = {tuple(it["gold_docs"]) for it in items}
    tries = 0
    while any(have[t] < n for t, n in targets.items()) and tries < 4 * sum(targets.values()):
        tries += 1
        kind = next(t for t, n in targets.items() if have[t] < n)
        if kind == "single":
            group = [rng.choice(fact_docs)]
        elif kind == "procedure":
            group = [rng.choice(proc_docs)]
        else:
            cats = [c for c, g in by_cat.items() if len([k for k in g if "(general)" not in k]) >= 3]
            prods = [k for k in by_cat[rng.choice(cats)] if "(general)" not in k]
            picks = rng.sample(prods, min(len(prods), rng.choice([2, 3])))
            cat = next(c for c, g in by_cat.items() if picks[0] in g)
            # the overview doc (_001) of each product carries its headline figures
            group = [sorted(by_cat[cat][p], key=lambda d: d["id"])[0] for p in picks]
        key = tuple(d["id"] for d in group)
        if key in used:
            continue
        body = "\n\n".join(f"### {d['title']}\n{d['content'][:6000]}" for d in group)
        res = ask(a.model, INSTRUCTIONS.format(task=PROMPTS[kind], docs=body))
        if not res or not res.get("question") or not res.get("key_facts"):
            continue
        corpus = norm(" ".join(d["title"] + " " + d["content"] for d in group))
        facts = [f for f in res["key_facts"] if isinstance(f, str) and f.strip() and norm(f) in corpus]
        if not facts or len(facts) < len(res["key_facts"]) * 0.5:
            continue  # ungrounded labels -> drop
        items.append({"id": f"q{len(items) + 1:03d}", "type": kind, "question": res["question"].strip(),
                      "answer": str(res.get("answer", "")).strip(), "key_facts": facts,
                      "gold_docs": list(key)})
        used.add(key)
        have[kind] += 1
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(items, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"[{kind}] {items[-1]['question'][:110]}  facts={facts}", flush=True)
    print(f"{len(items)} questions -> {out_path} ({dict(have)})")


if __name__ == "__main__":
    main()
