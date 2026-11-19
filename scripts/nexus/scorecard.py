#!/usr/bin/env python3
"""Scorecard for banking_knowledge runs (ours vs a baseline).

Metrics (Sierra conventions):
  pass^k     mean over tasks of C(successes, k) / C(trials, k)   (pass^1 = mean success rate)
  trial-1    share of tasks whose first trial succeeded
  per task:  tool calls, model (assistant) calls, KB queries, agent $, user $, knowledge-layer $

Usage:
  python scripts/nexus/scorecard.py OURS.json [--baseline BASE.json] [--usage OURS.nexus_usage.jsonl]
         [--label-ours nexus] [--label-base alltools] [--out SCORECARD.md]
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from math import comb
from pathlib import Path

KB_TOOLS = {"KB_query", "KB_search", "KB_search_bm25", "KB_search_dense", "shell", "grep",
            "KB_orient", "KB_sql", "KB_read_artifact", "KB_read_source", "KB_search_passages"}


def load(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def sim_stats(sim: dict) -> dict:
    tool_calls = model_calls = kb = 0
    for m in sim.get("messages") or []:
        if m.get("role") != "assistant":
            continue
        model_calls += 1
        for tc in m.get("tool_calls") or []:
            tool_calls += 1
            if tc.get("name") in KB_TOOLS:
                kb += 1
    reward = (sim.get("reward_info") or {}).get("reward") or 0.0
    return {"task": sim["task_id"], "trial": sim.get("trial", 0), "ok": reward >= 1 - 1e-6,
            "tool_calls": tool_calls, "model_calls": model_calls, "kb_calls": kb,
            "agent_usd": sim.get("agent_cost") or 0.0, "user_usd": sim.get("user_cost") or 0.0}


def summarize(results: dict, usage_path: str | None = None) -> dict:
    rows = [sim_stats(s) for s in results["simulations"]]
    by_task: dict[str, list] = defaultdict(list)
    for r in rows:
        by_task[r["task"]].append(r)
    n_trials = max(len(v) for v in by_task.values())
    pass_k = {}
    for k in range(1, n_trials + 1):
        vals = [comb(sum(r["ok"] for r in v), k) / comb(len(v), k) for v in by_task.values() if len(v) >= k]
        pass_k[k] = 100 * sum(vals) / len(vals)
    first = {t: min(v, key=lambda r: r["trial"]) for t, v in by_task.items()}
    kb_usd: dict[str, float] = defaultdict(float)
    kb_q: dict[str, int] = defaultdict(int)
    if usage_path and Path(usage_path).exists():
        for line in open(usage_path, encoding="utf-8"):
            rec = json.loads(line)
            kb_q[rec.get("task_id")] += 1
            kb_usd[rec.get("task_id")] += rec.get("cost_usd") or 0.0
    n = len(rows)
    return {
        "tasks": len(by_task), "trials": n_trials, "sims": n, "pass_k": pass_k,
        "trial1": 100 * sum(r["ok"] for r in first.values()) / len(first),
        "avg_tool_calls": sum(r["tool_calls"] for r in rows) / n,
        "avg_model_calls": sum(r["model_calls"] for r in rows) / n,
        "avg_kb_calls": sum(r["kb_calls"] for r in rows) / n,
        "agent_usd_per_task": sum(r["agent_usd"] for r in rows) / n,
        "user_usd_per_task": sum(r["user_usd"] for r in rows) / n,
        "kb_usd_total": sum(kb_usd.values()), "kb_queries_logged": sum(kb_q.values()),
        "first": first, "by_task": by_task,
        "retrieval": (results.get("info") or {}).get("retrieval_config"),
        "agent": ((results.get("info") or {}).get("agent_info") or {}).get("llm"),
    }


def fmt(s: dict, label: str) -> list[str]:
    pk = " · ".join(f"pass^{k} {v:.2f}%" for k, v in s["pass_k"].items())
    return [f"**{label}** ({s['agent']}, retrieval `{s['retrieval']}`): {s['tasks']} tasks × {s['trials']} trials",
            f"- {pk}", f"- trial-1 {s['trial1']:.2f}%",
            f"- per sim: {s['avg_tool_calls']:.1f} tool calls · {s['avg_model_calls']:.1f} model calls · "
            f"{s['avg_kb_calls']:.1f} KB calls",
            f"- $ per sim: agent {s['agent_usd_per_task']:.3f} · user {s['user_usd_per_task']:.3f}"
            + (f" · knowledge layer total {s['kb_usd_total']:.2f} ({s['kb_queries_logged']} queries)"
               if s["kb_queries_logged"] else ""), ""]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ours")
    ap.add_argument("--baseline")
    ap.add_argument("--usage")
    ap.add_argument("--label-ours", default="ours")
    ap.add_argument("--label-base", default="baseline")
    ap.add_argument("--out")
    a = ap.parse_args()

    ours = summarize(load(a.ours), a.usage)
    lines = ["# Scorecard: banking_knowledge", ""] + fmt(ours, a.label_ours)
    if a.baseline:
        base = summarize(load(a.baseline))
        lines += fmt(base, a.label_base)
        common = sorted(set(ours["first"]) & set(base["first"]))
        win = sum(ours["first"][t]["ok"] and not base["first"][t]["ok"] for t in common)
        lose = sum(base["first"][t]["ok"] and not ours["first"][t]["ok"] for t in common)
        both = sum(ours["first"][t]["ok"] and base["first"][t]["ok"] for t in common)
        lines += [f"Head-to-head on trial-1 over {len(common)} common tasks: **+{win} / -{lose}** "
                  f"(both pass {both}, net {win - lose:+d})", "",
                  "| task | ours | base | ours tools | base tools | ours KB | base KB | ours $ | base $ |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for t in common:
            o, b = ours["first"][t], base["first"][t]
            lines.append(f"| {t} | {int(o['ok'])} | {int(b['ok'])} | {o['tool_calls']} | {b['tool_calls']} | "
                         f"{o['kb_calls']} | {b['kb_calls']} | {o['agent_usd']:.2f} | {b['agent_usd']:.2f} |")
    text = "\n".join(lines) + "\n"
    if a.out:
        Path(a.out).write_text(text, encoding="utf-8")
    print(text if len(text) < 6000 else "\n".join(lines[:20]))


if __name__ == "__main__":
    main()
