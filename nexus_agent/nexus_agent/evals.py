"""Task-suite evals: run tasks with automatic checks, compare models.

suite.yaml::

    name: ops-basics
    model: gemini/gemini-3.1-flash-lite      # default; override/compare with -m A -m B
    trials: 1
    tasks:
      - id: top-customers
        task: Find the 3 customers with the highest total order value and write their names to top3.txt
        workspace: fixtures/shop             # copied to a fresh temp dir per trial (relative to suite file)
        tools: files,sqlite=shop.db          # tool spec, resolved inside the workspace
        max_steps: 20
        checks:
          - file_exists: top3.txt
          - file_contains: {path: top3.txt, text: Alice}
          - answer_contains: Alice
          - answer_regex: "\\b3\\b"
          - sql: {db: shop.db, query: "SELECT COUNT(*) FROM orders", equals: 12}
          - command: {run: "python check.py", exit_code: 0}
          - max_tool_calls: 15
          - judge: "The answer names the top 3 customers with their totals."   # LLM judge (uses judge_model)

A task passes when every check passes. Results: per-model pass rate, steps, tool calls, cost, time.
"""

from __future__ import annotations

import json
import re
import shutil
import sqlite3
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import yaml

from nexus_agent.agent import Agent, RunResult
from nexus_agent.config import parse_tool_spec


def run_check(check: dict, ws: Path, result: RunResult, judge_model: str | None) -> tuple[bool, str]:
    (kind, spec), = check.items()
    answer = result.text or ""
    if kind == "file_exists":
        return (ws / spec).exists(), f"{spec} exists"
    if kind == "file_contains":
        p = ws / spec["path"]
        ok = p.exists() and spec["text"].lower() in p.read_text(encoding="utf-8", errors="ignore").lower()
        return ok, f"{spec['path']} contains {spec['text']!r}"
    if kind == "answer_contains":
        items = spec if isinstance(spec, list) else [spec]
        return all(str(s).lower() in answer.lower() for s in items), f"answer contains {items}"
    if kind == "answer_regex":
        return bool(re.search(spec, answer, re.I | re.S)), f"answer matches /{spec}/"
    if kind == "sql":
        con = sqlite3.connect(ws / spec["db"])
        try:
            row = con.execute(spec["query"]).fetchone()
        finally:
            con.close()
        val = row[0] if row and len(row) == 1 else (list(row) if row else None)
        return val == spec["equals"], f"sql {spec['query']!r} == {spec['equals']!r} (got {val!r})"
    if kind == "command":
        p = subprocess.run(spec["run"], shell=True, cwd=ws, capture_output=True, text=True, timeout=300)
        return p.returncode == spec.get("exit_code", 0), f"`{spec['run']}` exit {p.returncode}"
    if kind == "max_tool_calls":
        return result.tool_calls <= int(spec), f"tool calls {result.tool_calls} <= {spec}"
    if kind == "stop_reason":
        return result.stop_reason == spec, f"stop_reason == {spec}"
    if kind == "judge":
        if not judge_model:
            return False, "judge check needs judge_model"
        from nexus_agent.llm import LLM

        verdict = LLM(judge_model).complete([
            {"role": "system", "content": "You grade an agent's final answer. Reply PASS or FAIL then one sentence."},
            {"role": "user", "content": f"Criterion: {spec}\n\nAgent answer:\n{answer[:8000]}"}]).text
        return verdict.strip().upper().startswith("PASS"), f"judge: {verdict.strip()[:160]}"
    return False, f"unknown check {kind}"


def run_task(task: dict, model: str, suite_dir: Path, trial: int, judge_model: str | None) -> dict:
    with tempfile.TemporaryDirectory(prefix="nxa-eval-", ignore_cleanup_errors=True) as tmp:
        ws = Path(tmp)
        if task.get("workspace"):
            shutil.copytree(suite_dir / task["workspace"], ws, dirs_exist_ok=True)
        for cmd in task.get("setup", []) or []:
            subprocess.run(cmd, shell=True, cwd=ws, check=True, capture_output=True)
        toolsets = parse_tool_spec(task.get("tools", "files"), str(ws))
        # paths in specs are relative to the workspace
        agent = Agent(model, toolsets, workdir=ws, permissions=task.get("permissions", "yolo"),
                      max_steps=task.get("max_steps", 30), system=task.get("system"))
        t0 = time.time()
        res = agent.run(task["task"])
        elapsed = time.time() - t0
        checks = []
        for c in task.get("checks", []) or []:
            try:
                ok, desc = run_check(c, ws, res, judge_model)
            except Exception as e:
                ok, desc = False, f"{next(iter(c))}: error {e}"
            checks.append({"ok": ok, "check": desc})
        for ts in list(agent.toolsets):
            agent.detach(ts)  # close db/mcp handles before the temp dir is removed
    passed = res.stop_reason == "done" and all(c["ok"] for c in checks)
    return {"task": task["id"], "model": model, "trial": trial, "passed": passed, "stop_reason": res.stop_reason,
            "steps": res.steps, "tool_calls": res.tool_calls, "cost_usd": res.cost_usd, "seconds": round(elapsed, 1),
            "checks": checks, "answer": (res.text or "")[:2000], "error": res.error}


def run_suite(suite_path: str, models: list[str] | None = None) -> list[dict]:
    path = Path(suite_path).resolve()
    suite = yaml.safe_load(path.read_text(encoding="utf-8"))
    models = models or [suite["model"]]
    rows = []
    for model in models:
        for t in suite["tasks"]:
            for trial in range(int(t.get("trials", suite.get("trials", 1)))):
                r = run_task(t, model, path.parent, trial, suite.get("judge_model"))
                rows.append(r)
                print(f"{'PASS' if r['passed'] else 'FAIL'}  {model}  {t['id']}#{trial}  "
                      f"{r['steps']} steps, {r['tool_calls']} tools, ${r['cost_usd']:.4f}, {r['seconds']}s"
                      + ("" if r["passed"] else "  <- " + "; ".join(c["check"] for c in r["checks"] if not c["ok"])
                         + (f" [{r['stop_reason']}: {r['error']}]" if r["error"] else "")), flush=True)
    return rows


def scorecard(rows: list[dict]) -> str:
    out = ["| model | pass rate | avg steps | avg tool calls | total $ | avg s |", "|---|---:|---:|---:|---:|---:|"]
    for model in dict.fromkeys(r["model"] for r in rows):
        rs = [r for r in rows if r["model"] == model]
        n = len(rs)
        out.append(f"| {model} | {100 * sum(r['passed'] for r in rs) / n:.0f}% ({sum(r['passed'] for r in rs)}/{n}) "
                   f"| {sum(r['steps'] for r in rs) / n:.1f} | {sum(r['tool_calls'] for r in rs) / n:.1f} "
                   f"| {sum(r['cost_usd'] for r in rs):.4f} | {sum(r['seconds'] for r in rs) / n:.0f} |")
    return "\n".join(out)


def main(suite: str, models: list[str] | None, out: str | None) -> None:
    rows = run_suite(suite, models)
    card = scorecard(rows)
    print("\n" + card)
    if out:
        Path(out).write_text(json.dumps(rows, indent=1, default=str), encoding="utf-8")
        Path(out).with_suffix(".md").write_text(card + "\n", encoding="utf-8")
