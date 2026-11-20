"""A persistent Python session for agents (code-interpreter style).

One long-lived worker process per session keeps variables, imports and loaded data between
``run_python`` calls, like a Jupyter kernel. Each call:

- runs the code in the session's globals, capturing stdout/stderr (including C-level output);
- if the last statement is an expression, shows its value (``df.head()`` works without print);
- on an exception returns the traceback and keeps the session alive;
- on timeout kills the worker and starts a fresh one (variables are lost, and the result says so).

Protocol: the parent writes one JSON line per request to the worker's stdin; the worker answers
with a sentinel-prefixed JSON line on stdout. Any other stdout lines (e.g. native library output)
are treated as part of that call's output.
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path

SENTINEL = "\x00NXA_RESULT "

WORKER = r'''
import ast, io, json, sys, traceback, contextlib
SENTINEL = "\x00NXA_RESULT "
_out = sys.stdout
G = {"__name__": "__main__"}

def _display(v):
    try:
        import pandas as pd
        if isinstance(v, (pd.DataFrame, pd.Series)):
            return v.to_string(max_rows=60, max_cols=30)
    except Exception:
        pass
    r = repr(v)
    return r if len(r) < 20000 else r[:20000] + "... (repr truncated)"

def run(code):
    buf_out, buf_err = io.StringIO(), io.StringIO()
    error = None
    try:
        tree = ast.parse(code, mode="exec")
        last = tree.body[-1] if tree.body and isinstance(tree.body[-1], ast.Expr) else None
        if last is not None:
            tree.body = tree.body[:-1]
        with contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(buf_err):
            exec(compile(tree, "<cell>", "exec"), G)
            if last is not None:
                v = eval(compile(ast.Expression(last.value), "<cell>", "eval"), G)
                if v is not None:
                    print(_display(v))
    except BaseException:
        error = traceback.format_exc(limit=8)
        error = "\n".join(l for l in error.splitlines() if not l.startswith('  File "<string>"'))
        error = error.replace('  File "<cell>"', '  cell')
    return {"stdout": buf_out.getvalue(), "stderr": buf_err.getvalue(), "error": error}

for line in sys.stdin:
    req = json.loads(line)
    if req.get("op") == "vars":
        info = []
        for k, v in G.items():
            if k.startswith("_") or k in ("In", "Out") or type(v).__name__ in ("module", "function", "type", "builtin_function_or_method"):
                continue
            shape = getattr(v, "shape", None)
            info.append(f"{k}: {type(v).__name__}" + (f" shape={tuple(shape)}" if shape is not None else ""))
        res = {"stdout": "\n".join(info) or "(no variables)", "stderr": "", "error": None}
    else:
        res = run(req["code"])
    _out.write(SENTINEL + json.dumps(res) + "\n")
    _out.flush()
'''


class PythonSession:
    def __init__(self, workdir: str | Path = ".", prelude: str = "", python: str | None = None,
                 env: dict | None = None):
        self.workdir = Path(workdir).resolve()
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.prelude = prelude
        self.python = python or sys.executable
        self.env = {**os.environ, "PYTHONUTF8": "1", "PYTHONUNBUFFERED": "1", **(env or {})}
        self.proc: subprocess.Popen | None = None
        self.lines: queue.Queue[str | None] = queue.Queue()
        self.lock = threading.Lock()

    # ------------------------------------------------------------------ process
    def _start(self) -> None:
        self.proc = subprocess.Popen([self.python, "-u", "-c", WORKER], cwd=self.workdir, env=self.env,
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     text=True, encoding="utf-8", errors="replace", bufsize=1)
        self.lines = queue.Queue()
        threading.Thread(target=self._pump, args=(self.proc, self.lines), daemon=True).start()
        if self.prelude:
            r = self._request({"op": "run", "code": self.prelude}, timeout=120)
            if r.get("error"):
                raise RuntimeError(f"python session prelude failed:\n{r['error']}")

    @staticmethod
    def _pump(proc: subprocess.Popen, q: queue.Queue) -> None:
        for line in proc.stdout:  # type: ignore[union-attr]
            q.put(line)
        q.put(None)

    def _request(self, req: dict, timeout: float) -> dict:
        assert self.proc and self.proc.stdin
        self.proc.stdin.write(json.dumps(req) + "\n")
        self.proc.stdin.flush()
        raw: list[str] = []
        import time

        deadline = time.time() + timeout
        while True:
            try:
                line = self.lines.get(timeout=max(0.1, deadline - time.time()))
            except queue.Empty:
                raise TimeoutError
            if line is None:
                return {"stdout": "".join(raw), "stderr": "", "error": "Python session process exited unexpectedly."}
            if line.startswith(SENTINEL):
                res = json.loads(line[len(SENTINEL):])
                if raw:
                    res["stdout"] = "".join(raw) + res.get("stdout", "")
                return res
            raw.append(line)

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    # ------------------------------------------------------------------ api
    def run(self, code: str, timeout: float = 300) -> str:
        with self.lock:
            note = ""
            if not self.alive():
                note = "[started a new Python session]\n" if self.proc is None else \
                    "[previous Python session had exited; started a new one, earlier variables are gone]\n"
                self._start()
            try:
                r = self._request({"op": "run", "code": code}, timeout)
            except TimeoutError:
                self.close()
                return (f"{note}Error: timed out after {timeout}s. The Python session was restarted, so all "
                        "variables are gone; reload what you need (and use smaller steps).")
        out = note + (r.get("stdout") or "")
        if r.get("stderr"):
            out += f"\n--- stderr ---\n{r['stderr'][-4000:]}"
        if r.get("error"):
            out += f"\n--- error (session still alive, variables kept) ---\n{r['error']}"
        return out.strip() or "(no output; the last line was not an expression and nothing was printed)"

    def variables(self) -> str:
        with self.lock:
            if not self.alive():
                return "(no Python session running yet)"
            return self._request({"op": "vars"}, 30)["stdout"]

    def reset(self) -> str:
        self.close()
        return "Python session reset; all variables cleared (the prelude runs again on the next call)."

    def close(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.kill()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
        self.proc = None
