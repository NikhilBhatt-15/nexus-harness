"""`nxa` command line.

    nxa                                   interactive chat (uses ./agent.yaml if present)
    nxa run "task" -m MODEL -t files,shell,sqlite=shop.db,kb=banking
    nxa run "task" --json                 JSON-lines event stream (for scripts)
    nxa chat --resume last                continue the most recent session
    nxa sessions | nxa tools | nxa models gemini
    nxa serve --port 8787                 HTTP API
    nxa eval suite.yaml -m MODEL          run a task suite and print a scorecard

Chat commands: /tools  /attach SPEC  /detach NAME  /model NAME  /mode MODE  /cost  /fork  /clear  /help  /exit
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Confirm

from nexus_agent.config import build_agent, load_config, parse_tool_spec
from nexus_agent.runtime import Session

console = Console()


def _load_dotenv() -> None:
    for p in (Path(".env"), Path.home() / ".nexus-agent" / ".env"):
        if p.exists():
            for line in p.read_text(encoding="utf-8").splitlines():
                k, sep, v = line.partition("=")
                if sep and k.strip() and not k.strip().startswith("#"):
                    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def cli_approver(t, args) -> bool:
    preview = json.dumps(args, ensure_ascii=False)[:600]
    console.print(Panel(f"[bold]{t.name}[/] ({t.risk}, toolset {t.toolset})\n{preview}", title="approve tool call?",
                        border_style="yellow"))
    return Confirm.ask("Allow", default=False)


def render_event(ev: dict) -> None:
    t = ev["type"]
    if t == "assistant":
        if ev.get("text") and ev.get("tool_calls"):
            console.print(f"[dim]{ev['text']}[/]")
    elif t == "tool_call":
        args = json.dumps(ev["arguments"], ensure_ascii=False)
        console.print(f"[cyan]> {ev['name']}[/] [dim]{args[:200]}[/]")
    elif t == "tool_result":
        mark = "[green]ok[/]" if ev["ok"] else "[red]error[/]"
        console.print(f"  {mark} [dim]{ev['ms']}ms, {ev['chars']} chars: {ev['preview'][:160]!r}[/]")
    elif t == "denied":
        console.print(f"  [red]denied:[/] {ev['reason']}")
    elif t == "compaction":
        console.print(f"[magenta]context compacted ({ev['summarized_messages']} messages summarized)[/]")
    elif t == "done":
        if ev.get("text"):
            console.print(Markdown(ev["text"]))
        style = "green" if ev["stop_reason"] == "done" else "yellow"
        extra = f" — {ev['error']}" if ev.get("error") else ""
        console.print(f"[{style}][{ev['stop_reason']}] {ev['steps']} steps, {ev['tool_calls']} tool calls, "
                      f"${ev['cost_usd']:.4f}{extra}[/]")


def _session_arg(resume: str | None):
    if not resume:
        return True
    if resume == "last":
        items = Session.list()
        return items[0]["id"] if items else True
    return resume


def cmd_run(a, interactive: bool) -> None:
    cfg = load_config(a.config)
    approver = cli_approver if sys.stdin.isatty() and not a.json else None
    agent = build_agent(cfg, model=a.model, tool_spec=a.tools, workdir=a.workdir, mode=a.mode, approver=approver,
                        session=_session_arg(getattr(a, "resume", None)), max_steps=a.max_steps)
    if not a.json:
        console.print(f"[bold]nexus-agent[/] · model [cyan]{agent.llm.model}[/] ({agent.llm.tool_mode} tools) · "
                      f"mode [cyan]{agent.permissions.mode}[/] · toolsets: "
                      f"{', '.join(agent.toolsets) or '(none)'} · session {agent.session.id if agent.session else '-'}")

    def one(task: str) -> None:
        for ev in agent.stream(task):
            if a.json:
                print(json.dumps(ev, ensure_ascii=False, default=str), flush=True)
            else:
                render_event(ev)

    if not interactive:
        one(a.task)
        return
    while True:
        try:
            line = console.input("[bold blue]you> [/]").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not line:
            continue
        if line.startswith("/"):
            cmd, _, arg = line[1:].partition(" ")
            if cmd in ("exit", "quit"):
                break
            elif cmd == "tools":
                for ts in agent.toolsets.values():
                    console.print(ts.describe())
            elif cmd == "attach":
                for ts in parse_tool_spec(arg, a.workdir):
                    agent.attach(ts)
                    console.print(f"attached {ts.describe()}")
            elif cmd == "detach":
                agent.detach(arg)
                console.print(f"detached {arg}")
            elif cmd == "model":
                from nexus_agent.llm import LLM

                agent.llm = LLM(arg)
                console.print(f"model -> {arg} ({agent.llm.tool_mode} tools)")
            elif cmd == "mode":
                agent.permissions.mode = arg  # type: ignore[assignment]
                console.print(f"permission mode -> {arg}")
            elif cmd == "cost":
                console.print(f"${agent.llm.total_cost:.4f} · {agent.llm.total_input_tokens} in / "
                              f"{agent.llm.total_output_tokens} out · {agent.llm.calls} calls")
            elif cmd == "fork" and agent.session:
                agent.session = agent.session.fork()
                console.print(f"forked -> session {agent.session.id}")
            elif cmd == "clear":
                agent.messages = []
                console.print("conversation cleared")
            else:
                console.print(__doc__)
            continue
        one(line)


def main(argv: list[str] | None = None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    _load_dotenv()
    ap = argparse.ArgumentParser(prog="nxa", description="nexus-agent: any LLM, pluggable tools")
    sub = ap.add_subparsers(dest="cmd")

    def common(p):
        p.add_argument("-m", "--model")
        p.add_argument("-t", "--tools", help="toolset spec, e.g. files,shell,sqlite=shop.db,kb=banking")
        p.add_argument("-c", "--config")
        p.add_argument("--mode", choices=["readonly", "ask", "auto", "yolo"])
        p.add_argument("--workdir", default=".")
        p.add_argument("--max-steps", type=int)
        p.add_argument("--json", action="store_true", help="emit JSON-lines events")

    r = sub.add_parser("run", help="run one task")
    r.add_argument("task")
    r.add_argument("--resume")
    common(r)
    c = sub.add_parser("chat", help="interactive chat (default)")
    c.add_argument("--resume", help="session id or 'last'")
    common(c)
    sub.add_parser("sessions", help="list saved sessions")
    m = sub.add_parser("models", help="list known models (optionally filtered)")
    m.add_argument("filter", nargs="?", default="")
    s = sub.add_parser("serve", help="HTTP API")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8787)
    s.add_argument("-c", "--config")
    e = sub.add_parser("eval", help="run a task suite")
    e.add_argument("suite")
    e.add_argument("-m", "--model", action="append", help="repeatable: compare several models")
    e.add_argument("--out")
    tl = sub.add_parser("tools", help="show toolsets from config/spec")
    tl.add_argument("-t", "--tools")
    tl.add_argument("-c", "--config")

    a = ap.parse_args(argv)
    if a.cmd in (None, "chat"):
        if a.cmd is None:
            a = ap.parse_args(["chat"] + (argv or sys.argv[1:]))
        cmd_run(a, interactive=True)
    elif a.cmd == "run":
        cmd_run(a, interactive=False)
    elif a.cmd == "sessions":
        for s_ in Session.list()[:30]:
            console.print(f"{s_['id']}  {s_['messages']:>4} msgs  {s_.get('model') or ''}")
    elif a.cmd == "models":
        import litellm

        names = sorted(k for k in litellm.model_cost if a.filter in k)
        console.print("\n".join(names[:200]) + (f"\n... {len(names) - 200} more" if len(names) > 200 else ""))
    elif a.cmd == "serve":
        from nexus_agent.server import serve

        serve(a.config, a.host, a.port)
    elif a.cmd == "eval":
        from nexus_agent.evals import main as eval_main

        eval_main(a.suite, a.model, a.out)
    elif a.cmd == "tools":
        cfg = load_config(a.config)
        from nexus_agent.config import toolset_from_entry

        sets = [toolset_from_entry(*next(iter(e.items()))) for e in cfg.get("toolsets", [])]
        sets += parse_tool_spec(a.tools) if a.tools else []
        for ts in sets:
            console.print(ts.describe())


if __name__ == "__main__":
    main()
