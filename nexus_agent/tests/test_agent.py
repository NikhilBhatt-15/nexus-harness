"""nexus-agent tests. A scripted FakeLLM drives the loop, so no API keys are needed."""

from __future__ import annotations

import json
import re
import sqlite3
import sys
from pathlib import Path

import pytest

from nexus_agent import Agent, tool, toolsets
from nexus_agent.llm import Completion, FakeLLM, LLM, ToolCall, parse_text_tool_calls
from nexus_agent.runtime import Permissions, Session

HERE = Path(__file__).parent


def call(_tool, **args):
    return ToolCall(id=f"c_{_tool}", name=_tool, arguments=args)


@pytest.fixture(autouse=True)
def _home(tmp_path, monkeypatch):
    monkeypatch.setenv("NXA_HOME", str(tmp_path / "home"))


@pytest.fixture
def shop(tmp_path):
    db = tmp_path / "shop.db"
    con = sqlite3.connect(db)
    con.executescript("""
        CREATE TABLE customers(id INTEGER PRIMARY KEY, name TEXT);
        CREATE TABLE orders(id INTEGER PRIMARY KEY, customer_id INT, total REAL, status TEXT);
        INSERT INTO customers VALUES (1,'Alice'),(2,'Bob'),(3,'Chen');
        INSERT INTO orders VALUES (1,1,120,'paid'),(2,1,80,'paid'),(3,2,50,'refunded'),(4,3,300,'paid');
    """)
    con.commit()
    con.close()
    return db


def test_tool_decorator_schema():
    @tool(risk="write")
    def save(path: str, count: int = 2, tags: list[str] | None = None) -> str:
        """Save something.

        Args:
            path: Where to save.
            count: How many.
            tags: Labels.
        """
        return path

    s = save.schema()["function"]
    assert s["name"] == "save" and s["description"] == "Save something."
    props = s["parameters"]["properties"]
    assert props["path"] == {"type": "string", "description": "Where to save."}
    assert props["count"]["type"] == "integer" and props["count"]["default"] == 2
    assert props["tags"]["type"] == "array" and props["tags"]["items"] == {"type": "string"}
    assert s["parameters"]["required"] == ["path"] and save.risk == "write"


def test_loop_writes_file_and_finishes(tmp_path):
    llm = FakeLLM([
        Completion(tool_calls=[call("write_file", path="hello.txt", content="hi there")]),
        Completion(tool_calls=[call("read_file", path="hello.txt")]),
        Completion(text="Created hello.txt"),
    ])
    agent = Agent(llm, [toolsets.files(tmp_path)], workdir=tmp_path)
    r = agent.run("make hello.txt")
    assert r.stop_reason == "done" and r.text == "Created hello.txt"
    assert (tmp_path / "hello.txt").read_text() == "hi there"
    assert r.tool_calls == 2 and r.steps == 3
    # the read result was fed back to the model
    assert "hi there" in llm.seen[2][-1]["content"]


def test_files_are_jailed(tmp_path):
    llm = FakeLLM([Completion(tool_calls=[call("read_file", path="../../etc/passwd")]), Completion(text="no")])
    agent = Agent(llm, [toolsets.files(tmp_path)], workdir=tmp_path)
    agent.run("x")
    assert "outside the workspace" in agent.messages[2]["content"]


def test_permissions_modes(tmp_path):
    def script():
        return [Completion(tool_calls=[call("write_file", path="a.txt", content="x")]), Completion(text="done")]

    ro = Agent(FakeLLM(script()), [toolsets.files(tmp_path)], permissions="readonly", workdir=tmp_path)
    ro.run("write")
    assert not (tmp_path / "a.txt").exists() and "read-only" in ro.messages[2]["content"]

    asked = []
    ask = Agent(FakeLLM(script()), [toolsets.files(tmp_path)], permissions="ask",
                approver=lambda t, a: asked.append(t.name) or False, workdir=tmp_path)
    ask.run("write")
    assert asked == ["write_file"] and not (tmp_path / "a.txt").exists()

    auto = Agent(FakeLLM(script()), [toolsets.files(tmp_path)], permissions="auto", workdir=tmp_path)
    auto.run("write")
    assert (tmp_path / "a.txt").exists()

    deny = Agent(FakeLLM(script()), [toolsets.files(tmp_path)], workdir=tmp_path,
                 permissions=Permissions("yolo", deny={"write_file"}))
    (tmp_path / "a.txt").unlink()
    deny.run("write")
    assert not (tmp_path / "a.txt").exists()


def test_sql_toolset_read_write_and_destructive_gate(shop):
    db = toolsets.sqlite(str(shop), name="shop")
    names = [t.name for t in db.tools]
    assert names == ["shop_schema", "shop_query", "shop_execute", "shop_admin"]
    q = {t.name: t for t in db.tools}
    assert "CREATE TABLE customers" in q["shop_schema"]() and "3 rows" in q["shop_schema"]()
    out = q["shop_query"](query="SELECT c.name, SUM(o.total) t FROM orders o JOIN customers c ON c.id=o.customer_id "
                                "WHERE status='paid' GROUP BY c.name ORDER BY t DESC")
    assert out.splitlines()[1].startswith("Chen | 300")
    assert "read-only" in q["shop_query"](query="DELETE FROM orders WHERE id=1")
    assert "destructive" in q["shop_execute"](statement="DROP TABLE orders")
    assert "1 rows affected" in q["shop_execute"](statement="UPDATE orders SET status='paid' WHERE id=3")
    # destructive admin tool is blocked in auto mode without an approver
    agent = Agent(FakeLLM([Completion(tool_calls=[call("shop_admin", statement="DROP TABLE orders")]),
                           Completion(text="ok")]), [db])
    agent.run("drop it")
    assert "needs approval" in agent.messages[2]["content"]
    assert sqlite3.connect(shop).execute("SELECT COUNT(*) FROM orders").fetchone()[0] == 4


def test_attach_detach_and_unknown_tool(tmp_path):
    agent = Agent(FakeLLM([Completion(tool_calls=[call("run_command", command="echo hi")]), Completion(text="x")]),
                  [toolsets.files(tmp_path)], workdir=tmp_path)
    assert "run_command" not in [t.name for t in agent.tools]
    agent.attach(toolsets.shell(tmp_path))
    assert "run_command" in [t.name for t in agent.tools]
    agent.detach("shell")
    agent.run("go")
    assert "unknown tool 'run_command'" in agent.messages[2]["content"]


def test_shell_runs(tmp_path):
    out = toolsets.shell(tmp_path).tools[0](command="echo nexus")
    assert "exit code: 0" in out and "nexus" in out


def test_text_protocol_parsing():
    text, calls = parse_text_tool_calls('Looking it up.\n```tool\n{"name": "kb_query", "arguments": {"question": "fees?"}}\n```')
    assert text == "Looking it up." and calls[0].name == "kb_query" and calls[0].arguments == {"question": "fees?"}
    msgs = LLM._to_text_protocol([{"role": "system", "content": "S"}, {"role": "user", "content": "u"},
                                  {"role": "assistant", "content": None, "tool_calls": [
                                      {"id": "1", "type": "function", "function": {"name": "f", "arguments": "{}"}}]},
                                  {"role": "tool", "tool_call_id": "1", "name": "f", "content": "R"}],
                                 toolsets.web().tools)
    assert "fetch_url" in msgs[0]["content"] and msgs[3]["content"].startswith("TOOL RESULT (f)")


def test_sessions_resume_and_fork(tmp_path):
    a = Agent(FakeLLM([Completion(text="first answer")]), session=True, workdir=tmp_path)
    a.run("hello")
    sid = a.session.id
    b = Agent(FakeLLM([Completion(text="second")]), session=sid, workdir=tmp_path)
    assert [m["content"] for m in b.messages] == ["hello", "first answer"]
    b.run("again")
    assert len(Session(sid).messages()) == 4
    f = Session(sid).fork(2)
    assert len(f.messages()) == 2 and any(s["id"] == f.id for s in Session.list())


def test_compaction(tmp_path, monkeypatch):
    monkeypatch.setenv("NXA_CONTEXT_WINDOW", "2000")
    big = " ".join(f"word{i}" for i in range(700))

    def script(messages, tools):
        if messages[-1]["content"].startswith("Summarize the conversation"):
            return Completion(text="SUMMARY: user wants files")
        n = sum(1 for m in messages if m["role"] == "tool")
        return Completion(tool_calls=[call("recall")]) if n < 3 else Completion(text="done")

    agent = Agent(FakeLLM(script), [tool(lambda: big, name="recall")], keep_recent=2, workdir=tmp_path)
    events = list(agent.stream("go"))
    assert any(e["type"] == "compaction" for e in events)
    assert agent.messages[0]["content"].startswith("[Summary of earlier conversation]")


def test_python_plugin_and_skills(tmp_path):
    plugin = tmp_path / "my_tools.py"
    plugin.write_text('"""Weather helpers."""\nfrom nexus_agent import tool\n\n@tool\ndef weather(city: str) -> str:\n'
                      '    """Get weather.\n\n    Args:\n        city: City.\n    """\n    return f"sunny in {city}"\n')
    ts = toolsets.python_tools(plugin)
    assert ts.name == "my_tools" and ts.tools[0](city="Pune") == "sunny in Pune" and ts.prompt == "Weather helpers."
    sk = tmp_path / "skills" / "refunds"
    sk.mkdir(parents=True)
    (sk / "SKILL.md").write_text("---\nname: refunds\ndescription: How to process refunds\n---\nStep 1: check order.")
    agent = Agent(FakeLLM([Completion(tool_calls=[call("load_skill", name="refunds")]), Completion(text="ok")]),
                  [ts], workdir=tmp_path)
    assert "refunds: How to process refunds" in agent.system_prompt()
    agent.run("refund")
    assert agent.messages[2]["content"] == "Step 1: check order."


def test_mcp_toolset_stdio():
    ts = toolsets.mcp("echo", command=sys.executable, args=[str(HERE / "mcp_echo_server.py")])
    try:
        tools = {t.name: t for t in ts.tools}
        assert set(tools) == {"echo__echo", "echo__add"}
        assert tools["echo__echo"](text="hi") == "HI"
        assert tools["echo__add"](a=2, b=3).strip() == "5"
    finally:
        ts.teardown()


def test_subagent_tool(tmp_path):
    from nexus_agent.toolsets import subagent

    child_llm = FakeLLM([Completion(text="child result: 42")])
    ts = subagent("unused", [])
    import nexus_agent.agent as agent_mod

    orig = agent_mod.Agent.__init__

    def patched(self, model, *a, **k):  # route the child agent to the fake model
        orig(self, child_llm if model == "unused" else model, *a, **k)

    agent_mod.Agent.__init__ = patched
    try:
        out = ts.tools[0](task="compute")
    finally:
        agent_mod.Agent.__init__ = orig
    assert "child result: 42" in out and "[delegate: done" in out


@pytest.mark.skipif(not Path("contexts/banking/context.sqlite").exists(), reason="no compiled nexus-lite context")
def test_knowledge_toolset():
    ts = toolsets.knowledge("banking", root="contexts")
    q = {t.name: t for t in ts.tools}
    out = q["kb_query"](question="What is the monthly maintenance fee for the Beige business checking account?")
    assert "Citations:" in out and "beige" in out.lower()
    assert "open_bank_account_4821" in q["kb_sql"](query="SELECT tool_name FROM tools")


def test_eval_runner(tmp_path, shop):
    from nexus_agent import evals

    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "shop.db").write_bytes(shop.read_bytes())
    task = {"id": "t1", "task": "who spent most?", "workspace": "ws", "tools": "files,sqlite=shop.db",
            "checks": [{"answer_contains": "Chen"}, {"file_exists": "answer.txt"},
                       {"sql": {"db": "shop.db", "query": "SELECT COUNT(*) FROM orders", "equals": 4}},
                       {"max_tool_calls": 5}]}
    llm = FakeLLM([Completion(tool_calls=[call("shop_query", query="SELECT 1")]),
                   Completion(tool_calls=[call("write_file", path="answer.txt", content="Chen")]),
                   Completion(text="Chen spent the most.")])
    r = evals.run_task(task, llm, tmp_path, 0, None)  # type: ignore[arg-type]
    assert r["passed"], r["checks"]
    assert "100%" in evals.scorecard([{**r, "model": "fake"}])


def test_http_server_runs_and_approval(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient

    import nexus_agent.server as srv

    def fake_build(cfg, **kw):
        llm = FakeLLM([Completion(tool_calls=[call("write_file", path="s.txt", content="1")]), Completion(text="ok")])
        return Agent(llm, [toolsets.files(tmp_path)], permissions="ask", session=kw.get("session"), workdir=tmp_path)

    monkeypatch.setattr(srv, "build_agent", fake_build)
    c = TestClient(srv.create_app(None))
    run = c.post("/v1/runs", json={"task": "write"}).json()
    import time

    for _ in range(100):
        st = c.get(f"/v1/runs/{run['id']}").json()
        if st["pending_approval"]:
            break
        time.sleep(0.05)
    assert st["pending_approval"]["tool"] == "write_file"
    assert c.post(f"/v1/runs/{run['id']}/approve", json={"allow": True}).json() == {"ok": True}
    for _ in range(100):
        st = c.get(f"/v1/runs/{run['id']}").json()
        if st["status"] == "completed":
            break
        time.sleep(0.05)
    assert st["result"]["stop_reason"] == "done" and (tmp_path / "s.txt").exists()
    ev = c.get(f"/v1/runs/{run['id']}/events").json()
    assert [e["type"] for e in ev["events"]][-1] == "done"


def test_tau2_adapter_roundtrip(monkeypatch):
    pytest.importorskip("tau2")
    from tau2.data_model.message import UserMessage
    from tau2.domains.mock.environment import get_environment

    from nexus_agent.tau2_agent import create_nexus_agent

    env = get_environment()
    agent = create_nexus_agent(env.get_tools(), env.policy, llm="gpt-4.1-mini", llm_args={"tool_mode": "native"})
    first_tool = env.get_tools()[0].name
    agent.client = FakeLLM([Completion(tool_calls=[call(first_tool)])])
    state = agent.get_init_state()
    msg, state = agent.generate_next_message(UserMessage(role="user", content="hi"), state)
    assert msg.tool_calls[0].name == first_tool


def test_relay_llm(tmp_path):
    """An external process plays the model through req/resp files."""
    import threading
    import time

    relay = tmp_path / "relay"

    def responder():
        replies = [{"tool_calls": [{"name": "write_file", "arguments": {"path": "r.txt", "content": "relay"}}]},
                   {"text": "wrote it", "tool_calls": []}]
        for i, rep in enumerate(replies, 1):
            req = relay / f"req_{i:03d}.json"
            for _ in range(200):
                if req.exists():
                    break
                time.sleep(0.05)
            data = json.loads(req.read_text(encoding="utf-8"))
            assert any(t["function"]["name"] == "write_file" for t in data["tools"])
            (relay / f"resp_{i:03d}.json").write_text(json.dumps(rep), encoding="utf-8")

    th = threading.Thread(target=responder)
    th.start()
    agent = Agent(f"relay:{relay}", [toolsets.files(tmp_path)], workdir=tmp_path)
    agent.llm.poll_s = 0.05
    r = agent.run("write r.txt")
    th.join()
    assert r.stop_reason == "done" and r.text == "wrote it" and (tmp_path / "r.txt").read_text() == "relay"


def test_offload_large_output_and_read_back(tmp_path):
    big = "\n".join(f"row {i}: value {i * 7}" for i in range(2000))
    script = [Completion(tool_calls=[call("dump")])]

    def drive(messages, tools):
        last = messages[-1]
        if last["role"] == "tool" and last["name"] == "dump":
            oid = re.search(r"saved as `([^`]+)`", last["content"]).group(1)
            return Completion(tool_calls=[call("grep_tool_output", output_id=oid, pattern=r"^row 1999:")])
        if last["role"] == "tool":
            return Completion(text=last["content"])
        return script.pop(0)

    agent = Agent(FakeLLM(drive), [tool(lambda: big, name="dump")], workdir=tmp_path, offload_chars=1000,
                  preview_chars=300)
    r = agent.run("dump it")
    dump_msg = next(m for m in agent.messages if m.get("name") == "dump")
    assert len(dump_msg["content"]) < 1000 and "Full output saved as" in dump_msg["content"]
    assert "row 1999: value 13993" in r.text
    ev = next(e for e in r.events if e["type"] == "tool_result" and e["name"] == "dump")
    assert ev["offloaded"] and ev["chars"] == len(big)


def test_clear_old_tool_results_and_plan(tmp_path, monkeypatch):
    monkeypatch.setenv("NXA_CONTEXT_WINDOW", "4000")
    text = " ".join(f"tok{i}" for i in range(250))

    def drive(messages, tools):
        n = sum(1 for m in messages if m["role"] == "tool")
        if n == 0:
            return Completion(tool_calls=[call("update_plan", items=["look", "answer"], done=[1])])
        return Completion(tool_calls=[call("peek")]) if n < 9 else Completion(text="done")

    agent = Agent(FakeLLM(drive), [tool(lambda: text, name="peek")], workdir=tmp_path, keep_tool_results=2,
                  clear_tool_results_at=0.3, compact_at=0.99)
    events = list(agent.stream("go"))
    assert any(e["type"] == "context_edit" for e in events)
    assert any(str(m.get("content", "")).startswith("[cleared") for m in agent.messages)
    assert "1. [x] look" in agent.system_prompt() and "2. [ ] answer" in agent.system_prompt()


def test_web_ui_and_info_endpoints(tmp_path):
    from fastapi.testclient import TestClient

    import nexus_agent.server as srv

    cfg = tmp_path / "agent.yaml"
    cfg.write_text("model: fake\npermissions: ask\ntoolsets:\n  - web: {}\n", encoding="utf-8")
    c = TestClient(srv.create_app(str(cfg)))
    assert "nexus-agent" in c.get("/").text
    info = c.get("/v1/info").json()
    assert info["model"] == "fake" and info["mode"] == "ask" and info["toolsets"][0].startswith("web:")
    s = Session()
    s.add_message({"role": "user", "content": "hello there"})
    assert c.get(f"/v1/sessions/{s.id}/messages").json()[0]["content"] == "hello there"
    assert any(x["title"] == "hello there" for x in c.get("/v1/sessions").json())


def test_python_session_persists_state(tmp_path):
    ts = toolsets.python_session(tmp_path, prelude="import math\nBASE = 10")
    t = {x.name: x for x in ts.tools}
    try:
        assert "started a new Python session" in t["run_python"](code="x = BASE * 2")
        assert t["run_python"](code="x + 1").strip() == "21"            # state kept, expression displayed
        err = t["run_python"](code="undefined_name")
        assert "NameError" in err and "variables kept" in err and '"<string>"' not in err
        assert t["run_python"](code="print(x)").strip() == "20"         # still alive after the error
        assert "x: int" in t["python_vars"]()
        assert "timed out" in t["run_python"](code="import time; time.sleep(3)", timeout=1)
        assert t["run_python"](code="'x' in globals()").endswith("False")
        t["reset_python"]()
    finally:
        ts.teardown()


def test_per_tool_offload_settings(tmp_path):
    big = "y" * 9000
    calls = [Completion(tool_calls=[call("kbish"), call("dumpish")]), Completion(text="ok")]
    agent = Agent(FakeLLM(calls), [tool(lambda: big, name="kbish", offload=False),
                                   tool(lambda: big, name="dumpish")], workdir=tmp_path, offload_chars=1000)
    agent.run("go")
    out = {m["name"]: m["content"] for m in agent.messages if m["role"] == "tool"}
    assert out["kbish"] == big                                   # not offloaded
    assert "Full output saved as `dumpish-" in out["dumpish"]    # offloaded, short hashed id
    assert len(re.search(r"saved as `([^`]+)`", out["dumpish"]).group(1)) <= 50


def test_readonly_python_classification_and_auto_approval(tmp_path):
    from nexus_agent.python_session import classify_code

    readonly = [
        "print(q('SELECT NAME_EDUCATION_TYPE, COUNT(*), AVG(TARGET) FROM application_train GROUP BY 1'))",
        "df = q('SELECT TARGET, EXT_SOURCE_3 FROM application_train')\ndf['d'] = pd.qcut(df.EXT_SOURCE_3, 10, labels=False)\n"
        "df.groupby('d').TARGET.agg(['mean','count'])",
        "import lightgbm as lgb\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.metrics import roc_auc_score\n"
        "m = lgb.LGBMClassifier().fit(X, y)\nroc_auc_score(yv, m.predict_proba(Xv)[:, 1])",
        "s = df.name.str.replace('a', 'b')\nlst.remove(3)",
    ]
    for code in readonly:
        assert classify_code(code) == ("read", []), code
    risky = {
        "conn2 = sqlite3.connect('..\\home_credit.db')": ".connect()",
        "df.to_csv('out.csv')": ".to_csv()",
        "import os\nos.listdir('.')": "imports os",
        "open('x.txt', 'w').write('hi')": "calls open()",
        "import subprocess": "imports subprocess",
        "eval('1+1')": "calls eval()",
        "x.__class__.__bases__": "dunder",
        "conn.execute('DELETE FROM t')": ".execute()",
    }
    for code, why in risky.items():
        risk, reasons = classify_code(code)
        assert risk == "exec" and any(why in r for r in reasons), (code, reasons)

    ts = toolsets.python_session(tmp_path, auto_approve_readonly=True)
    asked = []
    try:
        agent = Agent(FakeLLM([Completion(tool_calls=[call("run_python", code="1 + 1"),
                                                      call("run_python", code="open('f.txt','w').write('x')")]),
                               Completion(text="done")]),
                      [ts], permissions="auto", approver=lambda t, a: asked.append(a["code"]) or False, workdir=tmp_path)
        agent.run("go")
        results = [m["content"] for m in agent.messages if m["role"] == "tool"]
        assert results[0].endswith("2")                       # pure computation ran without asking
        assert "declined" in results[1] and asked == ["open('f.txt','w').write('x')"]
        assert not (tmp_path / "f.txt").exists()
    finally:
        ts.teardown()
