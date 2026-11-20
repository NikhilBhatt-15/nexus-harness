"""Server: projects, connect jobs, uploads, cancel (no API keys; fake model)."""

import time

import pytest
from fastapi.testclient import TestClient

from nexus_agent.llm import Completion, FakeLLM


@pytest.fixture(autouse=True)
def _home(tmp_path, monkeypatch):
    monkeypatch.setenv("NXA_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("NXA_EMBEDDER", "none")


def wait(c, url, key, done, timeout=60):
    for _ in range(int(timeout * 10)):
        r = c.get(url).json()
        if r[key] in done:
            return r
        time.sleep(0.1)
    raise AssertionError(f"timeout waiting for {url}")


def test_upload_connect_and_chat(tmp_path, monkeypatch):
    import nexus_agent.server as srv
    from nexus_agent import agent as agent_mod

    c = TestClient(srv.create_app(None))
    assert c.get("/v1/projects").json() == []
    assert c.post("/v1/runs", json={"task": "hi"}).status_code == 409   # no project yet
    batch = "b1"
    r = c.post(f"/v1/uploads?name=customers.csv&batch={batch}", content=b"id,name\n1,Ann\n2,Bo\n").json()
    c.post(f"/v1/uploads?name=orders.csv&batch={batch}",
           content=b"order_id,customer_id,total\n" + b"".join(f"{i},{1 + i % 2},{i}\n".encode() for i in range(10)))
    job = c.post("/v1/projects/connect", json={"source": r["folder"], "name": "Mini Shop"}).json()
    j = wait(c, f"/v1/jobs/{job['id']}", "status", {"done", "failed"})
    assert j["status"] == "done", j["log"]
    assert j["summary"]["relations"][0]["parent"] == "customers"
    projs = c.get("/v1/projects").json()
    assert projs[0]["name"] == "mini_shop" and projs[0]["relations"] == 1
    info = c.get("/v1/info?project=mini_shop").json()
    assert {t["name"] for ts in info["toolsets"] for t in ts["tools"]} >= {"db_query", "kb_query", "run_python"}
    assert c.get("/v1/projects/mini_shop/profile").json()["summary"]["self_check"]["passed"] > 0

    # chat with a fake model; session is tagged with the project
    orig = agent_mod.Agent.__init__

    def patched(self, model, *a, **k):
        orig(self, FakeLLM([Completion(text="hello from fake")]), *a, **k)

    monkeypatch.setattr(agent_mod.Agent, "__init__", patched)
    run = c.post("/v1/runs", json={"task": "hi", "project": "mini_shop"}).json()
    done = wait(c, f"/v1/runs/{run['id']}", "status", {"completed"})
    assert done["result"]["text"] == "hello from fake"
    sess = c.get("/v1/sessions?project=mini_shop").json()
    assert sess and sess[0]["project"] == "mini_shop"
    assert c.get("/v1/sessions?project=other").json() == []


def test_cancel_stops_run(tmp_path, monkeypatch):
    import nexus_agent.server as srv
    from nexus_agent import agent as agent_mod
    from nexus_agent.llm import ToolCall

    cfg = tmp_path / "p" / "agent.yaml"
    cfg.parent.mkdir()
    cfg.write_text("model: fake\npermissions: yolo\ntoolsets: []\n", encoding="utf-8")
    calls = {"n": 0}

    def slow(messages, tools):
        calls["n"] += 1
        time.sleep(0.3)
        return Completion(tool_calls=[ToolCall(id=f"c{calls['n']}", name="nothing", arguments={})])

    orig = agent_mod.Agent.__init__
    monkeypatch.setattr(agent_mod.Agent, "__init__", lambda self, model, *a, **k: orig(self, FakeLLM(slow), *a, **k))
    c = TestClient(srv.create_app(str(cfg)))
    run = c.post("/v1/runs", json={"task": "loop forever", "project": "p"}).json()
    time.sleep(0.8)
    assert c.post(f"/v1/runs/{run['id']}/cancel").json() == {"ok": True}
    done = wait(c, f"/v1/runs/{run['id']}", "status", {"completed"})
    assert done["result"]["stop_reason"] == "interrupted" and calls["n"] < 10


def test_token_auth(monkeypatch):
    import nexus_agent.server as srv

    monkeypatch.setenv("NXA_SERVER_TOKEN", "s3cret")
    c = TestClient(srv.create_app(None))
    assert c.get("/v1/projects").status_code == 401
    assert c.get("/v1/projects", headers={"Authorization": "Bearer s3cret"}).status_code == 200
    assert c.get("/v1/projects?token=s3cret").status_code == 200   # EventSource-style
    assert c.get("/v1/projects?token=wrong").status_code == 401
    assert c.get("/").status_code == 200                          # the page itself loads; API calls need the token
