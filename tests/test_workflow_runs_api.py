"""Phase 2 — workflow execution API: run store, run manager, REST + SSE endpoints.

Uses function-only workflow packages (no LLM calls) in a temp registry, a dummy
provider, and FastAPI's TestClient. Covers the full lifecycle: start → live events
→ completion record → per-node detail, plus awaiting_input → resume, cancel, and
the SSE integration stream ending in [DONE].
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
import yaml

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from neurosurfer.app.server.gateway import NeurosurferServer
from neurosurfer.app.server.workflow_runs import RunManager, RunStore
from neurosurfer.graph.workflow.registry import WorkflowRegistry

FN = "tests.test_workflow_runs_api"


class _DummyProvider:
    """Function-only workflows never call the LLM."""

    model = "dummy"
    capabilities = type("Caps", (), {"context_window": 8192, "max_output_tokens": 512})()


def _double(x=None, **kwargs):
    return (int(x) if x is not None else 0) * 2


def _shout(doubled=None, **kwargs):
    return f"result={doubled}"


def _slow(**kwargs):
    time.sleep(2.0)
    return "slow done"


def _write_pkg(root: Path, name: str, graph_nodes: list[dict], inputs: list[dict],
               outputs: list[str]) -> None:
    pkg = root / name
    pkg.mkdir(parents=True)
    (pkg / "workflow.yaml").write_text(
        yaml.dump({"name": name, "version": "1.0.0", "entrypoint": "graph.yaml",
                   "description": f"{name} test workflow"}),
        encoding="utf-8",
    )
    (pkg / "graph.yaml").write_text(
        yaml.dump({"name": name, "inputs": inputs, "nodes": graph_nodes,
                   "outputs": outputs}),
        encoding="utf-8",
    )


@pytest.fixture()
def manager(tmp_path: Path) -> RunManager:
    reg_dir = tmp_path / "registry"
    _write_pkg(
        reg_dir, "doubler",
        graph_nodes=[
            {"id": "doubled", "kind": "function", "callable": f"{FN}._double"},
            {"id": "shout", "kind": "function", "callable": f"{FN}._shout",
             "depends_on": ["doubled"]},
        ],
        inputs=[{"name": "x", "type": "integer", "required": True}],
        outputs=["shout"],
    )
    _write_pkg(
        reg_dir, "slowpoke",
        graph_nodes=[{"id": "nap", "kind": "function", "callable": f"{FN}._slow"}],
        inputs=[],
        outputs=["nap"],
    )
    _write_pkg(
        reg_dir, "needs_input",
        graph_nodes=[
            {"id": "approval", "kind": "input", "purpose": "Approve?"},
            {"id": "after", "kind": "function", "callable": f"{FN}._shout",
             "depends_on": ["approval"]},
        ],
        inputs=[{"name": "approval", "type": "string", "required": False}],
        outputs=["after"],
    )
    return RunManager(
        _DummyProvider(),
        registry=WorkflowRegistry(workflows_dir=reg_dir),
        store=RunStore(runs_dir=tmp_path / "runs"),
    )


@pytest.fixture()
def client(manager: RunManager) -> TestClient:
    server = NeurosurferServer(app_name="test", api_keys=None)
    server.run_manager = manager
    return TestClient(server.create_app())


def _wait(manager: RunManager, run_id: str, timeout: float = 10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        rec = manager.get(run_id)
        if rec and rec.status != "running":
            return rec
        time.sleep(0.05)
    raise AssertionError(f"run {run_id} still running after {timeout}s")


# ── manager lifecycle (no HTTP) ─────────────────────────────────────────────────

def test_manager_start_and_complete(manager: RunManager):
    rec = manager.start("doubler", {"x": 4})
    rec = _wait(manager, rec.id)
    assert rec.status == "succeeded"
    assert rec.final == {"shout": "result=8"}
    assert rec.nodes["doubled"]["output"] == 8
    assert rec.nodes["shout"]["status"] == "ok"
    # Event log has run start, node starts/oks, run finish — in seq order.
    seqs = [e["seq"] for e in rec.events]
    assert seqs == sorted(seqs)
    types = {e["type"] for e in rec.events}
    assert types == {"run", "node"}
    # Persisted to disk for replay.
    persisted = json.loads((manager.store.dir / f"{rec.id}.json").read_text())
    assert persisted["status"] == "succeeded"


def test_manager_awaiting_input_then_resume(manager: RunManager):
    rec = manager.start("needs_input", {})
    rec = _wait(manager, rec.id)
    assert rec.status == "awaiting_input"
    assert any(e["type"] == "input_required" and e["node_id"] == "approval"
               for e in rec.events)

    resumed = manager.resume(rec.id, {"approval": "yes"})
    resumed = _wait(manager, resumed.id)
    assert resumed.status == "succeeded"
    assert resumed.nodes["approval"]["output"] == "yes"
    assert any(e.get("resumed_from") == rec.id for e in resumed.events)


def test_manager_cancel(manager: RunManager):
    rec = manager.start("slowpoke", {})
    time.sleep(0.1)  # let the worker actually start the slow node
    out = manager.cancel(rec.id)
    assert out.status == "cancelled"
    time.sleep(2.5)  # worker finishes in background; must not overwrite cancelled
    assert manager.get(rec.id).status == "cancelled"


# ── REST contract ───────────────────────────────────────────────────────────────

def test_list_and_get_workflows(client: TestClient):
    r = client.get("/v1/workflows")
    assert r.status_code == 200
    names = {w["name"] for w in r.json()["workflows"]}
    assert names == {"doubler", "needs_input", "slowpoke"}

    g = client.get("/v1/workflows/doubler")
    assert g.status_code == 200
    body = g.json()
    assert body["name"] == "doubler"
    node_ids = [n["id"] for n in body["graph"]["nodes"]]
    assert node_ids == ["doubled", "shout"]

    assert client.get("/v1/workflows/nope").status_code == 404


def test_start_get_and_node_detail(client: TestClient, manager: RunManager):
    r = client.post("/v1/workflows/doubler/runs", json={"inputs": {"x": 21}})
    assert r.status_code == 202
    run_id = r.json()["id"]

    _wait(manager, run_id)
    got = client.get(f"/v1/runs/{run_id}")
    assert got.status_code == 200
    body = got.json()
    assert body["status"] == "succeeded"
    assert body["final"] == {"shout": "result=42"}
    assert "events" not in body  # events only on request

    with_events = client.get(f"/v1/runs/{run_id}", params={"events": "true"}).json()
    assert isinstance(with_events["events"], list) and with_events["events"]

    node = client.get(f"/v1/runs/{run_id}/nodes/doubled")
    assert node.status_code == 200
    assert node.json()["output"] == 42
    assert client.get(f"/v1/runs/{run_id}/nodes/ghost").status_code == 404

    runs = client.get("/v1/runs").json()["runs"]
    assert any(x["id"] == run_id for x in runs)


def test_start_unknown_workflow_404(client: TestClient):
    assert client.post("/v1/workflows/ghost/runs", json={"inputs": {}}).status_code == 404


def test_run_endpoints_404_for_unknown_run(client: TestClient):
    assert client.get("/v1/runs/deadbeef").status_code == 404
    assert client.post("/v1/runs/deadbeef/resume", json={"values": {}}).status_code == 404
    assert client.delete("/v1/runs/deadbeef").status_code == 404


def test_resume_endpoint(client: TestClient, manager: RunManager):
    r = client.post("/v1/workflows/needs_input/runs", json={"inputs": {}})
    run_id = r.json()["id"]
    _wait(manager, run_id)
    assert client.get(f"/v1/runs/{run_id}").json()["status"] == "awaiting_input"

    resumed = client.post(f"/v1/runs/{run_id}/resume", json={"values": {"approval": "ok"}})
    assert resumed.status_code == 202
    new_id = resumed.json()["id"]
    assert new_id != run_id
    rec = _wait(manager, new_id)
    assert rec.status == "succeeded"


def test_cancel_endpoint(client: TestClient, manager: RunManager):
    r = client.post("/v1/workflows/slowpoke/runs", json={"inputs": {}})
    run_id = r.json()["id"]
    out = client.delete(f"/v1/runs/{run_id}")
    assert out.status_code == 200
    assert out.json()["status"] == "cancelled"


# ── SSE integration ─────────────────────────────────────────────────────────────

def test_sse_stream_live_tail(client: TestClient, manager: RunManager):
    # Open the stream while the run is still executing: events must arrive live
    # over the connection and the stream must close after the final run event.
    r = client.post("/v1/workflows/slowpoke/runs", json={"inputs": {}})
    run_id = r.json()["id"]

    events: list[dict] = []
    with client.stream("GET", f"/v1/runs/{run_id}/events") as resp:
        for line in resp.iter_lines():
            if not line.startswith("data: "):
                continue
            payload = line[len("data: "):]
            if payload == "[DONE]":
                break
            events.append(json.loads(payload))

    statuses = [(e.get("node_id"), e["status"]) for e in events if e["type"] == "node"]
    assert (None, "running") == (events[0].get("node_id"), events[0]["status"])
    assert ("nap", "start") in statuses and ("nap", "ok") in statuses
    assert events[-1]["type"] == "run" and events[-1]["status"] == "succeeded"


def test_sse_stream_replays_and_closes(client: TestClient, manager: RunManager):
    r = client.post("/v1/workflows/doubler/runs", json={"inputs": {"x": 5}})
    run_id = r.json()["id"]
    _wait(manager, run_id)  # run finishes; stream must replay everything then close

    events: list[dict] = []
    done = False
    with client.stream("GET", f"/v1/runs/{run_id}/events") as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        for line in resp.iter_lines():
            if not line.startswith("data: "):
                continue
            payload = line[len("data: "):]
            if payload == "[DONE]":
                done = True
                break
            events.append(json.loads(payload))

    assert done, "stream must terminate with [DONE]"
    # Full replay: run-start first, run-end last, node lifecycle in between.
    assert events[0]["type"] == "run" and events[0]["status"] == "running"
    assert events[-1]["type"] == "run" and events[-1]["status"] == "succeeded"
    node_events = [e for e in events if e["type"] == "node"]
    assert {(e["node_id"], e["status"]) for e in node_events} >= {
        ("doubled", "start"), ("doubled", "ok"), ("shout", "start"), ("shout", "ok"),
    }
    seqs = [e["seq"] for e in events]
    assert seqs == sorted(seqs)
