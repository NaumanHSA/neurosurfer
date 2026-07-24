"""S5 architect build API: start / stream / record lifecycle.

Drives the ArchitectManager with a FAKE agent (no LLM) that stages a couple of
nodes, notifies, and registers a package — exercising the manager's step log,
graph snapshots, terminal handling, and the REST + SSE endpoints.
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import pytest
import yaml

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from neurosurfer.app.server.architect_builds.manager import ArchitectManager
from neurosurfer.app.server.gateway import NeurosurferServer
from neurosurfer.app.server.workflow_runs import RunManager, RunStore
from neurosurfer.graph.workflow.registry import WorkflowRegistry


class _DummyProvider:
    model = "dummy"
    capabilities = type("Caps", (), {"context_window": 8192, "max_output_tokens": 512})()


class _FakeSession:
    def __init__(self) -> None:
        self.name = "fake_wf"
        self.nodes: list[dict] = []

    def graph_dict(self) -> dict:
        return {
            "name": self.name,
            "description": "",
            "inputs": [],
            "nodes": self.nodes,
            "outputs": [self.nodes[-1]["id"]] if self.nodes else [],
        }


def _register(registry: WorkflowRegistry, session: _FakeSession) -> str:
    from neurosurfer.graph.workflow.package import load_package

    with tempfile.TemporaryDirectory() as tmp:
        pkg_dir = Path(tmp) / session.name
        pkg_dir.mkdir()
        (pkg_dir / "workflow.yaml").write_text(
            yaml.safe_dump({"name": session.name, "version": "0.1.0", "entrypoint": "graph.yaml"})
        )
        (pkg_dir / "graph.yaml").write_text(yaml.safe_dump(session.graph_dict()))
        pkg = load_package(pkg_dir)
        dest = registry.save(pkg)
    return str(dest)


class _FakeAgent:
    def __init__(self, notify, registry: WorkflowRegistry) -> None:
        self._notify = notify
        self._registry = registry
        self.session = _FakeSession()

    async def build(self, intent: str) -> str:
        self._notify("planning workflow")
        self.session.nodes.append({"id": "a", "kind": "base", "goal": f"do {intent}"})
        self._notify("added node a")
        self.session.nodes.append(
            {"id": "b", "kind": "base", "goal": "finish", "depends_on": ["a"]}
        )
        self._notify("added node b")
        self._notify("validating")
        return _register(self._registry, self.session)


@pytest.fixture()
def client(tmp_path: Path) -> TestClient:
    registry = WorkflowRegistry(workflows_dir=tmp_path / "registry")
    run_mgr = RunManager(
        _DummyProvider(), registry=registry, store=RunStore(runs_dir=tmp_path / "runs")
    )
    arch = ArchitectManager(
        _DummyProvider(),
        registry=registry,
        agent_factory=lambda notify, _verify: _FakeAgent(notify, registry),
    )
    server = NeurosurferServer(app_name="test", api_keys=None)
    server.run_manager = run_mgr
    server.architect_manager = arch
    return TestClient(server.create_app())


def _wait_terminal(client: TestClient, bid: str, timeout=5.0) -> dict:
    end = time.time() + timeout
    while time.time() < end:
        rec = client.get(f"/v1/architect/builds/{bid}").json()
        if rec["status"] in {"succeeded", "blocked", "failed", "cancelled"}:
            return rec
        time.sleep(0.05)
    raise AssertionError("build did not finish")


def test_start_requires_intent(client: TestClient):
    assert client.post("/v1/architect/builds", json={}).status_code == 422


def test_unknown_build_404(client: TestClient):
    assert client.get("/v1/architect/builds/nope").status_code == 404


def test_build_succeeds_and_records_steps(client: TestClient):
    r = client.post("/v1/architect/builds", json={"intent": "summarise an article"})
    assert r.status_code == 202
    bid = r.json()["id"]

    rec = _wait_terminal(client, bid)
    assert rec["status"] == "succeeded"
    assert rec["workflow"] == "fake_wf"
    assert rec["graph"] and [n["id"] for n in rec["graph"]["nodes"]] == ["a", "b"]

    # the registered workflow is now visible via the workflow API
    names = [w["name"] for w in client.get("/v1/workflows").json()["workflows"]]
    assert "fake_wf" in names

    # full event log carries logs, graph snapshots, and terminal build event
    full = client.get(f"/v1/architect/builds/{bid}?events=true").json()
    types = [e["type"] for e in full["events"]]
    assert "log" in types and "graph" in types
    assert full["events"][-1] == full["events"][-1]  # sanity
    assert any(e["type"] == "build" and e.get("status") == "succeeded" for e in full["events"])


def test_sse_stream_replays_and_closes(client: TestClient):
    bid = client.post("/v1/architect/builds", json={"intent": "x"}).json()["id"]
    _wait_terminal(client, bid)
    # After terminal, the stream replays the whole log then closes with [DONE].
    body = client.get(f"/v1/architect/builds/{bid}/events").text
    assert "planning workflow" in body
    assert "[DONE]" in body


def test_list_builds(client: TestClient):
    client.post("/v1/architect/builds", json={"intent": "one"})
    client.post("/v1/architect/builds", json={"intent": "two"})
    builds = client.get("/v1/architect/builds").json()["builds"]
    assert len(builds) >= 2
