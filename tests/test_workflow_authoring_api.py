"""S4 authoring API: validate / create / update / delete workflow packages.

Uses a temp registry + dummy provider + FastAPI TestClient (no LLM calls). Covers
the dry-run validator (ok / structural error / warnings) and the full CRUD
lifecycle including 409 (dup), 404 (missing), and 422 (invalid graph on save).
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from neurosurfer.app.server.gateway import NeurosurferServer
from neurosurfer.app.server.workflow_runs import RunManager, RunStore
from neurosurfer.graph.workflow.registry import WorkflowRegistry


class _DummyProvider:
    model = "dummy"
    capabilities = type("Caps", (), {"context_window": 8192, "max_output_tokens": 512})()


@pytest.fixture()
def client(tmp_path: Path) -> TestClient:
    mgr = RunManager(
        _DummyProvider(),
        registry=WorkflowRegistry(workflows_dir=tmp_path / "registry"),
        store=RunStore(runs_dir=tmp_path / "runs"),
    )
    server = NeurosurferServer(app_name="test", api_keys=None)
    server.run_manager = mgr
    return TestClient(server.create_app())


def _graph(nodes, inputs=None, outputs=None, name="wf"):
    return {"name": name, "inputs": inputs or [], "nodes": nodes, "outputs": outputs or []}


# ── validate ────────────────────────────────────────────────────────────────

def test_validate_ok(client: TestClient):
    g = _graph([{"id": "a", "kind": "base", "goal": "hi"}], outputs=["a"])
    r = client.post("/v1/workflows/validate", json={"graph": g})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["errors"] == []


def test_validate_structural_error(client: TestClient):
    # on_error target that does not depend on the source → control-flow error.
    g = _graph(
        [
            {"id": "a", "kind": "base", "goal": "hi", "on_error": "b"},
            {"id": "b", "kind": "base", "goal": "bye"},
        ],
        outputs=["a"],
    )
    r = client.post("/v1/workflows/validate", json={"graph": g})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is False
    assert body["errors"]


def test_validate_orphan_warning(client: TestClient):
    g = _graph(
        [
            {"id": "a", "kind": "base", "goal": "hi"},
            {"id": "b", "kind": "base", "goal": "orphan"},
        ],
        outputs=["a"],
    )
    body = client.post("/v1/workflows/validate", json={"graph": g}).json()
    assert body["ok"] is True  # warnings don't block
    assert any(w.get("node_id") == "b" for w in body["warnings"])


def test_validate_requires_graph(client: TestClient):
    assert client.post("/v1/workflows/validate", json={}).status_code == 422


# ── create / update / delete lifecycle ────────────────────────────────────────

def test_crud_lifecycle(client: TestClient):
    g1 = _graph(
        [{"id": "w", "kind": "base", "goal": "Write {topic}"}],
        inputs=[{"name": "topic", "type": "string", "required": True}],
        outputs=["w"],
        name="demo",
    )
    # create
    r = client.post("/v1/workflows", json={"name": "demo", "graph": g1})
    assert r.status_code == 201
    assert r.json()["name"] == "demo"

    # duplicate → 409
    assert client.post("/v1/workflows", json={"name": "demo", "graph": g1}).status_code == 409

    # it shows up in the list
    names = [w["name"] for w in client.get("/v1/workflows").json()["workflows"]]
    assert "demo" in names

    # update: add a node, bump version
    g2 = _graph(
        [
            {"id": "w", "kind": "base", "goal": "Write {topic}", "writes": "draft"},
            {"id": "p", "kind": "base", "goal": "Polish {draft}", "depends_on": ["w"]},
        ],
        inputs=[{"name": "topic", "type": "string", "required": True}],
        outputs=["p"],
        name="demo",
    )
    r = client.put("/v1/workflows/demo", json={"version": "1.1.0", "graph": g2})
    assert r.status_code == 200
    got = client.get("/v1/workflows/demo").json()
    assert got["version"] == "1.1.0"
    assert [n["id"] for n in got["graph"]["nodes"]] == ["w", "p"]

    # update with an invalid graph → 422, original preserved
    bad = _graph([{"id": "a", "kind": "base", "on_error": "ghost"}], outputs=["a"], name="demo")
    assert client.put("/v1/workflows/demo", json={"graph": bad}).status_code == 422
    assert client.get("/v1/workflows/demo").json()["version"] == "1.1.0"

    # delete, then 404 on second delete
    assert client.delete("/v1/workflows/demo").status_code == 200
    assert client.delete("/v1/workflows/demo").status_code == 404


def test_update_missing_404(client: TestClient):
    g = _graph([{"id": "a", "kind": "base", "goal": "hi"}], outputs=["a"], name="nope")
    assert client.put("/v1/workflows/nope", json={"graph": g}).status_code == 404
