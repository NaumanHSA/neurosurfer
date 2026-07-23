"""Real-LLM integration tests for the Phase 1 control-flow engine + Phase 2 API.

Runs against a REAL model and exercises the paths where a real model's output
drives control flow — the part unit tests with scripted providers cannot prove:

- a `when` guard evaluating real (whitespace-y, free-text) LLM output
- the LLM router choosing a branch semantically
- map/loop bodies containing real `base` nodes
- the execution API end-to-end (REST + SSE) with a real provider

Defaults to LM Studio (qwen/qwen3.5-9b) and auto-skips when it is down, so the
normal suite stays hermetic; override the model/endpoint via NEUROSURFER_TEST_*
(see ``tests/_llm_test_provider.py``) to run against a hosted model. Run with:

    conda run -n LLMs python -m pytest tests/test_graph_llm_integration.py -v
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
import yaml

from ._llm_test_provider import make_provider, provider_ready, skip_reason

pytestmark = pytest.mark.skipif(not provider_ready(), reason=skip_reason())


@pytest.fixture(scope="module")
def provider():
    return make_provider()


def _mark(**kwargs):
    return "ran"


FN = "tests.test_graph_llm_integration"


# ── 1c: when-guard over real LLM output ─────────────────────────────────────────

def _triage_graph():
    from neurosurfer.graph.engine.loader import load_graph_from_dict

    return load_graph_from_dict({
        "name": "llm_triage",
        "inputs": [{"name": "ticket", "type": "string", "required": True}],
        "nodes": [
            {"id": "classify", "kind": "base",
             "purpose": "You are a triage classifier. Reply with exactly one word.",
             "goal": "Classify this support ticket: \"{ticket}\". "
                     "Reply with exactly one word: urgent or normal.",
             "policy": {"max_new_tokens": 512, "temperature": 0.0}},
            # Real output arrives with whitespace/casing noise → contains(lower(...))
            # is the robust guard form (exact == would be brittle).
            {"id": "escalate", "kind": "function", "callable": f"{FN}._mark",
             "depends_on": ["classify"],
             "when": "contains(lower(nodes.classify), 'urgent')"},
            {"id": "archive", "kind": "function", "callable": f"{FN}._mark",
             "depends_on": ["classify"],
             "when": "not contains(lower(nodes.classify), 'urgent')"},
        ],
        "outputs": ["escalate", "archive"],
    })


def test_when_guard_on_real_llm_output_urgent(provider):
    from neurosurfer.graph import GraphExecutor

    ex = GraphExecutor(_triage_graph(), provider=provider, log_traces=False)
    res = ex.run({"ticket": "Production database is down; customers cannot pay!"})
    assert res.nodes["classify"].error is None
    assert "urgent" in str(res.nodes["classify"].raw_output).lower()
    assert res.nodes["escalate"].skipped is False
    assert res.nodes["archive"].skipped is True


def test_when_guard_on_real_llm_output_normal(provider):
    from neurosurfer.graph import GraphExecutor

    ex = GraphExecutor(_triage_graph(), provider=provider, log_traces=False)
    res = ex.run({"ticket": "Please update the footer copyright year when convenient."})
    assert res.nodes["archive"].skipped is False
    assert res.nodes["escalate"].skipped is True


# ── 1d: LLM router semantic routing ─────────────────────────────────────────────

def _router_graph():
    from neurosurfer.graph.engine.loader import load_graph_from_dict

    return load_graph_from_dict({
        "name": "llm_router_real",
        "inputs": [{"name": "ticket", "type": "string", "required": True}],
        "nodes": [
            {"id": "route", "kind": "router",
             "purpose": "Decide whether the support ticket in the workflow inputs "
                        "is urgent (production down, money lost, security breach) "
                        "or normal (routine, cosmetic, low impact).",
             "cases": [
                 {"to": "urgent_path", "label": "urgent"},
                 {"to": "normal_path", "label": "normal"},
             ]},
            {"id": "urgent_path", "kind": "function", "callable": f"{FN}._mark",
             "depends_on": ["route"]},
            {"id": "normal_path", "kind": "function", "callable": f"{FN}._mark",
             "depends_on": ["route"]},
        ],
        "outputs": ["urgent_path", "normal_path"],
    })


def test_llm_router_routes_urgent(provider):
    from neurosurfer.graph import GraphExecutor

    ex = GraphExecutor(_router_graph(), provider=provider, log_traces=False)
    res = ex.run({"ticket": "SECURITY BREACH: attacker has admin access right now."})
    assert res.nodes["route"].raw_output == "urgent_path"
    assert res.nodes["urgent_path"].skipped is False
    assert res.nodes["normal_path"].skipped is True


def test_llm_router_routes_normal(provider):
    from neurosurfer.graph import GraphExecutor

    ex = GraphExecutor(_router_graph(), provider=provider, log_traces=False)
    res = ex.run({"ticket": "Typo in the About page: 'teh' should be 'the'."})
    assert res.nodes["route"].raw_output == "normal_path"
    assert res.nodes["normal_path"].skipped is False
    assert res.nodes["urgent_path"].skipped is True


# ── 1e/1f: loop + map with real-LLM bodies ─────────────────────────────────────

def test_map_with_real_llm_body(provider):
    from neurosurfer.graph import GraphExecutor
    from neurosurfer.graph.engine.loader import load_graph_from_dict

    graph = load_graph_from_dict({
        "name": "capitals",
        "inputs": [{"name": "cities", "type": "array", "required": True}],
        "nodes": [
            {"id": "countries", "kind": "map", "over": "inputs.cities", "as": "item",
             "body": [
                 {"id": "ask", "kind": "base",
                  "purpose": "You answer geography questions with a single word.",
                  "goal": "{item} is the capital city of which country? "
                          "Reply with only the country name.",
                  "policy": {"max_new_tokens": 256, "temperature": 0.0}},
             ]},
        ],
        "outputs": ["countries"],
    })
    ex = GraphExecutor(graph, provider=provider, log_traces=False)
    res = ex.run({"cities": ["Paris", "Tokyo"]})
    out = res.nodes["countries"]
    assert out.error is None
    answers = [str(a).lower() for a in out.raw_output]
    assert len(answers) == 2
    assert "france" in answers[0]
    assert "japan" in answers[1]


def test_loop_with_real_llm_body(provider):
    from neurosurfer.graph import GraphExecutor
    from neurosurfer.graph.engine.loader import load_graph_from_dict

    graph = load_graph_from_dict({
        "name": "pinger",
        "nodes": [
            {"id": "pings", "kind": "loop", "max_iterations": 2,
             "accumulate": "history",
             "body": [
                 {"id": "say", "kind": "base",
                  "purpose": "You follow instructions exactly.",
                  "goal": "Reply with exactly the word: ping",
                  "policy": {"max_new_tokens": 128, "temperature": 0.0}},
             ]},
        ],
        "outputs": ["pings"],
    })
    ex = GraphExecutor(graph, provider=provider, log_traces=False)
    res = ex.run({})
    node = res.nodes["pings"]
    assert node.error is None
    assert node.structured_output["iterations"] == 2
    assert len(node.raw_output) == 2
    assert all("ping" in str(x).lower() for x in node.raw_output)


# ── Phase 2 API end-to-end with the real provider ───────────────────────────────

def test_api_run_with_real_llm_and_sse(provider, tmp_path: Path):
    fastapi = pytest.importorskip("fastapi")  # noqa: F841
    from fastapi.testclient import TestClient

    from neurosurfer.app.server.gateway import NeurosurferServer
    from neurosurfer.app.server.workflow_runs import RunManager, RunStore
    from neurosurfer.graph.workflow.registry import WorkflowRegistry

    # Register the LLM-router workflow as a real package.
    reg_dir = tmp_path / "registry"
    pkg = reg_dir / "triage"
    pkg.mkdir(parents=True)
    (pkg / "workflow.yaml").write_text(yaml.dump(
        {"name": "triage", "version": "1.0.0", "entrypoint": "graph.yaml"}))
    (pkg / "graph.yaml").write_text(yaml.dump({
        "name": "triage",
        "inputs": [{"name": "ticket", "type": "string", "required": True}],
        "nodes": [
            {"id": "route", "kind": "router",
             "purpose": "Decide whether the support ticket in the workflow inputs "
                        "is urgent (production down, money lost, security breach) "
                        "or normal (routine, cosmetic).",
             "cases": [
                 {"to": "urgent_path", "label": "urgent"},
                 {"to": "normal_path", "label": "normal"},
             ]},
            {"id": "urgent_path", "kind": "function",
             "callable": f"{FN}._mark", "depends_on": ["route"]},
            {"id": "normal_path", "kind": "function",
             "callable": f"{FN}._mark", "depends_on": ["route"]},
        ],
        "outputs": ["urgent_path", "normal_path"],
    }))

    manager = RunManager(
        provider,
        registry=WorkflowRegistry(workflows_dir=reg_dir),
        store=RunStore(runs_dir=tmp_path / "runs"),
    )
    server = NeurosurferServer(app_name="llm-int-test", api_keys=None)
    server.run_manager = manager
    client = TestClient(server.create_app())

    r = client.post("/v1/workflows/triage/runs",
                    json={"inputs": {"ticket": "Data center on fire, all services down."}})
    assert r.status_code == 202
    run_id = r.json()["id"]

    # Live SSE tail while the model thinks.
    events = []
    with client.stream("GET", f"/v1/runs/{run_id}/events") as resp:
        for line in resp.iter_lines():
            if not line.startswith("data: "):
                continue
            payload = line[len("data: "):]
            if payload == "[DONE]":
                break
            events.append(json.loads(payload))

    assert events[-1]["type"] == "run" and events[-1]["status"] == "succeeded"
    node_events = {(e.get("node_id"), e["status"]) for e in events if e["type"] == "node"}
    assert ("route", "ok") in node_events
    assert ("urgent_path", "ok") in node_events
    assert ("normal_path", "skipped") in node_events

    # Wait for the worker to fill the completion record, then check node detail.
    deadline = time.time() + 30
    while time.time() < deadline:
        body = client.get(f"/v1/runs/{run_id}").json()
        if body["status"] != "running":
            break
        time.sleep(0.1)
    assert body["status"] == "succeeded"
    node = client.get(f"/v1/runs/{run_id}/nodes/route").json()
    assert node["output"] == "urgent_path"
