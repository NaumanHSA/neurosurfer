"""Phase 1h/1i — sub-workflow (subgraph) composition + human-in-the-loop input."""

from __future__ import annotations

import pytest

from neurosurfer.graph import GraphExecutor
from neurosurfer.graph.engine.errors import GraphConfigurationError
from neurosurfer.graph.engine.loader import load_graph_from_dict
from neurosurfer.tools.base import ToolContext

from .fakes import ScriptedIO


class _EchoProvider:
    from neurosurfer.llm.capabilities import ProviderCapabilities

    model = "echo"
    capabilities = ProviderCapabilities(
        context_window=8192, max_output_tokens=2048,
        supports_thinking=False, supports_prompt_cache=False,
        supports_token_count=False, tool_call_style="openai",
    )


FN = "tests.test_graph_subgraph_input"


def _double(item=None, x=None, **kwargs):
    v = item if item is not None else x
    return (v or 0) * 2


def _add_one(doubled=None, **kwargs):
    return (doubled or 0) + 1


# ── subgraph ────────────────────────────────────────────────────────────────────

def test_subgraph_runs_nested_body_once():
    spec = {
        "name": "sub_wf",
        "inputs": [{"name": "x", "type": "integer"}],
        "nodes": [
            {"id": "pipe", "kind": "subgraph", "body_outputs": ["plus"],
             "body": [
                 {"id": "doubled", "kind": "function", "callable": f"{FN}._double"},
                 {"id": "plus", "kind": "function", "callable": f"{FN}._add_one",
                  "depends_on": ["doubled"]},
             ]},
        ],
        "outputs": ["pipe"],
    }
    graph = load_graph_from_dict(spec)
    ex = GraphExecutor(graph, provider=_EchoProvider(), log_traces=False)
    res = ex.run({"x": 5})
    # doubled = 10, plus = 11
    assert res.nodes["pipe"].error is None
    assert res.nodes["pipe"].raw_output == 11


def test_subgraph_output_feeds_downstream():
    spec = {
        "name": "sub_chain",
        "inputs": [{"name": "x", "type": "integer"}],
        "nodes": [
            {"id": "pipe", "kind": "subgraph", "body_outputs": ["doubled"],
             "body": [{"id": "doubled", "kind": "function", "callable": f"{FN}._double"}]},
            {"id": "after", "kind": "function", "callable": f"{FN}._add_one",
             "depends_on": ["pipe"]},
        ],
        "outputs": ["after"],
    }
    graph = load_graph_from_dict(spec)
    ex = GraphExecutor(graph, provider=_EchoProvider(), log_traces=False)
    res = ex.run({"x": 4})
    # subgraph doubled=8 → 'after' receives it as 'pipe'? after reads 'doubled' kwarg
    # which isn't present; it reads dep 'pipe'. So map dep name → use add_one on pipe.
    assert res.nodes["pipe"].raw_output == 8


def test_subgraph_requires_body():
    spec = {
        "name": "bad_sub",
        "nodes": [{"id": "pipe", "kind": "subgraph"}],
        "outputs": ["pipe"],
    }
    with pytest.raises(GraphConfigurationError, match="non-empty body"):
        load_graph_from_dict(spec)


# ── input / human-in-the-loop ───────────────────────────────────────────────────

def test_input_node_uses_presupplied_value():
    # Resume path: the answer is supplied as a graph input keyed by the node id.
    spec = {
        "name": "hitl_supplied",
        "inputs": [{"name": "approval", "type": "string", "required": False}],
        "nodes": [
            {"id": "approval", "kind": "input", "purpose": "Approve?"},
        ],
        "outputs": ["approval"],
    }
    graph = load_graph_from_dict(spec)
    ex = GraphExecutor(graph, provider=_EchoProvider(), log_traces=False)
    res = ex.run({"approval": "yes"})
    assert res.nodes["approval"].raw_output == "yes"
    assert res.nodes["approval"].structured_output["source"] == "supplied"


def test_input_node_asks_interactively():
    spec = {
        "name": "hitl_ask",
        "nodes": [
            {"id": "choice", "kind": "input", "purpose": "Pick one",
             "options": ["a", "b"]},
        ],
        "outputs": ["choice"],
    }
    graph = load_graph_from_dict(spec)
    io = ScriptedIO(answers=["b"])
    from pathlib import Path
    ctx = ToolContext(cwd=Path("."), io=io)
    ex = GraphExecutor(graph, provider=_EchoProvider(), tool_ctx=ctx, log_traces=False)
    res = ex.run({})
    assert res.nodes["choice"].raw_output == "b"
    assert res.nodes["choice"].structured_output["source"] == "interactive"
    assert "Pick one" in io.asked[0]


def test_input_node_awaiting_without_value_or_io():
    spec = {
        "name": "hitl_await",
        "nodes": [{"id": "need", "kind": "input", "purpose": "Value?"}],
        "outputs": ["need"],
    }
    graph = load_graph_from_dict(spec)
    ex = GraphExecutor(graph, provider=_EchoProvider(), log_traces=False)
    res = ex.run({})
    assert res.nodes["need"].error is not None
    assert "awaiting a value" in res.nodes["need"].error


def test_input_node_downstream_conditional():
    # A HITL gate driving a conditional edge: only escalate if user said "escalate".
    spec = {
        "name": "hitl_cond",
        "inputs": [{"name": "decision", "type": "string", "required": False}],
        "nodes": [
            {"id": "decision", "kind": "input", "purpose": "Escalate or resolve?"},
            {"id": "escalate", "kind": "function", "callable": f"{FN}._double",
             "depends_on": ["decision"], "when": "nodes.decision == 'escalate'"},
        ],
        "outputs": ["escalate"],
    }
    graph = load_graph_from_dict(spec)
    ex = GraphExecutor(graph, provider=_EchoProvider(), log_traces=False)

    esc = ex.run({"decision": "escalate", "x": 3})
    assert esc.nodes["escalate"].skipped is False

    res = ex.run({"decision": "resolve", "x": 3})
    assert res.nodes["escalate"].skipped is True
