"""Phase 1h/1i — sub-workflow (subgraph) composition + human-in-the-loop input."""

from __future__ import annotations

import pytest

from neurosurfer.graph import GraphExecutor
from neurosurfer.graph.engine.errors import GraphConfigurationError
from neurosurfer.graph.engine.loader import load_graph_from_dict
from neurosurfer.tools.base import ToolContext

from ..fakes import ScriptedIO


class _EchoProvider:
    from neurosurfer.llm.capabilities import ProviderCapabilities

    model = "echo"
    capabilities = ProviderCapabilities(
        context_window=8192, max_output_tokens=2048,
        supports_thinking=False, supports_prompt_cache=False,
        supports_token_count=False, tool_call_style="openai",
    )


FN = "tests.engine.test_graph_subgraph_input"


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


def test_input_node_asks_what_instructions_says():
    """The question a person is asked comes from `instructions` too.

    `_run_input_node` built its question from `purpose or goal`, so a node
    authored the new way asked "Input needed for 'choice'" — the studio writes
    `instructions` and clears the older three. Same defect as the router's, and
    the same silence: someone is simply asked a worse question.
    """
    spec = {
        "name": "hitl_ask_new",
        "nodes": [
            {"id": "choice", "kind": "input", "instructions": "Which environment?",
             "options": ["staging", "prod"]},
        ],
        "outputs": ["choice"],
    }
    graph = load_graph_from_dict(spec)
    io = ScriptedIO(answers=["prod"])
    from pathlib import Path
    ctx = ToolContext(cwd=Path("."), io=io)
    ex = GraphExecutor(graph, provider=_EchoProvider(), tool_ctx=ctx, log_traces=False)
    res = ex.run({})
    assert res.nodes["choice"].raw_output == "prod"
    assert "Which environment?" in io.asked[0]
    assert "Input needed for" not in io.asked[0]


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


# ── the input node's contract with its caller ────────────────────────────────
#
# The key an input node reads from — `writes` if set, else the node id — is the
# whole agreement between a workflow and whoever runs it. It was a convention
# rather than a declaration: the executor applied it, the studio reimplemented it
# in TypeScript, and `normalize_and_validate_graph_inputs` knew nothing about it.
# Both tests below describe a value a person typed being thrown away.

def _chat_graph(declared: list[dict] | None = None) -> dict:
    spec = {
        "name": "chat_wf",
        "nodes": [
            {"id": "ask", "kind": "input", "instructions": "What topic?"},
            {"id": "done", "kind": "output", "value": "{ask}", "depends_on": ["ask"]},
        ],
        "outputs": [],
    }
    if declared is not None:
        spec["inputs"] = declared
    return spec


def test_input_node_key_survives_a_declared_input_list():
    """The trap: with any `graph.inputs` declared, an undeclared key was warned
    about and **dropped** — so the chat message was discarded before the node
    that asked for it ever looked, and the run parked saying nothing was
    supplied."""
    graph = load_graph_from_dict(
        _chat_graph(declared=[{"name": "unrelated", "type": "string", "required": False}])
    )
    res = GraphExecutor(graph, provider=_EchoProvider(), log_traces=False).run(
        {"ask": "otters"}
    )
    assert res.nodes["ask"].raw_output == "otters"
    # The output node collects: the node wired into it, plus its own composed
    # `value` keyed by its id. Both are answers a caller asked to see.
    assert res.final["done"] == {"ask": "otters", "done": "otters"}


def test_bare_string_lands_where_the_graph_is_listening():
    """The other trap: a non-dict argument was wrapped as `{"query": ...}`, a key
    nothing reads. With one input node, the shortcut now lands on its key."""
    graph = load_graph_from_dict(_chat_graph())
    res = GraphExecutor(graph, provider=_EchoProvider(), log_traces=False).run("otters")
    assert res.nodes["ask"].raw_output == "otters"


def test_bare_string_still_falls_back_to_query_without_an_input_node():
    from neurosurfer.graph.engine.utils import normalize_and_validate_graph_inputs

    graph = load_graph_from_dict({
        "name": "no_input_node",
        "nodes": [{"id": "a", "kind": "base", "instructions": "hi"}],
        "outputs": ["a"],
    })
    assert normalize_and_validate_graph_inputs(graph, "otters") == {"query": "otters"}


def test_input_node_key_follows_writes():
    from neurosurfer.graph.engine.utils import input_node_keys

    graph = load_graph_from_dict({
        "name": "writes_wf",
        "nodes": [
            {"id": "ask", "kind": "input", "writes": "topic", "instructions": "?"},
            {"id": "other", "kind": "base", "instructions": "hi", "depends_on": ["ask"]},
        ],
        "outputs": ["other"],
    })
    assert input_node_keys(graph) == {"topic": "ask"}


# ── dict mode: the node collects the graph's declared inputs ─────────────────

def _dict_graph() -> dict:
    return {
        "name": "dict_wf",
        "inputs": [
            {"name": "object", "type": "string", "required": True},
            {"name": "color", "type": "string", "required": True},
        ],
        "nodes": [
            {"id": "input", "kind": "input", "input_mode": "dict"},
            {"id": "out", "kind": "output", "value": "{object}/{color}",
             "depends_on": ["input"]},
        ],
        "outputs": [],
    }


def test_dict_input_node_collects_the_declared_inputs():
    """The defect a form submission hit on its first run.

    A `dict`-mode node was still resolved by the single-key rule — `writes` or
    the node id — so a form supplying `{object, color}` left the node waiting for
    something called `input`, and the run parked as `awaiting_input` with both
    answers already in hand.
    """
    graph = load_graph_from_dict(_dict_graph())
    res = GraphExecutor(graph, provider=_EchoProvider(), log_traces=False).run(
        {"object": "tokyo", "color": "red"}
    )
    assert res.nodes["input"].error is None
    assert res.nodes["input"].raw_output == {"object": "tokyo", "color": "red"}
    assert res.final["out"] == {
        "input": {"object": "tokyo", "color": "red"},
        "out": "tokyo/red",
    }


def test_dict_input_node_names_what_is_missing():
    graph = load_graph_from_dict(_dict_graph())
    res = GraphExecutor(graph, provider=_EchoProvider(), log_traces=False).run(
        {"object": "tokyo"}
    )
    err = res.nodes["input"].error
    assert err and "'color'" in err
    assert "'object'" not in err  # it was supplied; don't ask for it again


def test_dict_input_node_with_no_declared_inputs_says_so():
    graph = load_graph_from_dict({
        "name": "dict_empty",
        "nodes": [{"id": "input", "kind": "input", "input_mode": "dict"}],
        "outputs": ["input"],
    })
    res = GraphExecutor(graph, provider=_EchoProvider(), log_traces=False).run({})
    assert "declares no inputs" in (res.nodes["input"].error or "")


def test_text_mode_is_unchanged_by_the_dict_arm():
    graph = load_graph_from_dict(_chat_graph())
    res = GraphExecutor(graph, provider=_EchoProvider(), log_traces=False).run(
        {"ask": "otters"}
    )
    assert res.nodes["ask"].raw_output == "otters"
