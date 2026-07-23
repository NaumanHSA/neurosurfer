"""Phase 1c/1d — conditional edges (`when` guards) + router nodes.

Exercises the dynamic scheduler: `when`-guard pruning, OR-join semantics, and
expression + LLM routers, using function-kind nodes so no real LLM is needed
(except the scripted-provider LLM-router test).
"""

from __future__ import annotations

import pytest

from neurosurfer.graph import Graph, GraphExecutor, GraphNode
from neurosurfer.graph.engine.errors import GraphConfigurationError
from neurosurfer.graph.engine.loader import load_graph_from_dict

from .fakes import ScriptedProvider


class _EchoProvider:
    model = "echo"
    from neurosurfer.llm.capabilities import ProviderCapabilities

    capabilities = ProviderCapabilities(
        context_window=8192, max_output_tokens=2048,
        supports_thinking=False, supports_prompt_cache=False,
        supports_token_count=False, tool_call_style="openai",
    )


# ── function-node helpers referenced by import path ─────────────────────────────

def _mark(**kwargs):
    return "ran"


def _const_high(**kwargs):
    return {"score": 9}


def _const_low(**kwargs):
    return {"score": 1}


def _collect(**kwargs):
    # A join node: returns which upstream branches produced the "ran" marker
    # (ignores graph inputs, which functions also receive as kwargs).
    return {k: v for k, v in kwargs.items() if v == "ran"}


FN = "tests.test_graph_control_flow"


# ── conditional edges (when guards) ─────────────────────────────────────────────

def test_when_guard_prunes_node_when_false():
    graph = Graph(
        name="cond",
        inputs=[{"name": "n", "type": "integer"}],
        nodes=[
            GraphNode(id="a", kind="function", callable=f"{FN}._mark"),
            GraphNode(id="b", kind="function", callable=f"{FN}._mark",
                      depends_on=["a"], when="inputs.n > 5"),
        ],
        outputs=["b"],
    )
    ex = GraphExecutor(graph, provider=_EchoProvider(), log_traces=False)

    hi = ex.run({"n": 10})
    assert hi.nodes["b"].skipped is False
    assert hi.nodes["b"].raw_output == "ran"

    lo = ex.run({"n": 1})
    assert lo.nodes["b"].skipped is True
    assert "condition false" in lo.nodes["b"].skip_reason


def test_when_guard_reads_upstream_node_output():
    graph = Graph(
        name="cond2",
        nodes=[
            GraphNode(id="score", kind="function", callable=f"{FN}._const_high"),
            GraphNode(id="escalate", kind="function", callable=f"{FN}._mark",
                      depends_on=["score"], when="nodes.score.score >= 5"),
        ],
        outputs=["escalate"],
    )
    ex = GraphExecutor(graph, provider=_EchoProvider(), log_traces=False)
    res = ex.run({})
    assert res.nodes["escalate"].raw_output == "ran"


def test_or_join_runs_when_any_branch_live():
    # b depends on a with a false guard (pruned); c depends on a and runs.
    # join depends on b and c — must still run because c is live (OR-join).
    graph = Graph(
        name="orjoin",
        inputs=[{"name": "n", "type": "integer"}],
        nodes=[
            GraphNode(id="a", kind="function", callable=f"{FN}._mark"),
            GraphNode(id="b", kind="function", callable=f"{FN}._mark",
                      depends_on=["a"], when="inputs.n > 100"),
            GraphNode(id="c", kind="function", callable=f"{FN}._mark", depends_on=["a"]),
            GraphNode(id="join", kind="function", callable=f"{FN}._collect",
                      depends_on=["b", "c"]),
        ],
        outputs=["join"],
    )
    ex = GraphExecutor(graph, provider=_EchoProvider(), log_traces=False)
    res = ex.run({"n": 1})
    assert res.nodes["b"].skipped is True
    assert res.nodes["c"].skipped is False
    assert res.nodes["join"].skipped is False
    # join only received c's output (b was pruned → None, filtered out)
    assert res.nodes["join"].raw_output == {"c": "ran"}


def test_all_branches_pruned_prunes_join():
    graph = Graph(
        name="deadjoin",
        inputs=[{"name": "n", "type": "integer"}],
        nodes=[
            GraphNode(id="a", kind="function", callable=f"{FN}._mark"),
            GraphNode(id="b", kind="function", callable=f"{FN}._mark",
                      depends_on=["a"], when="inputs.n > 100"),
            GraphNode(id="join", kind="function", callable=f"{FN}._collect",
                      depends_on=["b"]),
        ],
        outputs=["join"],
    )
    ex = GraphExecutor(graph, provider=_EchoProvider(), log_traces=False)
    res = ex.run({"n": 1})
    assert res.nodes["b"].skipped is True
    assert res.nodes["join"].skipped is True
    assert "no active branch" in res.nodes["join"].skip_reason


# ── router (expression) ─────────────────────────────────────────────────────────

def _router_graph() -> dict:
    return {
        "name": "router_wf",
        "inputs": [{"name": "n", "type": "integer"}],
        "nodes": [
            {"id": "classify", "kind": "function", "callable": f"{FN}._const_high"},
            {"id": "route", "kind": "router", "depends_on": ["classify"],
             "cases": [
                 {"when": "inputs.n > 5", "to": "big"},
                 {"when": "inputs.n <= 5", "to": "small"},
             ],
             "default": "small"},
            {"id": "big", "kind": "function", "callable": f"{FN}._mark", "depends_on": ["route"]},
            {"id": "small", "kind": "function", "callable": f"{FN}._mark", "depends_on": ["route"]},
            {"id": "join", "kind": "function", "callable": f"{FN}._collect",
             "depends_on": ["big", "small"]},
        ],
        "outputs": ["join"],
    }


def test_router_selects_branch_and_prunes_other():
    graph = load_graph_from_dict(_router_graph())
    ex = GraphExecutor(graph, provider=_EchoProvider(), log_traces=False)

    big = ex.run({"n": 10})
    assert big.nodes["route"].raw_output == "big"
    assert big.nodes["big"].skipped is False
    assert big.nodes["small"].skipped is True
    assert big.nodes["join"].raw_output == {"big": "ran"}

    small = ex.run({"n": 2})
    assert small.nodes["route"].raw_output == "small"
    assert small.nodes["small"].skipped is False
    assert small.nodes["big"].skipped is True
    assert small.nodes["join"].raw_output == {"small": "ran"}


def test_router_falls_back_to_default():
    spec = {
        "name": "router_default",
        "inputs": [{"name": "n", "type": "integer"}],
        "nodes": [
            {"id": "route", "kind": "router",
             "cases": [{"when": "inputs.n > 100", "to": "big"}],
             "default": "small"},
            {"id": "big", "kind": "function", "callable": f"{FN}._mark", "depends_on": ["route"]},
            {"id": "small", "kind": "function", "callable": f"{FN}._mark", "depends_on": ["route"]},
        ],
        "outputs": ["big", "small"],
    }
    graph = load_graph_from_dict(spec)
    ex = GraphExecutor(graph, provider=_EchoProvider(), log_traces=False)
    res = ex.run({"n": 1})
    assert res.nodes["route"].raw_output == "small"
    assert res.nodes["small"].skipped is False
    assert res.nodes["big"].skipped is True


# ── router (routes: the router IS the classifier) ──────────────────────────────

def _routes_graph(default: str | None = "reply", repair: bool = True) -> dict:
    spec = {
        "name": "routes_wf",
        "inputs": [{"name": "ticket", "type": "string"}],
        "nodes": [
            {"id": "route", "kind": "router", "repair": repair,
             "goal": "Route this support ticket by urgency: {ticket}",
             "routes": {"urgent": "escalate", "billing": "finance", "routine": "reply"}},
            {"id": "escalate", "kind": "function", "callable": f"{FN}._mark", "depends_on": ["route"]},
            {"id": "finance", "kind": "function", "callable": f"{FN}._mark", "depends_on": ["route"]},
            {"id": "reply", "kind": "function", "callable": f"{FN}._mark", "depends_on": ["route"]},
        ],
        "outputs": ["escalate", "finance", "reply"],
    }
    if default:
        spec["nodes"][0]["default"] = default
    return spec


def test_routes_router_is_n_way_classifier():
    # ONE router, THREE targets — the model's label picks the branch directly.
    graph = load_graph_from_dict(_routes_graph())
    provider = ScriptedProvider(turns=[("billing", [])])
    res = GraphExecutor(graph, provider=provider, log_traces=False).run(
        {"ticket": "I was charged twice for my subscription"})
    assert res.nodes["route"].raw_output == "finance"
    assert res.nodes["finance"].skipped is False
    assert res.nodes["escalate"].skipped is True
    assert res.nodes["reply"].skipped is True


def test_routes_router_tolerates_wrapping_prose():
    graph = load_graph_from_dict(_routes_graph())
    provider = ScriptedProvider(turns=[("  The route is: URGENT.\n", [])])
    res = GraphExecutor(graph, provider=provider, log_traces=False).run({"ticket": "x"})
    assert res.nodes["route"].raw_output == "escalate"


def test_routes_router_repair_retries_invalid_answer():
    graph = load_graph_from_dict(_routes_graph())
    # First answer matches no label → repair retry → valid answer.
    provider = ScriptedProvider(turns=[("I think it's about money?", []), ("billing", [])])
    res = GraphExecutor(graph, provider=provider, log_traces=False).run({"ticket": "x"})
    assert res.nodes["route"].raw_output == "finance"
    assert provider.calls == 2  # exactly one repair round-trip


def test_routes_router_falls_back_to_default_after_repair():
    graph = load_graph_from_dict(_routes_graph(default="reply"))
    provider = ScriptedProvider(turns=[("nonsense", []), ("still nonsense", [])])
    res = GraphExecutor(graph, provider=provider, log_traces=False).run({"ticket": "x"})
    assert res.nodes["route"].raw_output == "reply"
    assert res.nodes["reply"].skipped is False


def test_routes_router_no_default_errors_honestly():
    graph = load_graph_from_dict(_routes_graph(default=None, repair=False))
    provider = ScriptedProvider(turns=[("nonsense", [])])
    res = GraphExecutor(graph, provider=provider, log_traces=False).run({"ticket": "x"})
    assert res.nodes["route"].error is not None
    assert "no default" in res.nodes["route"].error


def test_routes_and_cases_together_rejected():
    spec = _routes_graph()
    spec["nodes"][0]["cases"] = [{"when": "True", "to": "reply"}]
    with pytest.raises(GraphConfigurationError, match="both"):
        load_graph_from_dict(spec)


def test_routes_target_must_depend_on_router():
    spec = _routes_graph()
    spec["nodes"][1]["depends_on"] = []  # escalate no longer depends on route
    with pytest.raises(GraphConfigurationError, match="depends_on"):
        load_graph_from_dict(spec)


# ── router (LLM) ────────────────────────────────────────────────────────────────

def test_llm_router_picks_label():
    # Cases carry only labels (no `when`) → LLM router. Scripted provider returns
    # the label text so the route is deterministic in the test.
    spec = {
        "name": "llm_router",
        "nodes": [
            {"id": "route", "kind": "router", "purpose": "Decide urgency",
             "cases": [
                 {"to": "urgent", "label": "urgent"},
                 {"to": "normal", "label": "normal"},
             ]},
            {"id": "urgent", "kind": "function", "callable": f"{FN}._mark", "depends_on": ["route"]},
            {"id": "normal", "kind": "function", "callable": f"{FN}._mark", "depends_on": ["route"]},
        ],
        "outputs": ["urgent", "normal"],
    }
    graph = load_graph_from_dict(spec)
    provider = ScriptedProvider(turns=[("urgent", [])])
    ex = GraphExecutor(graph, provider=provider, log_traces=False)
    res = ex.run({})
    assert res.nodes["route"].raw_output == "urgent"
    assert res.nodes["urgent"].skipped is False
    assert res.nodes["normal"].skipped is True


# ── validation ──────────────────────────────────────────────────────────────────

def test_router_target_must_depend_on_router():
    spec = {
        "name": "bad_router",
        "nodes": [
            {"id": "route", "kind": "router",
             "cases": [{"when": "True", "to": "x"}], "default": "x"},
            # x does NOT depend on route → must be rejected
            {"id": "x", "kind": "function", "callable": f"{FN}._mark"},
        ],
        "outputs": ["x"],
    }
    with pytest.raises(GraphConfigurationError, match="depends_on"):
        load_graph_from_dict(spec)


def test_router_unknown_target_rejected():
    spec = {
        "name": "bad_router2",
        "nodes": [
            {"id": "route", "kind": "router",
             "cases": [{"when": "True", "to": "ghost"}]},
        ],
        "outputs": ["route"],
    }
    with pytest.raises(GraphConfigurationError, match="unknown node id"):
        load_graph_from_dict(spec)


def test_invalid_when_expression_rejected_at_load():
    spec = {
        "name": "bad_expr",
        "nodes": [
            {"id": "a", "kind": "function", "callable": f"{FN}._mark",
             "when": "__import__('os')"},
        ],
        "outputs": ["a"],
    }
    with pytest.raises(GraphConfigurationError, match="invalid expression"):
        load_graph_from_dict(spec)


def test_router_fields_on_non_router_rejected():
    spec = {
        "name": "misplaced_cases",
        "nodes": [
            {"id": "a", "kind": "function", "callable": f"{FN}._mark",
             "cases": [{"when": "True", "to": "a"}]},
        ],
        "outputs": ["a"],
    }
    with pytest.raises(GraphConfigurationError, match="router fields"):
        load_graph_from_dict(spec)
