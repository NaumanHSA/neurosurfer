"""Phase 1e/1f — bounded loop + map/fan-out constructs.

Body sub-graphs use function-kind nodes so no LLM is needed. Covers loop
termination (break + ceiling), accumulation, map over a collection (serial and
concurrent), the implicit gather, and load-time validation.
"""

from __future__ import annotations

import pytest

from neurosurfer.graph import GraphExecutor
from neurosurfer.graph.engine.errors import GraphConfigurationError
from neurosurfer.graph.engine.loader import load_graph_from_dict


class _EchoProvider:
    from neurosurfer.llm.capabilities import ProviderCapabilities

    model = "echo"
    capabilities = ProviderCapabilities(
        context_window=8192, max_output_tokens=2048,
        supports_thinking=False, supports_prompt_cache=False,
        supports_token_count=False, tool_call_style="openai",
    )


FN = "tests.test_graph_iteration"

# Global counter so a loop body can make progress across iterations.
_counter = {"n": 0}


def _increment(**kwargs):
    _counter["n"] += 1
    return {"value": _counter["n"]}


def _double(item, **kwargs):
    return item * 2


def _echo_index(index=0, **kwargs):
    return index


# ── loop ────────────────────────────────────────────────────────────────────────

def test_loop_breaks_on_condition():
    _counter["n"] = 0
    spec = {
        "name": "loop_break",
        "nodes": [
            {"id": "count", "kind": "loop", "max_iterations": 10,
             "break_when": "nodes.step.value >= 3",
             "body": [
                 {"id": "step", "kind": "function", "callable": f"{FN}._increment"},
             ]},
        ],
        "outputs": ["count"],
    }
    graph = load_graph_from_dict(spec)
    ex = GraphExecutor(graph, provider=_EchoProvider(), log_traces=False)
    res = ex.run({})
    node = res.nodes["count"]
    assert node.error is None
    assert node.structured_output["iterations"] == 3
    assert node.structured_output["broke_early"] is True
    assert node.raw_output == {"value": 3}


def test_loop_respects_max_iterations_ceiling():
    _counter["n"] = 0
    spec = {
        "name": "loop_ceiling",
        "nodes": [
            {"id": "count", "kind": "loop", "max_iterations": 4,
             "break_when": "nodes.step.value >= 999",  # never true
             "body": [{"id": "step", "kind": "function", "callable": f"{FN}._increment"}]},
        ],
        "outputs": ["count"],
    }
    graph = load_graph_from_dict(spec)
    ex = GraphExecutor(graph, provider=_EchoProvider(), log_traces=False)
    res = ex.run({})
    assert res.nodes["count"].structured_output["iterations"] == 4
    assert res.nodes["count"].structured_output["broke_early"] is False


def test_loop_accumulates_results():
    _counter["n"] = 0
    spec = {
        "name": "loop_acc",
        "nodes": [
            {"id": "count", "kind": "loop", "max_iterations": 3,
             "accumulate": "history",
             "body": [{"id": "step", "kind": "function", "callable": f"{FN}._increment"}]},
        ],
        "outputs": ["count"],
    }
    graph = load_graph_from_dict(spec)
    ex = GraphExecutor(graph, provider=_EchoProvider(), log_traces=False)
    res = ex.run({})
    # accumulate → raw_output is the list of every iteration's output
    assert res.nodes["count"].raw_output == [{"value": 1}, {"value": 2}, {"value": 3}]


# ── map ─────────────────────────────────────────────────────────────────────────

def test_map_over_input_collection():
    spec = {
        "name": "map_wf",
        "inputs": [{"name": "nums", "type": "array"}],
        "nodes": [
            {"id": "doubled", "kind": "map", "over": "inputs.nums", "as": "item",
             "body": [{"id": "d", "kind": "function", "callable": f"{FN}._double"}]},
        ],
        "outputs": ["doubled"],
    }
    graph = load_graph_from_dict(spec)
    ex = GraphExecutor(graph, provider=_EchoProvider(), log_traces=False)
    res = ex.run({"nums": [1, 2, 3]})
    assert res.nodes["doubled"].raw_output == [2, 4, 6]
    assert res.nodes["doubled"].structured_output["count"] == 3


def test_map_concurrent_preserves_order():
    spec = {
        "name": "map_conc",
        "inputs": [{"name": "nums", "type": "array"}],
        "nodes": [
            {"id": "doubled", "kind": "map", "over": "inputs.nums", "as": "item",
             "concurrency": 4,
             "body": [{"id": "d", "kind": "function", "callable": f"{FN}._double"}]},
        ],
        "outputs": ["doubled"],
    }
    graph = load_graph_from_dict(spec)
    ex = GraphExecutor(graph, provider=_EchoProvider(), log_traces=False)
    res = ex.run({"nums": [5, 10, 15, 20]})
    assert res.nodes["doubled"].raw_output == [10, 20, 30, 40]


def test_map_output_feeds_downstream_gather():
    spec = {
        "name": "map_gather",
        "inputs": [{"name": "nums", "type": "array"}],
        "nodes": [
            {"id": "doubled", "kind": "map", "over": "inputs.nums", "as": "item",
             "body": [{"id": "d", "kind": "function", "callable": f"{FN}._double"}]},
            {"id": "total", "kind": "function",
             "callable": f"{FN}._sum_list", "depends_on": ["doubled"]},
        ],
        "outputs": ["total"],
    }
    graph = load_graph_from_dict(spec)
    ex = GraphExecutor(graph, provider=_EchoProvider(), log_traces=False)
    res = ex.run({"nums": [1, 2, 3]})
    assert res.nodes["total"].raw_output == 12  # (2+4+6)


def _sum_list(doubled=None, **kwargs):
    return sum(doubled or [])


def test_map_empty_collection_yields_empty_list():
    spec = {
        "name": "map_empty",
        "inputs": [{"name": "nums", "type": "array"}],
        "nodes": [
            {"id": "doubled", "kind": "map", "over": "inputs.nums", "as": "item",
             "body": [{"id": "d", "kind": "function", "callable": f"{FN}._double"}]},
        ],
        "outputs": ["doubled"],
    }
    graph = load_graph_from_dict(spec)
    ex = GraphExecutor(graph, provider=_EchoProvider(), log_traces=False)
    res = ex.run({"nums": []})
    assert res.nodes["doubled"].raw_output == []


# ── validation ──────────────────────────────────────────────────────────────────

def test_loop_requires_max_iterations():
    spec = {
        "name": "bad_loop",
        "nodes": [
            {"id": "l", "kind": "loop",
             "body": [{"id": "s", "kind": "function", "callable": f"{FN}._increment"}]},
        ],
        "outputs": ["l"],
    }
    with pytest.raises(GraphConfigurationError, match="max_iterations"):
        load_graph_from_dict(spec)


def test_map_requires_over_expression():
    spec = {
        "name": "bad_map",
        "nodes": [
            {"id": "m", "kind": "map",
             "body": [{"id": "d", "kind": "function", "callable": f"{FN}._double"}]},
        ],
        "outputs": ["m"],
    }
    with pytest.raises(GraphConfigurationError, match="over"):
        load_graph_from_dict(spec)


def test_loop_body_bad_internal_dep_rejected():
    spec = {
        "name": "bad_body",
        "nodes": [
            {"id": "l", "kind": "loop", "max_iterations": 2,
             "body": [
                 {"id": "s", "kind": "function", "callable": f"{FN}._increment",
                  "depends_on": ["ghost"]},
             ]},
        ],
        "outputs": ["l"],
    }
    with pytest.raises(GraphConfigurationError, match="not part of the body"):
        load_graph_from_dict(spec)
