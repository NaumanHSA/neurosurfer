"""Phase 1e/1f — bounded loop + map/fan-out constructs.

Body sub-graphs use function-kind nodes so no LLM is needed. Covers loop
termination (break + ceiling), accumulation, map over a collection (serial and
concurrent), the implicit gather, and load-time validation.
"""

from __future__ import annotations

from pathlib import Path

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


FN = "tests.engine.test_graph_iteration"

# Global counter so a loop body can make progress across iterations.
_counter = {"n": 0}


def _increment(**kwargs):
    _counter["n"] += 1
    return {"value": _counter["n"]}


def _double(item, **kwargs):
    return item * 2


def _triple_number(number, **kwargs):
    # Named for the `as: number` binding — fails loudly if the alias was dropped.
    return number * 3


def _echo_index(index=0, **kwargs):
    return index


# The `functions:` sidecar these loop tests point at — a standalone file, which
# is what a sidecar always is (see its docstring).
FNS_FILE = str((Path(__file__).parent / "loop_fns.py").resolve())


# ── loop ────────────────────────────────────────────────────────────────────────

def test_loop_breaks_on_condition():
    _counter["n"] = 0
    spec = {
        "name": "loop_break",
        "functions": FNS_FILE,
        "nodes": [
            {"id": "count", "kind": "loop", "max_iterations": 10,
             "until": "stop_at_three",
             "body": [
                 {"id": "step", "kind": "function", "callable": f"{FN}._increment"},
             ]},
        ],
        "outputs": ["count"],
    }
    graph = load_graph_from_dict(spec)
    graph.bind_functions(None)      # absolute path, so no base dir needed
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
        "functions": FNS_FILE,
        "nodes": [
            {"id": "count", "kind": "loop", "max_iterations": 4,
             "until": "never_stops",           # never true
             "body": [{"id": "step", "kind": "function", "callable": f"{FN}._increment"}]},
        ],
        "outputs": ["count"],
    }
    graph = load_graph_from_dict(spec)
    graph.bind_functions(None)
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


def test_map_custom_item_var_alias_survives_the_loader():
    """`as:` is an alias for `item_var` — the loader's unknown-key sanitizer must
    keep it. It once stripped aliases, silently rebinding the body to the default
    `item` (invisible in tests that used `as: item`)."""
    spec = {
        "name": "map_alias",
        "inputs": [{"name": "nums", "type": "array"}],
        "nodes": [
            {"id": "tripled", "kind": "map", "over": "inputs.nums", "as": "number",
             "body": [{"id": "t", "kind": "function", "callable": f"{FN}._triple_number"}]},
        ],
        "outputs": ["tripled"],
    }
    graph = load_graph_from_dict(spec)
    assert graph.nodes[0].item_var == "number"
    ex = GraphExecutor(graph, provider=_EchoProvider(), log_traces=False)
    res = ex.run({"nums": [1, 2, 3]})
    assert res.nodes["tripled"].raw_output == [3, 6, 9]


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


# ── loop with `until` (LLM-judged stop condition + feedback) ───────────────────

from ..fakes import ScriptedProvider  # noqa: E402

_feedbacks: list = []


def _echo_feedback(feedback="", **kwargs):
    _feedbacks.append(feedback)
    return f"draft {len(_feedbacks)}"


def _until_spec(repair: bool = True) -> dict:
    return {
        "name": "until_wf",
        "nodes": [
            {"id": "refine", "kind": "loop", "max_iterations": 3, "repair": repair,
             "until": "the draft is good enough",
             "accumulate": "attempts",
             "body": [{"id": "draft", "kind": "function",
                       "callable": f"{FN}._echo_feedback"}]},
        ],
        "outputs": ["refine"],
    }


def test_until_judge_stops_and_feeds_back():
    _feedbacks.clear()
    graph = load_graph_from_dict(_until_spec())
    provider = ScriptedProvider(turns=[
        ("CONTINUE - make it shorter and punchier", []),   # judge, iteration 1
        ("STOP - looks good now", []),                     # judge, iteration 2
    ])
    res = GraphExecutor(graph, provider=provider, log_traces=False).run({})
    node = res.nodes["refine"]
    assert node.error is None
    assert node.structured_output["iterations"] == 2
    assert node.structured_output["broke_early"] is True
    # The judge log records both verdicts with reasons.
    judge = node.structured_output["judge"]
    assert [j["stop"] for j in judge] == [False, True]
    assert "shorter" in judge[0]["reason"]
    # The CONTINUE reason reached iteration 2 as {feedback}.
    assert _feedbacks == ["", "make it shorter and punchier"]


def test_until_judge_repair_retries_unparseable_answer():
    _feedbacks.clear()
    graph = load_graph_from_dict(_until_spec())
    provider = ScriptedProvider(turns=[
        ("hmm, hard to say really", []),   # unparseable → repair retry
        ("STOP", []),                      # retry answer
    ])
    res = GraphExecutor(graph, provider=provider, log_traces=False).run({})
    assert res.nodes["refine"].structured_output["iterations"] == 1
    assert res.nodes["refine"].structured_output["broke_early"] is True
    assert provider.calls == 2


def test_until_judge_failure_fails_safe_to_ceiling():
    _feedbacks.clear()
    graph = load_graph_from_dict(_until_spec(repair=False))
    provider = ScriptedProvider(turns=[("???", []), ("???", []), ("???", [])])
    res = GraphExecutor(graph, provider=provider, log_traces=False).run({})
    node = res.nodes["refine"]
    assert node.error is None
    assert node.structured_output["iterations"] == 3  # ceiling still bounds it
    assert node.structured_output["broke_early"] is False


class TestUnrelatedCondition:
    """The judge's third verdict, for a condition about a different subject.

    "Stop when winter is here" over a body writing taglines can never be
    satisfied, so CONTINUE would be a lie costing the full ceiling — every
    iteration plus a judge call each — to tell. It rides on the call already
    being made, so noticing is free.
    """

    def _run(self, first_verdict: str, max_iterations: int = 3):
        _feedbacks.clear()
        spec = _until_spec()
        spec["nodes"][0]["max_iterations"] = max_iterations
        spec["nodes"][0]["until"] = "winter has arrived"
        graph = load_graph_from_dict(spec)
        provider = ScriptedProvider(turns=[
            (first_verdict, []), ("STOP - fine", []), ("STOP - fine", []),
        ])
        res = GraphExecutor(graph, provider=provider, log_traces=False).run({})
        assert res.nodes["refine"].error is None
        return res.nodes["refine"].structured_output

    def test_unrelated_stops_at_the_first_iteration(self):
        out = self._run("UNRELATED - the results are taglines, winter is another subject")
        assert out["iterations"] == 1, "must not spend the ceiling on it"
        assert out["broke_early"] is True
        assert out["stopped_reason"] == "condition_unrelated"
        assert "winter" in out["stopped_detail"]
        assert out["judge"][0]["related"] is False

    def test_the_loop_still_returns_its_work(self):
        """Stopping early is not failing — what ran is still the node's output."""
        out = self._run("UNRELATED - different subject")
        assert out["results"], "the completed iteration's output is kept"

    def test_continue_is_unaffected(self):
        out = self._run("CONTINUE - not yet", max_iterations=2)
        assert out["judge"][0]["related"] is True
        assert "stopped_reason" not in out

    def test_an_unreadable_verdict_never_counts_as_unrelated(self):
        """Failing safe means CONTINUE. A judge we could not parse must not be
        what stops someone's loop."""
        out = self._run("mumble mumble", max_iterations=2)
        assert out["judge"][0]["related"] is True
        assert "stopped_reason" not in out


class TestUntilAsAFunction:
    """`until` reads as a function or as prose, and which is a lookup.

    The sandboxed `break_when` expression it replaces could only see the parent
    state through a namespace; a function is handed the body run itself.
    """

    @staticmethod
    def _spec(until, **loop):
        return {
            "name": "fn_loop",
            "functions": FNS_FILE,
            "nodes": [
                {"id": "count", "kind": "loop", "max_iterations": 5, "until": until,
                 "body": [{"id": "step", "kind": "function",
                           "callable": f"{FN}._increment"}], **loop},
            ],
            "outputs": ["count"],
        }

    def _run(self, until, **loop):
        _counter["n"] = 0
        graph = load_graph_from_dict(self._spec(until, **loop))
        graph.bind_functions(None)
        res = GraphExecutor(graph, provider=_EchoProvider(), log_traces=False).run({})
        assert res.nodes["count"].error is None, res.nodes["count"].error
        return res.nodes["count"].structured_output

    def test_a_live_callable_is_used_directly(self):
        """A graph built in Python hands over the function itself — no lookup."""
        _counter["n"] = 0
        from neurosurfer.graph import FunctionNode, Graph, LoopNode  # noqa: PLC0415

        from .loop_fns import stop_at_three  # noqa: PLC0415

        node = LoopNode(
            id="count", max_iterations=5, until=stop_at_three,
            body=[FunctionNode(id="step", callable=f"{FN}._increment")],
        )
        graph = Graph(name="live", nodes=[node], outputs=["count"])
        res = GraphExecutor(graph, provider=_EchoProvider(), log_traces=False).run({})
        assert res.nodes["count"].structured_output["iterations"] == 3

    def test_a_name_in_the_sidecar_is_the_function(self):
        out = self._run("stop_at_three")
        assert out["iterations"] == 3
        assert out["broke_early"] is True
        assert "judge" not in out, "a function must not call the LLM judge"

    def test_a_function_that_never_stops_hits_the_ceiling(self):
        out = self._run("never_stops")
        assert (out["iterations"], out["broke_early"]) == (5, False)

    def test_a_reason_becomes_the_next_feedback(self):
        out = self._run("stop_with_reason")
        assert out["iterations"] == 2 and out["broke_early"] is True

    def test_a_raising_function_fails_the_node(self):
        """The author's own exit condition is broken — iterating past it would be
        running a loop whose bound has failed."""
        _counter["n"] = 0
        graph = load_graph_from_dict(self._spec("explodes"))
        graph.bind_functions(None)
        res = GraphExecutor(graph, provider=_EchoProvider(), log_traces=False).run({})
        assert res.nodes["count"].error is not None
        assert "this condition is broken" in res.nodes["count"].error

    def test_a_name_the_sidecar_lacks_is_an_error_not_a_prompt(self):
        """A typo'd identifier must not be silently sent to an LLM as English."""
        _counter["n"] = 0
        graph = load_graph_from_dict(self._spec("stop_at_thre"))
        graph.bind_functions(None)
        res = GraphExecutor(graph, provider=_EchoProvider(), log_traces=False).run({})
        err = res.nodes["count"].error or ""
        assert "names no function" in err and "stop_at_three" in err

    def test_prose_is_still_prose_when_a_sidecar_exists(self):
        """A sentence is not an identifier, so it never looks like a lookup miss."""
        from neurosurfer.graph.engine.executor.iteration import (  # noqa: PLC0415
            _resolve_until_function,
        )

        graph = load_graph_from_dict(self._spec("the counter has reached three"))
        graph.bind_functions(None)
        ex = GraphExecutor(graph, provider=_EchoProvider(), log_traces=False)
        assert _resolve_until_function(ex, graph.node_map()["count"]) is None

    def test_a_live_callable_serializes_as_its_name(self):
        """YAML cannot hold a function, so a dump names it — round-tripping via
        the sidecar that must define it."""
        from neurosurfer.graph import BaseNode, Graph, LoopNode  # noqa: PLC0415

        from .loop_fns import stop_at_three  # noqa: PLC0415

        graph = Graph(
            name="dump", functions=FNS_FILE,
            nodes=[LoopNode(id="l", max_iterations=2, until=stop_at_three,
                            body=[BaseNode(id="s")], body_outputs=["s"])],
            outputs=["l"],
        )
        assert graph.model_dump()["nodes"][0]["until"] == "stop_at_three"


def test_empty_until_rejected():
    spec = _until_spec()
    spec["nodes"][0]["until"] = "   "
    with pytest.raises(GraphConfigurationError, match="empty"):
        load_graph_from_dict(spec)


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


# ── nested body events (S3 [BE]) ────────────────────────────────────────────

def test_loop_body_emits_scoped_node_events():
    """Body nodes report their own start/ok, tagged with the container + iteration.

    Without this the UI sees a loop as one opaque node for its whole duration.
    """
    _counter["n"] = 0
    spec = {
        "name": "loop_events",
        "nodes": [
            {"id": "l", "kind": "loop", "max_iterations": 3,
             "body": [{"id": "step", "kind": "function", "callable": f"{FN}._increment"}]},
        ],
        "outputs": ["l"],
    }
    graph = load_graph_from_dict(spec)
    seen: list[tuple[str, str, dict | None]] = []

    def on_event(node_id, status, scope=None):
        seen.append((node_id, status, scope))

    GraphExecutor(graph, provider=_EchoProvider(), log_traces=False).run(
        {}, node_event=on_event
    )

    top = [e for e in seen if e[2] is None]
    body = [e for e in seen if e[2] is not None]
    assert [(n, s) for n, s, _ in top] == [("l", "start"), ("l", "ok")]
    # One start + one ok per iteration, each carrying its iteration index.
    assert [(n, s) for n, s, _ in body] == [("step", "start"), ("step", "ok")] * 3
    assert [sc["iteration"] for _, _, sc in body] == [0, 0, 1, 1, 2, 2]
    assert all(sc["parent"] == "l" and sc["kind"] == "loop" for _, _, sc in body)
    assert all(sc["total"] == 3 for _, _, sc in body)


def test_map_body_events_cover_every_item():
    spec = {
        "name": "map_events",
        "inputs": [{"name": "nums", "type": "array"}],
        "nodes": [
            {"id": "m", "kind": "map", "over": "inputs.nums", "as": "item",
             "body": [{"id": "d", "kind": "function", "callable": f"{FN}._double"}]},
        ],
        "outputs": ["m"],
    }
    graph = load_graph_from_dict(spec)
    seen = []

    def on_event(node_id, status, scope=None):
        seen.append((node_id, status, scope))

    GraphExecutor(graph, provider=_EchoProvider(), log_traces=False).run(
        {"nums": [1, 2, 3]}, node_event=on_event
    )

    body = [e for e in seen if e[2] is not None]
    assert {sc["iteration"] for _, _, sc in body} == {0, 1, 2}
    assert all(sc["parent"] == "m" and sc["kind"] == "map" for _, _, sc in body)
    assert all(sc["total"] == 3 for _, _, sc in body)


def test_subgraph_body_events_are_scoped():
    spec = {
        "name": "sub_events",
        "nodes": [
            {"id": "s", "kind": "subgraph",
             "body": [{"id": "inner", "kind": "function", "callable": f"{FN}._increment"}]},
        ],
        "outputs": ["s"],
    }
    graph = load_graph_from_dict(spec)
    seen = []

    GraphExecutor(graph, provider=_EchoProvider(), log_traces=False).run(
        {}, node_event=lambda n, s, sc=None: seen.append((n, s, sc))
    )
    body = [e for e in seen if e[2] is not None]
    assert [(n, s) for n, s, _ in body] == [("inner", "start"), ("inner", "ok")]
    assert all(sc["parent"] == "s" and sc["kind"] == "subgraph" for _, _, sc in body)
    assert all("iteration" not in sc for _, _, sc in body)


def test_two_arg_callbacks_still_work():
    """The old `(node_id, status)` callback shape must keep working — CLI progress
    bars and existing embedders use it, and would break on a third argument."""
    _counter["n"] = 0
    spec = {
        "name": "legacy_cb",
        "nodes": [
            {"id": "l", "kind": "loop", "max_iterations": 2,
             "body": [{"id": "step", "kind": "function", "callable": f"{FN}._increment"}]},
        ],
        "outputs": ["l"],
    }
    graph = load_graph_from_dict(spec)
    seen = []
    GraphExecutor(graph, provider=_EchoProvider(), log_traces=False).run(
        {}, node_event=lambda node_id, status: seen.append((node_id, status))
    )
    # Two-arg callbacks see only top-level events — never a TypeError.
    assert seen == [("l", "start"), ("l", "ok")]
