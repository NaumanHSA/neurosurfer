"""Phase 1j — the `output` node: a graph states what it returns.

`graph.outputs` could only *select* whole node outputs by id, and it lived in a
side panel where a canvas could not show it. An `output` node says the same thing
where you can see it, and can compose an answer out of several nodes without a
`python` node whose only job is to join two strings.

These cover the four things that can go wrong: the precedence between the new
form and the old, the passthrough that must not stringify, the refusal to hand a
caller an unresolved placeholder as content, and the terminality that stops a
graph having two endings.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from neurosurfer.graph import GraphExecutor
from neurosurfer.graph.engine.loader import load_graph_from_dict
from neurosurfer.graph.engine.schema import Graph


class _EchoProvider:
    from neurosurfer.llm.capabilities import ProviderCapabilities

    model = "echo"
    capabilities = ProviderCapabilities(
        context_window=8192, max_output_tokens=2048,
        supports_thinking=False, supports_prompt_cache=False,
        supports_token_count=False, tool_call_style="openai",
    )


FN = "tests.engine.test_graph_output_node"


def _summary(**kwargs):
    return "a short summary"


def _rows(**kwargs):
    return 42


def _payload(**kwargs):
    """A dict, to prove the passthrough does not flatten it to text."""
    return {"rows": 3, "ok": True}


def _run(spec: dict, inputs: dict | None = None, validate: bool = True):
    """Run a graph. `validate=False` is for the tests that deliberately build a
    graph the validator refuses, to assert what the *runtime* does with it."""
    graph = load_graph_from_dict(spec)
    ex = GraphExecutor(graph, provider=_EchoProvider(), log_traces=False, validate=validate)
    return ex.run(inputs or {})


# ── what it returns ─────────────────────────────────────────────────────────────

def test_passthrough_keeps_the_dependency_type():
    """A dict returned through an output node is still a dict.

    Stringifying here would silently change a workflow's contract just because
    its answer passed through a node on the way out.
    """
    res = _run({
        "name": "wf",
        "nodes": [
            {"id": "make", "kind": "function", "callable": f"{FN}._payload"},
            {"id": "out", "kind": "output", "depends_on": ["make"]},
        ],
    })
    assert res.nodes["out"].error is None
    assert res.nodes["out"].raw_output == {"rows": 3, "ok": True}
    assert res.final == {"out": {"rows": 3, "ok": True}}


def test_value_composes_several_upstream_nodes():
    res = _run({
        "name": "wf",
        "nodes": [
            {"id": "summary", "kind": "function", "callable": f"{FN}._summary"},
            {"id": "rows", "kind": "function", "callable": f"{FN}._rows"},
            {"id": "out", "kind": "output",
             "depends_on": ["summary", "rows"],
             "value": "{summary} ({rows} rows)"},
        ],
    })
    # The output node is a collector first: everything wired into it is
    # something a caller asked to see. A composed `value` is one MORE answer —
    # keyed by the output node's own id — not a reason to discard the two nodes
    # that were connected. Before this, wiring a node in and adding a template
    # silently dropped that node's output.
    assert res.nodes["out"].raw_output == {
        "summary": "a short summary",
        "rows": 42,
        "out": "a short summary (42 rows)",
    }


def test_value_can_reference_a_graph_input():
    res = _run(
        {
            "name": "wf",
            "inputs": [{"name": "who", "type": "string"}],
            "nodes": [
                {"id": "s", "kind": "function", "callable": f"{FN}._summary"},
                {"id": "out", "kind": "output", "depends_on": ["s"],
                 "value": "for {who}: {s}"},
            ],
        },
        {"who": "nomi"},
    )
    assert res.nodes["out"].raw_output == {
        "s": "a short summary",
        "out": "for nomi: a short summary",
    }


def test_several_dependencies_without_a_value_are_keyed_by_node():
    """Picking one of them would be a guess, so it returns both."""
    res = _run({
        "name": "wf",
        "nodes": [
            {"id": "summary", "kind": "function", "callable": f"{FN}._summary"},
            {"id": "rows", "kind": "function", "callable": f"{FN}._rows"},
            {"id": "out", "kind": "output", "depends_on": ["summary", "rows"]},
        ],
    })
    assert res.nodes["out"].raw_output == {"summary": "a short summary", "rows": 42}


# ── precedence ──────────────────────────────────────────────────────────────────

def test_output_node_wins_over_graph_outputs():
    """The older `outputs:` list stays legal, and stays lower precedence."""
    res = _run({
        "name": "wf",
        "nodes": [
            {"id": "summary", "kind": "function", "callable": f"{FN}._summary"},
            {"id": "rows", "kind": "function", "callable": f"{FN}._rows"},
            {"id": "out", "kind": "output", "depends_on": ["summary"]},
        ],
        "outputs": ["rows"],
    })
    assert res.final == {"out": "a short summary"}


def test_graph_outputs_still_work_with_no_output_node():
    res = _run({
        "name": "wf",
        "nodes": [
            {"id": "summary", "kind": "function", "callable": f"{FN}._summary"},
            {"id": "rows", "kind": "function", "callable": f"{FN}._rows"},
        ],
        "outputs": ["rows"],
    })
    assert res.final == {"rows": 42}


def test_falls_back_to_the_last_node_when_nothing_is_declared():
    res = _run({
        "name": "wf",
        "nodes": [
            {"id": "summary", "kind": "function", "callable": f"{FN}._summary"},
            {"id": "rows", "kind": "function", "callable": f"{FN}._rows",
             "depends_on": ["summary"]},
        ],
    })
    assert res.final == {"rows": 42}


def test_a_skipped_output_node_does_not_claim_the_result():
    """One arm of a branch returns; the arm that did not run must not.

    Without this a router taking the left branch would report an empty entry for
    the right one, and a caller reading `final` would see a key it never produced.
    """
    res = _run(
        {
            "name": "wf",
            "inputs": [{"name": "go", "type": "string"}],
            "nodes": [
                {"id": "left", "kind": "function", "callable": f"{FN}._summary",
                 "when": "inputs.go == 'left'"},
                {"id": "right", "kind": "function", "callable": f"{FN}._rows",
                 "when": "inputs.go == 'right'"},
                {"id": "out_left", "kind": "output", "depends_on": ["left"],
                 "when": "inputs.go == 'left'"},
                {"id": "out_right", "kind": "output", "depends_on": ["right"],
                 "when": "inputs.go == 'right'"},
            ],
        },
        {"go": "left"},
    )
    assert res.final == {"out_left": "a short summary"}


# ── refusals ────────────────────────────────────────────────────────────────────

def test_an_unresolved_placeholder_is_an_error_not_literal_text():
    """The answer a caller receives must never be a leftover `{name}`."""
    res = _run({
        "name": "wf",
        "nodes": [
            {"id": "s", "kind": "function", "callable": f"{FN}._summary"},
            {"id": "out", "kind": "output", "depends_on": ["s"],
             "value": "{nothing_called_this}"},
        ],
    }, validate=False)
    assert res.nodes["out"].error is not None
    assert "resolved to nothing" in res.nodes["out"].error


def test_no_value_and_no_dependency_is_an_error():
    res = _run({
        "name": "wf",
        "nodes": [{"id": "out", "kind": "output"}],
    }, validate=False)
    assert res.nodes["out"].error is not None
    assert "nothing to pass" in res.nodes["out"].error


# ── terminality ─────────────────────────────────────────────────────────────────

def test_nothing_may_depend_on_an_output_node():
    with pytest.raises(ValidationError, match="nothing runs after"):
        Graph(name="wf", nodes=[
            {"id": "out", "kind": "output"},
            {"id": "after", "kind": "base", "depends_on": ["out"]},
        ])


def test_an_output_node_is_not_an_error_handler():
    with pytest.raises(ValidationError, match="not an error handler"):
        Graph(name="wf", nodes=[
            {"id": "out", "kind": "output"},
            {"id": "risky", "kind": "base", "on_error": "out"},
        ])


def test_a_value_with_no_dependencies_is_still_just_the_value():
    """Nothing wired in — the template is the whole answer, not a one-key dict."""
    res = _run({
        "name": "wf",
        "inputs": [{"name": "who", "type": "string"}],
        "nodes": [{"id": "out", "kind": "output", "value": "hello {who}"}],
    }, {"who": "nomi"})
    assert res.nodes["out"].raw_output == "hello nomi"


def test_collecting_is_what_the_run_panel_lists():
    """Three nodes wired into one output node produce three named results.

    This is the shape the studio's Run panel lists: one row per connected node,
    each showing that node's own output. Keyed by node id so a row can name
    where its value came from.
    """
    res = _run({
        "name": "wf",
        "nodes": [
            {"id": "summary", "kind": "function", "callable": f"{FN}._summary"},
            {"id": "rows", "kind": "function", "callable": f"{FN}._rows"},
            {"id": "out", "kind": "output", "depends_on": ["summary", "rows"]},
        ],
    })
    assert res.nodes["out"].raw_output == {
        "summary": "a short summary",
        "rows": 42,
    }
