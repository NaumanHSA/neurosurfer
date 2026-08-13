"""Binding a tool's arguments to an upstream step's declared output.

The shape under test throughout:

    agent (structured, output_schema: {sql}) → query (tool, tool_args:{query:"{agent[sql]}"})

The agent chooses the arguments; the tool node still always runs.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from neurosurfer.graph.engine.schema import Graph, GraphNode
from neurosurfer.graph.engine.templates import render_template
from neurosurfer.graph.workflow.package import WorkflowPackage
from neurosurfer.graph.workflow.schema import WorkflowManifest
from neurosurfer.graph.workflow.validation import validate_package

SHAPE = {"type": "object", "properties": {"sql": {"type": "string"}}, "required": ["sql"]}


def pkg(nodes, tmp_path: Path) -> WorkflowPackage:
    graph = Graph(name="t", nodes=list(nodes), outputs=[])
    return WorkflowPackage(manifest=WorkflowManifest(name="t"), graph=graph, path=tmp_path)


def issues(nodes, tmp_path, kind: str) -> list:
    report = validate_package(pkg(nodes, tmp_path))
    return [i for i in report.issues if i.kind == kind]


def agent(schema=SHAPE) -> GraphNode:
    return GraphNode(id="agent", kind="base", instructions="Write SQL.",
                     mode="structured", output_schema=schema)


# ── the renderer this all rests on ──────────────────────────────────────────


class TestBothSpellingsReachAField:
    """`{x[k]}` and `{x.k}` must mean the same thing on both output kinds.

    They did not: the subscript form worked only on a dict and the attribute form
    only on a pydantic model — which is what a node with an `output_schema`
    returns. So the correct syntax depended on whether an output shape was set,
    a fact invisible from the canvas, and guessing wrong left the placeholder in
    the string and passed it to the tool verbatim.
    """

    class Shaped:  # stand-in for what `output_schema` produces
        pass

    @pytest.mark.parametrize("template", ["{agent[sql]}", "{agent.sql}"])
    def test_against_a_dict(self, template):
        text, unresolved = render_template(template, {"agent": {"sql": "SELECT 1"}})
        assert text == "SELECT 1" and not unresolved

    @pytest.mark.parametrize("template", ["{agent[sql]}", "{agent.sql}"])
    def test_against_a_model(self, template):
        from pydantic import BaseModel

        class Out(BaseModel):
            sql: str

        text, unresolved = render_template(template, {"agent": Out(sql="SELECT 2")})
        assert text == "SELECT 2" and not unresolved

    def test_the_whole_value_is_unchanged(self):
        """Wrapping must not alter what `{agent}` on its own renders."""
        text, _ = render_template("{agent}", {"agent": {"sql": "S"}})
        assert text == "{'sql': 'S'}"

    def test_a_miss_is_still_left_as_written(self):
        text, unresolved = render_template("{agent[nope]}", {"agent": {"sql": "S"}})
        assert text == "{agent[nope]}" and unresolved == ["agent[nope]"]


# ── R1 · reaching into something with no shape ──────────────────────────────


def test_reaching_into_an_unshaped_step_is_an_error(tmp_path):
    found = issues([
        GraphNode(id="agent", kind="base", instructions="Write SQL."),   # no schema
        GraphNode(id="q", kind="tool", tools=["read_file"],
                  tool_args={"path": "{agent[sql]}"}, depends_on=["agent"]),
    ], tmp_path, "binding.source_has_no_shape")
    assert len(found) == 1
    assert found[0].node_id == "q"


def test_reaching_into_a_shaped_step_is_fine(tmp_path):
    found = issues([
        agent(),
        GraphNode(id="q", kind="tool", tools=["read_file"],
                  tool_args={"path": "{agent[sql]}"}, depends_on=["agent"]),
    ], tmp_path, "binding.source_has_no_shape")
    assert not found


def test_taking_the_whole_answer_needs_no_shape(tmp_path):
    """`{agent}` is prose into a string parameter — a real and common binding."""
    found = issues([
        GraphNode(id="agent", kind="base", instructions="Write it."),
        GraphNode(id="q", kind="tool", tools=["read_file"],
                  tool_args={"path": "{agent}"}, depends_on=["agent"]),
    ], tmp_path, "binding.source_has_no_shape")
    assert not found


# ── R2 · reaching for a field that is not there ─────────────────────────────


def test_a_required_argument_bound_to_a_missing_field_is_an_error(tmp_path):
    found = issues([
        agent(),
        GraphNode(id="q", kind="tool", tools=["read_file"],
                  tool_args={"path": "{agent[sqll]}"}, depends_on=["agent"]),
    ], tmp_path, "binding.unknown_field")
    assert len(found) == 1
    assert found[0].severity == "error"
    # The fields that *do* exist are the part that saves the time.
    assert "sql" in (found[0].suggestion or "")


def test_an_optional_argument_bound_to_a_missing_field_only_warns(tmp_path):
    """It will be left out — degraded, not impossible."""
    found = issues([
        agent(),
        GraphNode(id="q", kind="tool", tools=["read_file"],
                  tool_args={"path": "{agent[sql]}", "offset": "{agent[nope]}"},
                  depends_on=["agent"]),
    ], tmp_path, "binding.unknown_field")
    optional = [i for i in found if i.subject == "offset"]
    assert len(optional) == 1
    assert optional[0].severity == "warning"


def test_a_field_that_exists_is_quiet(tmp_path):
    found = issues([
        agent(),
        GraphNode(id="q", kind="tool", tools=["read_file"],
                  tool_args={"path": "{agent[sql]}"}, depends_on=["agent"]),
    ], tmp_path, "binding.unknown_field")
    assert not found


# ── what the agent was never meant to produce ───────────────────────────────


class TestArgumentsDecidedElsewhere:
    """A credential or a literal is not the agent's job, and must never be
    reported against the agent's output."""

    def test_a_literal_is_never_checked(self, tmp_path):
        found = issues([
            agent(),
            GraphNode(id="q", kind="tool", tools=["read_file"],
                      tool_args={"path": "/etc/hosts"}, depends_on=["agent"]),
        ], tmp_path, "binding.unknown_field")
        assert not found

    def test_a_stored_secret_is_never_checked(self, tmp_path):
        found = issues([
            agent(),
            GraphNode(id="q", kind="tool", tools=["read_file"], secrets=["DB_URL"],
                      tool_args={"path": "${DB_URL}"}, depends_on=["agent"]),
        ], tmp_path, "binding.unknown_field")
        assert not found


# ── R3 · an argument no tool accepts ────────────────────────────────────────


def test_an_argument_no_tool_takes_is_a_warning(tmp_path):
    found = issues([
        GraphNode(id="q", kind="tool", tools=["read_file"],
                  tool_args={"path": "/tmp/x", "definitely_not_a_param": 1}),
    ], tmp_path, "binding.unused_argument")
    assert [i.subject for i in found] == ["definitely_not_a_param"]
    assert found[0].severity == "warning"


def test_an_argument_a_sibling_tool_takes_is_fine(tmp_path):
    """On a step holding several tools, an argument only needs *one* taker.

    `bind_pool` applies each bound value only to the tools whose schema names it,
    so an argument meant for the second tool is not a mistake on the first.
    """
    found = issues([
        GraphNode(id="a", kind="react", instructions="Do it.",
                  tools=["read_file", "write_file"],
                  tool_args={"path": "/tmp/x"}),
    ], tmp_path, "binding.unused_argument")
    assert not found


# ── a shape and tools cannot both happen ────────────────────────────────────


def test_a_shape_plus_tools_is_an_error(tmp_path):
    """The engine takes the shape and never reaches the tool loop."""
    found = issues([
        GraphNode(id="a", kind="base", instructions="Answer.", tools=["read_file"],
                  mode="structured", output_schema=SHAPE),
    ], tmp_path, "agent.shape_disables_tools")
    assert len(found) == 1 and found[0].severity == "error"


def test_a_shape_without_tools_is_fine(tmp_path):
    assert not issues([
        GraphNode(id="a", kind="base", instructions="Answer.",
                  mode="structured", output_schema=SHAPE),
    ], tmp_path, "agent.shape_disables_tools")


def test_tools_without_a_shape_are_fine(tmp_path):
    assert not issues([
        GraphNode(id="a", kind="base", instructions="Answer.", tools=["read_file"]),
    ], tmp_path, "agent.shape_disables_tools")


def test_a_react_node_is_offered_a_shape_only_now_that_it_honours_one():
    """The field is back, and the rule it was withdrawn under still holds.

    `output_schema` was removed from this spec because it was inert — written
    into the YAML, shown in the panel, reviewed, and read by nothing, so "return
    an object" silently returned prose. `run_react_node` takes one now (the loop
    runs, then a structured call shapes its answer), so offering it is honest
    again. The invariant this test really guards is unchanged: **a spec offers a
    field only when the runner reads it.**
    """
    import inspect

    from neurosurfer.graph.engine.kinds import node_kind_spec
    from neurosurfer.graph.engine.node_runner import run_react_node

    react = node_kind_spec("react")
    assert react.field("output_schema") is not None
    assert "output_schema" in inspect.signature(run_react_node).parameters

    # `mode` stays withdrawn — nothing reads it on a react node.
    assert react.field("mode") is None

    base = node_kind_spec("base")
    assert base.field("output_schema") and base.field("mode")



def test_an_unaccepted_argument_gets_one_message_not_two(tmp_path):
    """One mistake, one line.

    An argument the tool does not take was reported twice: once as unused, and
    once for where its value would have come from — a value nothing will ever
    read. The second is noise, and noise in a warnings list is what stops people
    reading it.
    """
    nodes = [
        agent(),
        GraphNode(id="q", kind="tool", tools=["read_file"],
                  tool_args={"path": "{agent[sql]}", "not_a_param": "{agent[nope]}"},
                  depends_on=["agent"]),
    ]
    assert not issues(nodes, tmp_path, "binding.unknown_field")
    unused = issues(nodes, tmp_path, "binding.unused_argument")
    assert [i.subject for i in unused] == ["not_a_param"]
