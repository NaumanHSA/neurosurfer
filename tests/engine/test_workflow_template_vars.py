"""Template-var resolution in the pre-registration validation gate.

A node's purpose/goal/expected_result are rendered with ``text.format(**scope)``.
An unresolvable name makes ``format`` raise, and the executor's fallback is to log
a WARNING and send the *unrendered* text — so the run succeeds while the model is
reading the five literal characters ``{summary}``. These tests pin the static
reconstruction of that scope to what the executor actually builds.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from neurosurfer.graph import Graph, GraphNode
from neurosurfer.graph.engine.executor import GraphExecutor
from neurosurfer.graph.workflow.package import WorkflowPackage
from neurosurfer.graph.workflow.schema import WorkflowManifest
from neurosurfer.graph.workflow.validate import validate_package

from ..fakes import ScriptedProvider


def _pkg(nodes: list[GraphNode], tmp_path: Path, *, inputs=None, outputs=None) -> WorkflowPackage:
    graph = Graph(
        name="t",
        inputs=[{"name": n, "type": "string", "required": True} for n in (inputs or [])],
        nodes=nodes,
        outputs=outputs or [nodes[-1].id],
    )
    return WorkflowPackage(manifest=WorkflowManifest(name="t"), graph=graph, path=tmp_path)


def _issues(report, node_id=None):
    """Template issues only — other checks (orphans, wiring floor) are not the subject."""
    def keep(items):
        return [i for i in items
                if i.kind.startswith("template") and (node_id is None or i.node_id == node_id)]
    return keep(report.errors), keep(report.warnings)


def _validate(nodes, tmp_path, *, inputs=None, outputs=None, node_id=None):
    report = validate_package(_pkg(nodes, tmp_path, inputs=inputs, outputs=outputs))
    return _issues(report, node_id)


# ── the runtime this check models ───────────────────────────────────────────────

class _RecordingProvider(ScriptedProvider):
    """ScriptedProvider that keeps every system prompt it was handed."""

    def __init__(self, turns):
        super().__init__(turns)
        self.systems: list[str] = []
        self.users: list[str] = []

    async def stream(self, messages, system, tools, config):
        self.systems.append(system)
        self.users.append("\n".join(str(getattr(m, "content", m)) for m in messages))
        async for ev in super().stream(messages, system, tools, config):
            yield ev


def test_runtime_leaves_undeclared_reference_literal_and_validation_says_so(tmp_path):
    """The bug, from both ends: the executor silently ships `{a}`, validation errors.

    `b` names another node without depending on it, so `dependency_results` — the
    only route by which a node id enters the scope — never carries it.
    """
    nodes = [
        GraphNode(id="a", kind="base", goal="write something"),
        GraphNode(id="b", kind="base", goal="uses {a}"),
    ]
    graph = _pkg(nodes, tmp_path, outputs=["a", "b"]).graph
    provider = _RecordingProvider([("out", "")] * 4)
    GraphExecutor(graph=graph, provider=provider).run({})

    assert any("uses {a}" in s for s in provider.systems), "executor rendered it after all"

    errors, _ = _validate(nodes, tmp_path, outputs=["a", "b"], node_id="b")
    assert [e.subject for e in errors] == ["a"]
    assert "add 'a' to depends_on" in (errors[0].suggestion or "")


def test_declared_dependency_renders_at_runtime_and_validates(tmp_path):
    """The mirror: with the edge declared, the same template resolves."""
    nodes = [
        GraphNode(id="a", kind="base", goal="write something", writes="alpha"),
        GraphNode(id="b", kind="base", depends_on=["a"], goal="uses {a}, {alpha} and {topic}"),
    ]
    graph = _pkg(nodes, tmp_path, inputs=["topic"]).graph
    provider = _RecordingProvider([("out", "")] * 4)
    GraphExecutor(graph=graph, provider=provider).run({"topic": "cats"})

    assert any("uses out, out and cats" in s for s in provider.systems)
    assert _validate(nodes, tmp_path, inputs=["topic"]) == ([], [])


# ── the three sources of scope ──────────────────────────────────────────────────

def test_graph_input_resolves(tmp_path):
    nodes = [GraphNode(id="a", kind="base", goal="summarise {article}")]
    assert _validate(nodes, tmp_path, inputs=["article"]) == ([], [])


def test_undeclared_name_is_an_error_listing_what_is_in_scope(tmp_path):
    nodes = [GraphNode(id="a", kind="base", goal="summarise {article}")]
    errors, _ = _validate(nodes, tmp_path, inputs=["topic"])
    assert len(errors) == 1
    assert errors[0].kind == "template_var"
    assert errors[0].subject == "article"
    assert "'topic'" in errors[0].message  # tells you what you could have used


def test_near_miss_suggests_the_name_in_scope(tmp_path):
    nodes = [GraphNode(id="a", kind="base", goal="summarise {artcle}")]
    errors, _ = _validate(nodes, tmp_path, inputs=["article"])
    assert "{article}" in (errors[0].suggestion or "")


def test_writes_of_a_transitive_ancestor_resolves(tmp_path):
    """`state.vars` is guaranteed for anything on this node's dependency path."""
    nodes = [
        GraphNode(id="a", kind="base", writes="summary", goal="go"),
        GraphNode(id="b", kind="base", depends_on=["a"], goal="go"),
        GraphNode(id="c", kind="base", depends_on=["b"], goal="title for {summary}"),
    ]
    assert _validate(nodes, tmp_path) == ([], [])


def test_writes_of_a_non_ancestor_is_a_warning_not_an_error(tmp_path):
    """It really may resolve — `vars` is shared and that node may have run first —
    so this is a warning about certainty, not a broken reference."""
    nodes = [
        GraphNode(id="a", kind="base", writes="summary", goal="go"),
        GraphNode(id="b", kind="base", goal="title for {summary}"),
    ]
    errors, warnings = _validate(nodes, tmp_path, outputs=["a", "b"], node_id="b")
    assert not errors
    assert len(warnings) == 1
    assert warnings[0].subject == "summary"
    assert "does not depend on" in warnings[0].message


def test_the_non_ancestor_case_really_does_resolve_at_runtime(tmp_path):
    """Corroborates the warning-not-error call above against the executor."""
    nodes = [
        GraphNode(id="a", kind="base", writes="summary", goal="go"),
        GraphNode(id="b", kind="base", goal="title for {summary}"),
    ]
    provider = _RecordingProvider([("out", "")] * 4)
    GraphExecutor(graph=_pkg(nodes, tmp_path, outputs=["a", "b"]).graph, provider=provider).run({})
    assert any("title for out" in s for s in provider.systems)


def test_attribute_and_index_access_check_the_root_name(tmp_path):
    ok = [GraphNode(id="a", kind="base", goal="{doc.title} then {doc[0]}")]
    assert _validate(ok, tmp_path, inputs=["doc"]) == ([], [])

    bad = [GraphNode(id="a", kind="base", goal="{doc.title}")]
    errors, _ = _validate(bad, tmp_path, inputs=["other"])
    assert errors[0].subject == "doc"


# ── containers: what a body can actually see ────────────────────────────────────

def _loop(body, **kw):
    return GraphNode(id="lp", kind="loop", max_iterations=2, body=body, **kw)


def test_loop_body_sees_index_item_feedback_and_the_parent_scope(tmp_path):
    nodes = [
        GraphNode(id="seed", kind="base", writes="slogan", goal="go"),
        _loop(
            depends_on=["seed"],
            body=[GraphNode(
                id="rewrite", kind="base",
                goal="pass {index} of {product}: fix {slogan} given {feedback} and {item}",
            )],
        ),
    ]
    assert _validate(nodes, tmp_path, inputs=["product"]) == ([], [])


def test_loop_body_honours_a_renamed_item_var(tmp_path):
    nodes = [_loop(body=[GraphNode(id="w", kind="base", goal="{draft}")], **{"as": "draft"})]
    assert _validate(nodes, tmp_path) == ([], [])

    nodes = [_loop(body=[GraphNode(id="w", kind="base", goal="{item}")], **{"as": "draft"})]
    errors, _ = _validate(nodes, tmp_path)
    assert errors[0].subject == "item"


def test_iteration_and_acc_are_expression_only_not_template_visible(tmp_path):
    """A subtle one: the loop puts `iteration`/`acc` on the child *scope*, which
    expressions read and `interp_scope` does not."""
    nodes = [_loop(body=[GraphNode(id="w", kind="base", goal="{iteration} {acc}")])]
    errors, _ = _validate(nodes, tmp_path)
    assert sorted(e.subject for e in errors) == ["acc", "iteration"]


def test_map_body_binds_item_and_index_but_not_feedback(tmp_path):
    nodes = [
        GraphNode(id="plan", kind="function", writes="sections",
                  callable="json:loads", goal="go"),
        GraphNode(
            id="mp", kind="map", depends_on=["plan"], over="vars.sections", **{"as": "section"},
            body=[GraphNode(id="w", kind="base", goal="{section} #{index} for {topic}")],
        ),
    ]
    assert _validate(nodes, tmp_path, inputs=["topic"]) == ([], [])

    nodes[1].body[0].goal = "{section} {feedback}"
    errors, _ = _validate(nodes, tmp_path, inputs=["topic"])
    assert [e.subject for e in errors] == ["feedback"]


def test_subgraph_body_inherits_parent_inputs_and_ancestor_writes(tmp_path):
    nodes = [
        GraphNode(id="a", kind="base", writes="brief", goal="go"),
        GraphNode(
            id="pack", kind="subgraph", depends_on=["a"],
            body=[GraphNode(id="sum", kind="base", goal="compress {brief} for {audience}")],
        ),
    ]
    assert _validate(nodes, tmp_path, inputs=["audience"]) == ([], [])


def test_body_node_can_reference_its_own_body_dependency(tmp_path):
    nodes = [_loop(body=[
        GraphNode(id="draft", kind="base", goal="write"),
        GraphNode(id="critique", kind="base", depends_on=["draft"], goal="review {draft}"),
    ])]
    assert _validate(nodes, tmp_path) == ([], [])


def test_body_node_cannot_reference_an_outer_node_id(tmp_path):
    """`dependency_results` is built per level, so an outer id is not in body scope."""
    nodes = [
        GraphNode(id="research", kind="base", goal="go"),
        _loop(depends_on=["research"], body=[GraphNode(id="w", kind="base", goal="{research}")]),
    ]
    errors, _ = _validate(nodes, tmp_path)
    assert [e.subject for e in errors] == ["research"]


# ── routers interpolate a narrower scope ────────────────────────────────────────

def test_routes_router_resolves_graph_inputs(tmp_path):
    nodes = [
        GraphNode(id="r", kind="router", routes={"a": "x", "b": "x"},
                  purpose="route this ticket: {ticket}"),
        GraphNode(id="x", kind="base", depends_on=["r"], goal="go"),
    ]
    assert _validate(nodes, tmp_path, inputs=["ticket"], outputs=["x"]) == ([], [])


def test_routes_router_cannot_reach_upstream_output(tmp_path):
    """`_route_by_classification` formats with `state.inputs` alone."""
    nodes = [
        GraphNode(id="research", kind="base", writes="findings", goal="go"),
        GraphNode(id="r", kind="router", depends_on=["research"], routes={"a": "x", "b": "x"},
                  purpose="given {research}, route it"),
        GraphNode(id="x", kind="base", depends_on=["r"], goal="go"),
    ]
    errors, _ = _validate(nodes, tmp_path, inputs=["topic"], outputs=["x"], node_id="r")
    assert len(errors) == 1
    assert errors[0].subject == "research"
    assert "graph inputs only" in errors[0].message


def test_routes_router_reference_to_a_non_ancestor_write_is_still_an_error(tmp_path):
    """A router reads no `vars` at all, so the "might resolve if that branch ran"
    softening that applies to base nodes must not apply here."""
    nodes = [
        GraphNode(id="other", kind="base", writes="notes", goal="go"),
        GraphNode(id="r", kind="router", routes={"a": "x", "b": "x"},
                  purpose="route using {notes}"),
        GraphNode(id="x", kind="base", depends_on=["r"], goal="go"),
    ]
    errors, warnings = _validate(nodes, tmp_path, outputs=["other", "x"], node_id="r")
    assert not warnings
    assert len(errors) == 1
    assert "graph inputs only" in errors[0].message


def test_routes_router_literal_stays_literal_at_runtime(tmp_path):
    """Corroboration: the router prompt really does carry the braces through."""
    nodes = [
        GraphNode(id="research", kind="base", goal="go"),
        GraphNode(id="r", kind="router", depends_on=["research"],
                  routes={"only": "x"}, purpose="given {research}, route it"),
        GraphNode(id="x", kind="base", depends_on=["r"], goal="go"),
    ]
    provider = _RecordingProvider([("go", ""), ("only", ""), ("done", "")])
    GraphExecutor(graph=_pkg(nodes, tmp_path, outputs=["x"]).graph, provider=provider).run({})
    # The classifier's instruction is the user turn, and it still carries the braces.
    assert any("given {research}, route it" in u for u in provider.users)


def test_cases_router_does_not_interpolate(tmp_path):
    nodes = [
        GraphNode(id="r", kind="router", default="x",
                  cases=[{"when": "inputs.n > 1", "to": "x"}], purpose="pick using {n}"),
        GraphNode(id="x", kind="base", depends_on=["r"], goal="go"),
    ]
    errors, warnings = _validate(nodes, tmp_path, inputs=["n"], outputs=["x"], node_id="r")
    assert not errors
    assert "never interpolated" in warnings[0].message


# ── kinds that never interpolate ────────────────────────────────────────────────

@pytest.mark.parametrize("kind,extra", [
    ("function", {"callable": "json:loads"}),
    ("python", {"callable": "json:loads"}),
    ("tool", {"tools": ["read_file"]}),
    ("input", {}),
])
def test_placeholder_in_a_non_interpolating_kind_is_a_warning(tmp_path, kind, extra):
    nodes = [GraphNode(id="a", kind=kind, purpose="handle {topic}", **extra)]
    errors, warnings = _validate(nodes, tmp_path, inputs=["topic"])
    assert not errors, "a name that resolves elsewhere must not become a hard error here"
    assert len(warnings) == 1
    assert warnings[0].subject == "topic"
    assert kind in warnings[0].message


def test_non_interpolating_kinds_tolerate_literal_braces(tmp_path):
    """Nothing formats these, so JSON in a description is simply not a problem."""
    nodes = [GraphNode(id="a", kind="function", callable="json:loads",
                       purpose='returns {"ok": true}')]
    assert _validate(nodes, tmp_path) == ([], [])


# ── braces that were never meant to be placeholders ─────────────────────────────

def test_literal_json_is_a_warning_not_an_error(tmp_path):
    nodes = [GraphNode(id="a", kind="base", goal='reply with {"title": "x"}')]
    errors, warnings = _validate(nodes, tmp_path)
    assert not errors
    assert warnings[0].kind == "template_braces"
    assert "{{" in (warnings[0].suggestion or "")


def test_literal_braces_do_not_disturb_the_vars_beside_them(tmp_path):
    """Rendering is per placeholder, so literal JSON costs only itself."""
    nodes = [GraphNode(id="a", kind="base", goal='summarise {topic} as {"t": "x"}')]
    errors, warnings = _validate(nodes, tmp_path, inputs=["topic"])
    assert not errors
    assert warnings[0].kind == "template_braces"

    provider = _RecordingProvider([("out", "")] * 2)
    GraphExecutor(graph=_pkg(nodes, tmp_path, inputs=["topic"]).graph,
                  provider=provider).run({"topic": "cats"})
    assert any('summarise cats as {"t": "x"}' in s for s in provider.systems)


def test_prose_after_a_colon_reads_as_a_literal_brace(tmp_path):
    """`{status: ok}` parses as field `status` with spec ' ok' — but that is not a
    format spec, so it is a literal brace, not a missing variable."""
    nodes = [GraphNode(id="a", kind="base", goal="emit {status: ok}")]
    errors, warnings = _validate(nodes, tmp_path)
    assert not errors
    assert warnings[0].kind == "template_braces"


def test_a_real_format_spec_is_still_a_variable(tmp_path):
    nodes = [GraphNode(id="a", kind="base", goal="{score:.2f} for {topic}")]
    errors, _ = _validate(nodes, tmp_path, inputs=["topic"])
    assert [e.subject for e in errors] == ["score"]


def test_positional_placeholders_are_flagged_as_literal(tmp_path):
    nodes = [GraphNode(id="a", kind="base", goal="an empty object is {} or {0}")]
    errors, warnings = _validate(nodes, tmp_path)
    assert not errors
    assert warnings[0].kind == "template_braces"


def test_unbalanced_brace_is_reported(tmp_path):
    nodes = [GraphNode(id="a", kind="base", goal="summarise {topic")]
    errors, warnings = _validate(nodes, tmp_path, inputs=["topic"])
    assert not errors
    assert "unbalanced" in warnings[0].message


def test_escaped_braces_are_silent(tmp_path):
    nodes = [GraphNode(id="a", kind="base", goal='reply with {{"title": "x"}} about {topic}')]
    assert _validate(nodes, tmp_path, inputs=["topic"]) == ([], [])


def test_every_template_field_is_checked(tmp_path):
    nodes = [GraphNode(id="a", kind="base", purpose="{p}", goal="{g}", expected_result="{e}")]
    errors, _ = _validate(nodes, tmp_path)
    assert sorted(e.subject for e in errors) == ["e", "g", "p"]


def test_a_clean_graph_reports_nothing(tmp_path):
    nodes = [
        GraphNode(id="a", kind="base", writes="summary", goal="summarise {article}"),
        GraphNode(id="b", kind="base", depends_on=["a"], goal="title for {summary}"),
    ]
    assert _validate(nodes, tmp_path, inputs=["article"]) == ([], [])
