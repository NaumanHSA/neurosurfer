"""Rules about the workflow as a whole.

Everything that cannot be decided by looking at one node: whether an edge points
at a node that exists, whether anything consumes a node's output, whether the
graph has a way in and a way out.
"""

from __future__ import annotations

from neurosurfer.graph.engine.kinds import node_kind_spec

from .models import Severity, ValidationIssue, ValidationReport
from .registry import graph_rule, node_rule


def _check_edges(node, node_ids: set[str], report: ValidationReport) -> None:
    for dep in node.depends_on or []:
        if dep not in node_ids:
            report.add(ValidationIssue(
        severity=Severity.ERROR,
                kind="dag",
                node_id=node.id,
                message="This step waits for a step that is not in the workflow.",
            suggestion="Connect it to a step that exists, or remove the connection.",
            detail=f"depends_on '{dep}'",
            ))




# ── rules ────────────────────────────────────────────────────────────────────


@node_rule(kinds=("*",), severity=Severity.ERROR, bodies=False)
def edges_point_at_real_nodes(node, ctx, report) -> None:
    """`depends_on` must name nodes that exist.

    Top level only: a body's edges are body-scoped and the loader already
    checked them against their own siblings.
    """
    _check_edges(node, ctx.node_ids, report)


@graph_rule(severity=Severity.ERROR)
def declared_outputs_exist(graph, ctx, report) -> None:
    """Every name in `outputs` is a node in the graph."""
    for out in graph.outputs or []:
        if out not in ctx.node_ids:
            report.add(ValidationIssue(
                severity=Severity.ERROR,
                kind="dag",
                message="The workflow is set to return a step that is not in it.",
                suggestion="Point it at a step that exists.",
                detail=f"outputs names '{out}'",
            ))


@graph_rule(severity=Severity.WARNING)
def nothing_consumes_this_node(graph, ctx, report) -> None:
    """A node whose output nothing reads.

    Dead weight that inflates cost, and a classic over-design smell — a
    "validation" or "formatting" step bolted on and wired to nothing. Only the
    top level is checked; body nodes are scoped separately.
    """
    if len(graph.nodes) <= 1:
        return
    for n in graph.nodes:
        # A terminal node is *meant* to have nothing after it — it is the value
        # the workflow returns, and the engine refuses anything that depends on
        # one. Reporting that as an orphan told authors their output node was a
        # mistake for being an output node.
        if _is_terminal(n):
            continue
        if n.id in ctx.referenced_ids:
            continue
        if getattr(n, "writes", None) and n.writes in ctx.referenced_vars:
            continue
        report.add(ValidationIssue(
            severity=Severity.WARNING,
            kind="structure",
            node_id=n.id,
            message=(
                "Nothing uses what this step produces, so it costs time and "
                "money without affecting the answer."
            ),
            suggestion="Connect it to the step that needs it, or remove it.",
            detail="not in `outputs`, and no node lists it in `depends_on`",
        ))


@graph_rule(severity=Severity.WARNING)
def declared_inputs_are_read_by_something(graph, ctx, report) -> None:
    """A graph input nothing reads. **The rule that makes the narrowing safe.**

    A node used to be printed every graph input under an `Inputs:` heading, so
    an input was consumed by existing: whether or not any step named it, a model
    somewhere saw it. That block is gone — a node's turn carries what its task
    text names plus its declared dependencies — which means an input nothing
    names now genuinely does nothing.

    That is the right contract and the wrong thing to discover at run time. The
    symptom would be a workflow that accepts a parameter, runs green, and
    answers as though it had never been passed, which is precisely the class of
    silent plausible success the rest of this package exists to catch.

    Every way a value can be read counts, not just prompt placeholders: an
    `over` expression, a `when` predicate, `tool_args`, an output `value`, and
    bodies as well as the top level. A rule that only looked at `instructions`
    would report a perfectly good `map` as ignoring its collection.
    """
    declared = [
        getattr(i, "name", None) or (i.get("name") if isinstance(i, dict) else None)
        for i in (graph.inputs or [])
    ]
    declared = [n for n in declared if n]
    if not declared:
        return

    read = _names_read_anywhere(ctx)
    for name in declared:
        if name in read:
            continue
        report.add(ValidationIssue(
            severity=Severity.WARNING,
            kind="structure",
            subject=name,
            message=(
                f"The workflow asks for '{name}' but no step uses it, so the "
                f"value a caller passes is ignored."
            ),
            suggestion=(
                f"Name it in a step's instructions as {{{name}}}, or drop it "
                f"from the workflow's inputs."
            ),
            detail=f"graph input {name!r} appears in no template, expression or argument",
        ))


def _names_read_anywhere(ctx) -> set[str]:
    """Every name any node reads, by any of the routes a value can travel."""
    from .templates import _TEMPLATE_FIELDS, _parse_placeholders

    def roots(text) -> set[str]:
        if not text or not isinstance(text, str) or "{" not in text:
            return set()
        return {p.root for p in (_parse_placeholders(text) or [])}

    found: set[str] = set()
    for node in ctx.all_nodes:
        for field_name in _TEMPLATE_FIELDS:
            found |= roots(getattr(node, field_name, None))
        # Expressions name things directly rather than through `{}`.
        for expr in (getattr(node, "over", None), getattr(node, "when", None),
                     getattr(node, "break_when", None)):
            found.update(_expression_names(expr))
        for case in getattr(node, "cases", None) or []:
            found.update(_expression_names(getattr(case, "when", None)))
        # `tool_args` / `tool_settings` values are templates too.
        for holder in (getattr(node, "tool_args", None) or {},
                       getattr(node, "tool_settings", None) or {}):
            for value in _flatten(holder):
                found |= roots(value)
        # A tool node is handed the inputs dict as kwargs, so a parameter name
        # matching an input is a read even with no template anywhere.
        if getattr(node, "kind", None) == "tool":
            found.update(getattr(node, "tool_args", None) or {})
    return found


def _expression_names(expr) -> set[str]:
    """Bare and `inputs.`-qualified names in a sandboxed expression."""
    import re

    if not expr or not isinstance(expr, str):
        return set()
    out = {m.group(1) for m in re.finditer(r"\binputs\.([A-Za-z_]\w*)", expr)}
    out.update(re.findall(r"\b([A-Za-z_]\w*)\b", expr))
    return out


def _flatten(value) -> list[str]:
    """Every string inside a nested dict/list, since `tool_args` nest freely."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [s for v in value.values() for s in _flatten(v)]
    if isinstance(value, (list, tuple)):
        return [s for v in value for s in _flatten(v)]
    return []


@graph_rule(severity=Severity.WARNING)
def steps_are_wired_to_each_other(graph, ctx, report) -> None:
    """Several nodes and no dependencies at all is a bag, not a pipeline."""
    if len(graph.nodes) > 1 and not any(n.depends_on for n in graph.nodes):
        report.add(ValidationIssue(
            severity=Severity.WARNING,
            kind="structure",
            message=(
                "No step is connected to any other, so they all run at once and "
                "none can use another's result."
            ),
            suggestion="Draw connections so each step feeds the next.",
        ))




def _is_terminal(node) -> bool:
    """Does the graph stop here? Asks the kind's own spec, never a kind name."""
    spec = node_kind_spec(getattr(node, "kind", ""))
    return bool(spec and spec.terminal)



# NOTE: there is deliberately no "this workflow has no way in" rule. A workflow
# that needs nothing to start — fetch today's headlines, summarise a fixed file —
# is ordinary and complete, so requiring an input step would reject correct
# graphs. The case that rule was reaching for is a prompt reading a value nothing
# supplies, and `templates.placeholders_resolve` already catches exactly that,
# naming the placeholder that does not resolve.


@graph_rule(severity=Severity.WARNING)
def workflow_has_a_way_out(graph, ctx, report) -> None:
    """A workflow that runs and hands nothing back.

    A warning rather than an error: the run genuinely succeeds and its per-step
    results are still readable in the trace. What it cannot do is *answer* —
    anything calling this workflow gets nothing.
    """
    if graph.outputs or ctx.has_kind("output"):
        return
    if not graph.nodes:
        return
    report.add(ValidationIssue(
        severity=Severity.WARNING,
        kind="workflow.no_result",
        message=(
            "This workflow does not say what to return, so it will run but hand "
            "nothing back to whoever asked."
        ),
        suggestion="Add an Output step and connect the result to it.",
    ))


@graph_rule(severity=Severity.WARNING)
def no_step_is_left_out(graph, ctx, report) -> None:
    """A step wired to nothing at either end.

    Narrower and more certain than the orphan rule: this one has no incoming
    connection *and* no outgoing one, so it is not part of the workflow at all —
    it runs alone, on whatever the graph started with, and its result is
    discarded.
    """
    if len(graph.nodes) <= 1:
        return
    for n in graph.nodes:
        if n.depends_on:
            continue
        if n.id in ctx.referenced_ids:
            continue
        if _is_terminal(n):
            continue  # covered by `output.no_source`, which says it better
        report.add(ValidationIssue(
            severity=Severity.WARNING,
            kind="structure.isolated",
            node_id=n.id,
            message=(
                "This step is not connected to anything — nothing feeds it and "
                "nothing uses its result."
            ),
            suggestion="Connect it into the workflow, or remove it.",
        ))
