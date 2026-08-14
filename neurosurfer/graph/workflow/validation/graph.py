"""Rules about the workflow as a whole.

Everything that cannot be decided by looking at one node: whether an edge points
at a node that exists, whether anything consumes a node's output, whether the
graph has a way in and a way out.
"""

from __future__ import annotations

from typing import NamedTuple

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


@graph_rule(severity=Severity.ERROR)
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

    **A code node's signature is one of those ways.** `function` and `python`
    nodes are called `fn(**{**graph_inputs, **dependency_results, **scope})`, so
    a parameter named `db_path` reads `db_path` with no template anywhere — the
    same argument that already exempted `tool` nodes. Judging one kind by its
    parameters and the other by its templates reported the capstone tutorial as
    ignoring two inputs its functions consume on every run.

    **This blocks, and it did not always.** As a warning the workflow stayed
    registerable and the backstop was a human noticing the answer ignored the
    parameter. That backstop does not exist for a workflow the Architect builds
    and verifies on its own: `gpt-5-mini`'s first build of a ticket-routing
    intent declared `ticket_text`, named it in none of its five steps, and would
    have registered — a workflow that accepts the ticket and never reads it. A
    warning it does not have to act on is a warning a model will not act on. As
    an error the repair loop must fix it before the build can register, which is
    the only enforcement in the loop that works on a weak model.

    **It downgrades itself when it cannot see.** Blocking is only honest while
    the analysis is complete, and one thing makes it incomplete: a code node
    whose signature could not be read, whose parameters are therefore invisible
    and whose inputs would look unread. Then this reports a warning and says so,
    because refusing to run a graph over a fact you could not establish is worse
    than the gap. `_analysis_is_complete` is that question, and the import
    failure behind it is `callable_resolves`' finding to report, not this one's.
    """
    declared = [
        getattr(i, "name", None) or (i.get("name") if isinstance(i, dict) else None)
        for i in (graph.inputs or [])
    ]
    declared = [n for n in declared if n]
    if not declared:
        return

    # A code node taking `**kwargs` receives the whole inputs mapping, so no
    # declared input is unread and there is nothing this rule can say.
    if _some_node_reads_every_input(ctx):
        return

    # Certain enough to refuse the run, or only to mention it? See the docstring:
    # a step whose parameters this cannot see hides the very reads that would
    # clear the input, so this is the difference between a finding and a guess.
    doubt = _analysis_is_incomplete(ctx)

    read = _names_read_anywhere(ctx)
    for name in declared:
        if name in read:
            continue
        report.add(ValidationIssue(
            severity=Severity.WARNING if doubt else Severity.ERROR,
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
            detail=(
                f"graph input {name!r} appears in no template, expression or "
                f"argument" + (f"; {doubt}" if doubt else "")
            ),
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
        for expr in (getattr(node, "over", None), getattr(node, "when", None)):
            found.update(_expression_names(expr))
        for case in getattr(node, "cases", None) or []:
            found.update(_expression_names(getattr(case, "when", None)))
        # `tool_args` / `tool_settings` values are templates too.
        for holder in (getattr(node, "tool_args", None) or {},
                       getattr(node, "tool_settings", None) or {}):
            for value in _flatten(holder):
                found |= roots(value)
        # **An output node's `value` is a template.** `_TEMPLATE_FIELDS` is the
        # prompt fields and stops there, so `value: "hello {who}"` — a whole
        # one-node workflow whose answer is composed from an input — read
        # nothing as far as this could tell. Docstring said it counted; only
        # this makes it true.
        for value in _flatten(getattr(node, "value", None)):
            found |= roots(value)
        # An input step's own key. The caller's value lands *on* the node, which
        # then yields it — `engine/utils.input_node_keys` is the same mapping
        # read from the other side. A resume that supplies `approval` for the
        # step named `approval` is that input being used, not ignored.
        if getattr(node, "kind", None) == "input":
            found.add(getattr(node, "writes", None) or getattr(node, "id", ""))
        # A tool node is handed the inputs dict as kwargs, so a parameter name
        # matching an input is a read even with no template anywhere.
        if getattr(node, "kind", None) == "tool":
            found.update(getattr(node, "tool_args", None) or {})
        # **A code node is handed the same dict** — `fn(**{**graph_inputs,
        # **dependency_results, **scope})` in `executor/deterministic.py`. So its
        # signature is a list of reads exactly as `tool_args` is, and judging one
        # by its parameters while judging the other by its templates reported a
        # working graph as ignoring the inputs its functions consume on every run.
        elif getattr(node, "kind", None) in {"function", "python"}:
            found |= _callable_parameters(node).names
    return found


def _code_nodes(ctx) -> list:
    """The nodes called with the inputs mapping as kwargs, top level and bodies."""
    return [
        node for node in ctx.all_nodes
        if getattr(node, "kind", None) in {"function", "python"}
    ]


def _has_a_tool_node(ctx) -> bool:
    """A `tool` node is handed the inputs mapping as kwargs, like a code node —
    but its parameters live in a registered tool's schema, not in a signature
    this rule can import.

    So the input a tool consumes is invisible here, the same way an uninspectable
    callable's is: `{"id": "get_motto", "kind": "tool", "tools": ["city_motto"]}`
    reads `city` on every run and names it nowhere. `engine/utils` reached the
    same conclusion from the other side and silences its check outright for these
    kinds. This is one step less blunt — the rule still says so, it just may not
    refuse the run over it.
    """
    return any(getattr(node, "kind", None) == "tool" for node in ctx.all_nodes)


def _analysis_is_incomplete(ctx) -> str | None:
    """Why this rule cannot be sure, or `None` when it can. **The gate on blocking.**

    A code node reads inputs by parameter name, so a signature this could not
    inspect is a set of reads it cannot see — and an input that is genuinely
    consumed there looks unread. Warning about that is a nuisance; refusing to
    run over it is a working graph stopped at the door on a fact that was never
    established. So the rule keeps its voice and loses its veto.

    Two ways for a callable to be uninspectable, both handled by their own rule
    and neither worth reporting twice: one that will not import
    (`callable_resolves` says so, with the path and the exception) and one whose
    signature `inspect` refuses — a C function, or an object whose `__call__` it
    cannot follow. A `tool` node is the third way and needs no failure at all;
    see `_has_a_tool_node`.

    The reason is returned rather than a bare `False` because it goes in the
    issue's `detail`: "no step names it" and "no step names it, but I could not
    read one of them" send a reader to different places.
    """
    if _has_a_tool_node(ctx):
        return "a tool step is handed the inputs directly, so this may be a false alarm"
    if not all(_callable_parameters(node).inspected for node in _code_nodes(ctx)):
        return "a step's callable could not be inspected, so this may be a false alarm"
    return None


def _some_node_reads_every_input(ctx) -> bool:
    """True when some node is handed the whole inputs mapping, so nothing is unread.

    Two shapes qualify, and the second is why this rule could not block before:

    - **A code node declaring `**kwargs`** — it receives whatever it is given.
    - **A `dict`-mode input node** — it *is* the declaration. It collects the
      graph's declared inputs by definition (`engine/utils.input_node_keys`
      skips it for exactly that reason), so every one of them is read by the
      step whose whole job is reading them. As a warning this misfired quietly
      on the smallest correct graph there is — one input step, one field; as an
      error it would have refused to run it.

    Kept out of `_names_read_anywhere` deliberately. That helper returns a set of
    names and is also called by `engine/utils.py`; a set that claims to contain
    everything would answer `in` correctly and set *difference* wrongly, which is
    exactly the operation that caller uses. So "reads everything" is a separate
    question with its own answer, and the sentinel never leaves this module.
    """
    if any(_callable_parameters(node).var_kw for node in _code_nodes(ctx)):
        return True
    return any(
        getattr(node, "kind", None) == "input"
        and (getattr(node, "input_mode", None) or "text") == "dict"
        for node in ctx.all_nodes
    )


class _Params(NamedTuple):
    """What a code node's signature said, and whether it could be read at all.

    `inspected` is the field that matters and the reason this is not a plain
    tuple: "no parameters" and "no answer" are both an empty `names`, and
    conflating them is the difference between blocking a workflow on a finding
    and blocking it on a guess. See `_analysis_is_complete`.
    """

    names: set[str]
    var_kw: bool
    inspected: bool


def _callable_parameters(node) -> _Params:
    """A code node's parameter names, whether it takes `**kwargs`, and whether
    the signature could be read.

    Best-effort by design. `callable_resolves` is the rule that reports an import
    failure, with the path and the exception; repeating that here would report the
    same defect twice in different words. So a callable this cannot inspect
    contributes no reads and, through `inspected`, costs the rule its veto rather
    than turning invisible parameters into a refusal to run.
    """
    import inspect

    from neurosurfer.graph.engine import import_string

    path = getattr(node, "callable", None)
    if not path or not isinstance(path, str):
        # No path at all: `callable_required` is already reporting this, and the
        # node is unrunnable for a reason that has nothing to do with inputs.
        return _Params(set(), False, False)
    try:
        fn = import_string(path)
        params = inspect.signature(fn).parameters.values()
    except Exception:  # noqa: BLE001 — an uninspectable callable is not this rule's finding
        return _Params(set(), False, False)

    names = {
        p.name for p in params
        if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
    }
    var_kw = any(p.kind is p.VAR_KEYWORD for p in params)
    return _Params(names, var_kw, True)


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
