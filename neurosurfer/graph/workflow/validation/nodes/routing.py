"""A router's branches: do they exist, and do they run after the decision.

## Why this is a rule and not only a loader check

`load_graph_from_dict` already refuses a router whose target is unknown, or whose
target does not list the router in `depends_on`. That covers YAML — and only YAML.

A graph built in **Python** never passes through the loader:

    Graph(nodes=[GraphNode(id="triage", kind="router", routes={"b": "GHOST"}), ...])

is constructed, validated by pydantic, and handed straight to the executor. So the
same mistake was a clear load-time error in one path and *silence* in the other,
with `validate_package(...).ok` reporting `True` on a graph that cannot route.

Two paths that disagree about whether a graph is valid is the defect this package
exists to remove — see `validation/__init__.py`. These rules put the loader's
guarantees on the Python path too, at the same severity.
"""

from __future__ import annotations

from typing import Any

from neurosurfer.graph.engine.templates import node_instruction

from ..models import Severity, ValidationIssue, ValidationReport
from ..registry import node_rule

__all__ = [
    "router_targets_exist",
    "router_targets_run_after_the_decision",
    "routes_router_classifies_on_something",
]


def _targets(node: Any) -> list[tuple[str, str]]:
    """Every `(label, target_id)` this router can send work to.

    Covers all three ways a router names a branch: `routes` (the classify form),
    `cases` (the expression form), and `default`. A rule that read only `routes`
    would pass a `cases` router with a typo in it.
    """
    out: list[tuple[str, str]] = []
    for label, target in (getattr(node, "routes", None) or {}).items():
        out.append((str(label), str(target)))
    for case in getattr(node, "cases", None) or []:
        target = getattr(case, "to", None)
        if target:
            out.append((str(getattr(case, "when", "") or "case"), str(target)))
    default = getattr(node, "default", None)
    if default:
        out.append(("default", str(default)))
    return out


@node_rule(kinds=("router",), severity=Severity.ERROR)
def router_targets_exist(node, ctx, report: ValidationReport) -> None:
    """A branch that points at a node the graph does not have.

    An error rather than a warning, and not a judgement call: the branch is
    unreachable, so one classification out of however many silently does nothing
    — and *which* one depends on what the model answers, so it may not show up
    until the workflow has been running for weeks.
    """
    for label, target in _targets(node):
        if target in ctx.node_ids:
            continue
        have = ", ".join(sorted(ctx.node_ids)) or "nothing"
        report.add(ValidationIssue(
            severity=Severity.ERROR,
            kind="router.unknown_target",
            node_id=node.id,
            subject=target,
            message=(
                f"This step routes '{label}' to '{target}', which is not a step "
                f"in this workflow."
            ),
            suggestion=f"The steps it could route to are: {have}.",
            detail=f"routes/cases target {target!r}",
        ))


@node_rule(kinds=("router",), severity=Severity.ERROR)
def router_targets_run_after_the_decision(node, ctx, report: ValidationReport) -> None:
    """A branch that exists but does not wait for the router.

    `depends_on` is what orders the run. A target that omits its router is not
    "after the decision" at all — it is an independent node in the same layer, so
    it runs *whatever* the router chose, and the branch that was supposed to be
    pruned executes anyway.

    The loader states this rule for YAML in as many words; this is the same rule
    for a graph built in Python.
    """
    for label, target in _targets(node):
        dep_of_target = ctx.by_id.get(target)
        if dep_of_target is None:
            continue  # `router_targets_exist` reports this one
        if node.id in (getattr(dep_of_target, "depends_on", None) or []):
            continue
        report.add(ValidationIssue(
            severity=Severity.ERROR,
            kind="router.target_not_downstream",
            node_id=node.id,
            subject=target,
            message=(
                f"This step routes '{label}' to '{target}', but '{target}' does "
                f"not wait for it — so it will run whichever branch is chosen."
            ),
            suggestion=f"Add '{node.id}' to what '{target}' depends on.",
            detail=f"{target}.depends_on does not contain {node.id!r}",
        ))


@node_rule(kinds=("router",), severity=Severity.WARNING)
def routes_router_classifies_on_something(node, ctx, report: ValidationReport) -> None:
    """A `routes` router with no evidence to classify on.

    The trap this catches was met while writing tutorial 03, and it is the most
    expensive shape in the engine because **nothing about it looks wrong**. A
    `routes` router builds its own prompt: the instruction, the allowed labels,
    and — only if it declared dependencies — their outputs. It is *not* handed
    the graph inputs as a block the way a base node is.

    So an instruction that never interpolates anything and never wires an
    upstream node is classifying a request it was never shown. Every answer
    still maps to a label, so the run is green, the branch is taken, and it is
    the wrong branch every time. A double-billing complaint and a 500 error went
    down the same path this way.

    A warning, not an error: a router can legitimately be pure prose when its
    dependency supplies the evidence, and that case is already excluded here.
    What is left is a real one, and `{}`-free prose is not proof of a mistake —
    only of a router with nothing in front of it.
    """
    if not getattr(node, "routes", None):
        return  # `cases` routers evaluate expressions; they read state directly
    if getattr(node, "depends_on", None):
        return  # upstream outputs are appended to the classifier prompt
    if "{" in node_instruction(node, ""):
        return  # it interpolates something — `templates.py` checks *what*

    report.add(ValidationIssue(
        severity=Severity.WARNING,
        kind="router.classifies_on_nothing",
        node_id=node.id,
        subject=node.id,
        message=(
            "This step decides which branch to take, but nothing tells it what "
            "it is deciding about — it has no earlier step to read and its "
            "instruction never mentions an input."
        ),
        suggestion=(
            "Put the value in the instruction (e.g. 'Classify this ticket: "
            "{ticket}'), or make it depend on the step that produces it."
        ),
        detail="routes router with no depends_on and no template placeholder",
    ))
