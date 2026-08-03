"""A rule declares what it applies to, and the runner finds it.

The previous shape called twenty-five module-private ``_check_*`` functions from
a hand-written sequence inside ``validate_package``. Adding a rule meant editing
that sequence; knowing *which rules can fire for an input node* meant reading all
eleven hundred lines and keeping the answer in your head.

A rule now says so:

    @node_rule(kinds=("base", "react"), severity=Severity.ERROR)
    def no_model(node, ctx, report): ...

which makes "what can fire on this kind" a filter over :func:`node_rules`, and
makes the audit a test rather than a reading exercise — see
``tests/engine/test_validation_rules.py``.

This is the same move ``engine/kinds`` made for node configuration: knowledge
that every consumer was re-deriving, stated once as data.

**Order is registration order**, which is import order in ``__init__``. Rules are
independent by construction — none reads another's output — so this only decides
the order issues are listed in, and listing them per-kind in the order the
modules are imported is stable and readable.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from .models import Severity, ValidationReport

__all__ = [
    "NodeRule",
    "GraphRule",
    "node_rule",
    "graph_rule",
    "node_rules",
    "graph_rules",
    "rules_for_kind",
]

#: A node rule: given the node, the shared context, and the report, add issues.
NodeCheck = Callable[[Any, Any, ValidationReport], None]
#: A graph rule: given the whole graph, the context, and the report.
GraphCheck = Callable[[Any, Any, ValidationReport], None]

#: Applies to every kind. A rule that says nothing about kinds means all of them.
ALL_KINDS = "*"


@dataclass(frozen=True)
class NodeRule:
    """One rule, and the kinds it is allowed to speak about."""

    name: str
    fn: NodeCheck
    kinds: tuple[str, ...]
    #: The severity this rule reports *by default*. A rule may still raise a
    #: lesser issue itself; this is what it declares, and what the audit reads.
    severity: Severity
    #: Does it run on nodes nested in a container body?
    #:
    #: Not all do, and the distinction is real: a body node's *tools* and
    #: *imports* must resolve exactly as a top-level node's must, but its edges
    #: are body-scoped and were already checked at load time.
    bodies: bool
    doc: str

    def applies_to(self, kind: str) -> bool:
        return ALL_KINDS in self.kinds or kind in self.kinds


@dataclass(frozen=True)
class GraphRule:
    """One rule about the workflow as a whole."""

    name: str
    fn: GraphCheck
    severity: Severity
    doc: str


_NODE_RULES: list[NodeRule] = []
_GRAPH_RULES: list[GraphRule] = []


def node_rule(
    *,
    kinds: Sequence[str] = (ALL_KINDS,),
    severity: Severity = Severity.ERROR,
    bodies: bool = True,
) -> Callable[[NodeCheck], NodeCheck]:
    """Register a per-node rule.

    ``kinds`` is the point of the decorator: a rule that only makes sense for an
    agent should not be reachable from an output node, and stating it here is
    what stops the body of every check beginning with a kind test.
    """

    def wrap(fn: NodeCheck) -> NodeCheck:
        _NODE_RULES.append(
            NodeRule(
                name=fn.__name__,
                fn=fn,
                kinds=tuple(kinds),
                severity=severity,
                bodies=bodies,
                doc=(fn.__doc__ or "").strip().split("\n")[0],
            )
        )
        return fn

    return wrap


def graph_rule(*, severity: Severity = Severity.ERROR) -> Callable[[GraphCheck], GraphCheck]:
    """Register a rule about the workflow as a whole."""

    def wrap(fn: GraphCheck) -> GraphCheck:
        _GRAPH_RULES.append(
            GraphRule(
                name=fn.__name__,
                fn=fn,
                severity=severity,
                doc=(fn.__doc__ or "").strip().split("\n")[0],
            )
        )
        return fn

    return wrap


def node_rules() -> tuple[NodeRule, ...]:
    return tuple(_NODE_RULES)


def graph_rules() -> tuple[GraphRule, ...]:
    return tuple(_GRAPH_RULES)


def rules_for_kind(kind: str) -> tuple[NodeRule, ...]:
    """Every rule that can fire on a node of *kind* — the question the old shape
    could not answer without reading the whole file."""
    return tuple(r for r in _NODE_RULES if r.applies_to(kind))


def run_node_rules(node: Any, ctx: Any, report: ValidationReport, *, in_body: bool) -> None:
    for rule in _NODE_RULES:
        if in_body and not rule.bodies:
            continue
        if not rule.applies_to(getattr(node, "kind", "")):
            continue
        rule.fn(node, ctx, report)


def run_graph_rules(graph: Any, ctx: Any, report: ValidationReport) -> None:
    for rule in _GRAPH_RULES:
        rule.fn(graph, ctx, report)


def iter_all(rules: Iterable[NodeRule | GraphRule]) -> Iterable[str]:
    return (r.name for r in rules)
