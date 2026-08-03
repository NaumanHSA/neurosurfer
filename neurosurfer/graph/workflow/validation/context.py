"""Everything the rules need about the workflow, worked out once.

Each ``_check_*`` used to derive what it needed from the graph itself: the id
set, the tool registry, the dependency map. With twenty-five of them that is the
same walk repeated, and — worse — the same *definition* repeated, so two checks
could disagree about what "reachable" meant without either being wrong on its own
terms.

The context is built once per validation and handed to every rule. It holds only
facts about the graph, never a partial verdict: a rule reads it and decides for
itself, so no rule can depend on another having run first.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any

__all__ = ["ValidationContext", "body_nodes"]


def body_nodes(nodes: Any) -> list:
    """Every node nested in a `body`, at any depth (bodies can contain bodies)."""
    found: list = []
    for node in nodes:
        body = getattr(node, "body", None) or []
        if body:
            found.extend(body)
            found.extend(body_nodes(body))
    return found


@dataclass
class ValidationContext:
    """Shared, read-only facts about the package under validation."""

    package: Any
    graph: Any
    #: Configured provider profiles, or ``None`` when the caller does not know.
    #:
    #: ``None`` is not "no providers" — it is "don't ask". A package is validated
    #: in plenty of places with no idea what a deployment has configured, and
    #: guessing there would reject valid graphs on an unconfigured machine.
    known_providers: set[str] | None = None
    #: Names in the tool registry.
    registered_tools: set[str] = field(default_factory=set)

    @cached_property
    def node_ids(self) -> set[str]:
        return {n.id for n in self.graph.nodes}

    @cached_property
    def by_id(self) -> dict[str, Any]:
        return {n.id: n for n in self.graph.nodes}

    @cached_property
    def body_nodes(self) -> list:
        return body_nodes(self.graph.nodes)

    @cached_property
    def all_nodes(self) -> list:
        return [*self.graph.nodes, *self.body_nodes]

    @cached_property
    def referenced_ids(self) -> set[str]:
        """Node ids something in the graph points at.

        Dependencies, error routes, the declared outputs, and any ``nodes.<id>``
        an expression reads. This is the definition of "consumed", and having it
        in one place is what stops two rules disagreeing about it.
        """
        found: set[str] = set(self.graph.outputs or [])
        for n in self.graph.nodes:
            found.update(n.depends_on or [])
            if getattr(n, "on_error", None):
                found.add(n.on_error)
            for e in self._expressions(n):
                found.update(re.findall(r"nodes\.([A-Za-z_]\w*)", e))
        return found

    @cached_property
    def referenced_vars(self) -> set[str]:
        """Variable names an expression reads as ``vars.<name>``."""
        found: set[str] = set()
        for n in self.graph.nodes:
            for e in self._expressions(n):
                found.update(re.findall(r"vars\.([A-Za-z_]\w*)", e))
        return found

    @staticmethod
    def _expressions(node: Any) -> list[str]:
        exprs = [
            getattr(node, "when", None),
            getattr(node, "until", None),
            getattr(node, "break_when", None),
            getattr(node, "over", None),
        ]
        exprs += [c.when for c in (getattr(node, "cases", None) or [])]
        return [e for e in exprs if e]

    def nodes_of_kind(self, *kinds: str) -> list:
        return [n for n in self.graph.nodes if n.kind in kinds]

    def has_kind(self, *kinds: str) -> bool:
        return any(n.kind in kinds for n in self.graph.nodes)
