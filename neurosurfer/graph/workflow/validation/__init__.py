"""Pre-registration validation for workflow packages.

``load_package`` only proves a package is *structurally* loadable — schema, node
kinds, DAG acyclicity. It does not prove the package can run: tool names may be
invented, import paths may not resolve, edges may point at nodes that do not
exist, and a prompt may read a variable nothing produces.

:func:`validate_package` is the gate. Nothing registers unless it passes, and it
is not the studio's gate — six callers use it, of which the studio is one: the
Architect validates every build against it, its ReAct agent checks its own work
with it, and the eval harness scores against it. A rule added here is a rule the
Architect obeys.

## The shape

Rules are **declared**, not sequenced. Each says which kinds it speaks about and
at what severity (:mod:`.registry`), reads shared facts computed once
(:mod:`.context`), and appends to one list of issues carrying their own severity
(:mod:`.models`). What that buys, beyond tidiness, is that *"what can go wrong
with an input node"* becomes a query rather than a careful read of a thousand
lines — which is the question anyone auditing this actually has.

    validation/
        models.py     Severity · ValidationIssue · ValidationReport
        registry.py   @node_rule / @graph_rule, and the rule tables
        context.py    ids, dependencies, the tool registry — derived once
        graph.py      rules about the workflow as a whole
        templates.py  the scope walk: placeholders, secrets, tool args
        nodes/        rules about a single node, by concern
"""

from __future__ import annotations

from .models import GAP_KINDS, Severity, ValidationIssue, ValidationReport
from .package_context import DEFER_MARKER, INFEASIBLE_MARKER, validate_package

__all__ = [
    "DEFER_MARKER",
    "GAP_KINDS",
    "INFEASIBLE_MARKER",
    "Severity",
    "ValidationIssue",
    "ValidationReport",
    "validate_package",
]
