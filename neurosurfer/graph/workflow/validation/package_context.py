"""The entry point: build the context, run every registered rule, return the report.

Kept apart from ``__init__`` so importing the package's types does not drag in
the package loader, and so the order of operations is one short readable
function rather than something spread across a module docstring.
"""

from __future__ import annotations

from ..package import WorkflowPackage, _PackagePathContext
from . import graph as _graph_rules  # noqa: F401  (registers the graph rules)
from . import nodes as _node_rules  # noqa: F401  (registers the per-node rules)
from . import templates as _template_rules  # noqa: F401  (registers the scope walk)
from .context import ValidationContext
from .models import ValidationReport
from .registry import run_graph_rules, run_node_rules
from .tool_schema import registered_tool_names

__all__ = ["validate_package", "DEFER_MARKER", "INFEASIBLE_MARKER"]

# Sentinel the `assemble` node returns (instead of a registered path) when a staged
# package did not pass validation. Returning it — rather than raising — keeps the
# executor from dumping a traceback; the ArchitectBuilder re-validates the staged dir
# and either renders a clean error (hard errors) or authors the missing tools (gaps).
DEFER_MARKER = "__STAGED_NEEDS_FINALIZE__:"

# Sentinel the `assemble` node returns when the tool_design step judged the workflow
# infeasible (a node needs a capability that cannot be built safely as described). The
# text after the marker is the human-readable feasibility report. The ArchitectBuilder
# turns this into a clean "not doable" message instead of registering a broken workflow.
INFEASIBLE_MARKER = "__WORKFLOW_INFEASIBLE__:"


def validate_package(
    pkg: WorkflowPackage, *, known_providers: set[str] | None = None
) -> ValidationReport:
    """Validate *pkg* beyond structural loading. Returns a :class:`ValidationReport`.

    *known_providers* is three-valued on purpose: a set says "these exist", and
    ``None`` says "do not ask". Most callers genuinely do not know what a given
    deployment has configured, and guessing would reject valid graphs on a
    machine that simply has not been set up yet.
    """
    report = ValidationReport()
    ctx = ValidationContext(
        package=pkg,
        graph=pkg.graph,
        known_providers=known_providers,
        registered_tools=registered_tool_names(),
    )

    # Import resolution needs the package directory on `sys.path` — a package may
    # carry its own `schemas.py` and `nodes/`. Every node rule runs inside it so
    # no rule has to know whether it is the one that imports something.
    with _PackagePathContext(pkg):
        for node in ctx.graph.nodes:
            run_node_rules(node, ctx, report, in_body=False)
        # Nodes nested in a container body are real nodes that really run, so
        # their tools and imports must resolve too. Rules that are about wiring
        # opt out via `bodies=False` — a body's edges are body-scoped and the
        # loader has already checked them against their own siblings.
        for node in ctx.body_nodes:
            run_node_rules(node, ctx, report, in_body=True)

        run_graph_rules(ctx.graph, ctx, report)

    return report
