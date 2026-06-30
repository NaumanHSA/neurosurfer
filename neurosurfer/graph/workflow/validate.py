"""Pre-registration validation gate for workflow packages (Phase E1).

`load_package` only proves a package is *structurally* loadable (schema, node kinds,
DAG acyclicity). It does NOT prove the package can actually run: tool names may be
invented, ``output_schema`` / ``callable`` import paths may not resolve, and edges may
point at non-existent nodes.

:func:`validate_package` is the hard gate the Architect's ``assemble`` step runs
**before** registering. Nothing registers unless it passes. Capability gaps (a node
that wants a tool no registered tool provides) are reported separately from hard
errors so a later phase (E4–E6) can resolve them by authoring a new tool — for now
they block registration with an actionable message.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass, field

from pydantic import BaseModel

from neurosurfer.graph.engine import import_string

from .package import WorkflowPackage, _PackagePathContext

__all__ = [
    "ValidationIssue",
    "ValidationReport",
    "validate_package",
    "DEFER_MARKER",
    "INFEASIBLE_MARKER",
]

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


@dataclass
class ValidationIssue:
    """A single problem found while validating a package."""

    kind: str  # tool_gap | tool_typo | schema | callable | dag | structure
    message: str
    node_id: str | None = None
    suggestion: str | None = None
    subject: str | None = None  # the offending name (e.g. the missing tool)

    def render(self) -> str:
        where = f"node '{self.node_id}': " if self.node_id else ""
        hint = f" → {self.suggestion}" if self.suggestion else ""
        return f"{where}{self.message}{hint}"


@dataclass
class ValidationReport:
    """Outcome of :func:`validate_package`.

    ``errors`` are hard problems (typos, bad imports, broken edges). ``gaps`` are
    capability gaps — a node wants a tool that does not exist anywhere. Both block
    registration today; gaps are kept separate so E4–E6 can route them to the
    tool-author agent instead of failing.
    """

    errors: list[ValidationIssue] = field(default_factory=list)
    gaps: list[ValidationIssue] = field(default_factory=list)
    warnings: list[ValidationIssue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors and not self.gaps

    def summary(self) -> str:
        lines: list[str] = []
        if self.errors:
            lines.append("Errors:")
            lines += [f"  • {e.render()}" for e in self.errors]
        if self.gaps:
            lines.append("Capability gaps (no registered tool provides this):")
            lines += [f"  • {g.render()}" for g in self.gaps]
        if self.warnings:
            lines.append("Warnings:")
            lines += [f"  • {w.render()}" for w in self.warnings]
        return "\n".join(lines) if lines else "Package is valid."


def validate_package(pkg: WorkflowPackage) -> ValidationReport:
    """Validate *pkg* beyond structural loading. Returns a :class:`ValidationReport`.

    Checks:
      1. every node tool resolves to a registered tool (else gap/typo)
      2. every ``output_schema`` imports to a ``BaseModel`` subclass
      3. every function-node ``callable`` imports to a callable
      4. ``depends_on`` / ``outputs`` reference existing node ids
    """
    report = ValidationReport()
    graph = pkg.graph
    node_ids = {n.id for n in graph.nodes}
    registered = _registered_tool_names()

    for node in graph.nodes:
        _check_tools(node, registered, report)
        _check_edges(node, node_ids, report)

    # Import-resolution checks need the package dir on sys.path (schemas.py / nodes/).
    with _PackagePathContext(pkg):
        for node in graph.nodes:
            _check_output_schema(node, report)
            _check_callable(node, report)

    # outputs must reference real nodes
    for out in graph.outputs:
        if out not in node_ids:
            report.errors.append(ValidationIssue(
                kind="dag",
                message=f"output '{out}' is not a node in the graph",
            ))

    # Depth-floor: a 1–2-node workflow is almost certainly under-designed.
    llm_nodes = [n for n in graph.nodes if n.kind in {"base", "react"}]
    if len(llm_nodes) < 3:
        report.warnings.append(ValidationIssue(
            kind="structure",
            message=(
                f"workflow has only {len(llm_nodes)} LLM node(s) — "
                "consider adding intermediate steps (validation, transformation, "
                "output formatting) to make it more robust"
            ),
        ))

    return report


# ── individual checks ──────────────────────────────────────────────────────────

def _registered_tool_names() -> set[str]:
    # Imported lazily so validate.py has no import-time dependency on the tool layer
    # (keeps the workflows package importable without the full tool registry).
    from neurosurfer.tools.registry import all_tools  # noqa: PLC0415

    return {t.name for t in all_tools()}


def _check_tools(node, registered: set[str], report: ValidationReport) -> None:
    for tool in node.tools or []:
        if tool in registered:
            continue
        match = difflib.get_close_matches(tool, registered, n=1, cutoff=0.7)
        if match:
            report.errors.append(ValidationIssue(
                kind="tool_typo",
                node_id=node.id,
                message=f"tool '{tool}' is not registered",
                suggestion=f"did you mean '{match[0]}'?",
                subject=tool,
            ))
        else:
            report.gaps.append(ValidationIssue(
                kind="tool_gap",
                node_id=node.id,
                message=f"tool '{tool}' does not exist; no registered tool provides it",
                suggestion="compose existing tools, or author a new tool",
                subject=tool,
            ))


def _check_edges(node, node_ids: set[str], report: ValidationReport) -> None:
    for dep in node.depends_on or []:
        if dep not in node_ids:
            report.errors.append(ValidationIssue(
                kind="dag",
                node_id=node.id,
                message=f"depends_on '{dep}' is not a node in the graph",
            ))


def _check_output_schema(node, report: ValidationReport) -> None:
    path = node.output_schema
    if not path:
        return
    try:
        obj = import_string(path)
    except Exception as exc:  # noqa: BLE001 - any import failure is a validation error
        report.errors.append(ValidationIssue(
            kind="schema",
            node_id=node.id,
            message=f"output_schema '{path}' does not import ({exc})",
        ))
        return
    if not (isinstance(obj, type) and issubclass(obj, BaseModel)):
        report.errors.append(ValidationIssue(
            kind="schema",
            node_id=node.id,
            message=f"output_schema '{path}' is not a pydantic BaseModel subclass",
        ))


def _check_callable(node, report: ValidationReport) -> None:
    if node.kind not in {"function", "python"}:
        return
    path = node.callable
    if not path:
        report.errors.append(ValidationIssue(
            kind="callable",
            node_id=node.id,
            message=f"{node.kind} node has no 'callable' set",
        ))
        return
    try:
        obj = import_string(path)
    except Exception as exc:  # noqa: BLE001 - any import failure is a validation error
        report.errors.append(ValidationIssue(
            kind="callable",
            node_id=node.id,
            message=f"callable '{path}' does not import ({exc})",
        ))
        return
    if not callable(obj):
        report.errors.append(ValidationIssue(
            kind="callable",
            node_id=node.id,
            message=f"callable '{path}' is not callable",
        ))
