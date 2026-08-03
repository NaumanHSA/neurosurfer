"""Tools a node names: do they exist.

Two failures that look alike and are handled very differently. A **typo** is an
error — the author meant a tool that exists and mistyped it. A **gap** is a tool
nothing provides at all, which the Architect routes to the tool-author agent
rather than simply failing. `models.GAP_KINDS` is what keeps them apart
downstream; both block registration.
"""

from __future__ import annotations

import difflib

from ..models import Severity, ValidationIssue, ValidationReport
from ..registry import node_rule


def _check_tools(node, registered: set[str], report: ValidationReport) -> None:
    for tool in node.tools or []:
        if tool in registered:
            continue
        match = difflib.get_close_matches(tool, registered, n=1, cutoff=0.7)
        if match:
            report.add(ValidationIssue(
        severity=Severity.ERROR,
                kind="tool_typo",
                node_id=node.id,
                message=f"This step uses a tool called '{tool}', which does not exist.",
                suggestion=f"did you mean '{match[0]}'?",
                subject=tool,
            ))
        else:
            report.add(ValidationIssue(
        severity=Severity.ERROR,
                kind="tool_gap",
                node_id=node.id,
                message=f"This step needs a tool called '{tool}', and nothing provides it.",
                suggestion="compose existing tools, or author a new tool",
                subject=tool,
            ))


def _check_capability(node, registered: set[str], report: ValidationReport) -> None:
    """Can this node do what it says? (See ``capability.py`` for the reasoning.)

    Two findings, deliberately at different severities:

    - a ``react`` node with no tools is an **error**. It is not a judgement call:
      the kind means "LLM that calls tools in a loop", and the executor now refuses
      to run one, so registering it would ship a package that cannot start.
    - prompt text describing an external action on a toolless node is a
      **warning**, because the evidence is a string match. The Architect's register
      gate escalates it (a build must attach a tool, author one, or explain
      itself); a library caller validating a hand-written package is only told.
    """
    from ...capability import (  # noqa: PLC0415 - keep the module import-light
        GROUNDED_KINDS,
        node_action_text,
        suspected_capability,
        toolless_react_node,
        upstream_may_satisfy,
    )

    if toolless_react_node(node):
        report.add(ValidationIssue(
        severity=Severity.ERROR,
            kind="capability",
            node_id=node.id,
            message=(
                "`react` node has no tools — a react node is an LLM that calls "
                "tools in a loop, so with none it cannot act and will describe "
                "work it never did"
            ),
            suggestion=(
                "give it the tools it needs, or use kind `base` if it is a pure "
                "reasoning step"
            ),
        ))
        return  # the missing-tool story is already told; don't also guess at it

    if node.kind not in GROUNDED_KINDS or (node.tools or []) or getattr(node, "tool_args", None):
        return

    capability = suspected_capability(*node_action_text(node))
    if capability is None or upstream_may_satisfy(node, capability):
        return

    candidates = [t for t in capability.tools if t in registered]
    if candidates:
        suggestion = (
            f"attach a tool that can do it ({', '.join(f'`{c}`' for c in candidates)})"
            f"{' — or kind `tool` for a single direct call' if len(candidates) == 1 else ''}"
        )
    else:
        suggestion = (
            "no built-in tool provides this — connect an MCP server that does, "
            "author a tool, or declare the request blocked"
        )
    report.add(ValidationIssue(
        severity=Severity.WARNING,
        kind="capability_gap",
        node_id=node.id,
        message=(
            f"appears to {capability.label} but holds no tool — an LLM step "
            f"cannot reach outside its prompt, so it will invent the result"
        ),
        suggestion=suggestion,
        subject=capability.label,
    ))


# ── rules ────────────────────────────────────────────────────────────────────


# The three kinds that carry tools. `base` is included and that is not a slip:
# the engine gives a base node one round with them, so a base node naming a tool
# that does not exist fails at run time exactly as a react node would. It was
# briefly excluded here because the *spec* had omitted the field — see
# `engine/kinds/base.py`, where the omission was the bug.
@node_rule(kinds=("base", "react", "tool"), severity=Severity.ERROR)
def tools_exist(node, ctx, report) -> None:
    """Every tool a node names is one the registry has."""
    _check_tools(node, ctx.registered_tools, report)


@node_rule(kinds=("*",), severity=Severity.ERROR)
def capability_is_available(node, ctx, report) -> None:
    """A node asking for a capability nothing provides is a gap, not a typo."""
    _check_capability(node, ctx.registered_tools, report)
