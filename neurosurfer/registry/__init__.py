"""The tool registry — what is available, what it can do, and what it needs.

One lookup path over three kinds of backing:

- **core** — a Python class in this package (`registry/core/<domain>/`);
- **imported** — a verified MCP server plus the tool name it knows;
- **authored** — generated source, kept after it passed its sandbox.

Callers ask for a *capability*, not a tool, and get back manifests. That is the
whole point: the Architect spent a long time matching plain English against tool
descriptions and picking wrong — a database-migration tool for "generate chart
images", a hosted gateway for a database on `localhost`. Resolution here matches
a declared tag, and where a tool cannot be reached the manifest says so.

Prose search remains, as a *tiebreak and a fallback for untagged tools*, never as
the primary signal.
"""

from __future__ import annotations

from typing import Any

from .capabilities import CAPABILITIES, describe, is_known
from .icons import icon_bytes, icon_slugs, resolve_icon
from .manifest import (
    ORIGINS,
    RUNTIMES,
    OperationManifest,
    ToolManifest,
    Verification,
    humanise,
    manifest_for,
)

__all__ = [
    "CAPABILITIES",
    "ORIGINS",
    "RUNTIMES",
    "OperationManifest",
    "ToolManifest",
    "Verification",
    "describe",
    "humanise",
    "icon_bytes",
    "icon_slugs",
    "is_known",
    "manifest_for",
    "resolve_icon",
    "manifests",
    "providers_of",
    "unsatisfied_capabilities",
]


def manifests(*, workflow_only: bool = True) -> list[ToolManifest]:
    """A manifest for every registered tool.

    Reads the live catalog rather than a stored list, so an MCP server started
    thirty seconds ago is included — and included *as what it is*, with no
    capabilities and an unknown credential story, rather than as an equal
    candidate to a core tool that has declared both.
    """
    from neurosurfer.tools.registry import all_tools, workflow_node_tools

    tools = workflow_node_tools() if workflow_only else all_tools()
    return [manifest_for(t) for t in tools]


def providers_of(
    capability: str,
    *,
    workflow_only: bool = True,
    reaches_localhost: bool | None = None,
) -> list[ToolManifest]:
    """Tools that declare *capability*, best-known first.

    *reaches_localhost* filters on where the tool runs. Pass ``True`` when the
    workflow targets something on this machine: a hosted service is a perfectly
    good tool that simply cannot see `localhost:1433`, and offering one for a
    database in a local container wasted an afternoon before this field existed.
    """
    out = [m for m in manifests(workflow_only=workflow_only) if m.covers(capability)]
    if reaches_localhost is not None:
        out = [m for m in out if m.reaches_localhost == reaches_localhost]
    # Verified first, then core over imported over unknown-origin: a tool that has
    # demonstrably run beats one that merely claims the tag.
    rank = {"core": 0, "authored": 1, "imported": 2, "mcp": 3}
    return sorted(
        out,
        key=lambda m: (0 if (m.verified and m.verified.ok) else 1,
                       rank.get(m.origin, 9), m.name),
    )


def unsatisfied_capabilities(*, workflow_only: bool = True) -> list[str]:
    """Vocabulary entries nothing provides.

    The honest answer to "generate a chart" is *"no tool here has that
    capability"* — which is a job to author or import, and a far better answer
    than whichever tool shared the most words with the request.
    """
    covered: set[str] = set()
    for m in manifests(workflow_only=workflow_only):
        covered |= m.capabilities
    return sorted(set(CAPABILITIES) - covered)


def registry_report() -> dict[str, Any]:
    """What the registry holds, for the catalog API and for a human at a prompt."""
    ms = manifests(workflow_only=False)
    return {
        "tools": len(ms),
        "by_origin": {o: sum(1 for m in ms if m.origin == o) for o in ORIGINS},
        "untagged": sorted(m.name for m in ms if not m.capabilities),
        "unsatisfied": unsatisfied_capabilities(),
    }
