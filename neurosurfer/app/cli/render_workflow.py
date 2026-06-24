"""Rich rendering helpers for workflow execution results."""

from __future__ import annotations

from typing import TYPE_CHECKING

from . import theme

if TYPE_CHECKING:
    from rich.console import Console

    from neurosurfer.graph import GraphExecutionResult
    from neurosurfer.graph.workflow.package import WorkflowPackage


def print_workflow_list(console: Console, names: list[str]) -> None:
    from rich.table import Table

    if not names:
        console.print(f"[{theme.DIM}]No workflows registered. Author one with /workflow build.[/{theme.DIM}]")
        return
    table = Table(title="Workflows", show_lines=False, title_style=f"bold {theme.ACCENT}")
    table.add_column("Name", style=theme.ACCENT_DIM, no_wrap=True)
    for name in names:
        table.add_row(name)
    console.print(table)


def print_workflow_info(console: Console, pkg: WorkflowPackage) -> None:
    from rich.table import Table

    console.print(f"[bold {theme.ACCENT}]{pkg.name}[/] v{pkg.version}")
    if pkg.description:
        console.print(f"[{theme.DIM}]{pkg.description}[/{theme.DIM}]")
    console.print()

    table = Table(show_header=True, show_lines=False, title_style=f"bold {theme.ACCENT}")
    table.add_column("Node", style=theme.ACCENT_DIM, no_wrap=True)
    table.add_column("Kind")
    table.add_column("Depends on")
    for node in pkg.graph.nodes:
        table.add_row(
            node.id,
            node.kind,
            ", ".join(node.depends_on) if node.depends_on else "—",
        )
    console.print(table)

    if pkg.graph.inputs:
        console.print(f"\n[{theme.DIM}]Inputs: {', '.join(i.name for i in pkg.graph.inputs)}[/{theme.DIM}]")
    if pkg.graph.outputs:
        console.print(f"[{theme.DIM}]Outputs: {', '.join(pkg.graph.outputs)}[/{theme.DIM}]")


def print_workflow_summary(console: Console, pkg: WorkflowPackage) -> None:
    """Render a human-readable markdown summary of what a workflow does.

    Shown right after the Architect registers a new workflow so the user
    understands the pipeline: its purpose, ordered steps, and each step's role.
    """
    from rich.markdown import Markdown

    lines: list[str] = [f"# {pkg.name}"]
    if pkg.description:
        lines.append(pkg.description)

    # Order nodes by dependency depth so the steps read top-to-bottom.
    ordered = _ordered_nodes(pkg)

    lines.append("\n## Pipeline")
    for i, node in enumerate(ordered, 1):
        kind_label = {
            "function": "Python step",
            "python": "Python step",
            "tool": "tool call",
            "react": "reasoning agent",
            "base": "LLM step",
        }.get(node.kind, node.kind)
        purpose = (node.purpose or node.goal or node.description or "").strip()
        # Keep each step to its first sentence / a sane length.
        if purpose:
            purpose = " ".join(purpose.split())
            if len(purpose) > 240:
                purpose = purpose[:240].rstrip() + "…"
        header = f"{i}. **{node.id}** _({kind_label})_"
        lines.append(header if not purpose else f"{header} — {purpose}")

    if pkg.graph.inputs:
        req = [i.name for i in pkg.graph.inputs if i.required]
        opt = [i.name for i in pkg.graph.inputs if not i.required]
        parts = []
        if req:
            parts.append("required: " + ", ".join(f"`{n}`" for n in req))
        if opt:
            parts.append("optional: " + ", ".join(f"`{n}`" for n in opt))
        lines.append("\n**Inputs** — " + "; ".join(parts))
    if pkg.graph.outputs:
        lines.append("**Output** — " + ", ".join(f"`{o}`" for o in pkg.graph.outputs))

    console.print()
    console.print(Markdown("\n".join(lines)))


def _ordered_nodes(pkg: WorkflowPackage) -> list:
    """Return nodes in dependency order (Kahn's algorithm; stable on ties)."""
    nodes = list(pkg.graph.nodes)
    by_id = {n.id: n for n in nodes}
    indeg = {n.id: len([d for d in n.depends_on if d in by_id]) for n in nodes}
    ordered: list = []
    ready = [n.id for n in nodes if indeg[n.id] == 0]
    seen: set[str] = set()
    while ready:
        nid = ready.pop(0)
        if nid in seen:
            continue
        seen.add(nid)
        ordered.append(by_id[nid])
        for n in nodes:
            if nid in n.depends_on:
                indeg[n.id] -= 1
                if indeg[n.id] <= 0 and n.id not in seen:
                    ready.append(n.id)
    # Append any nodes left out by a cycle, preserving declaration order.
    for n in nodes:
        if n.id not in seen:
            ordered.append(n)
    return ordered


def print_execution_result(
    console: Console,
    result: GraphExecutionResult,
    *,
    show_outputs: bool = True,
) -> None:
    from rich.table import Table

    table = Table(show_header=True, show_lines=False)
    table.add_column("Node", style=theme.ACCENT_DIM, no_wrap=True)
    table.add_column("Status", justify="center")
    table.add_column("Duration", justify="right")

    for node_id, nr in result.nodes.items():
        if nr.error:
            status = f"[{theme.ERR}]✗ error[/{theme.ERR}]"
        else:
            status = f"[{theme.OK if hasattr(theme, 'OK') else 'green'}]✓ ok[/]"
        duration = f"{nr.duration_ms / 1000:.2f}s"
        table.add_row(node_id, status, duration)

    console.print(table)

    if show_outputs and result.final:
        console.print(f"\n[bold {theme.ACCENT}]Final outputs[/]")
        for key, value in result.final.items():
            console.print(f"[{theme.DIM}]{key}:[/{theme.DIM}]")
            text = str(value)
            if len(text) > 2000:
                text = text[:2000] + " …"
            console.print(text)
