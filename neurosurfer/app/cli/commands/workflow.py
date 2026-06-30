"""/workflow — manage and run registered Workflow packages.

Sub-commands
------------
list     Print all registered workflows.
show     Inspect a workflow's graph (nodes, inputs, outputs).
run      Collect declared inputs and execute a workflow.
delete   Remove a workflow from the registry.
build    Launch the Architect to design and register a new workflow.
"""

from __future__ import annotations

from .. import theme
from ..context import CLIContext
from ..io import select_menu
from .base import SlashCommand

# ── helpers ───────────────────────────────────────────────────────────────────

def _registry(ctx: CLIContext):
    from neurosurfer.graph.workflow.registry import WorkflowRegistry

    return WorkflowRegistry()


async def _pick_workflow(ctx: CLIContext, prompt: str) -> str | None:
    reg = _registry(ctx)
    names = reg.list()
    if not names:
        ctx.console.print(f"[{theme.DIM}]No workflows registered.[/{theme.DIM}]")
        return None
    options = [(n, n) for n in names]
    return await select_menu(ctx.console, prompt, options)


async def _collect_inputs(ctx: CLIContext, graph) -> dict | None:
    """Prompt the user for each declared graph input. Returns None on cancel."""
    from ..io import InputCancelled, _ainput

    inputs: dict = {}
    for inp in graph.inputs:
        req_marker = "[red]*[/red] " if inp.required else ""
        label = f"{inp.name} ({inp.type})"
        desc = f" — {inp.description}" if inp.description else ""
        ctx.console.print(f"  {req_marker}[bold]{label}[/bold]{desc}")
        try:
            raw = (await _ainput(f"{inp.name}: ")).strip()
        except (InputCancelled, KeyboardInterrupt, EOFError):
            return None
        if not raw:
            if inp.required:
                ctx.console.print(f"[{theme.ERR}]'{inp.name}' is required.[/{theme.ERR}]")
                return None
            continue
        inputs[inp.name] = raw

    return inputs


# ── sub-command handlers ──────────────────────────────────────────────────────

def _list(ctx: CLIContext) -> None:
    from ..render_workflow import print_workflow_list

    names = _registry(ctx).list()
    print_workflow_list(ctx.console, names)


def _show(ctx: CLIContext, name: str) -> None:
    from ..render_workflow import print_workflow_info

    reg = _registry(ctx)
    try:
        pkg = reg.get(name)
    except Exception as e:  # noqa: BLE001
        ctx.console.print(f"[{theme.ERR}]{e}[/{theme.ERR}]")
        return
    print_workflow_info(ctx.console, pkg)


def _delete(ctx: CLIContext, name: str) -> None:
    reg = _registry(ctx)
    try:
        reg.delete(name)
        ctx.console.print(f"[{theme.OK}]✓[/{theme.OK}] Deleted workflow '{name}'.")
    except Exception as e:  # noqa: BLE001
        ctx.console.print(f"[{theme.ERR}]{e}[/{theme.ERR}]")


async def _run(ctx: CLIContext, name: str) -> None:
    from neurosurfer.llm.registry import resolve_provider
    from neurosurfer.graph.workflow.registry import WorkflowNotFoundError, WorkflowRegistry
    from neurosurfer.graph.workflow.runner import WorkflowRunner
    from ..render_workflow import print_execution_result

    reg = WorkflowRegistry()

    try:
        pkg = reg.get(name)
    except WorkflowNotFoundError as e:
        ctx.console.print(f"[{theme.ERR}]{e}[/{theme.ERR}]")
        return

    try:
        provider = resolve_provider(ctx.cfg, ctx.providers)
    except RuntimeError as e:
        ctx.console.print(f"[{theme.ERR}]No usable provider: {e}[/{theme.ERR}]")
        ctx.console.print(f"[{theme.DIM}]Configure one with /provider, then try again.[/{theme.DIM}]")
        return

    # Collect inputs interactively
    if pkg.graph.inputs:
        ctx.console.print(f"\n[bold {theme.ACCENT}]Inputs for '{name}'[/]")
        inputs = await _collect_inputs(ctx, pkg.graph)
        if inputs is None:
            ctx.console.print(f"[{theme.DIM}]Cancelled.[/{theme.DIM}]")
            return
    else:
        inputs = {}

    ctx.console.print(f"\n[{theme.DIM}]Running workflow '{name}'…[/{theme.DIM}]")

    import time

    from neurosurfer.config.paths import traces_dir

    trace_path = traces_dir() / f"run-{name}-{int(time.time())}.json"

    runner = WorkflowRunner(provider)
    try:
        result = runner.run(pkg, inputs, trace_path=trace_path)
    except Exception as e:  # noqa: BLE001
        ctx.console.print(f"[{theme.ERR}]Workflow failed: {e}[/{theme.ERR}]")
        return

    ctx.console.print()
    print_execution_result(ctx.console, result)
    if trace_path.exists():
        ctx.console.print(f"\n[{theme.DIM}]Trace: {trace_path}[/{theme.DIM}]")


# Friendly progress labels for the Architect's own pipeline nodes.
_ARCHITECT_NODE_LABELS = {
    "discover": "Researching your request online",
    "clarify": "Reviewing your answers",
    "decompose": "Breaking the work into stages",
    "design_nodes": "Designing workflow nodes",
    "critique": "Reviewing and refining the design",
    "tool_design": "Working out the tools each node needs",
    "plan": "Designing the workflow",  # kept for packages built before Q1
    "write_nodes": "Authoring workflow nodes",
    "assemble": "Assembling & registering",
}


async def _build(ctx: CLIContext, intent: str) -> None:
    import asyncio

    from neurosurfer.llm.registry import resolve_provider
    from neurosurfer.architect import (
        ArchitectBuilder,
        ArchitectConversation,
        WorkflowInfeasible,
    )
    from ..io import RichIOHandler

    try:
        provider = resolve_provider(ctx.cfg, ctx.providers)
    except RuntimeError as e:
        ctx.console.print(f"[{theme.ERR}]No usable provider: {e}[/{theme.ERR}]")
        ctx.console.print(f"[{theme.DIM}]Configure one with /provider, then try again.[/{theme.DIM}]")
        return

    if not intent:
        ctx.console.print(
            f"[{theme.DIM}]Tip: describe what you want to build to get started.[/{theme.DIM}]"
        )
        return

    ctx.console.print(
        f"\n[bold {theme.ACCENT}]Architect[/] — I'll ask a few quick questions, "
        f"then design your workflow.\n"
    )

    # ── conversational requirement gathering (multiple-choice questions) ──────
    io_handler = RichIOHandler(ctx.console)

    async def ask(question: str, choices: list[str]) -> str:
        return await io_handler.ask(question, choices or None)

    def say(text: str) -> None:
        ctx.console.print(f"\n[bold {theme.ACCENT}]Architect:[/] {text}")

    convo = ArchitectConversation(provider)
    try:
        refined_intent, answers = await convo.run(intent, ask=ask, say=say)
    except (KeyboardInterrupt, asyncio.CancelledError):
        ctx.console.print(f"\n[{theme.DIM}]Cancelled.[/{theme.DIM}]")
        return
    except Exception as e:  # noqa: BLE001
        ctx.console.print(f"[{theme.ERR}]Architect error: {e}[/{theme.ERR}]")
        return

    # ── live build with per-node progress so it never looks stuck ─────────────
    ctx.console.print(f"\n[bold {theme.ACCENT}]Building your workflow[/]")

    def on_node_event(node_id: str, status: str) -> None:
        label = _ARCHITECT_NODE_LABELS.get(node_id, f"Step '{node_id}'")
        if status == "start":
            ctx.console.print(f"  [{theme.ACCENT_DIM}]→[/] {label}…")
        elif status == "ok":
            ctx.console.print(f"  [{theme.OK}]✓[/{theme.OK}] [{theme.DIM}]{label}[/{theme.DIM}]")
        elif status == "error":
            ctx.console.print(f"  [{theme.ERR}]✗ {label} failed[/{theme.ERR}]")
        elif status == "skipped":
            ctx.console.print(f"  [{theme.DIM}]⊘ {label} skipped[/{theme.DIM}]")

    def notify(msg: str) -> None:
        ctx.console.print(f"  [{theme.DIM}]{msg}[/{theme.DIM}]")

    builder = ArchitectBuilder(provider)
    approve_tool = _make_tool_approver(ctx, io_handler)
    try:
        registered_path = await builder.run(
            refined_intent,
            answers=answers,
            on_node_event=on_node_event,
            approve_tool=approve_tool,
            notify=notify,
        )
    except WorkflowInfeasible as e:
        # The tool_design step judged this not buildable as described — say so clearly,
        # rather than registering a workflow that would silently do the wrong thing.
        from rich.box import ROUNDED
        from rich.panel import Panel

        ctx.console.print(
            Panel(
                f"{e.report}",
                title=f"[bold {theme.WARN}]This isn't doable as described[/]",
                border_style=theme.WARN,
                box=ROUNDED,
            )
        )
        ctx.console.print(
            f"[{theme.DIM}]Adjust the request (e.g. supply the missing resource, or "
            f"scope it to something self-contained) and try again.[/{theme.DIM}]"
        )
        return
    except Exception as e:  # noqa: BLE001
        ctx.console.print(f"[{theme.ERR}]Build failed: {e}[/{theme.ERR}]")
        return

    ctx.console.print(
        f"\n[{theme.OK}]✓[/{theme.OK}] Workflow registered at "
        f"[bold]{registered_path}[/bold]"
    )

    # ── show the user exactly what the new pipeline does ──────────────────────
    _print_built_summary(ctx, registered_path)

    ctx.console.print(
        f"\n[{theme.DIM}]Run it anytime with /workflow run[/{theme.DIM}]"
    )


def _make_tool_approver(ctx: CLIContext, io_handler):
    """Build the async approval callback shown when the Architect authors a new tool.

    Renders the generated source + sandbox result, surfaces any risky-token warnings,
    and requires an explicit choice before the tool is registered.
    """
    async def approve(draft, result) -> bool:
        from rich.box import ROUNDED
        from rich.panel import Panel
        from rich.syntax import Syntax

        ctx.console.print(
            f"\n[bold {theme.WARN}]⚠ The workflow needs a tool that doesn't exist yet.[/]"
        )
        ctx.console.print(
            f"[{theme.DIM}]The Architect wrote and sandbox-tested a new tool "
            f"'[bold]{draft.name}[/bold]'. Review it before it's registered.[/{theme.DIM}]\n"
        )
        ctx.console.print(
            Panel(
                Syntax(draft.code, "python", theme="ansi_dark", word_wrap=True),
                title=f"[bold]{draft.name}[/bold]",
                border_style=theme.ACCENT_DIM,
                box=ROUNDED,
            )
        )
        ctx.console.print(f"\n[{theme.DIM}]Sandbox checks:[/{theme.DIM}]")
        ctx.console.print(result.render())
        if result.warnings:
            ctx.console.print(
                f"\n[{theme.WARN}]Heads up — this tool {', '.join(result.warnings)}.[/{theme.WARN}]"
            )

        choice = await io_handler.ask(
            f"Register the generated tool '{draft.name}'?",
            ["Yes — register and use it", "No — cancel the build"],
        )
        return isinstance(choice, str) and choice.lower().startswith("yes")

    return approve


def _print_built_summary(ctx: CLIContext, registered_path: str) -> None:
    """Load the freshly-registered package and render its pipeline summary."""
    from pathlib import Path

    from neurosurfer.graph.workflow.package import load_package
    from ..render_workflow import print_workflow_summary

    try:
        pkg = load_package(Path(registered_path))
    except Exception:  # noqa: BLE001 - summary is best-effort
        return
    print_workflow_summary(ctx.console, pkg)


async def _refine(ctx: CLIContext, name: str) -> None:
    """Run a workflow and self-heal failing nodes from their run-time errors (E7)."""
    from neurosurfer.llm.registry import resolve_provider
    from neurosurfer.architect.refine import WorkflowRefiner
    from neurosurfer.graph.workflow.registry import WorkflowNotFoundError, WorkflowRegistry

    reg = WorkflowRegistry()
    try:
        pkg = reg.get(name)
    except WorkflowNotFoundError as e:
        ctx.console.print(f"[{theme.ERR}]{e}[/{theme.ERR}]")
        return

    try:
        provider = resolve_provider(ctx.cfg, ctx.providers)
    except RuntimeError as e:
        ctx.console.print(f"[{theme.ERR}]No usable provider: {e}[/{theme.ERR}]")
        return

    if pkg.graph.inputs:
        ctx.console.print(f"\n[bold {theme.ACCENT}]Inputs for '{name}'[/]")
        inputs = await _collect_inputs(ctx, pkg.graph)
        if inputs is None:
            ctx.console.print(f"[{theme.DIM}]Cancelled.[/{theme.DIM}]")
            return
    else:
        inputs = {}

    ctx.console.print(f"\n[bold {theme.ACCENT}]Refining '{name}'[/] — running and healing failures…")

    def notify(msg: str) -> None:
        ctx.console.print(f"  [{theme.DIM}]{msg}[/{theme.DIM}]")

    refiner = WorkflowRefiner(provider)
    try:
        result = await refiner.refine(name, inputs, notify=notify)
    except Exception as e:  # noqa: BLE001
        ctx.console.print(f"[{theme.ERR}]Refine failed: {e}[/{theme.ERR}]")
        return

    if result.ok:
        patched = ", ".join(result.patched_nodes) if result.patched_nodes else "none"
        ctx.console.print(
            f"\n[{theme.OK}]✓[/{theme.OK}] '{name}' now runs cleanly "
            f"(rounds: {result.rounds}, patched: {patched})."
        )
    else:
        ctx.console.print(f"\n[{theme.ERR}]✗ Could not heal '{name}'.[/{theme.ERR}]")
        if result.message:
            ctx.console.print(f"[{theme.DIM}]{result.message}[/{theme.DIM}]")


# ── command dispatcher ────────────────────────────────────────────────────────

async def _menu(ctx: CLIContext) -> None:
    """Arrow-key action menu shown when /workflow is called with no sub-command."""
    options = [
        ("list", "List workflows", "show all registered workflows"),
        ("run", "Run a workflow", "pick one and provide its inputs"),
        ("show", "Show a workflow", "inspect a workflow's pipeline"),
        ("build", "Build a new workflow", "describe what you want to automate"),
        ("refine", "Refine a workflow", "run it and auto-heal failing nodes"),
        ("delete", "Delete a workflow", "remove one from the registry"),
    ]
    choice = await select_menu(ctx.console, "Workflows", options)
    if choice is None:
        return
    if choice == "build":
        from ..io import InputCancelled, _ainput

        try:
            intent = (await _ainput("Describe the workflow to build: ")).strip()
        except (InputCancelled, KeyboardInterrupt, EOFError):
            return
        await _build(ctx, intent)
    else:
        # Re-enter the dispatcher for the chosen sub-command (it handles its own
        # interactive picker when no name is given).
        await handle(ctx, [choice])


async def handle(ctx: CLIContext, args: list[str]) -> None:
    sub = args[0].lower() if args else ""

    if sub == "":
        await _menu(ctx)

    elif sub == "list":
        _list(ctx)

    elif sub == "show":
        if len(args) >= 2:
            _show(ctx, args[1])
        else:
            name = await _pick_workflow(ctx, "Show which workflow?")
            if name:
                _show(ctx, name)

    elif sub == "run":
        if len(args) >= 2:
            await _run(ctx, args[1])
        else:
            name = await _pick_workflow(ctx, "Run which workflow?")
            if name:
                await _run(ctx, name)

    elif sub == "delete":
        if len(args) >= 2:
            _delete(ctx, args[1])
        else:
            name = await _pick_workflow(ctx, "Delete which workflow?")
            if name:
                _delete(ctx, name)

    elif sub == "build":
        intent = " ".join(args[1:]).strip() if len(args) > 1 else ""
        await _build(ctx, intent)

    elif sub == "refine":
        if len(args) >= 2:
            await _refine(ctx, args[1])
        else:
            name = await _pick_workflow(ctx, "Refine which workflow?")
            if name:
                await _refine(ctx, name)

    else:
        ctx.console.print(
            f"[{theme.DIM}]Usage: /workflow [list|show|run|refine|delete|build] [name][/{theme.DIM}]"
        )


COMMAND = SlashCommand(
    name="workflow",
    summary="Build, manage, and run compiled workflow packages",
    handler=handle,
    aliases=["wf"],
)
