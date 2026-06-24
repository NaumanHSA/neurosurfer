"""/task — manage and run Tasks (list / show / use / run / delete / new)."""

from __future__ import annotations

from neurosurfer.tasks.definition import TaskDefinition
from neurosurfer.tasks.registry import TaskNotFoundError, TaskRegistry
from .. import theme
from ..context import CLIContext
from ..io import select_menu
from .base import SlashCommand


# ── pure operations (unit-tested) ─────────────────────────────────────────────
def op_delete(registry: TaskRegistry, name: str) -> str:
    registry.delete(name)
    return f"Deleted task '{name}'."


def op_clone(registry: TaskRegistry, src: str, dst: str) -> str:
    registry.clone(src, dst)
    return f"Cloned '{src}' → '{dst}' (editable user task)."


def op_set_provider(task_providers, name: str, provider_name: str | None) -> str:
    """Pin (or unpin) which provider profile runs Task ``name``.

    Stored separately from the Task YAML so this works even on a protected
    readonly/system Task (e.g. the built-in ``code``/``general``).
    """
    if provider_name is None or provider_name.lower() == "default":
        task_providers.unset(name)
        return f"Task '{name}' now uses the default provider."
    task_providers.set(name, provider_name)
    return f"Task '{name}' is now pinned to provider '{provider_name}'."


# ── rendering ─────────────────────────────────────────────────────────────────
def _kind_label(td: TaskDefinition) -> str:
    """Compact kind badge; protected (readonly/system) tasks are marked locked."""
    if td.is_protected:
        return f"[{theme.ACCENT_DIM}]{td.kind} (locked)[/{theme.ACCENT_DIM}]"
    return f"[{theme.DIM}]user[/{theme.DIM}]"


def _print_list(ctx: CLIContext) -> None:
    from rich.table import Table

    names = ctx.registry.list(include_hidden=False)
    if not names:
        ctx.console.print(f"[{theme.DIM}]No tasks registered. Use /task new to author one.[/{theme.DIM}]")
        return
    table = Table(title="Tasks", show_lines=False, title_style=f"bold {theme.ACCENT}")
    table.add_column("", width=2)
    table.add_column("Name", style=theme.ACCENT_DIM, no_wrap=True)
    table.add_column("Description")
    table.add_column("Kind", justify="center")
    for name in names:
        try:
            td = ctx.registry.get(name)
            mark = "●" if name == ctx.active_task else " "
            table.add_row(mark, td.name, td.description or "—", _kind_label(td))
        except Exception as e:  # noqa: BLE001
            table.add_row(" ", name, f"[{theme.ERR}]parse error: {e}[/{theme.ERR}]", "")
    ctx.console.print(table)


def _show(ctx: CLIContext, name: str, *, full: bool = False) -> None:
    from rich.markup import escape
    from rich.panel import Panel

    try:
        td = ctx.registry.get(name)
    except TaskNotFoundError:
        ctx.console.print(f"[{theme.ERR}]Task '{name}' not found.[/{theme.ERR}]")
        return

    lock = f" [{theme.ACCENT_DIM}](locked)[/{theme.ACCENT_DIM}]" if td.is_protected else ""
    ctx.console.print()
    ctx.console.print(
        f"[bold {theme.ACCENT}]{escape(td.name)}[/bold {theme.ACCENT}] v{td.version}  "
        f"[{theme.DIM}]{td.kind}[/{theme.DIM}]{lock}"
    )
    if td.description:
        ctx.console.print(f"[{theme.DIM}]{escape(td.description)}[/{theme.DIM}]")

    path = ctx.registry.path_for(name)
    if path is not None:
        ctx.console.print(f"[{theme.ACCENT_DIM}]YAML:[/{theme.ACCENT_DIM}]  {escape(str(path))}")

    ctx.console.print(f"\n[{theme.ACCENT_DIM}]Tools:[/{theme.ACCENT_DIM}]  "
                      + (", ".join(td.tools) if td.tools else "[dim]all[/dim]"))

    if td.inputs:
        ctx.console.print(f"[{theme.ACCENT_DIM}]Inputs:[/{theme.ACCENT_DIM}]")
        for inp in td.inputs:
            req = "[bold red]*[/bold red]" if inp.required else ""
            ctx.console.print(f"  {req} [bold]{escape(inp.name)}[/bold] [{escape(inp.type)}]"
                              + (f"  — {escape(inp.prompt)}" if inp.prompt else ""))

    g = td.guardrails
    ctx.console.print(
        f"[{theme.ACCENT_DIM}]Guardrails:[/{theme.ACCENT_DIM}]  "
        f"shell={g.shell_policy}  max_turns={g.max_turns}"
        + (f"  write_scope={g.write_scope}" if g.write_scope else "")
    )
    ctx.console.print(f"[{theme.ACCENT_DIM}]Plan required:[/{theme.ACCENT_DIM}]  {td.plan_required}")
    pinned = ctx.task_providers.get(td.name)
    prov_label = f"{escape(pinned)} (pinned)" if pinned else "default — whatever /provider is active"
    ctx.console.print(f"[{theme.ACCENT_DIM}]Provider:[/{theme.ACCENT_DIM}]  {prov_label}")

    sp = td.system_prompt
    if full or len(sp) <= 500:
        body = escape(sp)
    else:
        body = escape(sp[:500]) + (
            f"\n[{theme.DIM}]… ({len(sp)} chars truncated — run "
            f"`/task show {name} full`, or open the YAML above)[/{theme.DIM}]"
        )
    ctx.console.print(f"\n[{theme.ACCENT_DIM}]System prompt:[/{theme.ACCENT_DIM}]")
    ctx.console.print(Panel(body, border_style=theme.DIM, padding=(0, 1)))
    ctx.console.print()


# ── interactive menu (arrow keys via the shared selector) ─────────────────────
async def _pick_task(ctx: CLIContext, title: str) -> str | None:
    names = ctx.registry.list(include_hidden=False)
    if not names:
        ctx.console.print(f"[{theme.DIM}]No tasks registered. Use 'New task' to author one.[/{theme.DIM}]")
        return None
    options: list[tuple[str, str]] = []
    for name in names:
        active = " ●" if name == ctx.active_task else ""
        try:
            tag = "  (locked)" if ctx.registry.get(name).is_protected else ""
        except Exception:  # noqa: BLE001
            tag = ""
        options.append((name, f"{name}{active}{tag}"))
    return await select_menu(ctx.console, title, options)


async def _pick_provider(ctx: CLIContext, title: str) -> str | None:
    """Arrow-key pick of a provider profile, plus a 'Use default' clear option."""
    profiles = ctx.providers.list()
    if not profiles:
        ctx.console.print(
            f"[{theme.DIM}]No provider profiles configured yet. Add one with /provider add.[/{theme.DIM}]"
        )
        return None
    active = ctx.providers.active_name()
    options = [("default", "Use default (whatever /provider is active)")]
    options += [(p.name, p.summary(active=p.name == active)) for p in profiles]
    return await select_menu(ctx.console, title, options)


async def _set_provider(ctx: CLIContext) -> None:
    name = await _pick_task(ctx, "Set provider for which task?")
    if not name:
        return
    choice = await _pick_provider(ctx, f"Provider for '{name}'")
    if choice is None:
        return
    ctx.console.print(f"[{theme.OK}]✓[/{theme.OK}] {op_set_provider(ctx.task_providers, name, choice)}")


async def _interactive_menu(ctx: CLIContext) -> None:
    from ..io import InputCancelled

    while True:
        _print_list(ctx)
        choice = await select_menu(ctx.console, "Task manager", [
            ("use",      "Set active task", "Switch the agent to a different task profile"),
            ("run",      "Run a task",      "Start a full guided session for a task"),
            ("show",     "Show a task",     "Inspect system prompt, tools, guardrails, and inputs"),
            ("provider", "Set provider",    "Pin a task to a specific provider profile (or clear it)"),
            ("new",      "New task",        "Author a new task interactively"),
            ("clone",    "Clone a task",    "Copy a task into a new editable user task"),
            ("delete",   "Delete a task",   "Permanently remove a task YAML"),
            ("done",     "Done",            "Return to the chat prompt"),
        ])
        if choice in (None, "done"):
            return
        try:
            if choice == "use":
                name = await _pick_task(ctx, "Set active task")
                if name:
                    ctx.registry.get(name)  # validates existence
                    ctx.active_task = name
                    ctx.console.print(f"[{theme.OK}]✓[/{theme.OK}] Active task is now '{name}'.")
            elif choice == "provider":
                await _set_provider(ctx)
            elif choice == "run":
                name = await _pick_task(ctx, "Run a task")
                if name:
                    from ..run import run_task

                    await run_task(ctx, ctx.registry.get(name))
                    return  # don't re-show the menu after a task completes
            elif choice == "show":
                name = await _pick_task(ctx, "Show a task")
                if name:
                    _show(ctx, name)
            elif choice == "new":
                await _new_task(ctx)
            elif choice == "clone":
                src = await _pick_task(ctx, "Clone which task?")
                if src:
                    from ..io import _ainput

                    dst = (await _ainput("New task name: ")).strip()
                    if dst:
                        ctx.console.print(
                            f"[{theme.OK}]✓[/{theme.OK}] {op_clone(ctx.registry, src, dst)}"
                        )
            elif choice == "delete":
                name = await _pick_task(ctx, "Delete a task")
                if name:
                    ctx.console.print(f"[{theme.OK}]✓[/{theme.OK}] {op_delete(ctx.registry, name)}")
                    if ctx.active_task == name:
                        ctx.active_task = None
        except InputCancelled:
            ctx.console.print(f"[{theme.DIM}]  ↩  cancelled[/{theme.DIM}]")
        except TaskNotFoundError as e:
            ctx.console.print(f"[{theme.ERR}]{e}[/{theme.ERR}]")
        except Exception as e:  # noqa: BLE001
            ctx.console.print(f"[{theme.ERR}]{e}[/{theme.ERR}]")


# ── command handler ───────────────────────────────────────────────────────────
async def handle(ctx: CLIContext, args: list[str]) -> None:
    if not args:
        await _interactive_menu(ctx)
        return
    sub, *rest = args
    try:
        if sub == "list":
            _print_list(ctx)
        elif sub == "show" and rest:
            _show(ctx, rest[0], full=len(rest) > 1 and rest[1].lower() in ("full", "-f", "--full"))
        elif sub == "use" and rest:
            ctx.registry.get(rest[0])  # validates existence
            ctx.active_task = rest[0]
            ctx.console.print(f"[{theme.OK}]✓[/{theme.OK}] Active task is now '{rest[0]}'.")
        elif sub == "provider" and len(rest) >= 2:
            ctx.registry.get(rest[0])  # validates existence
            ctx.console.print(
                f"[{theme.OK}]✓[/{theme.OK}] {op_set_provider(ctx.task_providers, rest[0], rest[1])}"
            )
        elif sub == "clone" and len(rest) >= 2:
            ctx.console.print(f"[{theme.OK}]✓[/{theme.OK}] {op_clone(ctx.registry, rest[0], rest[1])}")
        elif sub == "delete" and rest:
            ctx.console.print(f"[{theme.OK}]✓[/{theme.OK}] {op_delete(ctx.registry, rest[0])}")
            if ctx.active_task == rest[0]:
                ctx.active_task = None
        elif sub == "run" and rest:
            from ..run import run_task

            task = ctx.registry.get(rest[0])
            await run_task(ctx, task)
        elif sub == "new":
            await _new_task(ctx)
        else:
            ctx.console.print(
                f"[{theme.DIM}]Usage: /task "
                r"\[list|show <name> \[full]|use <name>|run <name>|provider <name> <profile|default>|"
                r"clone <src> <new>|delete <name>|new]"
                f"[/{theme.DIM}]"
            )
    except TaskNotFoundError:
        ctx.console.print(f"[{theme.ERR}]Task '{rest[0] if rest else ''}' not found.[/{theme.ERR}]")
    except Exception as e:  # noqa: BLE001
        ctx.console.print(f"[{theme.ERR}]{e}[/{theme.ERR}]")


async def _new_task(ctx: CLIContext) -> None:
    # Authoring is the task_builder meta-agent (Phase 8). Run it if registered.
    if "task_builder" in ctx.registry.list():
        from ..run import run_task

        await run_task(ctx, ctx.registry.get("task_builder"))
    else:
        ctx.console.print(f"[{theme.DIM}]Task authoring (task_builder) arrives in Phase 8. "
                          f"For now add a YAML under {ctx.cfg.tasks.dir}.[/{theme.DIM}]")


async def handle_run(ctx: CLIContext, args: list[str]) -> None:
    """/run <task> [k=v ...] — shortcut for /task run."""
    if not args:
        target = ctx.active_task
        if not target:
            ctx.console.print(f"[{theme.DIM}]Usage: /run <task>. Pick one with /task use or /task list.[/{theme.DIM}]")
            return
    else:
        target = args[0]
    overrides = {}
    for item in args[1:]:
        if "=" in item:
            k, _, v = item.partition("=")
            overrides[k.strip()] = v.strip()
    try:
        task = ctx.registry.get(target)
    except TaskNotFoundError:
        ctx.console.print(f"[{theme.ERR}]Task '{target}' not found.[/{theme.ERR}]")
        return
    from ..run import run_task

    await run_task(ctx, task, overrides or None)


COMMAND = SlashCommand(
    name="task",
    summary="List / show / use / run / delete / new Tasks",
    handler=handle,
    aliases=["tasks", "t"],
)

RUN_COMMAND = SlashCommand(
    name="run",
    summary="Run a task (shortcut for /task run)",
    handler=handle_run,
)
