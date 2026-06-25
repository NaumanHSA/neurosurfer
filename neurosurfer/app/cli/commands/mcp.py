"""/mcp — manage Model Context Protocol servers (list / add / remove / tools).

Configured servers persist via :class:`~neurosurfer.config.mcp.McpStore`. The live
connection is owned by :attr:`CLIContext.mcp`; mutating commands reconnect inside the
REPL task so the running agent immediately sees the new tool set.
"""

from __future__ import annotations

from neurosurfer.config.mcp import McpServerConfig

from .. import theme
from ..context import CLIContext
from ..io import _ainput, select_menu
from .base import SlashCommand

_USAGE = (
    "Usage: /mcp [list | add | remove <name> | enable <name> | "
    "disable <name> | tools [name] | reconnect]"
)


async def _prompt(label: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    raw = await _ainput(f"  {label}{suffix}: ")
    return raw.strip() or default


def _print_list(ctx: CLIContext) -> None:
    servers = ctx.mcp_store.list()
    if not servers:
        ctx.console.print(
            f"[{theme.DIM}]No MCP servers configured. Add one with /mcp add.[/{theme.DIM}]"
        )
        return
    live = {s.name: s for s in (ctx.mcp.status() if ctx.mcp else [])}
    ctx.console.print(f"[bold {theme.ACCENT}]MCP servers[/bold {theme.ACCENT}]")
    for s in servers:
        st = live.get(s.name)
        if st is None:
            dot, color = "○", theme.DIM
        elif st.connected:
            dot, color = "●", theme.OK_DIM
        else:
            dot, color = "✗", theme.WARN
        tail = f"  ({st.tool_count} tools)" if (st and st.connected) else ""
        ctx.console.print(f"  [{color}]{dot} {s.summary()}{tail}[/{color}]")


def _print_tools(ctx: CLIContext, name: str | None) -> None:
    if ctx.mcp is None:
        ctx.console.print(f"[{theme.DIM}]No MCP servers connected.[/{theme.DIM}]")
        return
    for st in ctx.mcp.status():
        if name and st.name != name:
            continue
        if not st.connected:
            ctx.console.print(f"[{theme.WARN}]{st.name}: not connected ({st.error})[/{theme.WARN}]")
            continue
        ctx.console.print(f"[bold {theme.ACCENT}]{st.name}[/bold {theme.ACCENT}] — {st.tool_count} tools")
        for t in st.tools:
            ctx.console.print(f"  [{theme.DIM}]· {t}[/{theme.DIM}]")


async def _add(ctx: CLIContext) -> None:
    name = await _prompt("Name")
    if not name:
        ctx.console.print(f"[{theme.ERR}]Name is required.[/{theme.ERR}]")
        return
    if ctx.mcp_store.get(name) is not None:
        ctx.console.print(f"[{theme.ERR}]A server named '{name}' already exists.[/{theme.ERR}]")
        return
    transport = await select_menu(
        ctx.console, "Transport", [("stdio", "stdio (local child process)"), ("http", "streamable HTTP")]
    )
    if transport is None:
        return
    cfg: McpServerConfig
    if transport == "stdio":
        command = await _prompt("Command (e.g. npx)")
        args_raw = await _prompt("Args (space-separated)")
        cfg = McpServerConfig(
            name=name, transport="stdio", command=command, args=args_raw.split() if args_raw else []
        )
    else:
        url = await _prompt("URL")
        cfg = McpServerConfig(name=name, transport="http", url=url)
    try:
        ctx.mcp_store.add(cfg)
    except ValueError as e:
        ctx.console.print(f"[{theme.ERR}]{e}[/{theme.ERR}]")
        return
    ctx.console.print(f"[{theme.OK}]✓[/{theme.OK}] Added MCP server '{name}'.")
    await _reconnect(ctx)


async def _reconnect(ctx: CLIContext) -> None:
    await ctx.close_mcp()
    await ctx.setup_mcp()


async def handle(ctx: CLIContext, args: list[str]) -> None:
    sub = args[0] if args else "list"
    rest = args[1:]

    if sub == "list":
        _print_list(ctx)
    elif sub == "add":
        await _add(ctx)
    elif sub == "tools":
        _print_tools(ctx, rest[0] if rest else None)
    elif sub == "reconnect":
        await _reconnect(ctx)
        _print_list(ctx)
    elif sub in ("remove", "enable", "disable"):
        if not rest:
            ctx.console.print(f"[{theme.ERR}]/mcp {sub} needs a server name.[/{theme.ERR}]")
            return
        name = rest[0]
        try:
            if sub == "remove":
                ctx.mcp_store.delete(name)
                ctx.console.print(f"[{theme.OK}]✓[/{theme.OK}] Removed '{name}'.")
            else:
                ctx.mcp_store.set_enabled(name, sub == "enable")
                ctx.console.print(f"[{theme.OK}]✓[/{theme.OK}] {sub.capitalize()}d '{name}'.")
        except KeyError as e:
            ctx.console.print(f"[{theme.ERR}]{e}[/{theme.ERR}]")
            return
        await _reconnect(ctx)
    else:
        ctx.console.print(f"[{theme.DIM}]{_USAGE}[/{theme.DIM}]")


COMMAND = SlashCommand(
    name="mcp",
    summary="Manage MCP servers (list, add, remove, tools)",
    handler=handle,
)
