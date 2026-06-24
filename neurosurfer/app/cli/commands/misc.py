"""Small built-in slash commands: /help, /doctor, /status, /clear, /exit."""

from __future__ import annotations

from .. import theme
from ..context import CLIContext
from .base import CommandRegistry, SlashCommand


async def _help(ctx: CLIContext, args: list[str]) -> None:
    from rich.table import Table

    registry: CommandRegistry = ctx._extra["registry"]
    table = Table(show_header=False, box=None, title="Commands", title_style=f"bold {theme.ACCENT}")
    table.add_column(style=theme.ACCENT, no_wrap=True)
    table.add_column()
    for cmd in registry.all():
        table.add_row(f"/{cmd.name}", cmd.summary)
    ctx.console.print(table)
    ctx.console.print(f"[{theme.DIM}]Describe what you want to automate, or type '/' for command suggestions.[/{theme.DIM}]")


async def _doctor(ctx: CLIContext, args: list[str]) -> None:
    from ..doctor import check_reachable

    ok, msg = await check_reachable(ctx)
    mark = f"[{theme.OK}]✓[/{theme.OK}]" if ok else f"[{theme.ERR}]✗[/{theme.ERR}]"
    ctx.console.print(f"{mark} {msg}")


async def _status(ctx: CLIContext, args: list[str]) -> None:
    from ..banner import print_status

    print_status(ctx)


async def _clear(ctx: CLIContext, args: list[str]) -> None:
    ctx.console.clear()
    from ..banner import print_banner

    await print_banner(ctx)


async def _exit(ctx: CLIContext, args: list[str]) -> None:
    ctx.should_exit = True


def register_misc(registry: CommandRegistry) -> None:
    registry.register(SlashCommand("help", "Show available commands", _help, aliases=["h", "?"]))
    registry.register(SlashCommand("doctor", "Check the active connection", _doctor))
    registry.register(SlashCommand("status", "Show provider + task status", _status))
    registry.register(SlashCommand("clear", "Clear the screen", _clear, aliases=["cls"]))
    registry.register(SlashCommand("exit", "Leave neurosurfer", _exit, aliases=["quit", "q"]))
