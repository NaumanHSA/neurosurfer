"""Small built-in slash commands: /help, /doctor, /status, /clear, /exit."""

from __future__ import annotations

from .. import theme
from ..context import CLIContext
from .base import CommandRegistry, SlashCommand


def _hard_clear(ctx: CLIContext) -> None:
    """Clear the visible screen AND the terminal scrollback buffer.

    Emits the exact byte sequence the ncurses ``clear`` command produces on
    xterm/VTE terminals with the E3 capability:

        \033[H   home the cursor
        \033[2J  clear the visible screen
        \033[3J  erase scrollback (saved lines)

    Order is what matters — this is the sequence `tput clear` actually emits.
    Getting it wrong (e.g. 3J first, or 2J before H) makes VTE/gnome-terminal
    scroll the old screen *into* scrollback instead of discarding it, which
    shows up as duplicated text plus a scrollbar.  We write it atomically to
    the same file Rich uses and flush so it lands before the reprinted banner.
    """
    f = ctx.console.file
    if not (hasattr(f, "isatty") and f.isatty()):
        return
    f.write("\033[H\033[2J\033[3J")
    f.flush()


async def _reset_screen(ctx: CLIContext) -> None:
    """Clear everything and reprint the full startup screen (banner + status)."""
    from neurosurfer import __version__
    from neurosurfer.app.banner import print_startup_banner

    from ..banner import print_banner

    _hard_clear(ctx)
    print_startup_banner(__version__)
    await print_banner(ctx)


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
    await _reset_screen(ctx)


async def _new(ctx: CLIContext, args: list[str]) -> None:
    from ..assistant import clear_assistant

    clear_assistant(ctx)
    await _reset_screen(ctx)
    ctx.console.print(f"[{theme.OK}]✓[/{theme.OK}] New session started — chat history cleared.")


async def _exit(ctx: CLIContext, args: list[str]) -> None:
    ctx.should_exit = True


def register_misc(registry: CommandRegistry) -> None:
    registry.register(SlashCommand("help", "Show available commands", _help, aliases=["h", "?"]))
    registry.register(SlashCommand("doctor", "Check the active connection", _doctor))
    registry.register(SlashCommand("status", "Show provider + task status", _status))
    registry.register(SlashCommand("clear", "Clear the screen", _clear, aliases=["cls"]))
    registry.register(SlashCommand("new", "Clear chat history and start a fresh session", _new))
    registry.register(SlashCommand("exit", "Leave neurosurfer", _exit, aliases=["quit", "q"]))
