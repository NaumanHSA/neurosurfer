"""The neurosurfer REPL.

Startup flow:
  banner → provider check → prompt loop

Input handling:
  /command args  → slash-command dispatcher
  <free text>    → routed to the general-purpose Assistant agent

The Assistant handles everything from quick one-liners to file/shell/web
automation.  Heavyweight recurring tasks can be escalated to the Architect
(/workflow build) which designs and registers a reusable workflow pipeline.
"""

from __future__ import annotations

import asyncio

from neurosurfer.config import Config
from . import theme
from .banner import print_banner, status_summary
from .commands import build_registry
from .completer import SlashCompleter
from .context import CLIContext


async def _maybe_confirm_default_provider(ctx: CLIContext) -> None:
    """Ask once at startup which provider is the default when 2+ are configured."""
    if ctx.providers.is_default_confirmed():
        return
    from .io import select_menu

    profiles = ctx.providers.list()
    if len(profiles) < 2:
        return
    active = ctx.providers.active_name()
    options = [(p.name, p.summary(active=p.name == active)) for p in profiles]
    choice = await select_menu(ctx.console, "Multiple providers configured — pick your default", options)
    if choice is None:
        return
    ctx.providers.confirm_default(choice)
    ctx.console.print(
        f"[{theme.OK}]✓[/{theme.OK}] Default provider set to '{choice}'. "
        f"Change anytime with /provider use."
    )


async def _dispatch_command(ctx: CLIContext, registry, line: str) -> None:
    parts = line[1:].split()
    if not parts:
        return
    name, args = parts[0], parts[1:]
    cmd = registry.get(name)
    if cmd is None:
        ctx.console.print(f"[{theme.DIM}]Unknown command /{name}. Type /help.[/{theme.DIM}]")
        return
    await cmd.handler(ctx, args)


def _print_rule(ctx: CLIContext) -> None:
    from rich.rule import Rule
    ctx.console.print(Rule(style=theme.DIM))


async def run_repl(cfg: Config) -> int:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.formatted_text import FormattedText
    from prompt_toolkit.history import InMemoryHistory
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.patch_stdout import patch_stdout
    from prompt_toolkit.styles import DynamicStyle

    ctx = CLIContext.create(cfg)
    registry = build_registry()
    ctx._extra["registry"] = registry

    await print_banner(ctx)
    await _maybe_confirm_default_provider(ctx)
    await ctx.setup_mcp()

    kb = KeyBindings()

    @kb.add("escape")
    def _(event) -> None:  # type: ignore[no-untyped-def]
        event.app.current_buffer.reset()

    session: PromptSession = PromptSession(
        completer=SlashCompleter(registry),
        complete_while_typing=True,
        history=InMemoryHistory(),
        style=DynamicStyle(theme.prompt_style),
        bottom_toolbar=lambda: FormattedText([("class:bottom-toolbar", " " + status_summary(ctx))]),
        key_bindings=kb,
    )

    while not ctx.should_exit:
        try:
            ctx.console.print()
            ctx.console.print(
                f"[italic {theme.DIM}]  Ask me to do something, "
                f"or type a slash command like /help or /workflow list."
                f"[/italic {theme.DIM}]"
            )
            _print_rule(ctx)
            with patch_stdout():
                line = await session.prompt_async(FormattedText([("class:prompt", "  > ")]))
        except (EOFError, KeyboardInterrupt):
            break

        line = line.strip()
        if not line:
            continue

        _print_rule(ctx)

        if line.startswith("/"):
            await _dispatch_command(ctx, registry, line)
            continue

        # Free-form text — route to the general-purpose Assistant.
        try:
            from .assistant import _assist
            await _assist(ctx, line)
        except (KeyboardInterrupt, asyncio.CancelledError):
            ctx.console.print(f"\n[{theme.WARN}]Interrupted.[/{theme.WARN}]")
        except Exception as exc:  # noqa: BLE001
            brief = str(exc)[:200].replace("\n", " ")
            ctx.console.print(f"[{theme.ERR}]Error: {brief}[/{theme.ERR}]")

    await ctx.close_mcp()
    ctx.console.print(f"\n[{theme.DIM}]Goodbye.[/{theme.DIM}]")
    return 0
