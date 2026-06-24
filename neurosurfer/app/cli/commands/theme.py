"""/theme — pick a color theme from the arrow-key menu."""

from __future__ import annotations

from .. import theme as _theme
from ..context import CLIContext
from ..io import select_menu
from .base import SlashCommand

_LABELS: dict[str, str] = {
    "chartreuse2":  "chartreuse2   — bright green",
    "bright_cyan":  "bright_cyan   — electric blue-teal",
    "gold3":        "gold3         — warm yellow-gold",
    "dark_orange3": "dark_orange3  — deep orange",
    "magenta2":     "magenta2      — vivid purple-pink",
    "grey74":       "grey74        — neutral silver",
}


async def _theme_cmd(ctx: CLIContext, args: list[str]) -> None:
    # Direct argument: /theme gold3
    if args:
        name = args[0].lower()
        if name not in _theme.THEMES:
            names = ", ".join(_theme.THEMES)
            ctx.console.print(
                f"[{_theme.ERR}]Unknown theme '{name}'.[/{_theme.ERR}] "
                f"[{_theme.DIM}]Options: {names}[/{_theme.DIM}]"
            )
            return
        await _apply(ctx, name)
        return

    # Interactive menu — current theme listed first, rest follow
    current = _theme.current().name
    ordered = [current] + [n for n in _theme.THEMES if n != current]
    options = [(name, _LABELS[name] + ("  ✓ current" if name == current else ""))
               for name in ordered]

    picked = await select_menu(ctx.console, "Choose a theme", options)
    if picked is None or picked == current:
        return
    await _apply(ctx, picked)


async def _apply(ctx: CLIContext, name: str) -> None:
    from ..banner import print_banner

    _theme.set_theme(name)
    ctx._save_session()
    ctx.console.clear()
    await print_banner(ctx)
    ctx.console.print(
        f"[{_theme.OK}]✓[/{_theme.OK}] Theme set to "
        f"[{_theme.ACCENT}]{name}[/{_theme.ACCENT}]"
    )


COMMAND = SlashCommand(
    name="theme",
    summary="Change the color theme",
    handler=_theme_cmd,
)
