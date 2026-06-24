"""Shared CLI theme — accent color, semantic colors, prompt_toolkit styles.

Six named themes; switch at runtime with /theme. The module exposes mutable
module-level globals (ACCENT, ACCENT_DIM) so all callers that do
``from . import theme`` and then ``theme.ACCENT`` pick up changes instantly.
Prompt-toolkit styles are rebuilt from the current theme on each call.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Theme:
    name: str
    # Rich color names
    accent: str       # banner, panel borders, headings, /help command names
    accent_dim: str   # spinner labels, status panel border, model name, tagline
    # prompt_toolkit color tokens
    pt_prompt: str    # "You ⟩" prompt + _ainput "❯"
    pt_menu_bg: str   # hex bg for completion dropdown selected row
    pt_select: str    # arrow-key selector "❯ item" style
    pt_scroll: str    # hex for scrollbar button


THEMES: dict[str, Theme] = {
    "chartreuse2": Theme(
        name="chartreuse2",
        accent="chartreuse2",
        accent_dim="dark_olive_green1",
        pt_prompt="ansibrightgreen",
        pt_menu_bg="#5f8700",
        pt_select="bold ansibrightgreen",
        pt_scroll="#5fd700",
    ),
    "bright_cyan": Theme(
        name="bright_cyan",
        accent="bright_cyan",
        accent_dim="cyan",
        pt_prompt="ansibrightcyan",
        pt_menu_bg="#008787",
        pt_select="bold ansibrightcyan",
        pt_scroll="#00afaf",
    ),
    "gold3": Theme(
        name="gold3",
        accent="gold3",
        accent_dim="dark_goldenrod",
        pt_prompt="ansibrightyellow",
        pt_menu_bg="#875f00",
        pt_select="bold ansibrightyellow",
        pt_scroll="#c7a800",
    ),
    "dark_orange3": Theme(
        name="dark_orange3",
        accent="dark_orange3",
        accent_dim="orange3",
        pt_prompt="ansibrightred",
        pt_menu_bg="#875f00",
        pt_select="bold ansibrightred",
        pt_scroll="#af5f00",
    ),
    "magenta2": Theme(
        name="magenta2",
        accent="magenta2",
        accent_dim="magenta",
        pt_prompt="ansibrightmagenta",
        pt_menu_bg="#870087",
        pt_select="bold ansibrightmagenta",
        pt_scroll="#af00af",
    ),
    "grey74": Theme(
        name="grey74",
        accent="grey74",
        accent_dim="grey62",
        pt_prompt="ansiwhite",
        pt_menu_bg="#444444",
        pt_select="bold ansiwhite",
        pt_scroll="#767676",
    ),
}

DEFAULT_THEME = "chartreuse2"

# ── mutable runtime state ─────────────────────────────────────────────────────

_current: Theme = THEMES[DEFAULT_THEME]

# Semantic colors — constant across all themes (they carry meaning)
OK = "bold green"
OK_DIM = "green"
WARN = "yellow"
ERR = "bold red"
DIM = "dim"

# Accent aliases — mutated by set_theme() so callers always see the live value
ACCENT: str = _current.accent
ACCENT_DIM: str = _current.accent_dim


# ── public API ────────────────────────────────────────────────────────────────

def current() -> Theme:
    return _current


def set_theme(name: str) -> Theme:
    """Apply a theme by name; updates module-level ACCENT / ACCENT_DIM immediately."""
    global _current, ACCENT, ACCENT_DIM
    t = THEMES[name]
    _current = t
    ACCENT = t.accent
    ACCENT_DIM = t.accent_dim
    return t


def prompt_style():
    """prompt_toolkit Style for the main PromptSession (rebuilt each call)."""
    from prompt_toolkit.styles import Style

    t = _current
    return Style.from_dict({
        "prompt":                                   f"bold {t.pt_prompt}",
        "completion-menu.completion":               "bg:#1c1c1c #aaaaaa",
        "completion-menu.completion.current":       f"bg:{t.pt_menu_bg} #ffffff bold",
        "completion-menu.meta.completion":          "bg:#1c1c1c #6c6c6c",
        "completion-menu.meta.completion.current":  f"bg:{t.pt_menu_bg} #ffffff",
        "scrollbar.background":                     "bg:#3a3a3a",
        "scrollbar.button":                         f"bg:{t.pt_scroll}",
        "bottom-toolbar":                           f"bg:#1a1a1a {t.pt_prompt}",
    })


def select_style():
    """prompt_toolkit Style for the arrow-key inline selector (rebuilt each call)."""
    from prompt_toolkit.styles import Style

    t = _current
    return Style.from_dict({
        "selected": t.pt_select,
        "choice":   "",
        "hint":     "ansidarkgray italic",
    })
