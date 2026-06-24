from __future__ import annotations

from typing import Any, Dict, Optional, List, Literal
import time
from pydantic import BaseModel


class TracerConfig(BaseModel):
    """
    Configuration options for Tracer.

    Attributes:
        enabled:
            If False, `step(...)` becomes a no-op. Your code can always call it.
        log_steps:
            If True, each step prints human-readable spans via `span_tracer`.
        max_output_preview_chars:
            When you store large outputs in `t.add(...)`, you may choose to
            also store a shortened preview in `outputs["preview"]`.
            This class itself doesn't enforce truncation — it's up to your
            usage — but the parameter is here for convenience / future use.
    """

    enabled: bool = True
    log_steps: bool = False
    max_output_preview_chars: int = 4000

    # pretty printing config
    indent_spaces: int = 4
    show_agent_banner: bool = True
    show_synthetic_phase_lines: bool = True

    show_inputs: bool = False
    show_outputs: bool = False
    max_preview_chars: int = 300

    show_logs: bool = True
    include_log_data: bool = False  # include log["data"] dict after message if present

    # cosmetic
    arrow_open: str = "▶"
    arrow_close: str = "◀"


RICH_LOG_TYPES_MAPPING = {
    # Core levels
    "info":     "[bold green]INFO: {message}[/bold green]",
    "warning":  "[bold yellow]WARNING: {message}[/bold yellow]",
    "error":    "[bold red]ERROR: {message}[/bold red]",
    "debug":    "[bold blue]DEBUG: {message}[/bold blue]",

    # Success / ok / done
    "success":  "[bold bright_green]SUCCESS: {message}[/bold bright_green]",
    "ok":       "[bright_green]OK: {message}[/bright_green]",

    # Trace / very low-level stuff (soft gray/white)
    "trace":    "[dim bright_black]TRACE: {message}[/dim bright_black]",
    "verbose":  "[dim grey50]VERBOSE: {message}[/dim grey50]",

    # White / neutral
    "neutral":  "[white]{message}[/white]",
    "white":    "[white]{message}[/white]",
    "whiteb":    "[bold white]{message}[/bold white]",

    # Grey variants
    "grey":         "[grey50]{message}[/grey50]",
    "gray":         "[grey50]{message}[/grey50]",
    "dim":          "[dim]{message}[/dim]",
    "muted":        "[dim bright_black]{message}[/dim bright_black]",

    # Orange / amber / highlight-ish
    "orange":       "[bold orange1]ORANGE: {message}[/bold orange1]",
    "orange_soft":  "[orange1]{message}[/orange1]",
    "amber":        "[dark_orange3]AMBER: {message}[/dark_orange3]",

    # Magenta / purple
    "magenta":  "[bold magenta]{message}[/bold magenta]",
    "purple":   "[bold medium_purple4]{message}[/bold medium_purple4]",

    # Cyan / teal
    "cyan":     "[bold cyan]{message}[/bold cyan]",
    "teal":     "[bright_cyan]{message}[/bright_cyan]",

    # Special semantic types for agent stuff (optional, but nice)
    "thought":      "[italic bright_black]{message}[/italic bright_black]",
    "action":       "[bold blue]Action: {message}[/bold blue]",
    "observation":  "[bold cyan]Observation: {message}[/bold cyan]",
    "tool":         "[bold magenta]TOOL: {message}[/bold magenta]",
    "prompt":       "[dim grey50]PROMPT: {message}[/dim grey50]",
    "meta":         "[dim]META: {message}[/dim]",

    # Critical / panic
    "critical": "[bold white on red]CRITICAL: {message}[/bold white on red]",
}