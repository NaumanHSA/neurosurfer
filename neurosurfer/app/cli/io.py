"""Interactive IOHandler — wires ask_user / approvals to the terminal.

Implements the ``tools.base.IOHandler`` protocol.

Multi-choice questions use a prompt_toolkit Application for arrow-key navigation
(↑/↓ + Enter). After the user confirms, the selector is erased with ANSI sequences
and Rich prints a clean confirmation line. Free-text questions use a styled
prompt_toolkit PromptSession (cyan ❯ prompt, async, history).

Non-interactive / test fallback: when stdin is not a TTY the numbered-list +
typed-input path is used so tests can patch ``_ainput``.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any

from neurosurfer.tools.base import ShellApproval

from . import theme

if TYPE_CHECKING:
    from rich.console import Console

    from neurosurfer.tools.base import WriteChoice

# Synthetic last row in a multi-choice prompt → lets the user type a custom answer.
_OTHER_LABEL = "✎  Something else (type my own answer)"


class InputCancelled(Exception):
    """Raised when the user presses Escape inside an _ainput prompt."""


# ──────────────────────────────────────────────────────────────────────────────
# Low-level primitives
# ──────────────────────────────────────────────────────────────────────────────

def _is_interactive() -> bool:
    return sys.stdin.isatty()


async def _ainput(prompt_text: str = "", *, is_password: bool = False) -> str:
    """Read a line without blocking the event loop (prompt_toolkit PromptSession).

    Raises ``InputCancelled`` if the user presses Escape.
    """
    from prompt_toolkit import PromptSession
    from prompt_toolkit.formatted_text import HTML
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.patch_stdout import patch_stdout

    kb = KeyBindings()

    @kb.add("escape")
    def _cancel(event) -> None:  # type: ignore[no-untyped-def]
        event.app.exit(exception=InputCancelled())

    session: PromptSession = PromptSession(key_bindings=kb)
    color = theme.current().pt_prompt
    styled = HTML(f"<{color}>  ❯</{color}>  {prompt_text}")
    with patch_stdout():
        return await session.prompt_async(styled, is_password=is_password)


async def _select_inline(choices: list[str], descs: list[str] | None = None) -> str | None:
    """Arrow-key inline selector.

    Renders a navigable list at the current cursor position. When the user
    confirms with Enter, the list is erased with ANSI sequences so the caller
    can print a clean confirmation without leftover selector text.

    ``descs`` is an optional parallel list of descriptions shown in a second
    column (dimmed). Returns the chosen label string, or ``None`` on cancel.
    """
    from prompt_toolkit.application import Application
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.layout import Layout, Window
    from prompt_toolkit.layout.controls import FormattedTextControl

    idx = [0]
    result: list[str | None] = [None]
    n = len(choices)

    # Align the label column so descriptions line up neatly.
    col_w = max((len(c) for c in choices), default=0) + 2

    def _get_text() -> list[tuple[str, str]]:
        lines: list[tuple[str, str]] = []
        lines.append(("class:hint", "\n"))  # blank line between title/question and first option
        for i, choice in enumerate(choices):
            desc = (descs[i] if descs else "") or ""
            if i == idx[0]:
                lines.append(("class:selected", f"  ❯  {choice:<{col_w}}"))
                if desc:
                    lines.append(("class:hint", f"  {desc}"))
                lines.append(("class:selected", "\n"))
            else:
                lines.append(("class:choice", f"     {choice:<{col_w}}"))
                if desc:
                    lines.append(("class:hint", f"  {desc}"))
                lines.append(("class:choice", "\n"))
        lines.append(("class:hint", "  ↑/↓  navigate   Enter  confirm   Esc  cancel"))
        return lines

    kb = KeyBindings()

    @kb.add("up")
    def _up(event) -> None:  # type: ignore[no-untyped-def]
        idx[0] = (idx[0] - 1) % n

    @kb.add("down")
    def _down(event) -> None:  # type: ignore[no-untyped-def]
        idx[0] = (idx[0] + 1) % n

    @kb.add("enter")
    def _enter(event) -> None:  # type: ignore[no-untyped-def]
        result[0] = choices[idx[0]]
        event.app.exit()

    @kb.add("c-c")
    @kb.add("escape")
    def _cancel(event) -> None:  # type: ignore[no-untyped-def]
        event.app.exit()

    style = theme.select_style()

    app: Application = Application(
        layout=Layout(Window(FormattedTextControl(_get_text, focusable=True))),
        key_bindings=kb,
        style=style,
        full_screen=False,
        mouse_support=False,
    )

    await app.run_async()

    # Erase the rendered selector area.
    # _get_text renders: 1 blank + n choices + hint (no \n).
    # prompt_toolkit emits a trailing \r\n on exit (non-fullscreen renderer reset),
    # landing the cursor one line below the hint.
    # Total lines above cursor: blank(1) + choices(n) + hint(1) + exit-newline(1) = n+3
    # but we count from hint+1 upward: need to jump n+2 to reach the blank line, then clear.
    sys.stdout.write(f"\r\033[{n + 2}A\033[J")
    sys.stdout.flush()

    return result[0]


async def select_menu(
    console: Console,
    title: str,
    options: list[tuple],  # (value, label) or (value, label, description)
) -> Any | None:
    """Shared arrow-key menu used everywhere for consistency.

    ``options`` is a list of ``(value, label)`` or ``(value, label, description)``
    tuples. The chosen ``value`` is returned (or ``None`` on cancel). Interactive
    terminals get the ↑/↓ + Enter selector with an optional description column;
    non-interactive stdin (tests / pipes) falls back to numbered input.
    """
    if title:
        console.print(f"\n[bold {theme.ACCENT}]{title}[/bold {theme.ACCENT}]")

    labels = [o[1] for o in options]
    descs: list[str] = [o[2] if len(o) >= 3 else "" for o in options]  # type: ignore[misc]
    has_descs = any(descs)

    if _is_interactive():
        picked = await _select_inline(labels, descs if has_descs else None)
        if picked is None:
            return None
        return options[labels.index(picked)][0]

    # Non-interactive fallback: numbered list with optional description.
    col_w = max((len(lb) for lb in labels), default=0)
    for i, (label, desc) in enumerate(zip(labels, descs, strict=False), 1):
        desc_part = f"  {desc}" if desc else ""
        console.print(f"  [{theme.ACCENT}]{i}[/{theme.ACCENT}]  {label:<{col_w}}{desc_part}")
    try:
        raw = (await _ainput(f"  Enter 1–{len(options)} (or Enter to cancel): ")).strip()
    except InputCancelled:
        return None
    if not raw or not raw.isdigit():
        return None
    idx = int(raw) - 1
    return options[idx][0] if 0 <= idx < len(options) else None


# ──────────────────────────────────────────────────────────────────────────────
# IOHandler
# ──────────────────────────────────────────────────────────────────────────────

class RichIOHandler:
    def __init__(self, console: Console) -> None:
        self._console = console
        # Injected by render.stream_events so approval prompts can stop the
        # Rich Live spinner before running their own prompt_toolkit UI.
        self.pause_live: object = None  # Callable | None

    def _pause(self) -> None:
        if callable(self.pause_live):
            self.pause_live()

    async def ask(self, question: str, options: list[str] | None = None) -> str:
        self._pause()
        from rich.box import ROUNDED
        from rich.markup import escape
        from rich.panel import Panel

        self._console.print()
        self._console.print(Panel(
            f"[bold]{escape(question)}[/bold]",
            border_style=theme.ACCENT_DIM,
            box=ROUNDED,
            padding=(0, 2),
            expand=False,
        ))
        self._console.print()

        if options:
            if _is_interactive():
                # Append a free-text escape so the user is never boxed into the
                # suggested answers.
                rows = [*options, _OTHER_LABEL]
                picked = await _select_inline(rows)
                if picked is None:
                    picked = options[0]
                if picked == _OTHER_LABEL:
                    typed = (await _ainput()).strip()
                    if typed:
                        _echo_selection(self._console, typed)
                    return typed
                _echo_selection(self._console, picked)
                return picked

            # Non-interactive fallback (tests / piped stdin).
            for i, opt in enumerate(options, 1):
                self._console.print(f"  [{theme.ACCENT}] {i} [/{theme.ACCENT}]  {escape(opt)}")
            self._console.print()
            while True:
                raw = (await _ainput()).strip()
                if raw.isdigit():
                    idx = int(raw) - 1
                    if 0 <= idx < len(options):
                        _echo_selection(self._console, options[idx])
                        return options[idx]
                elif raw in options:
                    _echo_selection(self._console, raw)
                    return raw
                elif raw:
                    # Treat any other text as a free-text answer.
                    _echo_selection(self._console, raw)
                    return raw
                self._console.print(
                    f"  [{theme.ERR}]Pick a number (1–{len(options)}) "
                    f"or type your own answer.[/{theme.ERR}]"
                )

        # Free-text
        answer = (await _ainput()).strip()
        if answer:
            _echo_selection(self._console, answer)
        return answer

    async def request_plan_approval(self, plan: str) -> tuple[bool, str]:
        self._pause()
        from rich.box import ROUNDED
        from rich.markup import escape
        from rich.panel import Panel

        self._console.print()
        self._console.print(Panel(
            escape(plan),
            title=f"[bold {theme.ACCENT}]  Proposed Plan  [/bold {theme.ACCENT}]",
            border_style=theme.ACCENT,
            box=ROUNDED,
            padding=(1, 2),
        ))
        self._console.print()

        choices = [
            "✓  Approve — proceed",
            "✏  Request changes",
            "✗  Reject",
        ]

        if _is_interactive():
            picked = await _select_inline(choices)
            action_idx = choices.index(picked) if picked in choices else 2
        else:
            for i, label in enumerate(choices, 1):
                self._console.print(f"  [{theme.ACCENT}] {i} [/{theme.ACCENT}]  {label}")
            self._console.print()
            while True:
                raw = (await _ainput()).strip().lower()
                if raw in ("1", "y", "yes", "approve"):
                    action_idx = 0
                    break
                if raw in ("2", "c", "change", "changes"):
                    action_idx = 1
                    break
                if raw in ("3", "n", "no", "reject"):
                    action_idx = 2
                    break
                self._console.print(
                    f"  [{theme.ERR}]Enter 1 (approve), 2 (changes), or 3 (reject).[/{theme.ERR}]"
                )

        if action_idx == 0:
            self._console.print(f"  [{theme.OK}]✓ Plan approved.[/{theme.OK}]")
            return (True, "")
        if action_idx == 1:
            self._console.print(
                f"\n  [{theme.DIM}]Describe what you'd like changed:[/{theme.DIM}]"
            )
            feedback = (await _ainput()).strip()
            return (False, feedback or "Please revise the plan.")
        self._console.print(f"  [{theme.ERR}]✗ Plan rejected.[/{theme.ERR}]")
        return (False, "Plan rejected.")

    async def request_shell_approval(self, command: str, reason: str) -> ShellApproval:
        self._pause()
        from rich.markup import escape

        self._console.print()
        self._console.print(
            f"  [{theme.WARN}]⚡ Shell command requested[/{theme.WARN}]"
        )
        self._console.print(
            f"  [{theme.ACCENT_DIM}]Command:[/{theme.ACCENT_DIM}]  "
            f"[bold]{escape(command)}[/bold]"
        )
        if reason:
            self._console.print(f"  [{theme.DIM}]Reason:  {escape(reason)}[/{theme.DIM}]")
        self._console.print()

        async def _redirect() -> ShellApproval:
            self._console.print(
                f"  [{theme.DIM}]Tell the agent what to do instead:[/{theme.DIM}]"
            )
            msg = (await _ainput("  > ")).strip()
            return ShellApproval(False, msg or None)

        allow, deny, redirect = (
            "✓  Allow",
            "✗  Deny",
            "✎  Deny & tell the agent what to do instead",
        )
        if _is_interactive():
            picked = await _select_inline([allow, deny, redirect])
            if picked == allow:
                return ShellApproval(True)
            if picked == redirect:
                return await _redirect()
            return ShellApproval(False)

        self._console.print(f"  [{theme.ACCENT}] 1 [/{theme.ACCENT}]  ✓  Allow")
        self._console.print(f"  [{theme.ACCENT}] 2 [/{theme.ACCENT}]  ✗  Deny")
        self._console.print(
            f"  [{theme.ACCENT}] 3 [/{theme.ACCENT}]  ✎  Deny & redirect the agent"
        )
        self._console.print()
        while True:
            raw = (await _ainput()).strip()
            low = raw.lower()
            if low in ("1", "y", "yes", "allow"):
                return ShellApproval(True)
            if low in ("2", "n", "no", "deny"):
                return ShellApproval(False)
            if low in ("3", "r", "redirect"):
                return await _redirect()
            self._console.print(
                f"  [{theme.ERR}]Enter 1 (allow), 2 (deny), or 3 (redirect).[/{theme.ERR}]"
            )

    async def request_write_approval(self, path: str, summary: str) -> WriteChoice:
        self._pause()
        import os

        from rich.markup import escape

        folder = os.path.dirname(path) or path

        self._console.print()
        self._console.print(
            f"  [{theme.WARN}]✎ The agent wants to write outside this task's scope[/{theme.WARN}]"
        )
        self._console.print(
            f"  [{theme.ACCENT_DIM}]File:[/{theme.ACCENT_DIM}]    [bold]{escape(path)}[/bold]"
        )
        if summary:
            self._console.print(f"  [{theme.DIM}]{escape(summary)}[/{theme.DIM}]")
        self._console.print()

        labels = [
            f"✓  Always allow this folder  ·  {escape(folder)}",
            "→  Allow once  ·  just this file",
            "✗  Don't allow",
        ]
        values: list[WriteChoice] = ["always", "once", "deny"]
        if _is_interactive():
            picked = await _select_inline(labels)
            return values[labels.index(picked)] if picked in labels else "deny"

        self._console.print(f"  [{theme.ACCENT}] 1 [/{theme.ACCENT}]  Always allow this folder ({escape(folder)})")
        self._console.print(f"  [{theme.ACCENT}] 2 [/{theme.ACCENT}]  Allow once (just this file)")
        self._console.print(f"  [{theme.ACCENT}] 3 [/{theme.ACCENT}]  Don't allow")
        self._console.print()
        while True:
            raw = (await _ainput()).strip().lower()
            if raw in ("1", "always", "a"):
                return "always"
            if raw in ("2", "once", "o", "y", "yes"):
                return "once"
            if raw in ("3", "deny", "n", "no"):
                return "deny"
            self._console.print(
                f"  [{theme.ERR}]Enter 1 (always), 2 (once), or 3 (don't allow).[/{theme.ERR}]"
            )

    def notify(self, message: str) -> None:
        from rich.markup import escape
        self._console.print(f"  [{theme.DIM}]· {escape(message)}[/{theme.DIM}]")


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _echo_selection(console: Console, value: str) -> None:
    from rich.markup import escape
    console.print(f"  [{theme.OK}]✓[/{theme.OK}]  [{theme.DIM}]{escape(value)}[/{theme.DIM}]")
