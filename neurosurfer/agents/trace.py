"""Live activity trace for the agent's ``verbose`` side-channel.

Shows what the agent is doing *while it works* — an animated one-line status whose label
shifts through verbs ("Thinking…", "Pondering…", "Cooking…") as the model streams reasoning,
switches to "Running <tool>…" while a tool executes, and drops a persistent line for each
completed tool call. It steps out of the way the instant the answer starts streaming so the
caller's own output (the ``TextDelta`` it prints) is never clobbered.

Implementation note: the status line is animated with a carriage return (``\\r``) — the same
trick progress bars use — *not* a Rich ``Live`` region. ``Live`` looks great in a real
terminal but leaves empty blocks behind in Jupyter every time it starts/stops, which shows up
as large vertical gaps. ``\\r`` updates a single line in place and behaves identically in a
terminal and in a notebook (VS Code / Lab / classic), with no gaps.

This is a side-channel: it observes the event stream but never consumes it. Both
:class:`~neurosurfer.agents.agentic_loop.AgenticLoop` and
:class:`~neurosurfer.agents.react.ReactAgent` get the identical experience because
:meth:`~neurosurfer.agents.base.BaseAgent.run` feeds every agent's events through here.

When stdout is neither a TTY nor a notebook (a pipe, CI, the test suite) it degrades to plain
newline-terminated lines — no spinner / carriage-return noise in captured output.
"""

from __future__ import annotations

import itertools
import sys
import time

from neurosurfer.agents.conversation import events

# Verbs the status cycles through while the model is thinking — keeps it feeling alive,
# the way Claude Code / Codex do.
_THINK_VERBS = [
    "Thinking", "Pondering", "Reasoning", "Mulling", "Cooking", "Noodling",
    "Percolating", "Synthesizing", "Computing", "Ruminating", "Scheming", "Brewing",
]
_SPINNER_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

_MAX_ARGS_LEN = 80
_MAX_RESULT_LEN = 120
_VERB_INTERVAL = 1.1   # seconds between verb changes while thinking

# ANSI styles (honoured by terminals and by Jupyter/VS Code output)
_DIM, _CYAN, _YELLOW, _RED, _RESET = "\033[2m", "\033[36m", "\033[33m", "\033[31m", "\033[0m"


def _short(text: str, limit: int) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _in_notebook() -> bool:
    try:
        from IPython import get_ipython  # noqa: PLC0415

        shell = get_ipython()
        return shell is not None and shell.__class__.__name__ == "ZMQInteractiveShell"
    except Exception:  # noqa: BLE001
        return False


class AgentTrace:
    """Animated (or plain) one-line activity trace, driven one event at a time."""

    def __init__(self) -> None:
        self._out = sys.stdout
        self._frames = itertools.cycle(_SPINNER_FRAMES)
        self._verbs = itertools.cycle(_THINK_VERBS)
        self._verb = next(self._verbs)
        self._verb_at = 0.0
        self._mode: str | None = None      # "think" | "tool" | None
        self._status_len = 0               # visible width of the live status line
        self._answer_open = False          # caller is mid-stream printing the answer
        try:
            self._interactive = bool(getattr(self._out, "isatty", lambda: False)()) or _in_notebook()
        except Exception:  # noqa: BLE001
            self._interactive = False

    # ── public API ────────────────────────────────────────────────────────────
    def handle(self, ev: events.Event) -> None:
        try:
            (self._handle_live if self._interactive else self._handle_plain)(ev)
        except Exception:  # noqa: BLE001 — a trace must never break the run
            self._status_len = 0

    def close(self) -> None:
        if not self._interactive:
            return
        try:
            self._clear_status()
            if self._answer_open:
                self._out.write("\n")
                self._out.flush()
                self._answer_open = False
        except Exception:  # noqa: BLE001
            pass

    # ── animated (interactive) path ──────────────────────────────────────────────
    def _handle_live(self, ev: events.Event) -> None:
        if isinstance(ev, events.ThinkingDelta):
            # While the answer is streaming, never draw the spinner — a stray reasoning
            # token mid-answer must not overwrite the text the caller is printing. The
            # spinner resumes after the next tool call (which commits a fresh line).
            if not self._answer_open:
                self._tick_thinking()
        elif isinstance(ev, events.ToolStarted):
            self._commit(f"{_YELLOW}→ {ev.name}({_short(repr(ev.args), _MAX_ARGS_LEN)}){_RESET}")
            self._mode = None
        elif isinstance(ev, events.ToolFinished):
            head = _short(ev.result.content, _MAX_RESULT_LEN)
            if ev.result.is_error:
                self._commit(f"{_RED}  ✗ {ev.name}: {head}{_RESET}")
            else:
                self._commit(f"{_DIM}  ↳ {head}{_RESET}")
            self._mode = None
        elif isinstance(ev, events.TextDelta):
            # Answer is streaming — clear the status and let the caller render it.
            self._clear_status()
            self._answer_open = True
            self._mode = None
        elif isinstance(ev, events.ModeChanged):
            self._commit(f"{_DIM}  ✓ mode: {ev.mode}{_RESET}")
            self._mode = None
        elif isinstance(ev, events.Compacted):
            self._commit(f"{_DIM}  ⟳ context compacted ({ev.tokens_before:,} → {ev.tokens_after:,} tokens){_RESET}")
            self._mode = None
        elif isinstance(ev, events.AgentError):
            self._commit(f"{_RED}✗ {ev.message}{_RESET}")
            self._mode = None
        elif isinstance(ev, events.RunFinished):
            self._clear_status()
            self._mode = None

    def _tick_thinking(self) -> None:
        now = time.monotonic()
        if self._mode != "think":
            self._mode = "think"
            self._verb_at = now
        elif now - self._verb_at >= _VERB_INTERVAL:
            self._verb = next(self._verbs)
            self._verb_at = now
        self._write_status(f"{self._verb}…", _CYAN)

    # ── carriage-return status plumbing ──────────────────────────────────────────
    # In-place updates: ``\r`` → column 0, ``\033[2K`` → erase line, plus trailing space
    # padding as a fallback for frontends that ignore ``\033[2K`` (so a longer→shorter
    # verb change never leaves residue like "king…"). *Leaving* a status finalises it
    # with a real newline — the only line break every frontend (terminal / VS Code /
    # Lab / classic) honours — so the next line is never glued onto "Thinking…".
    _ERASE = "\r\033[2K"

    def _fresh_line(self) -> None:
        """Drop to a new line if the caller was mid-printing the answer, so our output
        never overwrites it."""
        if self._answer_open:
            self._out.write("\n")
            self._answer_open = False

    def _write_status(self, label: str, color: str) -> None:
        """Render/overwrite the single animated status line in place."""
        frame = next(self._frames)
        visible = f"{frame} {label}"
        pad = max(0, self._status_len - len(visible))  # cover a shrink if \033[2K is ignored
        self._out.write(f"{self._ERASE}{color}{visible}{_RESET}{' ' * pad}")
        self._out.flush()
        self._status_len = len(visible)

    def _clear_status(self) -> None:
        """Finalise the live status line with a newline so following output starts fresh."""
        if self._status_len:
            self._out.write("\n")
            self._out.flush()
            self._status_len = 0

    def _commit(self, line: str) -> None:
        """Finalise any status line, then print a persistent line that stays in scrollback."""
        self._clear_status()
        self._fresh_line()
        self._out.write(line + "\n")
        self._out.flush()

    # ── plain (non-interactive) path ────────────────────────────────────────────
    def _handle_plain(self, ev: events.Event) -> None:
        if isinstance(ev, events.ThinkingDelta):
            if self._mode != "think":
                self._mode = "think"
                print("· thinking", flush=True)
        elif isinstance(ev, events.ToolStarted):
            self._mode = None
            print(f"→ {ev.name}({_short(repr(ev.args), _MAX_ARGS_LEN)})", flush=True)
        elif isinstance(ev, events.ToolFinished):
            self._mode = None
            head = _short(ev.result.content, _MAX_RESULT_LEN)
            if ev.result.is_error:
                print(f"  ✗ {ev.name}: {head}", flush=True)
            else:
                print(f"  ↳ {head}", flush=True)
        elif isinstance(ev, events.TextDelta):
            self._mode = None
        elif isinstance(ev, events.ModeChanged):
            self._mode = None
            print(f"  ✓ mode: {ev.mode}", flush=True)
        elif isinstance(ev, events.Compacted):
            self._mode = None
            print(f"  ⟳ context compacted ({ev.tokens_before} → {ev.tokens_after} tokens)", flush=True)
        elif isinstance(ev, events.AgentError):
            self._mode = None
            print(f"\n✗ {ev.message}", flush=True)
