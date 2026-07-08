"""Render the agent's streamed events to the terminal.

Text is buffered for the whole response turn and rendered at once via
rich.markdown.Markdown so headers, bullets, code blocks, and bold/italic
all display correctly.  A transient spinner ("Thinking…" / "Generating…" /
tool-specific label) gives visual feedback while the model works.

Spawn-agent events get a persistent muted line ("→ Calling in the X agent").
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import TYPE_CHECKING

from neurosurfer.agents import events

from . import theme

if TYPE_CHECKING:
    from rich.console import Console
    from rich.live import Live as _Live

    from .io import RichIOHandler

# ── friendly tool messages ────────────────────────────────────────────────────
_TOOL_BASE: dict[str, str] = {
    "list_dir":          "Scouting",
    "read_file":         "Reading",
    "search":            "Searching",
    "run_command":       "Running",
    "write_file":        "Writing",
    "apply_edit":        "Editing",
    "ask_user":          "Waiting for your input",
    "present_plan":      "Drawing up a plan",
    "propose_workflow":  "Proposing a workflow",
    "todo":              "Updating the to-do list",
    "spawn_agent":       "Working",
    "finish":            "Wrapping up",
    "web_search":        "Searching the web",
    "web_fetch":         "Fetching",
    "http":              "Fetching",
    "browse":            "Browsing",
    "data":              "Crunching the data",
}

_THINKING_LABEL   = "Thinking"
_GENERATING_LABEL = "Generating"

# Rich spinner style per tool — visual differentiation at a glance.
_TOOL_ANIM: dict[str, str] = {
    "list_dir":    "dots",
    "read_file":   "dots2",
    "search":      "line",
    "run_command": "dots3",
    "write_file":  "point",
    "apply_edit":  "point",
    "web_search":  "earth",
    "web_fetch":   "earth",
    "http":        "earth",
    "browse":      "earth",
    "data":        "dots",
    "spawn_agent": "bouncingBall",
    "finish":      "dots",
}

# Override per sub-agent type
_AGENT_ANIM: dict[str, str] = {
    "explore":  "bouncingBall",
    "analyzer": "dots12",
    "writer":   "point",
    "verifier": "toggle",
}

# Action labels per sub-agent type
_AGENT_LABEL: dict[str, str] = {
    "explore":  "Exploring",
    "analyzer": "Analysing",
    "writer":   "Drafting",
    "verifier": "Verifying",
}


# ── left-aligned Markdown ─────────────────────────────────────────────────────
# Rich centres Markdown headings (and boxes h1 in a heavy panel) by default.
# We render the whole response left-aligned, so headings should match. Build the
# subclass once, lazily, to keep the rich import off the startup path.
_LEFT_MD_CLS = None


def _left_markdown(text: str):
    """A rich Markdown renderable whose headings are left-aligned, not centred."""
    global _LEFT_MD_CLS
    if _LEFT_MD_CLS is None:
        from rich import box
        from rich.markdown import Heading, Markdown
        from rich.panel import Panel
        from rich.text import Text

        class _LeftHeading(Heading):
            def __rich_console__(self, console, options):  # type: ignore[no-untyped-def]
                self.text.justify = "left"
                if self.tag == "h1":
                    yield Panel(self.text, box=box.HEAVY, style="markdown.h1.border")
                else:
                    if self.tag == "h2":
                        yield Text("")
                    yield self.text

        class _LeftMarkdown(Markdown):
            elements = {**Markdown.elements, "heading_open": _LeftHeading}

        _LEFT_MD_CLS = _LeftMarkdown
    return _LEFT_MD_CLS(text)


def _fmt_tok(n: int) -> str:
    """'142 tokens' or '1.2k tokens'."""
    return f"{n / 1000:.1f}k tokens" if n >= 1000 else f"{n} tokens"


def _fmt_thinking(chars: int) -> str:
    return f"{_THINKING_LABEL} · {_fmt_tok(max(1, chars // 4))}"


def _fmt_generating(chars: int, tail: str = "") -> str:
    count = _fmt_tok(max(1, chars // 4))
    if tail:
        return f"{_GENERATING_LABEL} · {count}  …{tail}"
    return f"{_GENERATING_LABEL} · {count}"


# Tools that own the terminal with prompt_toolkit — must NOT get a Rich Live
# spinner (they would fight for the same terminal lines).
_INTERACTIVE_TOOLS = frozenset({"ask_user", "present_plan", "propose_workflow"})


_TODO_CONTENT_KEYS = ("content", "task", "text", "title", "description", "name", "item")
_TODO_DONE_WORDS   = {"completed", "complete", "done", "finished", "closed", "resolved"}
_TODO_WIP_WORDS    = {"in_progress", "in-progress", "inprogress", "in progress",
                      "doing", "active", "started", "wip"}


def _raw_todo_status(raw: str) -> str:
    """Normalise a raw status string the same way the TodoItem validator does."""
    s = raw.strip().lower()
    if s in _TODO_DONE_WORDS:
        return "completed"
    if s in _TODO_WIP_WORDS:
        return "in_progress"
    return "pending"


def _todo_content(item: object) -> str:
    """Extract the text content from a raw todo item dict (pre-pydantic)."""
    if not isinstance(item, dict):
        return str(item) if item else ""
    for key in _TODO_CONTENT_KEYS:
        val = item.get(key)
        if val:
            return str(val)
    return ""


def _todo_status(item: object) -> str:
    """Normalise the status from a raw todo item dict (pre-pydantic)."""
    if not isinstance(item, dict):
        return "pending"
    raw = item.get("status", "pending")
    return _raw_todo_status(str(raw)) if isinstance(raw, str) else "pending"


def _merge_todos(state: dict[str, str], raw_items: list) -> None:
    """Merge a new todo call's raw items into the accumulated *state* dict.

    Items that appear in *state* but NOT in *raw_items* are assumed done
    (the model removed them instead of marking them completed) and are
    kept as 'completed' so they remain visible in the output.
    """
    new_seen: set[str] = set()
    for item in raw_items:
        content = _todo_content(item)
        if not content:
            continue
        new_seen.add(content)
        state[content] = _todo_status(item)

    # Items the model removed → promote to completed so they stay visible.
    for content in list(state):
        if content not in new_seen and state[content] != "completed":
            state[content] = "completed"


_TODO_MARK = {
    "completed":   lambda: f"[{theme.OK}]✓[/{theme.OK}]",
    "in_progress": lambda: f"[{theme.WARN}]→[/{theme.WARN}]",
    "pending":     lambda: f"[{theme.DIM}]○[/{theme.DIM}]",
}


def _render_todo_state(state: dict[str, str], console: Console) -> None:
    """Render the accumulated todo state (content → status) with status markers."""
    from rich.markup import escape as _esc

    n = len(state)
    done = sum(1 for s in state.values() if s == "completed")
    console.print(f"  [{theme.DIM}]Todo  [{done}/{n}][/{theme.DIM}]")
    for content, status in state.items():
        mark_fn = _TODO_MARK.get(status, _TODO_MARK["pending"])
        mark = mark_fn()
        console.print(f"    {mark}  [{theme.DIM}]{_esc(content)}[/{theme.DIM}]")


def _tool_message(name: str, args: dict) -> str:
    if name == "read_file" and "path" in args:
        return f"Peeking inside {Path(str(args['path'])).name}"
    if name == "write_file" and "path" in args:
        return f"Crafting {Path(str(args['path'])).name}"
    if name == "apply_edit" and "path" in args:
        return f"Polishing {Path(str(args['path'])).name}"
    if name == "list_dir":
        p = str(args.get("path") or ".").rstrip("/") or "."
        return f"Scouting {p}"
    if name == "search" and "pattern" in args:
        pat = str(args["pattern"])
        return f"Hunting for {pat!r}" if len(pat) < 40 else "Hunting through the codebase"
    if name == "web_search" and "query" in args:
        q = str(args["query"])
        return f"Googling '{q[:40]}'" if len(q) <= 40 else "Surfing the web"
    if name == "spawn_agent" and "agent_type" in args:
        return _AGENT_LABEL.get(str(args["agent_type"]), "Working")
    return _TOOL_BASE.get(name, name)


def _spinner_label(active: dict[str, str]) -> str:
    if not active:
        return ""
    if len(active) == 1:
        return next(iter(active.values()))
    return f"Working ({len(active)} operations in parallel)"


def _resolve_anim(name: str, args: dict) -> str:
    """Pick a Rich spinner style for the given tool call."""
    if name == "spawn_agent":
        return _AGENT_ANIM.get(str(args.get("agent_type", "")), "bouncingBall")
    return _TOOL_ANIM.get(name, "dots")


# ── main renderer ─────────────────────────────────────────────────────────────
async def stream_events(
    console: Console,
    event_gen: AsyncIterator,
    io: RichIOHandler | None = None,
) -> events.RunFinished | None:
    """Consume an Agent/runner event stream and render it. Returns the final event."""
    from rich.live import Live
    from rich.markup import escape
    from rich.spinner import Spinner

    buffer: list[str] = []
    final: events.RunFinished | None = None
    live: _Live | None = None
    _spinner: Spinner | None = None
    _current_anim: str = "dots"   # animation style currently loaded in _spinner
    # tool_id → friendly label; special keys: "__thinking__", "__generating__"
    active_tools: dict[str, str] = {}
    # tool_id → Rich spinner style name
    active_anims: dict[str, str] = {}
    # Accumulated todo state: content → status (keeps completed items visible)
    todo_state: dict[str, str] = {}
    thinking_chars: int = 0
    generating_chars: int = 0
    _first_text: bool = True   # True until first text token for the current response block

    # ── spinner helpers ───────────────────────────────────────────────────────
    def _pick_anim() -> str:
        """Choose a spinner style from the current active tools."""
        real = [k for k in active_anims if not k.startswith("__")]
        if len(real) == 1:
            return active_anims[real[0]]
        return "dots"

    def _start_or_update_live() -> None:
        nonlocal live, _spinner, _current_anim
        label = _spinner_label(active_tools)
        if not label:
            return
        anim = _pick_anim()
        styled = f" [{theme.ACCENT_DIM}]{escape(label)}…[/{theme.ACCENT_DIM}]"
        if live is None or anim != _current_anim:
            # Start fresh (or swap animation style). Recreating the Spinner resets
            # its frame counter — a minor visual glitch but necessary when the
            # animation style itself changes between tool calls.
            _stop_live()
            _current_anim = anim
            _spinner = Spinner(anim, text=styled)
            live = Live(_spinner, console=console, transient=True, refresh_per_second=12)
            live.start()
        elif _spinner is not None:
            # Same style: update text only so the animation keeps ticking without
            # a frame-counter reset. Use .update() so markup tags are parsed
            # (raw .text= leaks the ANSI colour tag string to screen).
            _spinner.update(text=styled)

    def _stop_live() -> None:
        nonlocal live, _spinner
        if live is not None:
            live.stop()
            live = None
            _spinner = None

    def flush() -> None:
        """Render the buffered response turn as Markdown and clear state."""
        if not buffer:
            return
        _stop_live()
        active_tools.pop("__generating__", None)
        active_anims.pop("__generating__", None)
        from rich.rule import Rule
        console.print(Rule(style=theme.DIM))
        console.print(_left_markdown("".join(buffer)))
        buffer.clear()

    # Wire the stopper into the IO handler so approval prompts (shell, write)
    # can stop the Live spinner before running their own prompt_toolkit UI.
    # Without this, the two terminal renderers stomp each other and produce
    # spurious ANSI escape sequences in the output.
    if io is not None:
        io.pause_live = _stop_live

    # ── event loop ────────────────────────────────────────────────────────────
    # "Thinking…" spinner fires before the model produces its first token.
    active_tools["__thinking__"] = _THINKING_LABEL
    active_anims["__thinking__"] = "moon"
    _start_or_update_live()

    try:
        async for ev in event_gen:
            if isinstance(ev, events.TextDelta):
                active_tools.pop("__thinking__", None)
                active_anims.pop("__thinking__", None)
                generating_chars += len(ev.text)
                buffer.append(ev.text)
                # Buffer all tokens and show a generating spinner; the full text is
                # rendered as Markdown in flush() once the turn is complete.
                if _first_text:
                    _stop_live()
                    _first_text = False
                active_tools["__generating__"] = _fmt_generating(generating_chars)
                active_anims["__generating__"] = "dots"
                _start_or_update_live()

            elif isinstance(ev, events.ThinkingDelta):
                thinking_chars += len(ev.text)
                active_tools["__thinking__"] = _fmt_thinking(thinking_chars)
                _start_or_update_live()

            elif isinstance(ev, events.ToolStarted):
                flush()  # render any text that arrived before this tool call
                active_tools.pop("__thinking__", None)
                active_anims.pop("__thinking__", None)
                if ev.name in _INTERACTIVE_TOOLS:
                    _stop_live()
                elif ev.name == "todo" and isinstance(ev.args.get("todos"), list):
                    # Merge into accumulated state so completed items stay visible.
                    _stop_live()
                    _merge_todos(todo_state, ev.args["todos"])
                    _render_todo_state(todo_state, console)
                    active_tools[ev.id] = "Updating the to-do list"
                    active_anims[ev.id] = "dots"
                    _start_or_update_live()
                else:
                    msg = _tool_message(ev.name, ev.args)
                    anim = _resolve_anim(ev.name, ev.args)
                    active_tools[ev.id] = msg
                    active_anims[ev.id] = anim
                    # Persistent log line (not just the transient spinner) so the
                    # scrollback shows a running trace of what the agent did, in
                    # order, interleaved with its text/thinking output.
                    _stop_live()
                    console.print(f"  [{theme.DIM}]→ {escape(msg)}…[/{theme.DIM}]")
                    _start_or_update_live()

            elif isinstance(ev, events.ToolFinished):
                active_tools.pop(ev.id, None)
                active_anims.pop(ev.id, None)
                if active_tools:
                    _start_or_update_live()
                else:
                    _stop_live()
                    # After all tools finish the model will think before its next
                    # response — restart the thinking spinner so the user knows
                    # something is happening (especially important after ask_user).
                    thinking_chars = 0
                    generating_chars = 0
                    _first_text = True   # next response block gets a fresh separator
                    active_tools["__thinking__"] = _THINKING_LABEL
                    active_anims["__thinking__"] = "moon"
                    _start_or_update_live()
                if ev.result.is_error and ev.name != "spawn_agent":
                    first_line = (ev.result.content or "").splitlines()[0] if ev.result.content else ""
                    console.print(
                        f"  [{theme.ERR}]✗ {escape(ev.name)}: {escape(first_line)}[/{theme.ERR}]"
                    )

            elif isinstance(ev, events.TurnCompleted):
                flush()
                if ev.stop_reason == "max_tokens":
                    console.print(
                        f"\n[{theme.WARN}]⚠ Response cut off — the model hit the output "
                        f"token limit mid-generation. The reply above may be incomplete. "
                        f"Try asking it to continue, or raise the context window.[/{theme.WARN}]"
                    )

            elif isinstance(ev, events.ModeChanged):
                _stop_live()
                console.print(f"\n[{theme.OK_DIM}]  ✓ mode: {ev.mode}[/{theme.OK_DIM}]")

            elif isinstance(ev, events.Compacted):
                _stop_live()
                console.print(
                    f"\n[{theme.DIM}]  ⟳ context compacted "
                    f"({ev.tokens_before:,} → {ev.tokens_after:,} tokens)[/{theme.DIM}]"
                )

            elif isinstance(ev, events.RunFinished):
                flush()
                _stop_live()
                final = ev
                if ev.status in ("completed", "success"):
                    console.print(f"[{theme.OK}]✓ Done.[/{theme.OK}]")
                elif ev.status == "handoff_workflow":
                    pass  # _assist drives the Architect build; no status line here
                elif ev.status == "max_turns":
                    console.print(f"\n[{theme.WARN}]⚠ Reached turn limit.[/{theme.WARN}]")
                else:
                    console.print(f"\n[{theme.DIM}]Finished: {ev.status}[/{theme.DIM}]")
                if ev.report:
                    console.print(f"[{theme.DIM}]{escape(ev.report)}[/{theme.DIM}]")

            elif isinstance(ev, events.AgentError):
                flush()
                _stop_live()
                console.print(f"\n[{theme.ERR}]Error:[/{theme.ERR}] {escape(ev.message)}")

    finally:
        # Always clean up the spinner — catches Ctrl+C, CancelledError, and any
        # other exception that short-circuits the event loop above.
        flush()
        _stop_live()
        if io is not None:
            io.pause_live = None

    return final
