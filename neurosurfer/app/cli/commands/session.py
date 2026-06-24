"""/sessions — interactive session browser for the active task.

  /sessions              open the browser (list → select → use/delete/back)
  /sessions delete <id>  quick-delete without opening the browser

Sessions are per-task conversation histories stored under ~/.neurosurfer/sessions/.
"""

from __future__ import annotations

from .. import theme
from ..context import CLIContext
from .base import SlashCommand


def _fmt_age(dt: object) -> str:
    from datetime import UTC, datetime

    if not isinstance(dt, datetime):
        return "?"
    now = datetime.now(UTC)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    s = int((now - dt).total_seconds())
    if s < 3600:
        return f"{max(1, s // 60)} min ago"
    if s < 86400:
        return f"{s // 3600} h ago"
    return f"{s // 86400} days ago"


def print_session_summary(ctx: CLIContext, task: str, session_id: str) -> None:
    """Render a Rich panel with the chosen session's key details."""
    from rich.box import ROUNDED
    from rich.markup import escape
    from rich.panel import Panel

    store = ctx.session_store
    if store is None:
        return
    rec = store.get(task, session_id)
    if rec is None:
        return

    lines: list[str] = []

    updated = rec.updated_at.strftime("%Y-%m-%d %H:%M") if rec.updated_at else "?"
    lines.append(f"[{theme.DIM}]Task:[/{theme.DIM}]          {escape(rec.task)}")
    lines.append(f"[{theme.DIM}]Last active:[/{theme.DIM}]   {updated}")
    lines.append(
        f"[{theme.DIM}]History:[/{theme.DIM}]       "
        f"{rec.turn_count} turns · {rec.message_count} messages"
    )
    if rec.cwd:
        lines.append(f"[{theme.DIM}]Directory:[/{theme.DIM}]     {escape(rec.cwd)}")

    if rec.artifacts:
        lines.append("")
        lines.append(f"[{theme.DIM}]Files touched:[/{theme.DIM}]")
        for a in rec.artifacts[:5]:
            lines.append(f"  [{theme.DIM}]{escape(a)}[/{theme.DIM}]")
        if len(rec.artifacts) > 5:
            lines.append(f"  [{theme.DIM}]… and {len(rec.artifacts) - 5} more[/{theme.DIM}]")

    if rec.turn_count > 0:
        msgs = store.load_history(task, session_id)
        recent = [m for m in msgs if m.role in ("user", "assistant")][-4:]
        if recent:
            lines.append("")
            lines.append(f"[{theme.DIM}]Recent messages:[/{theme.DIM}]")
            for m in recent:
                txt = m.text()
                if not txt:
                    continue
                prefix = "You" if m.role == "user" else "Agent"
                snippet = txt.replace("\n", " ")[:120]
                if len(txt) > 120:
                    snippet += "…"
                c = theme.DIM if m.role == "user" else theme.ACCENT_DIM
                lines.append(
                    f"  [{c}]{escape(prefix)}:[/{c}] "
                    f"[{theme.DIM}]{escape(snippet)}[/{theme.DIM}]"
                )

    title_text = escape(rec.title or "(untitled)")
    ctx.console.print(Panel(
        "\n".join(lines),
        title=f"[bold {theme.ACCENT}]{title_text}[/bold {theme.ACCENT}]",
        border_style=theme.ACCENT_DIM,
        box=ROUNDED,
        padding=(0, 2),
    ))
    ctx.console.print(
        f"[{theme.DIM}]  Session active — send a message to resume.[/{theme.DIM}]"
    )


async def _session_browser(ctx: CLIContext, task: str) -> None:
    """Loop: list sessions → pick one → action sub-menu → repeat."""
    from ..io import select_menu

    store = ctx.session_store
    if store is None:
        ctx.console.print(f"[{theme.DIM}]Session store not available.[/{theme.DIM}]")
        return

    store.purge_empty(task)
    while True:
        sessions = store.list_for_task(task, limit=20)
        if not sessions:
            ctx.console.print(f"[{theme.DIM}]No sessions for task '{task}'.[/{theme.DIM}]")
            return

        main_opts: list[tuple] = [
            (s.id, s.title or "(untitled)", _fmt_age(s.updated_at))
            for s in sessions
        ]
        main_opts.append(("__cancel__", "Close", ""))

        chosen_id = await select_menu(ctx.console, f"Sessions — {task}", main_opts)
        if not chosen_id or chosen_id == "__cancel__":
            return

        rec = store.get(task, chosen_id)
        title = (rec.title or "(untitled)") if rec else chosen_id

        action = await select_menu(
            ctx.console,
            f'"{title}"',
            [
                ("use", "Use", "Switch to this session — next message resumes it"),
                ("delete", "Delete", "Remove this session and its history"),
                ("back", "Back", "Return to the session list"),
            ],
        )

        if action == "use":
            ctx.active_session_id = chosen_id
            print_session_summary(ctx, task, chosen_id)
            return

        if action == "delete":
            store.delete(task, chosen_id)
            ctx.console.print(f"[{theme.DIM}]Deleted '{title}'.[/{theme.DIM}]")
            # If we just deleted the currently active session, clear the reference
            # so the next message auto-creates a fresh one.
            if ctx.active_session_id == chosen_id:
                ctx.active_session_id = None
            continue  # refresh list

        # "back" or Esc → return to list
        continue


async def handle_sessions(ctx: CLIContext, args: list[str]) -> None:
    sub = args[0] if args else ""
    rest = args[1:]
    active = ctx.active_task or ""

    # Quick-delete shorthand: /sessions delete <id>
    if sub == "delete":
        if not rest:
            ctx.console.print(f"[{theme.ERR}]Usage: /sessions delete <id>[/{theme.ERR}]")
            return
        store = ctx.session_store
        if store and store.delete(active, rest[0]):
            ctx.console.print(f"[{theme.OK}]Deleted session {rest[0]}.[/{theme.OK}]")
        else:
            ctx.console.print(
                f"[{theme.ERR}]No session '{rest[0]}' for task '{active}'.[/{theme.ERR}]"
            )
        return

    # Resolve task from args or active task
    task = rest[0] if sub == "list" and rest else active
    if not task:
        ctx.console.print(
            f"[{theme.DIM}]No active task. Use /task use first, "
            f"or: /sessions list <task>[/{theme.DIM}]"
        )
        return

    await _session_browser(ctx, task)


COMMAND = SlashCommand(
    name="sessions",
    summary="Browse and manage REPL sessions (interactive)",
    handler=handle_sessions,
    aliases=["session"],
)
