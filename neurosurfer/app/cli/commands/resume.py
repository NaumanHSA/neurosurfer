"""/resume — continue an interrupted run from its persisted durable state.

``/resume`` with no id lists resumable runs (those with a saved state snapshot);
``/resume <run_id>`` reloads that run's task + durable state and continues it.
"""

from __future__ import annotations

from .. import theme
from ..context import CLIContext
from .base import SlashCommand


async def handle_resume(ctx: CLIContext, args: list[str]) -> None:
    from ..run import list_resumable, resume_run

    if not args:
        run_ids = list_resumable(ctx)
        if not run_ids:
            ctx.console.print(f"[{theme.DIM}]No resumable runs.[/{theme.DIM}]")
            return
        ctx.console.print(f"[{theme.ACCENT_DIM}]Resumable runs (most recent first):[/{theme.ACCENT_DIM}]")
        for rid in run_ids[:10]:
            ctx.console.print(f"  [{theme.ACCENT}]{rid}[/{theme.ACCENT}]")
        ctx.console.print(f"[{theme.DIM}]Continue one with /resume <run_id>.[/{theme.DIM}]")
        return

    await resume_run(ctx, args[0])


COMMAND = SlashCommand(
    name="resume",
    summary="Resume an interrupted run (list with no arg)",
    handler=handle_resume,
)
