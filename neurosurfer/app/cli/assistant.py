"""Assistant — the general-purpose front-agent for the neurosurfer REPL.

Wires the full tool pool (file ops, shell, code, web, present_plan,
propose_workflow, and any live MCP tools) to an AgenticLoop and streams
events through the CLI renderer.

Free-form text typed at the REPL prompt is routed here.  When the task is
heavy (heuristic: broad rewrites / mass file edits / risky shell) the agent
starts in plan mode so writes/shell are gated until present_plan is approved.
When the agent calls propose_workflow and the user confirms, control is handed
to the Architect (_build) which runs the deep decompose pipeline.
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path

from . import theme
from .context import CLIContext

# ── assistant persona ─────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are neurosurfer, a general automation assistant running in a terminal.
You help users get things done by reading and writing files, running shell
commands, executing code, and searching the web.

CAPABILITIES
  • Files    — read_file, list_dir, search, write_file, apply_edit
  • Shell    — run_command (gated: you'll ask before running broad or risky commands)
  • Code     — python_exec (sandboxed Python)
  • Web      — web_search, http, browse
  • Planning — present_plan (present a plan and wait for approval before major changes)

ROUTING
  1. Conversational / question  → answer directly; no tools needed.
  2. Simple task (1–3 actions)  → do it.  No ceremony.
  3. Large or risky task        → call present_plan first; summarise what you intend
                                  to do and which files/commands will be affected.
                                  Wait for approval, then execute.
  4. Recurring / pipeline task  → if the user asks for something they'll want to
                                  run again (daily digests, CI workflows, batch jobs),
                                  note that neurosurfer can design and register a
                                  reusable workflow pipeline for it — just say so and
                                  ask if they want that instead.

STYLE
  • Be direct: do the task, don't just describe it.
  • Be concise: one well-targeted action beats a long exploratory chain.
  • Confirm scope for writes/shell before touching many files or running broad commands.
  • If you hit an unexpected error, explain what you found and ask how to proceed.
"""


# ── heaviness heuristic ───────────────────────────────────────────────────────

# Keywords that strongly suggest a broad/risky operation where plan-first is safer.
_HEAVY_RE = re.compile(
    r"\b("
    r"refactor|restructure|rewrite|migrate|rename(?:\s+all)?|replace\s+all"
    r"|delete\s+all|remove\s+all|overwrite"
    r"|across\s+the\s+(repo|codebase|project)"
    r"|every\s+file|all\s+files|all\s+of\s+the"
    r"|large[\s-]scale|mass\s+update|bulk"
    r")\b",
    re.IGNORECASE,
)


def _is_heavy(line: str) -> bool:
    """Heuristic: return True if the request looks like a broad or risky operation."""
    return bool(_HEAVY_RE.search(line))


# ── factory ───────────────────────────────────────────────────────────────────

def build_assistant(ctx: CLIContext, *, plan_mode: bool = False):
    """Return a ready-to-use AgenticLoop wired to the full tool pool.

    Parameters
    ----------
    plan_mode:
        Start the loop in plan mode (writes/shell blocked until present_plan
        is approved).  Set by the heaviness pre-flight for broad/risky requests.

    Raises ``RuntimeError`` if no provider is configured.
    """
    # Import propose_workflow so its register_tool_factory side-effect fires and
    # the tool appears in default_pool().
    import neurosurfer.app.tools.propose_workflow  # noqa: F401

    from neurosurfer.agents.agentic_loop import AgenticLoop
    from neurosurfer.agents.runtime.permissions import Guardrails
    from neurosurfer.llm.registry import resolve_provider
    from neurosurfer.tools.registry import default_pool

    from .io import RichIOHandler

    try:
        provider = resolve_provider(ctx.cfg, ctx.providers)
    except RuntimeError as e:
        raise RuntimeError(str(e)) from e

    io = RichIOHandler(ctx.console)
    pool = default_pool()

    guardrails = Guardrails(
        max_turns=80,
        shell_policy="gated",
        network_policy="open",
        write_scope=["**"],
    )

    return AgenticLoop(
        provider=provider,
        tools=pool,
        system_prompt=_SYSTEM_PROMPT,
        guardrails=guardrails,
        io=io,
        cwd=Path.cwd(),
        mode="plan" if plan_mode else "default",
        verbose=False,
    ), io


# ── REPL handler ──────────────────────────────────────────────────────────────

async def _assist(ctx: CLIContext, line: str) -> None:
    """Run one assistant turn for *line* and stream the output to the console."""
    from .render import stream_events

    heavy = _is_heavy(line)
    if heavy:
        ctx.console.print(
            f"[{theme.DIM}]  ⚠ This looks like a broad change — "
            f"starting in plan mode (present_plan required before writes/shell).[/{theme.DIM}]"
        )

    try:
        agent, io = build_assistant(ctx, plan_mode=heavy)
    except RuntimeError as e:
        ctx.console.print(f"[{theme.ERR}]No usable provider: {e}[/{theme.ERR}]")
        ctx.console.print(
            f"[{theme.DIM}]Configure one with /provider add, then try again.[/{theme.DIM}]"
        )
        return

    try:
        final = await stream_events(ctx.console, agent.run(line), io=io)
    except (KeyboardInterrupt, asyncio.CancelledError):
        ctx.console.print(f"\n[{theme.WARN}]Interrupted.[/{theme.WARN}]")
        return

    # Tier-4 handoff: propose_workflow confirmed → hand off to the Architect.
    if final is not None and final.status == "handoff_workflow":
        intent = final.report or line
        ctx.console.print(
            f"\n[bold {theme.ACCENT}]Architect[/] — handing off to build a reusable workflow.\n"
        )
        from .commands.workflow import _build
        await _build(ctx, intent)
