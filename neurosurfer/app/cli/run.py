"""Run a registered Task interactively: collect inputs, resolve the provider,
stream events. Shared by the /task run command and the session picker.

Runs are interruptible (Ctrl-C): durable state is persisted each turn by the
runner, so an interrupted run can be continued later with ``resume_run``.
"""

from __future__ import annotations

import asyncio

from . import theme
from .context import CLIContext
from .io import RichIOHandler
from .render import stream_events


async def collect_inputs(ctx: CLIContext, task, provided: dict[str, str] | None = None) -> dict[str, str]:
    from neurosurfer.tasks.definition import TaskDefinition  # noqa: F401 (type clarity)

    io = RichIOHandler(ctx.console)
    inputs: dict[str, str] = dict(provided or {})
    for spec in task.inputs:
        if spec.name in inputs and inputs[spec.name]:
            continue
        answer = await io.ask(spec.prompt, options=spec.choices if spec.type == "choice" else None)
        if not answer and spec.default is not None:
            answer = str(spec.default)
        if spec.required and not answer:
            ctx.console.print(f"[{theme.ERR}]Input '{spec.name}' is required.[/{theme.ERR}]")
            raise ValueError(f"missing required input: {spec.name}")
        inputs[spec.name] = answer
    return inputs


def _resolve_provider_or_warn(ctx: CLIContext, task_name: str | None = None):
    from neurosurfer.llm.registry import resolve_provider_for_task

    try:
        return resolve_provider_for_task(ctx.cfg, ctx.providers, ctx.task_providers, task_name)
    except RuntimeError as e:
        ctx.console.print(f"[{theme.ERR}]No usable provider: {e}[/{theme.ERR}]")
        ctx.console.print(f"[{theme.DIM}]Configure one with /provider or in .env.[/{theme.DIM}]")
        return None


async def run_task(ctx: CLIContext, task, provided: dict[str, str] | None = None) -> None:
    from neurosurfer.observability.transcript import EventTranscript, new_run_id
    from neurosurfer.tasks.runner import TaskRunner

    try:
        inputs = await collect_inputs(ctx, task, provided)
    except ValueError:
        return

    io = RichIOHandler(ctx.console)
    provider = _resolve_provider_or_warn(ctx, task.name)
    if provider is None:
        return

    ctx.cfg.ensure_dirs()
    run_id = new_run_id()
    runner = TaskRunner(ctx.cfg, provider=provider)
    with EventTranscript(run_id, ctx.cfg.observability.transcripts_dir()) as transcript:
        events = runner.run(task, inputs, io, transcript=transcript, run_id=run_id)
        try:
            await stream_events(ctx.console, events, io=io)
        except (KeyboardInterrupt, asyncio.CancelledError):
            transcript.record("interrupted")
            ctx.console.print(
                f"\n[{theme.WARN}]Interrupted.[/{theme.WARN}] Durable state saved. "
                f"Resume with [{theme.ACCENT}]/resume {run_id}[/{theme.ACCENT}] "
                f"(or [{theme.ACCENT}]neurosurfer resume {run_id}[/{theme.ACCENT}])."
            )


async def resume_run(ctx: CLIContext, run_id: str) -> None:
    """Continue an interrupted run: reload its task/inputs + durable state."""
    from neurosurfer.agents.context.durable_state import DurableState
    from neurosurfer.observability.transcript import EventTranscript
    from neurosurfer.tasks.runner import TaskRunner

    ctx.cfg.ensure_dirs()

    # 1. Recover the original task + inputs from the run's transcript.
    meta = _read_run_meta(ctx, run_id)
    if meta is None:
        ctx.console.print(
            f"[{theme.ERR}]No resumable run '{run_id}'.[/{theme.ERR}] "
            f"List recent runs with [{theme.ACCENT}]/resume[/{theme.ACCENT}]."
        )
        return
    task_name, inputs = meta

    try:
        task = ctx.registry.get(task_name)
    except Exception as e:  # noqa: BLE001
        ctx.console.print(f"[{theme.ERR}]Task '{task_name}' no longer available: {e}[/{theme.ERR}]")
        return

    # 2. Reload durable state (plan/todos/decisions) if present.
    state_path = ctx.cfg.observability.runs_state_dir() / f"{run_id}.json"
    durable = DurableState.load(state_path) if state_path.exists() else DurableState()

    io = RichIOHandler(ctx.console)
    provider = _resolve_provider_or_warn(ctx, task_name)
    if provider is None:
        return

    ctx.console.print(f"[{theme.DIM}]Resuming run {run_id} of task '{task_name}'…[/{theme.DIM}]")
    runner = TaskRunner(ctx.cfg, provider=provider)
    with EventTranscript(run_id, ctx.cfg.observability.transcripts_dir()) as transcript:
        events = runner.run(
            task, inputs, io, transcript=transcript,
            run_id=run_id, durable=durable, resume=True,
        )
        try:
            await stream_events(ctx.console, events, io=io)
        except (KeyboardInterrupt, asyncio.CancelledError):
            transcript.record("interrupted")
            ctx.console.print(
                f"\n[{theme.WARN}]Interrupted again.[/{theme.WARN}] State saved; "
                f"resume with [{theme.ACCENT}]/resume {run_id}[/{theme.ACCENT}]."
            )


def _read_run_meta(ctx: CLIContext, run_id: str) -> tuple[str, dict[str, str]] | None:
    """Return (task_name, inputs) from a run's transcript, or None if not found."""
    import json

    path = ctx.cfg.observability.transcripts_dir() / f"{run_id}.jsonl"
    if not path.exists():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        if entry.get("type") in ("run_start", "run_resume"):
            return entry.get("task", ""), entry.get("inputs", {}) or {}
    return None


def list_resumable(ctx: CLIContext) -> list[str]:
    """Run ids that have a persisted durable-state snapshot (most recent first)."""
    state_dir = ctx.cfg.observability.runs_state_dir()
    if not state_dir.exists():
        return []
    files = sorted(state_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return [p.stem for p in files]
