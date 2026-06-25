"""Run a registered Task interactively: collect inputs, resolve the provider,
stream events. Shared by the /task run command and the REPL.

Runs are interruptible (Ctrl-C): the run stops and its transcript is closed.
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
            ctx.console.print(f"\n[{theme.WARN}]Interrupted.[/{theme.WARN}]")
