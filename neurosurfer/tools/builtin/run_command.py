from __future__ import annotations

import asyncio
import os
import signal
import time
import uuid
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field

from ..base import Tool, ToolContext, ToolResult
from ..utils import resolve_path

MAX_OUTPUT_CHARS = 30_000
DEFAULT_TIMEOUT = 120
_JOBS_KEY = "background_jobs"


@dataclass
class _BackgroundJob:
    proc: asyncio.subprocess.Process
    command: str
    output: bytearray = field(default_factory=bytearray)
    started_at: float = field(default_factory=time.time)
    done: bool = False
    exit_code: int | None = None


def _jobs(ctx: ToolContext) -> dict[str, _BackgroundJob]:
    return ctx.extra.setdefault(_JOBS_KEY, {})


async def _pump_output(job: _BackgroundJob) -> None:
    """Drain a background job's stdout as it runs, capping buffered size."""
    assert job.proc.stdout is not None
    cap = MAX_OUTPUT_CHARS * 4
    while True:
        chunk = await job.proc.stdout.read(4096)
        if not chunk:
            break
        job.output.extend(chunk)
        if len(job.output) > cap:
            del job.output[: len(job.output) - cap]
    job.exit_code = await job.proc.wait()
    job.done = True


def _kill_group(pid: int) -> None:
    try:
        os.killpg(os.getpgid(pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            os.kill(pid, signal.SIGKILL)
        except (ProcessLookupError, OSError):
            pass


def _decode_and_truncate(data: bytes) -> str:
    text = data.decode("utf-8", errors="replace")
    if len(text) > MAX_OUTPUT_CHARS:
        text = text[:MAX_OUTPUT_CHARS] + "\n… [output truncated]"
    return text


class RunCommandArgs(BaseModel):
    command: str = Field(
        default="", description="Shell command to execute. Required for action='run'."
    )
    description: str = Field(
        default="", description="Short description of what the command does (for the gate)."
    )
    timeout: int = Field(default=DEFAULT_TIMEOUT, ge=1, le=600)
    cwd: str | None = Field(
        default=None,
        description="Directory to run the command in (relative to the working directory, "
        "or absolute). Defaults to the working directory.",
    )
    run_in_background: bool = Field(
        default=False,
        description="Start the command and return immediately with a job_id instead of "
        "waiting for it to finish (e.g. a dev server or watcher). Check on it with "
        "action='status' and stop it with action='kill'.",
    )
    action: Literal["run", "status", "kill"] = Field(
        default="run",
        description="'run' starts a command (default). 'status' reads a background job's "
        "output/state so far (job_id required). 'kill' stops a background job (job_id required).",
    )
    job_id: str | None = Field(
        default=None,
        description="Background job id — required for action='status' or 'kill'; "
        "returned when a job is started with run_in_background=true.",
    )


class RunCommandTool(Tool):
    name = "run_command"
    description = (
        "Run a shell command in the working directory (or `cwd`, if given) and return its "
        "combined stdout/stderr and exit code. Subject to the Task's shell policy gate. "
        "Pass run_in_background=true for long-running commands (dev servers, watchers); "
        "then use action='status'/'kill' with the returned job_id to check on or stop it."
    )
    input_model = RunCommandArgs

    # Never read-only / never concurrency-safe — the loop runs these serially.
    def is_read_only(self, args: BaseModel) -> bool:
        return False

    def progress_message(self, args: dict) -> str:
        action = args.get("action", "run")
        if action != "run":
            return f"{action.capitalize()} background job {args.get('job_id', '')}…"
        cmd = (args.get("command") or "").strip()
        if len(cmd) > 60:
            cmd = cmd[:57] + "…"
        return f"Running `{cmd}`…" if cmd else "Running command…"

    async def call(self, args: RunCommandArgs, ctx: ToolContext) -> ToolResult:  # type: ignore[override]
        if args.action in ("status", "kill"):
            return await self._job_action(args, ctx)

        if not args.command:
            return ToolResult.error("command is required for action='run'.")

        workdir = resolve_path(ctx.cwd, args.cwd) if args.cwd else ctx.cwd
        if not workdir.is_dir():
            return ToolResult.error(f"cwd not found or not a directory: {args.cwd}")

        try:
            proc = await asyncio.create_subprocess_shell(
                args.command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=str(workdir),
                start_new_session=True,
            )
        except OSError as e:
            return ToolResult.error(f"Failed to launch: {e}")

        if args.run_in_background:
            job_id = uuid.uuid4().hex[:8]
            job = _BackgroundJob(proc=proc, command=args.command)
            _jobs(ctx)[job_id] = job
            asyncio.create_task(_pump_output(job))
            return ToolResult.ok(
                f"Started background job {job_id}: {args.command}\n"
                f"Use run_command(action='status', job_id='{job_id}') for output, "
                f"or action='kill' to stop it."
            )

        try:
            out, _ = await asyncio.wait_for(proc.communicate(), timeout=args.timeout)
        except TimeoutError:
            _kill_group(proc.pid)
            await proc.wait()
            return ToolResult.error(f"Command timed out after {args.timeout}s: {args.command}")

        text = _decode_and_truncate(out)
        code = proc.returncode or 0
        header = f"$ {args.command}\n(exit {code})\n"
        if code != 0:
            return ToolResult(content=header + text, is_error=True)
        return ToolResult.ok(header + (text or "(no output)"))

    async def _job_action(self, args: RunCommandArgs, ctx: ToolContext) -> ToolResult:
        if not args.job_id:
            return ToolResult.error(f"action='{args.action}' requires job_id.")
        job = _jobs(ctx).get(args.job_id)
        if job is None:
            return ToolResult.error(f"No background job with id {args.job_id!r}.")

        if args.action == "kill":
            _kill_group(job.proc.pid)
            return ToolResult.ok(f"Killed job {args.job_id} ({job.command}).")

        text = _decode_and_truncate(bytes(job.output))
        state = "exited" if job.done else "running"
        header = f"job {args.job_id} [{state}] $ {job.command}\n"
        if job.done:
            header += f"(exit {job.exit_code})\n"
        return ToolResult.ok(header + (text or "(no output yet)"))
