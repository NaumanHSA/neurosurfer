from __future__ import annotations

import asyncio

from pydantic import BaseModel, Field

from ..base import Tool, ToolContext, ToolResult

MAX_OUTPUT_CHARS = 30_000
DEFAULT_TIMEOUT = 120


class RunCommandArgs(BaseModel):
    command: str = Field(description="Shell command to execute.")
    description: str = Field(
        default="", description="Short description of what the command does (for the gate)."
    )
    timeout: int = Field(default=DEFAULT_TIMEOUT, ge=1, le=600)


class RunCommandTool(Tool):
    name = "run_command"
    description = (
        "Run a shell command in the working directory and return its combined "
        "stdout/stderr and exit code. Subject to the Task's shell policy gate."
    )
    input_model = RunCommandArgs

    # Never read-only / never concurrency-safe — the loop runs these serially.
    def is_read_only(self, args: BaseModel) -> bool:
        return False

    def progress_message(self, args: dict) -> str:
        cmd = (args.get("command") or "").strip()
        if len(cmd) > 60:
            cmd = cmd[:57] + "…"
        return f"Running `{cmd}`…" if cmd else "Running command…"

    async def call(self, args: RunCommandArgs, ctx: ToolContext) -> ToolResult:  # type: ignore[override]
        try:
            proc = await asyncio.create_subprocess_shell(
                args.command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=str(ctx.cwd),
            )
        except OSError as e:
            return ToolResult.error(f"Failed to launch: {e}")
        try:
            out, _ = await asyncio.wait_for(proc.communicate(), timeout=args.timeout)
        except TimeoutError:
            proc.kill()
            await proc.wait()
            return ToolResult.error(f"Command timed out after {args.timeout}s: {args.command}")

        text = out.decode("utf-8", errors="replace")
        if len(text) > MAX_OUTPUT_CHARS:
            text = text[:MAX_OUTPUT_CHARS] + "\n… [output truncated]"
        code = proc.returncode or 0
        header = f"$ {args.command}\n(exit {code})\n"
        if code != 0:
            return ToolResult(content=header + text, is_error=True)
        return ToolResult.ok(header + (text or "(no output)"))
