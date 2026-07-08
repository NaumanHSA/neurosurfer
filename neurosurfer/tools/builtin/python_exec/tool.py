"""``PythonExecTool`` — the native :class:`~neurosurfer.tools.base.Tool` wrapper.

Exposes the sandbox as a tool the agent can call.  All heavy lifting is in
:mod:`.sandbox`; this module only handles the Pydantic schema, the
:class:`~neurosurfer.tools.base.ToolResult` formatting, and the async
dispatch.
"""

from __future__ import annotations

import json

from pydantic import BaseModel, Field

from ...base import Tool, ToolContext, ToolResult
from .errors import CodeExecutionError
from .sandbox import SandboxResult, check_syntax, run_in_sandbox

DEFAULT_TIMEOUT = 30
DEFAULT_MEMORY_MB = 512


class PythonExecArgs(BaseModel):
    code: str = Field(description="Python code to execute.")
    timeout: int = Field(
        default=DEFAULT_TIMEOUT,
        ge=1,
        le=300,
        description="Maximum wall-clock execution time in seconds.",
    )
    memory_mb: int = Field(
        default=DEFAULT_MEMORY_MB,
        ge=64,
        le=8192,
        description=(
            "Soft memory limit for the child process in MiB. "
            "Enforced on Linux via resource.setrlimit; ignored on other platforms."
        ),
    )
    keep_sandbox: bool = Field(
        default=False,
        description=(
            "Preserve the sandbox directory after execution. "
            "Set to True when the code writes output files you need to read back."
        ),
    )


class PythonExecTool(Tool):
    """Subprocess-sandboxed Python code execution.

    **Security model:**

    * Runs in a dedicated temp directory (the only cwd the child sees).
    * ``HOME``, ``TMPDIR``, ``TEMP`` point at the sandbox so libraries write there by default.
    * Sensitive env vars (API keys, tokens, credentials, …) are stripped.
    * The process group is killed on timeout — no orphan grandchildren.
    * Memory is capped via ``resource.setrlimit`` on Linux.
    * Code is syntax-checked before spawning (fast failure).
    * The optional ``result`` variable is captured via a side-channel file,
      not stdout parsing.
    * Sandbox directory is cleaned up after each run (unless ``keep_sandbox=True``).

    **What is NOT sandboxed:**

    * Network access.
    * Filesystem reads outside the sandbox directory (``open('/etc/passwd')`` works).
      For true filesystem isolation, wrap this tool in a container.

    **Usage tip:**

    Assign ``result = <value>`` in your code to return a structured Python value
    (list, dict, number, string) that the agent can inspect directly.
    """

    name = "python_exec"
    description = (
        "Execute a Python code snippet in an isolated subprocess sandbox. "
        "Captures stdout, stderr, and exit code. "
        "Assign ``result = <value>`` in the code to return structured data "
        "(list, dict, number, string) the agent can reason about. "
        "Sensitive env vars are stripped; the process group is killed on timeout. "
        "All installed packages are available. "
        "Use for computation, data analysis, file operations, or any Python-specific task."
    )
    input_model = PythonExecArgs

    def is_read_only(self, args: BaseModel) -> bool:
        return False

    async def call(self, args: PythonExecArgs, ctx: ToolContext) -> ToolResult:  # type: ignore[override]
        # ── 1. Syntax check (no subprocess needed for invalid code) ──────────
        syntax_err = check_syntax(args.code)
        if syntax_err:
            return ToolResult(
                content=f"SyntaxError — code not executed:\n{syntax_err}",
                is_error=True,
            )

        # ── 2. Run in sandbox ────────────────────────────────────────────────
        try:
            result: SandboxResult = await run_in_sandbox(
                args.code,
                timeout=args.timeout,
                memory_mb=args.memory_mb,
                keep_sandbox=args.keep_sandbox,
            )
        except CodeExecutionError as exc:
            return ToolResult(
                content=f"Failed to launch Python subprocess: {exc}",
                is_error=True,
            )

        # ── 3. Format output ─────────────────────────────────────────────────
        return _format(result, keep_sandbox=args.keep_sandbox)


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


def _format(r: SandboxResult, *, keep_sandbox: bool) -> ToolResult:
    parts: list[str] = []

    if r.timed_out:
        parts.append("⚠ Execution timed out — process group killed.")

    if r.stdout.strip():
        parts.append(f"stdout:\n{r.stdout.rstrip()}")

    if r.stderr.strip():
        parts.append(f"stderr:\n{r.stderr.rstrip()}")

    if r.result_value is not None:
        try:
            serialised = json.dumps(r.result_value, ensure_ascii=False, default=str)
            parts.append(f"result: {serialised}")
        except Exception:
            parts.append(f"result: {r.result_value!r}")

    parts.append(f"exit_code: {r.exit_code}")

    if keep_sandbox:
        parts.append(f"sandbox: {r.sandbox_path}")

    content = "\n\n".join(parts) if parts else "(no output)"

    is_error = r.timed_out or r.exit_code != 0
    return ToolResult(content=content, is_error=is_error)
