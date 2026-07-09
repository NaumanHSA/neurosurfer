"""``PythonExecTool`` — the native :class:`~neurosurfer.tools.base.Tool` wrapper.

Exposes the sandbox as a tool the agent can call.  All heavy lifting is in
:mod:`.sandbox`; this module only handles the Pydantic schema, the
:class:`~neurosurfer.tools.base.ToolResult` formatting, and the async
dispatch.
"""

from __future__ import annotations

import asyncio
import json
import re

from pydantic import BaseModel, Field

from ...base import Tool, ToolContext, ToolResult
from .errors import CodeExecutionError
from .interpreter import resolve_interpreter
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
    """Python code execution in a managed environment.

    **Execution model:**

    * Runs via a resolved interpreter: a session-pinned env (``set_python_env``) >
      ``NEUROSURFER_PYENV`` > the neurosurfer-managed venv (``~/.neurosurfer/venv``,
      auto-provisioned with common data/PDF/image libraries) > the host interpreter.
    * The process **cwd is the current working directory** — relative file writes
      (e.g. an exported PDF or chart) land in the project, not a throwaway sandbox.
    * ``HOME``/``TMPDIR`` still point at a scratch directory used only for the code's
      side-channel result file; that scratch directory is deleted afterwards
      (unless ``keep_sandbox=True``) but never contains your output files.
    * Sensitive env vars (API keys, tokens, credentials, …) are stripped.
    * The process group is killed on timeout — no orphan grandchildren.
    * Memory is capped via ``resource.setrlimit`` on Linux.
    * Code is syntax-checked before spawning (fast failure).
    * The optional ``result`` variable is captured via a side-channel file,
      not stdout parsing.

    **What is NOT sandboxed:**

    * Network access.
    * Filesystem access — the child sees the real working directory.
      For true filesystem isolation, wrap this tool in a container.

    **Usage tip:**

    Assign ``result = <value>`` in your code to return a structured Python value
    (list, dict, number, string) that the agent can inspect directly. If a run fails
    with ``ModuleNotFoundError``, call ``install_python_package`` to add it (gated by
    user approval), then re-run.
    """

    name = "python_exec"
    description = (
        "Execute a Python code snippet using the managed Python environment. "
        "Captures stdout, stderr, and exit code. Runs with the current working "
        "directory as cwd, so relative file writes (charts, PDFs, exports) land "
        "where the user expects. "
        "Assign ``result = <value>`` in the code to return structured data "
        "(list, dict, number, string) the agent can reason about. "
        "Sensitive env vars are stripped; the process group is killed on timeout. "
        "If a package is missing, use install_python_package to add it. "
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
        # Resolving the interpreter can block on first-run venv provisioning
        # (installing numpy/pandas/matplotlib takes real time) — keep that off
        # the event loop so the CLI doesn't appear to hang.
        interpreter = await asyncio.to_thread(resolve_interpreter, ctx)
        try:
            result: SandboxResult = await run_in_sandbox(
                args.code,
                timeout=args.timeout,
                memory_mb=args.memory_mb,
                keep_sandbox=args.keep_sandbox,
                interpreter=interpreter,
                workdir=ctx.cwd,
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

_MODULE_NOT_FOUND_RE = re.compile(r"ModuleNotFoundError: No module named '([^']+)'")


def _missing_module_hint(stderr: str) -> str | None:
    match = _MODULE_NOT_FOUND_RE.search(stderr)
    if match is None:
        return None
    module = match.group(1).split(".")[0]
    return (
        f"Hint: '{module}' is not installed. Call "
        f"install_python_package(packages=['{module}'], reason=...) then re-run this code."
    )


def _format(r: SandboxResult, *, keep_sandbox: bool) -> ToolResult:
    parts: list[str] = []

    if r.timed_out:
        parts.append("⚠ Execution timed out — process group killed.")

    if r.stdout.strip():
        parts.append(f"stdout:\n{r.stdout.rstrip()}")

    if r.stderr.strip():
        parts.append(f"stderr:\n{r.stderr.rstrip()}")
        hint = _missing_module_hint(r.stderr)
        if hint:
            parts.append(hint)

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
