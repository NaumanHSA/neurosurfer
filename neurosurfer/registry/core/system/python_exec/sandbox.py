"""Core sandbox execution logic.

Responsibilities:
- Wrap user code in a template that captures the optional ``result`` variable
  via a side-channel temp file (no stdout pollution).
- Apply resource limits (memory cap on Linux via ``resource.setrlimit``).
- Launch the subprocess in a **new session** so that the entire process group
  can be killed on timeout (no orphan grandchildren).
- Sanitise output sizes.
- Return a structured ``SandboxResult``.
"""

from __future__ import annotations

import ast
import asyncio
import json
import os
import shutil
import signal
import sys
import tempfile
import textwrap
from dataclasses import dataclass
from pathlib import Path

from .env import build_sandbox_env
from .errors import CodeExecutionError

MAX_STDOUT_CHARS = 20_000
MAX_STDERR_CHARS = 8_000

# ---------------------------------------------------------------------------
# Wrapper template
# ---------------------------------------------------------------------------
# The user's code is indented and embedded between try/finally so that:
# 1. ``result`` (if defined) is serialised to a side-channel file.
# 2. ``SystemExit`` is re-raised so that sys.exit() works correctly.
# 3. Any other exception propagates normally (full traceback to stderr).
#
# NOTE: written at column-0 on purpose — {indented_code} has no leading
# whitespace, which would fool textwrap.dedent into not stripping anything.
_WRAPPER_TEMPLATE = """\
import sys as _sys, json as _json, os as _os

_RESULT_FILE = {result_file!r}
_result_written = False

try:
{indented_code}
except SystemExit as _se:
    _sys.exit(_se.code)
except Exception:
    raise
finally:
    if not _result_written and "result" in dir():
        try:
            _serialised = _json.dumps(result, default=str)  # noqa: F821
            with open(_RESULT_FILE, "w", encoding="utf-8") as _rf:
                _rf.write(_serialised)
            _result_written = True
        except Exception:
            pass
"""

# ---------------------------------------------------------------------------
# Resource-limit prelude
# ---------------------------------------------------------------------------
# Injected as ``python -c <prelude> <script_path>`` so tracebacks still
# reference the script's line numbers (runpy.run_path preserves them).
_RESOURCE_PRELUDE = textwrap.dedent(
    """\
    import sys, runpy
    try:
        import resource
        _limit = {limit_bytes}
        resource.setrlimit(resource.RLIMIT_AS, (_limit, _limit))
    except Exception:
        pass  # resource module unavailable (Windows / macOS without limits)
    runpy.run_path(sys.argv[1], run_name="__main__")
    """
)


@dataclass
class SandboxResult:
    stdout: str
    stderr: str
    exit_code: int
    result_value: object | None  # deserialised Python value from `result =`
    sandbox_path: str
    timed_out: bool = False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def check_syntax(code: str) -> str | None:
    """Return a human-readable error string if ``code`` has a syntax error, else ``None``."""
    try:
        ast.parse(code)
        return None
    except SyntaxError as exc:
        return str(exc)


async def run_in_sandbox(
    code: str,
    *,
    timeout: int,
    memory_mb: int,
    keep_sandbox: bool,
    interpreter: str = sys.executable,
    workdir: Path | None = None,
) -> SandboxResult:
    """Execute ``code`` with ``interpreter``, using ``workdir`` as the process cwd.

    A scratch temp directory holds only the wrapper script and the ``result``
    side-channel file — it is cleaned up afterwards (unless ``keep_sandbox`` is
    True) and is *not* where the child process runs. The child's cwd is
    ``workdir`` (the caller's project directory by default), so relative file
    writes — e.g. an exported PDF — land where the user expects and survive
    scratch cleanup.
    """
    sandbox = Path(tempfile.mkdtemp(prefix="ns_pyexec_"))
    try:
        return await _execute(
            code,
            sandbox=sandbox,
            timeout=timeout,
            memory_mb=memory_mb,
            interpreter=interpreter,
            workdir=workdir if workdir is not None else Path.cwd(),
        )
    finally:
        if not keep_sandbox:
            shutil.rmtree(sandbox, ignore_errors=True)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


async def _execute(
    code: str,
    *,
    sandbox: Path,
    timeout: int,
    memory_mb: int,
    interpreter: str,
    workdir: Path,
) -> SandboxResult:
    result_file = sandbox / "_result.json"
    script_file = sandbox / "_script.py"

    # Write wrapped user code
    indented = textwrap.indent(code, "    ")
    wrapped = _WRAPPER_TEMPLATE.format(
        result_file=str(result_file),
        indented_code=indented,
    )
    script_file.write_text(wrapped, encoding="utf-8")

    # Build child command: prelude sets resource limits then runs the script
    limit_bytes = memory_mb * 1024 * 1024
    prelude = _RESOURCE_PRELUDE.format(limit_bytes=limit_bytes)
    cmd = [interpreter, "-c", prelude, str(script_file)]

    # HOME/TMPDIR still point at the scratch sandbox (matplotlib config, etc.);
    # the process cwd is the caller's working directory, not the scratch dir.
    env = build_sandbox_env(sandbox)

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(workdir),
            env=env,
            # New session = new process group; lets us SIGKILL the whole tree.
            start_new_session=True,
        )
    except OSError as exc:
        raise CodeExecutionError(f"Failed to launch Python subprocess: {exc}") from exc

    timed_out = False
    try:
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            proc.communicate(), timeout=timeout
        )
    except TimeoutError:
        timed_out = True
        _kill_group(proc.pid)
        await proc.wait()
        stdout_bytes = b""
        stderr_bytes = b""

    exit_code = proc.returncode or 0

    result_value: object | None = None
    if not timed_out and result_file.exists():
        try:
            raw = result_file.read_text(encoding="utf-8").strip()
            if raw:
                result_value = json.loads(raw)
        except Exception:
            pass

    return SandboxResult(
        stdout=_truncate(stdout_bytes.decode("utf-8", errors="replace"), MAX_STDOUT_CHARS),
        stderr=_truncate(stderr_bytes.decode("utf-8", errors="replace"), MAX_STDERR_CHARS),
        exit_code=exit_code,
        result_value=result_value,
        sandbox_path=str(sandbox),
        timed_out=timed_out,
    )


def _kill_group(pid: int) -> None:
    """SIGKILL the entire process group (child + any grandchildren)."""
    try:
        os.killpg(os.getpgid(pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            os.kill(pid, signal.SIGKILL)
        except (ProcessLookupError, OSError):
            pass


def _truncate(text: str, limit: int) -> str:
    if len(text) > limit:
        omitted = len(text) - limit
        return text[:limit] + f"\n… [truncated — {omitted} chars omitted]"
    return text
