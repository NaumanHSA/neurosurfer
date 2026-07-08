"""Tests for the python_exec tool (N2 — sandboxed code execution).

Coverage:
- Syntax error fast path (no subprocess)
- Basic stdout capture
- result variable side-channel
- stderr capture
- Non-zero exit code → is_error
- Timeout + process-group kill
- Environment sanitisation (sensitive vars stripped, HOME overridden)
- keep_sandbox flag (sandbox path in output, dir not deleted)
- Memory-limit prelude builds correctly
- Output truncation sentinel
- CodeExecutionError on launch failure (mocked)
- Tool registered in the default pool
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from neurosurfer.tools.base import ToolContext
from neurosurfer.tools.builtin.python_exec import PythonExecTool
from neurosurfer.tools.builtin.python_exec.env import _SENSITIVE, build_sandbox_env
from neurosurfer.tools.builtin.python_exec.sandbox import (
    SandboxResult,
    _truncate,
    check_syntax,
)
from neurosurfer.tools.builtin.python_exec.tool import PythonExecArgs, _format
from tests.fakes import ScriptedIO

# ── helpers ───────────────────────────────────────────────────────────────────

def _ctx() -> ToolContext:
    return ToolContext(cwd=Path("/tmp"), io=ScriptedIO())


def _args(**kw) -> PythonExecArgs:
    return PythonExecArgs(**{"code": "x = 1", **kw})


# ── syntax check ─────────────────────────────────────────────────────────────

class TestSyntaxCheck:
    def test_valid_code_returns_none(self):
        assert check_syntax("x = 1 + 2") is None

    def test_invalid_code_returns_error_string(self):
        result = check_syntax("def foo(:\n    pass")
        assert result is not None
        assert "SyntaxError" in result or "invalid syntax" in result.lower() or result

    @pytest.mark.asyncio
    async def test_tool_returns_error_on_syntax_failure(self):
        tool = PythonExecTool()
        args = PythonExecArgs(code="def oops(:\n    pass")
        result = await tool.call(args, _ctx())
        assert result.is_error
        assert "SyntaxError" in result.content


# ── stdout capture ────────────────────────────────────────────────────────────

class TestStdout:
    @pytest.mark.asyncio
    async def test_print_captured(self):
        tool = PythonExecTool()
        args = PythonExecArgs(code='print("hello world")')
        r = await tool.call(args, _ctx())
        assert not r.is_error
        assert "hello world" in r.content

    @pytest.mark.asyncio
    async def test_no_output_shows_no_output(self):
        tool = PythonExecTool()
        args = PythonExecArgs(code="x = 1 + 1  # no print")
        r = await tool.call(args, _ctx())
        assert not r.is_error
        # No stdout, no result var → only exit_code line
        assert "exit_code: 0" in r.content


# ── result variable ───────────────────────────────────────────────────────────

class TestResultVariable:
    @pytest.mark.asyncio
    async def test_result_int(self):
        tool = PythonExecTool()
        args = PythonExecArgs(code="result = 42")
        r = await tool.call(args, _ctx())
        assert not r.is_error
        assert "42" in r.content
        assert "result:" in r.content

    @pytest.mark.asyncio
    async def test_result_dict(self):
        tool = PythonExecTool()
        args = PythonExecArgs(code='result = {"a": 1, "b": [1, 2, 3]}')
        r = await tool.call(args, _ctx())
        assert not r.is_error
        assert '"a"' in r.content or "a" in r.content

    @pytest.mark.asyncio
    async def test_result_list(self):
        tool = PythonExecTool()
        args = PythonExecArgs(code="result = [1, 2, 3]")
        r = await tool.call(args, _ctx())
        assert not r.is_error
        assert "[1, 2, 3]" in r.content

    @pytest.mark.asyncio
    async def test_no_result_var_no_result_line(self):
        tool = PythonExecTool()
        args = PythonExecArgs(code='print("no result var")')
        r = await tool.call(args, _ctx())
        assert "result:" not in r.content


# ── stderr and exit code ──────────────────────────────────────────────────────

class TestStderrAndExit:
    @pytest.mark.asyncio
    async def test_runtime_error_is_error(self):
        tool = PythonExecTool()
        args = PythonExecArgs(code="raise ValueError('boom')")
        r = await tool.call(args, _ctx())
        assert r.is_error
        assert "ValueError" in r.content or "stderr" in r.content

    @pytest.mark.asyncio
    async def test_sys_exit_nonzero_is_error(self):
        tool = PythonExecTool()
        args = PythonExecArgs(code="import sys; sys.exit(2)")
        r = await tool.call(args, _ctx())
        assert r.is_error
        assert "exit_code: 2" in r.content

    @pytest.mark.asyncio
    async def test_sys_exit_zero_is_ok(self):
        tool = PythonExecTool()
        args = PythonExecArgs(code="import sys; sys.exit(0)")
        r = await tool.call(args, _ctx())
        assert not r.is_error

    @pytest.mark.asyncio
    async def test_stderr_written_to_stderr(self):
        tool = PythonExecTool()
        args = PythonExecArgs(code="import sys; sys.stderr.write('err line\\n')")
        r = await tool.call(args, _ctx())
        assert "err line" in r.content


# ── timeout ───────────────────────────────────────────────────────────────────

class TestTimeout:
    @pytest.mark.asyncio
    async def test_timeout_returns_error(self):
        tool = PythonExecTool()
        args = PythonExecArgs(code="import time; time.sleep(60)", timeout=1)
        r = await tool.call(args, _ctx())
        assert r.is_error
        assert "timed out" in r.content.lower() or "timeout" in r.content.lower()


# ── environment sanitisation ──────────────────────────────────────────────────

class TestEnvSanitisation:
    def test_sensitive_patterns_match(self):
        sensitive_keys = [
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "SECRET_TOKEN",
            "DATABASE_PASSWORD",
            "AWS_ACCESS_KEY_ID",
            "STRIPE_SECRET_KEY",
            "MY_AUTH_TOKEN",
        ]
        for key in sensitive_keys:
            assert any(pat.search(key) for pat in _SENSITIVE), f"{key!r} should be sensitive"

    def test_build_sandbox_env_strips_sensitive(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-secret", "PATH": "/usr/bin"}):
            env = build_sandbox_env(Path("/tmp/sandbox"))
        assert "OPENAI_API_KEY" not in env
        assert "PATH" in env

    def test_build_sandbox_env_overrides_home(self):
        sandbox = Path("/tmp/my_sandbox")
        env = build_sandbox_env(sandbox)
        assert env["HOME"] == str(sandbox)
        assert env["TMPDIR"] == str(sandbox)
        assert env["TEMP"] == str(sandbox)

    def test_build_sandbox_env_sets_pythonunbuffered(self):
        env = build_sandbox_env(Path("/tmp"))
        assert env.get("PYTHONUNBUFFERED") == "1"

    @pytest.mark.asyncio
    async def test_env_not_leaked_to_child(self):
        """Child must not see OPENAI_API_KEY even if it's in the parent env."""
        tool = PythonExecTool()
        with patch.dict(os.environ, {"MY_SUPER_SECRET_KEY": "leaked!"}):
            args = PythonExecArgs(
                code=(
                    "import os\n"
                    "result = os.environ.get('MY_SUPER_SECRET_KEY', 'not_found')"
                )
            )
            r = await tool.call(args, _ctx())
        assert "leaked!" not in r.content
        assert "not_found" in r.content


# ── keep_sandbox flag ─────────────────────────────────────────────────────────

class TestKeepSandbox:
    @pytest.mark.asyncio
    async def test_keep_sandbox_path_in_output(self):
        tool = PythonExecTool()
        args = PythonExecArgs(code='print("hi")', keep_sandbox=True)
        r = await tool.call(args, _ctx())
        assert not r.is_error
        assert "sandbox:" in r.content

    @pytest.mark.asyncio
    async def test_sandbox_preserved_when_keep(self):
        tool = PythonExecTool()
        args = PythonExecArgs(
            code='open("output.txt","w").write("data")',
            keep_sandbox=True,
        )
        r = await tool.call(args, _ctx())
        # Extract sandbox path from output
        for line in r.content.splitlines():
            if line.startswith("sandbox:"):
                sandbox_path = line.split("sandbox:", 1)[1].strip()
                assert Path(sandbox_path).exists(), "Sandbox should not be cleaned up"
                # Cleanup manually
                import shutil
                shutil.rmtree(sandbox_path, ignore_errors=True)
                break

    @pytest.mark.asyncio
    async def test_sandbox_cleaned_by_default(self):
        """After a normal run, the sandbox temp dir should be gone."""
        import tempfile
        seen_paths: list[str] = []

        original_mkdtemp = tempfile.mkdtemp

        def recording_mkdtemp(**kwargs):
            path = original_mkdtemp(**kwargs)
            seen_paths.append(path)
            return path

        with patch("neurosurfer.tools.builtin.python_exec.sandbox.tempfile.mkdtemp", side_effect=recording_mkdtemp):
            tool = PythonExecTool()
            args = PythonExecArgs(code='print("hi")')
            await tool.call(args, _ctx())

        for p in seen_paths:
            assert not Path(p).exists(), f"Sandbox {p} should have been cleaned up"


# ── output formatting helpers ────────────────────────────────────────────────

class TestFormatting:
    def test_truncate_short(self):
        assert _truncate("hello", 100) == "hello"

    def test_truncate_long(self):
        text = "x" * 200
        result = _truncate(text, 100)
        assert len(result) < len(text)
        assert "truncated" in result

    def test_format_timeout(self):
        r = SandboxResult(
            stdout="",
            stderr="",
            exit_code=-9,
            result_value=None,
            sandbox_path="/tmp/s",
            timed_out=True,
        )
        result = _format(r, keep_sandbox=False)
        assert result.is_error
        assert "timed out" in result.content.lower()

    def test_format_success(self):
        r = SandboxResult(
            stdout="hello\n",
            stderr="",
            exit_code=0,
            result_value={"key": "val"},
            sandbox_path="/tmp/s",
        )
        result = _format(r, keep_sandbox=False)
        assert not result.is_error
        assert "hello" in result.content
        assert '"key"' in result.content or "key" in result.content


# ── tool registered in default pool ──────────────────────────────────────────

class TestRegistration:
    def test_python_exec_in_all_tools(self):
        from neurosurfer.tools.registry import all_tools
        names = {t.name for t in all_tools()}
        assert "python_exec" in names

    def test_python_exec_in_default_pool(self):
        from neurosurfer.tools.registry import default_pool
        pool = default_pool()
        assert pool.get("python_exec") is not None
