"""Tests for the python_exec tool (N2 — sandboxed code execution).

Coverage:
- Syntax error fast path (no subprocess)
- result variable side-channel
- Non-zero exit code → is_error
- Timeout + process-group kill
- Environment sanitisation (sensitive vars stripped, HOME overridden)
- Sandbox cleaned up by default
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from neurosurfer.tools.base import ToolContext
from neurosurfer.tools.builtin.python_exec import PythonExecTool
from neurosurfer.tools.builtin.python_exec.env import _SENSITIVE, build_sandbox_env
from neurosurfer.tools.builtin.python_exec.tool import PythonExecArgs
from tests.fakes import ScriptedIO

# ── helpers ───────────────────────────────────────────────────────────────────

def _ctx() -> ToolContext:
    return ToolContext(cwd=Path("/tmp"), io=ScriptedIO())


# ── syntax check ─────────────────────────────────────────────────────────────

class TestSyntaxCheck:
    @pytest.mark.asyncio
    async def test_tool_returns_error_on_syntax_failure(self):
        tool = PythonExecTool()
        args = PythonExecArgs(code="def oops(:\n    pass")
        result = await tool.call(args, _ctx())
        assert result.is_error
        assert "SyntaxError" in result.content


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
