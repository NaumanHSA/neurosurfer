"""Tests for the managed Python environment, interpreter resolution, gated package
installation, the environment-context prompt section, and user-selected envs
(Workstreams A/B/D of the CLI-agent Python tooling plan).
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from neurosurfer.agents.agentic_loop import AgenticLoop
from neurosurfer.agents.context.durable_state import DurableState
from neurosurfer.agents.runtime.permissions import Guardrails
from neurosurfer.llm.providers.anthropic import AnthropicProvider
from neurosurfer.prompts.environment import environment_section
from neurosurfer.tools import default_pool
from neurosurfer.tools.base import ToolContext
from neurosurfer.tools.builtin.install_package import (
    InstallPythonPackageArgs,
    InstallPythonPackageTool,
)
from neurosurfer.tools.builtin.python_exec.interpreter import (
    EnvResolutionError,
    describe_interpreter,
    resolve_env_spec,
    resolve_interpreter,
)
from neurosurfer.tools.builtin.set_python_env import SetPythonEnvArgs, SetPythonEnvTool

from .fakes import FakeAnthropicClient, ScriptedIO


def _ctx(tmp_path: Path, io: ScriptedIO | None = None, durable: DurableState | None = None) -> ToolContext:
    return ToolContext(cwd=tmp_path, io=io or ScriptedIO(), durable=durable)


# ── interpreter resolution precedence ────────────────────────────────────────

class TestResolveInterpreter:
    def test_session_pin_wins(self, tmp_path):
        ctx = _ctx(tmp_path)
        ctx.extra["python_interpreter"] = "/pinned/python"
        assert resolve_interpreter(ctx) == "/pinned/python"

    def test_describe_reports_pin_as_ready(self, tmp_path):
        ctx = _ctx(tmp_path)
        ctx.extra["python_interpreter"] = "/pinned/python"
        interp, ready = describe_interpreter(ctx)
        assert interp == "/pinned/python"
        assert ready is True

    def test_describe_never_provisions(self, tmp_path, monkeypatch):
        """describe_interpreter must not shell out to create the managed venv —
        it's called on every prompt turn and must stay cheap."""
        monkeypatch.setenv("NEUROSURFER_HOME", str(tmp_path / "ns_home"))
        with patch(
            "neurosurfer.tools.builtin.python_exec.interpreter.ensure_managed_venv"
        ) as ensure:
            interp, ready = describe_interpreter(_ctx(tmp_path))
        ensure.assert_not_called()
        assert ready is False
        assert "venv" in interp


# ── resolve_env_spec ──────────────────────────────────────────────────────────

class TestResolveEnvSpec:
    def test_managed_spec_provisions(self):
        with patch(
            "neurosurfer.tools.builtin.python_exec.interpreter.ensure_managed_venv",
            return_value=Path("/managed/bin/python"),
        ) as ensure:
            result = resolve_env_spec("managed")
        assert result == "/managed/bin/python"
        ensure.assert_called_once()

    def test_path_spec_to_interpreter_file(self, tmp_path):
        interp = tmp_path / "myenv"
        interp.write_text("#!/bin/sh")
        assert resolve_env_spec(str(interp)) == str(interp)

    def test_path_spec_to_venv_dir(self, tmp_path):
        venv_dir = tmp_path / "myvenv"
        bin_dir = venv_dir / "bin"
        bin_dir.mkdir(parents=True)
        (bin_dir / "python").write_text("#!/bin/sh")
        assert resolve_env_spec(str(venv_dir)) == str(bin_dir / "python")

    def test_conda_spec_found(self):
        payload = {"envs": ["/opt/conda/envs/myenv"]}
        with patch(
            "neurosurfer.tools.builtin.python_exec.interpreter.subprocess.run",
            return_value=MagicMock(returncode=0, stdout=json.dumps(payload)),
        ), patch("pathlib.Path.exists", return_value=True):
            result = resolve_env_spec("conda:myenv")
        assert result == "/opt/conda/envs/myenv/bin/python"

    def test_conda_spec_not_found(self):
        payload = {"envs": ["/opt/conda/envs/other"]}
        with patch(
            "neurosurfer.tools.builtin.python_exec.interpreter.subprocess.run",
            return_value=MagicMock(returncode=0, stdout=json.dumps(payload)),
        ):
            with pytest.raises(EnvResolutionError):
                resolve_env_spec("conda:missing")

    def test_bad_spec_raises(self, tmp_path):
        with pytest.raises(EnvResolutionError):
            resolve_env_spec(str(tmp_path / "does_not_exist"))


# ── install_python_package ────────────────────────────────────────────────────

class TestInstallPythonPackage:
    @pytest.mark.asyncio
    async def test_skips_already_importable(self, tmp_path):
        tool = InstallPythonPackageTool()
        ctx = _ctx(tmp_path)
        with (
            patch(
                "neurosurfer.tools.builtin.install_package.resolve_interpreter",
                return_value="/usr/bin/python3",
            ),
            patch(
                "neurosurfer.tools.builtin.install_package._is_importable",
                new=AsyncMock(return_value=True),
            ),
        ):
            r = await tool.call(InstallPythonPackageArgs(packages=["os"]), ctx)
        assert not r.is_error
        assert "Already available" in r.content

    @pytest.mark.asyncio
    async def test_denial_returns_feedback(self, tmp_path):
        tool = InstallPythonPackageTool()
        io = ScriptedIO(approve_shell=False, shell_feedback="use stdlib instead")
        ctx = _ctx(tmp_path, io)
        with (
            patch(
                "neurosurfer.tools.builtin.install_package.resolve_interpreter",
                return_value="/usr/bin/python3",
            ),
            patch(
                "neurosurfer.tools.builtin.install_package._is_importable",
                new=AsyncMock(return_value=False),
            ),
        ):
            r = await tool.call(InstallPythonPackageArgs(packages=["reportlab"]), ctx)
        assert r.is_error
        assert r.content == "use stdlib instead"

    @pytest.mark.asyncio
    async def test_approval_runs_pip_against_resolved_interpreter(self, tmp_path):
        tool = InstallPythonPackageTool()
        ctx = _ctx(tmp_path)  # ScriptedIO defaults approve_shell=True

        fake_proc = AsyncMock()
        fake_proc.communicate = AsyncMock(return_value=(b"", b""))
        fake_proc.returncode = 0

        captured: dict = {}

        async def fake_exec(*cmd, **kwargs):
            captured["cmd"] = cmd
            return fake_proc

        with (
            patch(
                "neurosurfer.tools.builtin.install_package.resolve_interpreter",
                return_value="/usr/bin/python3",
            ),
            patch(
                "neurosurfer.tools.builtin.install_package._is_importable",
                new=AsyncMock(return_value=False),
            ),
            patch("asyncio.create_subprocess_exec", new=fake_exec),
        ):
            r = await tool.call(
                InstallPythonPackageArgs(packages=["reportlab"], reason="pdf export"), ctx
            )

        assert not r.is_error
        assert "Installed reportlab" in r.content
        assert captured["cmd"][:4] == ("/usr/bin/python3", "-m", "pip", "install")
        assert "reportlab" in captured["cmd"]


# ── set_python_env ────────────────────────────────────────────────────────────

class TestSetPythonEnv:
    @pytest.mark.asyncio
    async def test_pins_interpreter_and_durable_state(self, tmp_path):
        tool = SetPythonEnvTool()
        durable = DurableState()
        ctx = _ctx(tmp_path, durable=durable)
        with (
            patch(
                "neurosurfer.tools.builtin.set_python_env.resolve_env_spec",
                return_value="/some/python",
            ),
            patch(
                "neurosurfer.tools.builtin.set_python_env.installed_packages",
                return_value=["foo==1.0"],
            ),
        ):
            r = await tool.call(SetPythonEnvArgs(spec="conda:myenv"), ctx)
        assert not r.is_error
        assert ctx.extra["python_interpreter"] == "/some/python"
        assert durable.python_env == "/some/python"
        assert "## Python Environment" in durable.to_context_block()

    @pytest.mark.asyncio
    async def test_bad_spec_returns_error(self, tmp_path):
        tool = SetPythonEnvTool()
        ctx = _ctx(tmp_path)
        with patch(
            "neurosurfer.tools.builtin.set_python_env.resolve_env_spec",
            side_effect=EnvResolutionError("nope"),
        ):
            r = await tool.call(SetPythonEnvArgs(spec="conda:bad"), ctx)
        assert r.is_error
        assert "nope" in r.content


# ── environment_section ───────────────────────────────────────────────────────

class TestEnvironmentSection:
    def test_renders_without_git(self, tmp_path):
        text = environment_section(_ctx(tmp_path))
        assert "Working directory" in text
        assert "Platform" in text
        assert "Git:" not in text  # tmp_path isn't inside a git repo

    def test_renders_ready_interpreter_with_packages(self, tmp_path):
        with (
            patch(
                "neurosurfer.tools.builtin.python_exec.interpreter.describe_interpreter",
                return_value=("/fake/python", True),
            ),
            patch(
                "neurosurfer.tools.builtin.python_exec.managed_env.installed_packages",
                return_value=["numpy==1.0", "pandas==2.0"],
            ),
        ):
            text = environment_section(_ctx(tmp_path))
        assert "/fake/python" in text
        assert "numpy" in text

    def test_renders_not_yet_provisioned(self, tmp_path):
        with patch(
            "neurosurfer.tools.builtin.python_exec.interpreter.describe_interpreter",
            return_value=("/would/be/python", False),
        ):
            text = environment_section(_ctx(tmp_path))
        assert "not yet provisioned" in text

    def test_internal_failure_propagates(self, tmp_path):
        """environment_section itself doesn't swallow errors — the caller does.

        BaseAgent._environment_block (agents/base.py) wraps this call in try/except
        so a failure here can never break the streaming path; see
        TestBaseAgentEnvironmentBlock below for that safety net.
        """
        with patch(
            "neurosurfer.tools.builtin.python_exec.interpreter.describe_interpreter",
            side_effect=RuntimeError("boom"),
        ):
            with pytest.raises(RuntimeError):
                environment_section(_ctx(tmp_path))


# ── BaseAgent wiring (show_environment / _environment_block safety net) ──────

def _agent(tmp_path: Path, **kwargs) -> AgenticLoop:
    provider = AnthropicProvider(api_key="test", model="claude-opus-4-8")
    provider._client = FakeAnthropicClient([])  # type: ignore[attr-defined]
    return AgenticLoop(
        provider=provider,
        tools=default_pool(),
        system_prompt="Do the task.",
        guardrails=Guardrails(write_scope=["docs/"]),
        io=ScriptedIO(),
        cwd=tmp_path,
        **kwargs,
    )


class TestBaseAgentEnvironmentBlock:
    def test_effective_system_includes_environment_section_by_default(self, tmp_path):
        system = _agent(tmp_path)._effective_system()
        assert "# Environment" in system
        assert str(tmp_path) in system

    def test_show_environment_false_omits_section(self, tmp_path):
        system = _agent(tmp_path, show_environment=False)._effective_system()
        assert "# Environment" not in system

    def test_environment_block_never_raises(self, tmp_path):
        agent = _agent(tmp_path)
        with patch(
            "neurosurfer.prompts.environment.environment_section",
            side_effect=RuntimeError("boom"),
        ):
            system = agent._effective_system()  # must not raise
        assert "(unavailable)" in system
