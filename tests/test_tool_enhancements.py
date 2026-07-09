"""Tests for the Workstream E tool polish: ripgrep-backed search (with pure-Python
fallback), multi-hunk apply_edit, run_command cwd override + background jobs, and
.gitignore-aware list_dir.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from neurosurfer.tools.base import ToolContext
from neurosurfer.tools.builtin.apply_edit import ApplyEditTool
from neurosurfer.tools.builtin.list_dir import ListDirTool
from neurosurfer.tools.builtin.run_command import RunCommandTool
from neurosurfer.tools.builtin.search import SearchTool

from .fakes import ScriptedIO


def ctx_for(tmp_path: Path, io: ScriptedIO | None = None) -> ToolContext:
    return ToolContext(cwd=tmp_path, io=io or ScriptedIO())


# ── search: ripgrep + fallback ────────────────────────────────────────────────

class TestSearchRipgrep:
    @pytest.mark.asyncio
    async def test_uses_ripgrep_when_available(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "neurosurfer.tools.builtin.search.shutil.which", lambda name: "/usr/bin/rg"
        )
        (tmp_path / "a.py").write_text("class Foo:\n    pass\n")

        fake_proc = AsyncMock()
        fake_proc.communicate = AsyncMock(return_value=(b"a.py:1:class Foo:\n", b""))
        fake_proc.returncode = 0
        captured: dict = {}

        async def fake_exec(*cmd, **kwargs):
            captured["cmd"] = cmd
            return fake_proc

        with patch("asyncio.create_subprocess_exec", new=fake_exec):
            r = await SearchTool().run({"pattern": "class Foo"}, ctx_for(tmp_path))

        assert not r.is_error
        assert "a.py:1:class Foo:" in r.content
        assert captured["cmd"][0] == "rg"

    @pytest.mark.asyncio
    async def test_falls_back_when_rg_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "neurosurfer.tools.builtin.search.shutil.which", lambda name: None
        )
        (tmp_path / "a.py").write_text("class Foo:\n    pass\n")
        r = await SearchTool().run({"pattern": "class Foo"}, ctx_for(tmp_path))
        assert not r.is_error
        assert "a.py:1: class Foo:" in r.content

    @pytest.mark.asyncio
    async def test_falls_back_when_rg_errors(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "neurosurfer.tools.builtin.search.shutil.which", lambda name: "/usr/bin/rg"
        )
        (tmp_path / "a.py").write_text("class Foo:\n    pass\n")

        fake_proc = AsyncMock()
        fake_proc.communicate = AsyncMock(return_value=(b"", b"bad pattern"))
        fake_proc.returncode = 2  # rg's "error" exit code, not "no matches" (1)

        async def fake_exec(*cmd, **kwargs):
            return fake_proc

        with patch("asyncio.create_subprocess_exec", new=fake_exec):
            r = await SearchTool().run({"pattern": "class Foo"}, ctx_for(tmp_path))

        assert not r.is_error
        assert "a.py" in r.content  # pure-Python fallback still finds the match

    @pytest.mark.asyncio
    async def test_rg_no_matches_is_not_an_error(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "neurosurfer.tools.builtin.search.shutil.which", lambda name: "/usr/bin/rg"
        )
        fake_proc = AsyncMock()
        fake_proc.communicate = AsyncMock(return_value=(b"", b""))
        fake_proc.returncode = 1  # rg's "no matches" exit code

        async def fake_exec(*cmd, **kwargs):
            return fake_proc

        with patch("asyncio.create_subprocess_exec", new=fake_exec):
            r = await SearchTool().run({"pattern": "nope"}, ctx_for(tmp_path))

        assert not r.is_error
        assert "No matches" in r.content


# ── apply_edit: multi-hunk ────────────────────────────────────────────────────

class TestApplyEditMultiHunk:
    @pytest.mark.asyncio
    async def test_multiple_hunks_applied_in_order(self, tmp_path):
        f = tmp_path / "f.txt"
        f.write_text("one two three\n")
        r = await ApplyEditTool().run(
            {
                "path": "f.txt",
                "edits": [
                    {"old_string": "one", "new_string": "ONE"},
                    {"old_string": "three", "new_string": "THREE"},
                ],
            },
            ctx_for(tmp_path),
        )
        assert not r.is_error
        assert f.read_text() == "ONE two THREE\n"

    @pytest.mark.asyncio
    async def test_atomic_failure_writes_nothing(self, tmp_path):
        f = tmp_path / "g.txt"
        f.write_text("alpha beta\n")
        r = await ApplyEditTool().run(
            {
                "path": "g.txt",
                "edits": [
                    {"old_string": "alpha", "new_string": "ALPHA"},
                    {"old_string": "nomatch", "new_string": "X"},
                ],
            },
            ctx_for(tmp_path),
        )
        assert r.is_error
        assert f.read_text() == "alpha beta\n"  # nothing written — all or nothing

    @pytest.mark.asyncio
    async def test_mixing_single_and_multi_mode_rejected(self, tmp_path):
        f = tmp_path / "h.txt"
        f.write_text("x\n")
        r = await ApplyEditTool().run(
            {
                "path": "h.txt",
                "old_string": "x",
                "new_string": "y",
                "edits": [{"old_string": "a", "new_string": "b"}],
            },
            ctx_for(tmp_path),
        )
        assert r.is_error


# ── run_command: cwd override + background jobs ──────────────────────────────

class TestRunCommandCwdAndBackground:
    @pytest.mark.asyncio
    async def test_cwd_override(self, tmp_path):
        sub = tmp_path / "sub"
        sub.mkdir()
        r = await RunCommandTool().run({"command": "pwd", "cwd": "sub"}, ctx_for(tmp_path))
        assert not r.is_error
        assert str(sub) in r.content

    @pytest.mark.asyncio
    async def test_background_job_reaches_completion(self, tmp_path):
        ctx = ctx_for(tmp_path)
        tool = RunCommandTool()
        start = await tool.run(
            {"command": "echo hi && sleep 0.1", "run_in_background": True}, ctx
        )
        assert not start.is_error
        job_id = start.content.split("job ")[1].split(":")[0]

        for _ in range(20):
            status = await tool.run({"action": "status", "job_id": job_id}, ctx)
            if "exited" in status.content:
                break
            await asyncio.sleep(0.05)
        assert "exited" in status.content
        assert "hi" in status.content

    @pytest.mark.asyncio
    async def test_kill_background_job(self, tmp_path):
        ctx = ctx_for(tmp_path)
        tool = RunCommandTool()
        start = await tool.run({"command": "sleep 30", "run_in_background": True}, ctx)
        job_id = start.content.split("job ")[1].split(":")[0]
        killed = await tool.run({"action": "kill", "job_id": job_id}, ctx)
        assert not killed.is_error
        assert "Killed" in killed.content

    @pytest.mark.asyncio
    async def test_status_unknown_job_errors(self, tmp_path):
        r = await RunCommandTool().run(
            {"action": "status", "job_id": "does-not-exist"}, ctx_for(tmp_path)
        )
        assert r.is_error


# ── list_dir: .gitignore awareness ────────────────────────────────────────────

class TestListDirGitignore:
    @pytest.mark.asyncio
    async def test_gitignore_filters_entries(self, tmp_path):
        (tmp_path / ".gitignore").write_text("**/build/\n*.log\n")
        (tmp_path / "build").mkdir()
        (tmp_path / "keep.txt").write_text("x")
        (tmp_path / "debug.log").write_text("x")
        r = await ListDirTool().run({"path": "."}, ctx_for(tmp_path))
        assert "build" not in r.content
        assert "debug.log" not in r.content
        assert "keep.txt" in r.content
        assert ".gitignore" in r.content

    @pytest.mark.asyncio
    async def test_no_gitignore_is_a_no_op(self, tmp_path):
        (tmp_path / "keep.txt").write_text("x")
        r = await ListDirTool().run({"path": "."}, ctx_for(tmp_path))
        assert "keep.txt" in r.content

    @pytest.mark.asyncio
    async def test_negation_not_supported_known_limitation(self, tmp_path):
        """Negation (!pattern) isn't implemented; such lines are simply skipped, so
        a broad pattern still hides the file the negation would have un-hidden."""
        (tmp_path / ".gitignore").write_text("*.log\n!keep.log\n")
        (tmp_path / "keep.log").write_text("x")
        r = await ListDirTool().run({"path": "."}, ctx_for(tmp_path))
        assert "keep.log" not in r.content
