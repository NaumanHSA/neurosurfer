from __future__ import annotations

from pathlib import Path

import pytest

from neurosurfer.app.tools.present_plan import PresentPlanTool
from neurosurfer.llm.providers.anthropic import to_anthropic_tools
from neurosurfer.llm.providers.openai import to_openai_tools
from neurosurfer.tools import all_tools, build_pool, default_pool
from neurosurfer.tools.base import ToolContext
from neurosurfer.tools.builtin.apply_edit import ApplyEditTool
from neurosurfer.tools.builtin.ask_user import AskUserTool
from neurosurfer.tools.builtin.finish import FinishTool
from neurosurfer.tools.builtin.list_dir import ListDirTool
from neurosurfer.tools.builtin.read_file import ReadFileTool
from neurosurfer.tools.builtin.run_command import RunCommandTool
from neurosurfer.tools.builtin.search import SearchTool
from neurosurfer.tools.builtin.todo import TodoTool
from neurosurfer.tools.builtin.write_file import WriteFileTool

from .fakes import ScriptedIO


def ctx_for(tmp_path: Path, io: ScriptedIO | None = None) -> ToolContext:
    return ToolContext(cwd=tmp_path, io=io or ScriptedIO())


@pytest.mark.asyncio
async def test_read_file_with_line_numbers_and_state(tmp_path):
    f = tmp_path / "a.py"
    f.write_text("line1\nline2\nline3\n")
    ctx = ctx_for(tmp_path)
    res = await ReadFileTool().run({"path": "a.py"}, ctx)
    assert not res.is_error
    assert "1\tline1" in res.content
    assert str(f) in ctx.file_state  # recorded for staleness


@pytest.mark.asyncio
async def test_read_file_missing(tmp_path):
    res = await ReadFileTool().run({"path": "nope.txt"}, ctx_for(tmp_path))
    assert res.is_error and "not found" in res.content.lower()


@pytest.mark.asyncio
async def test_list_dir_and_glob(tmp_path):
    (tmp_path / "x.py").write_text("a")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "y.py").write_text("b")
    ctx = ctx_for(tmp_path)
    res = await ListDirTool().run({"path": "."}, ctx)
    assert "x.py" in res.content and "sub/" in res.content
    res2 = await ListDirTool().run({"path": ".", "pattern": "**/*.py"}, ctx)
    assert "sub/y.py" in res2.content


@pytest.mark.asyncio
async def test_search(tmp_path):
    (tmp_path / "a.py").write_text("def foo():\n    return 1\n")
    (tmp_path / "b.py").write_text("x = 2\n")
    res = await SearchTool().run({"pattern": r"def \w+", "path": "."}, ctx_for(tmp_path))
    assert "a.py:1" in res.content
    assert "b.py" not in res.content


@pytest.mark.asyncio
async def test_run_command_success_and_failure(tmp_path):
    ok = await RunCommandTool().run({"command": "echo hello"}, ctx_for(tmp_path))
    assert not ok.is_error and "hello" in ok.content
    bad = await RunCommandTool().run({"command": "exit 3"}, ctx_for(tmp_path))
    assert bad.is_error and "exit 3" in bad.content


@pytest.mark.asyncio
async def test_write_then_edit(tmp_path):
    ctx = ctx_for(tmp_path)
    w = await WriteFileTool().run({"path": "docs/r.md", "content": "hello world\n"}, ctx)
    assert not w.is_error
    assert (tmp_path / "docs" / "r.md").read_text() == "hello world\n"

    e = await ApplyEditTool().run(
        {"path": "docs/r.md", "old_string": "world", "new_string": "there"}, ctx
    )
    assert not e.is_error
    assert (tmp_path / "docs" / "r.md").read_text() == "hello there\n"


@pytest.mark.asyncio
async def test_apply_edit_errors(tmp_path):
    ctx = ctx_for(tmp_path)
    (tmp_path / "f.txt").write_text("aaa")
    missing = await ApplyEditTool().run(
        {"path": "f.txt", "old_string": "zzz", "new_string": "y"}, ctx
    )
    assert missing.is_error and "not found" in missing.content
    (tmp_path / "g.txt").write_text("a a a")
    ambig = await ApplyEditTool().run(
        {"path": "g.txt", "old_string": "a", "new_string": "b"}, ctx
    )
    assert ambig.is_error and "ambiguous" in ambig.content


@pytest.mark.asyncio
async def test_apply_edit_staleness(tmp_path):
    ctx = ctx_for(tmp_path)
    f = tmp_path / "s.txt"
    f.write_text("original\n")
    await ReadFileTool().run({"path": "s.txt"}, ctx)  # records state
    import os
    import time

    time.sleep(0.01)
    f.write_text("changed externally\n")
    os.utime(f, (time.time() + 5, time.time() + 5))  # bump mtime
    res = await ApplyEditTool().run(
        {"path": "s.txt", "old_string": "changed", "new_string": "x"}, ctx
    )
    assert res.is_error and "changed on disk" in res.content


@pytest.mark.asyncio
async def test_ask_user(tmp_path):
    io = ScriptedIO(answers=["developers"])
    res = await AskUserTool().run({"question": "audience?"}, ctx_for(tmp_path, io))
    assert "developers" in res.content
    assert io.asked == ["audience?"]


@pytest.mark.asyncio
async def test_present_plan_approval(tmp_path):
    io = ScriptedIO(approve_plan=True)
    res = await PresentPlanTool().run({"plan": "do X"}, ctx_for(tmp_path, io))
    assert res.control.get("plan_approved") is True

    io2 = ScriptedIO(approve_plan=False)
    res2 = await PresentPlanTool().run({"plan": "do Y"}, ctx_for(tmp_path, io2))
    assert res2.control.get("plan_approved") is False


@pytest.mark.asyncio
async def test_todo_and_finish(tmp_path):
    ctx = ctx_for(tmp_path)
    res = await TodoTool().run(
        {"todos": [{"content": "a", "status": "completed"}, {"content": "b"}]}, ctx
    )
    assert "1/2 done" in res.content
    fin = await FinishTool().run({"summary": "all done"}, ctx)
    assert fin.control.get("finished") is True
    assert fin.content == "all done"


def test_pool_selection():
    pool = default_pool()
    assert set(pool.names()) >= {"read_file", "write_file", "finish"}
    narrowed = build_pool(["read_file", "list_dir", "search"])
    assert narrowed.names() == ["read_file", "list_dir", "search"]


def test_schemas_render_for_both_providers():
    pool = default_pool()
    schemas = pool.schemas()
    a = to_anthropic_tools(schemas)
    o = to_openai_tools(schemas)
    assert len(a) == len(o) == len(schemas)
    # Anthropic uses input_schema; OpenAI nests under function.parameters.
    assert all("input_schema" in t for t in a)
    assert all(t["function"]["name"] for t in o)
    # read_file schema exposes its 'path' property.
    read = next(t for t in a if t["name"] == "read_file")
    assert "path" in read["input_schema"]["properties"]


def test_all_tools_have_unique_names_and_descriptions():
    tools = all_tools()
    names = [t.name for t in tools]
    assert len(names) == len(set(names))
    assert all(t.name and t.description for t in tools)
