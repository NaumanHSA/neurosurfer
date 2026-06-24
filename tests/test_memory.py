"""Pillar 1 — long-term memory: store, retrieval, distill, the memory tool, lineup.

Local-first and offline: BM25 retrieval (no embeddings), JSONL store in a tmp dir.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pytest

from neurosurfer.agents.context.durable_state import DurableState
from neurosurfer.memory.distill import distill_run
from neurosurfer.memory.embeddings import NullEmbedder, get_embedder
from neurosurfer.memory.models import MemoryEntry
from neurosurfer.memory.retrieval import retrieve
from neurosurfer.memory.store import MemoryStore
from neurosurfer.tools.base import ToolContext
from neurosurfer.tools.builtin.memory_tool import MemoryTool

from .fakes import ScriptedIO


def _store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(tmp_path / "memory")


# ── store: scope isolation ────────────────────────────────────────────────────
def test_global_and_agent_scopes_are_separate_files(tmp_path):
    s = _store(tmp_path)
    s.add(MemoryEntry(scope="global", text="user prefers conda env LLMs"))
    s.add(MemoryEntry(scope="agent", scope_key="code", text="project gates on ruff and mypy"))
    s.add(MemoryEntry(scope="agent", scope_key="bedtime", text="the kid loves dragons"))

    assert (tmp_path / "memory" / "global.jsonl").exists()
    assert (tmp_path / "memory" / "agents" / "code.jsonl").exists()

    code_scope = {e.text for e in s.all_in_scope("code")}
    assert "user prefers conda env LLMs" in code_scope         # global is always included
    assert "project gates on ruff and mypy" in code_scope      # this agent's own
    assert "the kid loves dragons" not in code_scope           # another agent's stays out


def test_all_in_scope_without_agent_is_global_only(tmp_path):
    s = _store(tmp_path)
    s.add(MemoryEntry(scope="global", text="global fact"))
    s.add(MemoryEntry(scope="agent", scope_key="code", text="agent fact"))
    assert {e.text for e in s.all_in_scope(None)} == {"global fact"}


# ── store: dedup / forget / record_use ────────────────────────────────────────
def test_add_dedups_near_identical_text(tmp_path):
    s = _store(tmp_path)
    s.add(MemoryEntry(scope="global", text="The user prefers the conda env named LLMs"))
    s.add(MemoryEntry(scope="global", text="the user prefers the conda env named LLMs!!"))
    assert len(s.list_scope("global")) == 1  # superseded, not duplicated


def test_forget_and_record_use(tmp_path):
    s = _store(tmp_path)
    e = s.add(MemoryEntry(scope="global", text="a durable fact about the project"))
    s.record_use([e.id])
    assert s.list_scope("global")[0].uses == 1
    assert s.forget(e.id) is True
    assert s.list_scope("global") == []
    assert s.forget(e.id) is False  # already gone


def test_corrupt_line_is_skipped_not_fatal(tmp_path):
    s = _store(tmp_path)
    s.add(MemoryEntry(scope="global", text="good entry"))
    path = tmp_path / "memory" / "global.jsonl"
    path.write_text(path.read_text() + "{not valid json\n", encoding="utf-8")
    assert [e.text for e in s.list_scope("global")] == ["good entry"]


# ── retrieval: BM25 ordering + budget + empty ─────────────────────────────────
def test_retrieve_ranks_relevant_first_and_renders_block():
    entries = [
        MemoryEntry(scope="global", kind="fact", text="the deployment uses docker compose"),
        MemoryEntry(scope="global", kind="preference", text="the user likes terse answers"),
        MemoryEntry(scope="global", kind="fact", text="linting is done with ruff"),
    ]
    res = retrieve(entries, "how do we run linting", budget_tokens=500)
    assert res.block.startswith("# Relevant memory")
    assert "ruff" in res.block
    # the lint fact should be the most relevant id surfaced
    assert entries[2].id in res.entry_ids


def test_retrieve_empty_entries_is_empty_block():
    assert retrieve([], "anything").block == ""


def test_retrieve_budget_caps_output():
    entries = [
        MemoryEntry(scope="global", text=f"fact number {i} about the system architecture")
        for i in range(50)
    ]
    res = retrieve(entries, "system architecture", budget_tokens=40)
    assert res.block  # at least the best one
    assert len(res.entry_ids) < len(entries)  # budget pruned the rest


def test_retrieve_salience_breaks_ties_toward_fresh_important():
    now = datetime.utcnow()
    old = MemoryEntry(scope="global", text="shared topic alpha", salience=1.0)
    old.created_at = now - timedelta(days=400)
    fresh = MemoryEntry(scope="global", text="shared topic alpha", salience=5.0)
    fresh.created_at = now
    # Same text → lexical tie; salience + decay should rank the fresh/important one first.
    res = retrieve([old, fresh], "shared topic alpha", budget_tokens=20, now=now)
    assert res.entry_ids[0] == fresh.id


# ── embeddings: always degrade to BM25 ────────────────────────────────────────
def test_get_embedder_returns_none_for_bm25_modes():
    assert get_embedder(None) is None
    assert get_embedder("none") is None
    assert get_embedder("bm25") is None
    assert get_embedder("totally-unknown-backend") is None  # degrades, never raises


def test_null_embedder_is_noop():
    assert NullEmbedder().embed(["x", "y"]) == []


# ── distill: decisions → candidate memories ───────────────────────────────────
def test_distill_routes_to_agent_scope_and_dedups(tmp_path):
    s = _store(tmp_path)
    d = DurableState()
    d.add_decision("Chose BM25 over embeddings to stay local-first")
    d.add_decision("Chose BM25 over embeddings to stay local-first")  # dup
    added = distill_run(s, d, agent="code", session_id="run123")
    assert len(added) == 1
    saved = s.list_scope("agent", "code")
    assert len(saved) == 1
    assert saved[0].source == "distill"
    assert saved[0].session_id == "run123"
    assert saved[0].salience < 1.0  # low-salience candidate


def test_distill_no_decisions_is_noop(tmp_path):
    s = _store(tmp_path)
    assert distill_run(s, DurableState(), agent="code") == []
    assert distill_run(s, None, agent="code") == []


# ── the memory tool ───────────────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_memory_tool_add_global(tmp_path):
    s = _store(tmp_path)
    ctx = ToolContext(cwd=tmp_path, io=ScriptedIO(), memory=s, memory_agent="code")
    res = await MemoryTool().run({"op": "add", "text": "remember this fact", "scope": "global"}, ctx)
    assert not res.is_error
    assert s.list_scope("global")[0].text == "remember this fact"


@pytest.mark.asyncio
async def test_memory_tool_add_agent_scope_uses_active_agent(tmp_path):
    s = _store(tmp_path)
    ctx = ToolContext(cwd=tmp_path, io=ScriptedIO(), memory=s, memory_agent="code")
    await MemoryTool().run({"op": "save", "text": "an agent fact", "scope": "task"}, ctx)
    assert [e.text for e in s.list_scope("agent", "code")] == ["an agent fact"]


@pytest.mark.asyncio
async def test_memory_tool_agent_scope_falls_back_to_global_without_agent(tmp_path):
    s = _store(tmp_path)
    ctx = ToolContext(cwd=tmp_path, io=ScriptedIO(), memory=s, memory_agent=None)
    await MemoryTool().run({"op": "add", "text": "no agent here", "scope": "agent"}, ctx)
    assert [e.text for e in s.list_scope("global")] == ["no agent here"]


@pytest.mark.asyncio
async def test_memory_tool_forget(tmp_path):
    s = _store(tmp_path)
    e = s.add(MemoryEntry(scope="global", text="to be forgotten"))
    ctx = ToolContext(cwd=tmp_path, io=ScriptedIO(), memory=s)
    res = await MemoryTool().run({"op": "forget", "id": e.id}, ctx)
    assert not res.is_error and s.list_scope("global") == []


@pytest.mark.asyncio
async def test_memory_tool_noops_without_store(tmp_path):
    ctx = ToolContext(cwd=tmp_path, io=ScriptedIO())  # memory is None
    res = await MemoryTool().run({"op": "add", "text": "x"}, ctx)
    assert not res.is_error
    assert "not enabled" in res.content


@pytest.mark.asyncio
async def test_memory_tool_accepts_content_alias(tmp_path):
    s = _store(tmp_path)
    ctx = ToolContext(cwd=tmp_path, io=ScriptedIO(), memory=s)
    await MemoryTool().run({"op": "add", "content": "aliased text"}, ctx)
    assert [e.text for e in s.list_scope("global")] == ["aliased text"]
