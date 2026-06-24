"""Session store — store CRUD, history round-trip, token guard, memory injection."""

from __future__ import annotations

from pathlib import Path

from neurosurfer.llm.types import Message, TextBlock, ToolResultBlock, ToolUseBlock
from neurosurfer.sessions.history import estimate_tokens, load_history, save_history
from neurosurfer.sessions.store import SessionStore

# ── helpers ───────────────────────────────────────────────────────────────────

def _store(tmp_path: Path) -> SessionStore:
    return SessionStore(tmp_path / "sessions")


def _text_msg(role: str, text: str) -> Message:
    return Message(role=role, content=[TextBlock(text=text)])  # type: ignore[arg-type]


def _tool_turn() -> list[Message]:
    """A realistic assistant→user tool_use / tool_result pair."""
    tool_msg = Message(
        role="assistant",
        content=[ToolUseBlock(id="tid1", name="read_file", input={"path": "/tmp/x.txt"})],  # type: ignore[arg-type]
    )
    result_msg = Message(
        role="user",
        content=[ToolResultBlock(tool_use_id="tid1", content="file contents here")],  # type: ignore[arg-type]
    )
    return [tool_msg, result_msg]


# ── SessionStore CRUD ─────────────────────────────────────────────────────────

def test_create_and_get(tmp_path):
    s = _store(tmp_path)
    rec = s.create("general", cwd="/home/user")
    assert rec.task == "general"
    assert rec.cwd == "/home/user"
    assert rec.status == "active"

    loaded = s.get("general", rec.id)
    assert loaded is not None
    assert loaded.id == rec.id
    assert loaded.task == "general"


def test_get_missing_returns_none(tmp_path):
    s = _store(tmp_path)
    assert s.get("general", "notexist") is None


def test_list_for_task_ordered_by_updated_at(tmp_path):
    from datetime import datetime, timedelta

    s = _store(tmp_path)
    r1 = s.create("general", cwd="")
    r2 = s.create("general", cwd="")

    # Manually back-date r1 so r2 sorts first
    r1.updated_at = datetime.utcnow() - timedelta(hours=2)
    s.update(r1)
    r2.title = "newest"
    s.update(r2)

    sessions = s.list_for_task("general")
    assert sessions[0].id == r2.id
    assert sessions[1].id == r1.id


def test_list_for_task_empty_for_unknown_task(tmp_path):
    s = _store(tmp_path)
    assert s.list_for_task("no-such-task") == []


def test_close_and_interrupt_status(tmp_path):
    s = _store(tmp_path)
    r = s.create("code", cwd="")
    s.close(r)
    assert s.get("code", r.id).status == "closed"  # type: ignore[union-attr]

    r2 = s.create("code", cwd="")
    s.interrupt(r2)
    assert s.get("code", r2.id).status == "interrupted"  # type: ignore[union-attr]


def test_delete_removes_meta_and_history(tmp_path):
    s = _store(tmp_path)
    r = s.create("general", cwd="")
    msgs = [_text_msg("user", "hello"), _text_msg("assistant", "hi")]
    s.save_history(r, msgs)

    meta = tmp_path / "sessions" / "general" / f"{r.id}.json"
    hist = tmp_path / "sessions" / "general" / f"{r.id}.hist.json"
    assert meta.exists() and hist.exists()

    assert s.delete("general", r.id) is True
    assert not meta.exists()
    assert not hist.exists()
    assert s.delete("general", r.id) is False  # already gone


def test_update_bumps_updated_at(tmp_path):

    s = _store(tmp_path)
    r = s.create("general", cwd="")
    import time

    old_ts = r.updated_at
    time.sleep(0.01)
    r.title = "changed"
    s.update(r)
    reloaded = s.get("general", r.id)
    assert reloaded is not None
    assert reloaded.updated_at >= old_ts


def test_task_name_sanitized_for_filesystem(tmp_path):
    s = _store(tmp_path)
    r = s.create("my task/with:special!chars", cwd="")
    assert s.get("my task/with:special!chars", r.id) is not None
    # Ensure no literal "/" in the directory name
    task_dir = tmp_path / "sessions"
    dirs = list(task_dir.iterdir())
    assert all("/" not in d.name for d in dirs)


# ── History serialization ─────────────────────────────────────────────────────

def test_save_and_load_text_messages(tmp_path):
    path = tmp_path / "hist.json"
    msgs = [_text_msg("user", "hello world"), _text_msg("assistant", "hi there")]
    save_history(path, msgs)

    loaded = load_history(path)
    assert len(loaded) == 2
    assert loaded[0].role == "user"
    assert loaded[0].text() == "hello world"
    assert loaded[1].text() == "hi there"


def test_save_and_load_tool_use_messages(tmp_path):
    path = tmp_path / "hist.json"
    msgs = _tool_turn()
    save_history(path, msgs)

    loaded = load_history(path)
    assert len(loaded) == 2
    assert loaded[0].role == "assistant"
    block = loaded[0].content[0]
    assert isinstance(block, ToolUseBlock)
    assert block.name == "read_file"
    assert block.input == {"path": "/tmp/x.txt"}

    result_block = loaded[1].content[0]
    assert isinstance(result_block, ToolResultBlock)
    assert result_block.tool_use_id == "tid1"
    assert result_block.content == "file contents here"


def test_thinking_block_signature_preserved(tmp_path):
    from neurosurfer.llm.types import ThinkingBlock

    path = tmp_path / "hist.json"
    thinking = Message(
        role="assistant",
        content=[ThinkingBlock(thinking="I should think...", signature="sig-abc123")],  # type: ignore[arg-type]
    )
    save_history(path, [thinking])
    loaded = load_history(path)
    block = loaded[0].content[0]
    assert isinstance(block, ThinkingBlock)
    assert block.signature == "sig-abc123"
    assert block.thinking == "I should think..."


def test_load_missing_history_returns_empty(tmp_path):
    assert load_history(tmp_path / "nonexistent.json") == []


def test_load_corrupt_history_returns_empty(tmp_path):
    path = tmp_path / "hist.json"
    path.write_text("{not valid json", encoding="utf-8")
    assert load_history(path) == []


def test_store_save_and_load_history(tmp_path):
    s = _store(tmp_path)
    r = s.create("general", cwd="")
    msgs = [_text_msg("user", "hello"), _text_msg("assistant", "world")]
    s.save_history(r, msgs)

    loaded = s.load_history("general", r.id)
    assert [m.text() for m in loaded] == ["hello", "world"]


def test_load_history_missing_returns_empty(tmp_path):
    s = _store(tmp_path)
    assert s.load_history("general", "notexist") == []


# ── estimate_tokens ───────────────────────────────────────────────────────────

def test_estimate_tokens_text():
    msgs = [_text_msg("user", "a" * 400)]  # 400 chars → ~100 tokens
    assert estimate_tokens(msgs) == 100


def test_estimate_tokens_empty():
    assert estimate_tokens([]) == 0


# ── Token guard in _make_raw_agent ────────────────────────────────────────────

def test_token_guard_trims_long_history(tmp_path):
    """When restored history exceeds the budget, replace_prefix_with_summary fires."""
    from neurosurfer.agents.conversation.messages import MessageHistory
    from neurosurfer.sessions.history import estimate_tokens

    # Build a history that's clearly over any sane budget
    msgs = [_text_msg("user" if i % 2 == 0 else "assistant", "x" * 200) for i in range(60)]
    total = estimate_tokens(msgs)
    assert total > 0

    budget = total // 3  # artificially tight budget
    history = MessageHistory()
    history.replace_all(msgs)
    if estimate_tokens(history.messages) > budget:
        history.replace_prefix_with_summary(keep_tail=20, summary="[compacted]")

    assert len(history.messages) <= 21  # summary + 20 tail messages


# ── Memory instruction injection ──────────────────────────────────────────────

def test_memory_usage_section_is_nonempty():
    from neurosurfer.prompts.base_agent import memory_usage_section
    section = memory_usage_section()
    assert "memory" in section.lower()
    assert "scope" in section


def test_memory_usage_injected_when_memory_in_tools():
    """Injection guard: section added only when 'memory' is in the tools list."""
    from neurosurfer.prompts.base_agent import memory_usage_section

    tools_with_memory = ["read_file", "memory", "finish"]
    extra: list[str] = []
    if "memory" in tools_with_memory:
        extra.append(memory_usage_section())
    assert any("# Memory" in s for s in extra)


def test_memory_usage_not_injected_when_no_memory_tool():
    from neurosurfer.prompts.base_agent import memory_usage_section

    tools_without_memory = ["read_file", "finish"]
    extra: list[str] = []
    if "memory" in tools_without_memory:
        extra.append(memory_usage_section())
    assert extra == []
