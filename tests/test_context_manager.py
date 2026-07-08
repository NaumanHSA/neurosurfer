"""Phase 4 tests: context management (compaction + durable state).

Covers:
  - format_compact_summary strips <analysis>, unwraps <summary>
  - DurableState mutators, to_context_block()
  - ContextManager.system_with_durable injects the durable block
  - Proactive compaction: tokens > threshold → Compacted event emitted,
    history replaced with summary, recent tail preserved
  - No compaction when tokens <= threshold
  - Reactive compaction: context-overflow error during stream → compact +
    retry, stream continues after compaction
  - Non-overflow errors are re-raised immediately (no compaction)
  - Small-window simulation: tiny configured window triggers compaction on a
    short history (local-model scenario)
  - Durable state survives compaction (re-injected via system_with_durable)
  - todo / present_plan tools write into DurableState
"""

from __future__ import annotations

from pathlib import Path

import pytest

from neurosurfer.agents.context.durable_state import DurableState
from neurosurfer.agents.context.manager import ContextManager
from neurosurfer.agents.context.summary_prompt import (
    format_compact_summary,
    get_compact_prompt,
    get_compact_user_summary_message,
)
from neurosurfer.agents.conversation.messages import MessageHistory
from neurosurfer.llm.capabilities import ProviderCapabilities
from neurosurfer.llm.tokens import auto_compact_threshold, effective_window
from neurosurfer.llm.types import (
    CanonicalResponse,
    Done,
    GenerationConfig,
    TextBlock,
    Usage,
)
from tests.fakes import ScriptedIO, ScriptedProvider

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _make_provider(
    summary_text: str = "<analysis>a</analysis><summary>Summary content.</summary>",
    context_window: int = 200_000,
    max_output_tokens: int = 32_000,
    reported_tokens: int | None = None,
) -> ScriptedProvider:
    """ScriptedProvider whose complete() returns summary_text.

    ``reported_tokens`` pins the count_tokens response to a fixed value, making
    compaction threshold tests deterministic regardless of the local tokenizer.
    When None, estimation is used (supports_token_count=False).
    """
    p = ScriptedProvider(turns=[("", [])], context_window=context_window)
    p.capabilities = ProviderCapabilities(
        supports_thinking=False,
        supports_prompt_cache=False,
        supports_token_count=reported_tokens is not None,
        tool_call_style="anthropic",
        context_window=context_window,
        max_output_tokens=max_output_tokens,
    )

    async def _fake_complete(messages, system, tools, config):
        return CanonicalResponse(
            content=[TextBlock(text=summary_text)],
            stop_reason="end_turn",
            usage=Usage(input_tokens=100, output_tokens=50),
        )
    p.complete = _fake_complete  # type: ignore[method-assign]

    if reported_tokens is not None:
        _fixed = reported_tokens

        async def _fake_count_tokens(messages, system, tools):
            return _fixed

        p.count_tokens = _fake_count_tokens  # type: ignore[method-assign]

    return p


def _fat_history(n_pairs: int = 30) -> MessageHistory:
    """Build a history large enough to look over-budget to the estimator."""
    h = MessageHistory()
    for _ in range(n_pairs):
        h.add_user_text("x" * 400)            # ~100 tokens each
        h.add_assistant_response(
            CanonicalResponse(
                content=[TextBlock(text="y" * 400)],
                stop_reason="end_turn",
                usage=Usage(input_tokens=10, output_tokens=10),
            )
        )
    return h


# ──────────────────────────────────────────────────────────────────────────────
# summary_prompt helpers
# ──────────────────────────────────────────────────────────────────────────────

def test_format_strips_analysis_and_unwraps_summary():
    raw = (
        "<analysis>my scratchpad notes</analysis>\n\n"
        "<summary>\n1. Request: do X\n2. Key: Y\n</summary>"
    )
    result = format_compact_summary(raw)
    assert "<analysis>" not in result
    assert "<summary>" not in result
    assert "Summary:" in result
    assert "do X" in result


def test_format_no_tags_passthrough():
    raw = "Some plain text with no XML tags."
    result = format_compact_summary(raw)
    assert "Some plain text" in result


def test_get_compact_user_summary_message_wraps():
    raw = "<summary>Content here.</summary>"
    msg = get_compact_user_summary_message(raw)
    assert "continued from a previous conversation" in msg
    assert "Content here." in msg
    assert "do not acknowledge the summary" in msg


def test_get_compact_prompt_task_must_preserve():
    prompt = get_compact_prompt(task_must_preserve=["approved plan", "todo list"])
    assert "approved plan" in prompt
    assert "todo list" in prompt
    assert "MUST be preserved" in prompt


def test_get_compact_prompt_no_tools():
    prompt = get_compact_prompt()
    assert "Do NOT call any tools" in prompt
    assert "REMINDER" in prompt


# ──────────────────────────────────────────────────────────────────────────────
# DurableState
# ──────────────────────────────────────────────────────────────────────────────

def test_durable_state_empty():
    d = DurableState()
    assert d.is_empty()
    assert d.to_context_block() == ""


def test_durable_state_set_plan():
    d = DurableState()
    d.set_plan("Sprint plan", "Step 1: do A\nStep 2: do B")
    assert not d.is_empty()
    block = d.to_context_block()
    assert "Sprint plan" in block
    assert "Step 1" in block
    assert "<durable_state>" in block
    assert "</durable_state>" in block


def test_durable_state_todos():
    d = DurableState()
    d.set_todos([
        {"content": "write tests", "status": "completed"},
        {"content": "ship it", "status": "pending"},
    ])
    block = d.to_context_block()
    assert "[x] write tests" in block
    assert "[ ] ship it" in block


def test_durable_state_decisions():
    d = DurableState()
    d.add_decision("use async generators for streaming")
    d.add_decision("keep durable state outside compactable history")
    block = d.to_context_block()
    assert "async generators" in block
    assert "durable state" in block


# ──────────────────────────────────────────────────────────────────────────────
# Token math
# ──────────────────────────────────────────────────────────────────────────────

def test_effective_window_reservation():
    # window=200k, max_output=32k → effective = 200k − 20k = 180k
    assert effective_window(200_000, 32_000) == 180_000


def test_auto_compact_threshold():
    # threshold = effective_window − 13_000
    assert auto_compact_threshold(200_000, 32_000) == 167_000


def test_effective_window_floor():
    # Tiny window (issue #635): effective must not go below reserved+buffer.
    eff = effective_window(30_000, 32_000)
    assert eff > 0
    # floor: reserved=20k, buffer=13k → floor=33k; effective=30k-20k=10k < floor
    assert eff == 20_000 + 13_000


# ──────────────────────────────────────────────────────────────────────────────
# ContextManager — system_with_durable
# ──────────────────────────────────────────────────────────────────────────────

def test_system_with_durable_no_state():
    provider = _make_provider()
    cm = ContextManager(provider)
    assert cm.system_with_durable("base") == "base"


def test_system_with_durable_injects_block():
    d = DurableState()
    d.set_plan("Plan", "step A")
    provider = _make_provider()
    cm = ContextManager(provider, durable=d)
    result = cm.system_with_durable("base system")
    assert "base system" in result
    assert "step A" in result
    assert "<durable_state>" in result


# ──────────────────────────────────────────────────────────────────────────────
# ContextManager — proactive compaction
# ──────────────────────────────────────────────────────────────────────────────

async def test_no_compaction_below_threshold():
    """Small history: should produce zero Compacted events."""
    provider = _make_provider(context_window=200_000)
    cm = ContextManager(provider)
    history = MessageHistory()
    history.add_user_text("hello")

    collected = []
    async for ev in cm.maybe_compact(provider, history, "sys", []):
        collected.append(ev)

    assert collected == [], "No compaction expected on a short history"
    assert len(history.messages) == 1  # untouched


async def test_proactive_compact_over_threshold():
    """Fat history over threshold: Compacted event emitted, prefix replaced.

    Uses reported_tokens to pin the token count well above the threshold,
    making the test independent of the local tokenizer implementation.
    With context_window=200_000, max_output=32_000:
      threshold = effective_window(200k, 32k) - 13k = 180k - 13k = 167k.
    We report 180_000 tokens so compaction fires reliably.
    """
    provider = _make_provider(
        summary_text="<summary>Summary of everything.</summary>",
        context_window=200_000,
        max_output_tokens=32_000,
        reported_tokens=180_000,  # above threshold of 167_000
    )
    cm = ContextManager(provider)
    history = _fat_history(n_pairs=5)

    msg_count_before = len(history.messages)
    collected = []
    async for ev in cm.maybe_compact(provider, history, "sys", []):
        collected.append(ev)

    from neurosurfer.agents.conversation.events import Compacted
    assert any(isinstance(ev, Compacted) for ev in collected), (
        "Expected a Compacted event"
    )
    # History should be shorter than before (prefix replaced with summary).
    assert len(history.messages) < msg_count_before


async def test_proactive_compact_preserves_recent_tail():
    """The last _KEEP_RECENT_MESSAGES messages must survive compaction verbatim."""
    provider = _make_provider(
        summary_text="<summary>Compacted summary.</summary>",
        context_window=200_000,
        max_output_tokens=32_000,
        reported_tokens=180_000,
    )
    cm = ContextManager(provider)
    history = _fat_history(n_pairs=5)

    # Tag the last message so we can verify it's still there.
    sentinel = "SENTINEL_MESSAGE_MUST_SURVIVE"
    history.add_user_text(sentinel)

    async for _ in cm.maybe_compact(provider, history, "sys", []):
        pass

    # Tail messages must still be present.
    all_texts = [
        b.text
        for m in history.messages
        for b in m.content
        if isinstance(b, TextBlock)
    ]
    assert any(sentinel in t for t in all_texts), (
        "Recent tail message must survive compaction"
    )


async def test_proactive_compact_summary_injected():
    """The summary message must appear at the head of history after compaction."""
    provider = _make_provider(
        summary_text="<summary>THE SUMMARY CONTENT.</summary>",
        context_window=200_000,
        max_output_tokens=32_000,
        reported_tokens=180_000,
    )
    cm = ContextManager(provider)
    history = _fat_history(n_pairs=5)

    async for _ in cm.maybe_compact(provider, history, "sys", []):
        pass

    # First message should carry the formatted summary.
    first_msg = history.messages[0]
    first_text = " ".join(
        b.text for b in first_msg.content if isinstance(b, TextBlock)
    )
    assert "THE SUMMARY CONTENT." in first_text or "continued from a previous" in first_text


# ──────────────────────────────────────────────────────────────────────────────
# ContextManager — reactive compaction (stream_with_recovery)
# ──────────────────────────────────────────────────────────────────────────────

class _OverflowThenOkProvider:
    """Provider that raises a context-overflow error on the first stream() call,
    then succeeds on the second call."""

    def __init__(self):
        self._call = 0
        self.capabilities = ProviderCapabilities(
            supports_thinking=False,
            supports_prompt_cache=False,
            supports_token_count=False,
            tool_call_style="anthropic",
            context_window=200_000,
            max_output_tokens=32_000,
        )
        self.model = "fake"
        self.compact_calls = 0

    async def stream(self, messages, system, tools, config):
        self._call += 1
        if self._call == 1:
            raise ValueError("prompt is too long — context overflow")
        # Second call: yield a normal Done event.
        yield Done(
            response=CanonicalResponse(
                content=[TextBlock(text="recovered text")],
                stop_reason="end_turn",
                usage=Usage(input_tokens=5, output_tokens=5),
            )
        )

    async def complete(self, messages, system, tools, config):
        self.compact_calls += 1
        # Return a canned summary.
        return CanonicalResponse(
            content=[TextBlock(text="<summary>Compact result.</summary>")],
            stop_reason="end_turn",
            usage=Usage(input_tokens=50, output_tokens=20),
        )

    async def count_tokens(self, messages, system, tools):
        from neurosurfer.llm.tokens import estimate_messages_tokens
        return estimate_messages_tokens(messages, system, tools)


async def test_reactive_compact_on_overflow():
    """Context overflow during stream → compact + retry → stream resumes."""
    overflow_provider = _OverflowThenOkProvider()
    cm = ContextManager(overflow_provider)
    history = MessageHistory()
    history.add_user_text("trigger overflow")

    collected = []
    config = GenerationConfig(max_tokens=1024, enable_thinking=False, stream=True)
    async for ev in cm.stream_with_recovery(
        overflow_provider, history, "sys", [], config
    ):
        collected.append(ev)

    assert overflow_provider._call == 2, "Stream must be called twice (fail + retry)"
    assert overflow_provider.compact_calls == 1, "Compact must be called once"
    # Final Done should carry the recovered text.
    done_events = [ev for ev in collected if isinstance(ev, Done)]
    assert done_events, "Must end with a Done event"
    assert done_events[-1].response.text() == "recovered text"


async def test_reactive_compact_non_overflow_reraises():
    """Non-overflow errors must propagate immediately without compaction."""

    class _BrokenProvider:
        capabilities = ProviderCapabilities(
            supports_thinking=False,
            supports_prompt_cache=False,
            supports_token_count=False,
            tool_call_style="anthropic",
            context_window=200_000,
            max_output_tokens=32_000,
        )
        model = "fake"
        compact_calls = 0

        async def stream(self, messages, system, tools, config):
            raise RuntimeError("some unrelated error")
            yield  # make it an async generator

        async def complete(self, messages, system, tools, config):
            self.compact_calls += 1
            return CanonicalResponse(
                content=[TextBlock(text="")],
                stop_reason="end_turn",
                usage=Usage(),
            )

        async def count_tokens(self, messages, system, tools):
            return 100

    bad_provider = _BrokenProvider()
    cm = ContextManager(bad_provider)
    history = MessageHistory()
    history.add_user_text("hello")
    config = GenerationConfig(max_tokens=1024, enable_thinking=False)

    with pytest.raises(RuntimeError, match="some unrelated error"):
        async for _ in cm.stream_with_recovery(bad_provider, history, "sys", [], config):
            pass

    assert bad_provider.compact_calls == 0, "Compaction must NOT be called for non-overflow errors"


# ──────────────────────────────────────────────────────────────────────────────
# Small-window simulation (local-model scenario)
# ──────────────────────────────────────────────────────────────────────────────

async def test_small_window_triggers_compaction():
    """Simulate a local model with a tiny context window (e.g. 4k tokens).

    Reports a token count above threshold to simulate a full context window.
    threshold = effective_window(4_000, 2_048) - 13_000.
    With floor: effective = max(4000-2048, 2048+13000) = 15048; threshold = 2048.
    We report 10_000 tokens — above 2048 — to ensure compaction fires.
    """
    tiny_window = 4_000
    provider = _make_provider(
        summary_text="<summary>Short session summary.</summary>",
        context_window=tiny_window,
        max_output_tokens=2_048,
        reported_tokens=10_000,  # above threshold=2048 for this window/output combo
    )
    cm = ContextManager(provider)

    history = MessageHistory()
    for _ in range(4):
        history.add_user_text("A" * 200)
        history.add_assistant_response(
            CanonicalResponse(
                content=[TextBlock(text="B" * 200)],
                stop_reason="end_turn",
                usage=Usage(input_tokens=5, output_tokens=5),
            )
        )

    initial_count = len(history.messages)
    compacted_events = []
    async for ev in cm.maybe_compact(provider, history, "system prompt", []):
        compacted_events.append(ev)

    from neurosurfer.agents.conversation.events import Compacted
    assert compacted_events, "Expected compaction to fire with reported_tokens above threshold"
    assert isinstance(compacted_events[0], Compacted)
    # History must be shorter: the summary replaced the old prefix.
    assert len(history.messages) < initial_count


# ──────────────────────────────────────────────────────────────────────────────
# Durable state survives compaction
# ──────────────────────────────────────────────────────────────────────────────

async def test_durable_state_in_system_after_compaction():
    """Durable state is re-injected via system_with_durable even after compaction."""
    provider = _make_provider(
        summary_text="<summary>Post-compact summary.</summary>",
        context_window=200_000,
        max_output_tokens=32_000,
        reported_tokens=180_000,
    )
    d = DurableState()
    d.set_plan("Approved Plan", "Phase 1: build X\nPhase 2: test Y")
    d.set_todos([{"content": "build X", "status": "in_progress"}])

    cm = ContextManager(provider, durable=d)
    history = _fat_history(n_pairs=5)

    # After compaction the system prompt must still carry durable state.
    async for _ in cm.maybe_compact(provider, history, "base", []):
        pass

    effective_system = cm.system_with_durable("base")
    assert "Phase 1: build X" in effective_system
    assert "[~] build X" in effective_system


# ──────────────────────────────────────────────────────────────────────────────
# Tool integration: todo / present_plan write to DurableState
# ──────────────────────────────────────────────────────────────────────────────

async def test_todo_tool_writes_to_durable_state(tmp_path: Path):
    """TodoTool must persist items to DurableState when ctx.durable is set."""
    from neurosurfer.tools.base import ToolContext
    from neurosurfer.tools.builtin.todo import TodoTool

    d = DurableState()
    io = ScriptedIO()
    ctx = ToolContext(cwd=tmp_path, io=io, durable=d)

    tool = TodoTool()
    result = await tool.call(
        tool.input_model.model_validate(
            {
                "todos": [
                    {"content": "task A", "status": "pending"},
                    {"content": "task B", "status": "completed"},
                ]
            }
        ),
        ctx,
    )
    assert not result.is_error
    assert d.todos == [
        {"content": "task A", "status": "pending"},
        {"content": "task B", "status": "completed"},
    ]
    assert "[x] task B" in d.to_context_block()


async def test_present_plan_tool_writes_to_durable_state(tmp_path: Path):
    """PresentPlanTool must save plan to DurableState when ctx.durable is set."""
    from neurosurfer.app.tools.present_plan import PresentPlanTool
    from neurosurfer.tools.base import ToolContext

    d = DurableState()
    io = ScriptedIO(approve_plan=True)
    ctx = ToolContext(cwd=tmp_path, io=io, durable=d)

    tool = PresentPlanTool()
    result = await tool.call(
        tool.input_model.model_validate(
            {"plan": "## Phase 1\nDo X\n## Phase 2\nDo Y", "title": "Sprint Plan"}
        ),
        ctx,
    )
    assert not result.is_error
    assert d.plan_title == "Sprint Plan"
    assert "Do X" in (d.plan_text or "")
