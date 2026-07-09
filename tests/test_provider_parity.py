"""Provider-parity suite (Phase 1 exit criterion).

The same scripted conversation runs through both adapters behind the single
``Provider`` interface and must yield equivalent canonical results: same streamed
event categories, same assembled text / tool calls / stop reason.
"""

from __future__ import annotations

import pytest

from neurosurfer.llm.providers.anthropic import (
    AnthropicProvider,
    _system_param,
    to_anthropic_messages,
    to_anthropic_tools,
)
from neurosurfer.llm.providers.openai import (
    OpenAICompatProvider,
    repair_json_args,
    to_openai_messages,
)
from neurosurfer.llm.retry import is_retryable_error, with_retry
from neurosurfer.llm.types import (
    Done,
    GenerationConfig,
    Message,
    TextBlock,
    TextDelta,
    ToolResultBlock,
    ToolSchema,
    ToolUseArgsDelta,
    ToolUseBlock,
    ToolUseStart,
)

from .fakes import FakeAnthropicClient, FakeOpenAIClient

TOOLS = [ToolSchema(name="read_file", description="read", input_schema={"type": "object"})]
CFG = GenerationConfig(max_tokens=256)


def make_anthropic(turns):
    p = AnthropicProvider(api_key="test", model="claude-opus-4-8")
    p._client = FakeAnthropicClient(turns)  # type: ignore[attr-defined]
    return p


def make_openai(turns):
    p = OpenAICompatProvider(
        base_url="http://x/v1", api_key="test", model="local", context_window=32768
    )
    p._client = FakeOpenAIClient(turns)  # type: ignore[attr-defined]
    return p


async def collect(provider, messages):
    events = []
    response = None
    async for ev in provider.stream(messages, "system prompt", TOOLS, CFG):
        events.append(ev)
        if isinstance(ev, Done):
            response = ev.response
    return events, response


def summarize(response):
    return {
        "text": response.text(),
        "tools": [(t.name, t.input) for t in response.tool_uses()],
        "stop_reason": response.stop_reason,
    }


def event_categories(events):
    cats = []
    for e in events:
        if isinstance(e, TextDelta):
            cats.append("text")
        elif isinstance(e, ToolUseStart):
            cats.append("tool_start")
        elif isinstance(e, ToolUseArgsDelta):
            cats.append("tool_args")
        elif isinstance(e, Done):
            cats.append("done")
    return cats


@pytest.mark.asyncio
async def test_parity_text_then_tool_call():
    turn = ("Let me read that.", [("read_file", {"path": "a.py"})])
    msgs = [Message.user_text("document this repo")]

    a_events, a_resp = await collect(make_anthropic([turn]), msgs)
    o_events, o_resp = await collect(make_openai([turn]), msgs)

    # Assembled responses are equivalent.
    assert summarize(a_resp) == summarize(o_resp)
    assert summarize(a_resp) == {
        "text": "Let me read that.",
        "tools": [("read_file", {"path": "a.py"})],
        "stop_reason": "tool_use",
    }
    # Both stream the same high-level event categories in the same order.
    assert event_categories(a_events) == ["text", "tool_start", "tool_args", "done"]
    assert event_categories(o_events) == ["text", "tool_start", "tool_args", "done"]


@pytest.mark.asyncio
async def test_parity_plain_text_turn():
    turn = ("All done — docs written.", [])
    msgs = [Message.user_text("finish up")]

    _, a_resp = await collect(make_anthropic([turn]), msgs)
    _, o_resp = await collect(make_openai([turn]), msgs)

    assert summarize(a_resp) == summarize(o_resp)
    assert a_resp.stop_reason == "end_turn"
    assert a_resp.text() == "All done — docs written."


@pytest.mark.asyncio
async def test_parity_multi_turn_tool_loop():
    # Turn 1: tool call. Turn 2: final text. Same script for both providers.
    turns = [("", [("read_file", {"path": "x"})]), ("Here is the summary.", [])]
    msgs = [Message.user_text("go")]

    for make in (make_anthropic, make_openai):
        provider = make(list(turns))
        _, r1 = await collect(provider, msgs)
        assert r1.stop_reason == "tool_use"
        tu = r1.tool_uses()[0]
        # Append assistant turn + tool result, then second turn.
        convo = msgs + [
            r1.as_message(),
            Message(
                role="user",
                content=[ToolResultBlock(tool_use_id=tu.id, content="file contents")],
            ),
        ]
        _, r2 = await collect(provider, convo)
        assert r2.text() == "Here is the summary."
        assert r2.stop_reason == "end_turn"


def test_openai_message_conversion_tool_roundtrip():
    convo = [
        Message.user_text("go"),
        Message(
            role="assistant",
            content=[
                TextBlock(text="reading"),
                ToolUseBlock(id="c1", name="read_file", input={"path": "x"}),
            ],
        ),
        Message(
            role="user",
            content=[ToolResultBlock(tool_use_id="c1", content="data", is_error=False)],
        ),
    ]
    out = to_openai_messages(convo, "sys")
    roles = [m["role"] for m in out]
    assert roles == ["system", "user", "assistant", "tool"]
    assert out[2]["tool_calls"][0]["function"]["name"] == "read_file"
    assert out[3]["tool_call_id"] == "c1"


def test_anthropic_message_conversion_blocks():
    convo = [
        Message(
            role="assistant",
            content=[ToolUseBlock(id="c1", name="read_file", input={"path": "x"})],
        ),
        Message(
            role="user",
            content=[ToolResultBlock(tool_use_id="c1", content="data", is_error=True)],
        ),
    ]
    out = to_anthropic_messages(convo)
    assert out[0]["content"][0]["type"] == "tool_use"
    assert out[1]["content"][0]["type"] == "tool_result"
    assert out[1]["content"][0]["is_error"] is True


def test_malformed_tool_args_repair():
    # Trailing comma — repairable.
    parsed, ok = repair_json_args('{"path": "a.py",}')
    assert ok and parsed == {"path": "a.py"}
    # Junk around JSON — repairable by extraction.
    parsed, ok = repair_json_args('here you go: {"x": 1} thanks')
    assert ok and parsed == {"x": 1}
    # Empty args are valid (no-arg tool).
    parsed, ok = repair_json_args("")
    assert ok and parsed == {}
    # Truly broken — flagged, returns empty so loop self-corrects.
    parsed, ok = repair_json_args("{not json at all")
    assert not ok


@pytest.mark.asyncio
async def test_retry_on_transient_then_success():
    calls = {"n": 0}

    class Boom(Exception):
        status_code = 529

    async def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise Boom()
        return "ok"

    assert is_retryable_error(Boom())
    result = await with_retry(flaky, max_attempts=5, base_delay=0.001, max_delay=0.01)
    assert result == "ok"
    assert calls["n"] == 3


def test_terminal_error_not_retried():
    class AuthError(Exception):
        status_code = 401

    assert not is_retryable_error(AuthError())


# ── Phase 9 hardening: prompt-cache audit (Anthropic) ─────────────────────────
# Relocated from the former tests/test_hardening.py — these protect a genuine
# correctness property: cache_control breakpoints are emitted on the system
# block and the last tool, and cache_read usage is parsed back correctly.
class TestPromptCacheAudit:
    def test_anthropic_cache_breakpoints_emitted(self):
        sys_param = _system_param("You are an agent.")
        assert sys_param is not None
        assert sys_param[0]["cache_control"] == {"type": "ephemeral"}

        tools = [
            ToolSchema(name="a", description="d", input_schema={"type": "object"}),
            ToolSchema(name="b", description="d", input_schema={"type": "object"}),
        ]
        rendered = to_anthropic_tools(tools)
        # Only the last tool carries the cache breakpoint.
        assert "cache_control" not in rendered[0]
        assert rendered[-1]["cache_control"] == {"type": "ephemeral"}

    def test_anthropic_parses_cache_read_usage(self):
        from types import SimpleNamespace

        p = AnthropicProvider(api_key="test", model="claude-opus-4-8")
        final = SimpleNamespace(
            content=[SimpleNamespace(type="text", text="hi")],
            stop_reason="end_turn",
            usage=SimpleNamespace(
                input_tokens=10,
                output_tokens=5,
                cache_read_input_tokens=1234,
                cache_creation_input_tokens=42,
            ),
        )
        resp = p._final_to_response(final)
        assert resp.usage.cache_read_input_tokens == 1234
        assert resp.usage.cache_creation_input_tokens == 42
