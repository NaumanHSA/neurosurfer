"""Phase 9 — hardening tests: prompt-cache audit.

- Prompt-cache audit (Anthropic): cache_control breakpoints are emitted on the
  system block and the last tool, and cache_read usage is parsed back.
"""

from __future__ import annotations


# ── prompt-cache audit (Anthropic) ────────────────────────────────────────────
def test_anthropic_cache_breakpoints_emitted():
    from neurosurfer.llm.providers.anthropic import _system_param, to_anthropic_tools
    from neurosurfer.llm.types import ToolSchema

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


def test_anthropic_parses_cache_read_usage():
    from types import SimpleNamespace

    from neurosurfer.llm.providers.anthropic import AnthropicProvider

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
