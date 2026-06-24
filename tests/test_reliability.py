"""Per-provider reliability profile + scaffold injection (Pillar 0b)."""

from __future__ import annotations

from pathlib import Path

from neurosurfer.agents.runtime.permissions import Guardrails
from neurosurfer.llm.capabilities import (
    anthropic_capabilities,
    openai_capabilities,
    reliability_profile,
)
from neurosurfer.prompts.base_agent import think_scaffold_section
from neurosurfer.prompts.system import build_system_prompt


def test_profile_frontier_thinking_model():
    p = reliability_profile(anthropic_capabilities("claude-opus-4-8"))
    assert p.think_scaffold is False
    assert p.tool_arg_repair is False
    assert p.max_concurrent_subagents == 4


def test_profile_nonthinking_anthropic_gets_scaffold_only():
    p = reliability_profile(anthropic_capabilities("claude-haiku-4-5"))
    assert p.think_scaffold is True          # haiku has no native thinking
    assert p.max_concurrent_subagents == 4   # but it is not the weaker "local" tier
    assert p.tool_arg_repair is False


def test_profile_local_openai_model():
    p = reliability_profile(openai_capabilities("local-model", 8192))
    assert p.think_scaffold is True
    assert p.tool_arg_repair is True
    assert p.max_concurrent_subagents == 2


def test_think_scaffold_section_text():
    s = think_scaffold_section()
    assert s.strip() and "Working method" in s


def test_extra_sections_injected_after_base_prompt():
    out = build_system_prompt(
        task_instructions="do the thing",
        guardrails=Guardrails(),
        cwd=Path("/tmp"),
        model="test-model",
        extra_sections=[think_scaffold_section()],
    )
    assert "# Working method" in out
    # The scaffold rides the extra_sections tail — after the task instructions.
    assert out.index("# Working method") > out.index("do the thing")
