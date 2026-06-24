"""Per-provider / per-model feature flags consumed by the engine.

The engine asks capabilities (never the SDK) whether thinking, prompt caching, or
a token-count endpoint are available, and what the context window is. This is how
compaction thresholds stay provider-neutral.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

# Reserved for compaction summary output; also used as the max-output default.
DEFAULT_MAX_OUTPUT_TOKENS = 32_000

# Known Anthropic context windows. Unknown models fall back to 200k.
_ANTHROPIC_WINDOWS: dict[str, int] = {
    "claude-opus-4-8": 200_000,
    "claude-opus-4": 200_000,
    "claude-sonnet-4-6": 200_000,
    "claude-sonnet-4": 200_000,
    "claude-haiku-4-5": 200_000,
}


@dataclass
class ProviderCapabilities:
    supports_thinking: bool
    supports_prompt_cache: bool
    supports_token_count: bool
    tool_call_style: Literal["anthropic", "openai"]
    context_window: int
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS


def _anthropic_window(model: str) -> int:
    for prefix, window in _ANTHROPIC_WINDOWS.items():
        if model.startswith(prefix):
            return window
    return 200_000


def anthropic_capabilities(model: str) -> ProviderCapabilities:
    # Thinking is available on 4.x opus/sonnet families; haiku is treated as
    # non-thinking. Defensive: unknown models default to thinking-capable.
    no_thinking = "haiku" in model
    return ProviderCapabilities(
        supports_thinking=not no_thinking,
        supports_prompt_cache=True,
        supports_token_count=True,
        tool_call_style="anthropic",
        context_window=_anthropic_window(model),
        max_output_tokens=DEFAULT_MAX_OUTPUT_TOKENS,
    )


def openai_capabilities(
    model: str, context_window: int, max_output_tokens: int = 8192
) -> ProviderCapabilities:
    # Local / OpenAI-compatible servers: no count endpoint, no prompt caching,
    # thinking is model-dependent and treated as unavailable. Window is
    # authoritative from config.
    return ProviderCapabilities(
        supports_thinking=False,
        supports_prompt_cache=False,
        supports_token_count=False,
        tool_call_style="openai",
        context_window=context_window,
        # Capped per profile setting (default 8192). Reasoning models spend this
        # budget on chain-of-thought; smaller values keep turns short on local HW.
        max_output_tokens=min(max_output_tokens, context_window),
    )


@dataclass
class ReliabilityProfile:
    """Engine-side knobs that adapt behaviour to a model's strength.

    Frontier models barely need help; smaller / local models benefit from a
    tighter sub-agent fan-out, more aggressive tool-argument repair, and an
    explicit "reason before each tool call" scaffold (they lack native thinking).
    This is the local-first reliability layer: one place that decides how hard the
    engine should compensate for a weaker model.
    """

    max_concurrent_subagents: int
    tool_arg_repair: bool
    think_scaffold: bool


def reliability_profile(caps: ProviderCapabilities) -> ReliabilityProfile:
    """Derive a :class:`ReliabilityProfile` from a provider's capabilities.

    OpenAI-compatible / local servers (``tool_call_style == "openai"``) are the
    weaker tier: fewer parallel sub-agents and tool-arg repair on. Any model
    without native thinking gets the reasoning scaffold (local models, plus haiku).
    """
    is_local = caps.tool_call_style == "openai"
    return ReliabilityProfile(
        max_concurrent_subagents=2 if is_local else 4,
        tool_arg_repair=is_local,
        think_scaffold=not caps.supports_thinking,
    )
