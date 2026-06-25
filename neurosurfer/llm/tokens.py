"""Token counting and context-window math.

Anthropic providers use the ``messages.count_tokens`` endpoint (in the adapter);
OpenAI-compatible providers have no such endpoint, so we estimate locally with
``tiktoken`` when available and a character heuristic otherwise.

The window math:
    effective_window = window - min(max_output, MAX_OUTPUT_TOKENS_FOR_SUMMARY)
    auto_compact_threshold = effective_window - AUTOCOMPACT_BUFFER
with a floor so the threshold can never go non-positive.
"""

from __future__ import annotations

import json
from functools import lru_cache

from .types import (
    ImageBlock,
    Message,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolSchema,
    ToolUseBlock,
)

MAX_OUTPUT_TOKENS_FOR_SUMMARY = 20_000
AUTOCOMPACT_BUFFER_TOKENS = 13_000
# Flat per-image estimate. Real cost depends on resolution/provider tiling; this is a
# deliberately coarse upper-ish bound so compaction never undercounts an image-heavy
# history. (Anthropic ~1.15k–1.6k for a typical screenshot; OpenAI similar.)
IMAGE_TOKENS = 1_400


@lru_cache(maxsize=1)
def _tiktoken_encoder():
    try:
        import tiktoken

        return tiktoken.get_encoding("cl100k_base")
    except Exception:  # noqa: BLE001 - tiktoken optional
        return None


def estimate_text_tokens(text: str) -> int:
    enc = _tiktoken_encoder()
    if enc is not None:
        return len(enc.encode(text, disallowed_special=()))
    # Heuristic: ~4 chars per token, never below word count.
    return max(len(text) // 4, len(text.split()))


def _block_to_text(block: object) -> str:
    if isinstance(block, TextBlock):
        return block.text
    if isinstance(block, ThinkingBlock):
        return block.thinking
    if isinstance(block, ToolUseBlock):
        return f"{block.name} {json.dumps(block.input, default=str)}"
    if isinstance(block, ToolResultBlock):
        return block.content
    return str(block)


def estimate_messages_tokens(
    messages: list[Message],
    system: str | None = None,
    tools: list[ToolSchema] | None = None,
) -> int:
    """Local token estimate for a full request payload."""
    total = 0
    if system:
        total += estimate_text_tokens(system)
    for tool in tools or []:
        total += estimate_text_tokens(tool.name + " " + tool.description)
        total += estimate_text_tokens(json.dumps(tool.input_schema, default=str))
    for msg in messages:
        # ~4 tokens of role/structure overhead per message.
        total += 4
        for block in msg.content:
            if isinstance(block, ImageBlock):
                total += IMAGE_TOKENS
            else:
                total += estimate_text_tokens(_block_to_text(block))
    return total


def effective_window(context_window: int, max_output_tokens: int) -> int:
    """Context window minus the reservation for the compaction summary output."""
    reserved = min(max_output_tokens, MAX_OUTPUT_TOKENS_FOR_SUMMARY)
    effective = context_window - reserved
    # Floor: never let the threshold go negative.
    return max(effective, reserved + AUTOCOMPACT_BUFFER_TOKENS)


def auto_compact_threshold(context_window: int, max_output_tokens: int) -> int:
    return effective_window(context_window, max_output_tokens) - AUTOCOMPACT_BUFFER_TOKENS
