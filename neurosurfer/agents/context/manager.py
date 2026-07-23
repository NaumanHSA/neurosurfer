"""Context management: proactive + reactive compaction, durable-state injection.

Threshold math:
  effective_window       = context_window − min(max_output_tokens, 20_000)
  auto_compact_threshold = effective_window − 13_000

The ContextManager is injected into the Agent and is called at two points:

  1. maybe_compact()         — proactive: before each model call, compact if
                               token count exceeds the threshold.
  2. stream_with_recovery()  — reactive: wraps provider.stream(); on a
                               context-overflow API error it compacts in-place
                               and retries the stream once.

system_with_durable() appends the durable state block to the system prompt so
that compaction can never drop the plan, manifest, todos, or decisions.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

from neurosurfer.agents.conversation import events
from neurosurfer.agents.conversation.messages import MessageHistory
from neurosurfer.llm.retry import is_context_overflow_error
from neurosurfer.llm.tokens import auto_compact_threshold, estimate_messages_tokens
from neurosurfer.llm.types import GenerationConfig, Message, StreamEvent, TextBlock
from neurosurfer.observability.logging import get_logger

from .summary_prompt import get_compact_prompt, get_compact_user_summary_message

if TYPE_CHECKING:
    from neurosurfer.llm.base import Provider
    from neurosurfer.llm.capabilities import ProviderCapabilities
    from neurosurfer.llm.types import ToolSchema

    from .durable_state import DurableState

log = get_logger("core.context_manager")

# How many messages to keep from the tail when replacing history with a summary.
# 4 messages = 2 full turns (user+assistant pairs), enough recent context to
# avoid abrupt context breaks while keeping the compacted history small.
_KEEP_RECENT_MESSAGES = 4


class ContextManager:
    """Owns compaction and durable-state injection for one Agent session."""

    def __init__(
        self,
        provider: Provider,
        *,
        durable: DurableState | None = None,
        task_must_preserve: list[str] | None = None,
    ) -> None:
        self._caps: ProviderCapabilities = provider.capabilities
        self._durable = durable
        self._task_must_preserve = task_must_preserve

    # ── helpers ───────────────────────────────────────────────────────────────

    async def _count_tokens(
        self,
        provider: Provider,
        history: MessageHistory,
        system: str | None,
        tools: list[ToolSchema],
    ) -> int:
        """Use the provider's token-count endpoint when available, else estimate."""
        if self._caps.supports_token_count:
            return await provider.count_tokens(history.messages, system, tools)
        return estimate_messages_tokens(history.messages, system, tools)

    def _threshold(self) -> int:
        return auto_compact_threshold(self._caps.context_window, self._caps.max_output_tokens)

    # ── durable-state injection ───────────────────────────────────────────────

    def system_with_durable(self, base_system: str) -> str:
        """Append the durable-state block to the system prompt if non-empty."""
        if self._durable is None:
            return base_system
        block = self._durable.to_context_block()
        if not block:
            return base_system
        return f"{base_system}\n\n{block}"

    # ── compaction core ───────────────────────────────────────────────────────

    async def _do_compact(
        self,
        provider: Provider,
        history: MessageHistory,
        system: str | None,
        tools: list[ToolSchema],
    ) -> tuple[int, int]:
        """Compact history in-place; returns (tokens_before, tokens_after).

        Sends the entire current conversation plus a compaction user-message to
        the provider (no tools, one shot), parses the summary, then replaces the
        old message prefix with the formatted summary keeping the recent tail.
        Mirrors compactConversation() in compact.ts.
        """
        tokens_before = await self._count_tokens(provider, history, system, tools)
        log.info(
            "compacting: %d tokens (threshold=%d, window=%d)",
            tokens_before,
            self._threshold(),
            self._caps.context_window,
        )

        compact_prompt = get_compact_prompt(task_must_preserve=self._task_must_preserve)

        # The summarization request is the conversation so far plus the prompt.
        summarize_msgs: list[Message] = [
            *history.messages,
            Message(role="user", content=[TextBlock(text=compact_prompt)]),
        ]

        # No tools, no thinking — the preamble explicitly tells the model not
        # to call tools; thinking adds unnecessary latency + tokens here.
        summary_config = GenerationConfig(
            enable_thinking=False,
            stream=False,
        )

        response = await provider.complete(
            summarize_msgs,
            None,  # no system — conversation already has full context
            [],    # no tools — model explicitly instructed not to call any
            summary_config,
        )

        raw_summary = response.text()
        formatted = get_compact_user_summary_message(raw_summary)

        # Swap the old prefix for the summary, keep the most recent tail.
        history.replace_prefix_with_summary(_KEEP_RECENT_MESSAGES, formatted)

        tokens_after = await self._count_tokens(provider, history, system, tools)
        log.info("compacted: %d → %d tokens", tokens_before, tokens_after)
        return tokens_before, tokens_after

    # ── public API (called by Agent) ──────────────────────────────────────────

    async def maybe_compact(
        self,
        provider: Provider,
        history: MessageHistory,
        system: str | None,
        tools: list[ToolSchema],
    ) -> AsyncIterator[events.Event]:
        """Proactive compaction: yield a Compacted event if above threshold.

        Called at the top of every agent turn, before the model call.
        Mirrors shouldAutoCompact() + autoCompactIfNeeded() in autoCompact.ts.
        """
        current = await self._count_tokens(provider, history, system, tools)
        if current <= self._threshold():
            return
        log.info("proactive compact triggered (tokens=%d, threshold=%d)", current, self._threshold())
        tokens_before, tokens_after = await self._do_compact(provider, history, system, tools)
        yield events.Compacted(tokens_before=tokens_before, tokens_after=tokens_after)

    async def stream_with_recovery(
        self,
        provider: Provider,
        history: MessageHistory,
        system: str | None,
        tools: list[ToolSchema],
        config: GenerationConfig,
    ) -> AsyncIterator[StreamEvent]:
        """Stream with reactive compaction on context-overflow.

        Attempts the normal stream; if the provider raises a context-overflow
        error (prompt-too-long / context_length_exceeded), compacts in-place and
        retries the stream exactly once.  Mirrors the reactive-compact path in
        compact.ts (compactConversation called from the overflow error handler).
        """
        try:
            async for ev in provider.stream(history.messages, system, tools, config):
                yield ev
        except Exception as exc:  # noqa: BLE001
            if not is_context_overflow_error(exc):
                raise
            log.warning("reactive compact: context overflow — compacting and retrying")
            try:
                await self._do_compact(provider, history, system, tools)
            except Exception as compact_exc:  # noqa: BLE001
                log.error("reactive compaction failed: %s", compact_exc)
                raise exc from compact_exc
            # One retry after compaction.
            async for ev in provider.stream(history.messages, system, tools, config):
                yield ev
