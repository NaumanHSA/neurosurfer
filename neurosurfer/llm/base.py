"""The provider-neutral interface the engine depends on.

Two implementations exist (Anthropic, OpenAI-compatible). The engine never
imports ``anthropic`` or ``openai`` — it talks only to this protocol and the
canonical types in :mod:`neurosurfer.llm.types`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from .capabilities import ProviderCapabilities
from .types import (
    CanonicalResponse,
    Done,
    GenerationConfig,
    Message,
    StreamEvent,
    ToolSchema,
    Usage,
)


class Provider(ABC):
    """Canonical message/tool/stream interface behind one type."""

    capabilities: ProviderCapabilities
    model: str

    @abstractmethod
    def stream(
        self,
        messages: list[Message],
        system: str | None,
        tools: list[ToolSchema],
        config: GenerationConfig,
    ) -> AsyncIterator[StreamEvent]:
        """Stream a single assistant turn as normalized canonical events.

        The final event is always :class:`~neurosurfer.llm.types.Done`, carrying
        the fully assembled :class:`CanonicalResponse`.
        """
        ...

    async def complete(
        self,
        messages: list[Message],
        system: str | None,
        tools: list[ToolSchema],
        config: GenerationConfig,
    ) -> CanonicalResponse:
        """Non-streaming convenience: drain :meth:`stream` and return the result."""
        response: CanonicalResponse | None = None
        async for event in self.stream(messages, system, tools, config):
            if isinstance(event, Done):
                response = event.response
        if response is None:  # pragma: no cover - stream always ends with Done
            raise RuntimeError("provider stream ended without a Done event")
        return response

    @abstractmethod
    async def count_tokens(
        self,
        messages: list[Message],
        system: str | None,
        tools: list[ToolSchema],
    ) -> int:
        """Count input tokens. Anthropic uses the endpoint; OpenAI-compatible
        providers return a local estimate."""
        ...

    def empty_usage(self) -> Usage:
        return Usage()
