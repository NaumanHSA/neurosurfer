"""CachedProvider — transparent Provider wrapper that caches complete() calls."""
from __future__ import annotations

import hashlib
import json
from collections.abc import AsyncIterator

from neurosurfer.llm.base import Provider
from neurosurfer.llm.types import (
    CanonicalResponse,
    GenerationConfig,
    Message,
    StreamEvent,
    ToolSchema,
)

from .base import CacheKey, ResponseCache


def _make_key(
    model: str,
    messages: list[Message],
    system: str | None,
    tools: list[ToolSchema],
    config: GenerationConfig,
) -> CacheKey:
    payload = {
        "model": model,
        "messages": [m.model_dump() for m in messages],
        "system": system,
        "tools": [t.model_dump() for t in tools],
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()
    ).hexdigest()
    return CacheKey(key=digest)


class CachedProvider(Provider):
    """Wraps any Provider and caches complete() responses.

    stream() is intentionally **not** cached — streaming events can't be
    replayed from a stored response without the consumer interface changing.

    When ``cache=None`` this is a zero-overhead transparent pass-through: all
    calls are forwarded directly to the underlying provider.

    Usage::

        provider = build_provider(cfg)
        cache = InMemoryResponseCache(maxsize=256, ttl=3600)
        provider = CachedProvider(provider, cache=cache)
        # or: CachedProvider(provider, cache=None)  # off, no overhead
    """

    def __init__(self, provider: Provider, cache: ResponseCache | None) -> None:
        self._provider = provider
        self._cache = cache
        self.model = provider.model
        self.capabilities = provider.capabilities

    # ------------------------------------------------------------------ #
    def stream(
        self,
        messages: list[Message],
        system: str | None,
        tools: list[ToolSchema],
        config: GenerationConfig,
    ) -> AsyncIterator[StreamEvent]:
        return self._provider.stream(messages, system, tools, config)

    async def complete(
        self,
        messages: list[Message],
        system: str | None,
        tools: list[ToolSchema],
        config: GenerationConfig,
    ) -> CanonicalResponse:
        if self._cache is None:
            return await self._provider.complete(messages, system, tools, config)

        key = _make_key(self.model, messages, system, tools, config)
        hit = self._cache.get(key)
        if hit is not None:
            return hit

        response = await self._provider.complete(messages, system, tools, config)
        self._cache.set(key, response)
        return response

    async def count_tokens(
        self,
        messages: list[Message],
        system: str | None,
        tools: list[ToolSchema],
    ) -> int:
        return await self._provider.count_tokens(messages, system, tools)
