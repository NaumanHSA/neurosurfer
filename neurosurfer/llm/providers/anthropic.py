"""Anthropic Messages API adapter.

Translates canonical messages ↔ Anthropic's content-block wire format, applies
adaptive thinking + effort and ``cache_control`` breakpoints, normalizes the SSE
stream into canonical events, and exposes the ``count_tokens`` endpoint.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from ...observability.logging import get_logger
from ..base import Provider
from ..capabilities import anthropic_capabilities
from ..retry import with_retry
from ..tokens import estimate_messages_tokens
from ..types import (
    CanonicalResponse,
    Done,
    GenerationConfig,
    Message,
    StreamEvent,
    TextBlock,
    TextDelta,
    ThinkingBlock,
    ThinkingDelta,
    ToolResultBlock,
    ToolSchema,
    ToolUseArgsDelta,
    ToolUseBlock,
    ToolUseStart,
    Usage,
)

log = get_logger("llm.anthropic")


def _block_to_param(block: Any) -> dict[str, Any] | None:
    if isinstance(block, TextBlock):
        return {"type": "text", "text": block.text}
    if isinstance(block, ThinkingBlock):
        # Thinking can only be sent back if it carries its signature.
        if not block.signature:
            return None
        return {
            "type": "thinking",
            "thinking": block.thinking,
            "signature": block.signature,
        }
    if isinstance(block, ToolUseBlock):
        return {
            "type": "tool_use",
            "id": block.id,
            "name": block.name,
            "input": block.input,
        }
    if isinstance(block, ToolResultBlock):
        return {
            "type": "tool_result",
            "tool_use_id": block.tool_use_id,
            "content": block.content,
            "is_error": block.is_error,
        }
    return None


def to_anthropic_messages(messages: list[Message]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for msg in messages:
        params = [p for p in (_block_to_param(b) for b in msg.content) if p is not None]
        if not params:
            # Anthropic rejects empty content; keep a placeholder.
            params = [{"type": "text", "text": ""}]
        out.append({"role": msg.role, "content": params})
    return out


def to_anthropic_tools(tools: list[ToolSchema]) -> list[dict[str, Any]]:
    rendered = [
        {"name": t.name, "description": t.description, "input_schema": t.input_schema}
        for t in tools
    ]
    # Cache the (static) tools block: breakpoint on the last tool.
    if rendered:
        rendered[-1]["cache_control"] = {"type": "ephemeral"}
    return rendered


def _system_param(system: str | None) -> list[dict[str, Any]] | None:
    if not system:
        return None
    # Single cacheable system block — the static prefix is the cache boundary.
    return [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]


class AnthropicProvider(Provider):
    def __init__(self, api_key: str, model: str):
        import anthropic

        self._client = anthropic.AsyncAnthropic(api_key=api_key)
        self.model = model
        self.capabilities = anthropic_capabilities(model)

    def _build_kwargs(
        self,
        messages: list[Message],
        system: str | None,
        tools: list[ToolSchema],
        config: GenerationConfig,
        *,
        with_thinking: bool,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": config.max_tokens,
            "messages": to_anthropic_messages(messages),
        }
        sys_param = _system_param(system)
        if sys_param is not None:
            kwargs["system"] = sys_param
        if tools:
            kwargs["tools"] = to_anthropic_tools(tools)
        if config.stop_sequences:
            kwargs["stop_sequences"] = config.stop_sequences

        if with_thinking and self.capabilities.supports_thinking and config.enable_thinking:
            # Adaptive thinking requires temperature == 1.0.
            kwargs["thinking"] = {"type": "adaptive"}
            kwargs["temperature"] = 1.0
            # effort is passed through the raw body for forward-compatibility.
            kwargs["extra_body"] = {"effort": config.effort}
        else:
            kwargs["temperature"] = config.temperature
        return kwargs

    async def stream(
        self,
        messages: list[Message],
        system: str | None,
        tools: list[ToolSchema],
        config: GenerationConfig,
    ) -> AsyncIterator[StreamEvent]:  # type: ignore[override]
        async for event in self._stream_inner(messages, system, tools, config):
            yield event

    async def _stream_inner(
        self,
        messages: list[Message],
        system: str | None,
        tools: list[ToolSchema],
        config: GenerationConfig,
    ):
        with_thinking = True
        while True:
            kwargs = self._build_kwargs(
                messages, system, tools, config, with_thinking=with_thinking
            )
            try:
                async for ev in self._run_stream(kwargs):
                    yield ev
                return
            except TypeError as e:
                # SDK too old for thinking/effort kwargs — retry once without them.
                if with_thinking:
                    log.warning("retrying without thinking/effort kwargs: %s", e)
                    with_thinking = False
                    continue
                raise

    async def _run_stream(self, kwargs: dict[str, Any]):
        index_to_block: dict[int, ToolUseBlock] = {}

        async def open_stream():
            return self._client.messages.stream(**kwargs)

        stream_cm = await with_retry(open_stream)
        async with stream_cm as stream:
            async for event in stream:
                etype = getattr(event, "type", None)
                if etype == "content_block_start":
                    block: Any = getattr(event, "content_block", None)
                    if getattr(block, "type", None) == "tool_use":
                        idx = getattr(event, "index", 0)
                        index_to_block[idx] = ToolUseBlock(
                            id=block.id, name=block.name, input={}
                        )
                        yield ToolUseStart(index=idx, id=block.id, name=block.name)
                elif etype == "content_block_delta":
                    delta: Any = getattr(event, "delta", None)
                    dtype = getattr(delta, "type", None)
                    if dtype == "text_delta":
                        yield TextDelta(text=delta.text)
                    elif dtype == "thinking_delta":
                        yield ThinkingDelta(text=delta.thinking)
                    elif dtype == "input_json_delta":
                        yield ToolUseArgsDelta(
                            index=getattr(event, "index", 0),
                            partial_json=delta.partial_json,
                        )
            final = await stream.get_final_message()
        yield Done(response=self._final_to_response(final))

    def _final_to_response(self, final: Any) -> CanonicalResponse:
        content: list[Any] = []
        for block in final.content:
            btype = getattr(block, "type", None)
            if btype == "text":
                content.append(TextBlock(text=block.text))
            elif btype == "thinking":
                content.append(
                    ThinkingBlock(
                        thinking=block.thinking,
                        signature=getattr(block, "signature", None),
                    )
                )
            elif btype == "tool_use":
                content.append(
                    ToolUseBlock(id=block.id, name=block.name, input=block.input or {})
                )
        u = final.usage
        usage = Usage(
            input_tokens=getattr(u, "input_tokens", 0) or 0,
            output_tokens=getattr(u, "output_tokens", 0) or 0,
            cache_read_input_tokens=getattr(u, "cache_read_input_tokens", 0) or 0,
            cache_creation_input_tokens=getattr(u, "cache_creation_input_tokens", 0) or 0,
        )
        return CanonicalResponse(
            content=content,
            stop_reason=getattr(final, "stop_reason", "end_turn") or "end_turn",
            usage=usage,
            model=self.model,
        )

    async def count_tokens(
        self,
        messages: list[Message],
        system: str | None,
        tools: list[ToolSchema],
    ) -> int:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": to_anthropic_messages(messages),
        }
        sys_param = _system_param(system)
        if sys_param is not None:
            kwargs["system"] = sys_param
        if tools:
            kwargs["tools"] = to_anthropic_tools(tools)
        try:
            result = await with_retry(lambda: self._client.messages.count_tokens(**kwargs))
            return int(result.input_tokens)
        except Exception as e:  # noqa: BLE001 - fall back to estimate
            log.debug("count_tokens endpoint failed, estimating: %s", e)
            return estimate_messages_tokens(messages, system, tools)
