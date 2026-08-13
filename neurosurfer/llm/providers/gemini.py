"""Google Gemini, over its native REST API.

The third shape. Anthropic and OpenAI cover a great deal of the ecosystem
between them, but Gemini is neither — different envelope, different names for
the same ideas, and a genuinely different tool-call protocol.

**Over `httpx` rather than `google-genai`.** `httpx` is already a required
dependency; the Google SDK would be a new one for a single provider, and its
surface changes faster than the REST endpoint it wraps. Everything here is the
documented wire format, which also means it is testable against a fake server —
`tests/test_gemini_provider.py` runs the full stream against one.

The translation, in the places it is not obvious:

* Roles are ``user`` and ``model``, not ``user`` and ``assistant``.
* The system prompt is its own top-level ``systemInstruction``, not a turn.
* A tool call is a ``functionCall`` part with **already-parsed args** — no JSON
  string to assemble, so `ToolUseArgsDelta` carries the whole payload at once.
* A tool *result* is a ``functionResponse`` part on a **user** turn, keyed by
  the function's *name* rather than by a call id — so the id has to be mapped
  back, which is what `_call_names` exists for.
"""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator
from typing import Any

import httpx

from ..base import Provider
from ..capabilities import ProviderCapabilities
from ..types import (
    CanonicalResponse,
    Done,
    GenerationConfig,
    ImageBlock,
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

__all__ = ["GeminiProvider"]

DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
DEFAULT_MODEL = "gemini-2.5-flash"

#: Gemini's own stop reasons → the canonical vocabulary the engine understands.
_STOP_REASONS = {
    "STOP": "end_turn",
    "MAX_TOKENS": "max_tokens",
    "SAFETY": "refusal",
    "RECITATION": "refusal",
    "PROHIBITED_CONTENT": "refusal",
    "BLOCKLIST": "refusal",
}


class GeminiProvider(Provider):
    """Gemini via `generativelanguage.googleapis.com`."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        *,
        context_window: int = 1_048_576,
        max_output_tokens: int = 65_536,
        supports_vision: bool = True,
        timeout: float = 600.0,
    ) -> None:
        key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set (nor GOOGLE_API_KEY). Pass api_key= "
                "explicitly, or set one of them."
            )
        self.model = model
        self.api_key = key
        self.base_url = base_url.rstrip("/")
        self._timeout = timeout
        self.capabilities = ProviderCapabilities(
            supports_thinking=True,
            supports_prompt_cache=False,
            supports_token_count=True,
            # Not "openai" — the wire shape is Gemini's own. The engine reads
            # this to pick AgenticLoop over the text-parsing ReactAgent, and
            # `functionCall` parts are native tool use, so `openai` is the
            # honest answer to "does this provider call tools natively".
            tool_call_style="openai",
            context_window=context_window,
            max_output_tokens=max_output_tokens,
            supports_vision=supports_vision,
        )

    # ── the stream ───────────────────────────────────────────────────────────

    async def stream(
        self,
        messages: list[Message],
        system: str | None,
        tools: list[ToolSchema],
        config: GenerationConfig,
    ) -> AsyncIterator[StreamEvent]:
        payload = self._build_payload(messages, system, tools, config)
        url = f"{self.base_url}/models/{self.model}:streamGenerateContent"

        content: list[Any] = []
        usage = Usage()
        stop_reason = "end_turn"
        tool_index = 0

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            async with client.stream(
                "POST",
                url,
                params={"alt": "sse", "key": self.api_key},
                json=payload,
                headers={"Content-Type": "application/json"},
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    chunk = _parse_sse_line(line)
                    if chunk is None:
                        continue

                    if (meta := chunk.get("usageMetadata")):
                        usage = _usage_from(meta)

                    for candidate in chunk.get("candidates") or []:
                        if (reason := candidate.get("finishReason")):
                            stop_reason = _STOP_REASONS.get(reason, "end_turn")
                        for part in (candidate.get("content") or {}).get("parts") or []:
                            for event in self._events_for(part, content, tool_index):
                                if isinstance(event, ToolUseStart):
                                    tool_index += 1
                                yield event

        if any(isinstance(b, ToolUseBlock) for b in content):
            stop_reason = "tool_use"

        yield Done(
            response=CanonicalResponse(
                content=content, stop_reason=stop_reason, usage=usage, model=self.model
            )
        )

    def _events_for(self, part: dict, content: list, index: int):
        """One Gemini part → the canonical events it produces, appending blocks."""
        if (call := part.get("functionCall")) is not None:
            # **Args arrive parsed, not as a JSON string.** Every other provider
            # here streams argument text that the caller assembles; Gemini hands
            # over the object. It is re-serialised for `ToolUseArgsDelta` so a
            # consumer written against the other two sees the shape it expects.
            args = call.get("args") or {}
            call_id = f"gemini_call_{index}"
            content.append(ToolUseBlock(id=call_id, name=call.get("name", ""), input=args))
            yield ToolUseStart(index=index, id=call_id, name=call.get("name", ""))
            yield ToolUseArgsDelta(index=index, partial_json=json.dumps(args))
            return

        text = part.get("text")
        if not text:
            return
        # `thought: true` marks reasoning rather than answer — the same split
        # `TextDelta`/`ThinkingDelta` carries everywhere else in this framework.
        if part.get("thought"):
            _append_text(content, ThinkingBlock, "thinking", text)
            yield ThinkingDelta(text=text)
        else:
            _append_text(content, TextBlock, "text", text)
            yield TextDelta(text=text)

    # ── request building ─────────────────────────────────────────────────────

    def _build_payload(
        self,
        messages: list[Message],
        system: str | None,
        tools: list[ToolSchema],
        config: GenerationConfig,
    ) -> dict[str, Any]:
        generation: dict[str, Any] = {}
        if config.max_tokens is not None:
            generation["maxOutputTokens"] = config.max_tokens
        elif self.capabilities.max_output_tokens:
            generation["maxOutputTokens"] = self.capabilities.max_output_tokens
        if config.temperature is not None:
            generation["temperature"] = config.temperature
        if config.stop_sequences:
            generation["stopSequences"] = list(config.stop_sequences)

        payload: dict[str, Any] = {
            "contents": to_gemini_contents(messages),
            "generationConfig": generation,
        }
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}
        if tools:
            payload["tools"] = [{"functionDeclarations": to_gemini_tools(tools)}]
        return payload

    # ── token counting ───────────────────────────────────────────────────────

    async def count_tokens(
        self,
        messages: list[Message],
        system: str | None = None,
        tools: list[ToolSchema] | None = None,
    ) -> int:
        """Gemini has a real endpoint for this, so use it rather than estimating."""
        payload: dict[str, Any] = {"contents": to_gemini_contents(messages)}
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(
                    f"{self.base_url}/models/{self.model}:countTokens",
                    params={"key": self.api_key},
                    json=payload,
                )
                response.raise_for_status()
                return int(response.json().get("totalTokens", 0))
        except Exception:  # noqa: BLE001
            # Counting is advisory — a failure here must not fail the run that
            # was only trying to decide whether to compact.
            from ..tokens import estimate_messages_tokens

            return estimate_messages_tokens(messages, system, tools or [])


# ── translation helpers, exported so tests can check them directly ──────────


def to_gemini_tools(tools: list[ToolSchema]) -> list[dict[str, Any]]:
    """Canonical tool schemas → Gemini `functionDeclarations`.

    Gemini rejects a parameters object with no properties, so a tool that takes
    nothing is sent with the `parameters` key omitted entirely rather than with
    an empty object.
    """
    out = []
    for tool in tools:
        declaration: dict[str, Any] = {"name": tool.name, "description": tool.description}
        schema = _strip_unsupported(tool.input_schema or {})
        if schema.get("properties"):
            declaration["parameters"] = schema
        out.append(declaration)
    return out


#: Gemini's schema dialect is a subset of JSON Schema and errors on the rest.
_UNSUPPORTED_SCHEMA_KEYS = frozenset(
    {"additionalProperties", "$schema", "$defs", "definitions", "default", "examples"}
)


def _strip_unsupported(schema: Any) -> Any:
    if isinstance(schema, dict):
        return {
            k: _strip_unsupported(v)
            for k, v in schema.items()
            if k not in _UNSUPPORTED_SCHEMA_KEYS
        }
    if isinstance(schema, list):
        return [_strip_unsupported(v) for v in schema]
    return schema


def to_gemini_contents(messages: list[Message]) -> list[dict[str, Any]]:
    """Canonical messages → Gemini `contents`.

    Two things differ from every other adapter here. The assistant role is
    called ``model``; and a tool result is a ``functionResponse`` **keyed by the
    function's name**, not by the call id — so the id has to be resolved back to
    the name of the call it answers, which is what `call_names` tracks.
    """
    call_names: dict[str, str] = {}
    contents: list[dict[str, Any]] = []

    for message in messages:
        parts: list[dict[str, Any]] = []
        for block in message.content:
            if isinstance(block, TextBlock):
                if block.text:
                    parts.append({"text": block.text})
            elif isinstance(block, ThinkingBlock):
                # Reasoning is not replayed: Gemini does not accept it back, and
                # sending it as ordinary text would put the model's private
                # working notes into its own prompt as though they were answers.
                continue
            elif isinstance(block, ToolUseBlock):
                call_names[block.id] = block.name
                parts.append({"functionCall": {"name": block.name, "args": block.input}})
            elif isinstance(block, ToolResultBlock):
                parts.append(
                    {
                        "functionResponse": {
                            "name": call_names.get(block.tool_use_id, block.tool_use_id),
                            "response": {"result": block.content},
                        }
                    }
                )
            elif isinstance(block, ImageBlock):
                if (part := _image_part(block)) is not None:
                    parts.append(part)

        if parts:
            contents.append(
                {"role": "model" if message.role == "assistant" else "user", "parts": parts}
            )
    return contents


def _image_part(block: ImageBlock) -> dict[str, Any] | None:
    source = getattr(block, "source", None) or {}
    if source.get("type") == "base64":
        return {
            "inlineData": {
                "mimeType": source.get("media_type", "image/png"),
                "data": source.get("data", ""),
            }
        }
    # A URL image would need fetching first; Gemini's inline API takes bytes.
    return None


def _append_text(content: list, block_cls, field: str, text: str) -> None:
    """Merge into the trailing block of the same kind, or start a new one.

    Streamed text arrives in fragments; one block per fragment would make the
    assembled response a list of hundreds of one-word blocks.
    """
    if content and isinstance(content[-1], block_cls):
        setattr(content[-1], field, getattr(content[-1], field) + text)
    else:
        content.append(block_cls(**{field: text}))


def _parse_sse_line(line: str) -> dict[str, Any] | None:
    """One SSE line → its JSON payload, or ``None`` for keep-alives and noise."""
    if not line or not line.startswith("data:"):
        return None
    body = line[len("data:") :].strip()
    if not body or body == "[DONE]":
        return None
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        return None


def _usage_from(meta: dict[str, Any]) -> Usage:
    """Gemini's `usageMetadata` → canonical `Usage`.

    `candidatesTokenCount` excludes thinking tokens, which are reported
    separately — they are billed as output, so they are added here rather than
    silently dropped from the total the cost table will price.
    """
    return Usage(
        input_tokens=int(meta.get("promptTokenCount", 0)),
        output_tokens=int(meta.get("candidatesTokenCount", 0))
        + int(meta.get("thoughtsTokenCount", 0)),
        cache_read_input_tokens=int(meta.get("cachedContentTokenCount", 0)),
    )
