"""OpenAI-compatible adapter (LM Studio / vLLM / Ollama /v1 / llama.cpp / LiteLLM).

Projects the canonical block model onto chat-completions: ``tool_use`` →
``tool_calls`` (function name + JSON-string args), ``tool_result`` → ``tool`` role
messages, system → a system role message. Streamed ``tool_calls`` argument
fragments are reassembled by index, then validated and repaired before being
surfaced as canonical ``tool_use`` blocks (weak local models routinely emit
malformed or partial JSON).
"""

from __future__ import annotations

import json
import re
from collections.abc import AsyncIterator
from typing import Any

from ...observability.logging import get_logger
from ..base import Provider
from ..capabilities import openai_capabilities, openai_native_capabilities, resolve_openai_context_window
from ..retry import with_retry
from ..tokens import estimate_messages_tokens
from ..types import (
    CanonicalResponse,
    ContentBlock,
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

log = get_logger("llm.openai")

_FINISH_REASON_MAP = {
    "tool_calls": "tool_use",
    "function_call": "tool_use",
    "stop": "end_turn",
    "length": "max_tokens",
    "content_filter": "refusal",
}


def _image_url_part(block: ImageBlock) -> dict[str, Any]:
    if block.source == "url":
        url = block.url or ""
    else:
        url = f"data:{block.media_type};base64,{block.data or ''}"
    return {"type": "image_url", "image_url": {"url": url}}


def to_openai_messages(
    messages: list[Message], system: str | None, *, supports_vision: bool = True
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if system:
        out.append({"role": "system", "content": system})
    for msg in messages:
        if msg.role == "assistant":
            text_parts: list[str] = []
            tool_calls: list[dict[str, Any]] = []
            for block in msg.content:
                if isinstance(block, TextBlock):
                    text_parts.append(block.text)
                elif isinstance(block, ThinkingBlock):
                    pass  # OpenAI has no thinking channel; drop on send.
                elif isinstance(block, ToolUseBlock):
                    tool_calls.append(
                        {
                            "id": block.id,
                            "type": "function",
                            "function": {
                                "name": block.name,
                                "arguments": json.dumps(block.input, default=str),
                            },
                        }
                    )
            assistant: dict[str, Any] = {"role": "assistant"}
            assistant["content"] = "".join(text_parts) or None
            if tool_calls:
                assistant["tool_calls"] = tool_calls
            out.append(assistant)
        else:  # user
            tool_results = [b for b in msg.content if isinstance(b, ToolResultBlock)]
            text_parts = [b.text for b in msg.content if isinstance(b, TextBlock)]
            images = [b for b in msg.content if isinstance(b, ImageBlock)]
            for tr in tool_results:
                content = tr.content
                if tr.is_error:
                    content = f"[error] {content}"
                out.append(
                    {"role": "tool", "tool_call_id": tr.tool_use_id, "content": content}
                )
            joined = "".join(text_parts)
            if images and supports_vision:
                # Multimodal user turn: text part (if any) + image_url parts.
                parts: list[dict[str, Any]] = []
                if joined:
                    parts.append({"type": "text", "text": joined})
                parts.extend(_image_url_part(im) for im in images)
                out.append({"role": "user", "content": parts})
            else:
                if images and not supports_vision:
                    note = "[image omitted: model has no vision support]"
                    joined = f"{joined}\n{note}".strip() if joined else note
                if joined or not tool_results:
                    out.append({"role": "user", "content": joined})
    return out


def to_openai_tools(tools: list[ToolSchema]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.input_schema,
            },
        }
        for t in tools
    ]


def repair_json_args(raw: str) -> tuple[dict[str, Any], bool]:
    """Best-effort parse of a tool-call argument string.

    Returns ``(parsed_dict, ok)``. ``ok`` is False when even repair failed, in
    which case the caller surfaces a tool_result error so the loop self-corrects.
    """
    raw = (raw or "").strip()
    if not raw:
        return {}, True
    try:
        val = json.loads(raw)
        return (val if isinstance(val, dict) else {"value": val}), True
    except json.JSONDecodeError:
        pass
    # Repair pass: extract the outermost {...}, strip trailing commas.
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        candidate = re.sub(r",\s*([}\]])", r"\1", match.group(0))
        try:
            val = json.loads(candidate)
            if isinstance(val, dict):
                log.warning("repaired malformed tool-call args")
                return val, True
        except json.JSONDecodeError:
            pass
    log.warning("unrepairable tool-call args: %.120s", raw)
    return {}, False


_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_FN_NAME_RE = re.compile(r"<function\s*=\s*([^>\s]+)\s*>", re.DOTALL)
_FN_PARAM_RE = re.compile(r"<parameter\s*=\s*([^>\s]+)\s*>(.*?)</parameter>", re.DOTALL)


def _coerce_param(raw: str) -> Any:
    """Best-effort typing of a hermes ``<parameter>`` value (JSON scalar or string)."""
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def _parse_inline_tool_call(body: str) -> tuple[str, dict[str, Any]] | None:
    """Parse one ``<tool_call>`` body into ``(name, args)``.

    Handles the two shapes weak local models emit as text: a JSON object
    (``{"name": ..., "arguments": {...}}``) and the hermes XML form
    (``<function=name><parameter=k>v</parameter></function>``).
    """
    body = body.strip()
    fn = _FN_NAME_RE.search(body)
    if fn:
        args = {k: _coerce_param(v) for k, v in _FN_PARAM_RE.findall(body)}
        return fn.group(1), args
    parsed, ok = repair_json_args(body)
    if ok and isinstance(parsed.get("name"), str):
        raw_args = parsed.get("arguments", parsed.get("parameters", {}))
        return parsed["name"], raw_args if isinstance(raw_args, dict) else {}
    return None


def recover_inline_tool_calls(channel: str) -> tuple[list[ToolUseBlock], str]:
    """Extract ``<tool_call>`` markup that leaked into a text/reasoning channel.

    Some servers (LM Studio + Qwen, …) fail to hoist a model's tool call out of
    its content or ``reasoning_content`` into the OpenAI ``tool_calls`` field, so
    the call arrives as raw text and the turn looks like an empty final answer.
    Returns the recovered blocks and the channel text with that markup stripped.
    """
    blocks: list[ToolUseBlock] = []
    for i, match in enumerate(_TOOL_CALL_RE.finditer(channel)):
        parsed = _parse_inline_tool_call(match.group(1))
        if parsed is not None:
            name, args = parsed
            blocks.append(ToolUseBlock(id=f"call_inline_{i}", name=name, input=args))
    if blocks:
        channel = _TOOL_CALL_RE.sub("", channel).strip()
        log.warning("recovered %d tool call(s) leaked into text channel", len(blocks))
    return blocks, channel


class _ToolCallAccumulator:
    """Reassembles streamed tool_call fragments keyed by stream index."""

    def __init__(self) -> None:
        self.by_index: dict[int, dict[str, Any]] = {}
        self.order: list[int] = []

    def add(self, index: int, call_id: str | None, name: str | None, args_fragment: str | None):
        if index not in self.by_index:
            self.by_index[index] = {"id": call_id or "", "name": name or "", "args": ""}
            self.order.append(index)
        slot = self.by_index[index]
        if call_id:
            slot["id"] = call_id
        if name:
            slot["name"] = name
        if args_fragment:
            slot["args"] += args_fragment

    def finalize(self, strict: bool) -> list[ToolUseBlock]:
        blocks: list[ToolUseBlock] = []
        for i, index in enumerate(self.order):
            slot = self.by_index[index]
            parsed, ok = repair_json_args(slot["args"])
            if not ok and strict:
                parsed = {}
            call_id = slot["id"] or f"call_{index}_{i}"
            blocks.append(ToolUseBlock(id=call_id, name=slot["name"], input=parsed))
        return blocks


class OpenAICompatProvider(Provider):
    # Subclasses may override to use 'max_completion_tokens' (newer OpenAI API).
    _tokens_param = "max_tokens"

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        context_window: int | None = None,
        max_output_tokens: int = 8192,
    ):
        import warnings

        import httpx
        from openai import AsyncOpenAI

        resolved = context_window or resolve_openai_context_window(model)
        if resolved is None:
            raise ValueError(
                f"context_window is required for model {model!r}.\n"
                "Local and custom models don't advertise their context size, so you "
                "must pass it explicitly:\n\n"
                "    OpenAICompatProvider(\n"
                "        model=..., base_url=..., api_key=...,\n"
                "        context_window=32_768,  # match your model's actual context size\n"
                "    )\n\n"
                "Common values: 4_096, 8_192, 16_384, 32_768, 65_536, 131_072."
            )
        if context_window is None:
            # Auto-resolved from known frontier model table — inform the user.
            warnings.warn(
                f"context_window not provided for {model!r}; "
                f"using known value of {resolved:,} tokens.",
                stacklevel=2,
            )

        # Local models can take several minutes to generate large tool-call arguments
        # (e.g. writing a whole file).  300 s still catches genuine hangs (OOM,
        # deadlock, server crash) without timing out mid-generation.
        self._client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key or "not-needed",
            timeout=httpx.Timeout(connect=10.0, read=300.0, write=10.0, pool=5.0),
        )
        self.model = model
        self.capabilities = openai_capabilities(model, resolved, max_output_tokens)
        self.strict_tools = False

    async def stream(
        self,
        messages: list[Message],
        system: str | None,
        tools: list[ToolSchema],
        config: GenerationConfig,
    ) -> AsyncIterator[StreamEvent]:  # type: ignore[override]
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": to_openai_messages(
                messages, system, supports_vision=self.capabilities.supports_vision
            ),
            self._tokens_param: config.max_tokens,
            "temperature": config.temperature,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if tools:
            kwargs["tools"] = to_openai_tools(tools)
            kwargs["tool_choice"] = "auto"
        if config.stop_sequences:
            kwargs["stop"] = config.stop_sequences

        async def open_stream():
            return await self._client.chat.completions.create(**kwargs)

        text_buf: list[str] = []
        thinking_buf: list[str] = []
        accumulator = _ToolCallAccumulator()
        started: set[int] = set()
        finish_reason: str | None = None
        usage = Usage()

        stream = await with_retry(open_stream)
        async for chunk in stream:
            choices = getattr(chunk, "choices", None) or []
            if choices:
                choice = choices[0]
                delta = getattr(choice, "delta", None)
                if delta is not None:
                    # Reasoning ("thinking") models served over the OpenAI API
                    # (DeepSeek-R1, QwQ, gemma-qat, …) stream their chain-of-thought
                    # in a separate `reasoning_content` field. Surface it as a
                    # ThinkingDelta so the spinner shows live progress instead of a
                    # frozen "Thinking…" while the model reasons for thousands of
                    # tokens we'd otherwise drop on the floor.
                    delta_reasoning = (
                        getattr(delta, "reasoning_content", None)
                        or getattr(delta, "reasoning", None)
                    )
                    if delta_reasoning:
                        thinking_buf.append(delta_reasoning)
                        yield ThinkingDelta(text=delta_reasoning)
                    delta_text = getattr(delta, "content", None)
                    if delta_text:
                        text_buf.append(delta_text)
                        yield TextDelta(text=delta_text)
                    for tc in getattr(delta, "tool_calls", None) or []:
                        idx = getattr(tc, "index", 0) or 0
                        fn = getattr(tc, "function", None)
                        name = getattr(fn, "name", None) if fn else None
                        args = getattr(fn, "arguments", None) if fn else None
                        call_id = getattr(tc, "id", None)
                        accumulator.add(idx, call_id, name, args)
                        if idx not in started and (call_id or name):
                            started.add(idx)
                            yield ToolUseStart(
                                index=idx,
                                id=call_id or f"call_{idx}",
                                name=name or "",
                            )
                        elif args:
                            yield ToolUseArgsDelta(index=idx, partial_json=args)
                if getattr(choice, "finish_reason", None):
                    finish_reason = choice.finish_reason
            chunk_usage = getattr(chunk, "usage", None)
            if chunk_usage is not None:
                usage = Usage(
                    input_tokens=getattr(chunk_usage, "prompt_tokens", 0) or 0,
                    output_tokens=getattr(chunk_usage, "completion_tokens", 0) or 0,
                )

        tool_blocks = accumulator.finalize(self.strict_tools)
        thinking = "".join(thinking_buf)
        text = "".join(text_buf)
        # Recover tool calls the server left as raw <tool_call> markup in the text
        # or reasoning channel instead of the native `tool_calls` field — without
        # this the turn looks like an empty final answer and the agent loop stalls.
        if not tool_blocks:
            recovered_text, text = recover_inline_tool_calls(text)
            recovered_think, thinking = recover_inline_tool_calls(thinking)
            tool_blocks = recovered_text + recovered_think
        out_content: list[ContentBlock] = []
        if thinking:
            out_content.append(ThinkingBlock(thinking=thinking))
        if text:
            out_content.append(TextBlock(text=text))
        out_content.extend(tool_blocks)
        if usage.total() == 0:
            usage = Usage(
                input_tokens=estimate_messages_tokens(messages, system, tools),
                output_tokens=estimate_messages_tokens(
                    [Message(role="assistant", content=out_content)] if out_content else []
                ),
            )
        stop_reason = _FINISH_REASON_MAP.get(finish_reason or "stop", "end_turn")
        if tool_blocks and stop_reason == "end_turn":
            stop_reason = "tool_use"
        yield Done(
            response=CanonicalResponse(
                content=out_content, stop_reason=stop_reason, usage=usage, model=self.model
            )
        )

    async def count_tokens(
        self,
        messages: list[Message],
        system: str | None = None,
        tools: list[ToolSchema] | None = None,
    ) -> int:
        # No count endpoint on local servers — local estimate.
        return estimate_messages_tokens(messages, system, tools or [])


class OpenAIProvider(OpenAICompatProvider):
    """Official OpenAI API (api.openai.com).

    Requires only a model ID and an API key — no base URL, no context window
    configuration.  Uses ``max_completion_tokens`` (the current OpenAI requirement)
    instead of the deprecated ``max_tokens``.
    """

    _tokens_param = "max_completion_tokens"

    def __init__(self, api_key: str, model: str, max_output_tokens: int = 16384):
        import httpx
        from openai import AsyncOpenAI

        self._client = AsyncOpenAI(
            api_key=api_key,
            timeout=httpx.Timeout(connect=10.0, read=300.0, write=10.0, pool=5.0),
        )
        self.model = model
        self.capabilities = openai_native_capabilities(model, max_output_tokens)
        self.strict_tools = False
