from __future__ import annotations

import asyncio
import inspect
import json
import time
import uuid
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import Any

import anyio

from ..errors import OpenAIHTTPError
from ..schemas.openai import (
    ChatCompletionChoice,
    ChatCompletionResponse,
    ChatMessage,
    CompletionUsage,
)
from ..streaming.openai_chunks import chunk_end, chunk_role, chunk_text, chunk_usage
from .base import Backend


def _obj_usage(obj: Any) -> tuple[int, int] | None:
    """(input_tokens, output_tokens) from anything carrying a neurosurfer ``Usage``
    (an agent or a RunResult), or ``None`` if it exposes no usage."""
    u = getattr(obj, "usage", None)
    if u is None:
        return None
    it, ot = getattr(u, "input_tokens", None), getattr(u, "output_tokens", None)
    if it is None and ot is None:
        return None
    return int(it or 0), int(ot or 0)


def _completion_usage(input_tokens: int, output_tokens: int) -> CompletionUsage:
    it, ot = max(0, int(input_tokens)), max(0, int(output_tokens))
    return CompletionUsage(prompt_tokens=it, completion_tokens=ot, total_tokens=it + ot)


def _default_result_to_text(result: Any) -> str:
    if result is None:
        return ""
    if hasattr(result, "model_dump"):
        try:
            d = result.model_dump()
            if isinstance(d, dict):
                if "final" in d and isinstance(d["final"], dict):
                    final = d["final"]
                    if len(final) == 1:
                        return str(next(iter(final.values())))
                    return json.dumps(final, ensure_ascii=False, indent=2, default=str)
                return json.dumps(d, ensure_ascii=False, indent=2, default=str)
        except Exception:
            pass
    if isinstance(result, dict):
        for k in ("final_answer", "answer", "output", "result", "text"):
            if k in result:
                return str(result[k])
        if len(result) == 1:
            return str(next(iter(result.values())))
        return json.dumps(result, ensure_ascii=False, indent=2, default=str)
    return str(result)


@dataclass
class AgentSpec:
    agent: Any
    model_id: str
    description: str = "Neurosurfer agent"
    owned_by: str = "neurosurfer"
    max_model_len: int = 8192
    # Custom invocation: run_fn(agent, user_query, chat_history) → text | Awaitable[text]
    run_fn: Callable[[Any, str, list], Any] | None = None
    result_to_text: Callable[[Any], str] = _default_result_to_text


class AgentBackend(Backend):
    """Backend that drives a native agent (AgenticLoop / ReactAgent / Agent / custom).

    Detection order for the agent's ``run`` method:
    1. ``run_fn`` provided → call it (sync or async).
    2. Async generator → stream TextDelta events into OpenAI chunks.
    3. Async coroutine → await, convert result to text.
    4. Sync callable → run in a thread via anyio, convert result to text.
    """

    def __init__(self, spec: AgentSpec):
        self.spec = spec

    @property
    def name(self) -> str:
        return self.spec.model_id

    async def list_models(self) -> dict:
        now = int(time.time())
        perm = [
            {
                "id": f"modelperm-{uuid.uuid4().hex[:16]}",
                "object": "model_permission",
                "created": now,
                "allow_sampling": True,
                "allow_logprobs": True,
                "allow_view": True,
                "is_blocking": False,
            }
        ]
        return {
            "object": "list",
            "data": [
                {
                    "id": self.spec.model_id,
                    "object": "model",
                    "created": now,
                    "owned_by": self.spec.owned_by,
                    "max_model_len": self.spec.max_model_len,
                    "permission": perm,
                }
            ],
        }

    async def _invoke_streaming(self, user_query: str, chat_history: list) -> AsyncIterator[str]:
        """Run an async-generator agent and yield text fragments."""
        async for ev in self.spec.agent.run(user_query):
            text = getattr(ev, "text", None)
            if text:
                yield text

    async def _invoke_blocking(self, user_query: str, chat_history: list) -> str:
        """Run a sync agent in a thread and convert its result to text."""
        agent = self.spec.agent
        run = agent.run

        def _call():
            try:
                return run(inputs={"query": user_query})
            except TypeError:
                pass
            try:
                return run(user_query=user_query, chat_history=chat_history)
            except TypeError:
                pass
            return run(user_query)

        result = await anyio.to_thread.run_sync(_call)
        return self.spec.result_to_text(result)

    async def chat_completions(self, req: dict, *, request_id: str) -> tuple[bool, object]:
        model = req.get("model") or self.spec.model_id
        messages = req.get("messages") or []
        user_query = ""
        chat_history: list = []

        for m in messages:
            if not isinstance(m, dict):
                continue
            role = m.get("role")
            content = m.get("content")
            if role == "user":
                user_query = (
                    content if isinstance(content, str) else json.dumps(content, default=str)
                )
            if role in ("user", "assistant"):
                chat_history.append({"role": role, "content": content})

        if not user_query:
            raise OpenAIHTTPError(400, "No user message found")

        want_stream = bool(req.get("stream"))
        # OpenAI emits the usage chunk on a stream only when asked; match that.
        include_usage = bool((req.get("stream_options") or {}).get("include_usage"))
        created = int(time.time())
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        # Snapshot the agent's cumulative token counters so we can report *this*
        # request's usage as the delta (a long-lived agent accumulates across calls).
        usage_before = _obj_usage(self.spec.agent)

        # ── 1. Custom run_fn ──────────────────────────────────────────────────
        if self.spec.run_fn is not None:
            result = self.spec.run_fn(self.spec.agent, user_query, chat_history)
            if asyncio.iscoroutine(result) or inspect.isawaitable(result):
                result = await result
            text = self.spec.result_to_text(result)
            usage = self._run_usage(usage_before, result)
            return self._static_response(
                text, want_stream, model, completion_id, created,
                usage=usage, include_usage=include_usage,
            )

        run = getattr(self.spec.agent, "run", None)
        if run is None:
            raise RuntimeError("Agent has no run() method")

        # ── 2. Async generator — native streaming agent ───────────────────────
        if inspect.isasyncgenfunction(run):
            if want_stream:
                async def gen_stream() -> AsyncIterator[dict]:
                    before = _obj_usage(self.spec.agent)
                    yield chunk_role(id=completion_id, created=created, model=model)
                    async for text_frag in self._invoke_streaming(user_query, chat_history):
                        yield chunk_text(
                            id=completion_id, created=created, model=model, text=text_frag
                        )
                    yield chunk_end(id=completion_id, created=created, model=model)
                    if include_usage and (usage := self._run_usage(before, None)):
                        yield chunk_usage(
                            id=completion_id, created=created, model=model, usage=usage
                        )

                return True, gen_stream()

            # Non-streaming request: collect all text fragments
            parts: list[str] = []
            async for frag in self._invoke_streaming(user_query, chat_history):
                parts.append(frag)
            return self._static_response(
                "".join(parts), False, model, completion_id, created,
                usage=self._run_usage(usage_before, None), include_usage=include_usage,
            )

        # ── 3. Async coroutine ────────────────────────────────────────────────
        if asyncio.iscoroutinefunction(run):
            result = await run(user_query)
            text = self.spec.result_to_text(result)
            return self._static_response(
                text, want_stream, model, completion_id, created,
                usage=self._run_usage(usage_before, result), include_usage=include_usage,
            )

        # ── 4. Sync fallback ──────────────────────────────────────────────────
        text = await self._invoke_blocking(user_query, chat_history)
        return self._static_response(
            text, want_stream, model, completion_id, created,
            usage=self._run_usage(usage_before, None), include_usage=include_usage,
        )

    def _run_usage(
        self, before: tuple[int, int] | None, result: Any
    ) -> CompletionUsage | None:
        """This request's usage: a result's own ``usage`` if it has one, else the
        agent's cumulative-counter delta since ``before``. ``None`` when neither the
        result nor the agent reports tokens."""
        ru = _obj_usage(result) if result is not None else None
        if ru is not None:
            return _completion_usage(*ru)
        au = _obj_usage(self.spec.agent)
        if au is None:
            return None
        if before is not None:
            return _completion_usage(au[0] - before[0], au[1] - before[1])
        return _completion_usage(*au)

    def _static_response(
        self,
        text: str,
        want_stream: bool,
        model: str,
        completion_id: str,
        created: int,
        *,
        usage: CompletionUsage | None = None,
        include_usage: bool = False,
    ) -> tuple[bool, object]:
        if want_stream:
            async def gen() -> AsyncIterator[dict]:
                yield chunk_role(id=completion_id, created=created, model=model)
                if text:
                    yield chunk_text(id=completion_id, created=created, model=model, text=text)
                yield chunk_end(id=completion_id, created=created, model=model)
                if include_usage and usage is not None:
                    yield chunk_usage(
                        id=completion_id, created=created, model=model, usage=usage
                    )

            return True, gen()

        resp = ChatCompletionResponse(
            id=completion_id,
            created=created,
            model=model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=text),
                    finish_reason="stop",
                )
            ],
            usage=usage,
        ).model_dump()
        return False, resp
