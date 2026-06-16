from __future__ import annotations

import logging
import time
import uuid
import contextvars
from abc import ABC, abstractmethod
from dataclasses import dataclass
from threading import Lock
from typing import Any, AsyncIterator, Dict, Generator, List, Optional, Tuple, Union

import anyio

from neurosurfer.config import config
from neurosurfer.server.schemas import ChatCompletionResponse, ChatCompletionChunk

LLM_SYNC_RESPONSE = Union[ChatCompletionResponse, Generator[ChatCompletionChunk, None, None]]
LLM_ASYNC_RESPONSE = Union[ChatCompletionResponse, AsyncIterator[ChatCompletionChunk]]

# request-local state
_current_call_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("ns_call_id", default=None)
_current_stop_event: contextvars.ContextVar[Optional[anyio.Event]] = contextvars.ContextVar("ns_stop_event", default=None)


@dataclass(frozen=True)
class GenerationHandle:
    call_id: str
    stop_event: anyio.Event


class BaseChatModel(ABC):
    """
    Concurrency-safe base for all model backends.

    Subclasses can implement either:
      - sync:  _call() and _stream()
      - async: _acall() and _astream() (preferred)

    Base provides:
      - ask()  (sync)
      - aask() (async)
      - per-request stop: stop_generation(call_id) / stop_generation() (all)
      - message building helpers shared across backends
    """

    def __init__(
        self,
        max_seq_length: int = config.base_model.max_seq_length,
        enable_thinking: bool = config.base_model.enable_thinking,
        stop_words: Optional[List[str]] = None,
        verbose: bool = config.base_model.verbose,
        logger: Optional[logging.Logger] = None,
        **_: Any,
    ):
        self.max_seq_length = max_seq_length
        self.enable_thinking = enable_thinking
        self.stop_words = stop_words or []
        self.verbose = verbose
        self.logger = logger or logging.getLogger(__name__)

        # set by subclasses
        self.model_name: str = getattr(self, "model_name", "unknown-model")
        self.model = getattr(self, "model", None)
        self.tokenizer = getattr(self, "tokenizer", None)

        # in-flight calls
        self._active: Dict[str, GenerationHandle] = {}
        self._active_lock = Lock()

    # -------------------------
    # request-local state
    # -------------------------
    @property
    def call_id(self) -> Optional[str]:
        return _current_call_id.get()

    def should_stop(self) -> bool:
        ev = _current_stop_event.get()
        return bool(ev and ev.is_set())

    def _new_call_id(self) -> str:
        return f"chatcmpl-{uuid.uuid4().hex[:24]}"

    def _register_call(self) -> GenerationHandle:
        handle = GenerationHandle(call_id=self._new_call_id(), stop_event=anyio.Event())
        with self._active_lock:
            self._active[handle.call_id] = handle
        return handle

    def _unregister_call(self, call_id: str) -> None:
        with self._active_lock:
            self._active.pop(call_id, None)

    def stop_generation(self, call_id: Optional[str] = None) -> None:
        """
        Stop a single in-flight request (recommended) or all if call_id is None.
        """
        with self._active_lock:
            if call_id is None:
                for h in self._active.values():
                    h.stop_event.set()
                return
            h = self._active.get(call_id)
            if h:
                h.stop_event.set()

    # -------------------------
    # lifecycle
    # -------------------------
    @abstractmethod
    def init_model(self) -> None:
        raise NotImplementedError

    # -------------------------
    # Public APIs
    # -------------------------
    def ask(
        self,
        user_prompt: str,
        system_prompt: str = config.base_model.system_prompt,
        chat_history: Optional[List[dict]] = None,
        temperature: float = config.base_model.temperature,
        max_new_tokens: int = config.base_model.max_new_tokens,
        stream: bool = False,
        **kwargs: Any,
    ) -> LLM_SYNC_RESPONSE:
        """
        Sync entrypoint (safe for concurrency). If stream=True returns a sync generator.
        """
        chat_history = chat_history or []
        handle = self._register_call()

        tok_id = _current_call_id.set(handle.call_id)
        tok_stop = _current_stop_event.set(handle.stop_event)
        try:
            # Allow OpenAI-style param alias
            if "max_tokens" in kwargs and kwargs.get("max_tokens") is not None:
                max_new_tokens = int(kwargs.pop("max_tokens"))

            params = dict(
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                chat_history=chat_history,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                **kwargs,
            )
            return self._stream(**params) if stream else self._call(**params)
        finally:
            _current_call_id.reset(tok_id)
            _current_stop_event.reset(tok_stop)
            self._unregister_call(handle.call_id)

    async def aask(
        self,
        user_prompt: str,
        system_prompt: str = config.base_model.system_prompt,
        chat_history: Optional[List[dict]] = None,
        temperature: float = config.base_model.temperature,
        max_new_tokens: int = config.base_model.max_new_tokens,
        stream: bool = False,
        **kwargs: Any,
    ) -> LLM_ASYNC_RESPONSE:
        """
        Async entrypoint (safe for concurrency). If stream=True returns an async iterator.
        """
        chat_history = chat_history or []
        handle = self._register_call()

        tok_id = _current_call_id.set(handle.call_id)
        tok_stop = _current_stop_event.set(handle.stop_event)
        try:
            if "max_tokens" in kwargs and kwargs.get("max_tokens") is not None:
                max_new_tokens = int(kwargs.pop("max_tokens"))

            params = dict(
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                chat_history=chat_history,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                **kwargs,
            )

            if self._has_async_impl():
                return self._astream(**params) if stream else await self._acall(**params)

            # Wrap sync models (Transformers/Unsloth/Llama.cpp) in threads
            if not stream:
                return await anyio.to_thread.run_sync(lambda: self._call(**params))

            return self._bridge_sync_stream_to_async(lambda: self._stream(**params))
        finally:
            _current_call_id.reset(tok_id)
            _current_stop_event.reset(tok_stop)
            self._unregister_call(handle.call_id)

    def _has_async_impl(self) -> bool:
        return (
            self.__class__._acall is not BaseChatModel._acall
            or self.__class__._astream is not BaseChatModel._astream
        )

    # -------------------------
    # Subclass contracts
    # -------------------------
    # Sync (legacy)
    def _call(
        self,
        user_prompt: str,
        system_prompt: str,
        chat_history: List[dict],
        temperature: float,
        max_new_tokens: int,
        **kwargs: Any,
    ) -> ChatCompletionResponse:
        raise NotImplementedError("Implement _call() or async _acall().")

    def _stream(
        self,
        user_prompt: str,
        system_prompt: str,
        chat_history: List[dict],
        temperature: float,
        max_new_tokens: int,
        **kwargs: Any,
    ) -> Generator[ChatCompletionChunk, None, None]:
        raise NotImplementedError("Implement _stream() or async _astream().")

    # Async (preferred)
    async def _acall(
        self,
        user_prompt: str,
        system_prompt: str,
        chat_history: List[dict],
        temperature: float,
        max_new_tokens: int,
        **kwargs: Any,
    ) -> ChatCompletionResponse:
        raise NotImplementedError("Implement _acall() or sync _call().")

    async def _astream(
        self,
        user_prompt: str,
        system_prompt: str,
        chat_history: List[dict],
        temperature: float,
        max_new_tokens: int,
        **kwargs: Any,
    ) -> AsyncIterator[ChatCompletionChunk]:
        raise NotImplementedError("Implement _astream() or sync _stream().")

    # -------------------------
    # Common helpers for all backends
    # -------------------------
    def _build_messages(
        self,
        system_prompt: str,
        chat_history: List[Dict[str, Any]],
        user_prompt: str,
    ) -> List[Dict[str, Any]]:
        # Example: Qwen no-think toggle
        if "qwen" in (self.model_name or "").lower() and not self.enable_thinking:
            system_prompt = (system_prompt or "") + "/nothink"

        system_prompt, normalized = self._normalize_for_chat_template(
            system_prompt=system_prompt,
            chat_history=chat_history or [],
            user_prompt=user_prompt,
        )

        messages: List[Dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(normalized)
        return messages

    def _normalize_for_chat_template(
        self,
        *,
        system_prompt: Optional[str],
        chat_history: List[dict],
        user_prompt: str,
    ) -> Tuple[str, List[Dict[str, str]]]:
        system_prompt = system_prompt or ""

        # merge system messages from history into system_prompt
        extra_system_parts = [
            (m.get("content") or "")
            for m in chat_history
            if m.get("role") == "system" and m.get("content")
        ]
        if extra_system_parts:
            merged = "\n\n".join(extra_system_parts)
            system_prompt = (system_prompt + "\n\n" + merged).strip() if system_prompt else merged

        # map roles to user/assistant (tool/function -> assistant)
        filtered: List[Dict[str, str]] = []
        for m in chat_history:
            role = m.get("role")
            content = (m.get("content") or "").strip()
            if not content:
                continue
            if role in ("user", "assistant"):
                filtered.append({"role": role, "content": content})
            elif role in ("tool", "function"):
                filtered.append({"role": "assistant", "content": content})

        # enforce alternation by merging repeats
        normalized: List[Dict[str, str]] = []
        expected = "user"
        for m in filtered:
            role = m["role"]
            content = m["content"]
            if not normalized:
                if role != "user":
                    role = "user"
                normalized.append({"role": role, "content": content})
                expected = "assistant"
                continue
            if role == expected:
                normalized.append({"role": role, "content": content})
                expected = "user" if expected == "assistant" else "assistant"
            else:
                normalized[-1]["content"] += "\n\n" + content

        # append current user prompt
        user_prompt = (user_prompt or "").strip()
        if user_prompt:
            if normalized and normalized[-1]["role"] == "user":
                normalized[-1]["content"] += "\n\n" + user_prompt
            else:
                normalized.append({"role": "user", "content": user_prompt})

        return system_prompt, normalized

    def _find_first_stop(self, text: str, stops: List[str]) -> Tuple[Optional[str], Optional[int]]:
        if not stops:
            return None, None
        earliest = None
        which = None
        for s in stops:
            if not s:
                continue
            i = text.find(s)
            if i != -1 and (earliest is None or i < earliest):
                earliest = i
                which = s
        return (which, earliest) if which is not None else (None, None)

    # -------------------------
    # Stream bridging (sync gen -> async iterator)
    # -------------------------
    def _bridge_sync_stream_to_async(
        self,
        make_gen,
        buffer_size: int = 64,
    ) -> AsyncIterator[ChatCompletionChunk]:
        send, recv = anyio.create_memory_object_stream(buffer_size)

        async def runner():
            try:
                gen = make_gen()
                for item in gen:
                    await send.send(item)
            finally:
                await send.aclose()

        async def aiter():
            async with anyio.create_task_group() as tg:
                tg.start_soon(runner)
                async with recv:
                    async for item in recv:
                        yield item

        return aiter()
