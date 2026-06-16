from __future__ import annotations

import json
import logging
import re
from typing import Any, AsyncIterator, Generator, List, Optional

import httpx

from .base import BaseChatModel
from neurosurfer.config import config
from neurosurfer.server.schemas import ChatCompletionResponse, ChatCompletionChunk


class OpenAIModel(BaseChatModel):
    """
    OpenAI-compatible HTTP client that talks to:
      - vLLM OpenAI server
      - OpenAI
      - any OpenAI-compatible proxy

    Implements both sync + async for convenience.
    """

    def __init__(
        self,
        model_name: str,
        base_url: str,
        api_key: str,
        timeout: float = 120.0,
        stop_words: Optional[List[str]] = None,
        strip_reasoning: bool = False,
        max_seq_length: int = config.base_model.max_seq_length,
        enable_thinking: bool = config.base_model.enable_thinking,
        verbose: bool = config.base_model.verbose,
        logger: Optional[logging.Logger] = None,
        **kwargs: Any,
    ):
        super().__init__(
            max_seq_length=max_seq_length,
            enable_thinking=enable_thinking,
            stop_words=stop_words or config.base_model.stop_words,
            verbose=verbose,
            logger=logger,
            **kwargs,
        )
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.strip_reasoning = strip_reasoning

        self._client = httpx.Client(timeout=httpx.Timeout(timeout, read=None))
        self._aclient = httpx.AsyncClient(timeout=httpx.Timeout(timeout, read=None))

        # NOTE: regex stripping is not perfect across chunk boundaries,
        # but good enough as a baseline. For perfect behavior use hooks in server.
        self._think_re = re.compile(r"<think>.*?</think>", re.DOTALL)

    def init_model(self) -> None:
        # HTTP client model: nothing to load
        return

    def _headers(self) -> dict:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def _strip(self, s: str) -> str:
        if not self.strip_reasoning or not isinstance(s, str):
            return s
        return self._think_re.sub("", s).strip()

    def _payload(
        self,
        *,
        user_prompt: str,
        system_prompt: str,
        chat_history: List[dict],
        temperature: float,
        max_new_tokens: int,
        stream: bool,
        **kwargs: Any,
    ) -> dict:
        messages = self._build_messages(system_prompt, chat_history, user_prompt)

        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_new_tokens,
            "stream": stream,
        }

        # Pass stop words upstream if not overridden (OpenAI API supports "stop")
        if self.stop_words and "stop" not in kwargs:
            payload["stop"] = self.stop_words

        # Allow any OpenAI-compatible extra args (top_p, presence_penalty, tools, tool_choice, etc.)
        payload.update(kwargs or {})
        return payload

    # -------------------------
    # sync implementations
    # -------------------------
    def _call(
        self,
        user_prompt: str,
        system_prompt: str,
        chat_history: List[dict],
        temperature: float,
        max_new_tokens: int,
        **kwargs: Any,
    ) -> ChatCompletionResponse:
        if self.should_stop():
            # immediate stop: return empty assistant message
            return ChatCompletionResponse(
                id=self.call_id or "chatcmpl-stopped",
                created=int(__import__("time").time()),
                model=self.model_name,
                choices=[{"index": 0, "message": {"role": "assistant", "content": ""}, "finish_reason": "stop"}],
                usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            )

        payload = self._payload(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            chat_history=chat_history,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            stream=False,
            **kwargs,
        )

        r = self._client.post(f"{self.base_url}/chat/completions", headers=self._headers(), json=payload)
        r.raise_for_status()
        data = r.json()

        if self.strip_reasoning:
            try:
                for ch in data.get("choices", []):
                    msg = ch.get("message") or {}
                    if msg.get("role") == "assistant" and isinstance(msg.get("content"), str):
                        msg["content"] = self._strip(msg["content"])
            except Exception:
                pass

        return ChatCompletionResponse(**data)

    def _stream(
        self,
        user_prompt: str,
        system_prompt: str,
        chat_history: List[dict],
        temperature: float,
        max_new_tokens: int,
        **kwargs: Any,
    ) -> Generator[ChatCompletionChunk, None, None]:
        payload = self._payload(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            chat_history=chat_history,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            stream=True,
            **kwargs,
        )

        with self._client.stream(
            "POST",
            f"{self.base_url}/chat/completions",
            headers=self._headers(),
            json=payload,
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if self.should_stop():
                    # close early: exiting context manager ends upstream connection
                    break
                if not line:
                    continue
                s = line.decode("utf-8", "ignore")
                if s.startswith(":") or not s.startswith("data:"):
                    continue

                data = s[len("data:") :].strip()
                if data == "[DONE]":
                    break

                try:
                    obj = json.loads(data)
                except Exception:
                    continue

                if self.strip_reasoning:
                    try:
                        for ch in obj.get("choices", []):
                            delta = ch.get("delta") or {}
                            if isinstance(delta.get("content"), str):
                                delta["content"] = self._strip(delta["content"])
                    except Exception:
                        pass

                yield ChatCompletionChunk(**obj)

    # -------------------------
    # async implementations (preferred)
    # -------------------------
    async def _acall(
        self,
        user_prompt: str,
        system_prompt: str,
        chat_history: List[dict],
        temperature: float,
        max_new_tokens: int,
        **kwargs: Any,
    ) -> ChatCompletionResponse:
        if self.should_stop():
            return ChatCompletionResponse(
                id=self.call_id or "chatcmpl-stopped",
                created=int(__import__("time").time()),
                model=self.model_name,
                choices=[{"index": 0, "message": {"role": "assistant", "content": ""}, "finish_reason": "stop"}],
                usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            )

        payload = self._payload(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            chat_history=chat_history,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            stream=False,
            **kwargs,
        )

        r = await self._aclient.post(f"{self.base_url}/chat/completions", headers=self._headers(), json=payload)
        r.raise_for_status()
        data = r.json()

        if self.strip_reasoning:
            try:
                for ch in data.get("choices", []):
                    msg = ch.get("message") or {}
                    if msg.get("role") == "assistant" and isinstance(msg.get("content"), str):
                        msg["content"] = self._strip(msg["content"])
            except Exception:
                pass

        return ChatCompletionResponse(**data)

    async def _astream(
        self,
        user_prompt: str,
        system_prompt: str,
        chat_history: List[dict],
        temperature: float,
        max_new_tokens: int,
        **kwargs: Any,
    ) -> AsyncIterator[ChatCompletionChunk]:
        payload = self._payload(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            chat_history=chat_history,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            stream=True,
            **kwargs,
        )

        async with self._aclient.stream(
            "POST",
            f"{self.base_url}/chat/completions",
            headers=self._headers(),
            json=payload,
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if self.should_stop():
                    break
                if not line:
                    continue
                if line.startswith(":") or not line.startswith("data:"):
                    continue

                data = line[len("data:") :].strip()
                if data == "[DONE]":
                    break

                try:
                    obj = json.loads(data)
                except Exception:
                    continue

                if self.strip_reasoning:
                    try:
                        for ch in obj.get("choices", []):
                            delta = ch.get("delta") or {}
                            if isinstance(delta.get("content"), str):
                                delta["content"] = self._strip(delta["content"])
                    except Exception:
                        pass

                yield ChatCompletionChunk(**obj)

    async def aclose(self) -> None:
        await self._aclient.aclose()
        self._client.close()
