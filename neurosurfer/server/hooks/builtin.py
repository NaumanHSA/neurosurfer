from __future__ import annotations

import re

from .base import Hook, HookContext


class StripReasoningHook(Hook):
    """Remove ``<think>...</think>`` blocks from assistant responses."""

    def __init__(self, pattern: str = r"<think>.*?</think>", flags: int = re.DOTALL):
        self._re = re.compile(pattern, flags)

    def _strip(self, text: str) -> str:
        return self._re.sub("", text).strip()

    async def after_chat(self, ctx: HookContext, resp: dict) -> dict:
        try:
            for ch in resp.get("choices") or []:
                msg = ch.get("message") or {}
                if msg.get("role") == "assistant" and isinstance(msg.get("content"), str):
                    msg["content"] = self._strip(msg["content"])
        except Exception:
            pass
        return resp

    async def stream_chunk(self, ctx: HookContext, chunk: dict) -> dict:
        try:
            for ch in chunk.get("choices") or []:
                delta = ch.get("delta") or {}
                if isinstance(delta.get("content"), str):
                    delta["content"] = self._strip(delta["content"])
        except Exception:
            pass
        return chunk


class SystemPromptInjectorHook(Hook):
    """Prepend a fixed system prompt if none is present in the request."""

    def __init__(self, system_prompt: str):
        self.system_prompt = system_prompt

    async def before_chat(self, ctx: HookContext, req: dict) -> dict:
        msgs = req.get("messages") or []
        if msgs and not (isinstance(msgs[0], dict) and msgs[0].get("role") == "system"):
            req["messages"] = [{"role": "system", "content": self.system_prompt}] + msgs
        return req
