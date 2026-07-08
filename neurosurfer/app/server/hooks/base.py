from __future__ import annotations

from dataclasses import dataclass


@dataclass
class HookContext:
    request_id: str
    model: str
    user: str | None = None
    client_ip: str | None = None


class Hook:
    """Base hook — override any method to intercept the request/response lifecycle."""

    async def before_chat(self, ctx: HookContext, req: dict) -> dict:
        return req

    async def after_chat(self, ctx: HookContext, resp: dict) -> dict:
        return resp

    async def stream_chunk(self, ctx: HookContext, chunk: dict) -> dict:
        return chunk
