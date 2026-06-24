from __future__ import annotations

import time
import uuid
from typing import AsyncIterator

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from ..errors import OpenAIHTTPError
from ..hooks.base import HookContext
from ..streaming.sse import sse_data, sse_done, sse_ping


def mount_chat_routes(router: APIRouter, server) -> None:
    @router.post("/v1/chat/completions")
    async def chat_completions(req: Request):
        body = await req.json()
        model = body.get("model")
        if not model:
            raise OpenAIHTTPError(400, "Missing 'model'")

        request_id = f"req-{uuid.uuid4().hex[:24]}"
        ctx = HookContext(
            request_id=request_id,
            model=model,
            user=body.get("user"),
            client_ip=req.client.host if req.client else None,
        )

        for hk in server.hooks:
            body = await hk.before_chat(ctx, body)

        try:
            target = server.router.resolve(model)
        except KeyError:
            raise OpenAIHTTPError(404, f"Model not found: {model!r}")

        if target.upstream_model is not None and server._upstream_backend is not None:
            body["model"] = target.upstream_model

        is_stream, result = await target.backend.chat_completions(body, request_id=request_id)

        if not is_stream:
            resp = result
            for hk in server.hooks:
                resp = await hk.after_chat(ctx, resp)
            return JSONResponse(resp)

        async def event_gen() -> AsyncIterator[bytes]:
            last_ping = time.time()
            async for chunk in result:
                now = time.time()
                if now - last_ping >= server.settings.sse_ping_interval_s:
                    yield sse_ping()
                    last_ping = now

                for hk in server.hooks:
                    chunk = await hk.stream_chunk(ctx, chunk)

                yield sse_data(chunk)

                if await req.is_disconnected():
                    break

            yield sse_done()

        headers = {
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
        return StreamingResponse(event_gen(), media_type="text/event-stream", headers=headers)
