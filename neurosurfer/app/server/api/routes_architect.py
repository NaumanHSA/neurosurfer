"""Architect build API (S5) — start a build, stream its steps, read the record.

    POST /v1/architect/builds              start a build   {"intent": "...", "verify": "..."}
    GET  /v1/architect/builds              list builds (summaries)
    GET  /v1/architect/builds/{id}         build record (?events=true for the log)
    GET  /v1/architect/builds/{id}/events  SSE live stream (replay + tail, ends [DONE])

The ArchitectManager reuses the RunManager's provider + registry so builds
register into the same store the studio reads from.
"""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from ..architect_builds.store import BUILD_TERMINAL
from ..streaming.sse import sse_data, sse_done, sse_ping

_POLL_S = 0.1


def _arch_manager(server):
    mgr = getattr(server, "architect_manager", None)
    if mgr is not None:
        return mgr
    try:
        from ..architect_builds.manager import ArchitectManager

        # Reuse the run manager's provider + registry when available.
        run_mgr = getattr(server, "run_manager", None)
        if run_mgr is not None:
            mgr = ArchitectManager(run_mgr.provider, registry=run_mgr.registry)
        else:
            from neurosurfer.config import Config
            from neurosurfer.llm.registry import resolve_provider

            mgr = ArchitectManager(resolve_provider(Config()))
        server.architect_manager = mgr
        return mgr
    except Exception as e:  # noqa: BLE001 - no provider → clean 503
        raise HTTPException(
            status_code=503,
            detail=f"Architect unavailable: no usable LLM provider ({e})",
        ) from e


def mount_architect_routes(router: APIRouter, server) -> None:
    @router.post("/v1/architect/builds", status_code=202)
    async def start_build(body: dict[str, Any] | None = None):
        body = body or {}
        intent = (body.get("intent") or "").strip()
        if not intent:
            raise HTTPException(status_code=422, detail="'intent' is required")
        verify = body.get("verify")
        rec = _arch_manager(server).start(intent, verify=verify)
        return rec.to_dict(include_events=False)

    @router.get("/v1/architect/builds")
    async def list_builds():
        return {
            "builds": [r.to_dict(include_events=False) for r in _arch_manager(server).list()]
        }

    @router.get("/v1/architect/builds/{build_id}")
    async def get_build(build_id: str, events: bool = False):
        rec = _arch_manager(server).get(build_id)
        if rec is None:
            raise HTTPException(status_code=404, detail=f"Build '{build_id}' not found")
        return rec.to_dict(include_events=events)

    @router.get("/v1/architect/builds/{build_id}/events")
    async def stream_build_events(build_id: str):
        rec = _arch_manager(server).get(build_id)
        if rec is None:
            raise HTTPException(status_code=404, detail=f"Build '{build_id}' not found")
        ping_every = float(getattr(server.settings, "sse_ping_interval_s", 15.0) or 15.0)

        async def gen():
            idx = 0
            since_ping = 0.0
            while True:
                while idx < len(rec.events):
                    yield sse_data(rec.events[idx])
                    idx += 1
                    since_ping = 0.0
                if rec.status in BUILD_TERMINAL:
                    while idx < len(rec.events):
                        yield sse_data(rec.events[idx])
                        idx += 1
                    yield sse_done()
                    return
                await asyncio.sleep(_POLL_S)
                since_ping += _POLL_S
                if since_ping >= ping_every:
                    yield sse_ping()
                    since_ping = 0.0

        return StreamingResponse(gen(), media_type="text/event-stream")
