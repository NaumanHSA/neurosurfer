"""Workflow execution API (Phase 2).

REST + SSE surface over the Phase 1 engine:

    GET    /v1/workflows                     list registered workflows
    GET    /v1/workflows/{name}              full graph JSON (nodes/edges/control flow)
    POST   /v1/workflows/{name}/runs         start a run          {"inputs": {...}}
    GET    /v1/runs                          list runs (summaries)
    GET    /v1/runs/{id}                     run record (?events=true for the log)
    GET    /v1/runs/{id}/events              SSE live stream (replay + tail)
    GET    /v1/runs/{id}/nodes/{node_id}     one node's inputs/output/error/timing
    POST   /v1/runs/{id}/resume              re-run with values   {"values": {...}}
    DELETE /v1/runs/{id}                     cancel (best-effort)

The SSE stream replays the full event log from seq 1 and then tails live events,
so late subscribers miss nothing; it closes with ``[DONE]`` once the run reaches a
terminal state (succeeded / failed / cancelled / awaiting_input).

A ``RunManager`` can be injected by setting ``server.run_manager`` (tests do this);
otherwise one is built lazily from the configured LLM provider on first use.
"""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from ..streaming.sse import sse_data, sse_done, sse_ping
from ..workflow_runs.store import TERMINAL

# Statuses after which the event stream will receive nothing further.
_STREAM_END = TERMINAL | {"awaiting_input"}
_POLL_S = 0.1


def _manager(server):
    """Return the server's RunManager, building one lazily if not injected."""
    mgr = getattr(server, "run_manager", None)
    if mgr is not None:
        return mgr
    try:
        from neurosurfer.config import Config
        from neurosurfer.llm.registry import resolve_provider

        from ..workflow_runs.manager import RunManager

        provider = resolve_provider(Config())
        server.run_manager = RunManager(provider)
        return server.run_manager
    except Exception as e:  # noqa: BLE001 - no provider configured → clean 503
        raise HTTPException(
            status_code=503,
            detail=f"Workflow execution unavailable: no usable LLM provider ({e})",
        ) from e


def _get_run_or_404(server, run_id: str):
    rec = _manager(server).get(run_id)
    if rec is None:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
    return rec


def mount_workflow_routes(router: APIRouter, server) -> None:
    # ── workflows ───────────────────────────────────────────────────────────
    @router.get("/v1/workflows")
    async def list_workflows():
        return {"workflows": _manager(server).list_workflows()}

    @router.get("/v1/workflows/{name}")
    async def get_workflow(name: str):
        graph = _manager(server).get_workflow_graph(name)
        if graph is None:
            raise HTTPException(status_code=404, detail=f"Workflow '{name}' not found")
        return graph

    @router.post("/v1/workflows/{name}/runs", status_code=202)
    async def start_run(name: str, body: dict[str, Any] | None = None):
        from neurosurfer.graph.workflow.registry import WorkflowNotFoundError

        inputs = (body or {}).get("inputs") or {}
        if not isinstance(inputs, dict):
            raise HTTPException(status_code=422, detail="'inputs' must be an object")
        try:
            rec = _manager(server).start(name, inputs)
        except WorkflowNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e
        return rec.to_dict(include_events=False)

    # ── runs ────────────────────────────────────────────────────────────────
    @router.get("/v1/runs")
    async def list_runs():
        return {
            "runs": [r.to_dict(include_events=False) for r in _manager(server).list_runs()]
        }

    @router.get("/v1/runs/{run_id}")
    async def get_run(run_id: str, events: bool = False):
        rec = _get_run_or_404(server, run_id)
        return rec.to_dict(include_events=events)

    @router.get("/v1/runs/{run_id}/nodes/{node_id}")
    async def get_run_node(run_id: str, node_id: str):
        rec = _get_run_or_404(server, run_id)
        node = rec.nodes.get(node_id)
        if node is None:
            raise HTTPException(
                status_code=404, detail=f"Node '{node_id}' has no record in run '{run_id}'"
            )
        return {"run_id": run_id, "node_id": node_id, **node}

    @router.post("/v1/runs/{run_id}/resume", status_code=202)
    async def resume_run(run_id: str, body: dict[str, Any] | None = None):
        values = (body or {}).get("values") or {}
        if not isinstance(values, dict):
            raise HTTPException(status_code=422, detail="'values' must be an object")
        _get_run_or_404(server, run_id)  # 404 before starting anything
        rec = _manager(server).resume(run_id, values)
        return rec.to_dict(include_events=False)

    @router.delete("/v1/runs/{run_id}")
    async def cancel_run(run_id: str):
        _get_run_or_404(server, run_id)
        rec = _manager(server).cancel(run_id)
        return rec.to_dict(include_events=False)

    # ── live stream ─────────────────────────────────────────────────────────
    @router.get("/v1/runs/{run_id}/events")
    async def stream_run_events(run_id: str):
        rec = _get_run_or_404(server, run_id)
        ping_every = float(getattr(server.settings, "sse_ping_interval_s", 15.0) or 15.0)

        async def gen():
            idx = 0
            since_ping = 0.0
            while True:
                # Replay everything appended since our cursor (list is append-only).
                while idx < len(rec.events):
                    yield sse_data(rec.events[idx])
                    idx += 1
                    since_ping = 0.0
                if rec.status in _STREAM_END:
                    # Flush any tail appended between the check and now, then close.
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
