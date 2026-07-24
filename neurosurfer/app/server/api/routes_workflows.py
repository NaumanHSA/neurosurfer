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


# ── authoring helpers (validate / create / update) ───────────────────────────

def _issue_to_dict(issue) -> dict[str, Any]:
    return {
        "kind": issue.kind,
        "message": issue.message,
        "node_id": issue.node_id,
        "suggestion": issue.suggestion,
        "subject": issue.subject,
    }


def _stage_package(name: str, graph_dict: dict[str, Any], meta: dict[str, Any]):
    """Write a staged package (workflow.yaml + graph.yaml) to a temp dir and load
    it — this runs the engine's structural validation. Returns (tmp_dir, package).

    The caller MUST remove tmp_dir. Raises ValueError with a readable message on
    a structural/schema problem so the route can turn it into a 422.
    """
    import tempfile
    from pathlib import Path

    import yaml

    from neurosurfer.graph.workflow.package import load_package

    tmp = Path(tempfile.mkdtemp(prefix="ns-studio-"))
    pkg_dir = tmp / name
    pkg_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "name": name,
        "version": str(meta.get("version") or "1.0.0"),
        "description": meta.get("description") or graph_dict.get("description") or "",
        "entrypoint": "graph.yaml",
    }
    if meta.get("tags"):
        manifest["tags"] = list(meta["tags"])
    if meta.get("created_by"):
        manifest["created_by"] = meta["created_by"]

    graph_out = dict(graph_dict)
    graph_out.setdefault("name", name)

    (pkg_dir / "workflow.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False))
    (pkg_dir / "graph.yaml").write_text(yaml.safe_dump(graph_out, sort_keys=False))

    try:
        pkg = load_package(pkg_dir)
    except Exception as e:  # noqa: BLE001 - surface as a clean 422
        import shutil

        shutil.rmtree(tmp, ignore_errors=True)
        raise ValueError(str(e)) from e
    return tmp, pkg


def _validate_graph(name: str, graph_dict: dict[str, Any], meta: dict[str, Any]):
    """Structural + semantic validation. Returns a JSON-able report dict (never
    raises for validation failures — structural errors become an error issue)."""
    import shutil

    from neurosurfer.graph.workflow.validate import validate_package

    try:
        tmp, pkg = _stage_package(name, graph_dict, meta)
    except ValueError as e:
        return {
            "ok": False,
            "errors": [{"kind": "structure", "message": str(e), "node_id": None,
                        "suggestion": None, "subject": None}],
            "gaps": [],
            "warnings": [],
        }, None, None
    report = validate_package(pkg)
    result = {
        "ok": report.ok,
        "errors": [_issue_to_dict(i) for i in report.errors],
        "gaps": [_issue_to_dict(i) for i in report.gaps],
        "warnings": [_issue_to_dict(i) for i in report.warnings],
    }
    # Return tmp + pkg so a save path can reuse the staged package; caller cleans up.
    return result, tmp, pkg


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

    # ── authoring (S4): validate / create / update / delete ───────────────────
    @router.post("/v1/workflows/validate")
    async def validate_workflow(body: dict[str, Any] | None = None):
        body = body or {}
        graph = body.get("graph")
        if not isinstance(graph, dict):
            raise HTTPException(status_code=422, detail="'graph' object is required")
        name = body.get("name") or graph.get("name") or "draft"
        import shutil

        result, tmp, _ = _validate_graph(name, graph, body)
        if tmp is not None:
            shutil.rmtree(tmp, ignore_errors=True)
        return result

    def _save(name: str, body: dict[str, Any], *, must_exist: bool, must_not_exist: bool):
        import shutil

        graph = body.get("graph")
        if not isinstance(graph, dict):
            raise HTTPException(status_code=422, detail="'graph' object is required")
        mgr = _manager(server)
        exists = mgr.registry.exists(name)
        if must_exist and not exists:
            raise HTTPException(status_code=404, detail=f"Workflow '{name}' not found")
        if must_not_exist and exists:
            raise HTTPException(status_code=409, detail=f"Workflow '{name}' already exists")

        # Preserve created_by/at on update if present.
        meta = dict(body)
        if exists:
            try:
                prev = mgr.registry.get(name)
                meta.setdefault("created_by", prev.manifest.created_by)
            except Exception:  # noqa: BLE001 - best-effort metadata carry-over
                pass

        result, tmp, pkg = _validate_graph(name, graph, meta)
        try:
            if not result["ok"]:
                raise HTTPException(
                    status_code=422,
                    detail={"message": "Workflow is invalid", "validation": result},
                )
            mgr.registry.save(pkg)
        finally:
            if tmp is not None:
                shutil.rmtree(tmp, ignore_errors=True)
        graph_json = mgr.get_workflow_graph(name)
        return {**graph_json, "validation": result}

    @router.post("/v1/workflows", status_code=201)
    async def create_workflow(body: dict[str, Any] | None = None):
        body = body or {}
        name = body.get("name") or (body.get("graph") or {}).get("name")
        if not name:
            raise HTTPException(status_code=422, detail="'name' is required")
        return _save(name, body, must_exist=False, must_not_exist=True)

    @router.put("/v1/workflows/{name}")
    async def update_workflow(name: str, body: dict[str, Any] | None = None):
        return _save(name, body or {}, must_exist=True, must_not_exist=False)

    @router.delete("/v1/workflows/{name}")
    async def delete_workflow(name: str):
        from neurosurfer.graph.workflow.registry import WorkflowNotFoundError

        mgr = _manager(server)
        try:
            mgr.registry.delete(name)
        except WorkflowNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e
        return {"deleted": name}

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
