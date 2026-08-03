"""Workflow execution API (Phase 2).

REST + SSE surface over the Phase 1 engine:

    GET    /v1/workflows                     list registered workflows
    GET    /v1/workflows/{name}              full graph JSON (nodes/edges/control flow)
    POST   /v1/workflows/{name}/runs         start a run          {"inputs": {...}}
    GET    /v1/runs                          list runs (summaries)
    GET    /v1/runs/{id}                     run record (?events=true for the log)
    GET    /v1/runs/{id}/events              SSE live stream (replay + tail)
    GET    /v1/runs/{id}/nodes/{node_id}     one node's output/error/timing/usage/tools
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

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from ..streaming.sse import sse_data, sse_done, sse_ping
from ..workflow_runs.store import TERMINAL

# Statuses after which the event stream will receive nothing further.
_STREAM_END = TERMINAL | {"awaiting_input"}
_POLL_S = 0.1


def _manager(server, request: Request):
    """The RunManager for *this request's* user.

    Scoped per account: without it every signed-in user shared one registry, so
    one person's workflows and runs were visible to everyone.
    """
    from ..workspaces import run_manager_for

    try:
        return run_manager_for(server, getattr(request.state, "user", None))
    except Exception as e:  # noqa: BLE001 - no provider configured → clean 503
        raise HTTPException(
            status_code=503,
            detail=f"Workflow execution unavailable: no usable LLM provider ({e})",
        ) from e


def _get_run_or_404(server, request: Request, run_id: str):
    rec = _manager(server, request).get(run_id)
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


def _validate_graph(name: str, graph_dict: dict[str, Any], meta: dict[str, Any],
                    known_providers: set[str] | None = None):
    """Structural + semantic validation. Returns a JSON-able report dict (never
    raises for validation failures — structural errors become an error issue)."""
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
            "info": [],
        }, None, None
    report = validate_package(pkg, known_providers=known_providers)
    result = {
        "ok": report.ok,
        "errors": [_issue_to_dict(i) for i in report.errors],
        "gaps": [_issue_to_dict(i) for i in report.gaps],
        "warnings": [_issue_to_dict(i) for i in report.warnings],
        # Suggestions. Nothing is wrong and nothing blocks; the studio shows
        # them under their own tab so they never dilute a real warning.
        "info": [_issue_to_dict(i) for i in report.infos],
    }
    # Return tmp + pkg so a save path can reuse the staged package; caller cleans up.
    return result, tmp, pkg


def mount_workflow_routes(router: APIRouter, server) -> None:
    # ── workflows ───────────────────────────────────────────────────────────
    @router.get("/v1/workflows")
    async def list_workflows(request: Request):
        return {"workflows": _manager(server, request).list_workflows()}

    @router.get("/v1/workflows/{name}")
    async def get_workflow(name: str, request: Request):
        graph = _manager(server, request).get_workflow_graph(name)
        if graph is None:
            raise HTTPException(status_code=404, detail=f"Workflow '{name}' not found")
        return {**graph, "requirements": _requirements(server, request, name)}

    @router.get("/v1/workflows/{name}/requirements")
    async def workflow_requirements_route(name: str, request: Request):
        """Stored values this workflow needs, and whether each is already set.

        Derived rather than recorded: both halves — a server's `${VAR}`s and a
        node's `secrets:` — are authoritative elsewhere, and a copy in the package
        would go stale the first time a server is reconfigured.
        """
        if _manager(server, request).get_workflow_graph(name) is None:
            raise HTTPException(status_code=404, detail=f"Workflow '{name}' not found")
        return {"requirements": _requirements(server, request, name)}

    def _requirements(server_, request_: Request, name: str) -> list[dict[str, Any]]:
        from neurosurfer.graph.workflow.requirements import workflow_requirements
        from neurosurfer.mcp.credentials import use_credentials

        from ..workspaces import known_credentials_for

        try:
            pkg = _manager(server_, request_).registry.get(name)
            held = known_credentials_for(server_, getattr(request_.state, "user", None))
            # Bound rather than passed: satisfaction then goes through the same
            # lookup a connection uses, so the gateway's own environment counts
            # here exactly as much as it will at connect time.
            with use_credentials(held):
                return [r.to_dict() for r in workflow_requirements(pkg)]
        except Exception:  # noqa: BLE001 - a listing must not fail over this
            return []

    @router.post("/v1/workflows/{name}/runs", status_code=202)
    async def start_run(name: str, request: Request, body: dict[str, Any] | None = None):
        from neurosurfer.graph.workflow.registry import WorkflowNotFoundError

        inputs = (body or {}).get("inputs") or {}
        if not isinstance(inputs, dict):
            raise HTTPException(status_code=422, detail="'inputs' must be an object")
        from ..workflow_runs.manager import MissingSecrets

        try:
            rec = _manager(server, request).start(name, inputs)
        except WorkflowNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e
        except MissingSecrets as e:
            # 428 Precondition Required: the request is well-formed and the
            # workflow exists — something has to be set first. Same status the MCP
            # install uses for "a human must do something before this can work".
            raise HTTPException(status_code=428, detail=str(e)) from e
        return rec.to_dict(include_events=False)

    # ── authoring (S4): validate / create / update / delete ───────────────────
    def _provider_names(request: Request) -> set[str]:
        """Profiles configured for this account, so a node's `provider` is checked.

        Empty means "nothing configured", which the validator reads as "don't judge"
        — a package validated on a machine that hasn't been set up must not be
        rejected for naming providers that exist elsewhere.
        """
        from ..workspaces import settings_store_for, workspace_key

        try:
            owner = workspace_key(getattr(request.state, "user", None))
            return {r.name for r in settings_store_for(server).list_providers(owner)}
        except Exception:  # noqa: BLE001 - validation must not fail on a settings issue
            return set()

    @router.post("/v1/workflows/validate")
    async def validate_workflow(request: Request, body: dict[str, Any] | None = None):
        body = body or {}
        graph = body.get("graph")
        if not isinstance(graph, dict):
            raise HTTPException(status_code=422, detail="'graph' object is required")
        name = body.get("name") or graph.get("name") or "draft"
        import shutil

        known = _provider_names(request) or None
        result, tmp, _ = _validate_graph(name, graph, body, known)
        if tmp is not None:
            shutil.rmtree(tmp, ignore_errors=True)
        return result

    def _save(name: str, body: dict[str, Any], request: Request, *, must_exist: bool, must_not_exist: bool):
        import shutil

        graph = body.get("graph")
        if not isinstance(graph, dict):
            raise HTTPException(status_code=422, detail="'graph' object is required")
        mgr = _manager(server, request)
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

        result, tmp, pkg = _validate_graph(name, graph, meta, _provider_names(request) or None)
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
    async def create_workflow(request: Request, body: dict[str, Any] | None = None):
        body = body or {}
        name = body.get("name") or (body.get("graph") or {}).get("name")
        if not name:
            raise HTTPException(status_code=422, detail="'name' is required")
        return _save(name, body, request, must_exist=False, must_not_exist=True)

    @router.put("/v1/workflows/{name}")
    async def update_workflow(name: str, request: Request, body: dict[str, Any] | None = None):
        return _save(name, body or {}, request, must_exist=True, must_not_exist=False)

    @router.delete("/v1/workflows/{name}")
    async def delete_workflow(name: str, request: Request):
        from neurosurfer.graph.workflow.registry import WorkflowNotFoundError

        mgr = _manager(server, request)
        try:
            mgr.registry.delete(name)
        except WorkflowNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e
        return {"deleted": name}

    # ── runs ────────────────────────────────────────────────────────────────
    @router.get("/v1/runs")
    async def list_runs(
        request: Request,
        workflow: str | None = None,
        status: str | None = None,
        limit: int = 200,
    ):
        """Run summaries, newest first — the runs list / dashboard feed.

        Summaries omit per-node outputs and final results (see
        ``RunRecord.summary``); fetch a single run for those.
        """
        runs = _manager(server, request).list_runs()
        if workflow:
            runs = [r for r in runs if r.workflow == workflow]
        if status:
            wanted = {s.strip() for s in status.split(",") if s.strip()}
            runs = [r for r in runs if r.status in wanted]
        return {"runs": [r.summary() for r in runs[: max(1, limit)]], "total": len(runs)}

    @router.get("/v1/runs/{run_id}")
    async def get_run(run_id: str, request: Request, events: bool = False):
        rec = _get_run_or_404(server, request, run_id)
        return rec.to_dict(include_events=events)

    @router.get("/v1/runs/{run_id}/nodes/{node_id}")
    async def get_run_node(run_id: str, node_id: str, request: Request):
        rec = _get_run_or_404(server, request, run_id)
        node = rec.nodes.get(node_id)
        if node is None:
            raise HTTPException(
                status_code=404, detail=f"Node '{node_id}' has no record in run '{run_id}'"
            )
        return {"run_id": run_id, "node_id": node_id, **node}

    @router.get("/v1/runs/{run_id}/trace")
    async def get_run_trace(run_id: str, request: Request, node_id: str | None = None):
        """The run's traced steps — one per model/tool call, grouped by node.

        Read from the persisted `<run>.trace.json` rather than kept in memory, so
        a restored run from a previous process is just as inspectable.
        """
        import json as _json
        from pathlib import Path as _Path

        rec = _get_run_or_404(server, request, run_id)
        if not rec.trace_path:
            return {"run_id": run_id, "steps": [], "by_node": {}, "available": False}
        path = _Path(rec.trace_path)
        if not path.exists():
            return {"run_id": run_id, "steps": [], "by_node": {}, "available": False}
        try:
            data = _json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 - a broken trace must not 500 the run view
            return {"run_id": run_id, "steps": [], "by_node": {}, "available": False}

        steps = data.get("steps") or []
        if node_id:
            steps = [s for s in steps if s.get("node_id") == node_id]
        # Index by node so the UI can render a per-node tree without regrouping;
        # steps with no node id (should be none) fall under "".
        by_node: dict[str, list[int]] = {}
        for step in steps:
            by_node.setdefault(step.get("node_id") or "", []).append(step.get("step_id"))
        return {
            "run_id": run_id,
            "available": True,
            "meta": data.get("meta") or {},
            "steps": steps,
            "by_node": by_node,
        }

    @router.post("/v1/runs/{run_id}/resume", status_code=202)
    async def resume_run(run_id: str, request: Request, body: dict[str, Any] | None = None):
        values = (body or {}).get("values") or {}
        if not isinstance(values, dict):
            raise HTTPException(status_code=422, detail="'values' must be an object")
        _get_run_or_404(server, request, run_id)  # 404 before starting anything
        rec = _manager(server, request).resume(run_id, values)
        return rec.to_dict(include_events=False)

    @router.delete("/v1/runs/{run_id}")
    async def cancel_run(run_id: str, request: Request):
        _get_run_or_404(server, request, run_id)
        rec = _manager(server, request).cancel(run_id)
        return rec.to_dict(include_events=False)

    # ── live stream ─────────────────────────────────────────────────────────
    @router.get("/v1/runs/{run_id}/events")
    async def stream_run_events(run_id: str, request: Request):
        rec = _get_run_or_404(server, request, run_id)
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
