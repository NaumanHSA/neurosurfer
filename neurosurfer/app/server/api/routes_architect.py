"""Architect build API (S5) — start a build, stream its steps, read the record.

    POST /v1/architect/plans               plan only, no build  {"intent": "..."}
    POST /v1/architect/builds              start a build   {"intent": "...", "verify": "...",
                                                            "review_plan": bool, "plan": {...}}
    GET  /v1/architect/builds              list builds (summaries)
    GET  /v1/architect/builds/{id}         build record (?events=true for the log)
    GET  /v1/architect/builds/{id}/events  SSE live stream (replay + tail, ends [DONE])

`/plans` answers "what would you build?" for one LLM call and a few catalog
lookups, without staging a package or spending a build. It is also where a caller
sees which steps reach outside the model and what would provide them — the thing
that decides whether a build is worth starting at all.

The ArchitectManager reuses the RunManager's provider + registry so builds
register into the same store the studio reads from.
"""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from ..architect_builds.store import BUILD_TERMINAL
from ..streaming.sse import sse_data, sse_done, sse_ping

_POLL_S = 0.1


#: Bounds on the conversation history a client may send. This is browser-supplied
#: text that becomes model context, so it is capped here rather than trusted: an
#: unbounded list would let one request blow the context window (and the bill) for
#: a build that only needed the last exchange.
_MAX_HISTORY_TURNS = 12
_MAX_HISTORY_CHARS = 2000


def _history(value: Any) -> list[dict[str, str]] | None:
    """Normalise a client-supplied transcript into `{role, text}` turns."""
    if not isinstance(value, list):
        return None
    out: list[dict[str, str]] = []
    for item in value[-_MAX_HISTORY_TURNS:]:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        role = "user" if str(item.get("role")) == "user" else "assistant"
        out.append({"role": role, "text": text[:_MAX_HISTORY_CHARS]})
    return out or None


def _arch_manager(server, request: Request):
    """The ArchitectManager for *this request's* user.

    Scoped per account so a build registers into that user's own workflow
    registry rather than a registry everyone shares.
    """
    from ..workspaces import architect_manager_for

    try:
        return architect_manager_for(server, getattr(request.state, "user", None))
    except Exception as e:  # noqa: BLE001 - no provider → clean 503
        raise HTTPException(
            status_code=503,
            detail=f"Architect unavailable: no usable LLM provider ({e})",
        ) from e


def mount_architect_routes(router: APIRouter, server) -> None:
    @router.post("/v1/architect/plans")
    async def make_plan(request: Request, body: dict[str, Any] | None = None):
        """Plan a workflow without building it.

        Cheap on purpose: one model call plus the capability lookups. Returns the
        steps, which of them reach outside the model, and what provides each.
        """
        body = body or {}
        intent = (body.get("intent") or "").strip()
        if not intent:
            raise HTTPException(status_code=422, detail="'intent' is required")

        manager = _arch_manager(server, request)
        from neurosurfer.architect.planner import plan_and_resolve

        answers = body.get("answers")
        try:
            plan = await plan_and_resolve(
                manager.provider,
                intent,
                answers=answers if isinstance(answers, dict) else None,
            )
        except Exception as e:  # noqa: BLE001 - upstream/model failure, not a bug here
            raise HTTPException(
                status_code=502, detail=f"Planning failed: {type(e).__name__}: {e}"
            ) from e
        return {
            "plan": plan.to_dict(),
            "rendered": plan.render(),
            "unresolved": [s.id for s in plan.unresolved_steps],
        }

    @router.post("/v1/architect/builds", status_code=202)
    async def start_build(request: Request, body: dict[str, Any] | None = None):
        body = body or {}
        intent = (body.get("intent") or "").strip()
        if not intent:
            raise HTTPException(status_code=422, detail="'intent' is required")
        verify = body.get("verify")
        refines = (body.get("refines") or "").strip() or None
        if refines is not None:
            # Refining something that isn't there would fail deep inside the
            # agent as a confusing "cannot load package"; say so up front.
            registry = _arch_manager(server, request).registry
            if not registry.exists(refines):
                raise HTTPException(
                    status_code=404, detail=f"Workflow '{refines}' not found"
                )
        plan = body.get("plan")
        if plan is not None and not (isinstance(plan, dict) and plan.get("steps")):
            raise HTTPException(
                status_code=422, detail="'plan' must be a plan object with steps"
            )
        from ..workspaces import (
            discovery_source_for,
            known_credentials_for,
            secret_saver_for,
        )

        user = getattr(request.state, "user", None)
        rec = _arch_manager(server, request).start(
            intent,
            verify=verify,
            # `auto` | `always` | `off`, defaulting to auto. Passed through
            # unconverted so an older caller's `true`/`false` still means what it
            # meant — the manager normalises both forms.
            clarify=body.get("clarify", "auto"),
            approve_tools=bool(body.get("approve_tools")),
            review_plan=bool(body.get("review_plan")),
            refines=refines,
            plan=plan,
            # Earlier turns of this conversation. The panel is a chat, so a
            # follow-up ("here is the connection string") has to be readable
            # as an answer rather than a brand-new request.
            history=_history(body.get("history")),
            # Resolved here, on the request thread — see `start`.
            discovery=discovery_source_for(server, user),
            credentials=known_credentials_for(server, user),
            save_secret=secret_saver_for(server, user),
        )
        return rec.to_dict(include_events=False)

    @router.post("/v1/architect/builds/{build_id}/respond")
    async def respond_to_build(build_id: str, request: Request, body: dict[str, Any] | None = None):
        """Answer whatever the build is currently parked on."""
        body = body or {}
        manager = _arch_manager(server, request)
        rec = manager.get(build_id)
        if rec is None:
            raise HTTPException(status_code=404, detail=f"Build '{build_id}' not found")
        interaction_id = (body.get("interaction_id") or "").strip()
        if not interaction_id:
            raise HTTPException(status_code=422, detail="'interaction_id' is required")
        if "value" not in body:
            raise HTTPException(status_code=422, detail="'value' is required")
        if not manager.respond(build_id, interaction_id, body["value"]):
            # Either the build moved on or this is a stale tab answering a
            # question that is no longer the one being asked.
            raise HTTPException(
                status_code=409,
                detail="That interaction is no longer awaiting an answer.",
            )
        return rec.to_dict(include_events=False)

    @router.post("/v1/architect/builds/{build_id}/cancel")
    async def cancel_build(build_id: str, request: Request):
        manager = _arch_manager(server, request)
        rec = manager.get(build_id)
        if rec is None:
            raise HTTPException(status_code=404, detail=f"Build '{build_id}' not found")
        if not manager.cancel(build_id):
            raise HTTPException(
                status_code=409, detail=f"Build is already {rec.status}."
            )
        return rec.to_dict(include_events=False)

    @router.get("/v1/architect/builds")
    async def list_builds(request: Request):
        return {
            "builds": [r.to_dict(include_events=False) for r in _arch_manager(server, request).list()]
        }

    @router.get("/v1/architect/builds/{build_id}")
    async def get_build(build_id: str, request: Request, events: bool = False):
        rec = _arch_manager(server, request).get(build_id)
        if rec is None:
            raise HTTPException(status_code=404, detail=f"Build '{build_id}' not found")
        return rec.to_dict(include_events=events)

    @router.get("/v1/architect/builds/{build_id}/events")
    async def stream_build_events(build_id: str, request: Request):
        rec = _arch_manager(server, request).get(build_id)
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

    # ── runtime repair (V3 Phase 5d) ────────────────────────────────────────
    @router.post("/v1/architect/repairs")
    async def diagnose_workflow(request: Request, body: dict[str, Any] | None = None):
        """Run a registered workflow and report what to do about how it went.

        Diagnosis only: nothing is written to the package. A caller that wants the
        proposal applied posts it back to ``/v1/architect/repairs/apply``, which is
        the whole point of the phase — the legacy refiner patched a registered
        workflow with nobody's approval.
        """
        body = body or {}
        name = (body.get("workflow") or "").strip()
        if not name:
            raise HTTPException(status_code=422, detail="'workflow' is required")
        inputs = body.get("inputs")
        if inputs is not None and not isinstance(inputs, dict):
            raise HTTPException(status_code=422, detail="'inputs' must be an object")

        manager = _arch_manager(server, request)
        if not manager.registry.exists(name):
            raise HTTPException(status_code=404, detail=f"Workflow '{name}' not found")

        from neurosurfer.architect.refine import WorkflowRefiner

        refiner = WorkflowRefiner(manager.provider, registry=manager.registry)
        try:
            proposal = await refiner.diagnose(name, inputs or {})
        except Exception as e:  # noqa: BLE001 - a model/runtime failure, not a bug here
            raise HTTPException(
                status_code=502, detail=f"Diagnosis failed: {type(e).__name__}: {e}"
            ) from e

        return {
            "workflow": proposal.workflow,
            "ran": proposal.ran,
            "ok": proposal.ok,
            "actionable": proposal.actionable,
            "rendered": proposal.render(),
            "failed_nodes": [
                {
                    "node_id": d.node_id,
                    "kind": d.kind,
                    "error": d.error,
                    "diagnosis": d.diagnosis,
                    "patch": d.patch,
                    "external_reason": d.external_reason,
                    "rejected_tools": d.rejected_tools,
                    "inert_fields": d.inert_fields,
                    "actionable": d.actionable,
                    "answered": d.answered,
                }
                for d in proposal.failed_nodes
            ],
        }

    @router.post("/v1/architect/repairs/apply")
    async def apply_repair(request: Request, body: dict[str, Any] | None = None):
        """Write an approved proposal's patches into the registered package.

        Takes the patches explicitly rather than a proposal id: the diagnosis is
        not persisted, and a reviewer who edited a patch before approving it should
        be able to send back what they actually approved.
        """
        body = body or {}
        name = (body.get("workflow") or "").strip()
        patches = body.get("patches")
        if not name:
            raise HTTPException(status_code=422, detail="'workflow' is required")
        if not isinstance(patches, dict) or not patches:
            raise HTTPException(
                status_code=422,
                detail="'patches' must be a non-empty {node_id: {field: value}} object",
            )

        manager = _arch_manager(server, request)
        if not manager.registry.exists(name):
            raise HTTPException(status_code=404, detail=f"Workflow '{name}' not found")

        from neurosurfer.architect.refine import NodeDiagnosis, RepairProposal, WorkflowRefiner

        proposal = RepairProposal(
            workflow=name, ran=True,
            failed_nodes=[
                NodeDiagnosis(node_id=nid, error="", patch=dict(patch))
                for nid, patch in patches.items()
                if isinstance(patch, dict) and patch
            ],
        )
        if not proposal.failed_nodes:
            raise HTTPException(status_code=422, detail="no non-empty patch supplied")

        refiner = WorkflowRefiner(manager.provider, registry=manager.registry)
        applied = refiner.apply(proposal)

        # Re-validate what the patch produced, so a repair cannot quietly leave a
        # registered workflow in a worse state than it was found in.
        from neurosurfer.graph.workflow.validate import validate_package

        pkg = manager.registry.get(name)
        report = validate_package(pkg)
        return {
            "workflow": name,
            "applied": applied,
            "validation": {
                "ok": report.ok,
                "errors": [i.__dict__ for i in report.errors],
                "gaps": [i.__dict__ for i in report.gaps],
                "info": [i.__dict__ for i in report.infos],
                "warnings": [i.__dict__ for i in report.warnings],
            },
        }
