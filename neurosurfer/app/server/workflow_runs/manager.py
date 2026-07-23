"""RunManager — background workflow execution behind the API (Phase 2).

Bridges the synchronous :class:`WorkflowRunner` to the async gateway: each run
executes in a daemon worker thread while its :class:`RunRecord` collects the live
node-event log (which the SSE endpoint tails) and, on completion, the per-node
results and final outputs.

Semantics:

- **start** — validates the workflow exists, creates a record, spawns the worker.
- **awaiting_input** — a run whose input-kind node had no value finishes in status
  ``awaiting_input`` (not ``failed``), with the node id in the record, so a client
  knows to resume.
- **resume** — re-runs the workflow with the original inputs merged with the
  supplied values (the Phase 1i input-node contract: a pre-supplied value satisfies
  the pause). Returns a NEW run linked to the old one via ``resumed_from``.
- **cancel** — best-effort: the record is marked cancelled immediately and the
  worker's eventual result is discarded. Interrupting a node mid-flight is future
  work (needs cooperative checks in the executor).
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

from neurosurfer.graph.workflow.registry import WorkflowRegistry
from neurosurfer.graph.workflow.runner import WorkflowRunner
from neurosurfer.llm.base import Provider

from .store import TERMINAL, RunRecord, RunStore

logger = logging.getLogger("neurosurfer.server.runs")

# Marker the Phase 1i input node embeds in its error when it needs a human value.
_AWAITING_MARKER = "is awaiting a value"


def _jsonable(value: Any) -> Any:
    """Best-effort JSON-safe conversion (mirrors WorkflowState snapshots)."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        try:
            return _jsonable(dump())
        except Exception:  # noqa: BLE001
            pass
    text = repr(value)
    return text if len(text) <= 4000 else text[:4000] + "…"


class RunManager:
    """Start, observe, resume, and cancel workflow runs for the execution API."""

    def __init__(
        self,
        provider: Provider,
        *,
        registry: WorkflowRegistry | None = None,
        store: RunStore | None = None,
        cwd: Path | None = None,
        tool_context: Any = None,
    ) -> None:
        self.provider = provider
        self.registry = registry or WorkflowRegistry()
        self.store = store or RunStore()
        self._cwd = cwd
        self._tool_ctx = tool_context

    # ── queries ─────────────────────────────────────────────────────────────
    def list_workflows(self) -> list[dict[str, Any]]:
        out = []
        for name in self.registry.list():
            try:
                pkg = self.registry.get(name)
                out.append({
                    "name": name,
                    "description": pkg.manifest.description,
                    "version": pkg.manifest.version,
                    "tags": pkg.manifest.tags,
                })
            except Exception as e:  # noqa: BLE001 - one broken package must not hide the rest
                out.append({"name": name, "error": str(e)})
        return out

    def get_workflow_graph(self, name: str) -> dict[str, Any] | None:
        """The full graph as JSON — nodes, edges, control flow — for the UI."""
        if not self.registry.exists(name):
            return None
        pkg = self.registry.get(name)
        return {
            "name": pkg.manifest.name,
            "description": pkg.manifest.description,
            "version": pkg.manifest.version,
            "graph": pkg.graph.model_dump(mode="json"),
        }

    def get(self, run_id: str) -> RunRecord | None:
        return self.store.get(run_id)

    def list_runs(self) -> list[RunRecord]:
        return self.store.list()

    # ── lifecycle ───────────────────────────────────────────────────────────
    def start(
        self, workflow: str, inputs: dict[str, Any], *, resumed_from: str | None = None
    ) -> RunRecord:
        pkg = self.registry.get(workflow)  # raises WorkflowNotFoundError → 404 upstream
        rec = self.store.create(workflow, inputs)
        if resumed_from:
            rec.add_event("run", status="resumed", resumed_from=resumed_from)

        trace_path = self.store.dir / f"{rec.id}.trace.json"

        def _work() -> None:
            try:
                def on_event(node_id: str, status: str) -> None:
                    if rec.cancelled:
                        return
                    rec.add_event("node", node_id=node_id, status=status)
                    rec.set_node(node_id, status=status)

                runner = WorkflowRunner(
                    self.provider, cwd=self._cwd, tool_context=self._tool_ctx
                )
                result = runner.run(
                    pkg, dict(inputs), on_node_event=on_event, trace_path=trace_path
                )
                if rec.status == "cancelled":
                    self.store.persist(rec)
                    return

                awaiting: str | None = None
                for nid, nr in result.nodes.items():
                    node_status = (
                        "error" if nr.error and not nr.skipped
                        else ("skipped" if nr.skipped else "ok")
                    )
                    rec.set_node(
                        nid,
                        status=node_status,
                        output=_jsonable(nr.raw_output),
                        error=nr.error,
                        skipped=nr.skipped,
                        skip_reason=nr.skip_reason,
                        duration_ms=nr.duration_ms,
                    )
                    # Only the input node itself counts — skipped dependents embed
                    # the upstream error text and must not steal the attribution.
                    if nr.error and not nr.skipped and _AWAITING_MARKER in nr.error:
                        awaiting = nid

                rec.trace_path = str(trace_path) if trace_path.exists() else None
                final = {k: _jsonable(v) for k, v in (result.final or {}).items()}
                if awaiting is not None:
                    rec.add_event("input_required", node_id=awaiting)
                    rec.finish("awaiting_input", final=final,
                               error=result.nodes[awaiting].error)
                elif result.errors:
                    rec.finish("failed", final=final,
                               error="; ".join(f"{k}: {v}" for k, v in result.errors.items()))
                else:
                    rec.finish("succeeded", final=final)
            except Exception as e:  # noqa: BLE001 - a worker crash must land in the record
                logger.exception("run %s crashed", rec.id)
                if rec.status not in TERMINAL:
                    rec.finish("failed", error=str(e))
            finally:
                self.store.persist(rec)

        threading.Thread(target=_work, name=f"wfrun-{rec.id[:8]}", daemon=True).start()
        return rec

    def resume(self, run_id: str, values: dict[str, Any]) -> RunRecord | None:
        """Re-run the workflow with the paused run's inputs + the supplied values."""
        rec = self.store.get(run_id)
        if rec is None:
            return None
        merged = {**rec.inputs, **(values or {})}
        return self.start(rec.workflow, merged, resumed_from=run_id)

    def cancel(self, run_id: str) -> RunRecord | None:
        rec = self.store.get(run_id)
        if rec is None:
            return None
        if rec.status not in TERMINAL:
            rec.request_cancel()
            rec.finish("cancelled")
            self.store.persist(rec)
        return rec
