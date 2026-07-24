"""ArchitectManager (S5) — runs Architect builds in the background and records
their step log + staged-graph snapshots for the studio to stream.

Each build runs the ReAct ``ArchitectAgent`` in a worker thread. The agent's
``notify`` callback appends a ``log`` event and snapshots the live staged graph
(``agent.session.graph_dict()``) so the studio canvas can animate the build.
The terminal outcome is a registered workflow (``succeeded``), a
``WorkflowInfeasible`` (``blocked``), or an error (``failed``).

``agent_factory`` is injectable so tests can drive the plumbing without an LLM.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Callable

from .store import BuildRecord


class ArchitectManager:
    def __init__(
        self,
        provider: Any,
        *,
        registry: Any = None,
        staging_root: Path | None = None,
        agent_factory: Callable[[Callable[[str], None], str], Any] | None = None,
        verify_default: str = "encouraged",
    ) -> None:
        self.provider = provider
        if registry is None:
            from neurosurfer.graph.workflow.registry import WorkflowRegistry

            registry = WorkflowRegistry()
        self.registry = registry
        self.staging_root = staging_root
        self._agent_factory = agent_factory
        self._verify_default = verify_default
        self._builds: dict[str, BuildRecord] = {}
        self._lock = threading.Lock()

    # ── read ──────────────────────────────────────────────────────────────
    def get(self, build_id: str) -> BuildRecord | None:
        return self._builds.get(build_id)

    def list(self) -> list[BuildRecord]:
        return sorted(self._builds.values(), key=lambda r: r.created_at, reverse=True)

    # ── lifecycle ─────────────────────────────────────────────────────────
    def start(self, intent: str, *, verify: str | None = None) -> BuildRecord:
        rec = BuildRecord(intent=intent)
        with self._lock:
            self._builds[rec.id] = rec
        verify_mode = verify or self._verify_default

        def _snapshot(agent: Any) -> None:
            try:
                sess = getattr(agent, "session", None)
                g = sess.graph_dict() if sess is not None else None
            except Exception:  # noqa: BLE001 - snapshot is best-effort
                g = None
            if g and g.get("nodes") and g != rec.graph:
                rec.graph = g
                rec.add_event("graph", graph=g)

        def _work() -> None:
            holder: dict[str, Any] = {}
            rec.add_event("build", status="running")

            def notify(msg: str) -> None:
                rec.add_event("log", message=msg)
                _snapshot(holder.get("agent"))

            try:
                import asyncio

                agent = self._make_agent(notify, verify_mode)
                holder["agent"] = agent
                path = asyncio.run(agent.build(intent))

                _snapshot(agent)
                name = getattr(getattr(agent, "session", None), "name", None)
                name = name or Path(path).name
                rec.workflow = name
                rec.path = path
                # Authoritative final graph from the registry, if available.
                try:
                    pkg = self.registry.get(name)
                    rec.graph = pkg.graph.model_dump(mode="json")
                    rec.add_event("graph", graph=rec.graph)
                except Exception:  # noqa: BLE001 - fall back to the last snapshot
                    pass
                rec.status = "succeeded"
                rec.add_event("build", status="succeeded", workflow=name, path=path)
            except Exception as e:  # noqa: BLE001 - map outcomes to terminal states
                from neurosurfer.architect.agent.agent import WorkflowInfeasible

                if isinstance(e, WorkflowInfeasible):
                    rec.status = "blocked"
                    rec.error = str(e)
                    rec.add_event("build", status="blocked", error=str(e))
                else:
                    rec.status = "failed"
                    rec.error = str(e)
                    rec.add_event("build", status="failed", error=str(e))

        threading.Thread(target=_work, name=f"arch-{rec.id[:8]}", daemon=True).start()
        return rec

    # ── agent construction (overridable for tests) ────────────────────────
    def _make_agent(self, notify: Callable[[str], None], verify: str) -> Any:
        if self._agent_factory is not None:
            return self._agent_factory(notify, verify)
        from neurosurfer.architect import ArchitectAgent

        async def _approve(draft: Any, _res: Any) -> bool:
            notify(f"authored tool: {getattr(draft, 'name', '?')} (auto-approved)")
            return True

        return ArchitectAgent(
            self.provider,
            registry=self.registry,
            staging_root=self.staging_root,
            notify=notify,
            verify=verify,
            approve_tool=_approve,
        )
