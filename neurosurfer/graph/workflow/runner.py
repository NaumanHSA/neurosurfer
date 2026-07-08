"""WorkflowRunner — wire a WorkflowPackage to GraphExecutor and run it.

Responsibilities:
- Build a native ``ToolPool`` from registered neurosurfer ``Tool`` objects
  (restricted to tools declared in the graph).
- Patch ``sys.path`` so function-node ``callable`` imports resolve against the
  package directory.
- Drive ``GraphExecutor.run`` with the native provider + ToolPool path (R3+R4)
  and return the ``GraphExecutionResult``.

Callers (CLI, tests) can optionally pass a *progress* callback that receives
``(node_id, status, duration_ms)`` tuples as nodes complete — used by the CLI
renderer to update a live table without coupling the runner to Rich.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from neurosurfer.graph.engine import GraphExecutionResult, GraphExecutor, InputValidationError
from neurosurfer.llm.base import Provider
from neurosurfer.observability.run import traced_run
from neurosurfer.tools.base import (
    BaseIOHandler,
    ShellApproval,
    ToolContext,
    ToolPool,
    WriteChoice,
)
from neurosurfer.tools.registry import all_tools

from .package import WorkflowPackage, _PackagePathContext
from .schema import Graph

__all__ = ["WorkflowRunner", "run_workflow"]

# (node_id, "ok" | "error", duration_ms) — post-run summary hook for callers
ProgressCallback = Callable[[str, str, int], None]
# (node_id, "start" | "ok" | "error" | "skipped") — fired live as nodes run
NodeEventCallback = Callable[[str, str], None]


class WorkflowRunner:
    """Run a :class:`WorkflowPackage` against a neurosurfer :class:`Provider`.

    Parameters
    ----------
    provider:
        The LLM provider powering all ``agent``-kind nodes.
    cwd:
        Working directory used for tool contexts (defaults to ``Path.cwd()``).
    tool_context:
        If given, used as-is for all native tool calls.
        If omitted, a minimal :class:`ToolContext` is constructed from *cwd*.
    allowed_tools:
        Explicit set of neurosurfer tool names to expose.  Defaults to all
        tools returned by :func:`~neurosurfer.tools.registry.all_tools`.
    """

    def __init__(
        self,
        provider: Provider,
        *,
        cwd: Path | None = None,
        tool_context: ToolContext | None = None,
        allowed_tools: set[str] | None = None,
    ) -> None:
        self._provider = provider
        self._cwd = cwd or Path.cwd()
        self._tool_ctx = tool_context or _make_tool_context(self._cwd)
        self._allowed_tools = allowed_tools

    # ── public ────────────────────────────────────────────────────────────────

    def run(
        self,
        pkg: WorkflowPackage,
        inputs: dict[str, Any],
        *,
        progress: ProgressCallback | None = None,
        on_node_event: NodeEventCallback | None = None,
        trace_path: Path | None = None,
    ) -> GraphExecutionResult:
        """Execute *pkg* with the supplied *inputs* dict.

        The package directory is prepended to ``sys.path`` for the duration of
        the call so that function nodes can import from ``nodes/``.
        """
        self._validate_inputs(pkg, inputs)

        tracer = None
        if trace_path is not None:
            from neurosurfer.tracing import Tracer, TracerConfig

            tracer = Tracer(
                config=TracerConfig(enabled=True, log_steps=False),
                meta={"workflow": pkg.manifest.name},
            )

        tool_pool = self._build_tool_pool(pkg.graph)
        executor = GraphExecutor(
            pkg.graph,
            provider=self._provider,
            native_tools=tool_pool,
            tool_ctx=self._tool_ctx,
            tracer=tracer,
            log_traces=False,
        )

        with _PackagePathContext(pkg), traced_run(
            f"workflow:{pkg.manifest.name}",
            metadata={"workflow": pkg.manifest.name, "kind": "workflow"},
        ):
            result = executor.run(inputs, node_event=on_node_event)

        if tracer is not None and trace_path is not None:
            self._dump_trace(tracer, trace_path)

        if progress is not None:
            for node_id, node_result in result.nodes.items():
                status = "error" if node_result.error else "ok"
                progress(node_id, status, node_result.duration_ms)

        return result

    # ── private ───────────────────────────────────────────────────────────────

    def _dump_trace(self, tracer: Any, trace_path: Path) -> None:
        import json

        try:
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            trace_path.write_text(
                json.dumps(tracer.export_json(), indent=2, default=str),
                encoding="utf-8",
            )
        except Exception:  # noqa: BLE001 - tracing must never break a run
            pass

    def _validate_inputs(self, pkg: WorkflowPackage, inputs: dict[str, Any]) -> None:
        declared = pkg.graph.inputs
        if not declared:
            return

        missing = [s.name for s in declared if s.required and s.name not in inputs]
        if missing:
            raise InputValidationError(
                f"Workflow '{pkg.manifest.name}' is missing required input(s): "
                + ", ".join(f"'{m}'" for m in missing)
            )

    def _build_tool_pool(self, graph: Graph) -> ToolPool:
        """Build a native ToolPool containing only tools declared across all nodes."""
        needed: set[str] = set()
        for node in graph.nodes:
            needed.update(node.tools)

        if not needed:
            return ToolPool([])

        tool_map = {t.name: t for t in all_tools()}
        unknown = sorted(
            name for name in needed
            if name not in tool_map
            and (self._allowed_tools is None or name in self._allowed_tools)
        )
        if unknown:
            available = ", ".join(sorted(tool_map))
            raise ValueError(
                "This workflow references tool(s) that are not registered: "
                + ", ".join(f"'{n}'" for n in unknown)
                + ".\nRegistered tools are: "
                + available
                + ".\nThis workflow was likely generated before tool validation was "
                "added — rebuild it with /workflow build to fix the wiring."
            )

        tools = []
        for name in sorted(needed):
            if self._allowed_tools is not None and name not in self._allowed_tools:
                continue
            tool = tool_map.get(name)
            if tool is not None:
                tools.append(tool)

        return ToolPool(tools)


# ── convenience wrapper ───────────────────────────────────────────────────────

def run_workflow(
    pkg: WorkflowPackage,
    inputs: dict[str, Any],
    *,
    provider: Provider,
    cwd: Path | None = None,
    progress: ProgressCallback | None = None,
) -> GraphExecutionResult:
    """One-shot helper: build a :class:`WorkflowRunner` and execute *pkg*."""
    runner = WorkflowRunner(provider, cwd=cwd)
    return runner.run(pkg, inputs, progress=progress)


# ── internal helpers ──────────────────────────────────────────────────────────

class _HeadlessIO(BaseIOHandler):
    """Non-interactive workflow IO: no human present, so *deny* anything that
    would otherwise prompt (shell, out-of-scope writes) rather than silently
    auto-approving. Everything else inherits the base defaults."""

    async def request_shell_approval(self, command: str, reason: str) -> ShellApproval:
        return ShellApproval(False)

    async def request_write_approval(self, path: str, summary: str) -> WriteChoice:
        return "deny"


def _make_tool_context(cwd: Path) -> ToolContext:
    """Minimal ToolContext for non-interactive workflow execution."""
    return ToolContext(cwd=cwd, io=_HeadlessIO())
