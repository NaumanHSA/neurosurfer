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

from pydantic import BaseModel

from neurosurfer.graph.engine import GraphExecutionResult, GraphExecutor, InputValidationError
from neurosurfer.llm.base import Provider
from neurosurfer.observability.run import traced_run
from neurosurfer.tools.base import (
    BaseIOHandler,
    ShellApproval,
    Tool,
    ToolContext,
    ToolPool,
    ToolResult,
    WriteChoice,
)
from neurosurfer.tools.registry import all_tools

from .package import WorkflowPackage, _PackagePathContext
from .schema import Graph

__all__ = ["NotExercisedTool", "WorkflowRunner", "run_workflow"]

# Prefixes the output of every stubbed call, so a stub's trace is greppable and a
# downstream node reading it has been told, in words, that it is not real data.
NOT_EXERCISED_MARK = "[NOT EXERCISED]"


class _AnyArgs(BaseModel):
    """Accepts whatever the graph passes — a stub cannot validate a real schema."""

    model_config = {"extra": "allow"}


class NotExercisedTool(Tool):
    """Stands in for a tool this machine cannot provide (V3 Phase 5b).

    A workflow whose Gmail step needs an MCP server nobody has installed cannot be
    tested on that step — but the *rest* of the graph can be, and refusing to run
    any of it teaches nothing. The stub keeps the run going and marks its output
    loudly, so the judge is not handed fabricated data as though it were real and
    the report can say which steps were never exercised.
    """

    input_model = _AnyArgs

    def __init__(self, name: str, reason: str) -> None:
        self.name = name
        self.reason = reason
        self.description = f"Unavailable in this environment: {reason}"

    def is_read_only(self, args: BaseModel) -> bool:
        return True

    async def call(self, args: BaseModel, ctx: ToolContext) -> ToolResult:
        return ToolResult.ok(
            f"{NOT_EXERCISED_MARK} `{self.name}` was not called — {self.reason}. "
            "No real data was produced for this step."
        )


def uncredentialed_reason(tool: Tool) -> str | None:
    """Why an MCP-backed *tool* cannot really run here, or None if it can.

    The case Phase 5b is actually about: the server is installed and connected —
    so the tool exists and the graph validates — but the credentials it needs
    were never supplied. Calling it would fail with an auth error partway through
    a run, which reads like a broken workflow rather than a missing secret.

    An `env`/`headers` value written as `${VAR}` is a declaration that the value
    comes from the environment; expanding to empty means nobody supplied it.
    """
    if not getattr(tool, "is_mcp", False):
        return None
    server_name = getattr(tool, "server_name", None)
    if not server_name:
        return None
    try:
        from neurosurfer.config.mcp import McpStore

        cfg = McpStore.default().get(server_name)
    except Exception:  # noqa: BLE001 - no config store is not a credential problem
        return None
    if cfg is None:
        return None

    unset = sorted(
        [k for k, v in cfg.env.items() if "${" in v and not cfg.resolved_env().get(k)]
        + [k for k, v in cfg.headers.items() if "${" in v and not cfg.resolved_headers().get(k)]
    )
    if not unset:
        return None
    return (
        f"its MCP server '{server_name}' has no value for "
        + ", ".join(unset)
        + " (set them in the environment to test this step for real)"
    )

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
    stub_missing_tools:
        When True, a tool the graph names but this machine cannot provide becomes
        a :class:`NotExercisedTool` instead of raising, and its name lands in
        :attr:`stubbed_tools`. Verification uses this to test the reachable part
        of a graph whose external steps it cannot exercise; a real run must keep
        the default and fail loudly.
    """

    def __init__(
        self,
        provider: Provider,
        *,
        cwd: Path | None = None,
        tool_context: ToolContext | None = None,
        allowed_tools: set[str] | None = None,
        provider_resolver: Any = None,
        stub_missing_tools: bool = False,
    ) -> None:
        self._provider = provider
        self._cwd = cwd or Path.cwd()
        self._tool_ctx = tool_context or _make_tool_context(self._cwd)
        self._allowed_tools = allowed_tools
        self._stub_missing_tools = stub_missing_tools
        #: Tools replaced by a stub on the most recent run (empty unless stubbing).
        self.stubbed_tools: set[str] = set()
        # Lets a node name its own provider. Omitted, every node runs on `provider`
        # and `node.model` rebinds a copy of it.
        self._provider_resolver = provider_resolver

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
        self.stubbed_tools = set()

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
            provider_resolver=self._provider_resolver,
        )

        with _PackagePathContext(pkg), traced_run(
            f"workflow:{pkg.manifest.name}",
            metadata={"workflow": pkg.manifest.name, "kind": "workflow"},
            input=inputs,
        ) as wspan:
            result = executor.run(inputs, node_event=on_node_event)
            # Attach the run's outcome to the workflow span so the trace root shows
            # the final outputs (or the failure) instead of null/undefined.
            if wspan is not None:
                if result.errors:
                    wspan.error("; ".join(
                        f"{nid}: {err[:150]}" for nid, err in result.errors.items()
                    ))
                else:
                    from neurosurfer.graph.engine.state import _jsonable

                    wspan.output = {
                        k: _jsonable(v) for k, v in (result.final or {}).items()
                    }

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
            # The missing names may live on a configured MCP server — connect on
            # demand (idempotent; publishes tools to the live registry) and retry.
            try:
                from neurosurfer.mcp.runtime import ensure_mcp_tools

                if ensure_mcp_tools():
                    tool_map = {t.name: t for t in all_tools()}
                    unknown = [n for n in unknown if n not in tool_map]
            except Exception:  # noqa: BLE001 - fall through to the clear error below
                pass
        if unknown and not self._stub_missing_tools:
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
            if tool is None:
                if self._stub_missing_tools:
                    self.stubbed_tools.add(name)
                    tools.append(NotExercisedTool(
                        name,
                        "it is not available on this machine (an MCP server that "
                        "is not installed or not connected, or a tool that no "
                        "longer exists)",
                    ))
                continue
            reason = uncredentialed_reason(tool) if self._stub_missing_tools else None
            if reason:
                self.stubbed_tools.add(name)
                tools.append(NotExercisedTool(name, reason))
            else:
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
