from __future__ import annotations

import logging
import time
from contextvars import copy_context
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from pydantic import BaseModel as PydModel

# Native-stack (R3+R4)
from neurosurfer.llm.base import Provider
from neurosurfer.llm.types import GenerationConfig
from neurosurfer.observability.run import traced_run
from neurosurfer.tools.base import ToolContext, ToolPool
from neurosurfer.tracing import Tracer, TracerConfig, TraceStepContext

if TYPE_CHECKING:
    from neurosurfer.rag.agent import RAGAgent

from .artifacts import ArtifactStore
from .errors import (
    GraphConfigurationError,
    GraphExecutionError,
    NodeFailedError,
    NodeSkippedError,
    NodeTimeoutError,
    StructuredOutputError,
)
from .export import GraphExporter
from .manager import ManagerAgent, ManagerConfig
from .schema import Graph, GraphExecutionResult, GraphNode, NodeExecutionResult
from .templates import DEFAULT_NODE_SYSTEM_TEMPLATE
from .utils import import_string, normalize_and_validate_graph_inputs, topo_sort


class GraphExecutor:
    """Execute a Graph DAG using a native Provider + ToolPool (R4 native path).

    Parameters
    ----------
    graph:       The loaded Graph spec.
    provider:    Native LLM provider for base/react nodes.
    native_tools: Native ToolPool for tool nodes and react agents.
    tool_ctx:    ToolContext supplied to tool/react nodes.
    llm, toolkit:  Accepted but ignored (legacy compat — pass provider= instead).
    """

    def __init__(
        self,
        graph: Graph,
        *,
        provider: Optional[Provider] = None,
        native_tools: Optional[ToolPool] = None,
        tool_ctx: Optional[ToolContext] = None,
        # Legacy params — accepted but ignored so old call-sites don't crash.
        llm: Any = None,
        toolkit: Any = None,
        manager_llm: Any = None,
        rag_agent: Optional[Any] = None,
        manager_config: Optional[ManagerConfig] = None,
        exporter: Optional[GraphExporter] = None,
        tracer: Optional[Tracer] = None,
        artifact_store: Optional[ArtifactStore] = None,
        logger: Optional[logging.Logger] = None,
        log_traces: bool = True,
        parallelism: int = 1,
    ) -> None:
        self.graph = graph
        self.provider = provider
        self.native_tools = native_tools
        self._tool_ctx = tool_ctx
        self.rag_agent = rag_agent
        self.exporter = exporter
        self.logger = logger or logging.getLogger(__name__)
        self.tracer = tracer
        self.log_traces = log_traces
        self.artifacts = artifact_store or ArtifactStore()
        self.parallelism = max(1, parallelism)

        self.manager = ManagerAgent(
            config=manager_config,
            tracer=tracer,
            log_traces=log_traces,
        )

        self._node_map: Dict[str, GraphNode] = self.graph.node_map()
        self._order = topo_sort(self.graph.nodes)
        self._layers: List[List[str]] = _topo_layers(self.graph.nodes)

        self._validate_tools()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    @staticmethod
    def _topo_layers_static(nodes) -> List[List[str]]:
        return _topo_layers(nodes)

    def run(
        self,
        inputs: Any,
        *,
        manager_temperature: float = None,
        manager_max_new_tokens: int = None,
        trace_step: Optional[TraceStepContext] = None,
        node_event: Optional[Any] = None,
    ) -> GraphExecutionResult:
        """
        Execute the entire graph once.

        Parameters
        ----------
        inputs:
            Runtime inputs to the graph.

            If the graph declares `inputs` in YAML:
              - Must be a mapping (dict)
              - Validated and cast according to GraphInput
              - Extra keys are warned and ignored

            If the graph does NOT declare `inputs`:
              - If `inputs` is a dict, it's used as-is
              - Otherwise, it's wrapped as: {"query": inputs}
        manager_temperature:
            Temperature used for ManagerAgent when composing prompts.
        manager_max_new_tokens:
            Max new tokens for ManagerAgent responses.

        Returns
        -------
        GraphExecutionResult
            Contains the graph spec, all node results, and the final outputs.
        """
        graph_inputs = normalize_and_validate_graph_inputs(self.graph, inputs)
        nodes_results: Dict[str, NodeExecutionResult] = {}
        # Track which nodes failed (error set, raw_output None) — their dependents
        # must be skipped rather than receiving None in dep_results.
        failed_ids: Set[str] = set()

        def _emit(node_id: str, status: str) -> None:
            """Fire the optional per-node lifecycle callback, ignoring callback errors."""
            if node_event is None:
                return
            try:
                node_event(node_id, status)
            except Exception:  # noqa: BLE001 - progress UI must never break execution
                pass

        for layer in self._layers:
            # Determine which nodes in this layer are blocked by a failed upstream.
            to_skip = [nid for nid in layer if any(d in failed_ids for d in self._node_map[nid].depends_on)]
            to_run  = [nid for nid in layer if nid not in to_skip]

            # Mark skipped nodes immediately (no LLM call needed).
            for nid in to_skip:
                node = self._node_map[nid]
                failed_upstream = next(d for d in node.depends_on if d in failed_ids)
                upstream_err = nodes_results[failed_upstream].error or "unknown error"
                skip_result = NodeExecutionResult(
                    node_id=nid,
                    mode=node.mode,
                    raw_output=None,
                    started_at=time.time(),
                    duration_ms=0,
                    error=f"Skipped: upstream node '{failed_upstream}' failed: {upstream_err}",
                    skipped=True,
                    skip_reason=f"upstream '{failed_upstream}' failed",
                )
                nodes_results[nid] = skip_result
                failed_ids.add(nid)
                _emit(nid, "skipped")
                self._log(
                    f"Node {nid} skipped (upstream '{failed_upstream}' failed)",
                    tracer=trace_step,
                    type="warning",
                )

            if not to_run:
                continue

            # Run nodes in this layer — serially (parallelism=1) or in a thread pool.
            def _execute_one(nid: str) -> NodeExecutionResult:
                node = self._node_map[nid]
                dep_results = {d: nodes_results[d].raw_output for d in node.depends_on}
                # Find the most-recently completed non-skipped node for prev context.
                prev_result = None
                for past_id in reversed(list(nodes_results.keys())):
                    past = nodes_results[past_id]
                    if not past.skipped and past.raw_output is not None:
                        prev_result = past.raw_output
                        break
                # One trace span per node: makes non-agent (function/tool) nodes
                # visible and nests each node's agent under its *node* row. Pushes an
                # ambient TraceContext the node agent inherits (across threads too,
                # via the copy_context() used for parallel/timeout nodes).
                with traced_run(
                    f"node:{node.id}",
                    metadata={"node_id": node.id, "kind": node.kind, "mode": node.mode},
                    flush=False,
                ) as span:
                    result = self._run_node(
                        node=node,
                        graph_inputs=graph_inputs,
                        dependency_results=dep_results,
                        previous_result=prev_result,
                        manager_temperature=manager_temperature,
                        manager_max_new_tokens=manager_max_new_tokens,
                        trace_step=trace_step,
                    )
                    if span is not None and result.error and not result.skipped:
                        span.error(result.error)
                    return result

            if self.parallelism == 1 or len(to_run) == 1:
                for nid in to_run:
                    _emit(nid, "start")
                    result = _execute_one(nid)
                    nodes_results[nid] = result
                    if result.error and not result.skipped:
                        failed_ids.add(nid)
                        _emit(nid, "error")
                        self._log(f"Node {nid} failed: {result.error}", tracer=trace_step, type="error")
                        if self.graph.fail_fast:
                            raise GraphExecutionError(
                                f"Node '{nid}' failed (fail_fast=True): {result.error}",
                                failed_node=nid,
                            )
                    else:
                        _emit(nid, "ok")
            else:
                for nid in to_run:
                    _emit(nid, "start")
                with ThreadPoolExecutor(max_workers=min(self.parallelism, len(to_run))) as pool:
                    # Each worker thread runs inside a *fresh* copy of the current
                    # context so the ambient observability TraceContext propagates and
                    # parallel node agents nest under the workflow trace. One snapshot
                    # per node — a Context can't be entered by two threads at once.
                    futures: Dict[str, Future] = {
                        nid: pool.submit(copy_context().run, _execute_one, nid)
                        for nid in to_run
                    }
                    for nid, fut in futures.items():
                        try:
                            result = fut.result()
                        except Exception as exc:
                            # Wrap any unexpected exception from the thread.
                            node = self._node_map[nid]
                            result = NodeExecutionResult(
                                node_id=nid,
                                mode=node.mode,
                                raw_output=None,
                                started_at=time.time(),
                                duration_ms=0,
                                error=f"Unexpected thread error: {exc}",
                            )
                        nodes_results[nid] = result
                        if result.error and not result.skipped:
                            failed_ids.add(nid)
                            _emit(nid, "error")
                            self._log(f"Node {nid} failed: {result.error}", tracer=trace_step, type="error")
                            if self.graph.fail_fast:
                                raise GraphExecutionError(
                                    f"Node '{nid}' failed (fail_fast=True): {result.error}",
                                    failed_node=nid,
                                )
                        else:
                            _emit(nid, "ok")

        final = self._select_final_outputs(nodes_results)
        all_errors = {nid: r.error for nid, r in nodes_results.items() if r.error and not r.skipped}
        all_skipped = [nid for nid, r in nodes_results.items() if r.skipped]
        if all_errors:
            self._log("Graph completed with errors. Returning partial results.", tracer=trace_step, type="warning")
        return GraphExecutionResult(
            graph=self.graph,
            nodes=nodes_results,
            final=final,
            errors=all_errors,
            skipped=all_skipped,
        )

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _validate_tools(self) -> None:
        """Ensure all YAML tool names exist in whichever tool source is active."""
        if self.native_tools is not None:
            missing = {
                name
                for node in self.graph.nodes
                for name in node.tools
                if self.native_tools.get(name) is None
            }
            if missing:
                raise GraphConfigurationError(
                    f"YAML refers to unknown tools not in ToolPool: {sorted(missing)}"
                )
            return


    # ------------------------------------------------------------------ #
    # Non-LLM node runners
    # ------------------------------------------------------------------ #
    def _run_function_node(
        self,
        node: GraphNode,
        graph_inputs: Dict[str, Any],
        dependency_results: Dict[str, Any],
    ) -> NodeExecutionResult:
        started_at = time.time()
        try:
            if not node.callable:
                raise GraphConfigurationError(
                    f"function node '{node.id}' has no 'callable' set."
                )
            fn = import_string(node.callable)
            kwargs = {**graph_inputs, **dependency_results}
            raw = fn(**kwargs)
            return NodeExecutionResult(
                node_id=node.id,
                mode=node.mode,
                raw_output=raw,
                started_at=started_at,
                duration_ms=int((time.time() - started_at) * 1000),
            )
        except Exception as e:
            self.logger.exception("Function node %s failed: %s", node.id, e)
            return NodeExecutionResult(
                node_id=node.id,
                mode=node.mode,
                raw_output=None,
                started_at=started_at,
                duration_ms=int((time.time() - started_at) * 1000),
                error=str(e),
            )

    def _run_tool_node(
        self,
        node: GraphNode,
        graph_inputs: Dict[str, Any],
        dependency_results: Dict[str, Any],
    ) -> NodeExecutionResult:
        started_at = time.time()
        try:
            if not node.tools:
                raise GraphConfigurationError(
                    f"tool node '{node.id}' has no 'tools' declared."
                )
            tool_name = node.tools[0]
            kwargs = {**graph_inputs, **dependency_results, **(node.tool_args or {})}

            if self.native_tools is None:
                raise GraphConfigurationError(
                    f"tool node '{node.id}' requires native_tools (a ToolPool) "
                    "but none was provided to the executor."
                )
            if self._tool_ctx is None:
                raise GraphConfigurationError(
                    f"tool node '{node.id}' requires a tool_ctx (ToolContext) "
                    "but none was provided to the executor."
                )
            from .node_runner import run_tool_node
            raw = run_tool_node(self.native_tools, tool_name, kwargs, self._tool_ctx)
            return NodeExecutionResult(
                node_id=node.id,
                mode=node.mode,
                raw_output=raw,
                started_at=started_at,
                duration_ms=int((time.time() - started_at) * 1000),
            )
        except Exception as e:
            self.logger.exception("Tool node %s failed: %s", node.id, e)
            return NodeExecutionResult(
                node_id=node.id,
                mode=node.mode,
                raw_output=None,
                started_at=started_at,
                duration_ms=int((time.time() - started_at) * 1000),
                error=str(e),
            )

    def _run_node(
        self,
        *,
        node: GraphNode,
        graph_inputs: Dict[str, Any],
        dependency_results: Dict[str, Any],
        previous_result: Any,
        manager_temperature: float,
        manager_max_new_tokens: int,
        trace_step: Optional[TraceStepContext] = None,
    ) -> NodeExecutionResult:

        # Non-LLM dispatch — no prompt building, no agent needed
        if node.kind in {"function", "python"}:
            return self._run_function_node(node, graph_inputs, dependency_results)
        if node.kind == "tool":
            return self._run_tool_node(node, graph_inputs, dependency_results)

        # LLM-based node (base | react)
        user_prompt = self.manager.compose_user_prompt(
            node=node,
            graph_inputs=graph_inputs,
            dependency_results=dependency_results,
            previous_result=previous_result,
            temperature=manager_temperature,
            max_new_tokens=manager_max_new_tokens,
        )
        system_prompt = self._build_system_prompt(node, graph_inputs)
        output_schema = self._load_output_schema_if_needed(node)
        timeout_s = node.policy.timeout_s if node.policy and node.policy.timeout_s else None

        if self.provider is None:
            raise GraphConfigurationError(
                f"Node '{node.id}' is a base/react node but no provider was given to the executor."
            )
        return self._run_node_native(
            node=node,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            output_schema=output_schema,
            timeout_s=timeout_s,
        )

    def _run_node_native(
        self,
        *,
        node: GraphNode,
        system_prompt: str,
        user_prompt: str,
        output_schema: Optional[type] = None,
        timeout_s: Optional[float] = None,
    ) -> NodeExecutionResult:
        """Execute a base or react node using the native provider + ToolPool stack."""
        from concurrent.futures import ThreadPoolExecutor
        from concurrent.futures import TimeoutError as FuturesTimeout

        from .node_runner import run_base_node, run_react_node

        # Build per-node GenerationConfig from NodePolicy if present.
        gen_config = None
        if node.policy and (node.policy.max_new_tokens or node.policy.temperature):
            from neurosurfer.llm.types import GenerationConfig
            gen_config = GenerationConfig(
                max_tokens=node.policy.max_new_tokens or self.provider.capabilities.max_output_tokens,
                temperature=node.policy.temperature,
            )

        started_at = time.time()

        def _execute() -> Any:
            if node.kind == "react":
                pool = (
                    self.native_tools.select(node.tools)
                    if self.native_tools and node.tools
                    else (self.native_tools or ToolPool([]))
                )
                tool_ctx = self._tool_ctx
                if tool_ctx is None:
                    from pathlib import Path

                    from neurosurfer.tools.base import ToolContext

                    from .node_runner import _HeadlessIO
                    tool_ctx = ToolContext(cwd=Path.cwd(), io=_HeadlessIO())
                return run_react_node(
                    self.provider, pool, tool_ctx, system_prompt, user_prompt,
                    gen_config=gen_config,
                )
            else:
                return run_base_node(
                    self.provider, system_prompt, user_prompt,
                    output_schema=output_schema,
                    gen_config=gen_config,
                )

        try:
            if timeout_s is not None:
                with ThreadPoolExecutor(max_workers=1) as _pool:
                    # Carry the ambient context (observability TraceContext) into the
                    # timeout worker so the node agent still nests under the workflow.
                    _fut = _pool.submit(copy_context().run, _execute)
                    try:
                        raw = _fut.result(timeout=timeout_s)
                    except FuturesTimeout:
                        duration_ms = int((time.time() - started_at) * 1000)
                        self.logger.warning("Node %s timed out after %ss", node.id, timeout_s)
                        return NodeExecutionResult(
                            node_id=node.id,
                            mode=node.mode,
                            raw_output=None,
                            started_at=started_at,
                            duration_ms=duration_ms,
                            error=f"Timeout after {timeout_s}s",
                        )
            else:
                raw = _execute()

            # Structured output: native stack returns a Pydantic model directly.
            structured = raw if (output_schema and isinstance(raw, output_schema)) else None
            duration_ms = int((time.time() - started_at) * 1000)
            result = NodeExecutionResult(
                node_id=node.id,
                mode=node.mode,
                raw_output=raw,
                structured_output=structured,
                started_at=started_at,
                duration_ms=duration_ms,
            )
            self.artifacts.put(node.id, raw)
            if self.exporter and node.export:
                self.exporter.export_single_node(node=node, result=result)
            return result

        except Exception as e:
            duration_ms = int((time.time() - started_at) * 1000)
            self.logger.exception("Node %s failed: %s", node.id, e)
            return NodeExecutionResult(
                node_id=node.id,
                mode=node.mode,
                raw_output=None,
                started_at=started_at,
                duration_ms=duration_ms,
                error=str(e),
            )

    def _build_system_prompt(self, node: GraphNode, graph_inputs: Dict[str, Any]) -> str:
        """
        Build the system prompt for a node, interpolating graph-level
        inputs into purpose/goal/expected_result using `{name}` syntax.

        Example:
            purpose: "Perform research on {company_title}."
        """
        def tmpl(text: Optional[str]) -> str:
            if not text:
                return ""
            try:
                return text.format(**graph_inputs)
            except Exception as e:
                self.logger.warning(
                    "Failed to format node %s template %r with graph inputs: %s",
                    node.id,
                    text,
                    e,
                )
                return text

        purpose = tmpl(node.purpose or node.description or f"Node {node.id}")
        goal = tmpl(node.goal or "Follow the instructions in the user prompt.")
        expected = tmpl(node.expected_result or "A useful, correct, and concise answer.")
        return DEFAULT_NODE_SYSTEM_TEMPLATE.format(
            purpose=purpose,
            goal=goal,
            expected_result=expected,
        )

    def _load_output_schema_if_needed(self, node: GraphNode) -> Optional[type[PydModel]]:
        if not node.output_schema:
            return None

        obj = import_string(node.output_schema)
        if not isinstance(obj, type) or not issubclass(obj, PydModel):
            raise GraphConfigurationError(
                f"output_schema {node.output_schema!r} is not a Pydantic model"
            )
        return obj

    def _select_final_outputs(
        self, results: Dict[str, NodeExecutionResult]
    ) -> Dict[str, Any]:
        """
        Pick which node outputs are considered "final" for the graph.

        If `graph.outputs` is empty, use the last node in the topological order.
        """
        if self.graph.outputs:
            return {
                nid: results[nid].raw_output
                for nid in self.graph.outputs
                if nid in results
            }

        if not self._order:
            return {}
        last_nid = self._order[-1]
        if last_nid not in results:
            return {}
        return {last_nid: results[last_nid].raw_output}

    def _log(self, message: str, tracer: Optional[TraceStepContext] = None, type: str = "info") -> None:
        if tracer:
            tracer.log(message=message, type=type)
        else:
            self.logger.info(message)


# ── Module-level helpers ────────────────────────────────────────────────────────

def _topo_layers(nodes) -> List[List[str]]:
    """Group nodes into topological execution layers.

    All nodes in the same layer have their dependencies satisfied by earlier layers
    and can therefore run in parallel.  This is the approach used by LangChain's
    ``RunnableParallel`` and Apache Airflow's task-group scheduling.

    Example (diamond graph A→B, A→C, B→D, C→D):
        Layer 0: [A]
        Layer 1: [B, C]   ← can run in parallel
        Layer 2: [D]
    """
    if not nodes:
        return []

    node_ids = {n.id for n in nodes}
    deps: Dict[str, Set[str]] = {n.id: set(n.depends_on) & node_ids for n in nodes}
    remaining = dict(deps)
    layers: List[List[str]] = []
    completed: Set[str] = set()

    while remaining:
        # Nodes whose all deps are already in completed layers.
        ready = [nid for nid, ds in remaining.items() if ds <= completed]
        if not ready:
            # Cycle or unresolvable — fall back to serial (topo_sort will catch the cycle).
            ready = list(remaining.keys())
        layers.append(ready)
        completed.update(ready)
        for nid in ready:
            del remaining[nid]

    return layers