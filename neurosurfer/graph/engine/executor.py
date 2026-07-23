from __future__ import annotations

import logging
import time
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout
from contextvars import copy_context
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel as PydModel

# Native-stack (R3+R4)
from neurosurfer.llm.base import Provider
from neurosurfer.observability.run import traced_run
from neurosurfer.tools.base import ToolContext, ToolPool
from neurosurfer.tracing import Tracer, TraceStepContext

if TYPE_CHECKING:
    pass

from .artifacts import ArtifactStore
from .errors import (
    GraphConfigurationError,
    GraphExecutionError,
)
from .export import GraphExporter
from .manager import ManagerAgent, ManagerConfig
from .schema import Graph, GraphExecutionResult, GraphNode, NodeExecutionResult
from .state import WorkflowState
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
        provider: Provider | None = None,
        native_tools: ToolPool | None = None,
        tool_ctx: ToolContext | None = None,
        # Legacy params — accepted but ignored so old call-sites don't crash.
        llm: Any = None,
        toolkit: Any = None,
        manager_llm: Any = None,
        rag_agent: Any | None = None,
        manager_config: ManagerConfig | None = None,
        exporter: GraphExporter | None = None,
        tracer: Tracer | None = None,
        artifact_store: ArtifactStore | None = None,
        logger: logging.Logger | None = None,
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

        self._node_map: dict[str, GraphNode] = self.graph.node_map()
        self._order = topo_sort(self.graph.nodes)
        self._layers: list[list[str]] = _topo_layers(self.graph.nodes)

        self._validate_tools()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    @staticmethod
    def _topo_layers_static(nodes) -> list[list[str]]:
        return _topo_layers(nodes)

    def run(
        self,
        inputs: Any,
        *,
        manager_temperature: float = None,
        manager_max_new_tokens: int = None,
        trace_step: TraceStepContext | None = None,
        node_event: Any | None = None,
        seed_state: WorkflowState | None = None,
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
        # Typed shared state threaded through the whole run (Phase 1a). Node outputs
        # and explicit `writes` land here so conditional edges / routers / loops can
        # read them via the expression evaluator. A `seed_state` (passed by loop/map
        # body execution) pre-populates prior node outputs / vars / iteration scope.
        if seed_state is not None:
            state = WorkflowState(
                inputs=dict(graph_inputs),
                nodes=dict(seed_state.nodes),
                vars=dict(seed_state.vars),
                scope=dict(seed_state.scope),
            )
        else:
            state = WorkflowState(inputs=dict(graph_inputs))
        nodes_results: dict[str, NodeExecutionResult] = {}
        # Nodes that errored (or were skipped because an upstream errored). These
        # propagate AND-skip semantics to dependents (an error taints the branch).
        failed_ids: set[str] = set()
        # Nodes deliberately not taken — a false `when` guard, a router not selecting
        # them, or all incoming branches pruned. Distinct from `failed_ids`: a join
        # node still runs if *any* incoming branch is live (OR-join).
        pruned_ids: set[str] = set()

        def _emit(node_id: str, status: str) -> None:
            """Fire the optional per-node lifecycle callback, ignoring callback errors."""
            if node_event is None:
                return
            try:
                node_event(node_id, status)
            except Exception:  # noqa: BLE001 - progress UI must never break execution
                pass

        def _prune(nid: str, reason: str) -> None:
            node = self._node_map[nid]
            nodes_results[nid] = NodeExecutionResult(
                node_id=nid, mode=node.mode, raw_output=None,
                started_at=time.time(), duration_ms=0,
                skipped=True, skip_reason=reason,
            )
            pruned_ids.add(nid)
            _emit(nid, "skipped")
            self._log(f"Node {nid} pruned ({reason})", tracer=trace_step, type="info")

        def _post_run(nid: str, result: NodeExecutionResult) -> None:
            """Record a completed node: update state, propagate failure/pruning, emit."""
            nodes_results[nid] = result
            node = self._node_map[nid]
            if result.error and not result.skipped:
                # Error/fallback routing: a handled error activates the on_error branch
                # and prunes the normal successors, instead of AND-skipping dependents.
                if node.on_error:
                    state.set_var(f"{nid}__error", result.error)
                    for other in self.graph.nodes:
                        if nid in (other.depends_on or []) and other.id != node.on_error:
                            pruned_ids.add(other.id)
                    _emit(nid, "error")
                    self._log(
                        f"Node {nid} errored; routing to fallback '{node.on_error}': "
                        f"{result.error}",
                        tracer=trace_step, type="warning",
                    )
                    return
                failed_ids.add(nid)
                _emit(nid, "error")
                self._log(f"Node {nid} failed: {result.error}", tracer=trace_step, type="error")
                if self.graph.fail_fast:
                    raise GraphExecutionError(
                        f"Node '{nid}' failed (fail_fast=True): {result.error}",
                        failed_node=nid,
                    )
                return
            # Success: publish output to state (+ named variable), then handle routing.
            state.set_node_output(nid, result.raw_output)
            if node.writes:
                state.set_var(node.writes, result.raw_output)
            if node.kind == "router":
                self._apply_router_pruning(node, result, pruned_ids)
            _emit(nid, "ok")

        for layer in self._layers:
            to_run: list[str] = []
            for nid in layer:
                node = self._node_map[nid]
                deps = node.depends_on
                # 1. Upstream error → skip (AND-propagation; preserves prior behaviour).
                failed_dep = next((d for d in deps if d in failed_ids), None)
                if failed_dep is not None:
                    upstream_err = nodes_results[failed_dep].error or "unknown error"
                    nodes_results[nid] = NodeExecutionResult(
                        node_id=nid, mode=node.mode, raw_output=None,
                        started_at=time.time(), duration_ms=0,
                        error=f"Skipped: upstream node '{failed_dep}' failed: {upstream_err}",
                        skipped=True, skip_reason=f"upstream '{failed_dep}' failed",
                    )
                    failed_ids.add(nid)
                    _emit(nid, "skipped")
                    continue
                # 2. Explicitly pruned by an upstream router.
                if nid in pruned_ids:
                    _prune(nid, "not selected by router")
                    continue
                # 3. OR-join: prune only if the node has deps and EVERY dep was pruned
                #    (no live branch reached it). A single live dep keeps it alive.
                if deps and all(d in pruned_ids for d in deps):
                    _prune(nid, "no active branch reached this node")
                    continue
                # 4. Conditional-edge guard.
                if node.when:
                    from .expressions import safe_bool
                    if not safe_bool(node.when, state.namespace(), default=False):
                        _prune(nid, f"condition false: {node.when}")
                        continue
                to_run.append(nid)

            if not to_run:
                continue

            def _execute_one(nid: str) -> NodeExecutionResult:
                node = self._node_map[nid]
                dep_results = {d: state.get_node_output(d) for d in node.depends_on}
                # Find the most-recently completed live node for prev context.
                prev_result = None
                for past_id in reversed(list(nodes_results.keys())):
                    past = nodes_results[past_id]
                    if not past.skipped and past.raw_output is not None:
                        prev_result = past.raw_output
                        break
                # One trace span per node: makes non-agent (function/tool/router) nodes
                # visible and nests each node's agent under its *node* row. Pushes an
                # ambient TraceContext the node agent inherits (across threads too,
                # via the copy_context() used for parallel/timeout nodes).
                retries = node.policy.retries if (node.policy and node.policy.retries) else 0
                with traced_run(
                    f"node:{node.id}",
                    metadata={"node_id": node.id, "kind": node.kind, "mode": node.mode},
                    flush=False,
                ) as span:
                    attempt = 0
                    while True:
                        result = self._run_node(
                            node=node,
                            graph_inputs=graph_inputs,
                            dependency_results=dep_results,
                            previous_result=prev_result,
                            state=state,
                            manager_temperature=manager_temperature,
                            manager_max_new_tokens=manager_max_new_tokens,
                            trace_step=trace_step,
                        )
                        # Retry a genuinely-failed node up to policy.retries times.
                        if result.error and not result.skipped and attempt < (retries or 0):
                            attempt += 1
                            self._log(
                                f"Node {nid} retry {attempt}/{retries} after error: {result.error}",
                                tracer=trace_step, type="warning",
                            )
                            continue
                        break
                    if span is not None and result.error and not result.skipped:
                        span.error(result.error)
                    return result

            if self.parallelism == 1 or len(to_run) == 1:
                for nid in to_run:
                    _emit(nid, "start")
                    _post_run(nid, _execute_one(nid))
            else:
                for nid in to_run:
                    _emit(nid, "start")
                with ThreadPoolExecutor(max_workers=min(self.parallelism, len(to_run))) as pool:
                    # Each worker thread runs inside a *fresh* copy of the current
                    # context so the ambient observability TraceContext propagates and
                    # parallel node agents nest under the workflow trace. One snapshot
                    # per node — a Context can't be entered by two threads at once.
                    futures: dict[str, Future] = {
                        nid: pool.submit(copy_context().run, _execute_one, nid)
                        for nid in to_run
                    }
                    for nid, fut in futures.items():
                        try:
                            result = fut.result()
                        except Exception as exc:
                            node = self._node_map[nid]
                            result = NodeExecutionResult(
                                node_id=nid, mode=node.mode, raw_output=None,
                                started_at=time.time(), duration_ms=0,
                                error=f"Unexpected thread error: {exc}",
                            )
                        _post_run(nid, result)

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
        graph_inputs: dict[str, Any],
        dependency_results: dict[str, Any],
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
        graph_inputs: dict[str, Any],
        dependency_results: dict[str, Any],
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

    # ------------------------------------------------------------------ #
    # Router node (Phase 1d)
    # ------------------------------------------------------------------ #
    def _run_router_node(
        self, node: GraphNode, state: WorkflowState
    ) -> NodeExecutionResult:
        """Evaluate a router node and return the selected target node id as output.

        Two flavours:
          - **Expression router** (cases carry ``when`` predicates): the first case
            whose predicate is truthy wins; otherwise ``default``.
          - **LLM router** (cases carry only labels, node has a purpose): the model
            picks one label from the case list; that label maps to its ``to`` target.
        The selected id becomes ``raw_output`` so the scheduler can prune the
        non-selected branches.
        """
        started_at = time.time()
        cases = node.cases or []
        try:
            if not cases:
                raise GraphConfigurationError(
                    f"router node '{node.id}' has no 'cases' declared."
                )
            has_predicates = any(c.when and c.when.strip() for c in cases)
            if has_predicates:
                selected, label = self._route_by_expression(cases, node.default, state)
            else:
                selected, label = self._route_by_llm(node, cases, state)
            return NodeExecutionResult(
                node_id=node.id,
                mode=node.mode,
                raw_output=selected,
                structured_output={"selected": selected, "label": label},
                started_at=started_at,
                duration_ms=int((time.time() - started_at) * 1000),
            )
        except Exception as e:  # noqa: BLE001
            self.logger.exception("Router node %s failed: %s", node.id, e)
            return NodeExecutionResult(
                node_id=node.id,
                mode=node.mode,
                raw_output=None,
                started_at=started_at,
                duration_ms=int((time.time() - started_at) * 1000),
                error=str(e),
            )

    @staticmethod
    def _route_by_expression(cases, default, state: WorkflowState):
        from .expressions import safe_bool

        ns = state.namespace()
        for case in cases:
            # An empty/None `when` is a catch-all (always matches).
            if not case.when or not case.when.strip() or safe_bool(case.when, ns, default=False):
                return case.to, case.label
        return default, None

    def _route_by_llm(self, node: GraphNode, cases, state: WorkflowState):
        """Ask the LLM to choose exactly one route label from the case list."""
        if self.provider is None:
            raise GraphConfigurationError(
                f"LLM router '{node.id}' needs a provider but none was given."
            )
        from .node_runner import run_base_node

        labels = [(c.label or c.to) for c in cases]
        purpose = (node.purpose or node.goal or f"Route node {node.id}").strip()
        # Compact, JSON-safe state context so the classifier can decide.
        import json as _json
        context = _json.dumps(state.snapshot(), ensure_ascii=False)[:4000]
        system = (
            "You are a routing classifier inside a workflow engine. Read the context "
            "and choose EXACTLY ONE route from the allowed list. Reply with only the "
            "route label, nothing else."
        )
        user = (
            f"Routing decision: {purpose}\n\n"
            f"Allowed routes: {labels}\n\n"
            f"Workflow state:\n{context}\n\n"
            f"Answer with exactly one of: {labels}"
        )
        raw = run_base_node(self.provider, system, user)
        answer = str(raw or "").strip().lower()
        # Match the answer to a case by label/to (exact, then substring).
        for case in cases:
            key = (case.label or case.to).lower()
            if key == answer:
                return case.to, case.label
        for case in cases:
            key = (case.label or case.to).lower()
            if key and key in answer:
                return case.to, case.label
        # No confident match → default (or first case as a last resort).
        return (node.default or cases[0].to), None

    @staticmethod
    def _apply_router_pruning(
        node: GraphNode, result: NodeExecutionResult, pruned_ids: set[str]
    ) -> None:
        """Mark every router-controlled target except the selected one as pruned."""
        selected = result.raw_output
        controlled: set[str] = {c.to for c in (node.cases or [])}
        if node.default:
            controlled.add(node.default)
        for target in controlled:
            if target != selected:
                pruned_ids.add(target)

    # ------------------------------------------------------------------ #
    # Iteration nodes (Phase 1e loop / 1f map)
    # ------------------------------------------------------------------ #
    def _child_executor(self, node: GraphNode) -> GraphExecutor:
        """Build a nested executor for a loop/map ``body`` sub-graph."""
        body_graph = Graph(
            name=f"{node.id}__body",
            nodes=node.body or [],
            outputs=list(node.body_outputs or []),
        )
        return GraphExecutor(
            body_graph,
            provider=self.provider,
            native_tools=self.native_tools,
            tool_ctx=self._tool_ctx,
            exporter=self.exporter,
            tracer=self.tracer,
            log_traces=self.log_traces,
            parallelism=self.parallelism,
        )

    @staticmethod
    def _body_value(result: GraphExecutionResult) -> Any:
        """Reduce a body run's final outputs to a single value (unwrap 1-key dicts)."""
        final = result.final or {}
        if len(final) == 1:
            return next(iter(final.values()))
        return final

    def _run_loop_node(self, node: GraphNode, state: WorkflowState) -> NodeExecutionResult:
        """Run ``node.body`` repeatedly until ``break_when`` or ``max_iterations``.

        Each iteration sees ``index`` and the previous output (bound to ``item_var``)
        in its scope; body node outputs are published back to the parent state so the
        break predicate and downstream nodes can read them.
        """
        from .expressions import safe_bool

        started_at = time.time()
        try:
            if not node.body:
                raise GraphConfigurationError(f"loop node '{node.id}' has no body.")
            if not node.max_iterations or node.max_iterations < 1:
                raise GraphConfigurationError(
                    f"loop node '{node.id}' requires max_iterations >= 1 (a hard ceiling)."
                )
            child = self._child_executor(node)
            acc: list[Any] = []
            last_output: Any = None
            iterations = 0
            broke = False
            for i in range(node.max_iterations):
                iterations = i + 1
                scope = {"index": i, "iteration": i, node.item_var: last_output, "acc": list(acc)}
                child_state = state.child_scope(scope)
                iter_inputs = {**state.inputs, "index": i, node.item_var: last_output}
                body_result = child.run(iter_inputs, seed_state=child_state)
                # Publish body outputs to the parent state (readable by break_when).
                for nid, r in body_result.nodes.items():
                    if not r.skipped and r.error is None:
                        state.set_node_output(nid, r.raw_output)
                last_output = self._body_value(body_result)
                acc.append(last_output)
                if node.accumulate:
                    state.set_var(node.accumulate, list(acc))
                if body_result.errors:
                    # A failing body stops the loop (surface it below).
                    return self._error_result(
                        node, started_at,
                        f"loop body failed on iteration {iterations}: {body_result.errors}",
                    )
                if node.break_when:
                    ns = state.child_scope(
                        {"index": i, "iteration": i, node.item_var: last_output, "acc": list(acc)}
                    ).namespace()
                    if safe_bool(node.break_when, ns, default=False):
                        broke = True
                        break
            result_value = state.vars.get(node.accumulate) if node.accumulate else last_output
            return NodeExecutionResult(
                node_id=node.id,
                mode=node.mode,
                raw_output=result_value,
                structured_output={"iterations": iterations, "broke_early": broke, "results": acc},
                started_at=started_at,
                duration_ms=int((time.time() - started_at) * 1000),
            )
        except Exception as e:  # noqa: BLE001
            self.logger.exception("Loop node %s failed: %s", node.id, e)
            return self._error_result(node, started_at, str(e))

    def _run_map_node(self, node: GraphNode, state: WorkflowState) -> NodeExecutionResult:
        """Fan ``node.body`` out over the collection from the ``over`` expression.

        Returns the list of per-item body outputs (implicit gather); a downstream
        node depending on this map receives that list.
        """
        from .expressions import ExpressionError, evaluate

        started_at = time.time()
        try:
            if not node.body:
                raise GraphConfigurationError(f"map node '{node.id}' has no body.")
            if not node.over:
                raise GraphConfigurationError(f"map node '{node.id}' requires an 'over' expression.")
            try:
                collection = evaluate(node.over, state.namespace())
            except ExpressionError as e:
                raise GraphConfigurationError(
                    f"map '{node.id}' over-expression {node.over!r} failed: {e}"
                ) from e
            if collection is None:
                collection = []
            if not isinstance(collection, (list, tuple)):
                raise GraphConfigurationError(
                    f"map '{node.id}' over-expression must yield a list, got "
                    f"{type(collection).__name__}."
                )
            items = list(collection)
            child = self._child_executor(node)
            results: list[Any] = [None] * len(items)

            def _run_item(i: int) -> Any:
                scope = {"index": i, node.item_var: items[i]}
                child_state = state.child_scope(scope)
                iter_inputs = {**state.inputs, "index": i, node.item_var: items[i]}
                body_result = child.run(iter_inputs, seed_state=child_state)
                if body_result.errors:
                    raise GraphExecutionError(
                        f"map body failed for item {i}: {body_result.errors}"
                    )
                return self._body_value(body_result)

            if node.concurrency > 1 and len(items) > 1:
                with ThreadPoolExecutor(max_workers=min(node.concurrency, len(items))) as pool:
                    futures = {
                        pool.submit(copy_context().run, _run_item, i): i
                        for i in range(len(items))
                    }
                    for fut in futures:
                        idx = futures[fut]
                        results[idx] = fut.result()
            else:
                for i in range(len(items)):
                    results[i] = _run_item(i)

            return NodeExecutionResult(
                node_id=node.id,
                mode=node.mode,
                raw_output=results,
                structured_output={"count": len(items)},
                started_at=started_at,
                duration_ms=int((time.time() - started_at) * 1000),
            )
        except Exception as e:  # noqa: BLE001
            self.logger.exception("Map node %s failed: %s", node.id, e)
            return self._error_result(node, started_at, str(e))

    def _error_result(self, node: GraphNode, started_at: float, message: str) -> NodeExecutionResult:
        return NodeExecutionResult(
            node_id=node.id,
            mode=node.mode,
            raw_output=None,
            started_at=started_at,
            duration_ms=int((time.time() - started_at) * 1000),
            error=message,
        )

    # ------------------------------------------------------------------ #
    # Sub-workflow (Phase 1h) + human-in-the-loop (Phase 1i)
    # ------------------------------------------------------------------ #
    def _run_subgraph_node(self, node: GraphNode, state: WorkflowState) -> NodeExecutionResult:
        """Run ``node.body`` once as a nested sub-graph (composition).

        The body sees the parent inputs/state; its final outputs become this node's
        output (unwrapped when there's a single output).
        """
        started_at = time.time()
        try:
            if not node.body:
                raise GraphConfigurationError(f"subgraph node '{node.id}' has no body.")
            child = self._child_executor(node)
            body_result = child.run(dict(state.inputs), seed_state=state.child_scope({}))
            if body_result.errors:
                return self._error_result(
                    node, started_at, f"subgraph body failed: {body_result.errors}"
                )
            return NodeExecutionResult(
                node_id=node.id,
                mode=node.mode,
                raw_output=self._body_value(body_result),
                structured_output={"body_nodes": list(body_result.nodes.keys())},
                started_at=started_at,
                duration_ms=int((time.time() - started_at) * 1000),
            )
        except Exception as e:  # noqa: BLE001
            self.logger.exception("Subgraph node %s failed: %s", node.id, e)
            return self._error_result(node, started_at, str(e))

    def _run_input_node(self, node: GraphNode, state: WorkflowState) -> NodeExecutionResult:
        """Human-in-the-loop node: obtain a value from the user (or a supplied answer).

        Resolution order:
          1. A pre-supplied value (``state.inputs[key]`` / ``state.vars[key]``) — this
             is the **resume** path a durable API run uses (Phase 2).
          2. An interactive ask through the ToolContext IO handler (CLI).
          3. Otherwise an error signalling the run is awaiting input.
        The answer key is ``writes`` if set, else the node id.
        """
        from neurosurfer.tools.base import AutoApproveIOHandler

        started_at = time.time()
        key = node.writes or node.id
        supplied = state.inputs.get(key)
        if supplied is None:
            supplied = state.vars.get(key)
        if supplied is not None:
            return NodeExecutionResult(
                node_id=node.id, mode=node.mode, raw_output=supplied,
                structured_output={"source": "supplied"},
                started_at=started_at, duration_ms=int((time.time() - started_at) * 1000),
            )

        io = self._tool_ctx.io if self._tool_ctx else None
        question = (node.purpose or node.goal or f"Input needed for '{node.id}'").strip()
        # Headless auto-approvers are non-interactive — don't fabricate an answer.
        if io is not None and not isinstance(io, AutoApproveIOHandler):
            from .node_runner import run_coro_blocking
            try:
                answer = run_coro_blocking(io.ask(question, node.options or None))
            except Exception as e:  # noqa: BLE001
                return self._error_result(node, started_at, f"input ask failed: {e}")
            return NodeExecutionResult(
                node_id=node.id, mode=node.mode, raw_output=answer,
                structured_output={"source": "interactive"},
                started_at=started_at, duration_ms=int((time.time() - started_at) * 1000),
            )

        return self._error_result(
            node, started_at,
            f"input node '{node.id}' is awaiting a value — supply '{key}' as an input "
            f"to resume this run.",
        )

    def _run_node(
        self,
        *,
        node: GraphNode,
        graph_inputs: dict[str, Any],
        dependency_results: dict[str, Any],
        previous_result: Any,
        state: WorkflowState | None = None,
        manager_temperature: float,
        manager_max_new_tokens: int,
        trace_step: TraceStepContext | None = None,
    ) -> NodeExecutionResult:

        # Non-LLM dispatch — no prompt building, no agent needed
        if node.kind in {"function", "python"}:
            return self._run_function_node(node, graph_inputs, dependency_results)
        if node.kind == "tool":
            return self._run_tool_node(node, graph_inputs, dependency_results)
        _state = state or WorkflowState(inputs=dict(graph_inputs))
        if node.kind == "router":
            return self._run_router_node(node, _state)
        if node.kind == "loop":
            return self._run_loop_node(node, _state)
        if node.kind == "map":
            return self._run_map_node(node, _state)
        if node.kind == "subgraph":
            return self._run_subgraph_node(node, _state)
        if node.kind == "input":
            return self._run_input_node(node, _state)

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
        output_schema: type | None = None,
        timeout_s: float | None = None,
    ) -> NodeExecutionResult:
        """Execute a base or react node using the native provider + ToolPool stack."""
        from concurrent.futures import ThreadPoolExecutor

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

    def _build_system_prompt(self, node: GraphNode, graph_inputs: dict[str, Any]) -> str:
        """
        Build the system prompt for a node, interpolating graph-level
        inputs into purpose/goal/expected_result using `{name}` syntax.

        Example:
            purpose: "Perform research on {company_title}."
        """
        def tmpl(text: str | None) -> str:
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

    def _load_output_schema_if_needed(self, node: GraphNode) -> type[PydModel] | None:
        if not node.output_schema:
            return None

        obj = import_string(node.output_schema)
        if not isinstance(obj, type) or not issubclass(obj, PydModel):
            raise GraphConfigurationError(
                f"output_schema {node.output_schema!r} is not a Pydantic model"
            )
        return obj

    def _select_final_outputs(
        self, results: dict[str, NodeExecutionResult]
    ) -> dict[str, Any]:
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

    def _log(self, message: str, tracer: TraceStepContext | None = None, type: str = "info") -> None:
        if tracer:
            tracer.log(message=message, type=type)
        else:
            self.logger.info(message)


# ── Module-level helpers ────────────────────────────────────────────────────────

def _topo_layers(nodes) -> list[list[str]]:
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
    deps: dict[str, set[str]] = {n.id: set(n.depends_on) & node_ids for n in nodes}
    remaining = dict(deps)
    layers: list[list[str]] = []
    completed: set[str] = set()

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
