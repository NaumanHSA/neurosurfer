from __future__ import annotations

import inspect
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

from ..artifacts import ArtifactStore
from ..errors import (
    GraphConfigurationError,
    GraphExecutionError,
)
from ..export import GraphExporter
from ..json_schema import JsonSchemaError, model_from_json_schema
from ..manager import ManagerAgent, ManagerConfig
from ..nodes import (
    Function,
    Input,
    Loop,
    Map,
    Output,
    Python,
    React,
    Router,
    Subgraph,
    Tool,
)
from ..schema import Graph, GraphExecutionResult, GraphNode, NodeExecutionResult
from ..state import WorkflowState
from ..templates import (
    DEFAULT_NODE_SYSTEM_TEMPLATE,
    NODE_SYSTEM_TEMPLATE,
    recited_names,
    render_scope,
    render_template,
)
from ..utils import import_string, normalize_and_validate_graph_inputs, topo_sort
from . import deterministic, io_nodes, iteration, routing
from ._trace import _trace_step, _trace_text


def _input_root(expression: str | None) -> str | None:
    """The graph input a `map`'s `over` reads, if it reads one.

    `inputs.reviews` and a bare `reviews` both name an input; `nodes.fetch` and
    `vars.acc` name something else and yield nothing to hide. Anything with an
    index or a call in it (`inputs.a[0]`, `len(x)`) is left alone rather than
    guessed at — the cost of missing one is a prompt that is merely verbose,
    and the cost of guessing wrong is a body that cannot see an input it needs.
    """
    text = (expression or "").strip()
    if not text:
        return None
    head, _, rest = text.partition(".")
    if head in {"nodes", "vars", "state"}:
        return None
    name = rest if head == "inputs" else text
    return name if name.isidentifier() else None


def _accepts_scope(callback: Any) -> bool:
    """True if *callback* can take a third ``scope`` argument.

    The node-event callback was historically ``(node_id, status)``. Nested body
    events carry a scope dict as a third argument, so older two-argument callbacks
    (CLI progress bars, tests) keep working unchanged — they simply never see the
    nested events' context.
    """
    try:
        sig = inspect.signature(callback)
    except (TypeError, ValueError):  # builtins / C callables — assume the old shape
        return False
    positional = 0
    for p in sig.parameters.values():
        if p.kind is p.VAR_POSITIONAL:
            return True
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD):
            positional += 1
    return positional >= 3


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
        provider_resolver: Any = None,
        validate: bool = True,
    ) -> None:
        self.graph = graph
        # Validation is the first step of every run — see `_validate_before_running`.
        # The escape hatch exists for the validator's own tests and for
        # deliberately running a graph you know is broken; it is not a
        # performance switch, and nothing in the library sets it.
        self.validate = validate
        self.provider = provider
        # Per-node provider selection. Without one, every node runs on `provider`
        # and `node.model` rebinds a copy of it — so a graph that names a model
        # finally gets that model rather than a trace that merely claims it.
        if provider_resolver is None:
            from neurosurfer.llm.resolver import ProviderResolver
            provider_resolver = ProviderResolver(provider)

        self.providers = provider_resolver
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

        # Set for the duration of run(); nested body executors inherit it so their
        # node events reach the same consumer.
        self._active_node_event: Any | None = None

        # Graph inputs this run resolves but does **not** recite to a model. Set
        # by `_child_executor` for a container body; empty for a top-level run.
        # See `_hidden_body_inputs` for what earns a place here and why.
        self._hidden_inputs: frozenset[str] = frozenset()

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

    def _validate_before_running(self) -> None:
        """Refuse a graph the validator says cannot run. **The first step of a run.**

        Validation used to be an *optional* gate that only some callers passed
        through: the Architect ran it, the registry ran it, and a graph handed
        straight to `GraphExecutor` — which is what the Python API and every
        tutorial does — ran no checks at all.

        The cost of that was demonstrated by renaming one node: a router still
        routing to the old name selected a branch that did not exist, a node that
        depended on the router but was nobody's target ran unconditionally, and
        the whole thing reported success. Every fact needed to refuse it was
        available before the first model call.

        **Errors block; warnings do not.** An error means the graph will not run
        correctly, so starting it only spends tokens on the way to a worse
        message. A warning means it will run and may surprise you, which is the
        author's call and not the engine's.

        Validation failing to *run* — no tool registry, an import this
        deployment cannot resolve — is logged and skipped rather than raised.
        Refusing to execute because the checker itself broke would make the gate
        more fragile than the thing it guards.
        """
        if not self.validate:
            return
        try:
            from pathlib import Path

            from neurosurfer.graph.workflow.package import WorkflowPackage
            from neurosurfer.graph.workflow.schema import WorkflowManifest
            from neurosurfer.graph.workflow.validation import validate_package

            pkg = getattr(self, "_package", None) or WorkflowPackage(
                manifest=WorkflowManifest(name=self.graph.name or "graph"),
                graph=self.graph,
                path=Path("."),
            )
            # Tools the executor was *given* are as real as registered ones for
            # this run — see `validate_package`'s `extra_tools`.
            pool = self.native_tools
            given = set(pool.names()) if pool is not None else set()
            report = validate_package(pkg, extra_tools=given)
        except Exception as e:  # noqa: BLE001 - see the docstring
            self.logger.debug("Pre-run validation could not run: %s", e)
            return

        for issue in report.warnings:
            self.logger.warning("%s: %s", issue.node_id or self.graph.name, issue.message)

        blocking = report.errors + report.gaps
        if blocking:
            lines = "\n".join(f"  - {i.render()}" for i in blocking)
            raise GraphConfigurationError(
                f"This workflow cannot run as written:\n{lines}\n\n"
                f"Validation runs before every graph, so this was caught without "
                f"spending a model call. Pass `validate=False` to the executor to "
                f"run it anyway."
            )

    def run(
        self,
        inputs: Any,
        *,
        manager_temperature: float = None,
        manager_max_new_tokens: int = None,
        trace_step: TraceStepContext | None = None,
        node_event: Any | None = None,
        event_scope: dict[str, Any] | None = None,
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
        self._validate_before_running()
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
        # `on_error` targets that a real failure activated. A handler is pruned
        # when its guarded node succeeds, but several nodes may share one handler,
        # so an activation by any of them outranks a prune by the others —
        # regardless of the order the two complete in.
        error_handled: set[str] = set()

        # Nested bodies (loop/map/subgraph) run on a child executor that is handed
        # this same callback plus a scope describing where it sits — so a UI can
        # show per-iteration progress instead of one opaque container node.
        self._active_node_event = node_event
        emit_scope = _accepts_scope(node_event) if node_event is not None else False

        def _emit(node_id: str, status: str) -> None:
            """Fire the optional per-node lifecycle callback, ignoring callback errors."""
            if node_event is None:
                return
            if event_scope and not emit_scope:
                # We're inside a body and the consumer takes only (node_id, status):
                # it has no way to tell a body node from a top-level one, and would
                # file `step` alongside real graph nodes. Stay silent for it.
                return
            try:
                if emit_scope:
                    node_event(node_id, status, event_scope)
                else:
                    node_event(node_id, status)
            except Exception:  # noqa: BLE001 - progress UI must never break execution
                pass

        def _prune(nid: str, reason: str, *, quiet: bool = False) -> None:
            node = self._node_map[nid]
            nodes_results[nid] = NodeExecutionResult(
                node_id=nid, mode=node.mode, raw_output=None,
                started_at=time.time(), duration_ms=0,
                skipped=True, skip_reason=reason,
            )
            pruned_ids.add(nid)
            _emit(nid, "skipped")
            if quiet:
                # Router non-selection is already announced positively by the
                # "routing to" line — per-target prune lines would just be noise.
                self.logger.debug("Node %s pruned (%s)", nid, reason)
            else:
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
                    error_handled.add(node.on_error)
                    # Another node guarded by this same handler may have already
                    # pruned it by succeeding; a real error wins.
                    pruned_ids.discard(node.on_error)
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
            # Success: the error branch is not taken. Without this the handler is
            # just an ordinary dependent whose dependency was satisfied, so every
            # workflow with a fallback ran its fallback on the happy path.
            if node.on_error and node.on_error not in error_handled:
                pruned_ids.add(node.on_error)
            # Success: publish output to state (+ named variable), then handle routing.
            state.set_node_output(nid, result.raw_output)
            if node.writes:
                state.set_var(node.writes, result.raw_output)
            if isinstance(node, Router):
                self._apply_router_pruning(node, result, pruned_ids)
                selected = result.raw_output
                label = (result.structured_output or {}).get("label")
                if selected is None:
                    msg = f"Node {nid}: no route matched — all targets pruned"
                elif label and str(label) != str(selected):
                    msg = f"Node {nid}: routing to '{selected}' (label: {label})"
                else:
                    msg = f"Node {nid}: routing to '{selected}'"
                self._log(msg, tracer=trace_step, type="info")
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
                    reason = (
                        "error handler not needed — the guarded node succeeded"
                        if any(n.on_error == nid for n in self.graph.nodes)
                        else "not selected by router"
                    )
                    _prune(nid, reason, quiet=True)
                    continue
                # 3. OR-join: prune only if the node has deps and EVERY dep was pruned
                #    (no live branch reached it). A single live dep keeps it alive.
                if deps and all(d in pruned_ids for d in deps):
                    _prune(nid, "no active branch reached this node")
                    continue
                # 4. Switched off by hand. Same outcome as a false guard — the
                #    node is *not taken*, which is a normal branch and not an
                #    error, so dependents still run if another branch is live.
                #    Checked before `when` because it is unconditional: a
                #    disabled node should not evaluate an expression that could
                #    itself fail.
                if node.disabled:
                    _prune(nid, "disabled")
                    continue
                # 5. Conditional-edge guard.
                if node.when:
                    from ..expressions import safe_bool
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
                # The span carries the node's real I/O so the trace UI shows what
                # went in (graph inputs + upstream outputs) and what came out —
                # not just on the nested agent generation.
                from ..state import _jsonable

                span_input: dict[str, Any] = {"graph_inputs": _jsonable(graph_inputs)}
                if dep_results:
                    span_input["dependencies"] = _jsonable(dep_results)
                with traced_run(
                    f"node:{node.id}",
                    metadata={"node_id": node.id, "kind": node.kind, "mode": node.mode},
                    input=span_input,
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
                    if span is not None:
                        if result.error and not result.skipped:
                            span.error(result.error)
                        else:
                            span.output = _jsonable(result.raw_output)
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
    # ── deterministic kinds: function / python / tool ────────────────────
    #
    # Thin forwarders; the runners live in `deterministic.py`.

    def _run_function_node(
        self,
        node: GraphNode,
        graph_inputs: dict[str, Any],
        dependency_results: dict[str, Any],
        state: WorkflowState | None = None,
    ) -> NodeExecutionResult:
        return deterministic.run_function_node(
            self, node, graph_inputs, dependency_results, state
        )

    def _run_tool_node(
        self,
        node: GraphNode,
        graph_inputs: dict[str, Any],
        dependency_results: dict[str, Any],
        state: WorkflowState | None = None,
    ) -> NodeExecutionResult:
        return deterministic.run_tool_node(
            self, node, graph_inputs, dependency_results, state
        )

    # ------------------------------------------------------------------ #
    # Router node (Phase 1d)
    # ------------------------------------------------------------------ #
    # ── routing ──────────────────────────────────────────────────────────
    #
    # Thin forwarders; the runners live in `routing.py`.

    def _run_router_node(
        self, node: GraphNode, state: WorkflowState
    ) -> NodeExecutionResult:
        return routing.run_router_node(self, node, state)

    @staticmethod
    def _route_by_expression(cases, default, state: WorkflowState):
        return routing.route_by_expression(cases, default, state)

    @staticmethod
    def _apply_router_pruning(
        node: GraphNode, result: NodeExecutionResult, pruned_ids: set[str]
    ) -> None:
        routing.apply_router_pruning(node, result, pruned_ids)

    # ------------------------------------------------------------------ #
    # Iteration nodes (Phase 1e loop / 1f map)
    # ------------------------------------------------------------------ #
    def _child_executor(self, node: GraphNode) -> GraphExecutor:
        """Build a nested executor for a loop/map ``body`` sub-graph.

        **The body is not re-validated**, and that is not an oversight. A body is
        not a standalone workflow: `{item}`, `{index}` and `{iteration}` are
        injected by the container at run time and are *not* graph inputs, so
        validating the body in isolation reports them as references nothing
        provides — and a perfectly good `map` fails before its first iteration.

        The parent's validation already covered these nodes: the rule runner
        walks into `body` (see `validation/context.py::body_nodes`), where it can
        see the container that supplies the loop variable.
        """
        body_graph = Graph(
            name=f"{node.id}__body",
            nodes=node.body or [],
            outputs=list(node.body_outputs or []),
        )
        child = GraphExecutor(
            body_graph,
            validate=False,
            provider=self.provider,
            native_tools=self.native_tools,
            tool_ctx=self._tool_ctx,
            exporter=self.exporter,
            tracer=self.tracer,
            log_traces=self.log_traces,
            parallelism=self.parallelism,
            # Shared, not rebuilt: body nodes may name providers too, and a fresh
            # resolver per iteration would rebuild a client on every pass.
            provider_resolver=self.providers,
        )
        # Inherited, then extended: a `map` nested inside a `loop` hides both
        # containers' plumbing, since both are still in the body's inputs.
        child._hidden_inputs = self._hidden_inputs | self._hidden_body_inputs(node)
        return child

    @staticmethod
    def _hidden_body_inputs(node: GraphNode) -> frozenset[str]:
        """Names a container puts in its body's inputs that no body node should be *told*.

        A container binds its iteration values into the body's graph inputs
        (`{**state.inputs, "index": i, item_var: …}`), and every LLM node prints
        every graph input. So a two-item `map` told each body node the item it
        was on, the index it was at, **and the entire collection both came
        from** — the same list, once per item, in every prompt. At fifty items
        that is fifty copies of fifty reviews.

        Three kinds of name are hidden, and none of them stops *resolving* —
        `{item}` renders, `inputs.index` evaluates, a function node still
        receives them as kwargs. They are only not recited:

        - **the iteration values** (`index`, the item, a loop's `feedback`) —
          already interpolated into the instruction by the author who used them,
          and meaningless plumbing to a model that did not;
        - **the collection being mapped over** — a body handles one element, and
          the list it was drawn from is the parent's business. This is the one
          that was quadratic;
        - nothing else. A shared input a body genuinely reads (`{criteria}`) is
          untouched, because narrowing to "only what this node references" would
          silently starve every workflow that leans on the inputs block instead
          of placeholders.

        A `subgraph` hides nothing: it is composition, not iteration — its body
        is meant to see the parent's inputs, and there is no per-item anything.
        """
        if isinstance(node, Subgraph):
            return frozenset()
        names = {"index", node.item_var}
        if isinstance(node, Loop):
            names.add("feedback")
        if isinstance(node, Map):
            root = _input_root(node.over)
            if root:
                names.add(root)
        return frozenset(names)

    # ── iteration: loop / map / subgraph ─────────────────────────────────
    #
    # Thin forwarders. The runners live in `iteration.py`; these keep the method
    # surface a reader (and `_run_node`'s dispatch, and the tests) already knows.

    @staticmethod
    def _body_value(result: GraphExecutionResult) -> Any:
        return iteration.body_value(result)

    def _run_loop_node(self, node: GraphNode, state: WorkflowState) -> NodeExecutionResult:
        return iteration.run_loop_node(self, node, state)

    def _run_map_node(self, node: GraphNode, state: WorkflowState) -> NodeExecutionResult:
        return iteration.run_map_node(self, node, state)

    def _run_subgraph_node(self, node: GraphNode, state: WorkflowState) -> NodeExecutionResult:
        return iteration.run_subgraph_node(self, node, state)

    def _error_result(self, node: GraphNode, started_at: float, message: str) -> NodeExecutionResult:
        return NodeExecutionResult(
            node_id=node.id,
            mode=node.mode,
            raw_output=None,
            started_at=started_at,
            duration_ms=int((time.time() - started_at) * 1000),
            error=message,
        )

    # ── the edges of a run: input / output ───────────────────────────────
    #
    # Thin forwarders; the runners live in `io_nodes.py`.

    def _run_input_node(self, node: GraphNode, state: WorkflowState) -> NodeExecutionResult:
        return io_nodes.run_input_node(self, node, state)

    def _run_output_node(
        self,
        node: GraphNode,
        graph_inputs: dict[str, Any],
        dependency_results: dict[str, Any],
        state: WorkflowState,
    ) -> NodeExecutionResult:
        return io_nodes.run_output_node(
            self, node, graph_inputs, dependency_results, state
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

        # Non-LLM dispatch — no prompt building, no agent needed.
        #
        # `isinstance` rather than `node.kind == ...`: `Graph` upgrades every node
        # to its kind's class on the way in (schema._as_kind_classes), including
        # nodes that arrived as `kind=` strings or from YAML, so the two are
        # equivalent — but the class is the thing a reader can follow to a
        # docstring, and mypy narrows it.
        _state = state or WorkflowState(inputs=dict(graph_inputs))
        if isinstance(node, (Function, Python)):
            return self._run_function_node(node, graph_inputs, dependency_results, _state)
        if isinstance(node, Tool):
            return self._run_tool_node(node, graph_inputs, dependency_results, _state)
        if isinstance(node, Router):
            return self._run_router_node(node, _state)
        if isinstance(node, Loop):
            return self._run_loop_node(node, _state)
        if isinstance(node, Map):
            return self._run_map_node(node, _state)
        if isinstance(node, Subgraph):
            return self._run_subgraph_node(node, _state)
        if isinstance(node, Input):
            return self._run_input_node(node, _state)
        if isinstance(node, Output):
            return self._run_output_node(node, graph_inputs, dependency_results, _state)

        # LLM-based node (base | react)
        #
        # The system prompt is built **first**, because what it says decides what
        # the user prompt should not repeat.
        #
        # Interpolate templates over graph inputs *and* upstream state — a node's
        # `goal`/`purpose` commonly references an upstream node's output by its
        # `writes` name (e.g. "…based on the summary: {summary}"). `writes` vars are
        # in `_state.vars`; dependency outputs are also exposed by node id. Explicit
        # `writes` take precedence over a same-named graph input.
        interp_scope = render_scope(
            graph_inputs,
            nodes=dependency_results,
            variables=_state.vars,
            scope=_state.scope,
        )
        system_prompt, recited = self._build_system_prompt(node, interp_scope)
        user_prompt = self.manager.compose_user_prompt(
            node=node,
            graph_inputs=graph_inputs,
            dependency_results=dependency_results,
            previous_result=previous_result,
            temperature=manager_temperature,
            max_new_tokens=manager_max_new_tokens,
            hidden=self._hidden_inputs | recited,
        )
        output_schema = self._load_output_schema_if_needed(node)
        timeout_s = node.policy.timeout_s if node.policy and node.policy.timeout_s else None

        if self.provider is None and node.provider is None:
            raise GraphConfigurationError(
                f"Node '{node.id}' is a base/react node but no provider was given to the executor."
            )
        return self._run_node_native(
            node=node,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            output_schema=output_schema,
            timeout_s=timeout_s,
            provider=self._provider_for(node),
            # For a react node's bound `tool_args`; the same scope its
            # prompts are rendered against.
            scope=interp_scope,
        )

    def _provider_for(self, node: GraphNode) -> Provider:
        """The client this node runs on — its own, or the run's default.

        A bad name fails here, before the call, naming the alternatives; the same
        treatment an unresolvable tool gets, and for the same reason: the alternative
        is a confusing provider-side error halfway through a paid run.
        """
        from neurosurfer.llm.resolver import UnknownProviderError

        try:
            return self.providers.resolve(node.provider, node.model)
        except UnknownProviderError as e:
            raise GraphConfigurationError(f"Node '{node.id}': {e}") from e

    def _run_node_native(
        self,
        *,
        node: GraphNode,
        system_prompt: str,
        user_prompt: str,
        output_schema: type | None = None,
        timeout_s: float | None = None,
        provider: Provider | None = None,
        scope: dict[str, Any] | None = None,
    ) -> NodeExecutionResult:
        """Execute a base or react node using the native provider + ToolPool stack."""
        from concurrent.futures import ThreadPoolExecutor

        from ..node_runner import run_base_node, run_react_node

        # The node's own client when it named one; otherwise the run's.
        provider = provider or self.provider

        # Build per-node GenerationConfig from NodePolicy if present.
        gen_config = None
        if node.policy and (node.policy.max_new_tokens or node.policy.temperature):
            from neurosurfer.llm.types import GenerationConfig
            gen_config = GenerationConfig(
                max_tokens=node.policy.max_new_tokens or provider.capabilities.max_output_tokens,
                temperature=node.policy.temperature,
            )

        started_at = time.time()

        def _agent_tools() -> tuple[ToolPool, Any]:
            """This node's tools, bound, plus the context to call them in.

            Shared by `react` and `base`, because "which tools does this node have"
            has one answer whatever it does with them.

            `tool_args` are **bound arguments**, not the whole call: values the
            engine supplies on every call the model makes, and removes from the
            schema it is offered. This is the only way an agent node can use a
            credential — a secret reaches a tool as `${NAME}` in `tool_args`.
            Rendered with the same pass a tool node uses, so `{input}` and
            `${SECRET}` behave alike.
            """
            pool = (
                self.native_tools.select(node.tools)
                if self.native_tools
                else ToolPool([])
            )
            if node.tool_args:
                from ..bound_tools import bind_pool

                pool = bind_pool(
                    pool,
                    self._render_tool_args(node, dict(node.tool_args), scope or {}),
                )
            # Settings are applied **after** binding, so a configured tool wraps a
            # bound one rather than the other way round. The order matters: the
            # root has to scope the call that actually happens, and a bound
            # argument is part of that call.
            if node.tool_settings:
                from ..configured_tools import configure_pool

                pool = configure_pool(pool, self._render_tool_settings(node, scope or {}))
            tool_ctx = self._tool_ctx
            if tool_ctx is None:
                from pathlib import Path

                from neurosurfer.tools.base import ToolContext

                from ..node_runner import _HeadlessIO
                tool_ctx = ToolContext(cwd=Path.cwd(), io=_HeadlessIO())
            return pool, tool_ctx

        def _execute() -> Any:
            if isinstance(node, React):
                # A react node is an LLM that calls tools in a loop. With none, it
                # used to fall through to an empty pool and quietly become a base
                # node that narrates actions it never took — the failure mode is a
                # confident, entirely invented answer. Refuse instead: an
                # unrunnable graph must say so, not improvise.
                if not node.tools:
                    raise GraphConfigurationError(
                        f"react node '{node.id}' has no tools — it cannot act. "
                        "Give it the tools it needs, or use kind 'base' for a "
                        "pure reasoning step."
                    )
                pool, tool_ctx = _agent_tools()
                return run_react_node(
                    provider, pool, tool_ctx, system_prompt, user_prompt,
                    gen_config=gen_config,
                )
            else:
                # **A base node uses the tools it was given.**
                #
                # It used to be handed `ToolPool([])` unconditionally, so a tool
                # attached to one was accepted by validation, drawn on the canvas,
                # and then silently dropped at run time — the worst of the three
                # outcomes, because nothing anywhere said so.
                #
                # The rung between `base` and `react` is one LLM call *with* tools:
                # the model may call them once, sees the results, and answers.
                # `OneShotAgent` already did this and the empty pool threw it away.
                # A base node with no tools is unchanged — same single call, same
                # empty pool — so this only ever adds behaviour where a tool was
                # deliberately attached.
                pool, tool_ctx = _agent_tools() if node.tools else (ToolPool([]), None)
                return run_base_node(
                    provider, system_prompt, user_prompt,
                    tool_pool=pool,
                    tool_ctx=tool_ctx,
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
                        call = _fut.result(timeout=timeout_s)
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
                with _trace_step(
                    self.tracer,
                    kind="llm",
                    label=f"node.{node.kind}",
                    node_id=node.id,
                    agent_id=node.id,
                    inputs={"system": system_prompt, "user": user_prompt},
                ) as step:
                    call = _execute()
                    if step is not None:
                        step.outputs(
                            output=_trace_text(call.output),
                            tool_calls=list(call.tool_calls),
                        )
                        step.add_meta(
                            usage=call.usage.model_dump() if call.usage else None,
                            # The model the call was made on, read off the client that
                            # made it. Reading `node.model` here is how the trace used
                            # to name models nothing had called.
                            model=getattr(provider, "model", None),
                            provider=node.provider,
                        )

            raw = call.output
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
                usage=call.usage,
                tool_calls=call.tool_calls,
                node_input=_trace_text(user_prompt),
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

    def _build_system_prompt(
        self, node: GraphNode, scope: dict[str, Any]
    ) -> tuple[str, frozenset[str]]:
        """
        Build the system prompt for a node, interpolating available scope
        (graph inputs + upstream `writes` vars + dependency outputs) using
        `{name}` syntax.

        Two shapes, and which one is used depends on the node:

        - **`instructions`** — one field saying what this step should do. This is
          what new nodes set, and it wins outright when present.
        - **`purpose` / `goal` / `expected_result`** — the three fields that came
          first. Still read, so every workflow already on disk keeps running; see
          `GraphNode.instructions` for why they were collapsed.

        Example:
            instructions: "Research {company_title} and write a title based on
                           the summary: {summary}"

        Returns the prompt **and the names it recited**, so the user prompt can
        avoid saying the same thing again. The two are returned together because
        which fields get rendered is this method's rule — `instructions` winning
        outright means a name mentioned only in `purpose` was never stated, and a
        second copy of that rule elsewhere is a second copy that can fall out of
        date. It is the field precedence in `node_instruction`'s docstring that
        already went wrong twice this way.
        """
        def tmpl(text: str | None) -> str:
            if not text:
                return ""
            rendered, unresolved = render_template(text, scope)
            if unresolved:
                self.logger.warning(
                    "Node %s: left %s unresolved in %r (available scope: %s)",
                    node.id,
                    ", ".join(f"{{{u}}}" for u in unresolved),
                    text,
                    sorted(scope),
                )
            return rendered

        if node.instructions and node.instructions.strip():
            return (
                NODE_SYSTEM_TEMPLATE.format(instructions=tmpl(node.instructions)),
                recited_names(node.instructions),
            )

        purpose = tmpl(node.purpose or node.description or f"Node {node.id}")
        goal = tmpl(node.goal or "Follow the instructions in the user prompt.")
        expected = tmpl(node.expected_result or "A useful, correct, and concise answer.")
        return (
            DEFAULT_NODE_SYSTEM_TEMPLATE.format(
                purpose=purpose,
                goal=goal,
                expected_result=expected,
            ),
            recited_names(
                node.purpose or node.description, node.goal, node.expected_result
            ),
        )

    def _load_output_schema_if_needed(self, node: GraphNode) -> type[PydModel] | None:
        if not node.output_schema:
            return None

        # Written on the node as JSON Schema — built, never executed. See
        # `engine.json_schema` for why that distinction is the point.
        if isinstance(node.output_schema, dict):
            try:
                return model_from_json_schema(node.output_schema, name=f"{node.id}_output")
            except JsonSchemaError as exc:
                raise GraphConfigurationError(
                    f"node '{node.id}' output_schema is not a usable JSON Schema: {exc}"
                ) from exc

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

        Three sources, in order of how *deliberately* they say it:

        1. **`output` nodes** — the graph states its return value as a node you
           can see. Only ones that actually ran count, so a router that took the
           left branch returns the left branch's output and not an empty entry
           for the right one.
        2. **`graph.outputs`** — the older form, a list of node ids. Kept working
           because every workflow on disk uses it.
        3. **The last node in topological order** — a guess, and the reason the
           first two exist.
        """
        ran = {
            n.id for n in self.graph.nodes
            if isinstance(n, Output) and n.id in results and not results[n.id].skipped
        }
        if ran:
            return {nid: results[nid].raw_output for nid in ran}

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
