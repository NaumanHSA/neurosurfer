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
from neurosurfer.llm.types import Usage
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
from ..secrets import expand_node_secrets, redact
from ..state import WorkflowState
from ..templates import (
    DEFAULT_NODE_SYSTEM_TEMPLATE,
    NODE_SYSTEM_TEMPLATE,
    node_instruction,
    recited_names,
    render_scope,
    render_template,
)
from ..utils import import_string, normalize_and_validate_graph_inputs, topo_sort

_TRACE_TEXT_LIMIT = 4000


def _trace_text(value: Any) -> Any:
    """JSON-safe, length-capped copy of a value for the trace.

    A trace is a debugging artifact, not a second store of every payload: a long
    map output would otherwise be duplicated in full on disk for every run.
    """
    if value is None or isinstance(value, (bool, int, float)):
        return value
    text = value if isinstance(value, str) else repr(value)
    return text if len(text) <= _TRACE_TEXT_LIMIT else text[:_TRACE_TEXT_LIMIT] + "…"


def _trace_step(tracer, **kwargs):
    """Open a trace step, or a no-op context when there's no tracer.

    Keeps call sites free of `if self.tracer is not None` noise; the Tracer's own
    disabled path already returns a no-op, this covers `tracer=None` too.
    """
    if tracer is None:
        from contextlib import nullcontext

        return nullcontext(None)
    return tracer(**kwargs)


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
    def _run_function_node(
        self,
        node: GraphNode,
        graph_inputs: dict[str, Any],
        dependency_results: dict[str, Any],
        state: WorkflowState | None = None,
    ) -> NodeExecutionResult:
        started_at = time.time()
        try:
            if not node.callable:
                raise GraphConfigurationError(
                    f"function node '{node.id}' has no 'callable' set."
                )
            fn = import_string(node.callable)
            # The container scope is passed **explicitly**. It used to arrive by
            # being merged into the body's graph inputs, which is the same thing
            # from the callable's side and a very different thing from a prompt's
            # — see `render_scope`. Last, so the current `item` wins over a graph
            # input that happens to share its name.
            kwargs = {**graph_inputs, **dependency_results, **(state.scope if state else {})}
            # Deterministic nodes belong in the trace too — a run's story is
            # incomplete if only the model calls show up.
            with _trace_step(
                self.tracer,
                kind="function",
                label=f"call.{node.callable}",
                node_id=node.id,
                agent_id=node.id,
                inputs={"kwargs": _trace_text(sorted(kwargs))},
            ) as step:
                raw = fn(**kwargs)
                if step is not None:
                    step.outputs(result=_trace_text(raw))
            return NodeExecutionResult(
                node_id=node.id,
                mode=node.mode,
                raw_output=raw,
                started_at=started_at,
                duration_ms=int((time.time() - started_at) * 1000),
                # Redacted: `kwargs` holds tool_args **after** `${NAME}`
                # substitution, so for a node using a secret this is the
                # plaintext credential. The trace span was already
                # redacted; this field was not, and it is persisted in the
                # run record and rendered in the studio. Verified leaking a
                # real database password before this line existed.
                node_input=_trace_text(redact(kwargs)),
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

    def _render_tool_args(
        self, node: GraphNode, args: Any, scope: dict[str, Any]
    ) -> Any:
        """Interpolate `{name}` through a tool node's static kwargs.

        Recurses into lists and dicts because a model writing `tool_args` nests
        them as readily as it writes a flat one.
        """
        if isinstance(args, str):
            rendered, unresolved = render_template(args, scope)
            # `${NAME}` contains `{NAME}`, so the template pass reports every
            # secret as unresolved and then leaves it intact for the pass below.
            # Warning about it says "your secret did not resolve" about the one
            # case where everything is working, which is worse than silence.
            declared = set(getattr(node, "secrets", None) or ())
            unresolved = [u for u in unresolved if u not in declared]
            if unresolved:
                # Refused, not warned. This value is about to be handed to a tool
                # as an argument, and nothing downstream can interpret a leftover
                # `{name}` — a `tool` node makes no model call, and a `react`
                # node's bound args are supplied verbatim. A build that wrote
                # `content: "{output of generate_markdown_table}"` warned here,
                # wrote the placeholder to disk, and reported success at every
                # level: build, run, and all three nodes.
                raise GraphConfigurationError(
                    f"node '{node.id}': tool_args contains "
                    f"{', '.join(f'{{{u}}}' for u in unresolved)}, which resolved "
                    f"to nothing — the tool would receive it as literal text. "
                    f"In scope: {', '.join(sorted(scope)) or 'nothing'}. "
                    f"Reference a graph input or an upstream node id, or double "
                    f"the brace ('{{{{' / '}}}}') if it is meant literally."
                )
            # `${NAME}` last, and only for names the node declared. A second pass
            # rather than an entry in `scope`: the scope is also what builds a
            # prompt, and a secret must not be reachable from there by any path.
            return expand_node_secrets(rendered, node)
        if isinstance(args, dict):
            return {k: self._render_tool_args(node, v, scope) for k, v in args.items()}
        if isinstance(args, list):
            return [self._render_tool_args(node, v, scope) for v in args]
        return args

    def _render_tool_settings(
        self, node: GraphNode, scope: dict[str, Any]
    ) -> dict[str, dict[str, Any]]:
        """A node's `tool_settings`, with templates and `${SECRET}` resolved.

        The same pass `tool_args` gets, for the same reason: a root of
        `{inputs.workspace}/out` is an ordinary thing to want, and a setting that
        could not say it would be the one field on the node where a placeholder
        silently reached the filesystem as literal text.
        """
        out: dict[str, dict[str, Any]] = {}
        for tool_name, values in (node.tool_settings or {}).items():
            if isinstance(values, dict):
                out[str(tool_name)] = self._render_tool_args(node, dict(values), scope)
        return out

    def _run_tool_node(
        self,
        node: GraphNode,
        graph_inputs: dict[str, Any],
        dependency_results: dict[str, Any],
        state: WorkflowState | None = None,
    ) -> NodeExecutionResult:
        started_at = time.time()
        try:
            if not node.tools:
                raise GraphConfigurationError(
                    f"tool node '{node.id}' has no 'tools' declared."
                )
            tool_name = node.tools[0]
            # `tool_args` interpolate `{name}` from graph inputs and upstream
            # outputs, exactly as a node's prompt fields do. Without this the
            # Architect's own shape for "read this file" —
            # `tool_args: {path: "{file_path}"}` — hands `read_file` the literal
            # string `{file_path}` and fails as "no such file", which reads like
            # a missing file rather than an unresolved template.
            #
            # Two different things, deliberately not one dict: what a template
            # may *resolve* against is wide (it includes the `inputs.`/`nodes.`
            # namespaces and the container scope); what the tool is *called*
            # with stays the flat mapping it has always been, since every key
            # here becomes a keyword argument and a `_Namespace` object is not
            # something any tool signature accepts.
            iter_scope = state.scope if state else {}
            render_ctx = render_scope(
                graph_inputs,
                nodes=dependency_results,
                variables=state.vars if state else None,
                scope=iter_scope,
            )
            tool_args = self._render_tool_args(node, node.tool_args or {}, render_ctx)
            kwargs = {**graph_inputs, **dependency_results, **iter_scope, **tool_args}

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
            # A `tool` node's one tool, configured. Same mechanism as an attached
            # tool's — the node kind decides who composes the call, never where
            # the call is allowed to land.
            native_tools = self.native_tools
            if node.tool_settings:
                from ..configured_tools import configure_pool

                native_tools = configure_pool(
                    native_tools, self._render_tool_settings(node, render_ctx)
                )

            from ..node_runner import run_tool_node
            with _trace_step(
                self.tracer,
                kind="tool",
                label=f"tool.{tool_name}",
                node_id=node.id,
                agent_id=node.id,
                # The tool gets the real value; the record of the call does not.
                # `tool_args` still holds the `${NAME}` source here, but a tool's
                # *output* can echo what it was given — a connection error quoting
                # the URL it failed on is the usual way a password reaches a trace.
                inputs={"args": _trace_text(redact(node.tool_args or {}))},
            ) as step:
                raw = run_tool_node(native_tools, tool_name, kwargs, self._tool_ctx)
                if step is not None:
                    step.outputs(result=_trace_text(redact(raw)))
            return NodeExecutionResult(
                node_id=node.id,
                mode=node.mode,
                raw_output=raw,
                started_at=started_at,
                duration_ms=int((time.time() - started_at) * 1000),
                # No model call here, so no usage — but the tool invocation is
                # still worth showing in the trace.
                tool_calls=[tool_name],
                # Redacted: `kwargs` holds tool_args **after** `${NAME}`
                # substitution, so for a node using a secret this is the
                # plaintext credential. The trace span was already
                # redacted; this field was not, and it is persisted in the
                # run record and rendered in the studio. Verified leaking a
                # real database password before this line existed.
                node_input=_trace_text(redact(kwargs)),
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
            if not cases and not node.routes:
                raise GraphConfigurationError(
                    f"router node '{node.id}' has neither 'routes' nor 'cases'."
                )
            if node.routes:
                # The simple form: the router IS the classifier (one LLM call).
                selected, label, usage, prompt = self._route_by_classification(node, state)
            elif any(c.when and c.when.strip() for c in cases):
                # Expression routing is free — no model call, so no usage.
                selected, label = self._route_by_expression(cases, node.default, state)
                usage, prompt = None, None
            else:
                selected, label, usage = self._route_by_llm(node, cases, state)
                prompt = None
            return NodeExecutionResult(
                node_id=node.id,
                mode=node.mode,
                raw_output=selected,
                structured_output={"selected": selected, "label": label},
                started_at=started_at,
                duration_ms=int((time.time() - started_at) * 1000),
                usage=usage,
                node_input=prompt,
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

    def _route_by_classification(self, node: GraphNode, state: WorkflowState):
        """`routes` router: one LLM call classifies and picks a label → target.

        The node's purpose/goal (interpolated with graph inputs, e.g. ``{ticket}``)
        is the classification instruction. If the answer matches no label and
        ``repair`` is on, retry once telling the model exactly what went wrong;
        after that fall back to ``default`` (or error if none).
        """
        if self.provider is None:
            raise GraphConfigurationError(
                f"router '{node.id}' uses `routes` (LLM classification) but no "
                f"provider was given to the executor."
            )
        from ..node_runner import run_base_node

        routes: dict[str, str] = node.routes or {}
        labels = list(routes)

        # **Graph inputs only, still** — no dependency outputs and no `vars`. That
        # narrowness is a contract, not an oversight: `validation/templates.py`
        # raises an *error* on a router referencing a node id or a write, on the
        # grounds that upstream results are already appended to the classifier
        # prompt below. Widening here would leave the validator rejecting graphs
        # the engine had quietly started running.
        #
        # The container scope is the one addition, and it is not a widening: a
        # router inside a `map` body has always seen `{item}` — the container
        # merged it into the body's inputs, which is exactly what the validator
        # models in `_container_bindings`. It arrives by its own door now.
        route_scope = render_scope(state.inputs, scope=state.scope)

        def tmpl(text: str) -> str:
            rendered, unresolved = render_template(text, route_scope)
            if unresolved:
                self.logger.warning(
                    "Router %s: left %s unresolved in %r (available: %s)",
                    node.id,
                    ", ".join(f"{{{u}}}" for u in unresolved),
                    text,
                    sorted(route_scope),
                )
            return rendered

        instruction = tmpl(node_instruction(node, f"Route for node {node.id}"))
        # Upstream outputs are the routing evidence when the router has parents.
        context = ""
        if node.depends_on:
            deps = {d: str(state.get_node_output(d))[:1200] for d in node.depends_on}
            import json as _json

            context = f"\n\nUpstream results:\n{_json.dumps(deps, ensure_ascii=False)}"

        system = (
            "You are a routing classifier inside a workflow engine. Decide which "
            "route fits best. Reply with EXACTLY one route name from the allowed "
            "list — nothing else."
        )
        user = f"{instruction}{context}\n\nAllowed routes: {labels}"

        def _match(text: str) -> str | None:
            answer = (text or "").strip().lower()
            for lb in labels:
                if lb.lower() == answer:
                    return lb
            for lb in labels:  # tolerate wrapping prose, e.g. "Route: urgent."
                if lb.lower() in answer:
                    return lb
            return None

        # A `routes` router is itself an LLM call (plus a possible repair call) —
        # its tokens belong to the router node, not to nobody.
        usage = Usage()
        with _trace_step(
            self.tracer,
            kind="llm",
            label="router.classify",
            node_id=node.id,
            agent_id=node.id,
            inputs={"system": system, "user": user, "routes": labels},
        ) as step:
            # The router's own provider/model, not the run's — classification is
            # the textbook case for "route with the cheap model and write with
            # the expensive one", and this call ignored both, so a router that
            # named a model was answered by whatever the run was started with.
            # `_provider_for` returns the run's default when a node names
            # neither, so nothing changes for a graph that does not ask.
            call = run_base_node(self._provider_for(node), system, user)
            if step is not None:
                step.outputs(answer=_trace_text(call.output))
                step.add_meta(usage=call.usage.model_dump() if call.usage else None)
        usage = usage.add(call.usage)
        raw = call.output
        label = _match(str(raw))
        if label is None and node.repair:
            self.logger.info(
                "router %s: answer %r matched no route; repairing", node.id, str(raw)[:80]
            )
            retry = (
                f"{user}\n\nYour previous answer was invalid: {str(raw)[:200]!r}. "
                f"Answer again with EXACTLY one of: {labels} — a single word, no prose."
            )
            call = run_base_node(self._provider_for(node), system, retry)
            usage = usage.add(call.usage)
            raw = call.output
            label = _match(str(raw))
        if label is not None:
            return routes[label], label, usage, _trace_text(user)
        if node.default:
            return node.default, None, usage, _trace_text(user)
        raise GraphExecutionError(
            f"router '{node.id}' could not map the model's answer "
            f"({str(raw)[:120]!r}) to any route in {labels} and has no default."
        )

    @staticmethod
    def _route_by_expression(cases, default, state: WorkflowState):
        from ..expressions import safe_bool

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
        from ..node_runner import run_base_node

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
        call = run_base_node(self.provider, system, user)
        raw, usage = call.output, call.usage
        answer = str(raw or "").strip().lower()
        # Match the answer to a case by label/to (exact, then substring).
        for case in cases:
            key = (case.label or case.to).lower()
            if key == answer:
                return case.to, case.label, usage
        for case in cases:
            key = (case.label or case.to).lower()
            if key and key in answer:
                return case.to, case.label, usage
        # No confident match → default (or first case as a last resort).
        return (node.default or cases[0].to), None, usage

    @staticmethod
    def _apply_router_pruning(
        node: GraphNode, result: NodeExecutionResult, pruned_ids: set[str]
    ) -> None:
        """Mark every router-controlled target except the selected one as pruned."""
        selected = result.raw_output
        controlled: set[str] = {c.to for c in (node.cases or [])}
        controlled.update((node.routes or {}).values())
        if node.default:
            controlled.add(node.default)
        for target in controlled:
            if target != selected:
                pruned_ids.add(target)

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

    @staticmethod
    def _body_value(result: GraphExecutionResult) -> Any:
        """Reduce a body run's final outputs to a single value (unwrap 1-key dicts)."""
        final = result.final or {}
        if len(final) == 1:
            return next(iter(final.values()))
        return final

    def _run_loop_node(self, node: GraphNode, state: WorkflowState) -> NodeExecutionResult:
        """Run ``node.body`` repeatedly until the stop condition or ``max_iterations``.

        Stop conditions (mutually exclusive):
          - ``until``       — a plain-English condition judged by an internal LLM
            decision after each iteration. A CONTINUE verdict carries a reason,
            which the next iteration receives as ``{feedback}`` — directed
            refinement, not blind retry.
          - ``break_when``  — a sandboxed expression, evaluated for free.

        Each iteration sees ``index``, the previous output (bound to ``item_var``),
        and ``feedback``; body node outputs are published back to the parent state.
        """
        from ..expressions import safe_bool

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
            usage = Usage()
            judge_log: list[dict[str, Any]] = []
            last_output: Any = None
            feedback = ""
            iterations = 0
            broke = False
            for i in range(node.max_iterations):
                iterations = i + 1
                scope = {"index": i, "iteration": i, node.item_var: last_output,
                         "acc": list(acc), "feedback": feedback}
                child_state = state.child_scope(scope)
                iter_inputs = {**state.inputs, "index": i,
                               node.item_var: last_output, "feedback": feedback}
                body_result = child.run(
                    iter_inputs,
                    seed_state=child_state,
                    node_event=self._active_node_event,
                    event_scope={
                        "parent": node.id, "kind": "loop",
                        "iteration": i, "total": node.max_iterations,
                    },
                )
                # Body nodes aren't in the parent's result map, so their tokens
                # would vanish — fold every iteration into the loop node's usage.
                usage = usage.add(body_result.total_usage())
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
                if node.until:
                    stop, reason, judge_usage = self._judge_loop_until(node, last_output, i)
                    usage = usage.add(judge_usage)
                    judge_log.append(
                        {"iteration": iterations, "stop": stop, "reason": reason}
                    )
                    self._log(
                        f"Node {node.id}: iteration {iterations} → "
                        f"{'stop' if stop else 'continue'}"
                        + (f" ({reason[:80]})" if reason else ""),
                        tracer=None, type="info",
                    )
                    if stop:
                        broke = True
                        break
                    feedback = reason or feedback
                elif node.break_when:
                    ns = state.child_scope(
                        {"index": i, "iteration": i, node.item_var: last_output, "acc": list(acc)}
                    ).namespace()
                    if safe_bool(node.break_when, ns, default=False):
                        broke = True
                        break
            result_value = state.vars.get(node.accumulate) if node.accumulate else last_output
            structured: dict[str, Any] = {
                "iterations": iterations, "broke_early": broke, "results": acc,
            }
            if judge_log:
                structured["judge"] = judge_log
            return NodeExecutionResult(
                node_id=node.id,
                mode=node.mode,
                raw_output=result_value,
                structured_output=structured,
                started_at=started_at,
                duration_ms=int((time.time() - started_at) * 1000),
                usage=usage,
            )
        except Exception as e:  # noqa: BLE001
            self.logger.exception("Loop node %s failed: %s", node.id, e)
            return self._error_result(node, started_at, str(e))

    def _judge_loop_until(
        self, node: GraphNode, body_value: Any, index: int
    ) -> tuple[bool, str]:
        """The loop's hidden exit judge: one constrained LLM decision per iteration.

        Returns ``(stop, reason)``. The reason for a CONTINUE verdict becomes the
        next iteration's ``{feedback}``. Unparseable answers get one corrective
        retry (when ``repair``), then fail safe to CONTINUE — the mandatory
        ``max_iterations`` ceiling still bounds the loop.
        """
        import json as _json
        import re as _re

        if self.provider is None:
            raise GraphConfigurationError(
                f"loop '{node.id}' uses `until` (LLM-judged) but no provider was "
                f"given to the executor."
            )
        from ..node_runner import run_base_node

        evidence = _json.dumps(body_value, ensure_ascii=False, default=str)[:3000]
        system = (
            "You are a loop-exit judge inside a workflow engine. Decide whether the "
            "stop condition is met. Reply with exactly one word first — STOP or "
            "CONTINUE — optionally followed by ' - <one short reason>'. On CONTINUE, "
            "the reason must say what is still lacking: it is handed to the next "
            "attempt as feedback."
        )
        user = (
            f"Stop condition: {node.until}\n\n"
            f"Iteration {index + 1} of at most {node.max_iterations}. Its results:\n"
            f"{evidence}\n\nIs the stop condition met?"
        )

        def _parse(text: str) -> tuple[bool | None, str]:
            m = _re.search(r"\b(stop|continue)\b", (text or "").lower())
            if m is None:
                return None, ""
            reason = (text or "")[m.end():].strip(" \t\n-—:.,*")
            return m.group(1) == "stop", reason

        # The judge is a hidden LLM call per iteration — often a third of a loop's
        # cost. Report it so it lands on the loop node's usage.
        usage = Usage()
        with _trace_step(
            self.tracer,
            kind="llm",
            label="loop.until_judge",
            node_id=node.id,
            agent_id=node.id,
            inputs={"until": node.until, "iteration": index + 1},
        ) as step:
            call = run_base_node(self.provider, system, user)
            if step is not None:
                step.outputs(verdict=_trace_text(call.output))
                step.add_meta(usage=call.usage.model_dump() if call.usage else None)
        usage = usage.add(call.usage)
        raw = call.output
        decision, reason = _parse(str(raw))
        if decision is None and node.repair:
            retry = (
                f"{user}\n\nYour previous answer was invalid: {str(raw)[:200]!r}. "
                f"Reply with exactly STOP or CONTINUE (one word), optionally "
                f"' - <short reason>'."
            )
            call = run_base_node(self.provider, system, retry)
            usage = usage.add(call.usage)
            raw = call.output
            decision, reason = _parse(str(raw))
        if decision is None:
            # Fail safe: keep iterating — max_iterations still bounds the loop.
            self.logger.warning(
                "loop %s: until-judge answer unparseable (%r); continuing",
                node.id, str(raw)[:80],
            )
            return False, "", usage
        return decision, reason, usage

    def _run_map_node(self, node: GraphNode, state: WorkflowState) -> NodeExecutionResult:
        """Fan ``node.body`` out over the collection from the ``over`` expression.

        Returns the list of per-item body outputs (implicit gather); a downstream
        node depending on this map receives that list.
        """
        from ..expressions import ExpressionError, evaluate

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
            # Items may run concurrently, so each one records its own usage and we
            # sum afterwards rather than mutating a shared accumulator.
            item_usage: list[Usage] = [Usage() for _ in items]

            def _run_item(i: int) -> Any:
                scope = {"index": i, node.item_var: items[i]}
                child_state = state.child_scope(scope)
                iter_inputs = {**state.inputs, "index": i, node.item_var: items[i]}
                body_result = child.run(
                    iter_inputs,
                    seed_state=child_state,
                    node_event=self._active_node_event,
                    # Passed per call, not stored on the child — map items may run
                    # concurrently on the same child executor.
                    event_scope={
                        "parent": node.id, "kind": "map",
                        "iteration": i, "total": len(items),
                    },
                )
                item_usage[i] = body_result.total_usage()
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

            total = Usage()
            for u in item_usage:
                total = total.add(u)
            return NodeExecutionResult(
                node_id=node.id,
                mode=node.mode,
                raw_output=results,
                structured_output={"count": len(items)},
                started_at=started_at,
                duration_ms=int((time.time() - started_at) * 1000),
                usage=total,
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
            body_result = child.run(
                dict(state.inputs),
                seed_state=state.child_scope({}),
                node_event=self._active_node_event,
                event_scope={"parent": node.id, "kind": "subgraph"},
            )
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
                usage=body_result.total_usage(),
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

        **What "the value" is depends on ``input_mode``.** In ``text`` it is one
        answer under ``writes`` or the node id. In ``dict`` it is the graph's
        *declared inputs* — the node collects several named values, so there is
        no single key to look under, and looking for one is exactly the bug this
        arm fixes: a form supplying ``{object, color}`` left a node waiting for
        something called ``input``, and the run parked with both answers already
        in hand.
        """
        from neurosurfer.tools.base import AutoApproveIOHandler

        started_at = time.time()

        if (node.input_mode or "text") == "dict":
            declared = [i.name for i in (self.graph.inputs or [])]
            required = {i.name for i in (self.graph.inputs or []) if i.required}
            collected = {n: state.inputs[n] for n in declared if n in state.inputs}
            missing = sorted(required - set(collected))
            if declared and not missing:
                return NodeExecutionResult(
                    node_id=node.id, mode=node.mode, raw_output=collected,
                    structured_output={"source": "supplied", "mode": "dict"},
                    started_at=started_at,
                    duration_ms=int((time.time() - started_at) * 1000),
                )
            if not declared:
                reason = (
                    f"input node '{node.id}' is set to collect a dict, but the "
                    f"graph declares no inputs. Add them on the node, or switch "
                    f"it to text."
                )
            else:
                reason = (
                    f"input node '{node.id}' is awaiting "
                    f"{', '.join(f'{m!r}' for m in missing)} — supply "
                    f"{'it' if len(missing) == 1 else 'them'} to resume this run."
                )
            return self._error_result(node, started_at, reason)

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
        question = node_instruction(node, f"Input needed for '{node.id}'")
        # Headless auto-approvers are non-interactive — don't fabricate an answer.
        if io is not None and not isinstance(io, AutoApproveIOHandler):
            from ..node_runner import run_coro_blocking
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

    def _run_output_node(
        self,
        node: GraphNode,
        graph_inputs: dict[str, Any],
        dependency_results: dict[str, Any],
        state: WorkflowState,
    ) -> NodeExecutionResult:
        """Declare what the graph returns.

        An output node runs nothing and calls nothing. It exists so a graph can
        *say* what its answer is, in a place you can see on a canvas, instead of
        leaving a caller to infer it from `graph.outputs` or from whichever node
        happened to be last in topological order.

        It is a **collector**: everything wired into it is something a caller
        asked to see. The shapes, in the order they are decided:

        - **one dependency, no `value`** — that dependency's output *unchanged*,
          preserving its type. It matters that this is not `str(...)`: a workflow
          returning a dict or a file reference must not be flattened to text on
          the way out just because it passed through here.
        - **several dependencies** — all of them, keyed by node id. Picking one
          would be a guess.
        - **`value` set** — an interpolated template over the same scope every
          other template field sees, so `"{summary} ({rows} rows)"` composes an
          answer without a `python` node whose only job is to join two strings.
          With dependencies present it is an *extra* entry keyed by this node's
          own id, not a replacement for them.
        """
        started_at = time.time()

        def done(out: Any, source: str) -> NodeExecutionResult:
            return NodeExecutionResult(
                node_id=node.id, mode=node.mode, raw_output=out,
                structured_output={"source": source},
                started_at=started_at,
                duration_ms=int((time.time() - started_at) * 1000),
            )

        if node.value is None:
            deps = [d for d in node.depends_on if d in dependency_results]
            if len(deps) == 1:
                return done(dependency_results[deps[0]], "passthrough")
            if not deps:
                return self._error_result(
                    node, started_at,
                    f"output node '{node.id}' has no `value` and nothing to pass "
                    f"through — give it a dependency, or set `value`.",
                )
            # Several dependencies and no template: return them keyed by node id
            # rather than picking one, which would be a guess.
            return done({d: dependency_results[d] for d in deps}, "merged")

        scope = render_scope(
            graph_inputs,
            nodes=dependency_results,
            variables=state.vars,
            scope=state.scope,
        )
        rendered, unresolved = render_template(node.value, scope)
        if unresolved:
            # Refused rather than warned, for the same reason `tool_args` refuses:
            # this text *is* the answer handed back to a caller, and a leftover
            # `{name}` in it is a defect that reads as content.
            return self._error_result(
                node, started_at,
                f"output node '{node.id}': value contains "
                f"{', '.join(f'{{{u}}}' for u in unresolved)}, which resolved to "
                f"nothing — the caller would receive it as literal text. "
                f"Available: {sorted(k for k in scope if not k.startswith('_'))}",
            )

        # A `value` alongside dependencies is an *extra* answer, not a
        # replacement for them. The node is a collector first: everything wired
        # into it is something a caller asked to see, and a composed sentence is
        # one more of those rather than a reason to discard the rest. Keyed by
        # the output node's own id so it names itself in the result.
        deps = [d for d in node.depends_on if d in dependency_results]
        if deps:
            collected = {d: dependency_results[d] for d in deps}
            return done({**collected, node.id: rendered}, "merged+template")
        return done(rendered, "template")

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
