"""Choosing a branch: the `router` kind, and the pruning that follows it.

Three flavours behind one kind, which is why they are together: a `routes`
router classifies with one LLM call, a `cases` router with `when` predicates
evaluates expressions and costs nothing, and a `cases` router with bare labels
asks the model to pick one. All three answer the same question — *which single
target runs next* — and `apply_router_pruning` marks the rest.

`apply_router_pruning` is a scheduling step, called from `GraphExecutor.run`
rather than from the runner, because pruning is about the nodes that have not
run yet. It lives here anyway: what a router controls is a fact about routers,
and the two definitions have to agree about `routes`, `cases` and `default`.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from neurosurfer.llm.types import Usage

from ..errors import GraphConfigurationError, GraphExecutionError
from ..schema import GraphNode, NodeExecutionResult
from ..state import WorkflowState
from ..templates import node_instruction, render_scope, render_template
from ._trace import _trace_step, _trace_text

if TYPE_CHECKING:  # a type hint only — importing `core` here would be a cycle
    from .core import GraphExecutor

__all__ = ["apply_router_pruning", "route_by_expression", "run_router_node"]


def run_router_node(
    ex: GraphExecutor, node: GraphNode, state: WorkflowState
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
            selected, label, usage, prompt = _route_by_classification(ex, node, state)
        elif any(c.when and c.when.strip() for c in cases):
            # Expression routing is free — no model call, so no usage.
            selected, label = route_by_expression(cases, node.default, state)
            usage, prompt = None, None
        else:
            selected, label, usage = _route_by_llm(ex, node, cases, state)
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
        ex.logger.exception("Router node %s failed: %s", node.id, e)
        return NodeExecutionResult(
            node_id=node.id,
            mode=node.mode,
            raw_output=None,
            started_at=started_at,
            duration_ms=int((time.time() - started_at) * 1000),
            error=str(e),
        )


def _route_by_classification(ex: GraphExecutor, node: GraphNode, state: WorkflowState):
    """`routes` router: one LLM call classifies and picks a label → target.

    The node's purpose/goal (interpolated with graph inputs, e.g. ``{ticket}``)
    is the classification instruction. If the answer matches no label and
    ``repair`` is on, retry once telling the model exactly what went wrong;
    after that fall back to ``default`` (or error if none).
    """
    if ex.provider is None:
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
            ex.logger.warning(
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
        ex.tracer,
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
        call = run_base_node(ex._provider_for(node), system, user)
        if step is not None:
            step.outputs(answer=_trace_text(call.output))
            step.add_meta(usage=call.usage.model_dump() if call.usage else None)
    usage = usage.add(call.usage)
    raw = call.output
    label = _match(str(raw))
    if label is None and node.repair:
        ex.logger.info(
            "router %s: answer %r matched no route; repairing", node.id, str(raw)[:80]
        )
        retry = (
            f"{user}\n\nYour previous answer was invalid: {str(raw)[:200]!r}. "
            f"Answer again with EXACTLY one of: {labels} — a single word, no prose."
        )
        call = run_base_node(ex._provider_for(node), system, retry)
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


def route_by_expression(cases, default, state: WorkflowState):
    from ..expressions import safe_bool

    ns = state.namespace()
    for case in cases:
        # An empty/None `when` is a catch-all (always matches).
        if not case.when or not case.when.strip() or safe_bool(case.when, ns, default=False):
            return case.to, case.label
    return default, None


def _route_by_llm(ex: GraphExecutor, node: GraphNode, cases, state: WorkflowState):
    """Ask the LLM to choose exactly one route label from the case list."""
    if ex.provider is None:
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
    call = run_base_node(ex.provider, system, user)
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


def apply_router_pruning(
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
