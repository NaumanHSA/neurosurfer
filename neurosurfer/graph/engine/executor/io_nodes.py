"""The two kinds that talk to the edges of a run: `input` and `output`.

An `input` node asks a person for a value; an `output` node decides what the
caller gets back. Neither is a step of the work — they are where a run meets
something that is not the graph, which is why the pair is one module.

The `input` node is also the resume point of a durable run: a pre-supplied value
under its key is how an API run continues after having parked to ask.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from ..schema import GraphNode, NodeExecutionResult
from ..state import WorkflowState
from ..templates import node_instruction, render_scope, render_template

if TYPE_CHECKING:  # a type hint only — importing `core` here would be a cycle
    from .core import GraphExecutor

__all__ = ["run_input_node", "run_output_node"]


def run_input_node(ex: GraphExecutor, node: GraphNode, state: WorkflowState) -> NodeExecutionResult:
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
        declared = [i.name for i in (ex.graph.inputs or [])]
        required = {i.name for i in (ex.graph.inputs or []) if i.required}
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
        return ex._error_result(node, started_at, reason)

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

    io = ex._tool_ctx.io if ex._tool_ctx else None
    question = node_instruction(node, f"Input needed for '{node.id}'")
    # Headless auto-approvers are non-interactive — don't fabricate an answer.
    if io is not None and not isinstance(io, AutoApproveIOHandler):
        from ..node_runner import run_coro_blocking
        try:
            answer = run_coro_blocking(io.ask(question, node.options or None))
        except Exception as e:  # noqa: BLE001
            return ex._error_result(node, started_at, f"input ask failed: {e}")
        return NodeExecutionResult(
            node_id=node.id, mode=node.mode, raw_output=answer,
            structured_output={"source": "interactive"},
            started_at=started_at, duration_ms=int((time.time() - started_at) * 1000),
        )

    return ex._error_result(
        node, started_at,
        f"input node '{node.id}' is awaiting a value — supply '{key}' as an input "
        f"to resume this run.",
    )


def run_output_node(
    ex: GraphExecutor,
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
            return ex._error_result(
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
        return ex._error_result(
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
