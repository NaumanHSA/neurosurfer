"""Running a body more than once: `loop`, `map` and `subgraph`.

The three kinds that build a nested `Graph` from `node.body` and hand it to a
child executor. What differs is only *how often* and *with what bound to the
iteration scope* — a `loop` re-runs until a condition, a `map` runs once per
element with the elements independent, and a `subgraph` runs the thing once.

`_child_executor` and `_hidden_body_inputs` stay on `GraphExecutor`: they build
an executor, which is the class's own business, and keeping them there is what
lets this module hold no runtime import of `core` at all.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from typing import TYPE_CHECKING, Any

from neurosurfer.llm.types import Usage

from ..errors import GraphConfigurationError, GraphExecutionError
from ..schema import GraphExecutionResult, GraphNode, NodeExecutionResult
from ..state import WorkflowState
from ._trace import _trace_step, _trace_text

if TYPE_CHECKING:  # a type hint only — importing `core` here would be a cycle
    from .core import GraphExecutor

__all__ = ["body_value", "run_loop_node", "run_map_node", "run_subgraph_node"]



def body_value(result: GraphExecutionResult) -> Any:
    """Reduce a body run's final outputs to a single value (unwrap 1-key dicts)."""
    final = result.final or {}
    if len(final) == 1:
        return next(iter(final.values()))
    return final


def run_loop_node(ex: GraphExecutor, node: GraphNode, state: WorkflowState) -> NodeExecutionResult:
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
        child = ex._child_executor(node)
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
                node_event=ex._active_node_event,
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
            last_output = body_value(body_result)
            acc.append(last_output)
            if node.accumulate:
                state.set_var(node.accumulate, list(acc))
            if body_result.errors:
                # A failing body stops the loop (surface it below).
                return ex._error_result(
                    node, started_at,
                    f"loop body failed on iteration {iterations}: {body_result.errors}",
                )
            if node.until:
                stop, reason, judge_usage = _judge_loop_until(ex, node, last_output, i)
                usage = usage.add(judge_usage)
                judge_log.append(
                    {"iteration": iterations, "stop": stop, "reason": reason}
                )
                ex._log(
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
        ex.logger.exception("Loop node %s failed: %s", node.id, e)
        return ex._error_result(node, started_at, str(e))


def _judge_loop_until(
    ex: GraphExecutor, node: GraphNode, body_value: Any, index: int
) -> tuple[bool, str]:
    """The loop's hidden exit judge: one constrained LLM decision per iteration.

    Returns ``(stop, reason)``. The reason for a CONTINUE verdict becomes the
    next iteration's ``{feedback}``. Unparseable answers get one corrective
    retry (when ``repair``), then fail safe to CONTINUE — the mandatory
    ``max_iterations`` ceiling still bounds the loop.
    """
    import json as _json
    import re as _re

    if ex.provider is None:
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
        ex.tracer,
        kind="llm",
        label="loop.until_judge",
        node_id=node.id,
        agent_id=node.id,
        inputs={"until": node.until, "iteration": index + 1},
    ) as step:
        call = run_base_node(ex.provider, system, user)
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
        call = run_base_node(ex.provider, system, retry)
        usage = usage.add(call.usage)
        raw = call.output
        decision, reason = _parse(str(raw))
    if decision is None:
        # Fail safe: keep iterating — max_iterations still bounds the loop.
        ex.logger.warning(
            "loop %s: until-judge answer unparseable (%r); continuing",
            node.id, str(raw)[:80],
        )
        return False, "", usage
    return decision, reason, usage


def run_map_node(ex: GraphExecutor, node: GraphNode, state: WorkflowState) -> NodeExecutionResult:
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
        child = ex._child_executor(node)
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
                node_event=ex._active_node_event,
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
            return body_value(body_result)

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
        ex.logger.exception("Map node %s failed: %s", node.id, e)
        return ex._error_result(node, started_at, str(e))



def run_subgraph_node(ex: GraphExecutor, node: GraphNode, state: WorkflowState) -> NodeExecutionResult:
    """Run ``node.body`` once as a nested sub-graph (composition).

    The body sees the parent inputs/state; its final outputs become this node's
    output (unwrapped when there's a single output).
    """
    started_at = time.time()
    try:
        if not node.body:
            raise GraphConfigurationError(f"subgraph node '{node.id}' has no body.")
        child = ex._child_executor(node)
        body_result = child.run(
            dict(state.inputs),
            seed_state=state.child_scope({}),
            node_event=ex._active_node_event,
            event_scope={"parent": node.id, "kind": "subgraph"},
        )
        if body_result.errors:
            return ex._error_result(
                node, started_at, f"subgraph body failed: {body_result.errors}"
            )
        return NodeExecutionResult(
            node_id=node.id,
            mode=node.mode,
            raw_output=body_value(body_result),
            structured_output={"body_nodes": list(body_result.nodes.keys())},
            started_at=started_at,
            duration_ms=int((time.time() - started_at) * 1000),
            usage=body_result.total_usage(),
        )
    except Exception as e:  # noqa: BLE001
        ex.logger.exception("Subgraph node %s failed: %s", node.id, e)
        return ex._error_result(node, started_at, str(e))
