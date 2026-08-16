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
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from neurosurfer.llm.types import Usage

from ..errors import GraphConfigurationError, GraphExecutionError
from ..schema import GraphExecutionResult, GraphNode, NodeExecutionResult
from ..state import WorkflowState
from ._trace import _trace_step, _trace_text

if TYPE_CHECKING:  # a type hint only — importing `core` here would be a cycle
    from .core import GraphExecutor

__all__ = [
    "LoopIteration", "body_value", "run_loop_node", "run_map_node", "run_subgraph_node",
]



def body_value(result: GraphExecutionResult) -> Any:
    """Reduce a body run's final outputs to a single value (unwrap 1-key dicts)."""
    final = result.final or {}
    if len(final) == 1:
        return next(iter(final.values()))
    return final


@dataclass(frozen=True)
class LoopIteration:
    """What one pass of a loop looked like — the argument an `until` function gets.

    The body run's `GraphExecutionResult` is the substance here and is handed
    over whole as `result`: every node's output, errors, skips and token usage,
    with nothing hidden behind a reduction. But a stop decision usually needs
    more than the latest run — "has this stopped improving", "have I tried
    three times", "did the score go down" are all questions about the *sequence*
    — and a result object cannot answer them. So it arrives alongside the
    position and the history rather than on its own.

    One object rather than several arguments, because this contract is public
    the moment anyone writes an `until` function: adding a field later must not
    break every one of them.

    Return `True` to stop. Return `(True, "reason")` — or `(False, "reason")` —
    to also set the next iteration's `{feedback}`, the same channel the
    plain-English judge uses, so a function can steer the next attempt too.
    """

    #: 0-based, matching the `index` the body's templates see.
    index: int
    #: 1-based — the human count, and what `structured_output["iterations"]` reports.
    iteration: int
    #: The reduced output of this pass (`body_value` of `result`).
    output: Any
    #: This pass's full body run.
    result: GraphExecutionResult
    #: Every iteration's `output` so far, including this one.
    history: list[Any]
    #: The reason carried out of the previous iteration; "" on the first.
    feedback: str
    #: The loop's ceiling, so a function can tell "last chance" from "keep going".
    max_iterations: int
    #: The parent workflow's variables, readable (a copy — writes do not leak).
    vars: dict[str, Any]

    @property
    def is_last(self) -> bool:
        """True when the ceiling will stop the loop after this pass anyway."""
        return self.iteration >= self.max_iterations


def run_loop_node(ex: GraphExecutor, node: GraphNode, state: WorkflowState) -> NodeExecutionResult:
    """Run ``node.body`` repeatedly until ``until`` says stop, or ``max_iterations``.

    One stop condition, in one field, read as whichever of two things it is:

      - **a function** — a callable, or the name of one in the graph's
        ``functions:`` sidecar. Called with a :class:`LoopIteration`; returns
        ``True`` to stop, or ``(stop, reason)`` to also set ``{feedback}``.
        Deterministic and free.
      - **plain English** — judged by an internal LLM decision each iteration,
        which can also report a condition it cannot relate to the work at all
        and stop rather than burn the ceiling on it.

    Which one it is is looked up, not guessed: a name the sidecar defines as a
    callable is a function, anything else is prose.

    Each iteration sees ``index``, the previous output (bound to ``item_var``),
    and ``feedback``; body node outputs are published back to the parent state.
    """
    started_at = time.time()
    try:
        if not node.body:
            raise GraphConfigurationError(f"loop node '{node.id}' has no body.")
        if not node.max_iterations or node.max_iterations < 1:
            raise GraphConfigurationError(
                f"loop node '{node.id}' requires max_iterations >= 1 (a hard ceiling)."
            )
        child = ex._child_executor(node)
        # Resolved once, before any work: a loop naming a function that does not
        # exist should say so now, not on iteration one after paying for a body.
        stop_fn = _resolve_until_function(ex, node)
        acc: list[Any] = []
        usage = Usage()
        judge_log: list[dict[str, Any]] = []
        last_output: Any = None
        feedback = ""
        iterations = 0
        broke = False
        unrelated: str | None = None
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
            # Publish body outputs to the parent state (readable by `until`).
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
            if stop_fn is not None:
                stop, reason = _call_until_function(
                    ex, node, stop_fn,
                    LoopIteration(
                        index=i,
                        iteration=iterations,
                        output=last_output,
                        result=body_result,
                        history=list(acc),
                        feedback=feedback,
                        max_iterations=node.max_iterations,
                        vars=dict(state.vars),
                    ),
                )
            elif node.until:
                stop, reason, related, judge_usage = _judge_loop_until(
                    ex, node, last_output, i
                )
                usage = usage.add(judge_usage)
                judge_log.append(
                    {"iteration": iterations, "stop": stop,
                     "reason": reason, "related": related}
                )
                if not related:
                    # The condition cannot be judged against this work at all, so
                    # iterating is spending the ceiling to learn nothing. Stop and
                    # say why — the loop's own output is still returned.
                    unrelated = reason
                    broke = True
                    ex.logger.warning(
                        "loop %s: stop condition %r does not relate to what the body "
                        "produces (%s). Stopped after iteration %d rather than running "
                        "to max_iterations=%d.",
                        node.id, node.until, reason or "no reason given",
                        iterations, node.max_iterations,
                    )
                    ex._log(
                        f"Node {node.id}: stop condition unrelated to the body's work "
                        f"— stopped at iteration {iterations}"
                        + (f" ({reason[:80]})" if reason else ""),
                        tracer=None, type="warning",
                    )
                    break
            else:
                stop, reason = False, ""      # no condition: run to the ceiling

            if stop_fn is not None or node.until:
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
        result_value = state.vars.get(node.accumulate) if node.accumulate else last_output
        structured: dict[str, Any] = {
            "iterations": iterations, "broke_early": broke, "results": acc,
        }
        if judge_log:
            structured["judge"] = judge_log
        if unrelated is not None:
            structured["stopped_reason"] = "condition_unrelated"
            structured["stopped_detail"] = unrelated
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


def _resolve_until_function(
    ex: GraphExecutor, node: GraphNode
) -> Callable[[LoopIteration], Any] | None:
    """The callable behind `until`, or None when `until` is prose (or absent).

    Three cases, and only the middle one needs looking anything up:

      * already a callable — a graph built in Python passed the function itself;
      * a name the graph's `functions:` sidecar defines — that is the function;
      * anything else — plain English, judged by the LLM.

    The distinction is a lookup, never a guess about the shape of the string. The
    one case worth stopping for is a graph that *declares* a sidecar and names
    something absent from it: "is_don" against a file defining `is_done` is a
    typo, and silently sending it to an LLM judge as though it were English
    would be a very expensive way to not find out.
    """
    until = node.until
    if until is None:
        return None
    if callable(until):
        return until
    graph = getattr(ex, "graph", None)
    sidecar = getattr(graph, "sidecar", None) if graph is not None else None
    if sidecar is None:
        return None                       # no sidecar ⇒ it can only be prose
    fn = graph.function(until)
    if fn is not None:
        return fn
    # A sidecar exists and does not define this name. Prose is still the likely
    # reading for anything sentence-shaped; a bare identifier is not.
    if until.isidentifier():
        raise GraphConfigurationError(
            f"loop '{node.id}': `until: {until}` names no function in "
            f"{getattr(sidecar, 'path', 'the functions file')}. "
            f"It defines: {', '.join(sidecar.names()) or '(nothing)'}. "
            f"Write a plain-English condition, or define {until}()."
        )
    return None


def _call_until_function(
    ex: GraphExecutor,
    node: GraphNode,
    fn: Callable[[LoopIteration], Any],
    it: LoopIteration,
) -> tuple[bool, str]:
    """Run the loop's stop function. Returns ``(stop, reason)``.

    Accepts ``True``/``False`` or ``(stop, reason)`` — the reason becomes the
    next iteration's ``{feedback}``, the same channel the LLM judge uses, so a
    function can steer the next attempt and not merely end it.

    A raising function stops the loop rather than being swallowed. This is the
    author's own code stating the exit condition; continuing to iterate past it
    would be running a loop whose bound has failed, and the ceiling is a
    backstop for a condition that never comes true, not for one that is broken.
    """
    try:
        verdict = fn(it)
    except Exception as e:
        raise GraphExecutionError(
            f"loop '{node.id}': `until` function {getattr(fn, '__name__', fn)!r} "
            f"raised on iteration {it.iteration}: {e}"
        ) from e
    if isinstance(verdict, tuple):
        stop, reason = (list(verdict) + [""])[:2]
        return bool(stop), str(reason or "")
    return bool(verdict), ""


def _judge_loop_until(
    ex: GraphExecutor, node: GraphNode, body_value: Any, index: int
) -> tuple[bool, str, bool, Usage]:
    """The loop's hidden exit judge: one constrained LLM decision per iteration.

    Returns ``(stop, reason, related, usage)``.

    Three verdicts, not two. STOP and CONTINUE are the loop's business as usual —
    a CONTINUE reason becomes the next iteration's ``{feedback}``. The third,
    UNRELATED, is for a condition that cannot be judged against this work at
    all: "stop when winter is here" over a body writing coffee taglines. Nothing
    the body produces can ever satisfy it, so CONTINUE would be a lie that costs
    the full ceiling — every iteration, plus a judge call each — to tell.

    It rides on the call that was already being made, so noticing costs nothing,
    and it is asked *after* an iteration rather than before any: the judge then
    has the body's real output to compare against, which is far better evidence
    than the author's description of what the body was supposed to do.

    Deliberately narrow. The judge is told to use UNRELATED only when the
    condition is about a different subject entirely — "not yet true", "vague"
    and "hard to tell" are all CONTINUE. A condition that merely *looks*
    demanding must not be reclassified as nonsense, because the cost of a false
    UNRELATED is a loop that stops on its first iteration for no reason.

    Unparseable answers get one corrective retry (when ``repair``), then fail
    safe to CONTINUE — the mandatory ``max_iterations`` ceiling still bounds it.
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
        "stop condition is met. Reply with exactly one word first — STOP, "
        "CONTINUE or UNRELATED — optionally followed by ' - <one short reason>'.\n"
        "STOP: the condition is met.\n"
        "CONTINUE: not met yet. The reason must say what is still lacking — it is "
        "handed to the next attempt as feedback.\n"
        "UNRELATED: the condition is about a different subject than the results, "
        "so no amount of further work could ever satisfy it. Use this ONLY for a "
        "genuine mismatch of subject. A condition that is merely demanding, "
        "vague, or not yet satisfied is CONTINUE, not UNRELATED."
    )
    user = (
        f"Stop condition: {node.until}\n\n"
        f"Iteration {index + 1} of at most {node.max_iterations}. Its results:\n"
        f"{evidence}\n\nIs the stop condition met?"
    )

    def _parse(text: str) -> tuple[bool | None, str, bool]:
        """→ (stop, reason, related). `stop is None` means unparseable."""
        m = _re.search(r"\b(stop|continue|unrelated)\b", (text or "").lower())
        if m is None:
            return None, "", True
        reason = (text or "")[m.end():].strip(" \t\n-—:.,*")
        verdict = m.group(1)
        if verdict == "unrelated":
            return True, reason, False
        return verdict == "stop", reason, True

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
    decision, reason, related = _parse(str(raw))
    if decision is None and node.repair:
        retry = (
            f"{user}\n\nYour previous answer was invalid: {str(raw)[:200]!r}. "
            f"Reply with exactly STOP, CONTINUE or UNRELATED (one word), "
            f"optionally ' - <short reason>'."
        )
        call = run_base_node(ex.provider, system, retry)
        usage = usage.add(call.usage)
        raw = call.output
        decision, reason, related = _parse(str(raw))
    if decision is None:
        # Fail safe: keep iterating — max_iterations still bounds the loop. Note
        # this fails toward CONTINUE, never toward UNRELATED: a judge we could
        # not read must not be what stops someone's loop.
        ex.logger.warning(
            "loop %s: until-judge answer unparseable (%r); continuing",
            node.id, str(raw)[:80],
        )
        return False, "", True, usage
    return decision, reason, related, usage


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
