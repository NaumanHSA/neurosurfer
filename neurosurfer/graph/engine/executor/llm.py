"""The kinds that call a model: `base` and `react`, and the prompt they get.

One function does the call for both, because the difference between them is a
parameter and not a code path — a `base` node gets one round of tools, a `react`
node loops until it has an answer. What surrounds the call is the same: bind the
tools, run it, fold the usage in, and refuse a step that produced nothing.

`build_system_prompt` returns the prompt **and the names it recited**, so the
user prompt can avoid saying the same thing twice. The two travel together
because which fields get rendered is that function's rule; see its docstring.
"""

from __future__ import annotations

import time
from concurrent.futures import TimeoutError as FuturesTimeout
from contextvars import copy_context
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel as PydModel

from neurosurfer.llm.base import Provider
from neurosurfer.tools.base import ToolPool

from ..errors import GraphConfigurationError
from ..json_schema import JsonSchemaError, model_from_json_schema
from ..nodes import ReactNode
from ..schema import GraphNode, NodeExecutionResult
from ..templates import (
    DEFAULT_NODE_TASK_TEMPLATE,
    NODE_TASK_TEMPLATE,
    render_template,
)
from ..utils import import_string
from ._trace import _trace_step, _trace_text

if TYPE_CHECKING:  # a type hint only — importing `core` here would be a cycle
    from .core import GraphExecutor

__all__ = ["load_output_schema_if_needed", "render_task", "run_node_native"]


def run_node_native(
    ex: GraphExecutor,
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
    provider = provider or ex.provider

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
            ex.native_tools.select(node.tools)
            if ex.native_tools
            else ToolPool([])
        )
        if node.tool_args:
            from ..bound_tools import bind_pool
            from .deterministic import _render_tool_args

            pool = bind_pool(
                pool,
                _render_tool_args(ex, node, dict(node.tool_args), scope or {}),
            )
        # Settings are applied **after** binding, so a configured tool wraps a
        # bound one rather than the other way round. The order matters: the
        # root has to scope the call that actually happens, and a bound
        # argument is part of that call.
        if node.tool_settings:
            from ..configured_tools import configure_pool
            from .deterministic import _render_tool_settings

            pool = configure_pool(pool, _render_tool_settings(ex, node, scope or {}))
        tool_ctx = ex._tool_ctx
        if tool_ctx is None:
            from pathlib import Path

            from neurosurfer.tools.base import ToolContext

            from ..node_runner import _HeadlessIO
            tool_ctx = ToolContext(cwd=Path.cwd(), io=_HeadlessIO())
        return pool, tool_ctx

    def _execute() -> Any:
        if isinstance(node, ReactNode):
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
                output_schema=output_schema,
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
                    ex.logger.warning("Node %s timed out after %ss", node.id, timeout_s)
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
                ex.tracer,
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
        # The other half of the debugging view — see the prompt log in
        # `core._run_node` for why this is `debug` and not `print`.
        ex.logger.debug("node %s output: %s", node.id, _trace_text(raw))
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
            model=getattr(provider, "model", None),
            tool_calls=call.tool_calls,
            node_input=_trace_text(user_prompt),
        )
        ex.artifacts.put(node.id, raw)
        if ex.exporter and node.export:
            ex.exporter.export_single_node(node=node, result=result)
        return result

    except Exception as e:
        duration_ms = int((time.time() - started_at) * 1000)
        ex.logger.exception("Node %s failed: %s", node.id, e)
        return NodeExecutionResult(
            node_id=node.id,
            mode=node.mode,
            raw_output=None,
            started_at=started_at,
            duration_ms=duration_ms,
            error=str(e),
        )


def render_task(ex: GraphExecutor, node: GraphNode, scope: dict[str, Any]) -> str:
    """What this node is being asked to do, with its `{placeholders}` filled.

    This is the **task block of the user turn**, not the system prompt. It used
    to be rendered into the system prompt, which made that prompt different for
    every node and, inside a `map`, different for every item — see
    `NODE_SYSTEM_PROMPT` for the caching cost of that and for the convention it
    was on the wrong side of.

    Two shapes, and which one is used depends on the node:

    - **`instructions`** — one field saying what this step should do. This is
      what new nodes set, and it wins outright when present.
    - **`purpose` / `goal` / `expected_result`** — the three fields that came
      first. Still read, so every workflow already on disk keeps running; see
      `GraphNode.instructions` for why they were collapsed.

    Example:
        instructions: "Research {company_title} and write a title based on
                       the summary: {summary}"

    **The placeholders here are the whole of what a node sees from graph
    state.** There is no ambient block of every graph input any more, so a
    value this text does not name is a value this node was not given — which is
    what `agent_has_something_to_work_from` checks before a run starts.
    """
    def tmpl(text: str | None) -> str:
        if not text:
            return ""
        rendered, unresolved = render_template(text, scope)
        if unresolved:
            ex.logger.warning(
                "Node %s: left %s unresolved in %r (available scope: %s)",
                node.id,
                ", ".join(f"{{{u}}}" for u in unresolved),
                text,
                sorted(scope),
            )
        return rendered

    if node.instructions and node.instructions.strip():
        return NODE_TASK_TEMPLATE.format(instructions=tmpl(node.instructions))

    return DEFAULT_NODE_TASK_TEMPLATE.format(
        purpose=tmpl(node.purpose or node.description or f"Node {node.id}"),
        goal=tmpl(node.goal or "Follow the instructions in this turn."),
        expected_result=tmpl(
            node.expected_result or "A useful, correct, and concise answer."
        ),
    )


def load_output_schema_if_needed(ex: GraphExecutor, node: GraphNode) -> type[PydModel] | None:
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
