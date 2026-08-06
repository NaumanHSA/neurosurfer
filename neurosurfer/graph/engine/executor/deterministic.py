"""The kinds that make no model call: `function`, `python` and `tool`.

Deterministic in the sense that matters here — nothing composes their arguments.
A `function` node is a Python callable reached by import path; a `tool` node is
one registered tool invoked directly, where `tool_args` *is* the whole
instruction because there is no model to write one.

Both are handed the same flat mapping as keyword arguments — graph inputs,
dependency outputs, then the container scope — which is why `render_scope`'s wide
template scope and this narrow kwargs mapping are deliberately two different
things here. A `_Namespace` object is not something any tool signature accepts.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from ..errors import GraphConfigurationError
from ..schema import GraphNode, NodeExecutionResult
from ..secrets import expand_node_secrets, redact
from ..state import WorkflowState
from ..templates import render_scope, render_template
from ..utils import import_string
from ._trace import _trace_step, _trace_text

if TYPE_CHECKING:  # a type hint only — importing `core` here would be a cycle
    from .core import GraphExecutor

__all__ = ["run_function_node", "run_tool_node"]


def run_function_node(
    ex: GraphExecutor,
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
            ex.tracer,
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
        ex.logger.exception("Function node %s failed: %s", node.id, e)
        return NodeExecutionResult(
            node_id=node.id,
            mode=node.mode,
            raw_output=None,
            started_at=started_at,
            duration_ms=int((time.time() - started_at) * 1000),
            error=str(e),
        )


def _render_tool_args(
    ex: GraphExecutor, node: GraphNode, args: Any, scope: dict[str, Any]
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
        return {k: _render_tool_args(ex, node, v, scope) for k, v in args.items()}
    if isinstance(args, list):
        return [_render_tool_args(ex, node, v, scope) for v in args]
    return args


def _render_tool_settings(
    ex: GraphExecutor, node: GraphNode, scope: dict[str, Any]
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
            out[str(tool_name)] = _render_tool_args(ex, node, dict(values), scope)
    return out


def run_tool_node(
    ex: GraphExecutor,
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
        tool_args = _render_tool_args(ex, node, node.tool_args or {}, render_ctx)
        kwargs = {**graph_inputs, **dependency_results, **iter_scope, **tool_args}

        if ex.native_tools is None:
            raise GraphConfigurationError(
                f"tool node '{node.id}' requires native_tools (a ToolPool) "
                "but none was provided to the executor."
            )
        if ex._tool_ctx is None:
            raise GraphConfigurationError(
                f"tool node '{node.id}' requires a tool_ctx (ToolContext) "
                "but none was provided to the executor."
            )
        # A `tool` node's one tool, configured. Same mechanism as an attached
        # tool's — the node kind decides who composes the call, never where
        # the call is allowed to land.
        native_tools = ex.native_tools
        if node.tool_settings:
            from ..configured_tools import configure_pool

            native_tools = configure_pool(
                native_tools, _render_tool_settings(ex, node, render_ctx)
            )

        from ..node_runner import run_tool_node
        with _trace_step(
            ex.tracer,
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
            raw = run_tool_node(native_tools, tool_name, kwargs, ex._tool_ctx)
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
        ex.logger.exception("Tool node %s failed: %s", node.id, e)
        return NodeExecutionResult(
            node_id=node.id,
            mode=node.mode,
            raw_output=None,
            started_at=started_at,
            duration_ms=int((time.time() - started_at) * 1000),
            error=str(e),
        )
