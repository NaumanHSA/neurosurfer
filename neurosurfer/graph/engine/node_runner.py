"""Native-stack execution for base and react workflow nodes (R3+R4).

Called from GraphExecutor._run_node() when a provider + native_tools pair is
supplied to the executor.  Replaces the vendored _runtime Agent / ReActAgent.

- base nodes (no tools): single provider.complete() call.
- base nodes (output_schema): structured_completion() via native tool-use.
- react nodes: native agents.Agent.run_collect() with a ToolPool subset.
- tool nodes: direct native Tool.run() call (no BaseTool adapter).
"""
from __future__ import annotations

import asyncio
import contextvars
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from neurosurfer.agents.agentic_loop import AgenticLoop
from neurosurfer.agents.oneshot import Agent as OneShotAgent
from neurosurfer.agents.react import ReactAgent
from neurosurfer.agents.runtime.permissions import Guardrails
from neurosurfer.llm.base import Provider
from neurosurfer.llm.types import GenerationConfig, ToolUseBlock, Usage
from neurosurfer.tools.base import AutoApproveIOHandler, ToolContext, ToolPool

from .errors import GraphConfigurationError

# ── async→sync bridge ────────────────────────────────────────────────────────

def run_coro_blocking(coro: Any) -> Any:
    """Run an async coroutine from synchronous (graph executor) code.

    Uses asyncio.run() when no loop is running; otherwise spawns a fresh
    daemon thread with its own loop to avoid "loop already running" errors.

    The thread inherits a snapshot of the caller's ``contextvars`` (via
    :func:`contextvars.copy_context`) so ambient state — notably the
    observability :class:`~neurosurfer.observability.context.TraceContext` — rides
    across the thread boundary and node agents still nest under the workflow trace.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    box: dict[str, Any] = {}
    ctx = contextvars.copy_context()

    def _thread() -> None:
        try:
            box["value"] = ctx.run(asyncio.run, coro)
        except BaseException as exc:  # noqa: BLE001
            box["error"] = exc

    t = threading.Thread(target=_thread, daemon=True)
    t.start()
    t.join()
    if "error" in box:
        raise box["error"]
    return box["value"]


# ── headless IO for workflow nodes ────────────────────────────────────────────
# Non-interactive nodes run unattended, so they auto-approve every gated step.
_HeadlessIO = AutoApproveIOHandler


_WORKFLOW_GUARDRAILS = Guardrails(
    max_turns=50,
    shell_policy="gated",
    network_policy="open",
    write_scope=["**"],
)


# ── per-kind node runners ─────────────────────────────────────────────────────

def _tool_names(agent: Any) -> list[str]:
    """Tools the agent actually invoked, in order, read back off its history."""
    names: list[str] = []
    try:
        for msg in agent.history.messages:
            for block in msg.content:
                if isinstance(block, ToolUseBlock):
                    names.append(block.name)
    except Exception:  # noqa: BLE001 - telemetry must never fail a node
        return names
    return names


@dataclass
class NodeCall:
    """What one node's LLM work produced: the output plus what it cost.

    Node runners used to return the output alone, so the token usage the agent had
    already tallied was discarded at the boundary and never reached the run record
    or the API. Callers that only want the value can use ``.output``.
    """

    output: Any
    usage: Usage = field(default_factory=Usage)
    tool_calls: list[str] = field(default_factory=list)


def run_base_node(
    provider: Provider,
    system_prompt: str,
    user_prompt: str,
    *,
    tool_pool: ToolPool | None = None,
    tool_ctx: ToolContext | None = None,
    output_schema: type | None = None,
    gen_config: GenerationConfig | None = None,
) -> NodeCall:
    """Execute a base node: one LLM call, optionally with tools.

    ``NodeCall.output`` is a Pydantic model instance when output_schema is set,
    plain text otherwise.

    **With tools, this is the rung between `base` and `react`.** `OneShotAgent`
    offers the tools, lets the model call them for as many rounds as the kind
    declares (`base` says one — see `kinds/base.py`), feeds the results back and
    takes the answer. That is the right shape for "fetch this, then tell me about
    it", which needed a `react` node and its whole loop before.

    The budget is read from the spec rather than written here as a literal, so
    the card, validation and the Architect all describe the same limit the
    executor enforces.

    An empty pool leaves the behaviour exactly as it was: one call, no tools
    offered. Nothing changes for a node that never had any.

    ``output_schema`` and tools do not combine — the structured path in
    `OneShotAgent` answers in one shot by construction and never reaches the tool
    loop. A node asking for both gets the schema, which is the field it declared
    explicitly.
    """
    cfg = gen_config or GenerationConfig(
        max_tokens=provider.capabilities.max_output_tokens
    )
    pool = tool_pool if tool_pool is not None else ToolPool([])

    from .kinds import NODE_KIND_SPECS

    rounds = NODE_KIND_SPECS["base"].tool_rounds

    async def _run() -> NodeCall:
        agent = OneShotAgent(
            provider=provider,
            tools=pool,
            system_prompt=system_prompt,
            guardrails=_WORKFLOW_GUARDRAILS,
            io=(tool_ctx.io if tool_ctx else _HeadlessIO()),
            cwd=(tool_ctx.cwd if tool_ctx else Path.cwd()),
            gen_config=cfg,
            mode="bypass",
            output_schema=output_schema,
            max_tool_rounds=rounds if rounds is not None else 1,
        )
        out = await agent.complete(user_prompt)

        # A cut-short run that produced nothing is a failure, not a result.
        #
        # `base` gets one round of tool calls. A model asked to "fetch the page
        # then write it to a file" spends that round on the fetch, is refused the
        # second, and its last turn is tool calls with no prose — so `out` is `""`.
        # Reported as success, that is a green run, three green nodes, and a blank
        # answer, with nothing anywhere saying the plan was truncated.
        #
        # **Empty is the line, not cut-short.** A truncated run that still produced
        # text produced a partial answer, and a partial answer is sometimes exactly
        # what was wanted; refusing it would fail runs that are useful today. What
        # is never useful is nothing at all.
        if getattr(agent, "cut_short", False) and not str(out or "").strip():
            names = ", ".join(dict.fromkeys(_tool_names(agent))) or "its tools"
            raise GraphConfigurationError(
                "This step ran out of tool calls before it finished. It called "
                f"{names}, still wanted to call more, and returned nothing. A "
                "`base` step gets one round of tool calls, so it cannot use one "
                "tool's result to decide the next — use kind `react` for work "
                "that needs a sequence."
            )

        return NodeCall(output=out, usage=agent.usage, tool_calls=_tool_names(agent))

    return run_coro_blocking(_run())


def run_react_node(
    provider: Provider,
    tool_pool: ToolPool,
    tool_ctx: ToolContext,
    system_prompt: str,
    user_prompt: str,
    *,
    gen_config: GenerationConfig | None = None,
    output_schema: type | None = None,
) -> NodeCall:
    """Execute a react (tool-using agent loop) node via the native agents.Agent.

    **With `output_schema`, the loop runs first and its answer is then shaped.**
    Two steps rather than one, deliberately: a react node's whole purpose is to
    decide what to do next from what the last tool returned, and constraining
    every turn to a schema would break that. So the loop works normally, and one
    structured call afterwards turns its prose into the declared shape.

    That costs one extra model call, which is the honest price of the feature —
    and it is why the field was withdrawn rather than faked: the alternative was
    a spec field that silently returned prose.
    """
    cfg = gen_config or GenerationConfig(
        max_tokens=provider.capabilities.max_output_tokens
    )
    io = _HeadlessIO()
    # Native tool-use ⇒ the AgenticLoop; providers without it ⇒ text-parsing ReAct.
    native = getattr(provider.capabilities, "tool_call_style", None) in ("anthropic", "openai")
    agent_cls = AgenticLoop if native else ReactAgent

    async def _run() -> NodeCall:
        agent = agent_cls(
            provider=provider,
            tools=tool_pool,
            system_prompt=system_prompt,
            guardrails=_WORKFLOW_GUARDRAILS,
            io=io,
            cwd=tool_ctx.cwd,
            gen_config=cfg,
            mode="bypass",
        )
        result = await agent.run_collect(user_prompt)

        # **An agent that ends with `finish()` puts its answer in `report`.**
        #
        # `RunResult` has two channels: `final_text` accumulates text deltas, and
        # `report` carries what `RunFinished` was given. A model that does its
        # work and then calls `finish(summary=…)` — which is the shape the finish
        # tool exists to encourage — produces *no* text deltas at all, so reading
        # `final_text` alone returned `""` from a node that had just done the
        # work and written the answer down.
        #
        # Nothing failed. The node was green, its tool calls were in the trace,
        # and the *next* node was handed an empty string and improvised around
        # it: "I don't have the scout findings above to summarise." One node's
        # silence became the next node's invention.
        #
        # `subagents/runner.py` had this right already (`report or final_text or
        # …`); this is the same rule, in the place a workflow reads it.
        # **And a thinking-only turn is an answer, not a silence.**
        #
        # A local reasoning model sometimes ends its last turn having emitted
        # only `ThinkingDelta` — no tool call, no text. `final_text` is then
        # empty for a run that plainly produced something, and this node failed,
        # taking every node downstream with it. Measured on `qwen/qwen3.5-9b`
        # driving the capstone tutorial's vision node: two runs in six.
        #
        # `CanonicalResponse.text()` already resolves this the same way for the
        # one-shot path — *"thinking-only models put everything in
        # reasoning_content; fall back so callers always get a non-empty
        # string"*. The streamed path disagreed with it purely because it
        # accumulates deltas. Last in the order, so a real answer always wins.
        output = (
            (result.report or "").strip()
            or (result.final_text or "").strip()
            or (result.final_thinking or "").strip()
        )

        # Empty is the line, exactly as it is for `base` — see `run_base_node`. A
        # react node that ran tools and produced no answer has not succeeded at
        # anything a downstream node can use.
        if not output:
            names = ", ".join(dict.fromkeys(_tool_names(agent))) or "no tools"
            raise GraphConfigurationError(
                f"This step finished without producing an answer. It called "
                f"{names} and returned nothing, so anything downstream would be "
                f"working from an empty result. Check that the step's "
                f"instructions ask for an answer, not only for actions."
            )

        usage = result.usage or Usage()

        if output_schema is not None:
            from neurosurfer.agents.runtime.structured import structured_completion

            shaping = Usage()

            def _add(u: Usage) -> None:
                nonlocal shaping
                shaping = shaping.add(u)

            # The loop's answer is the *input* to the shaping call, not the
            # original task: re-running the task without the tools would ask the
            # model to invent what it just spent a loop finding out.
            output = await structured_completion(
                provider,
                output_schema,
                user=(
                    "Convert the following result into the required structure. "
                    "Use only what it says; do not add facts.\n\n" + output
                ),
                system="You restructure an existing answer. You never invent content.",
                config=cfg,
                on_usage=_add,
            )
            # Shaping is billed to the node that asked for it — otherwise a
            # structured react node under-reports its cost by a whole call.
            usage = usage.add(shaping)

        return NodeCall(
            output=output,
            usage=usage,
            tool_calls=_tool_names(agent),
        )

    return run_coro_blocking(_run())


def run_tool_node(
    tool_pool: ToolPool,
    tool_name: str,
    kwargs: dict[str, Any],
    tool_ctx: ToolContext,
) -> Any:
    """Execute a tool node directly via the native ToolPool (no adapter needed)."""
    tool = tool_pool.get(tool_name)
    if tool is None:
        raise KeyError(f"Tool {tool_name!r} not found in ToolPool")

    async def _run() -> Any:
        result = await tool.run(kwargs, tool_ctx)
        if result.is_error:
            raise RuntimeError(result.content)
        return result.content

    return run_coro_blocking(_run())
