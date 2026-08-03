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
    offers the tools, lets the model call them once (`max_tool_rounds=1`), feeds the
    results back and takes the answer — a single round rather than a loop. That is
    the right shape for "fetch this, then tell me about it", which needed a `react`
    node and its whole loop before.

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
        )
        out = await agent.complete(user_prompt)
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
) -> NodeCall:
    """Execute a react (tool-using agent loop) node via the native agents.Agent."""
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
        return NodeCall(
            output=result.final_text,
            usage=result.usage or Usage(),
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
