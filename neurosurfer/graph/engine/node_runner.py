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
import threading
from pathlib import Path
from typing import Any

from neurosurfer.agents.agentic_loop import AgenticLoop
from neurosurfer.agents.oneshot import Agent as OneShotAgent
from neurosurfer.agents.react_agent import ReactAgent
from neurosurfer.agents.runtime.permissions import Guardrails
from neurosurfer.llm.base import Provider
from neurosurfer.llm.types import GenerationConfig
from neurosurfer.tools.base import ToolContext, ToolPool

# ── async→sync bridge ────────────────────────────────────────────────────────

def run_coro_blocking(coro: Any) -> Any:
    """Run an async coroutine from synchronous (graph executor) code.

    Uses asyncio.run() when no loop is running; otherwise spawns a fresh
    daemon thread with its own loop to avoid "loop already running" errors.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    box: dict[str, Any] = {}

    def _thread() -> None:
        try:
            box["value"] = asyncio.run(coro)
        except BaseException as exc:  # noqa: BLE001
            box["error"] = exc

    t = threading.Thread(target=_thread, daemon=True)
    t.start()
    t.join()
    if "error" in box:
        raise box["error"]
    return box["value"]


# ── headless IO for workflow nodes ────────────────────────────────────────────

class _HeadlessIO:
    """Non-interactive IO: auto-approves everything; no prompts."""

    async def ask(self, question: str, options: list[str] | None = None) -> str:
        return ""

    async def request_plan_approval(self, plan: str) -> tuple[bool, str]:
        return (True, "")

    async def request_shell_approval(self, command: str, reason: str) -> bool:
        return True

    async def request_write_approval(self, path: str, summary: str) -> str:
        return "always"

    def notify(self, message: str) -> None:
        pass


_WORKFLOW_GUARDRAILS = Guardrails(
    max_turns=50,
    shell_policy="gated",
    network_policy="open",
    write_scope=["**"],
)


# ── per-kind node runners ─────────────────────────────────────────────────────

def run_base_node(
    provider: Provider,
    system_prompt: str,
    user_prompt: str,
    *,
    output_schema: type | None = None,
    gen_config: GenerationConfig | None = None,
) -> Any:
    """Execute a base (single-LLM-call) node on the native provider stack.

    Returns a Pydantic model instance when output_schema is set, plain text
    otherwise.
    """
    cfg = gen_config or GenerationConfig(
        max_tokens=provider.capabilities.max_output_tokens
    )

    async def _run() -> Any:
        agent = OneShotAgent(
            provider=provider,
            tools=ToolPool([]),
            system_prompt=system_prompt,
            guardrails=_WORKFLOW_GUARDRAILS,
            io=_HeadlessIO(),
            cwd=Path.cwd(),
            gen_config=cfg,
            mode="bypass",
            output_schema=output_schema,
        )
        return await agent.complete(user_prompt)

    return run_coro_blocking(_run())


def run_react_node(
    provider: Provider,
    tool_pool: ToolPool,
    tool_ctx: ToolContext,
    system_prompt: str,
    user_prompt: str,
    *,
    gen_config: GenerationConfig | None = None,
) -> str:
    """Execute a react (tool-using agent loop) node via the native agents.Agent."""
    cfg = gen_config or GenerationConfig(
        max_tokens=provider.capabilities.max_output_tokens
    )
    io = _HeadlessIO()
    # Native tool-use ⇒ the AgenticLoop; providers without it ⇒ text-parsing ReAct.
    native = getattr(provider.capabilities, "tool_call_style", None) in ("anthropic", "openai")
    agent_cls = AgenticLoop if native else ReactAgent

    async def _run() -> str:
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
        return result.final_text

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
