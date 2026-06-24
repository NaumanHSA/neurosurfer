"""BaseAgent — shared state + plumbing for every agent type.

Holds the provider / tools / history / permissions / context-manager wiring and the
model-streaming helper. Subclasses implement :meth:`run` with their own strategy:

- :class:`~neurosurfer.agents.agentic_loop.AgenticLoop` — multi-step native tool-use.
- :class:`~neurosurfer.agents.react_agent.ReactAgent` — multi-step text-parsing ReAct
  (for providers without a native tool-calling API).
- :class:`~neurosurfer.agents.oneshot.Agent` — a single bounded call (+ optional tools
  / structured output).
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import TYPE_CHECKING

from neurosurfer.agents.conversation import events
from neurosurfer.agents.conversation.messages import MessageHistory
from neurosurfer.agents.runtime.permissions import Guardrails, PermissionMode, Permissions
from neurosurfer.llm import types as lt
from neurosurfer.llm.base import Provider
from neurosurfer.tools.base import IOHandler, SpawnFn, ToolContext, ToolPool

if TYPE_CHECKING:
    from neurosurfer.agents.context.durable_state import DurableState
    from neurosurfer.agents.context.manager import ContextManager
    from neurosurfer.memory.store import MemoryStore


class BaseAgent:
    """Common construction + streaming shared by all agent types."""

    def __init__(
        self,
        *,
        provider: Provider,
        tools: ToolPool,
        system_prompt: str,
        guardrails: Guardrails,
        io: IOHandler,
        cwd: Path,
        gen_config: lt.GenerationConfig | None = None,
        mode: PermissionMode = "default",
        durable: DurableState | None = None,
        spawn: SpawnFn | None = None,
        context_manager: ContextManager | None = None,
        depth: int = 0,
        persist_scope: Callable[[str], None] | None = None,
        memory: MemoryStore | None = None,
        memory_agent: str | None = None,
        session_id: str | None = None,
    ):
        self.provider = provider
        self.tools = tools
        self.system_prompt = system_prompt
        self.guardrails = guardrails
        self.io = io
        self.cwd = cwd
        self.gen_config = gen_config or lt.GenerationConfig(
            max_tokens=provider.capabilities.max_output_tokens
        )
        self.mode: PermissionMode = mode
        self.durable = durable
        self.depth = depth
        self.context_manager = context_manager

        self.history = MessageHistory()
        self.file_state: dict = {}
        self.permissions = Permissions(guardrails, cwd)
        self.usage = lt.Usage()
        self.turns = 0

        self._ctx = ToolContext(
            cwd=cwd,
            io=io,
            file_state=self.file_state,
            durable=durable,
            spawn=spawn,
            guardrails=guardrails,
            depth=depth,
            persist_scope=persist_scope,
            memory=memory,
            memory_agent=memory_agent,
            session_id=session_id,
        )

    # ── system prompt (dynamic suffix injected by context manager) ────────────
    def _effective_system(self) -> str:
        if self.context_manager is not None:
            return self.context_manager.system_with_durable(self.system_prompt)
        return self.system_prompt

    async def _stream_model(
        self, *, tool_schemas: list | None = None
    ) -> AsyncIterator[lt.StreamEvent]:
        """Stream one model turn, applying reactive compaction on overflow.

        ``tool_schemas`` defaults to the full tool pool's schemas (native tool-use).
        Pass ``[]`` for prompt-driven agents (e.g. ReAct) that must not advertise
        native tools to the provider.
        """
        schemas = self.tools.schemas() if tool_schemas is None else tool_schemas
        if self.context_manager is not None:
            async for ev in self.context_manager.stream_with_recovery(
                self.provider,
                self.history,
                self._effective_system(),
                schemas,
                self.gen_config,
            ):
                yield ev
            return
        async for ev in self.provider.stream(
            self.history.messages,
            self._effective_system(),
            schemas,
            self.gen_config,
        ):
            yield ev

    # ── public entry points ───────────────────────────────────────────────────
    async def run(self, user_input: str) -> AsyncIterator[events.Event]:
        """Run the agent over *user_input*, yielding events. Implemented per type."""
        raise NotImplementedError
        yield  # pragma: no cover — makes this an async generator for typing

    async def run_collect(self, user_input: str) -> events.RunResult:
        """Drive :meth:`run` to completion and collect a :class:`RunResult`."""
        result = events.RunResult()
        text_parts: list[str] = []
        async for ev in self.run(user_input):
            if isinstance(ev, events.TextDelta):
                text_parts.append(ev.text)
            elif isinstance(ev, events.RunFinished):
                result.status = ev.status
                result.report = ev.report
        result.final_text = "".join(text_parts)
        result.usage = self.usage
        result.turns = self.turns
        return result
