"""BaseAgent — shared state + plumbing for every agent type.

Holds the provider / tools / history / permissions / context-manager wiring and the
model-streaming helper. Subclasses implement :meth:`_run` with their own strategy
(the public :meth:`run` wraps it to emit the verbose activity trace):

- :class:`~neurosurfer.agents.agentic_loop.AgenticLoop` — multi-step native tool-use.
- :class:`~neurosurfer.agents.react.ReactAgent` — multi-step text-parsing ReAct
  (for providers without a native tool-calling API).
- :class:`~neurosurfer.agents.oneshot.Agent` — a single bounded call (+ optional tools
  / structured output).
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from neurosurfer.agents.conversation import events
from neurosurfer.agents.conversation.messages import MessageHistory
from neurosurfer.agents.runtime.permissions import Guardrails, PermissionMode, Permissions
from neurosurfer.agents.trace import AgentTrace
from neurosurfer.llm import types as lt
from neurosurfer.llm.base import Provider
from neurosurfer.observability.context import TraceContext, current_trace_context
from neurosurfer.observability.exporters import get_active_exporters
from neurosurfer.observability.exporters.stream import TraceStreamObserver
from neurosurfer.tools.base import (
    AutoApproveIOHandler,
    IOHandler,
    SpawnFn,
    TerminalIOHandler,
    ToolContext,
    ToolPool,
)

# How a gated tool (shell / write / network / plan) gets its yes-or-no:
#   "auto" → approve everything, never block (the zero-config default)
#   "ask"  → prompt the human on stdin (terminal or notebook)
#   an IOHandler instance → your own approval UI (CLI, server, web, …)
Approval = Literal["auto", "ask"] | IOHandler


def _resolve_io(approval: Approval) -> IOHandler:
    if isinstance(approval, str):
        if approval == "auto":
            return AutoApproveIOHandler()
        if approval == "ask":
            return TerminalIOHandler()
        raise ValueError(
            f"approval must be 'auto', 'ask', or an IOHandler — got {approval!r}"
        )
    return approval  # already an IOHandler

if TYPE_CHECKING:
    from neurosurfer.agents.context.durable_state import DurableState
    from neurosurfer.agents.context.manager import ContextManager


class BaseAgent:
    """Common construction + streaming shared by all agent types."""

    def __init__(
        self,
        *,
        provider: Provider,
        tools: ToolPool,
        system_prompt: str,
        guardrails: Guardrails,
        approval: Approval = "auto",
        io: IOHandler | None = None,
        cwd: Path | None = None,
        gen_config: lt.GenerationConfig | None = None,
        mode: PermissionMode = "default",
        durable: DurableState | None = None,
        spawn: SpawnFn | None = None,
        context_manager: ContextManager | None = None,
        depth: int = 0,
        persist_scope: Callable[[str], None] | None = None,
        verbose: bool = True,
        session_id: str | None = None,
        trace_name: str | None = None,
    ):
        self.provider = provider
        self.tools = tools
        self.system_prompt = system_prompt
        self.guardrails = guardrails
        # ``io`` (an explicit handler) always wins for back-compat; otherwise the
        # friendly ``approval`` preset picks one so callers never hand-write a handler.
        self.io = io if io is not None else _resolve_io(approval)
        cwd = cwd if cwd is not None else Path.cwd()
        self.cwd = cwd
        self.gen_config = gen_config or lt.GenerationConfig(
            max_tokens=provider.capabilities.max_output_tokens
        )
        self.mode: PermissionMode = mode
        self.durable = durable
        self.depth = depth
        self.context_manager = context_manager
        # Optional grouping id: all runs of this agent share it, so a multi-message
        # CLI conversation appears as one Langfuse *session* instead of N traces.
        self.session_id = session_id
        # Optional human-friendly trace name; falls back to ``<AgentType>.run``.
        self._trace_name = trace_name
        # When True, the agent renders a live activity trace (animated spinner +
        # tool lines) as events flow past — so a caller that only handles ``TextDelta``
        # still sees the agent working. Front-ends with their own renderer (the CLI,
        # nested sub-agents) pass ``verbose=False``.
        self.verbose = verbose

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
        )

    # ── system prompt (dynamic suffix injected by context manager) ────────────
    def _effective_system(self) -> str:
        if self.context_manager is not None:
            return self.context_manager.system_with_durable(self.system_prompt)
        return self.system_prompt

    async def _stream_model(
        self, *, tool_schemas: list | None = None, system: str | None = None
    ) -> AsyncIterator[lt.StreamEvent]:
        """Stream one model turn, applying reactive compaction on overflow.

        ``tool_schemas`` defaults to the full tool pool's schemas (native tool-use).
        Pass ``[]`` for prompt-driven agents (e.g. ReAct) that must not advertise
        native tools to the provider.

        ``system`` overrides the agent's base system prompt for this turn. Prompt-driven
        agents (e.g. ReAct) pass their assembled prompt — with the tool catalog and
        format spec — here so the model actually receives it; otherwise the bare
        ``self.system_prompt`` is used.
        """
        schemas = self.tools.schemas() if tool_schemas is None else tool_schemas
        effective_system = system if system is not None else self._effective_system()
        if self.context_manager is not None:
            async for ev in self.context_manager.stream_with_recovery(
                self.provider,
                self.history,
                effective_system,
                schemas,
                self.gen_config,
            ):
                yield ev
            return
        async for ev in self.provider.stream(
            self.history.messages,
            effective_system,
            schemas,
            self.gen_config,
        ):
            yield ev

    # ── public entry points ───────────────────────────────────────────────────
    async def run(self, user_input: str) -> AsyncIterator[events.Event]:
        """Run the agent over *user_input*, yielding events.

        The per-type strategy lives in :meth:`_run`; this wrapper taps the stream to
        emit the activity trace (see :attr:`verbose`). Events are passed through
        unchanged, so consuming them is identical whether ``verbose`` is on or off.
        """
        async for ev in self._tap(self._run(user_input), user_input=user_input):
            yield ev

    async def _run(self, user_input: str) -> AsyncIterator[events.Event]:
        """The agent's event stream. Implemented per type."""
        raise NotImplementedError
        yield  # pragma: no cover — makes this an async generator for typing

    async def _tap(
        self, stream: AsyncIterator[events.Event], *, user_input: str = ""
    ) -> AsyncIterator[events.Event]:
        """Yield every event untouched, driving side-channel observers alongside it.

        Independent of how the caller consumes the events: even a minimal
        ``async for ev in agent.run(task)`` that only handles ``TextDelta`` still shows
        the animated thinking/tool spinner when ``verbose`` is set, and still ships a
        trace to any configured backend (Langfuse / OTel). Both observers are created
        per run and always torn down (they must not outlive the stream).
        """
        trace = AgentTrace() if self.verbose else None
        observer = self._make_trace_observer()
        if observer is not None:
            observer.start(input=user_input or None)
        try:
            async for ev in stream:
                if trace is not None:
                    trace.handle(ev)
                if observer is not None:
                    observer.handle(ev)
                yield ev
        except BaseException as exc:  # noqa: BLE001 — record then re-raise
            if observer is not None:
                observer.on_run_exception(exc)
            raise
        finally:
            if trace is not None:
                trace.close()
            if observer is not None:
                observer.close()

    def _make_trace_observer(self) -> TraceStreamObserver | None:
        """A per-run trace observer if any exporter is active, else ``None`` (no overhead).

        If this run starts *inside* another traced run (a spawned sub-agent, or a
        graph node's agent), it inherits that run's trace and nests under its span;
        otherwise it opens a fresh top-level trace.
        """
        exporters = get_active_exporters()
        if not exporters:
            return None
        model = getattr(self.provider, "model", None)
        meta = {
            "agent_type": type(self).__name__,
            "provider": type(self.provider).__name__,
            "model": model,
        }
        parent = current_trace_context()
        if parent is not None:
            ctx = parent.child(metadata={**parent.metadata, **meta})
        else:
            ctx = TraceContext(session_id=self.session_id, metadata=meta)
        name = self._trace_name or f"{type(self).__name__}.run"
        return TraceStreamObserver(ctx, exporters, model=model, name=name)

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
