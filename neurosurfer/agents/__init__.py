"""The agent family + the shared engine primitives it's built on.

Agent types (pick per task):
- :class:`~neurosurfer.agents.agentic_loop.AgenticLoop` — multi-step native tool-use.
- :class:`~neurosurfer.agents.react_agent.ReactAgent` — multi-step text-parsing ReAct
  (for providers without a native tool-calling API).
- :class:`~neurosurfer.agents.oneshot.Agent` — a single bounded call (+ optional tools
  / structured output).

Shared primitives live in subpackages: :mod:`~neurosurfer.agents.conversation`
(messages + events), :mod:`~neurosurfer.agents.context` (compaction / durable state),
:mod:`~neurosurfer.agents.runtime` (tool dispatch, permissions, structured output,
background tasks), :mod:`~neurosurfer.agents.subagents` (definitions + spawning).
Public names are re-exported here so ``from neurosurfer.agents import X`` is stable.
"""
from .agentic_loop import AgenticLoop
from .base import BaseAgent
from .context import ContextManager, DurableState
from .conversation import events  # noqa: F401  — back-compat: neurosurfer.agents.events
from .conversation.events import (
    AgentError,
    Compacted,
    Event,
    ModeChanged,
    RunFinished,
    RunResult,
    TextDelta,
    ThinkingDelta,
    ToolFinished,
    ToolStarted,
    TurnCompleted,
)
from .oneshot import Agent
from .react_agent import ReactAgent
from .runtime import Guardrails, PermissionMode, Permissions, TaskHandle, TasksRuntime, initial_mode
from .subagents import SubAgentRunner

__all__ = [
    "BaseAgent",
    "AgenticLoop",
    "ReactAgent",
    "Agent",
    "ContextManager",
    "DurableState",
    "Event",
    "RunResult",
    "TextDelta",
    "ThinkingDelta",
    "ToolStarted",
    "ToolFinished",
    "TurnCompleted",
    "ModeChanged",
    "Compacted",
    "RunFinished",
    "AgentError",
    "Guardrails",
    "Permissions",
    "PermissionMode",
    "initial_mode",
    "SubAgentRunner",
    "TaskHandle",
    "TasksRuntime",
]
