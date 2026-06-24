"""Conversation plumbing: model-facing message history + the event stream."""
from . import events  # noqa: F401  (also exposed as neurosurfer.agents.events)
from .events import (  # noqa: F401
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
from .messages import MessageHistory  # noqa: F401
