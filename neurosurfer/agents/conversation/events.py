"""Events the agent loop yields to the front-end (CLI / SDK / tests).

The loop is an async generator of these; the consumer renders them. Approvals are
handled out-of-band via the IOHandler (so they can block), not as events.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from neurosurfer.llm.types import Usage
from neurosurfer.tools.base import ToolResult


@dataclass
class TextDelta:
    text: str


@dataclass
class ThinkingDelta:
    text: str


@dataclass
class ToolStarted:
    id: str
    name: str
    args: dict[str, Any]


@dataclass
class ToolFinished:
    id: str
    name: str
    result: ToolResult


@dataclass
class TurnCompleted:
    usage: Usage
    stop_reason: str


@dataclass
class ModeChanged:
    mode: str
    reason: str = ""


@dataclass
class Compacted:
    tokens_before: int
    tokens_after: int


@dataclass
class RunFinished:
    status: str
    report: str = ""


@dataclass
class AgentError:
    message: str


Event = (
    TextDelta
    | ThinkingDelta
    | ToolStarted
    | ToolFinished
    | TurnCompleted
    | ModeChanged
    | Compacted
    | RunFinished
    | AgentError
)


@dataclass
class RunResult:
    """Aggregate outcome of an Agent.run() invocation."""

    final_text: str = ""
    status: str = "completed"
    report: str = ""
    usage: Usage = field(default_factory=Usage)
    turns: int = 0
