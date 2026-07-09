"""Events the agent loop yields to the front-end (CLI / SDK / tests).

The loop is an async generator of these; the consumer renders them. Approvals are
handled out-of-band via the IOHandler (so they can block), not as events.

The two text channels are the ones consumers care about most:

- :class:`TextDelta` — a chunk of the **user-facing answer**, streamed token by token.
  This is "the answer": concatenate the ``.text`` of every ``TextDelta`` to rebuild it,
  or render them live. (``RunFinished.report`` carries the same final answer in one
  piece for callers that don't stream.)
- :class:`ThinkingDelta` — a chunk of the model's **reasoning / scaffolding** (e.g. ReAct
  Thought/Action text, or a native thinking channel). Not part of the answer; front-ends
  typically collapse it to a "Thinking…" indicator.

You don't have to handle these to *see* what the agent is doing: when an agent runs with
``verbose=True`` (the default) it renders a live animated status (a spinner whose label shifts
while the model thinks, "Running <tool>…" while a tool runs, a line per tool call) as a
side-channel, independent of which events you consume. See :mod:`neurosurfer.agents.trace`.
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
    # Human-friendly status line (e.g. "Reading file README.md…"). Front-ends render
    # this; falls back to name/args when empty.
    title: str = ""


@dataclass
class ToolFinished:
    id: str
    name: str
    result: ToolResult


@dataclass
class TurnCompleted:
    usage: Usage
    stop_reason: str
    # Snapshot of the messages sent to the model for this turn — the LLM
    # generation's *input*. Carried here so trace exporters can record it.
    input: Any = None
    # The model's assistant response for this turn (thinking + text + tool_use
    # blocks) — the generation's *output*. Distinct from the streamed answer text.
    output: Any = None


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


@dataclass
class Notice:
    """A one-line informational message surfaced to the user (not to the model)."""

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
    | Notice
)


@dataclass
class RunResult:
    """Aggregate outcome of an Agent.run() invocation."""

    final_text: str = ""
    status: str = "completed"
    report: str = ""
    usage: Usage = field(default_factory=Usage)
    turns: int = 0
