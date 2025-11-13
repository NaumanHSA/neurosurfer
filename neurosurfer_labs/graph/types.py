from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional


class NodeMode(str, Enum):
    """How a graph node prefers to use its underlying Agent."""

    AUTO = "auto"          # Let Agent decide based on its config/toolkit
    TEXT = "text"          # Plain text output (no schema)
    STRUCTURED = "structured"  # Structured output via Pydantic schema
    TOOL = "tool"          # Node is primarily about using tools


@dataclass
class NodeExecutionResult:
    """
    Canonical result of a single node execution in the graph.

    This is what you can log, inspect, or show in a UI.
    """

    node_id: str
    mode: NodeMode
    raw_output: Any
    structured_output: Optional[Any]
    tool_call_output: Optional[Any]
    started_at: float
    duration_ms: int
    error: Optional[str] = None
