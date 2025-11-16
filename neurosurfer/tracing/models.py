# neurosurfer/tracing/models.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Literal
from pydantic import BaseModel, Field, model_validator


class TraceLog(BaseModel):
    """
    A single log entry inside a traced step.
    """
    ts: float
    message: str
    data: Dict[str, Any] = Field(default_factory=dict)
    type: Literal["info", "warning", "error", "debug"] = "info"

class TraceStep(BaseModel):
    """
    One traced step in a workflow.

    Flexible enough to cover:
      - LLM calls
      - tool calls
      - any custom step

    You decide how to populate `inputs`, `outputs`, and `meta`.
    """

    step_id: int
    kind: str                       # e.g. "llm", "tool", "graph", "other"
    label: Optional[str] = None     # e.g. "agent.llm.ask", "tool.web_search"
    node_id: Optional[str] = None   # graph node id (if any)
    agent_id: Optional[str] = None  # agent id/name (if any)

    started_at: float
    duration_ms: int

    inputs: Dict[str, Any] = Field(default_factory=dict)
    outputs: Dict[str, Any] = Field(default_factory=dict)
    meta: Dict[str, Any] = Field(default_factory=dict)

    ok: bool
    error: Optional[str] = None

    logs: List[TraceLog] = Field(default_factory=list)


class TraceResult(BaseModel):
    """
    Structured tracing results for a single run.

    You can attach this object directly to Agent / GraphExecutor outputs.
    """

    steps: List[TraceStep] = Field(default_factory=list)
    meta: Dict[str, Any] = Field(default_factory=dict)

    def summary(self) -> str:
        ok_count = sum(1 for s in self.steps if s.ok)
        err_count = sum(1 for s in self.steps if not s.ok)
        return (
            f"TraceResult(steps={len(self.steps)}, ok={ok_count}, "
            f"errors={err_count})"
        )

    @model_validator(mode="after")
    def _sort_steps_by_step_id(cls, m: "TraceResult") -> "TraceResult":
        """
        Ensure steps are always ordered by step_id ascending.

        So step_id=0 is first, step_id=n is last, regardless of insert order.
        """
        m.steps = sorted(m.steps, key=lambda s: s.step_id)
        return m
