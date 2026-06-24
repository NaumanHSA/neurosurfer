"""Pydantic output schemas for the Architect workflow nodes.

These are referenced by ``output_schema`` fields in the architect's graph.yaml
and imported at runtime via ``import_string``.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# ---------------------------------------------------------------------------
# Node 1 — discover
# ---------------------------------------------------------------------------

class ClarifyingQuestion(BaseModel):
    """A single clarifying question with exactly 3 choices."""

    id: str = Field(description="Short snake_case identifier for this question.")
    question: str = Field(description="The question text shown to the user.")
    choices: list[str] = Field(
        description="Exactly 3 descriptive options the user picks from.",
        min_length=3,
        max_length=3,
    )

    @field_validator("choices")
    @classmethod
    def _exactly_three(cls, v: list[str]) -> list[str]:
        if len(v) != 3:
            raise ValueError(f"choices must have exactly 3 items, got {len(v)}")
        return v


class DiscoveryOutput(BaseModel):
    """Structured output from the discover node."""

    model_config = ConfigDict(populate_by_name=True)

    summary: str = Field(
        description="One-sentence summary of what the user wants to achieve."
    )
    web_findings: list[str] = Field(
        description=(
            "List of key findings from web research — each entry is a short string "
            "describing one tool, pattern, or insight relevant to the user's request."
        ),
        min_length=1,
    )
    questions: list[ClarifyingQuestion] = Field(
        description="3-4 clarifying questions to guide workflow design.",
        min_length=2,
        max_length=5,
    )

    @model_validator(mode="before")
    @classmethod
    def _normalize(cls, data: object) -> object:
        if not isinstance(data, dict):
            return data
        # Accept 'clarifying_questions' as alias for 'questions'.
        if "questions" not in data and "clarifying_questions" in data:
            data["questions"] = data.pop("clarifying_questions")
        # Accept web_findings as a list of dicts — convert each to a readable string.
        wf = data.get("web_findings")
        if isinstance(wf, list):
            normalized = []
            for item in wf:
                if isinstance(item, dict):
                    parts = []
                    for k, v in item.items():
                        parts.append(f"{k}: {v}")
                    normalized.append("; ".join(parts))
                elif isinstance(item, str):
                    normalized.append(item)
                else:
                    normalized.append(str(item))
            data["web_findings"] = normalized
        elif isinstance(wf, str):
            # Backwards compat: wrap a plain string in a list.
            data["web_findings"] = [wf]
        return data


# ---------------------------------------------------------------------------
# Node 3 — plan
# ---------------------------------------------------------------------------

# The Architect can only author 'base' (single LLM call) and 'react' (LLM + tools)
# nodes — it cannot write Python callables for 'function'/'python' nodes or wire
# 'tool' nodes reliably, so those are coerced to 'react' (the most capable kind).
NodeKind = Literal["base", "react"]


class NodePlan(BaseModel):
    """Plan for a single node in the designed workflow."""

    id: str = Field(description="Unique node identifier (snake_case).")
    kind: NodeKind = Field(
        default="base",
        description="Node kind: 'base' (a single LLM call — for writing/generating/"
                    "summarising text) or 'react' (an LLM that calls tools — for file "
                    "I/O, search, shell, web). No other kinds are allowed.",
    )

    @field_validator("kind", mode="before")
    @classmethod
    def _coerce_kind(cls, v: object) -> object:
        # Map kinds the Architect can't fulfil onto 'react' so the plan still parses
        # (the strengthened prompt steers it to base/react in the first place).
        if isinstance(v, str) and v.strip().lower() in {"function", "python", "tool"}:
            return "react"
        return v
    purpose: str = Field(description="What this node does — used as the system prompt.")
    goal: str = Field(
        default="",
        description="Specific goal / success criterion for this node.",
    )
    expected_result: str = Field(
        default="",
        description="What the node's output should look like.",
    )
    depends_on: list[str] = Field(
        default_factory=list,
        description="IDs of nodes this node waits for before running.",
    )
    tools: list[str] = Field(
        default_factory=list,
        description="Tool names available to this node (for base/react kinds).",
    )
    mode: str = Field(
        default="text",
        description="Output mode: text, json, structured, or tool.",
    )
    output_schema: str | None = Field(
        default=None,
        description="Import path to a Pydantic model for structured output (module:Class).",
    )
    callable: str | None = Field(
        default=None,
        description="Import path to the Python callable for function-kind nodes.",
    )
    tool_args: dict | None = Field(
        default=None,
        description="Static kwargs for tool-kind nodes.",
    )
    rationale: str = Field(
        default="",
        description="Why this node exists — used for documentation.",
    )

    @field_validator("id")
    @classmethod
    def _snake_case(cls, v: str) -> str:
        v = v.strip().lower().replace(" ", "_").replace("-", "_")
        if not v:
            raise ValueError("node id must not be empty")
        return v


class WorkflowPlan(BaseModel):
    """Complete workflow design produced by the plan node."""

    name: str = Field(description="Workflow name (snake_case, no spaces).")
    description: str = Field(description="One-sentence description of what this workflow does.")
    nodes: list[NodePlan] = Field(
        description="All nodes in execution order (topological).",
        min_length=1,
    )
    outputs: list[str] = Field(
        description="IDs of nodes whose output is the final workflow result.",
        min_length=1,
    )

    @field_validator("name")
    @classmethod
    def _clean_name(cls, v: str) -> str:
        import re
        v = re.sub(r"[^a-z0-9_]", "_", v.lower().strip())
        v = re.sub(r"_+", "_", v).strip("_")
        return v or "workflow"
