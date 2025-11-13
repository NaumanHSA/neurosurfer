from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field, field_validator

from .types import NodeMode


class GraphNode(BaseModel):
    """
    One node in a graph.

    Each node:
      - Is backed by an Agent instance created automatically by the executor.
      - Has PURPOSE / GOAL / EXPECTED_RESULT (used for manager + system prompt).
      - Declares dependencies via `depends_on` (forming a DAG).
      - Declares tools it is allowed to use.
      - Can optionally specify a Pydantic `output_schema` for structured mode.
      - Can optionally override `strict_tool_call`.
    """

    id: str = Field(..., description="Unique node identifier.")
    description: Optional[str] = Field(
        None, description="Short human description of this node."
    )
    purpose: Optional[str] = Field(
        None, description="High-level purpose of this node in the workflow."
    )
    goal: Optional[str] = Field(
        None,
        description="What this node is trying to accomplish in this step.",
    )
    expected_result: Optional[str] = Field(
        None,
        description="What shape / content we expect from this node's output.",
    )
    tools: List[str] = Field(
        default_factory=list,
        description="Tool names this node is allowed to use (subset of the master Toolkit).",
    )
    depends_on: List[str] = Field(
        default_factory=list,
        description="List of predecessor node IDs this node depends on.",
    )
    mode: NodeMode = Field(
        default=NodeMode.AUTO,
        description=(
            "Execution mode preference. AUTO = let Agent decide; TEXT = plain LLM; "
            "STRUCTURED = use Pydantic schema; TOOL = emphasize tools."
        ),
    )
    output_schema: Optional[str] = Field(
        default=None,
        description=(
            "Optional fully-qualified import path to a Pydantic model "
            "for structured output (e.g. 'myproj.schemas.Answer')."
        ),
    )
    strict_tool_call: Optional[bool] = Field(
        default=None,
        description="Override Agent's strict_tool_call setting for this node.",
    )
    model: Optional[str] = Field(
        default=None,
        description=(
            "Optional model identifier (e.g. 'openai/gpt-4o-mini'). "
            "Reserved for future per-node LLM routing; currently unused."
        ),
    )

    @field_validator("id")
    def _id_not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("node id must not be empty")
        return v


class GraphSpec(BaseModel):
    """
    Full graph specification, typically loaded from YAML.

    Example:

    ```yaml
    name: blog_workflow
    description: "Research -> outline -> draft -> review"
    nodes:
      - id: research
        purpose: ...
        goal: ...
        tools: ["web_search"]
      - id: outline
        depends_on: ["research"]
        ...
    outputs:
      - draft
      - review
    ```
    """

    name: str
    description: Optional[str] = None
    nodes: List[GraphNode]
    outputs: List[str] = Field(
        default_factory=list,
        description="Which node IDs are considered 'final outputs' of the graph.",
    )

    @field_validator("nodes")
    def _unique_node_ids(cls, v: List[GraphNode]) -> List[GraphNode]:
        ids = [n.id for n in v]
        if len(ids) != len(set(ids)):
            dupes = {i for i in ids if ids.count(i) > 1}
            raise ValueError(f"duplicate node IDs in graph: {sorted(dupes)}")
        return v

    def node_map(self) -> dict[str, GraphNode]:
        return {n.id: n for n in self.nodes}
