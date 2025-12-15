from __future__ import annotations

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator
from enum import Enum
from neurosurfer.tracing import TraceResult


class NodeMode(str, Enum):
    AUTO = "auto"
    TEXT = "text"
    STRUCTURED = "structured"
    JSON = "json"
    TOOL = "tool"

# ---------------------------
# Graph-level input spec
# ---------------------------
class GraphInput(BaseModel):
    """
    Specification for a top-level graph input.

    Normalized form:
      name: str
      type: str     (string|integer|float|boolean|object|array, or synonyms)
      required: bool
      description: Optional[str]
    """

    name: str = Field(..., description="Input name (key expected at runtime).")
    type: str = Field(
        default="string",
        description="Logical type: string|integer|float|boolean|object|array (or 'str', 'int', etc.).",
    )
    required: bool = Field(
        default=True,
        description="If True, this key must be present in runtime inputs.",
    )
    description: Optional[str] = Field(
        default=None,
        description="Optional human description of the input.",
    )

    @field_validator("type")
    @classmethod
    def _normalize_type(cls, v: str) -> str:
        v = v.strip()
        lower = v.lower()
        if lower in {"str", "string", "text"}:
            return "string"
        if lower in {"int", "integer"}:
            return "integer"
        if lower in {"float", "number"}:
            return "float"
        if lower in {"bool", "boolean"}:
            return "boolean"
        if lower in {"dict", "object"}:
            return "object"
        if lower in {"list", "array"}:
            return "array"
        return lower

    model_config = dict(extra="ignore")


# ---------------------------
# Node policy (per-node AgentConfig overrides)
# ---------------------------
class NodePolicy(BaseModel):
    """
    Per-node policy that can override some AgentConfig settings and add
    node-level execution constraints (e.g., timeout).

    YAML example:
        nodes:
          - id: research
            policy:
              retries: 1
              timeout_s: 30
              max_new_tokens: 180
              temperature: 0.2
              allow_input_pruning: false
              repair_with_llm: true
              strict_tool_call: true
    """
    max_new_tokens: Optional[int] = Field(default=None, description="Override AgentConfig.max_new_tokens for this node only.")
    temperature: Optional[float] = Field(default=None, description="Override AgentConfig.temperature for this node only.")
    retries: Optional[int] = Field(default=None, description="Override AgentConfig.retry.max_route_retries for this node.")
    timeout_s: Optional[int] = Field(
        default=None,
        description=(
            "Soft timeout for this node in seconds. Execution isn't forcibly "
            "cancelled but the node will be marked as errored if exceeded."
        ),
    )
    # Direct AgentConfig-like overrides
    allow_input_pruning: Optional[bool] = None
    repair_with_llm: Optional[bool] = None
    strict_tool_call: Optional[bool] = None
    strict_json: Optional[bool] = None
    max_json_repair_attempts: Optional[int] = None    # for malformed JSON repairs
    
    skip_special_tokens: Optional[bool] = None
    return_stream_by_default: Optional[bool] = None
    log_internal_thoughts: Optional[bool] = None

    class Config:
        extra = "ignore"  # ignore unknown keys under 'policy'


# ---------------------------
# Graph node & spec
# ---------------------------
class GraphNode(BaseModel):
    id: str
    description: Optional[str] = None
    kind: str = Field(default="base", description="Node kind: base | react")
    purpose: Optional[str] = None
    goal: Optional[str] = None
    expected_result: Optional[str] = None
    tools: list[str] = Field(default_factory=list)
    depends_on: list[str] = Field(default_factory=list)
    mode: NodeMode = Field(default=NodeMode.AUTO)
    output_schema: Optional[str] = None
    model: Optional[str] = None
    rag: Optional[bool] = False

    policy: Optional[NodePolicy] = Field(
        default=None,
        description="Optional per-node AgentConfig/policy overrides.",
    )
    # save node's output as files
    export: Optional[bool] = Field(default=False, description="Whether to export the node's output as files.")
    export_path: Optional[str] = Field(default=None, description="Optional custom path for exporting the node's output.")

    @field_validator("kind")
    def _kind_not_invalid(cls, v: str) -> str:
        v = v.strip()
        if v not in {"base", "react"}:
            raise ValueError(f"node kind must be 'base' or 'react', got '{v}'")
        return v

    @field_validator("id")
    def _id_not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("node id must not be empty")
        return v


class Graph(BaseModel):
    name: str
    description: Optional[str] = None

    inputs: list[GraphInput] = Field(
        default_factory=list,
        description="Optional declared graph inputs that runtime 'inputs' must satisfy.",
    )

    nodes: list[GraphNode]
    outputs: list[str] = Field(default_factory=list)

    @field_validator("nodes")
    def _unique_node_ids(cls, v: list[GraphNode]) -> list[GraphNode]:
        ids = [n.id for n in v]
        if len(ids) != len(set(ids)):
            dupes = {i for i in ids if ids.count(i) > 1}
            raise ValueError(f"duplicate node IDs in graph: {sorted(dupes)}")
        return v

    @field_validator("inputs", mode="before")
    @classmethod
    def _normalize_inputs(cls, v):
        """
        Accept flexible YAML forms and normalize to a list[GraphInput]-compatible dicts.

        Supported:

        1) List of compact mappings:
            inputs:
              - topic_title: str
              - query:
                  type: string
                  required: true

        2) List of full specs:
            inputs:
              - name: topic_title
                type: str
                required: true

        3) Single mapping:
            inputs:
              topic_title: str
              query:
                type: string
                required: true
        """
        if v is None:
            return []

        # Case 3: single mapping
        if isinstance(v, dict):
            out = []
            for name, spec in v.items():
                if isinstance(spec, dict):
                    # e.g. query: { type: string, required: true }
                    merged = {"name": name, **spec}
                else:
                    # e.g. topic_title: str
                    merged = {"name": name, "type": spec}
                out.append(merged)
            return out

        # Expect list otherwise
        if not isinstance(v, list):
            raise TypeError("inputs must be a list or mapping")

        normalized = []
        for item in v:
            if isinstance(item, dict) and "name" not in item and len(item) == 1:
                # Compact mapping: { topic_title: str } OR { query: {type, required} }
                name, spec = next(iter(item.items()))
                if isinstance(spec, dict):
                    # { query: {type: string, required: true} }
                    merged = {"name": name, **spec}
                else:
                    # { topic_title: str }
                    merged = {"name": name, "type": spec}
                normalized.append(merged)
            else:
                # Already in normalized or near-normalized form
                normalized.append(item)
        return normalized

    @field_validator("inputs")
    @classmethod
    def _unique_input_names(cls, v: List[GraphInput]) -> List[GraphInput]:
        names = [f.name for f in v]
        if len(names) != len(set(names)):
            dupes = {n for n in names if names.count(n) > 1}
            raise ValueError(f"duplicate graph input names: {sorted(dupes)}")
        return v

    def node_map(self) -> dict[str, GraphNode]:
        return {n.id: n for n in self.nodes}


class NodeExecutionResult(BaseModel):
    node_id: str
    mode: NodeMode
    raw_output: object
    structured_output: Optional[object] = None
    tool_call_output: Optional[object] = None
    started_at: float
    duration_ms: int
    error: Optional[str] = None
    traces: Optional[TraceResult] = None

class GraphExecutionResult(BaseModel):
    graph: Graph
    nodes: Dict[str, NodeExecutionResult]
    final: Dict[str, Any]
