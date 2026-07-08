from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, field_validator

from neurosurfer.tracing import TraceResult


class NodeMode(StrEnum):
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
    description: str | None = Field(
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
    max_new_tokens: int | None = Field(default=None, description="Override AgentConfig.max_new_tokens for this node only.")
    temperature: float | None = Field(default=None, description="Override AgentConfig.temperature for this node only.")
    retries: int | None = Field(default=None, description="Override AgentConfig.retry.max_route_retries for this node.")
    timeout_s: int | None = Field(
        default=None,
        description=(
            "Soft timeout for this node in seconds. Execution isn't forcibly "
            "cancelled but the node will be marked as errored if exceeded."
        ),
    )
    # Direct AgentConfig-like overrides
    allow_input_pruning: bool | None = None
    repair_with_llm: bool | None = None
    strict_tool_call: bool | None = None
    strict_json: bool | None = None
    max_json_repair_attempts: int | None = None    # for malformed JSON repairs

    skip_special_tokens: bool | None = None
    return_stream_by_default: bool | None = None
    log_internal_thoughts: bool | None = None

    class Config:
        extra = "ignore"  # ignore unknown keys under 'policy'


# ---------------------------
# Graph node & spec
# ---------------------------
_VALID_NODE_KINDS = {"base", "react", "function", "python", "tool"}


class GraphNode(BaseModel):
    id: str
    description: str | None = None
    kind: str = Field(default="base", description="Node kind: base | react | function | python | tool")
    purpose: str | None = None
    goal: str | None = None
    expected_result: str | None = None
    tools: list[str] = Field(default_factory=list)
    depends_on: list[str] = Field(default_factory=list)
    mode: NodeMode = Field(default=NodeMode.AUTO)
    output_schema: str | None = None
    model: str | None = None
    rag: bool | None = False

    # function / python node: import path to the callable (module:attr or module.attr)
    callable: str | None = Field(default=None, description="Import path for function/python nodes.")
    # tool node: static kwargs merged with graph inputs + dep outputs before calling the tool
    tool_args: dict[str, Any] | None = Field(default=None, description="Static kwargs for tool nodes.")

    policy: NodePolicy | None = Field(
        default=None,
        description="Optional per-node AgentConfig/policy overrides.",
    )
    # save node's output as files
    export: bool | None = Field(default=False, description="Whether to export the node's output as files.")
    export_path: str | None = Field(default=None, description="Optional custom path for exporting the node's output.")

    @field_validator("kind")
    def _kind_not_invalid(cls, v: str) -> str:
        v = v.strip()
        if v not in _VALID_NODE_KINDS:
            raise ValueError(f"node kind must be one of {sorted(_VALID_NODE_KINDS)}, got '{v}'")
        return v

    @field_validator("id")
    def _id_not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("node id must not be empty")
        return v


class Graph(BaseModel):
    name: str
    description: str | None = None

    # When True, the executor stops on the first failed node (LangChain's
    # RunnableSequence "short-circuit" behaviour). When False, all non-blocked
    # nodes still run and every failure is surfaced in GraphExecutionResult.
    fail_fast: bool = Field(
        default=False,
        description="Stop execution immediately on the first node failure.",
    )

    # When True, extra input keys not declared in `inputs` are rejected instead
    # of silently ignored (stricter boundary validation).
    strict_inputs: bool = Field(
        default=False,
        description="Reject undeclared input keys (default: warn and ignore).",
    )

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
    def _unique_input_names(cls, v: list[GraphInput]) -> list[GraphInput]:
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
    structured_output: object | None = None
    tool_call_output: object | None = None
    started_at: float
    duration_ms: int
    error: str | None = None
    # True when the node was not run because an upstream dependency failed.
    skipped: bool = False
    skip_reason: str = ""
    traces: TraceResult | None = None

    @property
    def ok(self) -> bool:
        """True iff the node ran without error and was not skipped."""
        return not self.error and not self.skipped


class GraphExecutionResult(BaseModel):
    graph: Graph
    nodes: dict[str, NodeExecutionResult]
    final: dict[str, Any]
    # Nodes that failed (error is set and not skipped).
    errors: dict[str, str] = Field(default_factory=dict)
    # Nodes skipped due to upstream failures.
    skipped: list[str] = Field(default_factory=list)
    traces: TraceResult | None = None
    logs: str | None = None

    @property
    def succeeded(self) -> bool:
        """True iff every non-skipped node completed without error."""
        return not self.errors

    def execution_summary(self) -> str:
        total = len(self.nodes)
        ok = sum(1 for r in self.nodes.values() if r.ok)
        err = len(self.errors)
        skip = len(self.skipped)
        return f"Graph '{self.graph.name}': {total} nodes — {ok} ok, {err} failed, {skip} skipped"
