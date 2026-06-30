"""Pydantic output schemas for the Architect workflow nodes.

These are referenced by ``output_schema`` fields in the architect's graph.yaml
and imported at runtime via ``import_string``.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# ---------------------------------------------------------------------------
# Tolerance helpers — weaker models often emit nested JSON as a *string* (e.g.
# nodes="[{...}]") or drop required list fields. These coercers keep the
# Architect resilient so a single formatting quirk doesn't fail the whole build.
# ---------------------------------------------------------------------------

def _maybe_json(value: Any) -> Any:
    """If *value* is a string that looks like JSON, parse it; else return as-is."""
    if isinstance(value, str):
        s = value.strip()
        if s[:1] in "[{":
            import json
            try:
                return json.loads(s)
            except (json.JSONDecodeError, ValueError):
                return value
    return value


def _as_list(value: Any) -> Any:
    """Coerce a scalar / JSON-string into a list where a list is expected."""
    parsed = _maybe_json(value)
    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, str):
        # Bare comma-free string → single-item list; comma-joined → split.
        return [p.strip() for p in parsed.split(",")] if "," in parsed else [parsed]
    return parsed


def _salvage_json_objects(s: str) -> list | None:
    """Extract complete top-level ``{...}`` objects from a possibly-truncated JSON
    array string via brace matching. Recovers partial output from weak models that
    hit a token limit mid-array. Returns the parsed dicts, or None if none parse."""
    import json
    objs: list = []
    depth = 0
    start: int | None = None
    in_str = False
    esc = False
    for i, ch in enumerate(s):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    try:
                        objs.append(json.loads(s[start : i + 1]))
                    except (json.JSONDecodeError, ValueError):
                        pass
                    start = None
    return objs or None


def _coerce_object_list(value: Any) -> Any:
    """Coerce a field that must be a list-of-objects. Parses a JSON string, and on
    failure *salvages* complete objects — but NEVER comma-splits (that would shred a
    broken object string into garbage).

    Unrecoverable input collapses to ``[]`` so the field's own constraints decide the
    outcome: schemas that require items (WorkflowPlan/StagePlan ``min_length``) error
    cleanly, while optional lists (CapabilityPlan.nodes) degrade gracefully to empty."""
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        return [value]  # a single object emitted without an array wrapper
    if isinstance(value, str):
        parsed = _maybe_json(value)
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict):
            return [parsed]
        salvaged = _salvage_json_objects(value)
        if salvaged is not None:
            return salvaged
        return []
    return value


def _derive_outputs(nodes: Any) -> list[str]:
    """Terminal node ids (those nothing depends on); fall back to the last node."""
    if not isinstance(nodes, list) or not nodes:
        return []
    ids: list[str] = []
    dep_targets: set[str] = set()
    for n in nodes:
        if isinstance(n, dict):
            if n.get("id"):
                ids.append(str(n["id"]))
            for d in _as_list(n.get("depends_on") or []):
                dep_targets.add(str(d))
    terminals = [i for i in ids if i not in dep_targets]
    return terminals or ([ids[-1]] if ids else [])

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
        # Weaker models may stringify these nested fields — parse JSON strings first.
        if isinstance(data.get("questions"), str):
            data["questions"] = _maybe_json(data["questions"])
        if isinstance(data.get("web_findings"), str):
            parsed = _maybe_json(data["web_findings"])
            if isinstance(parsed, list):
                data["web_findings"] = parsed
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
# Node 3 — decompose
# ---------------------------------------------------------------------------

class Stage(BaseModel):
    """A single logical phase of the workflow (not a node — a concept)."""

    id: str = Field(description="Short snake_case identifier for this stage.")
    name: str = Field(description="Human-readable stage name.")
    purpose: str = Field(description="What this stage achieves — one sentence.")
    rationale: str = Field(
        default="",
        description="Why this stage exists as a separate phase.",
    )
    min_nodes: int = Field(
        default=1,
        ge=1,
        description="Minimum number of nodes needed to implement this stage.",
    )
    capabilities: list[str] = Field(
        default_factory=list,
        description=(
            "What this stage needs: tools, external APIs, file I/O, LLM calls, etc."
        ),
    )

    @field_validator("id")
    @classmethod
    def _snake_case(cls, v: str) -> str:
        v = v.strip().lower().replace(" ", "_").replace("-", "_")
        if not v:
            raise ValueError("stage id must not be empty")
        return v


class StagePlan(BaseModel):
    """High-level stage decomposition produced by the decompose node."""

    intent: str = Field(
        description="The user's goal restated in one sentence.",
    )
    stages: list[Stage] = Field(
        description="2–6 logical stages of the workflow, in execution order.",
        min_length=2,
        max_length=6,
    )
    depth_rationale: str = Field(
        default="",
        description=(
            "One sentence explaining why this many stages are needed "
            "(why they can't be collapsed into fewer)."
        ),
    )

    @field_validator("stages", mode="before")
    @classmethod
    def _coerce_stages(cls, v: object) -> object:
        return _coerce_object_list(v)


# ---------------------------------------------------------------------------
# Node 4 — plan / design_nodes
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

    @field_validator("depends_on", "tools", mode="before")
    @classmethod
    def _coerce_lists(cls, v: object) -> object:
        if v in (None, ""):
            return []
        return _as_list(v)


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

    @model_validator(mode="before")
    @classmethod
    def _coerce_containers(cls, data: Any) -> Any:
        # Tolerate weaker models that stringify `nodes`/`outputs` or drop `outputs`.
        if not isinstance(data, dict):
            return data
        data = dict(data)
        if "nodes" in data:
            data["nodes"] = _coerce_object_list(data["nodes"])
        if data.get("outputs"):
            data["outputs"] = _as_list(data["outputs"])
        else:
            data["outputs"] = _derive_outputs(data.get("nodes"))
        return data

    @field_validator("name")
    @classmethod
    def _clean_name(cls, v: str) -> str:
        import re
        v = re.sub(r"[^a-z0-9_]", "_", v.lower().strip())
        v = re.sub(r"_+", "_", v).strip("_")
        return v or "workflow"


# ---------------------------------------------------------------------------
# Node 6 — tool_design (capability planning)
# ---------------------------------------------------------------------------

# Per-node verdict on how its required capability will be met.
ToolDecision = Literal["use_existing", "author_new", "infeasible"]


class ToolSpec(BaseModel):
    """Specification for a NEW tool the Architect must author to fill a capability gap.

    Carries enough detail for the tool-author engine to generate a real ``Tool``
    subclass AND to *functionally test* it in the sandbox before registration.
    """

    name: str = Field(
        description="Distinctive snake_case tool name (must NOT collide with an existing tool)."
    )
    purpose: str = Field(description="One sentence: what the tool does.")
    inputs: list[str] = Field(
        default_factory=list,
        description="Input fields the tool's args model needs, e.g. 'db_path: path to the SQLite file'.",
    )
    signature_hint: str = Field(
        default="",
        description="How call() should behave and what it returns on success.",
    )
    workflow_inputs: list[str] = Field(
        default_factory=list,
        description=(
            "Run-time values this tool needs that the USER must supply when the workflow "
            "runs (e.g. 'connection_string', 'api_key'). These become graph inputs the "
            "runtime prompts for. Use ONLY for values that genuinely come from the user."
        ),
    )
    test_setup: str = Field(
        default="",
        description=(
            "OPTIONAL self-contained Python run before the functional test to build "
            "fixtures under the current working directory (e.g. create a temp SQLite "
            "db with sample rows). May define a dict named ARGS to supply/override the "
            "call arguments. Stdlib only; no network."
        ),
    )
    test_args: dict = Field(
        default_factory=dict,
        description="Concrete sample arguments used to functionally test call().",
    )
    expected_behavior: str = Field(
        default="",
        description="What a successful functional test result should look like.",
    )

    @field_validator("name")
    @classmethod
    def _snake_case(cls, v: str) -> str:
        import re
        v = re.sub(r"[^a-z0-9_]", "_", v.lower().strip())
        v = re.sub(r"_+", "_", v).strip("_")
        if not v:
            raise ValueError("tool name must not be empty")
        return v

    @field_validator("inputs", "workflow_inputs", mode="before")
    @classmethod
    def _coerce_str_list(cls, v: object) -> object:
        # Accept a list of dicts (e.g. {"name": ..., "description": ...}) → "name: description".
        if isinstance(v, list):
            out: list[str] = []
            for item in v:
                if isinstance(item, dict):
                    name = item.get("name") or item.get("field") or ""
                    desc = item.get("description") or item.get("desc") or ""
                    out.append(f"{name}: {desc}".strip(": ").strip() if name else str(item))
                else:
                    out.append(str(item))
            return out
        if isinstance(v, str):
            return [v]
        return v

    @field_validator("test_args", mode="before")
    @classmethod
    def _coerce_test_args(cls, v: object) -> object:
        # Providers may emit test_args as a JSON string; tolerate it.
        if isinstance(v, str):
            import json
            try:
                parsed = json.loads(v)
                return parsed if isinstance(parsed, dict) else {}
            except (json.JSONDecodeError, ValueError):
                return {}
        if v is None:
            return {}
        return v


class NodeCapability(BaseModel):
    """How a single workflow node's tool need will be met."""

    node_id: str = Field(description="ID of the node this entry is about.")
    required_capability: str = Field(
        description="What this node must DO in I/O terms (e.g. 'run a SQL query against a database')."
    )
    decision: ToolDecision = Field(
        description=(
            "use_existing: a catalog tool covers it; author_new: a new tool must be "
            "built; infeasible: cannot be built safely in this framework as described."
        )
    )
    assigned_tools: list[str] = Field(
        default_factory=list,
        description="Existing catalog tool names this node will use (decision=use_existing).",
    )
    new_tools: list[ToolSpec] = Field(
        default_factory=list,
        description="Specs for tools to author for this node (decision=author_new).",
    )
    infeasible_reason: str = Field(
        default="",
        description=(
            "If infeasible: WHY it can't be built as asked, and exactly what the user "
            "would need to provide or change to make it possible."
        ),
    )

    @field_validator("node_id")
    @classmethod
    def _snake_case(cls, v: str) -> str:
        return v.strip().lower().replace(" ", "_").replace("-", "_")

    @field_validator("assigned_tools", mode="before")
    @classmethod
    def _coerce_assigned(cls, v: object) -> object:
        if v in (None, ""):
            return []
        return _as_list(v)  # bare "data" → ["data"]

    @field_validator("new_tools", mode="before")
    @classmethod
    def _coerce_new_tools(cls, v: object) -> object:
        if v in (None, ""):
            return []
        return _coerce_object_list(v)  # object list — salvage, never comma-split


class CapabilityPlan(BaseModel):
    """Per-node tool/capability decisions produced by the tool_design node."""

    nodes: list[NodeCapability] = Field(
        default_factory=list,
        description="One entry per workflow node that needs tools (base text-only nodes may be omitted).",
    )
    feasible: bool = Field(
        default=True,
        description="False if ANY node is infeasible or there are hard blockers.",
    )
    blockers: list[str] = Field(
        default_factory=list,
        description="Human-readable reasons the workflow can't be built as described.",
    )
    notes: str = Field(default="", description="Optional design notes.")

    @field_validator("nodes", mode="before")
    @classmethod
    def _coerce_nodes(cls, v: object) -> object:
        if v in (None, ""):
            return []
        return _coerce_object_list(v)  # object list — salvage, never comma-split

    @field_validator("blockers", mode="before")
    @classmethod
    def _coerce_blockers(cls, v: object) -> object:
        if v in (None, ""):
            return []
        return _as_list(v)  # scalar string list

    @model_validator(mode="after")
    def _derive_feasibility(self) -> CapabilityPlan:
        infeasible = [n for n in self.nodes if n.decision == "infeasible"]
        if infeasible:
            self.feasible = False
            existing = set(self.blockers)
            for n in infeasible:
                reason = n.infeasible_reason or f"node '{n.node_id}' cannot be built as described"
                line = f"{n.node_id}: {reason}"
                if line not in existing:
                    self.blockers.append(line)
        return self

    def new_tool_specs(self) -> list[ToolSpec]:
        """Flatten every authored-tool spec across all nodes (de-duplicated by name)."""
        seen: set[str] = set()
        out: list[ToolSpec] = []
        for node in self.nodes:
            for spec in node.new_tools:
                if spec.name and spec.name not in seen:
                    seen.add(spec.name)
                    out.append(spec)
        return out
