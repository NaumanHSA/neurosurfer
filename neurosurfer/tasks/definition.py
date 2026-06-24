"""Task data model.

A ``TaskDefinition`` is the complete specification for a specialised agent run:
system prompt, tool allow-list, enforced guardrails, declared inputs, and
optional model override.  It is stored as YAML in the registry and validated
against the ``PolicyCeiling`` before activation.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from ..agents.runtime.permissions import Guardrails

# ──────────────────────────────────────────────────────────────────────────────
# Input spec
# ──────────────────────────────────────────────────────────────────────────────

InputType = Literal["text", "path", "path_or_url", "bool", "int", "choice"]

# A Task's classification — drives CLI visibility and mutability:
#   user     — custom, editable & deletable (the default; what register_task creates)
#   readonly — a shipped built-in capability: visible & runnable, but protected
#   system   — an internal capability (e.g. task_builder): hidden from the CLI + protected
TaskKind = Literal["user", "readonly", "system"]


class InputSpec(BaseModel):
    name: str
    type: InputType = "text"
    required: bool = True
    prompt: str
    default: Any = None
    choices: list[str] | None = None  # valid only when type == "choice"


# ──────────────────────────────────────────────────────────────────────────────
# Provenance
# ──────────────────────────────────────────────────────────────────────────────

class Provenance(BaseModel):
    created_by: str = "hand"
    created_at: datetime = Field(default_factory=datetime.utcnow)
    source_model: str | None = None
    interview_transcript_ref: str | None = None


# ──────────────────────────────────────────────────────────────────────────────
# Task definition
# ──────────────────────────────────────────────────────────────────────────────

ALL_TOOLS = [
    "read_file", "list_dir", "search", "run_command", "web_search",
    "http", "data", "browse",
    "write_file", "apply_edit", "ask_user", "present_plan",
    "todo", "spawn_agent", "memory", "finish", "register_task",
]


class TaskDefinition(BaseModel):
    name: str
    version: int = 1
    kind: TaskKind = "user"
    description: str = ""
    system_prompt: str
    tools: list[str] = Field(default_factory=lambda: list(ALL_TOOLS))
    sub_agents: list[str] = Field(default_factory=list)
    guardrails: Guardrails = Field(default_factory=Guardrails)
    inputs: list[InputSpec] = Field(default_factory=list)
    plan_required: bool = False
    # Optional per-Task model id override, applied within whichever provider
    # profile is resolved for this Task. To pin the Task to a different
    # *provider* entirely, use TaskProviderStore (CLI: /task provider).
    model: str | None = None

    provenance: Provenance = Field(default_factory=Provenance)

    @field_validator("tools")
    @classmethod
    def _validate_tools(cls, v: list[str]) -> list[str]:
        unknown = set(v) - set(ALL_TOOLS)
        if unknown:
            raise ValueError(f"Unknown tools: {sorted(unknown)}")
        return v

    @property
    def is_protected(self) -> bool:
        """Readonly/system tasks cannot be edited, overridden, or deleted."""
        return self.kind in ("readonly", "system")

    @property
    def is_hidden(self) -> bool:
        """System tasks are hidden from the user-facing CLI listing."""
        return self.kind == "system"
