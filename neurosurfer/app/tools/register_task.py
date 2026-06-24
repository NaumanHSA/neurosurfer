"""register_task — validate and persist a new TaskDefinition to the user registry.

The task_builder meta-agent calls this after interviewing the user and receiving
plan approval. The tool validates the definition against the PolicyCeiling before
persisting, so authored Tasks can never exceed the hard caps.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from neurosurfer.tools.base import Tool, ToolContext, ToolResult
from neurosurfer.tools.coerce import SHELL_POLICY_ALIASES, coerce_enum, coerce_str_list, fill_key

# ──────────────────────────────────────────────────────────────────────────────
# Coercion maps — small / local models phrase these loosely; normalise them.
# Generic mechanics (coerce_enum / coerce_str_list / fill_key) live in coerce.py
# so every register_* tool is equally forgiving; the task-specific input-type map
# stays here. The shell-policy map is shared (SHELL_POLICY_ALIASES from coerce).
# ──────────────────────────────────────────────────────────────────────────────

_VALID_INPUT_TYPES = {"text", "path", "path_or_url", "bool", "int", "choice"}
_INPUT_TYPE_ALIASES: dict[str, str] = {
    "text": "text", "string": "text", "str": "text", "free_text": "text", "freetext": "text",
    "path": "path", "filepath": "path", "file": "path", "file_path": "path",
    "directory": "path", "dir": "path", "folder": "path",
    "path_or_url": "path_or_url", "url": "path_or_url", "uri": "path_or_url",
    "repo": "path_or_url", "repository": "path_or_url", "git": "path_or_url", "git_url": "path_or_url",
    "bool": "bool", "boolean": "bool", "flag": "bool", "yes_no": "bool",
    "int": "int", "integer": "int", "number": "int", "num": "int",
    "choice": "choice", "select": "choice", "enum": "choice", "option": "choice", "options": "choice",
}
_VALID_SUBAGENTS = {"explore", "analyzer", "writer", "verifier"}


# ──────────────────────────────────────────────────────────────────────────────
# Nested input models for structured tool-call schema
# ──────────────────────────────────────────────────────────────────────────────

class _GuardrailsInput(BaseModel):
    shell_policy: str = Field(
        default="gated",
        description="Shell access level: 'denied' (none), 'readonly' (read-only cmds), 'gated' (user approves each)",
    )
    write_scope: list[str] = Field(
        default_factory=list,
        description="Relative directory paths the task may write to, e.g. ['docs/', 'out/']",
    )
    path_deny: list[str] = Field(
        default_factory=list,
        description="Glob patterns the task may not read, e.g. ['.env', '**/secrets/**']",
    )
    max_turns: int = Field(default=100, description="Maximum agent turns before stopping (hard ceiling: 500)")
    max_subagent_depth: int = Field(default=2, description="Maximum sub-agent nesting depth (hard ceiling: 3)")
    max_concurrent_subagents: int = Field(default=4, description="Max parallel sub-agents (hard ceiling: 8)")

    @field_validator("shell_policy", mode="before")
    @classmethod
    def _coerce_shell(cls, v: Any) -> Any:
        return coerce_enum(v, SHELL_POLICY_ALIASES, "gated") if isinstance(v, str) else v

    @field_validator("write_scope", "path_deny", mode="before")
    @classmethod
    def _coerce_str_list(cls, v: Any) -> Any:
        return coerce_str_list(v)


class _InputSpecInput(BaseModel):
    name: str = Field(description="Snake_case input name, e.g. 'repo_path'")
    type: str = Field(
        default="text",
        description="Input type: text, path, path_or_url, bool, int, choice",
    )
    required: bool = True
    prompt: str = Field(default="", description="Question shown to the user at run time")
    default: str | None = None
    choices: list[str] | None = Field(
        default=None,
        description="Valid choices when type='choice'",
    )

    @model_validator(mode="before")
    @classmethod
    def _coerce_keys(cls, data: Any) -> Any:
        data = fill_key(data, "name", ("key", "id", "field", "arg"))
        return fill_key(data, "prompt", ("question", "label", "description", "help", "text"))

    @field_validator("type", mode="before")
    @classmethod
    def _coerce_type(cls, v: Any) -> Any:
        return coerce_enum(v, _INPUT_TYPE_ALIASES, "text")

    @field_validator("default", mode="before")
    @classmethod
    def _coerce_default(cls, v: Any) -> Any:
        return None if v is None else str(v)


# ──────────────────────────────────────────────────────────────────────────────
# Tool input
# ──────────────────────────────────────────────────────────────────────────────

class RegisterTaskArgs(BaseModel):
    name: str = Field(description="Kebab-case task identifier, e.g. 'code-review'")
    description: str = Field(description="One-line summary shown in /task list (under 120 chars)")
    system_prompt: str = Field(description="Task-specific instructions for the agent (no generic identity/tone — the engine adds those)")
    tools: list[str] = Field(
        default_factory=lambda: list(_SAFE_DEFAULT_TOOLS),
        description=(
            "Tool names this task may use. Valid values: "
            "read_file, list_dir, search, run_command, web_search, http, data, browse, "
            "write_file, apply_edit, ask_user, present_plan, todo, spawn_agent, finish"
        ),
    )
    sub_agents: list[str] = Field(
        default_factory=list,
        description="Built-in sub-agent types to allow: explore, analyzer, writer, verifier",
    )
    guardrails: _GuardrailsInput = Field(default_factory=_GuardrailsInput)
    inputs: list[_InputSpecInput] = Field(default_factory=list)
    plan_required: bool = Field(
        default=False,
        description="True = the agent must present_plan and get approval before writing anything",
    )
    model: str | None = Field(
        default=None,
        description="Optional model override for this task, e.g. 'claude-opus-4-8'",
    )


_SAFE_DEFAULT_TOOLS: frozenset[str] = frozenset(["read_file", "list_dir", "search", "ask_user", "present_plan", "todo", "finish"])


# ──────────────────────────────────────────────────────────────────────────────
# Tool
# ──────────────────────────────────────────────────────────────────────────────

class RegisterTaskTool(Tool):
    """Validate and save a TaskDefinition drafted by the task_builder agent."""

    name = "register_task"
    description = (
        "Validate and register a new TaskDefinition in the user Task registry. "
        "Call this only after the user has approved the proposed task via present_plan. "
        "Returns success (with the save path) or a detailed validation error."
    )
    input_model = RegisterTaskArgs
    def is_read_only(self, args: object) -> bool:
        return False

    def is_concurrency_safe(self, args: object) -> bool:
        return False

    def is_destructive(self, args: object) -> bool:
        return False

    def __init__(self, tasks_dir: Path) -> None:
        self._tasks_dir = tasks_dir

    async def call(self, args: RegisterTaskArgs, ctx: ToolContext) -> ToolResult:  # type: ignore[override]
        from ...agents.runtime.permissions import Guardrails
        from ...tasks.definition import ALL_TOOLS, InputSpec, Provenance, TaskDefinition
        from ...tasks.policy import PolicyCeiling, validate_task_all
        from ...tasks.registry import TaskRegistry

        ceiling = PolicyCeiling()
        _denied_scopes = {"", ".", "/", "~", ".git"}

        # 1. Construct the TaskDefinition, sanitising loose model output so it
        #    registers cleanly (clamp caps, drop unknown tools, fix scopes).
        try:
            g = args.guardrails

            # Normalise write scopes; drop dangerous roots instead of failing.
            write_scope: list[str] = []
            for s in g.write_scope:
                base = (s or "").strip().rstrip("/")
                if base in _denied_scopes or base.endswith(".git"):
                    continue
                write_scope.append(base + "/")
            guardrails = Guardrails(
                shell_policy=g.shell_policy,  # type: ignore[arg-type]
                write_scope=write_scope,
                path_deny=g.path_deny,
                max_turns=max(1, min(g.max_turns, ceiling.max_turns)),
                max_subagent_depth=max(0, min(g.max_subagent_depth, ceiling.max_subagent_depth)),
                max_concurrent_subagents=max(1, min(g.max_concurrent_subagents, ceiling.max_concurrent_subagents)),
            )

            inputs = []
            for i in args.inputs:
                if not i.name:
                    continue  # skip nameless inputs rather than fail
                # i.type is already coerced to a valid InputType literal.
                inputs.append(InputSpec.model_validate({
                    "name": i.name,
                    "type": i.type,
                    "required": i.required,
                    "prompt": i.prompt or f"Provide a value for '{i.name}'",
                    "default": i.default,
                    "choices": i.choices if i.type == "choice" else None,
                }))

            # Keep only real tool names; guarantee the essentials are present.
            tools = [t for t in args.tools if t in ALL_TOOLS]
            for essential in ("ask_user", "finish"):
                if essential not in tools:
                    tools.append(essential)

            sub_agents = [s for s in args.sub_agents if s in _VALID_SUBAGENTS]

            provenance = Provenance(created_by="task_builder")
            task = TaskDefinition(
                name=args.name,
                description=args.description,
                system_prompt=args.system_prompt,
                tools=tools,
                sub_agents=sub_agents,
                guardrails=guardrails,
                inputs=inputs,
                plan_required=args.plan_required,
                model=args.model,
                provenance=provenance,
            )
        except Exception as exc:
            return ToolResult.error(
                f"Could not build the task definition: {exc}. "
                "Re-call register_task with corrected fields."
            )

        # 2. Validate against the PolicyCeiling (collect all violations at once).
        violations = validate_task_all(task)
        if violations:
            return ToolResult.error(
                "Policy violations — adjust the task and try again:\n"
                + "\n".join(f"  • {v}" for v in violations)
            )

        # 2b. Tool-alignment check: every tool name mentioned in backticks inside
        #     system_prompt must appear in the tools list. Models often write the
        #     system prompt correctly but forget to add the tool to the list.
        import re as _re
        _mentioned = {m for m in _re.findall(r"`(\w+)`", args.system_prompt) if m in ALL_TOOLS}
        _tools_set = set(tools)
        _missing = _mentioned - _tools_set
        if _missing:
            listed = ", ".join(sorted(_missing))
            return ToolResult.error(
                f"Tool alignment error: the system_prompt references {listed} "
                f"but {'they are' if len(_missing) > 1 else 'it is'} not in the tools list.\n"
                f"Add {listed} to the tools list and call register_task again."
            )

        # 3. Persist to the user registry.
        try:
            registry = TaskRegistry(self._tasks_dir)
            registry.save(task, validate=False)  # already validated above
        except Exception as exc:
            return ToolResult.error(f"Failed to save task: {exc}")

        path = self._tasks_dir / f"{args.name}.yaml"
        return ToolResult.ok(
            f"Task '{args.name}' registered.\n"
            f"Saved to: {path}\n"
            f"Run it with: /task run {args.name}",
            registered_task=args.name,
        )
