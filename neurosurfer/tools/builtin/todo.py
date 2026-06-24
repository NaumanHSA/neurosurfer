from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from ..base import Tool, ToolContext, ToolResult

_MARK = {"pending": "[ ]", "in_progress": "[~]", "completed": "[x]"}

# Map the many ways a model phrases a status onto our three canonical values.
_STATUS_ALIASES: dict[str, str] = {
    "pending": "pending", "todo": "pending", "to_do": "pending", "to-do": "pending",
    "not_started": "pending", "not started": "pending", "open": "pending", "new": "pending",
    "in_progress": "in_progress", "in-progress": "in_progress", "inprogress": "in_progress",
    "in progress": "in_progress", "doing": "in_progress", "active": "in_progress",
    "started": "in_progress", "wip": "in_progress",
    "completed": "completed", "complete": "completed", "done": "completed",
    "finished": "completed", "closed": "completed", "resolved": "completed",
}


class TodoItem(BaseModel):
    content: str = Field(description="What needs to be done.")
    status: Literal["pending", "in_progress", "completed"] = "pending"

    @model_validator(mode="before")
    @classmethod
    def _coerce_shape(cls, data: Any) -> Any:
        # Accept a bare string item ("do X") as {content: "do X"}.
        if isinstance(data, str):
            return {"content": data}
        if isinstance(data, dict):
            # Common alternative keys for the text.
            if "content" not in data:
                for alt in ("task", "text", "title", "description", "name", "item"):
                    if alt in data:
                        data = {**data, "content": data[alt]}
                        break
        return data

    @field_validator("status", mode="before")
    @classmethod
    def _coerce_status(cls, v: Any) -> Any:
        if isinstance(v, str):
            return _STATUS_ALIASES.get(v.strip().lower(), "pending")
        return v


class TodoArgs(BaseModel):
    todos: list[TodoItem] = Field(description="The full, updated todo list.")

    @model_validator(mode="before")
    @classmethod
    def _coerce_list(cls, data: Any) -> Any:
        if isinstance(data, list):
            return {"todos": data}
        if isinstance(data, dict) and "todos" not in data:
            for alt in ("items", "tasks", "list", "todo"):
                if alt in data:
                    return {"todos": data[alt]}
        return data


class TodoTool(Tool):
    name = "todo"
    description = (
        "Record and update the task's todo list. "
        "IMPORTANT: always send ALL items — completed ones included. "
        "NEVER remove an item; mark it 'completed' instead. "
        "Workflow: set item to in_progress when starting it; set it to completed "
        "when done, then set the next item to in_progress. "
        "Call this tool after EVERY completed step — not just once at the start. "
        "Keep exactly one item in_progress at a time. "
        "The list is pinned in durable state so it survives compaction."
    )
    input_model = TodoArgs

    async def call(self, args: TodoArgs, ctx: ToolContext) -> ToolResult:  # type: ignore[override]
        items = [{"content": t.content, "status": t.status} for t in args.todos]
        if ctx.durable is not None:
            ctx.durable.set_todos(items)
        rendered = "\n".join(f"{_MARK[t.status]} {t.content}" for t in args.todos)
        done = sum(1 for t in args.todos if t.status == "completed")
        return ToolResult.ok(f"Todos updated ({done}/{len(args.todos)} done):\n{rendered}")
