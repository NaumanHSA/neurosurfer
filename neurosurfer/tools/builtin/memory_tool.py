"""The ``memory`` tool — let an agent record or forget a durable fact mid-run.

Adds to one of two scopes: ``global`` (about the user, recalled everywhere) or
``agent`` (about this agent's domain, recalled only when this agent runs). Needs no
permission change — unknown tools fall through to ``Decision(True)`` in
``permissions``. Loose op/scope phrasing is coerced so small models succeed. When no
``MemoryStore`` is wired (a bare engine run) the tool no-ops with a clear message
rather than erroring.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from ...memory.models import MemoryEntry, MemoryKind
from ..base import Tool, ToolContext, ToolResult
from ..coerce import coerce_enum

_OP_ALIASES = {
    "add": "add", "save": "add", "remember": "add", "store": "add", "record": "add",
    "forget": "forget", "delete": "forget", "remove": "forget", "drop": "forget",
}
_SCOPE_ALIASES = {
    "global": "global", "user": "global", "personal": "global", "all": "global",
    "agent": "agent", "task": "agent", "project": "agent", "local": "agent",
}
_KIND_ALIASES = {
    "preference": "preference", "pref": "preference", "fact": "fact",
    "decision": "decision", "glossary": "glossary", "term": "glossary",
}


class MemoryArgs(BaseModel):
    op: str = Field(default="add", description="add | forget")
    text: str = Field(
        default="",
        description="The fact to remember (for add), or ignored for forget.",
    )
    scope: str = Field(
        default="global",
        description="global (about the user, recalled everywhere) | agent (this agent's domain).",
    )
    kind: str = Field(
        default="fact", description="preference | fact | decision | glossary."
    )
    id: str = Field(default="", description="Entry id to forget (for op=forget).")


class MemoryTool(Tool):
    name = "memory"
    description = (
        "Record or forget a durable, long-term fact you should remember across future "
        "sessions. Use op='add' with a concise 'text' and scope='global' for facts about "
        "the user or scope='agent' for facts about your own domain/task. Use op='forget' "
        "with an 'id' to remove one. Save only stable, reusable facts — not transient "
        "run details."
    )
    input_model = MemoryArgs

    def is_read_only(self, args: BaseModel) -> bool:
        return False

    def is_destructive(self, args: BaseModel) -> bool:
        return False

    def parse_args(self, raw: dict[str, Any]) -> BaseModel:
        # Accept a loose 'content'/'memory' key as an alias for 'text'.
        if isinstance(raw, dict) and "text" not in raw:
            for alt in ("content", "memory", "value", "fact"):
                if alt in raw:
                    raw = {**raw, "text": raw[alt]}
                    break
        return super().parse_args(raw)

    async def call(self, args: MemoryArgs, ctx: ToolContext) -> ToolResult:  # type: ignore[override]
        store = getattr(ctx, "memory", None)
        if store is None:
            return ToolResult.ok("Memory is not enabled for this run; nothing saved.")

        op = coerce_enum(args.op, _OP_ALIASES, "add")
        if op == "forget":
            if not args.id:
                return ToolResult.error("forget requires an 'id' (see your relevant memory).")
            ok = store.forget(args.id)
            return ToolResult.ok(f"Forgot memory {args.id}." if ok else f"No memory {args.id}.")

        text = args.text.strip()
        if not text:
            return ToolResult.error("add requires non-empty 'text'.")
        scope = coerce_enum(args.scope, _SCOPE_ALIASES, "global")
        kind: MemoryKind = coerce_enum(args.kind, _KIND_ALIASES, "fact")  # type: ignore[assignment]
        agent = getattr(ctx, "memory_agent", None) or ""
        if scope == "agent" and not agent:
            scope = "global"  # no agent context → keep it global rather than orphan it
        entry = store.add(
            MemoryEntry(
                scope=scope,  # type: ignore[arg-type]
                scope_key=agent if scope == "agent" else "",
                kind=kind,
                text=text,
                source="agent",
                session_id=getattr(ctx, "session_id", None),
            )
        )
        where = "globally" if scope == "global" else f"for agent '{agent}'"
        return ToolResult.ok(f"Saved memory {where} (id {entry.id}).")
