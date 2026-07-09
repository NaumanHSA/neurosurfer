"""Durable state: plan, manifest, todos, decisions — pinned outside compactable history.

These items are re-injected each turn as a suffix on the system prompt so that
context compaction can never drop them.  They live only for the duration of one
run — there is no on-disk persistence.

The tool layer writes into this object (todo/present_plan already call the mutators);
the context manager reads it via to_context_block() and system_with_durable().
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

_TODO_MARK: dict[str, str] = {
    "pending": "[ ]",
    "in_progress": "[~]",
    "completed": "[x]",
}


@dataclass
class DurableState:
    """All state that must survive context compaction."""

    plan_title: str | None = None
    plan_text: str | None = None
    manifest: str | None = None
    todos: list[dict[str, Any]] = field(default_factory=list)
    decisions: list[str] = field(default_factory=list)
    # User-selected Python interpreter for python_exec / install_python_package
    # (set via the set_python_env tool / /pyenv command). Kept durable so an
    # explicit "use conda env ABC" survives context compaction.
    python_env: str | None = None

    # ── mutators (called by tools) ────────────────────────────────────────────

    def set_plan(self, title: str, text: str) -> None:
        self.plan_title = title
        self.plan_text = text

    def set_manifest(self, text: str) -> None:
        self.manifest = text

    def set_todos(self, items: list[dict[str, Any]]) -> None:
        self.todos = list(items)

    def add_decision(self, text: str) -> None:
        self.decisions.append(text)

    def set_python_env(self, interpreter: str) -> None:
        self.python_env = interpreter

    def is_empty(self) -> bool:
        return not any(
            [self.plan_text, self.manifest, self.todos, self.decisions, self.python_env]
        )

    # ── context injection ─────────────────────────────────────────────────────

    def to_context_block(self) -> str:
        """Render durable state as a system-prompt section.

        Returns an empty string when there is nothing to inject, so callers can
        safely append it without adding trailing whitespace.
        """
        if self.is_empty():
            return ""

        lines: list[str] = ["<durable_state>"]

        if self.plan_text:
            title = self.plan_title or "Approved Plan"
            lines.append(f"\n## {title}\n{self.plan_text}")

        if self.manifest:
            lines.append(f"\n## Manifest\n{self.manifest}")

        if self.todos:
            todo_lines = "\n".join(
                f"{_TODO_MARK.get(t.get('status', 'pending'), '[ ]')} {t.get('content', '')}"
                for t in self.todos
            )
            lines.append(f"\n## Todos\n{todo_lines}")

        if self.decisions:
            dec_lines = "\n".join(f"- {d}" for d in self.decisions)
            lines.append(f"\n## Decisions\n{dec_lines}")

        if self.python_env:
            lines.append(f"\n## Python Environment\npython_exec is pinned to: {self.python_env}")

        lines.append("\n</durable_state>")
        return "\n".join(lines)
