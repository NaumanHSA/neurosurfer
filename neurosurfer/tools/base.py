"""The Tool contract.

A tool declares its name, description, a pydantic input model, and three behaviour
flags the agent loop uses for scheduling and gating: ``is_read_only`` /
``is_concurrency_safe`` / ``is_destructive``. ``call`` returns a structured
:class:`ToolResult`; errors are *returned* (``is_error=True``), never raised, so
the loop can self-correct.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol

from pydantic import BaseModel, ValidationError

from ..llm.types import ToolSchema
from .schema import model_to_schema

if TYPE_CHECKING:  # avoid import cycles; these are wired in later phases
    from ..agents.context.durable_state import DurableState
    from ..memory.store import MemoryStore

# The user's answer to an out-of-scope write prompt:
#   "always" → allow + persist the folder to the task's write scope
#   "once"   → allow just this write, don't widen scope
#   "deny"   → refuse
WriteChoice = Literal["always", "once", "deny"]


# ──────────────────────────────────────────────────────────────────────────────
# Result
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class ToolResult:
    content: str
    is_error: bool = False
    # Out-of-band signals the agent loop acts on (plan approval, finish, etc.).
    control: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def ok(cls, content: str, **control: Any) -> ToolResult:
        return cls(content=content, is_error=False, control=control)

    @classmethod
    def error(cls, content: str) -> ToolResult:
        return cls(content=content, is_error=True)


# ──────────────────────────────────────────────────────────────────────────────
# IO + context
# ──────────────────────────────────────────────────────────────────────────────
class IOHandler(Protocol):
    """How tools talk to the human (or a scripted test driver)."""

    async def ask(self, question: str, options: list[str] | None = None) -> str: ...

    async def request_plan_approval(self, plan: str) -> tuple[bool, str]: ...

    async def request_shell_approval(self, command: str, reason: str) -> bool: ...

    async def request_write_approval(self, path: str, summary: str) -> WriteChoice: ...

    def notify(self, message: str) -> None: ...


@dataclass
class FileState:
    mtime: float
    size: int
    content: str


SpawnFn = Callable[[str, str], Awaitable[str]]  # (agent_type, prompt) -> report


@dataclass
class ToolContext:
    """Everything a tool needs that is not in its own arguments."""

    cwd: Path
    io: IOHandler
    file_state: dict[str, FileState] = field(default_factory=dict)
    durable: DurableState | None = None
    spawn: SpawnFn | None = None
    # Guardrails object (Phase 3/6). Tools may consult it; permissions enforce it.
    guardrails: Any = None
    depth: int = 0
    # Optional callback to persist a newly-approved write folder onto the active
    # Task's write_scope (wired by the runner/REPL; None ⇒ session-only widening).
    persist_scope: Callable[[str], None] | None = None
    # Long-term memory (Pillar 1). ``memory`` is the store; ``memory_agent`` is the
    # active agent/task name used for the agent-scoped layer; ``session_id`` tags
    # provenance. All None on a bare engine run with memory disabled.
    memory: MemoryStore | None = None
    memory_agent: str | None = None
    session_id: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


# ──────────────────────────────────────────────────────────────────────────────
# Tool
# ──────────────────────────────────────────────────────────────────────────────
class Tool(ABC):
    name: str = ""
    description: str = ""
    input_model: type[BaseModel] = BaseModel

    # Behaviour flags (defaults: not read-only, not concurrency-safe).
    def is_read_only(self, args: BaseModel) -> bool:
        return False

    def is_concurrency_safe(self, args: BaseModel) -> bool:
        # Safe iff read-only by default; override for nuance.
        return self.is_read_only(args)

    def is_destructive(self, args: BaseModel) -> bool:
        return not self.is_read_only(args)

    def is_enabled(self) -> bool:
        return True

    def progress_message(self, args: dict[str, Any]) -> str:
        """A short, human-friendly status line shown while this call runs.

        The agent loop puts this on the ``ToolStarted`` event so front-ends can render
        "Reading file README.md…" instead of "read_file {'path': 'README.md'}". Override
        per tool for context-aware text; the default humanises the tool name.
        """
        return f"{self.name.replace('_', ' ').capitalize()}…"

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description=self.description,
            input_schema=model_to_schema(self.input_model),
        )

    def parse_args(self, raw: dict[str, Any]) -> BaseModel:
        return self.input_model.model_validate(raw)

    @abstractmethod
    async def call(self, args: BaseModel, ctx: ToolContext) -> ToolResult: ...

    async def run(self, raw: dict[str, Any], ctx: ToolContext) -> ToolResult:
        """Validate raw args then dispatch. Validation errors are returned as
        tool errors so the model can correct its call."""
        try:
            args = self.parse_args(raw)
        except ValidationError as e:
            return ToolResult.error(f"Invalid arguments for {self.name}: {e}")
        try:
            return await self.call(args, ctx)
        except Exception as e:  # noqa: BLE001 - tool errors flow back as results
            return ToolResult.error(f"{self.name} failed: {type(e).__name__}: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# Pool
# ──────────────────────────────────────────────────────────────────────────────
class ToolPool:
    """The curated set of tools; a Task narrows it via an allow-list."""

    def __init__(self, tools: list[Tool]):
        self._tools: dict[str, Tool] = {t.name: t for t in tools}

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def progress_message(self, name: str, args: dict[str, Any]) -> str:
        """Friendly status line for a tool call; safe for unknown tools."""
        tool = self.get(name)
        if tool is None:
            return f"{name.replace('_', ' ').capitalize()}…"
        try:
            return tool.progress_message(args)
        except Exception:  # noqa: BLE001 - never let a status string break the loop
            return f"{name.replace('_', ' ').capitalize()}…"

    def all(self) -> list[Tool]:
        return [t for t in self._tools.values() if t.is_enabled()]

    def select(self, names: list[str]) -> ToolPool:
        chosen = [self._tools[n] for n in names if n in self._tools]
        return ToolPool(chosen)

    def schemas(self) -> list[ToolSchema]:
        return [t.schema for t in self.all()]

    def names(self) -> list[str]:
        return list(self._tools.keys())
