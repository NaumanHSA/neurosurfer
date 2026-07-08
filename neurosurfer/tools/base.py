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

from ..llm.types import ImageBlock, ToolSchema
from .schema import model_to_schema

if TYPE_CHECKING:  # avoid import cycles; these are wired in later phases
    from ..agents.context.durable_state import DurableState

# The user's answer to an out-of-scope write prompt:
#   "always" → allow + persist the folder to the task's write scope
#   "once"   → allow just this write, don't widen scope
#   "deny"   → refuse
WriteChoice = Literal["always", "once", "deny"]


@dataclass
class ShellApproval:
    """A human's answer to a shell / network / MCP approval prompt.

    ``approved`` gates the action. ``feedback`` is an optional free-text redirect:
    when the user denies *but* wants the agent to do something else instead, it is
    passed back to the model (as the tool-result error) in place of a generic
    "declined" — the Claude-Code-style "no, do this instead" affordance.
    """

    approved: bool
    feedback: str | None = None


# ──────────────────────────────────────────────────────────────────────────────
# Result
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class ToolResult:
    content: str
    is_error: bool = False
    # Out-of-band signals the agent loop acts on (plan approval, finish, etc.).
    control: dict[str, Any] = field(default_factory=dict)
    # Images the tool produced (screenshots, rendered files). The agent loop appends
    # these to the tool-results turn so a vision model can see them; non-vision models
    # drop them at the provider boundary.
    images: list[ImageBlock] = field(default_factory=list)

    @classmethod
    def ok(cls, content: str, **control: Any) -> ToolResult:
        return cls(content=content, is_error=False, control=control)

    @classmethod
    def with_images(cls, content: str, images: list[ImageBlock]) -> ToolResult:
        return cls(content=content, is_error=False, images=list(images))

    @classmethod
    def error(cls, content: str) -> ToolResult:
        return cls(content=content, is_error=True)


# ──────────────────────────────────────────────────────────────────────────────
# IO + context
# ──────────────────────────────────────────────────────────────────────────────
class IOHandler(Protocol):
    """How tools talk to the human (or a scripted test driver).

    This is the *structural* type used for annotations. To implement one, subclass
    :class:`BaseIOHandler` (below) so new hooks land with a default and your
    handler never breaks when the protocol grows.
    """

    async def ask(self, question: str, options: list[str] | None = None) -> str: ...

    async def request_plan_approval(self, plan: str) -> tuple[bool, str]: ...

    async def request_shell_approval(self, command: str, reason: str) -> ShellApproval: ...

    async def request_write_approval(self, path: str, summary: str) -> WriteChoice: ...

    def notify(self, message: str) -> None: ...


class BaseIOHandler:
    """Concrete :class:`IOHandler` whose every hook auto-approves.

    This is the base you subclass to customise approvals: override only the hooks
    you care about. Because the defaults live here, a new method added to the
    protocol lands with a sensible default and existing handlers keep working —
    no more re-implementing five methods just to run unattended.
    """

    async def ask(self, question: str, options: list[str] | None = None) -> str:
        return ""

    async def request_plan_approval(self, plan: str) -> tuple[bool, str]:
        return (True, "")

    async def request_shell_approval(self, command: str, reason: str) -> ShellApproval:
        return ShellApproval(True)

    async def request_write_approval(self, path: str, summary: str) -> WriteChoice:
        return "once"

    def notify(self, message: str) -> None:
        pass


# ``approval="auto"`` resolves to this — the zero-config default: run to
# completion, never block for a human. Named for intent at call sites.
AutoApproveIOHandler = BaseIOHandler


class TerminalIOHandler(BaseIOHandler):
    """Interactive handler that prompts on stdin — works in a terminal *and* a
    notebook. Selected via ``approval="ask"``.

    Blocks for a human decision at each gated step: y/N for shell & network,
    once/always/deny for out-of-scope writes, and a free-text reply (or plan
    feedback) otherwise. Uses ``input()`` on a worker thread so the event loop
    keeps turning while it waits.
    """

    async def _prompt(self, text: str) -> str:
        import asyncio

        return (await asyncio.to_thread(input, text)).strip()

    async def ask(self, question: str, options: list[str] | None = None) -> str:
        suffix = f"\n   options: {' / '.join(options)}" if options else ""
        return await self._prompt(f"\n❓ {question}{suffix}\n> ")

    async def request_plan_approval(self, plan: str) -> tuple[bool, str]:
        print(f"\n📋 Plan proposed:\n{plan}")
        answer = await self._prompt("Approve? [Y]es / type feedback to revise > ")
        if answer.lower() in {"y", "yes", ""}:
            return (True, "")
        return (False, answer)

    async def request_shell_approval(self, command: str, reason: str) -> ShellApproval:
        answer = await self._prompt(
            f"\n⚠️  Allow: {command}\n   ({reason})"
            f"\n   [y]es / [n]o / or type what to do instead > "
        )
        low = answer.lower().strip()
        if low in {"y", "yes"}:
            return ShellApproval(True)
        if low in {"", "n", "no"}:
            return ShellApproval(False)
        # Any other free text is a redirect: deny, and hand the message to the agent.
        return ShellApproval(False, answer.strip())

    async def request_write_approval(self, path: str, summary: str) -> WriteChoice:
        answer = (
            await self._prompt(
                f"\n✏️  Write {path}\n   {summary}\n   [o]nce / [a]lways / [d]eny > "
            )
        ).lower()
        if answer in {"a", "always"}:
            return "always"
        if answer in {"o", "once", "y", "yes"}:
            return "once"
        return "deny"

    def notify(self, message: str) -> None:
        print(message)


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
    # Task's write_scope (wired by the runner/REPL; None ⇒ run-only widening).
    persist_scope: Callable[[str], None] | None = None
    extra: dict[str, Any] = field(default_factory=dict)


# ──────────────────────────────────────────────────────────────────────────────
# Tool
# ──────────────────────────────────────────────────────────────────────────────
class Tool(ABC):
    name: str = ""
    description: str = ""
    input_model: type[BaseModel] = BaseModel

    # True for tools backed by an external MCP server. The permission layer reads
    # this (via ``getattr``) to apply the MCP gate without importing the mcp module.
    is_mcp: bool = False

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
