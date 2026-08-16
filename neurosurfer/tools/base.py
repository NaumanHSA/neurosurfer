"""The Tool contract.

A tool declares its name, description, a pydantic input model, and three behaviour
flags the agent loop uses for scheduling and gating: ``is_read_only`` /
``is_concurrency_safe`` / ``is_destructive``. ``call`` returns a structured
:class:`ToolResult`; errors are *returned* (``is_error=True``), never raised, so
the loop can self-correct.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Mapping
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
    # Session-scoped side-channel for cross-tool state that doesn't warrant a
    # dedicated field. Known keys:
    #   "python_interpreter" (str) — interpreter path pinned by set_python_env,
    #   read by python_exec / install_python_package's interpreter resolver
    #   (see tools/builtin/python_exec/interpreter.py).
    extra: dict[str, Any] = field(default_factory=dict)


# ──────────────────────────────────────────────────────────────────────────────
# Operations
# ──────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Operation:
    """One thing a tool can do.

    A tool is a **type** of integration — "SQL" — and an operation is a thing it
    does: `query`, `list_tables`, `table_schema`. Before this, each of those was
    its own top-level tool, and the relationship between them existed only as
    shared private helpers in one module: real in the code, invisible to
    everything above it. The same failure the registry was built to fix, one
    level up — a fact that is not written down gets guessed at.

    **Grouped by type, never by credential.** A credential is a value a tool
    *uses*, not what it *is*: it rotates, and two SQL nodes may legitimately
    point at different databases. Grouping by credential would rearrange the
    palette when somebody changed a password.

    Each operation carries its own `input_model`, which is the point: a caller
    configuring one is shown the arguments *that* operation needs, rather than
    the union of every operation's arguments with everything optional.
    """

    description: str
    input_model: type[BaseModel]
    #: Tags from `registry.capabilities`. Resolution matches a *tool* when any of
    #: its operations declares the tag; the operation says which one to call.
    capabilities: frozenset[str] = frozenset()
    #: Per operation, because `query` reads and an `insert` would not.
    read_only: bool = True
    #: What a person calls this operation — "Run a query", not `query`. Empty
    #: means "derive it from the key", which is right for `query` and wrong for
    #: `table_schema`; declare one wherever the derivation reads badly.
    title: str = ""


# ──────────────────────────────────────────────────────────────────────────────
# Tool
# ──────────────────────────────────────────────────────────────────────────────
class Tool(ABC):
    #: The identifier: what a graph writes in `tools:`, what the model is offered,
    #: what every log line and error message says. Stable, snake_case, ours.
    name: str = ""

    #: What a **person** calls it. `sql` is a good identifier and a poor label —
    #: a palette that lists `sql`, `http`, `apply_edit` is showing its own
    #: vocabulary to somebody who never agreed to learn it.
    #:
    #: Kept deliberately separate from `name` rather than replacing it: renaming
    #: the identifier breaks every workflow that references it, and a title is
    #: exactly the thing you want to be free to reword. Empty means "derive it"
    #: (`apply_edit` → "Apply Edit"), which is fine for the simple cases and
    #: wrong for the ones that carry a product name — declare those.
    title: str = ""

    description: str = ""
    input_model: type[BaseModel] = BaseModel

    #: What an **author** configures once, as against what the model fills in per
    #: call. Same shape as `input_model` and a completely different lifetime.
    #:
    #: The distinction exists because `write_file` in a hosted studio had nowhere
    #: to say *where files go*. Its `path` resolved against the gateway process's
    #: working directory, so an agent asked to write a report wrote into the
    #: server's own checkout, and there was no surface anywhere that could have
    #: said otherwise.
    #:
    #: Why not simply add `root` to `input_model`: everything in `input_model` is
    #: offered to the model, and a model offered a `root` invents one. That is the
    #: failure `secret_inputs` and `BoundTool` already exist to prevent for
    #: credentials, and a directory is the same kind of fact — decided by whoever
    #: built the workflow, identical on every call, and none of the model's
    #: business. So it is a second declaration, never a wider first one.
    #:
    #: `None` means "nothing to configure", which is the honest answer for `browse`
    #: and must not become an empty settings panel.
    settings_model: type[BaseModel] | None = None

    #: The `settings_model` field that is this tool's filesystem root, if it has
    #: one. Everything in `path_inputs` is resolved under it and refused outside it.
    #:
    #: Named explicitly rather than inferred from a field called `root`, because a
    #: convention that lives in a naming habit is one nothing checks — and the
    #: manifest, the validator and the studio all have to agree on which field
    #: this is.
    root_setting: str = ""

    #: Which **call arguments** name a path. Declared, because confinement cannot
    #: be guessed: `search` takes a `pattern` and a `path` and only one of them is
    #: a place on disk, and a wrapper that inspected the value would confine
    #: whatever happened to look path-shaped that day.
    #:
    #: Empty on a tool with no `root_setting`; the two are meaningless apart.
    path_inputs: frozenset[str] = frozenset()

    #: Icon slug, resolved against the registry's own icon set and served to
    #: front-ends as a URL. Empty means "use the family default", worked out from
    #: which registry domain the tool lives in — so an MCP tool nobody has ever
    #: seen still reads as *database* or *web* rather than as a generic box, and
    #: per-tool artwork stays optional instead of being a prerequisite.
    icon: str = ""

    # True for tools backed by an external MCP server. The permission layer reads
    # this (via ``getattr``) to apply the MCP gate without importing the mcp module.
    is_mcp: bool = False

    #: Input names whose value is a credential, so a workflow must pass them as
    #: ``${STORED_NAME}`` rather than writing them down.
    #:
    #: The tool is the only place that knows this — a schema says `dsn` is a
    #: string, not that the string contains a password. Without it, "what does
    #: this workflow need me to supply" had to be answered by whatever the model
    #: happened to list in `secrets:`, and a live build invented seven names
    #: (`…_TRUST_CERTIFICATE`, `…_ENCRYPTION`) that were not inputs of anything.
    #: Read via ``getattr`` so MCP tools, which cannot declare it, are unaffected.
    secret_inputs: frozenset[str] = frozenset()

    #: How a person *obtains* the value for `secret_inputs`. Required whenever
    #: those are set: "a connection URL from your database provider" is the
    #: difference between a checklist someone can act on and a field name they
    #: have to guess at. A build once invented seven credential names for want
    #: of this sentence.
    credential_help: str = ""

    #: Tags from :mod:`neurosurfer.registry.capabilities` — what this tool can
    #: *do*, in a closed vocabulary, so resolution matches on a declared fact
    #: rather than on words shared with a description. Untagged tools still work;
    #: they are simply never a match, which is the point.
    capabilities: frozenset[str] = frozenset()

    #: Where the work happens: ``in_process`` | ``local_subprocess`` | ``hosted``.
    #: A hosted service cannot reach ``localhost`` — the fact that made a hosted
    #: SQL gateway resolve as available for a database running in a container on
    #: the user's own machine.
    runtime: str = "in_process"

    #: ``core`` | ``authored`` | ``imported`` | ``mcp``. How much is known.
    origin: str = "core"

    #: name → :class:`Operation`, for a tool that is a *type* of integration
    #: rather than a single action. Empty for single-purpose tools (`http`,
    #: `web_search`), which must not grow an "Operation: [one choice]" selector.
    #:
    #: When set, `input_model` is expected to carry an `operation` field, and
    #: each operation declares the arguments *it* needs — so a caller can be
    #: shown one precise parameter set instead of the union of all of them.
    operations: Mapping[str, Operation] = {}

    def operation_for(self, args: Any) -> Operation | None:
        """The operation *args* selects, or None for a single-purpose tool."""
        if not self.operations:
            return None
        return self.operations.get(str(getattr(args, "operation", "") or ""))

    def check_secret(self, name: str, value: str) -> str:
        """Why *value* is unusable as this tool's *name* input, or "" if it is fine.

        Checked before a run, because "is it set" is a different question from
        "will it work" and only the tool can answer the second. A stored value
        left over from an abandoned naming scheme once satisfied a requirement
        by name — `AIAccessManagment_SQLSERVER_URL` held `127.0.0.1`, a hostname
        from a build that had invented a seven-field connection model — so the
        user was never asked, and the run failed on a malformed URL instead.

        Default: any non-empty value passes. Override where the shape is knowable
        and the check is cheap; a slow or uncertain one belongs at call time.
        """
        return "" if str(value or "").strip() else "is empty"

    # Behaviour flags (defaults: not read-only, not concurrency-safe).
    def is_read_only(self, args: BaseModel) -> bool:
        # A tool with operations answers per operation: `query` reads and an
        # `insert` would not, and the approval gates key off this. `args` is
        # already part of the signature — it was added for `run_command` — so
        # this needs no contract change.
        op = self.operation_for(args)
        return op.read_only if op is not None else False

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

    @property
    def settings_schema(self) -> dict[str, Any] | None:
        """JSON Schema for `settings_model`, or None when there is nothing to set.

        Served by the gateway and rendered by the studio, which is why it is a
        schema rather than a form: the same declaration has to reach a Python
        caller, an API caller and a canvas, and only one of those three can be
        shown a widget.
        """
        return None if self.settings_model is None else model_to_schema(self.settings_model)

    def required_settings(self) -> list[str]:
        """Settings that must have a value before this tool can run.

        Read by validation, so a workflow says *"this step has no output
        directory"* while it is being built rather than when it fails.
        """
        schema = self.settings_schema or {}
        return [str(r) for r in (schema.get("required") or [])]

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
