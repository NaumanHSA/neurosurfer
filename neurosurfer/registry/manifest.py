"""What a tool *is*, stated rather than inferred.

The Architect spent a long time reasoning about tools it knew almost nothing
about. An MCP tool reached it as a name, a sentence and an unchecked JSON schema,
so every resolution heuristic was a guess at a fact nobody had written down — and
each guess became a workflow that could not run:

- a database-migration tool answered *"generate chart images"*, because the only
  thing to match on was prose and both contained "generate";
- a **hosted** gateway resolved as available for a database on `localhost` it
  could never reach, because nothing could express where a tool runs;
- a plan invented seven credential names, because no tool could say what it
  actually needed.

A :class:`ToolManifest` is the answer to each: capability **tags** instead of
prose, a **runtime** instead of an assumption, a **credential spec** instead of
invention, and a **verification record** instead of hope.

**The manifest is the universal artifact; the implementation is a pointer.** A
core entry points at a Python class in this package, an imported entry at a server
plus a remote tool name, an authored entry at generated source. One lookup path,
three kinds of backing — which is what lets a caller stop caring which it got.

For core tools the manifest is *derived from the class*, so an author declares
these facts inline next to the code rather than in a file that drifts from it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from .capabilities import unknown as unknown_capabilities
from .icons import FALLBACK_ICON, resolve_icon

__all__ = [
    "ORIGINS",
    "RUNTIMES",
    "OperationManifest",
    "ToolManifest",
    "Verification",
    "humanise",
    "manifest_for",
]

#: Words the derived title should not sentence-case into something wrong. A
#: derivation is only worth having if its failures are rare and cheap; these are
#: the ones that would otherwise read as "Http" and "Sql" in a palette.
_INITIALISMS = {
    "api": "API",
    "cli": "CLI",
    "csv": "CSV",
    "db": "DB",
    "dir": "Directory",
    "html": "HTML",
    "http": "HTTP",
    "id": "ID",
    "io": "IO",
    "json": "JSON",
    "mcp": "MCP",
    "os": "OS",
    "pdf": "PDF",
    "sql": "SQL",
    "ssh": "SSH",
    "url": "URL",
    "xml": "XML",
    "yaml": "YAML",
}


def humanise(name: str) -> str:
    """A readable label derived from an identifier — `apply_edit` → "Apply Edit".

    The fallback, not the mechanism. A tool whose label carries a product name
    ("Microsoft SQL Server", "Google Drive") must declare `title`, because no
    derivation from a snake_case identifier can produce one. This exists so the
    other ninety percent — and every MCP tool, which cannot declare anything —
    still reads as words rather than as code.
    """
    parts = [p for p in str(name or "").replace("-", "_").split("_") if p]
    if not parts:
        return ""
    return " ".join(_INITIALISMS.get(p.lower(), p[:1].upper() + p[1:]) for p in parts)

#: Where a tool came from, and therefore how much is known about it.
ORIGINS = ("core", "authored", "imported", "mcp")

#: Where a tool's work actually happens. The field exists because a hosted
#: service cannot reach `localhost`, and for a long time nothing could say so:
#: a hosted SQL gateway was offered for a database in a container on the user's
#: own machine, resolved as satisfied, and failed at run time.
RUNTIMES = ("in_process", "local_subprocess", "hosted")


@dataclass(frozen=True)
class Verification:
    """Evidence that a tool did something, once, rather than merely existing."""

    ok: bool
    at: float = field(default_factory=time.time)
    #: What was tried, in a few words ("connected and listed 12 tables").
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "at": self.at, "detail": self.detail}


@dataclass(frozen=True)
class OperationManifest:
    """One operation of a tool, as a caller sees it.

    The static half of :class:`neurosurfer.tools.base.Operation` — its own
    schema, its own tags, its own read-only answer. A configuration dialog reads
    exactly this to know which fields to show once an operation is chosen.
    """

    name: str
    description: str
    #: What a person calls it — "Run a query" rather than `query`. Always set:
    #: `manifest_for` derives one from the key when the operation declares none,
    #: so a caller rendering a dropdown never needs a fallback of its own.
    title: str = ""
    input_schema: dict[str, Any] = field(default_factory=dict)
    capabilities: frozenset[str] = frozenset()
    read_only: bool = True

    @property
    def required_inputs(self) -> list[str]:
        return [str(r) for r in (self.input_schema.get("required") or [])]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "title": self.title or humanise(self.name),
            "input_schema": self.input_schema,
            "capabilities": sorted(self.capabilities),
            "read_only": self.read_only,
        }


@dataclass(frozen=True)
class ToolManifest:
    """Everything a caller needs to decide whether a tool fits, before calling it."""

    name: str
    description: str
    #: What a person calls it. `name` is the identifier a graph references and a
    #: log line prints; this is the label a palette shows. Two fields because
    #: renaming the identifier breaks every workflow that uses it, and the label
    #: is precisely the thing you want to be free to reword.
    title: str = ""
    #: Icon slug, always resolvable against `registry/icons/` — see `icons.py`
    #: for the three-tier resolution that guarantees it.
    icon: str = FALLBACK_ICON
    origin: str = "core"
    #: Tags from `capabilities.CAPABILITIES`. Empty means "nothing declares what
    #: this does" — usable, but never a *match*, because matching an untagged
    #: tool means falling back to prose and that is the failure being retired.
    capabilities: frozenset[str] = frozenset()
    runtime: str = "in_process"
    #: Inputs whose value is a credential and must arrive as `${STORED_NAME}`.
    secret_inputs: frozenset[str] = frozenset()
    #: How a person *obtains* the value. "A connection URL from your database
    #: provider" is the difference between a checklist someone can act on and a
    #: field name they have to guess at.
    credential_help: str = ""
    input_schema: dict[str, Any] = field(default_factory=dict)
    #: What an *author* configures once, as against `input_schema`, which is what
    #: the model fills in per call. None when the tool has nothing to configure —
    #: which must stay distinguishable from an empty object, or `browse` grows a
    #: settings panel with no settings in it.
    settings_schema: dict[str, Any] | None = None
    read_only: bool = True
    #: None when nobody has ever run it. Usable, and says so.
    verified: Verification | None = None
    #: For imported/mcp entries: which server, and the name it knows the tool by.
    server: str = ""
    remote_name: str = ""
    #: What this tool can do, when it is a type of integration rather than a
    #: single action. Empty for single-purpose tools. `capabilities` above is the
    #: union of these, so `covers()` and `providers_of()` answer "which tool" and
    #: this answers "which operation" — one lookup path, two levels of detail.
    operations: tuple[OperationManifest, ...] = ()

    # ── views ───────────────────────────────────────────────────────────────
    @property
    def required_inputs(self) -> list[str]:
        return [str(r) for r in (self.input_schema.get("required") or [])]

    @property
    def required_settings(self) -> list[str]:
        """Settings that must have a value before this tool is configured.

        Read by validation, so "this step has no output directory" is something a
        workflow says while it is being built rather than when it fails.
        """
        return [str(r) for r in ((self.settings_schema or {}).get("required") or [])]

    def operation(self, name: str) -> OperationManifest | None:
        return next((o for o in self.operations if o.name == name), None)

    def operations_for(self, tag: str) -> list[OperationManifest]:
        """The operations that provide *tag* — what to actually call."""
        return [o for o in self.operations if tag in o.capabilities]

    @property
    def reaches_localhost(self) -> bool:
        """Can this tool reach a service on the user's own machine?

        The one question `thinair/data` got wrong for a whole afternoon.
        """
        return self.runtime != "hosted"

    def covers(self, tag: str) -> bool:
        return tag in self.capabilities

    def problems(self) -> list[str]:
        """Why this manifest is not fit to register. Empty means it is."""
        out: list[str] = []
        if not self.name:
            out.append("no name")
        if not self.description:
            out.append("no description")
        if self.origin not in ORIGINS:
            out.append(f"origin {self.origin!r} is not one of {list(ORIGINS)}")
        if self.runtime not in RUNTIMES:
            out.append(f"runtime {self.runtime!r} is not one of {list(RUNTIMES)}")
        for tag in unknown_capabilities(self.capabilities):
            out.append(f"capability {tag!r} is not in the vocabulary")
        for op in self.operations:
            if not op.description:
                out.append(f"operation {op.name!r} has no description")
            for tag in unknown_capabilities(op.capabilities):
                out.append(
                    f"operation {op.name!r} declares capability {tag!r}, "
                    f"which is not in the vocabulary"
                )
        # The union is what resolution matches on, so an operation tagged with
        # something the tool does not claim would be unreachable.
        declared = {t for op in self.operations for t in op.capabilities}
        if self.operations and not declared <= self.capabilities:
            out.append(
                f"operations declare {sorted(declared - self.capabilities)} "
                f"which the tool does not"
            )
        # A credential nobody can be told how to get is a checklist item that
        # reads as an accusation. It cost a real build seven invented names.
        if self.secret_inputs and not self.credential_help:
            out.append("declares secret inputs but no credential_help")
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "title": self.title or humanise(self.name),
            "icon": self.icon,
            "origin": self.origin,
            "capabilities": sorted(self.capabilities),
            "runtime": self.runtime,
            "secret_inputs": sorted(self.secret_inputs),
            "credential_help": self.credential_help,
            "settings_schema": self.settings_schema,
            "read_only": self.read_only,
            "verified": self.verified.to_dict() if self.verified else None,
            "server": self.server,
            "remote_name": self.remote_name,
            "operations": [o.to_dict() for o in self.operations],
        }


def manifest_for(tool: Any) -> ToolManifest:
    """Read a live tool's manifest off the class that implements it.

    Everything is `getattr`-with-default, so a tool that declares none of it —
    an MCP tool, which cannot — still yields a valid manifest saying exactly that:
    no capabilities, unknown credentials, and (for MCP) a runtime we work out from
    how the server connects rather than from anything the server told us.
    """
    schema: dict[str, Any] = {}
    try:
        schema = dict(tool.schema.input_schema or {})
    except Exception:  # noqa: BLE001 - a schema we cannot read is not a crash
        pass

    settings: dict[str, Any] | None = None
    try:
        settings = tool.settings_schema
    except Exception:  # noqa: BLE001 - same rule as the input schema
        settings = None

    is_mcp = bool(getattr(tool, "is_mcp", False))
    origin = str(getattr(tool, "origin", "mcp" if is_mcp else "core"))

    read_only = True
    try:
        # `is_read_only` takes the parsed args on some tools; the manifest is a
        # static view, so a tool that needs them is reported conservatively.
        read_only = bool(tool.is_read_only(None))
    except Exception:  # noqa: BLE001
        read_only = not is_mcp

    operations = _operations_of(tool)
    if operations:
        # A tool with operations is read-only only if every one of them is —
        # the static view cannot know which will be called.
        read_only = all(o.read_only for o in operations)

    # The tool's own tags plus everything its operations claim. Resolution
    # matches the tool on the union; `operations_for(tag)` then says which
    # operation to call, so neither question needs a second lookup path.
    capabilities = frozenset(getattr(tool, "capabilities", None) or ()) | {
        t for o in operations for t in o.capabilities
    }

    name = getattr(tool, "name", "") or ""
    return ToolManifest(
        name=name,
        description=(getattr(tool, "description", "") or "").strip(),
        title=str(getattr(tool, "title", "") or "").strip() or humanise(name),
        icon=resolve_icon(tool),
        origin=origin,
        capabilities=capabilities,
        runtime=str(getattr(tool, "runtime", "in_process")),
        secret_inputs=frozenset(getattr(tool, "secret_inputs", None) or ()),
        credential_help=str(getattr(tool, "credential_help", "") or ""),
        input_schema=schema,
        settings_schema=settings,
        read_only=read_only,
        verified=getattr(tool, "verified", None),
        server=str(getattr(tool, "server_name", "") or ""),
        remote_name=str(getattr(tool, "remote_name", "") or ""),
        operations=operations,
    )


def _operations_of(tool: Any) -> tuple[OperationManifest, ...]:
    """A tool's operations, as static manifests.

    `getattr`-with-default like everything else here: a single-purpose tool
    declares none and yields an empty tuple, which is the honest answer rather
    than an invented one-item list.
    """
    from neurosurfer.tools.schema import model_to_schema

    declared = getattr(tool, "operations", None) or {}
    out: list[OperationManifest] = []
    for op_name, op in declared.items():
        schema: dict[str, Any] = {}
        try:
            schema = dict(model_to_schema(op.input_model) or {})
        except Exception:  # noqa: BLE001 - a schema we cannot read is not a crash
            pass
        out.append(OperationManifest(
            name=str(op_name),
            description=(op.description or "").strip(),
            title=str(getattr(op, "title", "") or "").strip() or humanise(str(op_name)),
            input_schema=schema,
            capabilities=frozenset(op.capabilities or ()),
            read_only=bool(op.read_only),
        ))
    return tuple(sorted(out, key=lambda o: o.name))
