"""Where MCP servers are discovered — one interface, more than one index.

The official registry (`registry.modelcontextprotocol.io`) is canonical, neutral and
stays the default. It is also a single-term keyword index that answers in ~12s about
a third of the time, and its schema carries *how to run* a server and nothing about
what the server exposes. So there is room for a second opinion, and this is the seam
that makes one an implementation rather than a fork of every call site.

Two rules hold the design together:

**Every extra is optional.** A source that cannot report tools, or use counts, or an
icon leaves them empty — it never invents them. Callers render what is present. That
is what lets the official source stay exactly as it is while a richer one sits beside
it, and it is why `capabilities` exists: a UI asks *can you do this* rather than
discovering the answer from an empty list.

**`RegistryHit` stays the currency.** Both sources produce the same type, so ranking,
install and the studio do not branch on which engine answered.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

__all__ = [
    "AuthorizationRequired",
    "Capability",
    "RegistrySource",
    "SourceUnavailable",
    "ToolSummary",
]


class SourceUnavailable(RuntimeError):
    """The source cannot be used as configured — usually a missing API key.

    Distinct from a registry *error*: nothing went wrong, the engine simply is not
    set up. Selecting one in this state is refused with the reason rather than
    silently returning nothing.
    """


class AuthorizationRequired(RuntimeError):
    """The server needs a human to authorise it in a browser before it will answer.

    The third thing an install can need, and neither of the other two: not a
    missing API key (that is `SourceUnavailable`) and not a value we could prompt
    for in a form (that is `CredentialRequirement`). An OAuth consent screen has
    to be *visited* — a Gmail server cannot be handed a password, only a token the
    user grants by signing in to Google.

    Carrying `setup_url` is the whole point. Without it this failure looks exactly
    like a broken endpoint: the connection is created, the MCP call answers
    `-32001 Authorization required`, and the client sits there until it times out.
    With it, the install has somewhere to send the user.
    """

    def __init__(self, message: str, *, setup_url: str = "", server: str = "") -> None:
        super().__init__(message)
        self.setup_url = setup_url
        self.server = server


@dataclass(frozen=True)
class ToolSummary:
    """One tool a server exposes, as an index reports it before installation.

    The official registry cannot supply these — its schema has no tools field — so
    an empty list means "not published", never "this server has no tools".
    """

    name: str
    description: str = ""
    input_schema: dict[str, Any] = field(default_factory=dict)
    output_schema: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
        }


class Capability:
    """What a source can answer, so a caller can ask instead of guessing.

    A missing capability is not a failure — it decides whether a control is offered
    at all. Sorting by popularity against an index with no use counts is a control
    that does nothing, which is worse than no control.
    """

    SEMANTIC_SEARCH = "semantic_search"
    """Natural-language queries work. Without it, callers reduce a phrase to
    keywords first — see `architect.capability.registry_terms`."""

    TOOL_LISTING = "tool_listing"
    """`detail()` populates `tools`, so what a server offers is knowable before it
    is installed."""

    POPULARITY = "popularity"
    """Hits carry `use_count`, so results can be ranked by use."""


@runtime_checkable
class RegistrySource(Protocol):
    """One place MCP servers can be discovered.

    Implementations live in this package and are registered in `__init__`. They must
    raise `McpRegistryError` for transport and protocol failures — callers already
    handle that — and `SourceUnavailable` when they are simply not configured.
    """

    #: Stable identifier, stored in settings and sent over the API.
    id: str
    #: Shown in the UI.
    label: str
    #: One line on what this index is and what using it implies.
    blurb: str
    #: Subset of `Capability`.
    capabilities: frozenset[str]
    #: True when this engine cannot answer without a user-supplied key. A chooser
    #: uses it to put the key field on the row that needs it, rather than in a
    #: section below that reads as unrelated to the option it unlocks.
    requires_key: bool

    def available(self) -> bool:
        """False when the source is not usable as configured (e.g. no API key)."""
        ...

    def key_set(self) -> bool:
        """True when a key is on file, from settings or the environment.

        Distinct from `available()` even where the two agree today: "a key is
        stored" is what a form renders, and it must not start meaning "the engine
        answered" if availability ever grows a second condition.
        """
        ...

    def search(self, query: str, *, limit: int = 20, cursor: str = "") -> list[Any]:
        """Search the index. Returns `RegistryHit`s; `cursor` pages where supported."""
        ...

    def detail(self, name: str) -> dict[str, Any]:
        """The full entry for one server, in that source's own shape."""
        ...

    def install_config(self, server: dict[str, Any], body: dict[str, Any] | None = None):
        """Turn an entry (+ user-supplied env/headers) into an `McpServerConfig`."""
        ...

    def credentials(self, server: dict[str, Any]) -> list[Any]:
        """What must be supplied before this server can run."""
        ...
