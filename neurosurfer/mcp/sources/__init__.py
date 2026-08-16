"""The discovery engines, and which one is active.

The official registry is the default and always available. Smithery is opt-in: it
needs the user's own API key, it is closed source, and servers found through it run
on its infrastructure — so it stays experimental until someone chooses it, and
choosing it without a key is refused rather than silently returning nothing.
"""

from __future__ import annotations

from .active import active_source, set_active_source, use_source
from .base import (
    AuthorizationRequired,
    Capability,
    RegistrySource,
    SourceUnavailable,
    ToolSummary,
)
from .official import OfficialSource
from .smithery import SmitherySource

__all__ = [
    "AuthorizationRequired",
    "Capability",
    "RegistrySource",
    "SourceUnavailable",
    "ToolSummary",
    "OfficialSource",
    "SmitherySource",
    "DEFAULT_SOURCE",
    "all_sources",
    "get_source",
    "active_source",
    "set_active_source",
    "use_source",
]

DEFAULT_SOURCE = "official"


def all_sources(*, smithery_key: str | None = None) -> list[RegistrySource]:
    """Every engine, in the order a chooser should offer them."""
    return [OfficialSource(), SmitherySource(smithery_key)]


def get_source(
    source_id: str | None = None, *, smithery_key: str | None = None
) -> RegistrySource:
    """The named engine, or the default.

    An unknown id falls back to the default rather than raising: a stale setting
    should not take the catalog down, and the official registry is always a correct
    answer to "where do I look for MCP servers".
    """
    wanted = (source_id or DEFAULT_SOURCE).strip().lower()
    for source in all_sources(smithery_key=smithery_key):
        if source.id == wanted:
            return source
    return OfficialSource()
