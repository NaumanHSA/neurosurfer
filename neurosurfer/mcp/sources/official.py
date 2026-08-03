"""The official MCP registry, behind the source interface.

Deliberately a thin adapter over `mcp/registry.py` rather than a move of it. That
module is the default engine, every existing caller and test points at it, and an
experimental second source is not a reason to disturb the one that works. It keeps
its keep-alive client, its retry, its 40s timeout, its 15-minute cache and its
`{placeholder}` credential handling; this file only states what it can do.
"""

from __future__ import annotations

from typing import Any

from ..registry import (
    RegistryHit,
    credential_requirements,
    install_config,
    registry_detail,
    search_registry,
)

__all__ = ["OfficialSource"]


class OfficialSource:
    id = "official"
    label = "Official MCP registry"
    blurb = (
        "The canonical, publisher-neutral index at registry.modelcontextprotocol.io. "
        "Carries how to run a server — including packages that run locally — but "
        "publishes no tool list and no usage data."
    )
    # Neither semantic search nor tools: its `search` is a single-term keyword index
    # (`read gmail` finds nothing, `gmail` finds five), which is exactly why
    # `architect.capability.registry_terms` reduces a phrase to keywords first.
    capabilities = frozenset()
    requires_key = False

    def available(self) -> bool:
        """Always usable — no key, no account."""
        return True

    def key_set(self) -> bool:
        """Nothing to set: this index is anonymous."""
        return False

    def search(self, query: str, *, limit: int = 20, cursor: str = "") -> list[RegistryHit]:
        hits = search_registry(query, limit=limit, cursor=cursor)
        for hit in hits:
            hit.source = self.id
        return hits

    def detail(self, name: str) -> dict[str, Any]:
        return registry_detail(name)

    def install_config(self, server: dict[str, Any], body: dict[str, Any] | None = None):
        """The registry's own config, plus the little provenance it publishes.

        Only a name, a description and a repository link — no icon, no usage, no
        tools. That thinness is the point of `ServerOrigin` being all-optional: the
        installed card shows what exists here and does not pretend the rest is zero.
        """
        from neurosurfer.config.mcp import ServerOrigin

        cfg = install_config(server, body)
        repo = server.get("repository") or {}
        return cfg.model_copy(update={"origin": ServerOrigin(
            source=self.id,
            qualified_name=str(server.get("name") or ""),
            description=str(server.get("description") or ""),
            homepage=str(repo.get("url") or "") if isinstance(repo, dict) else "",
        )})

    def credentials(self, server: dict[str, Any]) -> list[Any]:
        return credential_requirements(server)
