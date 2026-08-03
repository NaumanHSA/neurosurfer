"""Smithery, as an experimental second discovery engine.

Measured against the official registry on the same machine, six identical repeated
queries each: median **0.68s vs 11.94s**, and consistent rather than erratic. It also
answers two questions the official index structurally cannot —

* **what a server exposes**, before anything is installed: `tools` with input and
  output schemas, plus `prompts` and `resources`;
* **how much a server is used** (`useCount`, observed 0 → 25,335), which is the only
  popularity signal available anywhere in the ecosystem and the one honest way to
  rank a search where thousands of entries share a keyword.

Its search is genuinely semantic — `"send a text message"` returns telnyx and two SMS
gateways with no literal token match — which is the thing `registry_terms()` exists to
fake against a keyword index.

What it is not:

* **Not a health check.** `security.scanPassed` exists in the schema and came back
  `null` on every entry sampled, authenticated *and* not. It is a security scan
  rather than a functional one either way, so nothing here reads it.
* **Not free of runtime dependency.** Nothing here is installed in the sense the
  official registry means it. A server found on Smithery runs on Smithery, reached
  through a *connection* created under your namespace, and every request carries
  your key — so the traffic, and any credential the server holds, pass through
  their infrastructure. `install_config` refuses without a key rather than writing
  a config that provably cannot connect.
* **Not open source, and not self-hostable.** Hence experimental, opt-in, and never
  the default until someone chooses it.
"""

from __future__ import annotations

import os
import time
from typing import Any

from ..registry import CredentialRequirement, McpRegistryError, RegistryHit, _looks_secret
from .base import AuthorizationRequired, Capability, SourceUnavailable, ToolSummary

__all__ = ["SmitherySource", "SMITHERY_BASE"]

SMITHERY_BASE = "https://api.smithery.ai"
_HTTP_TIMEOUT = 20.0
_CACHE_TTL_S = 900.0

_cache: dict[str, tuple[float, Any]] = {}
_http: Any = None


def _client():
    """One keep-alive client for the process, as the official source uses."""
    global _http
    import httpx

    if _http is None:
        _http = httpx.Client(
            timeout=_HTTP_TIMEOUT,
            headers={"accept": "application/json"},
            limits=httpx.Limits(max_keepalive_connections=4, max_connections=8),
        )
    return _http


class SmitherySource:
    id = "smithery"
    label = "Smithery (experimental)"
    blurb = (
        "Faster, searches by meaning, and publishes each server's tools and usage "
        "count before you install. Needs a Smithery API key — servers found here are "
        "hosted by Smithery, so your traffic and credentials pass through them."
    )
    capabilities = frozenset({
        Capability.SEMANTIC_SEARCH,
        Capability.TOOL_LISTING,
        Capability.POPULARITY,
    })
    requires_key = True

    def __init__(self, api_key: str | None = None) -> None:
        # An explicit key wins; otherwise the environment, so a developer can try the
        # engine before any of it is wired into settings.
        self._key = (api_key or os.environ.get("SMITHERY_API_KEY") or "").strip()

    # ── plumbing ──────────────────────────────────────────────────────────────
    def available(self) -> bool:
        return bool(self._key)

    def key_set(self) -> bool:
        return bool(self._key)

    def _require_key(self) -> str:
        if not self._key:
            raise SourceUnavailable(
                "Smithery needs an API key. Add one in Settings → Discovery, or set "
                "SMITHERY_API_KEY."
            )
        return self._key

    def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:

        key = self._require_key()
        cache_key = f"{path}?{sorted((params or {}).items())}"
        hit = _cache.get(cache_key)
        now = time.time()
        if hit and now - hit[0] < _CACHE_TTL_S:
            return hit[1]

        try:
            resp = _client().get(
                f"{SMITHERY_BASE}{path}",
                params=params or {},
                headers={"Authorization": f"Bearer {key}"},
            )
        except Exception as e:  # noqa: BLE001 - network is the expected failure
            raise McpRegistryError(
                f"Smithery unreachable: {type(e).__name__}: {e}"
            ) from e
        if resp.status_code in (401, 403):
            raise McpRegistryError(
                "Smithery rejected the API key. Check it in Settings → Discovery.",
                status=resp.status_code,
            )
        if resp.status_code == 404:
            raise McpRegistryError("Not found on Smithery", status=404)
        if resp.status_code >= 400:
            raise McpRegistryError(f"Smithery returned {resp.status_code}")
        payload = resp.json()
        _cache[cache_key] = (now, payload)
        return payload

    # ── the interface ─────────────────────────────────────────────────────────
    def search(self, query: str, *, limit: int = 20, cursor: str = "") -> list[RegistryHit]:
        """Search by meaning. `cursor` carries a page number for this source."""
        page = int(cursor) if str(cursor).isdigit() else 1
        payload = self._get("/servers", {"q": query, "page": page, "pageSize": limit})
        return [self._hit(entry) for entry in payload.get("servers") or []]

    def detail(self, name: str) -> dict[str, Any]:
        """One entry, in the same `{summary, server, versions}` envelope the
        official source returns.

        Shape matters more than convenience here: the studio's install form reads
        one structure, so an engine that returned its own would need the UI to
        branch on which index answered — exactly what this seam exists to avoid.
        Smithery has no per-entry version list, so `versions` is empty rather than
        invented.
        """
        from urllib.parse import quote

        # `namespace/server`; the path is one segment, so the slash is encoded.
        entry = self._get(f"/servers/{quote(name, safe='')}")
        hit = self._hit({**entry, "useCount": entry.get("useCount")})
        summary = hit.to_dict()
        summary["tools"] = [t.to_dict() for t in self.tools(entry)]
        summary["credentials"] = [c.to_dict() for c in self.credentials(entry)]
        return {"summary": summary, "server": entry, "versions": []}

    def _hit(self, entry: dict[str, Any]) -> RegistryHit:
        """A list entry as a `RegistryHit`.

        The list endpoint does not carry tools — only `detail()` does — so `tools`
        stays empty here rather than being faked from the description.
        """
        return RegistryHit(
            name=str(entry.get("qualifiedName") or ""),
            description=str(entry.get("description") or ""),
            version="",  # Smithery versions releases, not catalogue entries
            runs_locally=False,   # everything here is Smithery-hosted
            runs_remotely=True,
            credentials=[],
            server=entry,
            use_count=entry.get("useCount"),
            verified=entry.get("verified"),
            icon_url=str(entry.get("iconUrl") or ""),
            source=self.id,
        )

    def tools(self, server: dict[str, Any]) -> list[ToolSummary]:
        """What the server exposes, as published — empty if the entry has none."""
        return [
            ToolSummary(
                name=str(t.get("name") or ""),
                description=str(t.get("description") or ""),
                input_schema=t.get("inputSchema") or {},
                output_schema=t.get("outputSchema") or {},
            )
            for t in (server.get("tools") or [])
            if isinstance(t, dict) and t.get("name")
        ]

    def credentials(self, server: dict[str, Any]) -> list[CredentialRequirement]:
        """Config a server needs, read from its connection's JSON Schema.

        A different shape from the official registry's `environmentVariables` /
        `headers` arrays, so deliberately not the same code: here it is a JSON
        Schema per connection, with `required` naming the mandatory properties.
        The `{placeholder}` rule does not apply — Smithery states requirement
        structurally instead of hiding it in a template string.
        """
        found: dict[str, CredentialRequirement] = {}
        for conn in server.get("connections") or []:
            if not isinstance(conn, dict):
                continue
            schema = conn.get("configSchema") or {}
            required = set(schema.get("required") or [])
            for name, spec in (schema.get("properties") or {}).items():
                if not isinstance(spec, dict):
                    continue
                found.setdefault(name, CredentialRequirement(
                    name=name,
                    description=str(spec.get("description") or ""),
                    required=name in required,
                    secret=_looks_secret(name),
                    where="config",
                ))
        return list(found.values())

    # ── connections ───────────────────────────────────────────────────────────
    def _namespace(self) -> str:
        """The account namespace connections are created under.

        `SMITHERY_NAMESPACE` wins so a developer can pin one; otherwise it is
        discovered, because every key has at least one and asking a user to find
        theirs to install a server is a step that answers itself.
        """
        pinned = (os.environ.get("SMITHERY_NAMESPACE") or "").strip()
        if pinned:
            return pinned
        payload = self._get("/namespaces")
        for ns in payload.get("namespaces") or []:
            if isinstance(ns, dict) and ns.get("name"):
                return str(ns["name"])
        raise McpRegistryError(
            "This Smithery key has no namespace, so there is nowhere to create a "
            "connection. Create one at smithery.ai first.",
            status=422,
        )

    def _connect(self, qualified: str, connection_id: str, config: dict[str, Any]) -> dict[str, Any]:
        """Create (or update) the connection that fronts a registry server.

        `PUT` is idempotent on `connectionId`, so reinstalling the same server
        reuses its connection rather than littering the namespace.
        """
        key = self._require_key()
        ns = self._namespace()
        payload: dict[str, Any] = {
            "transport": "http",
            # Documented as "use this instead of mcpUrl for registry servers".
            "server": qualified,
            "name": connection_id,
        }
        if config:
            payload["config"] = config
        try:
            resp = _client().put(
                f"{SMITHERY_BASE}/connect/{ns}/{connection_id}",
                json=payload,
                headers={"Authorization": f"Bearer {key}"},
            )
        except Exception as e:  # noqa: BLE001 - network is the expected failure
            raise McpRegistryError(f"Smithery unreachable: {type(e).__name__}: {e}") from e
        if resp.status_code in (401, 403):
            raise McpRegistryError(
                "Smithery rejected the API key. Check it in Settings → Discovery.",
                status=resp.status_code,
            )
        if resp.status_code >= 400:
            raise McpRegistryError(
                f"Smithery could not create a connection to '{qualified}' "
                f"({resp.status_code}): {resp.text[:200]}",
                status=422,
            )
        return {"namespace": ns, **(resp.json() or {})}

    def install_config(self, server: dict[str, Any], body: dict[str, Any] | None = None):
        """A Smithery *connection*, not the server's own address.

        This is the part that is not guessable from the catalogue. `deploymentUrl`
        looks like the endpoint — it is a URL, it is on the entry, it resolves —
        but it is Smithery's internal deployment address and answers `401
        invalid_token` to every key, including a valid one. Pointing an MCP client
        at it produces no error a user can act on, only a connect timeout.

        The client-facing endpoint has to be created first:

            PUT  /connect/{namespace}/{id}   {transport, server}   → a connection
            POST /connect/{namespace}/{id}/mcp                     → speak MCP here

        The connection is also where authorisation lives. A server fronting a
        third-party account (Gmail, Notion, Linear) comes back `auth_required`
        with a `setupUrl`, and no value we could have collected in a form would
        have changed that — so the install stops and hands the URL back rather
        than writing a config that cannot connect.
        """
        from neurosurfer.config.mcp import McpServerConfig

        key = self._require_key()
        body = body or {}
        qualified = str(server.get("qualifiedName") or "")
        if not qualified:
            raise McpRegistryError("Entry has no qualified name", status=422)
        local_name = body.get("local_name") or qualified.replace("/", "-") or "smithery"

        # Server config (API keys and the like) is held by Smithery against the
        # connection, so it travels in the PUT and never touches our config file.
        config = {k: v for k, v in (body.get("config") or {}).items() if v not in (None, "")}
        config.update({k: v for k, v in (body.get("env") or {}).items() if v not in (None, "")})

        conn_id = _connection_id(local_name)
        conn = self._connect(qualified, conn_id, config)

        status = conn.get("status") or {}
        if str(status.get("state") or "") == "auth_required":
            setup = str(status.get("setupUrl") or status.get("authorizationUrl") or "")
            raise AuthorizationRequired(
                f"'{qualified}' connects to an account that has to be authorised in a "
                "browser — sign in there, then install again.",
                setup_url=setup,
                server=qualified,
            )

        # The key is written as an `${ENV}` reference when that is where it came
        # from, so a shared mcp.json carries a pointer rather than the secret.
        env_key = (os.environ.get("SMITHERY_API_KEY") or "").strip()
        bearer = "${SMITHERY_API_KEY}" if key == env_key and env_key else key
        headers = {"Authorization": f"Bearer {bearer}"}
        headers.update(body.get("headers") or {})

        return McpServerConfig(
            name=local_name,
            transport="http",
            url=f"{SMITHERY_BASE}/connect/{conn['namespace']}/{conn_id}/mcp",
            headers=headers,
            enabled=False,
            # `useCount` and `verified` live on the *list* row, not on the detail
            # payload, so the caller passes back what it already has rather than
            # us spending a second search to recover it.
            origin=_origin(server, self.id, conn, extra=body),
        )


def _connection_id(local_name: str) -> str:
    """A stable, URL-safe id for one installed server's connection."""
    import re

    slug = re.sub(r"[^a-z0-9-]+", "-", local_name.lower()).strip("-") or "server"
    return f"ns-{slug}"[:64]


def _origin(
    server: dict[str, Any],
    source_id: str,
    conn: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
):
    """What the catalogue knew, kept for the installed server to show later.

    A server that has never started has no tool list of its own, and re-querying
    the index on every render is a network call to redraw a card. The entry is
    already in hand at install time, so it is carried along.
    """
    from neurosurfer.config.mcp import ServerOrigin

    extra = extra or {}
    uses = extra.get("use_count", server.get("useCount"))
    return ServerOrigin(
        source=source_id,
        qualified_name=str(server.get("qualifiedName") or ""),
        display_name=str(server.get("displayName") or ""),
        description=str(server.get("description") or ""),
        icon_url=str(server.get("iconUrl") or ""),
        homepage=str(server.get("homepage") or ""),
        use_count=uses if isinstance(uses, int) else None,
        verified=extra.get("verified", server.get("verified")),
        tools=[
            {"name": str(t.get("name") or ""), "description": str(t.get("description") or "")}
            for t in (server.get("tools") or [])
            if isinstance(t, dict) and t.get("name")
        ],
        setup_url=str(((conn or {}).get("status") or {}).get("setupUrl") or ""),
    )
