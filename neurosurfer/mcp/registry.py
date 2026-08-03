"""Client for the official MCP registry — FastAPI-free, so the agent can use it.

This logic used to live inside ``app/server/api/routes_mcp.py`` and raise
``HTTPException``, which made it reachable from a browser and from nowhere else.
The Architect needs exactly the same three things when it hits a capability it has
no tool for — *search*, *what would this cost me*, and *install it* — and it runs
in-process for CLI and library callers that have no web server at all.

So the client lives here and raises :class:`McpRegistryError`; the route module maps
that to HTTP. Both callers get the same answers from the same code.

The piece that did not exist before is :func:`credential_requirements`. The registry
declares what a server needs to run — ``environmentVariables`` on a package,
``headers`` on a remote, each with ``isRequired`` and ``isSecret`` — and we were
throwing it away. It is the difference between "I can't do Gmail" and "install
``…/gmail-mcp`` and set ``GOOGLE_CLIENT_ID`` and ``GOOGLE_CLIENT_SECRET``".
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "McpRegistryError",
    "CredentialRequirement",
    "REGISTRY_BASE",
    "CATEGORIES",
    "registry_get",
    "search_registry",
    "registry_detail",
    "entry_summary",
    "credential_requirements",
    "install_config",
    "default_runtime",
]

REGISTRY_BASE = "https://registry.modelcontextprotocol.io/v0.1"

# `{smithery_api_key}` in a declared header/env value: the publisher saying this
# has to be filled in before the server can run.
_PLACEHOLDER = re.compile(r"\{[a-zA-Z_][\w.-]*\}")
_SECRET_WORDS = ("token", "key", "secret", "password", "passwd", "auth", "credential")


def _looks_secret(name: str) -> bool:
    low = name.lower()
    return any(w in low for w in _SECRET_WORDS)

# Clear of the ~14s plateau the registry sits on for roughly a third of calls.
# At 15s a successful slow response and a real failure were indistinguishable.
_HTTP_TIMEOUT = 40.0
# Entries change on the order of days, and the API is slow enough that a short TTL
# is paid for in seconds of staring at a spinner.
_CACHE_TTL_S = 900.0

#: Process-wide keep-alive client, built on first use.
_http: Any = None

# url → (fetched_at, payload). The registry is public and slow-moving; caching
# keeps a browsing session from hammering it.
_cache: dict[str, tuple[float, Any]] = {}

# Canned searches for the landing view. These are *categories*, not a popularity
# ranking: the registry exposes no download counts or stars, so any "trending"
# list would be invented. Recently-updated + categories is what the data supports.
CATEGORIES: list[dict[str, str]] = [
    {"label": "Filesystem", "query": "filesystem"},
    {"label": "Database", "query": "postgres"},
    {"label": "Git & GitHub", "query": "github"},
    {"label": "Web & search", "query": "search"},
    {"label": "Cloud", "query": "aws"},
    {"label": "Communication", "query": "slack"},
    {"label": "Browser", "query": "browser"},
    {"label": "Docs & notes", "query": "notion"},
]


class McpRegistryError(RuntimeError):
    """A registry call failed. ``status`` mirrors HTTP where one is meaningful."""

    def __init__(self, message: str, *, status: int = 502) -> None:
        super().__init__(message)
        self.status = status


@dataclass(frozen=True)
class CredentialRequirement:
    """One thing a server needs before it can run, as the registry declares it."""

    name: str
    description: str = ""
    required: bool = True
    secret: bool = False
    where: str = "env"          # env | header | argument
    #: The declared value when it contains a `{placeholder}`, e.g.
    #: ``"Bearer {smithery_api_key}"``. Kept so the install can rebuild the header
    #: around whatever the user supplies rather than discarding the publisher's
    #: framing — a bare token in an `Authorization` header is not a bearer token.
    template: str = ""

    def render(self) -> str:
        # The *asked* name, not the slot: a templated header is called
        # `Authorization` and what a person can supply is the `{placeholder}`
        # inside it. Asking for "Authorization" names something nobody has.
        from .credentials import asked_name

        bits = [f"`{asked_name(self)}`"]
        if self.description:
            bits.append(f"— {self.description}")
        flags = [w for w, on in (("required", self.required), ("secret", self.secret)) if on]
        if flags:
            bits.append(f"({', '.join(flags)})")
        return " ".join(bits)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "description": self.description,
            "required": self.required, "secret": self.secret, "where": self.where,
            "template": self.template,
        }


@dataclass
class RegistryHit:
    """A search result, flattened to what a chooser (human or agent) needs."""

    name: str
    description: str = ""
    version: str = ""
    runs_locally: bool = False
    runs_remotely: bool = False
    credentials: list[CredentialRequirement] = field(default_factory=list)
    server: dict[str, Any] = field(default_factory=dict)   # the raw entry

    # ── optional, per source (see `mcp/sources/base.py`) ──────────────────────
    # This registry populates none of them: its schema is
    # [$schema, description, name, packages, remotes, repository, version,
    # websiteUrl] — how to *run* a server, nothing about what it exposes or how
    # much it is used. An index that does know fills them in, and callers render
    # what is present. Empty means "not published", never "none".
    #: Tools the server exposes, when the index publishes them.
    tools: list[Any] = field(default_factory=list)
    #: How many times the index has seen this server used.
    use_count: int | None = None
    #: The index vouches for the publisher's identity (not for the code).
    verified: bool | None = None
    icon_url: str = ""
    #: Which source produced this hit.
    source: str = "official"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "runs_locally": self.runs_locally,
            "runs_remotely": self.runs_remotely,
            "credentials": [c.to_dict() for c in self.credentials],
            "tools": [t.to_dict() for t in self.tools],
            "use_count": self.use_count,
            "verified": self.verified,
            "icon_url": self.icon_url,
            "source": self.source,
        }


def _client():
    """One keep-alive client for the process.

    Measured: a warm connection answers in ~0.4s where a fresh one takes ~1.3s. It
    does nothing for the slow responses (those are server-side) but it makes the
    fast path genuinely fast, and browsing the catalog is many small calls.
    """
    global _http
    import httpx

    if _http is None:
        _http = httpx.Client(
            timeout=_HTTP_TIMEOUT,
            headers={"accept": "application/json"},
            limits=httpx.Limits(max_keepalive_connections=4, max_connections=8),
        )
    return _http


def registry_get(path: str, params: dict[str, Any]) -> Any:
    """GET from the official registry, with a TTL cache, one retry, and clean errors.

    The public registry is erratic rather than uniformly slow: measured over
    identical repeated queries it answers in ~1s about two thirds of the time and
    ~14s the rest, independent of the query, the page size and connection reuse. So
    the timeout has to sit well clear of that plateau — at the old 15s it landed
    exactly on it, which is how a *successful* 15.2s response became the
    `ReadTimeout` the studio showed — and a timeout is worth retrying once, because
    the next attempt usually lands in the fast band.
    """
    import httpx

    query = "&".join(f"{k}={v}" for k, v in sorted(params.items()) if v not in (None, ""))
    key = f"{path}?{query}"
    hit = _cache.get(key)
    now = time.time()
    if hit and now - hit[0] < _CACHE_TTL_S:
        return hit[1]

    url = f"{REGISTRY_BASE}{path}"
    last: Exception | None = None
    for attempt in (1, 2):
        try:
            resp = _client().get(url, params=params)
            break
        except (httpx.TimeoutException, httpx.TransportError) as e:
            last = e
            if attempt == 2:
                raise McpRegistryError(
                    f"MCP registry did not answer within {_HTTP_TIMEOUT:.0f}s "
                    f"(tried twice): {type(e).__name__}"
                ) from e
        except Exception as e:  # noqa: BLE001 - network is the expected failure here
            raise McpRegistryError(
                f"MCP registry unreachable: {type(e).__name__}: {e}"
            ) from e
    else:  # pragma: no cover - the loop always breaks or raises
        raise McpRegistryError(f"MCP registry unreachable: {last}")

    if resp.status_code == 404:
        raise McpRegistryError("Not found in the MCP registry", status=404)
    if resp.status_code >= 400:
        raise McpRegistryError(f"MCP registry returned {resp.status_code}")
    payload = resp.json()
    _cache[key] = (now, payload)
    return payload


def entry_summary(entry: dict[str, Any]) -> dict[str, Any]:
    """Flatten a registry entry into what a list view needs."""
    server = entry.get("server", entry)
    meta = (entry.get("_meta") or {}).get("io.modelcontextprotocol.registry/official", {})
    packages = server.get("packages") or []
    remotes = server.get("remotes") or []
    return {
        "name": server.get("name"),
        "description": server.get("description"),
        "version": server.get("version"),
        "repository": server.get("repository"),
        # How it would run here: locally (a package we execute) or remotely (HTTP).
        "runs_locally": bool(packages),
        "runs_remotely": bool(remotes),
        "package_types": sorted({p.get("registryType", "?") for p in packages}),
        "remote_types": sorted({r.get("type", "?") for r in remotes}),
        "status": meta.get("status"),
        "updated_at": meta.get("updatedAt"),
        "is_latest": meta.get("isLatest"),
        # New in V3: what you would have to supply to actually run it.
        "credentials": [c.to_dict() for c in credential_requirements(server)],
    }


def credential_requirements(server: dict[str, Any]) -> list[CredentialRequirement]:
    """Everything the registry says this server needs supplied before it runs.

    Three declaration sites, all optional and all shaped the same way: a package's
    ``environmentVariables`` and ``packageArguments``, and a remote's ``headers``.
    Deduplicated by name, because a server offering both transports often declares
    the same secret twice.

    ``packageArguments`` are only counted when marked ``isSecret``. Publishers mark
    plain launch flags ``isRequired`` — one real entry declares ``stdio`` and
    ``http`` that way — and listing those as things the user must supply turns a
    credential checklist into noise.

    **A value containing a ``{placeholder}`` is required whatever ``isRequired``
    says.** Real entries declare a header as
    ``{"name": "Authorization", "value": "Bearer {smithery_api_key}"}`` and omit
    ``isRequired`` — read literally that is "optional", so the install gate passed
    and we wrote a server that could not possibly connect. An unfilled placeholder
    is the publisher stating the value must come from somewhere.
    """
    found: dict[str, CredentialRequirement] = {}

    def add(spec: Any, where: str) -> None:
        if not isinstance(spec, dict):
            return
        name = spec.get("name") or spec.get("value")
        if not name or not isinstance(name, str):
            return
        template = str(spec.get("value") or "")
        templated = bool(_PLACEHOLDER.search(template))
        # A bearer token declared without `isSecret` is still a bearer token; the
        # studio masks on this flag, so guessing low here leaks it into the UI.
        secret = bool(spec.get("isSecret")) or templated or _looks_secret(name)
        if where == "argument" and not secret:
            return
        found.setdefault(name, CredentialRequirement(
            name=name,
            description=str(spec.get("description") or ""),
            # The registry schema defaults `isRequired` to false, and so must we —
            # treating an optional variable as mandatory makes `install_mcp_server`
            # refuse a server that would have started perfectly well. An unfilled
            # placeholder is the exception: that one cannot be left alone.
            required=bool(spec.get("isRequired", False)) or templated,
            secret=secret,
            where=where,
            template=template if templated else "",
        ))

    for pkg in server.get("packages") or []:
        for spec in pkg.get("environmentVariables") or []:
            add(spec, "env")
        for spec in pkg.get("packageArguments") or []:
            add(spec, "argument")
    for remote in server.get("remotes") or []:
        for spec in remote.get("headers") or []:
            add(spec, "header")
    return list(found.values())


def search_registry(query: str, *, limit: int = 20, cursor: str = "") -> list[RegistryHit]:
    """Search the registry, returning flattened hits with their credential needs.

    ``version=latest`` is always applied: the registry stores every published
    version as its own row, so without it a list is mostly duplicates.
    """
    payload = registry_get(
        "/servers",
        {"search": query, "limit": limit, "cursor": cursor, "version": "latest"},
    )
    hits: list[RegistryHit] = []
    for entry in payload.get("servers") or []:
        server = entry.get("server", entry)
        if not server.get("name"):
            continue
        hits.append(RegistryHit(
            name=str(server["name"]),
            description=str(server.get("description") or ""),
            version=str(server.get("version") or ""),
            runs_locally=bool(server.get("packages")),
            runs_remotely=bool(server.get("remotes")),
            credentials=credential_requirements(server),
            server=server,
        ))
    return hits


def registry_detail(name: str) -> dict[str, Any]:
    """The latest published version of one registry entry."""
    from urllib.parse import quote

    # Registry names are reverse-DNS with a slash (`com.example/thing`); the slash
    # must be percent-encoded or the upstream path doesn't match.
    payload = registry_get(f"/servers/{quote(name, safe='')}/versions", {})
    versions = payload.get("servers") or []
    if not versions:
        raise McpRegistryError(f"'{name}' is not in the registry", status=404)
    latest = next(
        (
            v for v in versions
            if ((v.get("_meta") or {})
                .get("io.modelcontextprotocol.registry/official", {})
                .get("isLatest"))
        ),
        versions[-1],
    )
    return {
        "summary": entry_summary(latest),
        "server": latest.get("server", {}),
        "versions": [(v.get("server") or {}).get("version") for v in versions],
    }


def default_runtime(registry_type: str | None) -> str | None:
    return {"npm": "npx", "pypi": "uvx", "oci": "docker"}.get(registry_type or "")


def _remote_headers(
    remote: dict[str, Any], supplied: dict[str, str]
) -> dict[str, str]:
    """The headers a remote actually needs, built from what the publisher declared.

    The declared header used to be discarded and only `supplied` kept, so a server
    whose entry said ``Authorization: "Bearer {smithery_api_key}"`` was installed
    with **no** headers at all and could never connect.

    Three cases, in order: a header supplied by its exact name wins outright; a
    declared template has its `{placeholder}` filled from a supplied value of that
    placeholder's name (so ``smithery_api_key`` fills ``Bearer {…}`` and keeps the
    publisher's framing — a bare token in an `Authorization` header is not a bearer
    token); a declared literal is carried through as-is.
    """
    out: dict[str, str] = {}
    # Supplied keys consumed as placeholder *values* rather than as header names —
    # `smithery_api_key` fills `Bearer {…}` and must not also become a header of
    # its own.
    consumed: set[str] = set()

    for spec in remote.get("headers") or []:
        if not isinstance(spec, dict):
            continue
        name = spec.get("name")
        if not name or not isinstance(name, str):
            continue
        if name in supplied:
            out[name] = supplied[name]
            continue
        template = str(spec.get("value") or "")
        if not template:
            continue

        def fill(m: re.Match[str]) -> str:
            key = m.group(0)[1:-1]
            if key in supplied:
                consumed.add(key)
                return supplied[key]
            return m.group(0)

        filled = _PLACEHOLDER.sub(fill, template)
        # Still holding a placeholder ⇒ nobody supplied it. Writing the literal
        # `Bearer {smithery_api_key}` onto the wire is worse than omitting it: the
        # server rejects it as a malformed token rather than as a missing one.
        if not _PLACEHOLDER.search(filled):
            out[name] = filled

    # Anything supplied that the entry never declared is still the caller's call.
    for name, value in supplied.items():
        if name not in consumed:
            out.setdefault(name, value)
    return out


def install_config(server: dict[str, Any], body: dict[str, Any] | None = None):
    """Turn a registry entry (+ user-supplied env/args) into an McpServerConfig.

    Prefers a remote (HTTP) transport when the entry offers one — it runs no code
    on this machine. Pass ``prefer: "package"`` to force the stdio route.
    """
    from neurosurfer.config.mcp import McpServerConfig

    body = body or {}
    local_name = body.get("local_name") or (server.get("name") or "").split("/")[-1]
    env: dict[str, str] = dict(body.get("env") or {})
    headers: dict[str, str] = dict(body.get("headers") or {})
    prefer = body.get("prefer")

    remotes = server.get("remotes") or []
    packages = server.get("packages") or []

    if remotes and prefer != "package":
        remote = remotes[0]
        return McpServerConfig(
            name=local_name,
            transport="http",
            url=remote.get("url"),
            headers=_remote_headers(remote, headers),
            enabled=False,
        )

    if not packages:
        raise McpRegistryError(
            "This registry entry has neither a remote endpoint nor a package.",
            status=422,
        )

    pkg = packages[0]
    runtime = pkg.get("runtimeHint") or default_runtime(pkg.get("registryType"))
    if not runtime:
        raise McpRegistryError(
            f"Don't know how to run a '{pkg.get('registryType')}' package.",
            status=422,
        )
    args = [a.get("value") for a in (pkg.get("runtimeArguments") or []) if a.get("value")]
    identifier = pkg.get("identifier")
    version = pkg.get("version")
    spec = f"{identifier}@{version}" if identifier and version else identifier
    if spec:
        args.append(spec)
    args += [a.get("value") for a in (pkg.get("packageArguments") or []) if a.get("value")]

    # A package can declare that it serves HTTP rather than speaking over pipes.
    # Assuming stdio for every package installs those in a shape that can never
    # connect, and the failure ("Connection closed") says nothing about why.
    declared = (pkg.get("transport") or {})
    kind = (declared.get("type") or "stdio").strip().lower()
    is_http = kind in ("http", "streamable-http", "streamable_http", "sse")

    return McpServerConfig(
        name=local_name,
        transport="http" if is_http else "stdio",
        command=runtime,
        args=[a for a in args if a],
        env=env,
        url=declared.get("url") if is_http else None,
        headers=headers if is_http else {},
        enabled=False,
    )
