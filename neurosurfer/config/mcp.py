"""MCP server profiles — named, persisted Model Context Protocol connections.

A profile captures how to reach one MCP server: either a **stdio** child process
(``command``/``args``/``env``) or a **streamable-HTTP** endpoint (``url``/``headers``).
Profiles persist to ``~/.neurosurfer/mcp.json`` with mode ``0600`` (headers may carry
tokens), and secrets are masked for display.

Mirrors :mod:`neurosurfer.config.profiles` (the provider store) so the two config
surfaces feel the same. ``${ENV_VAR}`` references inside ``env`` / ``headers`` values
are expanded at connect time, so secrets need not be written into the file.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

McpTransport = Literal["stdio", "http"]

_ENV_REF = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def expand_env(value: str, extra: Mapping[str, str] | None = None) -> str:
    """Expand ``${VAR}`` references against *extra*, then the process environment.

    *extra* is the account's stored secrets and wins, because a value someone set
    in Settings is a deliberate choice and should not be overruled by whatever the
    gateway happened to be started with. Before it existed the only way to give a
    server a database password was to edit the gateway's `.env` and restart it.

    An unset variable expands to the empty string (the connection then likely
    fails with a clear auth error, which is friendlier than a silent literal
    ``${VAR}`` leaking onto the wire).
    """
    extra = extra or {}

    def _sub(m: re.Match[str]) -> str:
        name = m.group(1)
        if name in extra:
            return str(extra[name])
        return os.environ.get(name, "")

    return _ENV_REF.sub(_sub, value)


def env_refs(value: str) -> list[str]:
    """Variable names a ``${VAR}``-bearing string references."""
    return _ENV_REF.findall(value or "")


def mask_secret(value: str | None) -> str:
    if not value:
        return ""
    if len(value) <= 8:
        return "•" * len(value)
    return value[:4] + "…" + value[-2:]


class ServerOrigin(BaseModel):
    """What the catalogue knew about a server, kept from the moment it was installed.

    None of it is needed to *connect* — that is `command`/`url` below. It exists so
    an installed server can be shown as something recognisable rather than a name
    and a URL: a description, an icon, how many people use it, what it exposes.

    Every field is optional and every reader must treat it that way. The official
    registry supplies almost none of this (its schema describes how to run a server,
    not what it is), so absent has to render as nothing at all — never as a zero, a
    blank badge, or an empty "Tools" section implying the server has none.
    """

    source: str = ""
    qualified_name: str = ""
    display_name: str = ""
    description: str = ""
    icon_url: str = ""
    homepage: str = ""
    use_count: int | None = None
    verified: bool | None = None
    #: Tools as *published*, which is not the same as the tools a running server
    #: reports. Shown only until the real list is available.
    tools: list[dict[str, str]] = Field(default_factory=list)
    #: Where a human goes to authorise this server, when it needs that.
    setup_url: str = ""


class McpServerConfig(BaseModel):
    """One configured MCP server."""

    name: str
    transport: McpTransport = "stdio"
    enabled: bool = True
    # When set, every tool from this server is exposed as ``<tool_prefix>__<tool>``.
    # Leave unset to use the bare tool name (the manager still disambiguates clashes).
    tool_prefix: str | None = None

    # stdio transport
    command: str | None = None
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    cwd: str | None = None

    # http transport
    url: str | None = None
    headers: dict[str, str] = Field(default_factory=dict)

    # Provenance, for display. Absent on anything installed before this existed
    # and on anything configured by hand, which is why it defaults rather than
    # being required.
    origin: ServerOrigin | None = None

    def endpoint(self) -> str:
        if self.transport == "http":
            return self.url or "(no url)"
        cmd = " ".join([self.command or "(no command)", *self.args]).strip()
        return cmd or "(no command)"

    def resolved_env(self, extra: Mapping[str, str] | None = None) -> dict[str, str]:
        return {k: expand_env(v, extra) for k, v in self.env.items()}

    def resolved_headers(self, extra: Mapping[str, str] | None = None) -> dict[str, str]:
        return {k: expand_env(v, extra) for k, v in self.headers.items()}

    def required_vars(self) -> list[str]:
        """Every ``${VAR}`` this config references, across env and headers.

        What the server needs *supplied*, as opposed to what it was installed
        with — the answer to "why won't this connect" when a secret is missing.
        """
        names: list[str] = []
        for value in (*self.env.values(), *self.headers.values()):
            for ref in env_refs(value):
                if ref not in names:
                    names.append(ref)
        return names

    def with_secrets(self, values: Mapping[str, str] | None) -> McpServerConfig:
        """A copy with every ``${VAR}`` already expanded against *values*.

        Resolution has to happen on the caller's thread, not at connect time: the
        MCP host runs on its own daemon thread with its own event loop, so a
        context bound during a request is not there when the manager connects.
        Expanding first means the config that crosses that boundary carries
        literal values and nothing has to be ambient.

        The stored config is untouched — `mcp.json` keeps the `${VAR}`, so a
        secret is never written to a file shared by every account on the gateway.
        """
        if not values:
            return self
        return self.model_copy(update={
            "env": self.resolved_env(values),
            "headers": self.resolved_headers(values),
        })

    def summary(self) -> str:
        kind = "stdio" if self.transport == "stdio" else "http"
        state = "" if self.enabled else " [disabled]"
        prefix = f" · prefix={self.tool_prefix}" if self.tool_prefix else ""
        secret = ""
        if self.transport == "http" and self.headers:
            # Surface that auth headers exist without leaking them.
            secret = " · " + ", ".join(
                f"{k}={mask_secret(v)}" for k, v in self.headers.items()
            )
        return f"{self.name}: {kind} · {self.endpoint()}{prefix}{secret}{state}"


class _StoreFile(BaseModel):
    servers: list[McpServerConfig] = Field(default_factory=list)


@dataclass
class McpStore:
    """CRUD over the persisted MCP-servers file."""

    path: Path

    @classmethod
    def default(cls, state_home: Path | None = None) -> McpStore:
        """The host-level MCP config.

        Deliberately shared by every workspace: an MCP server is a process the
        gateway spawns as its own OS user, so a per-account copy would suggest
        an isolation that doesn't exist.
        """
        if state_home is not None:
            return cls(path=Path(state_home) / "mcp.json")
        from .paths import mcp_config_path

        return cls(path=mcp_config_path())

    # ── load / save ────────────────────────────────────────────────────────────
    def _read(self) -> _StoreFile:
        if not self.path.exists():
            return _StoreFile()
        try:
            return _StoreFile.model_validate_json(self.path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 - corrupt file → start fresh
            return _StoreFile()

    def _write(self, data: _StoreFile) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(data.model_dump_json(indent=2), encoding="utf-8")
        try:
            os.chmod(self.path, 0o600)  # headers may carry tokens
        except OSError:
            pass

    # ── queries ────────────────────────────────────────────────────────────────
    def list(self) -> list[McpServerConfig]:
        return self._read().servers

    def enabled(self) -> list[McpServerConfig]:
        return [s for s in self._read().servers if s.enabled]

    def names(self) -> list[str]:
        return [s.name for s in self._read().servers]

    def get(self, name: str) -> McpServerConfig | None:
        return next((s for s in self._read().servers if s.name == name), None)

    # ── mutations ──────────────────────────────────────────────────────────────
    def add(self, server: McpServerConfig) -> None:
        data = self._read()
        if any(s.name == server.name for s in data.servers):
            raise ValueError(f"An MCP server named '{server.name}' already exists.")
        data.servers.append(server)
        self._write(data)

    def update(self, name: str, **changes: object) -> McpServerConfig:
        data = self._read()
        for i, s in enumerate(data.servers):
            if s.name == name:
                updated = s.model_copy(
                    update={k: v for k, v in changes.items() if v is not None}
                )
                data.servers[i] = updated
                self._write(data)
                return updated
        raise KeyError(f"No MCP server named '{name}'.")

    def set_enabled(self, name: str, enabled: bool) -> McpServerConfig:
        return self.update(name, enabled=enabled)

    def delete(self, name: str) -> None:
        data = self._read()
        if not any(s.name == name for s in data.servers):
            raise KeyError(f"No MCP server named '{name}'.")
        data.servers = [s for s in data.servers if s.name != name]
        self._write(data)
