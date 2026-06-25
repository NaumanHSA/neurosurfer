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
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

McpTransport = Literal["stdio", "http"]

_ENV_REF = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def expand_env(value: str) -> str:
    """Expand ``${VAR}`` references against the process environment.

    An unset variable expands to the empty string (the connection then likely
    fails with a clear auth error, which is friendlier than a silent literal
    ``${VAR}`` leaking onto the wire).
    """
    return _ENV_REF.sub(lambda m: os.environ.get(m.group(1), ""), value)


def mask_secret(value: str | None) -> str:
    if not value:
        return ""
    if len(value) <= 8:
        return "•" * len(value)
    return value[:4] + "…" + value[-2:]


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

    def endpoint(self) -> str:
        if self.transport == "http":
            return self.url or "(no url)"
        cmd = " ".join([self.command or "(no command)", *self.args]).strip()
        return cmd or "(no command)"

    def resolved_env(self) -> dict[str, str]:
        return {k: expand_env(v) for k, v in self.env.items()}

    def resolved_headers(self) -> dict[str, str]:
        return {k: expand_env(v) for k, v in self.headers.items()}

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
        home = state_home or (Path.home() / ".neurosurfer")
        return cls(path=home / "mcp.json")

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
