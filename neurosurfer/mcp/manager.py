"""``McpManager`` — connect to configured MCP servers and expose their tools.

Lifecycle (one manager per CLI session / app process)::

    mgr = McpManager(store.enabled())
    await mgr.connect_all()        # opens sessions, discovers + registers tools
    ...                            # agents now see MCP tools via tools.all_tools()
    await mgr.aclose()             # tears every session down

Sessions are held open with a single :class:`contextlib.AsyncExitStack`, so
``connect_all`` and ``aclose`` **must run in the same task** (the anyio transports
own cancel scopes). In the CLI both run inside the long-lived REPL task; in tests use
the same coroutine. One unreachable server never breaks the others — failures are
collected per server and surfaced through :meth:`status`.

On success the discovered :class:`McpTool` instances are published to the global live
registry (:func:`neurosurfer.tools.registry.set_live_tools`) so every ``all_tools()``
caller (agents, sub-agents) picks them up without extra wiring.
"""

from __future__ import annotations

import logging
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from datetime import timedelta
from typing import TYPE_CHECKING

from ..config.mcp import McpServerConfig
from ..tools.base import Tool
from ..tools.registry import clear_live_tools, set_live_tools
from .tool import McpTool

if TYPE_CHECKING:
    from mcp import ClientSession

log = logging.getLogger(__name__)

# How long to wait for a server to initialize / respond to list_tools.
_CONNECT_TIMEOUT_S = 30.0


@dataclass
class ServerStatus:
    name: str
    connected: bool
    tool_count: int = 0
    tools: list[str] = field(default_factory=list)
    error: str | None = None


class McpManager:
    def __init__(self, servers: list[McpServerConfig]) -> None:
        self._servers = list(servers)
        self._stack: AsyncExitStack | None = None
        self._tools: list[McpTool] = []
        self._status: dict[str, ServerStatus] = {}

    # ── lifecycle ───────────────────────────────────────────────────────────────
    async def connect_all(self) -> list[ServerStatus]:
        """Connect every server, discover tools, and publish them. Returns statuses."""
        if self._stack is not None:
            await self.aclose()
        self._stack = AsyncExitStack()
        self._tools = []
        self._status = {}

        exposed_names: set[str] = set()
        for cfg in self._servers:
            status = await self._connect_one(cfg, exposed_names)
            self._status[cfg.name] = status

        set_live_tools(list(self._tools))
        return list(self._status.values())

    async def _connect_one(
        self, cfg: McpServerConfig, exposed_names: set[str]
    ) -> ServerStatus:
        try:
            session = await self._open_session(cfg)
            defs = (await session.list_tools()).tools
        except Exception as e:  # noqa: BLE001 - isolate a bad server
            log.warning("MCP server '%s' failed to connect: %s", cfg.name, e)
            return ServerStatus(cfg.name, connected=False, error=f"{type(e).__name__}: {e}")

        names: list[str] = []
        for d in defs:
            exposed = self._unique_name(cfg, d.name, exposed_names)
            exposed_names.add(exposed)
            self._tools.append(
                McpTool.from_def(
                    session=session,
                    server_name=cfg.name,
                    definition=d,
                    exposed_name=exposed,
                )
            )
            names.append(exposed)
        return ServerStatus(cfg.name, connected=True, tool_count=len(names), tools=names)

    async def _open_session(self, cfg: McpServerConfig) -> ClientSession:
        """Open + initialize a session on the shared exit stack."""
        from mcp import ClientSession  # noqa: PLC0415 - optional dep, import lazily

        assert self._stack is not None
        timeout = timedelta(seconds=_CONNECT_TIMEOUT_S)

        if cfg.transport == "http":
            from mcp.client.streamable_http import streamablehttp_client  # noqa: PLC0415

            if not cfg.url:
                raise ValueError(f"MCP server '{cfg.name}' has no url.")
            read, write, _ = await self._stack.enter_async_context(
                streamablehttp_client(cfg.url, headers=cfg.resolved_headers() or None)
            )
        else:
            from mcp import StdioServerParameters  # noqa: PLC0415
            from mcp.client.stdio import stdio_client  # noqa: PLC0415

            if not cfg.command:
                raise ValueError(f"MCP server '{cfg.name}' has no command.")
            params = StdioServerParameters(
                command=cfg.command,
                args=list(cfg.args),
                env=cfg.resolved_env() or None,
                cwd=cfg.cwd,
            )
            read, write = await self._stack.enter_async_context(stdio_client(params))

        session = await self._stack.enter_async_context(
            ClientSession(read, write, read_timeout_seconds=timeout)
        )
        await session.initialize()
        return session

    async def aclose(self) -> None:
        """Tear down every session/subprocess. Idempotent."""
        clear_live_tools()
        self._tools = []
        if self._stack is not None:
            stack, self._stack = self._stack, None
            try:
                await stack.aclose()
            except Exception as e:  # noqa: BLE001 - best-effort teardown
                log.debug("MCP teardown raised (ignored): %s", e)

    async def __aenter__(self) -> McpManager:
        await self.connect_all()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    # ── introspection ────────────────────────────────────────────────────────────
    def tools(self) -> list[Tool]:
        return list(self._tools)

    def status(self) -> list[ServerStatus]:
        return list(self._status.values())

    def tool_count(self) -> int:
        return len(self._tools)

    # ── helpers ──────────────────────────────────────────────────────────────────
    @staticmethod
    def _unique_name(cfg: McpServerConfig, remote: str, taken: set[str]) -> str:
        base = f"{cfg.tool_prefix}__{remote}" if cfg.tool_prefix else remote
        if base not in taken:
            return base
        # Collision across servers without an explicit prefix → namespace by server.
        prefixed = f"{cfg.name}__{remote}"
        if prefixed not in taken:
            return prefixed
        i = 2
        while f"{prefixed}_{i}" in taken:
            i += 1
        return f"{prefixed}_{i}"
