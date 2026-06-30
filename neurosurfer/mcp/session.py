"""McpSession — synchronous, thread-safe MCP connection.

Wraps McpManager with a private background event loop so MCP tools work
from any context without the caller needing to understand asyncio::

    with McpSession([McpServerConfig(name="demo", ...)]) as session:
        pool = ToolPool([*session.tools(), FinishTool()])
        executor = GraphExecutor(graph, provider=..., native_tools=pool)
        result = executor.run(inputs)

The session runs its own daemon thread with a dedicated event loop. All
tool calls are automatically routed back to that loop via the bridge built
into :class:`~neurosurfer.mcp.tool.McpTool`, so they work correctly from
graph executor threads, pytest fixtures, scripts, and anywhere else.
"""
from __future__ import annotations

import asyncio
import threading
from typing import TYPE_CHECKING

from ..tools.base import Tool
from .manager import McpManager, ServerStatus

if TYPE_CHECKING:
    from ..config.mcp import McpServerConfig


class McpSession:
    """Synchronous context manager for MCP connections.

    Unlike :class:`~neurosurfer.mcp.McpManager` (which is async-first),
    ``McpSession`` is entirely synchronous and safe to use from any thread or
    environment::

        with McpSession(servers) as session:
            for st in session.status():
                print(st.name, st.tools)
            pool = ToolPool([*session.tools(), FinishTool()])

    The backing event loop runs on a private daemon thread. The loop is torn
    down cleanly on ``__exit__``/``close``.
    """

    def __init__(self, servers: list[McpServerConfig]) -> None:
        self._servers = list(servers)
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._mgr: McpManager | None = None

    # ── lifecycle ────────────────────────────────────────────────────────────
    def connect(self) -> list[ServerStatus]:
        """Start the background loop and connect all servers.

        Called automatically by ``__enter__``; call manually if not using the
        context-manager form.
        """
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._loop.run_forever,
            daemon=True,
            name="neurosurfer-mcp",
        )
        self._thread.start()
        self._mgr = McpManager(self._servers)
        statuses = asyncio.run_coroutine_threadsafe(
            self._mgr.connect_all(), self._loop
        ).result()
        return statuses

    def close(self) -> None:
        """Disconnect all servers and stop the background loop.

        Called automatically by ``__exit__``; call manually to match a manual
        ``connect()``.
        """
        if self._mgr is not None and self._loop is not None:
            try:
                asyncio.run_coroutine_threadsafe(
                    self._mgr.aclose(), self._loop
                ).result(timeout=10)
            except Exception:  # noqa: BLE001 - best-effort teardown
                pass
            self._mgr = None
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._loop.stop)
            if self._thread is not None:
                self._thread.join(timeout=5)
            self._loop = None
            self._thread = None

    def __enter__(self) -> McpSession:
        self.connect()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ── introspection ────────────────────────────────────────────────────────
    def _require_connected(self) -> McpManager:
        if self._mgr is None:
            raise RuntimeError(
                "McpSession is not connected. "
                "Use it as a context manager or call connect() first."
            )
        return self._mgr

    def tools(self) -> list[Tool]:
        return self._require_connected().tools()

    def status(self) -> list[ServerStatus]:
        return self._require_connected().status()

    def tool_count(self) -> int:
        return self._require_connected().tool_count()
