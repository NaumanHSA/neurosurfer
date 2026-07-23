"""Process-lifetime MCP host for non-CLI contexts (Phase 6).

The CLI owns its own :class:`McpManager` on its main event loop. Everything else —
the workflow runner (synchronous), the Architect agent, the server — needs MCP
tools without owning an event loop that outlives the call. This module hosts the
manager on a dedicated daemon thread's loop:

- ``connect_all`` and ``aclose`` run inside **one** coroutine on that loop (the
  anyio stdio transports require same-task setup/teardown).
- The host loop stays alive for the process, so :class:`McpTool`'s cross-loop
  marshalling (``run_coroutine_threadsafe`` back to its home loop) works from any
  thread or loop — graph-executor threads included.
- Discovered tools are published to the global live-tool registry by the manager,
  making them visible to ``all_tools()`` / ``workflow_node_tools()``.

``ensure_mcp_tools()`` is sync, idempotent, and safe to call opportunistically:
no configured servers → no-op; already hosted → returns the cached statuses.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from neurosurfer.config.mcp import McpStore

    from .manager import ServerStatus

logger = logging.getLogger(__name__)

__all__ = ["ensure_mcp_tools", "shutdown_mcp", "mcp_statuses"]

_lock = threading.Lock()
_thread: threading.Thread | None = None
_loop: asyncio.AbstractEventLoop | None = None
_stop: asyncio.Event | None = None
_statuses: list[Any] = []


def ensure_mcp_tools(
    store: McpStore | None = None, *, timeout: float = 30.0
) -> list[ServerStatus]:
    """Connect all enabled configured MCP servers (once) and return their statuses.

    Publishes discovered tools to the live registry as a side effect. Returns []
    when nothing is configured. Raises TimeoutError if connection setup hangs.
    """
    global _thread, _statuses
    with _lock:
        if _thread is not None and _thread.is_alive():
            return list(_statuses)

        from neurosurfer.config.mcp import McpStore

        servers = [s for s in (store or McpStore.default()).list() if s.enabled]
        if not servers:
            return []

        ready: threading.Event = threading.Event()
        box: dict[str, Any] = {}

        def _run_host() -> None:
            global _loop, _stop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            _loop = loop
            try:
                loop.run_until_complete(_host(servers, ready, box))
            finally:
                loop.close()
                _loop = None

        _thread = threading.Thread(target=_run_host, name="mcp-host", daemon=True)
        _thread.start()
        if not ready.wait(timeout):
            raise TimeoutError(f"MCP servers did not connect within {timeout}s")
        if "error" in box:
            _thread = None
            raise box["error"]
        _statuses = box.get("statuses", [])
        return list(_statuses)


async def _host(servers: list[Any], ready: threading.Event, box: dict[str, Any]) -> None:
    """One task: connect, publish, wait for shutdown, close — in that order."""
    global _stop
    from .manager import McpManager

    _stop = asyncio.Event()
    manager = McpManager(servers)
    try:
        box["statuses"] = await manager.connect_all()
    except Exception as e:  # noqa: BLE001 - surface to the waiting caller
        box["error"] = e
        ready.set()
        return
    ready.set()
    try:
        await _stop.wait()
    finally:
        await manager.aclose()  # also clears the live-tool registry


def mcp_statuses() -> list[Any]:
    """Statuses from the last successful ``ensure_mcp_tools`` (may be empty)."""
    return list(_statuses)


def shutdown_mcp(timeout: float = 10.0) -> None:
    """Disconnect and stop the host thread (mainly for tests / clean shutdown)."""
    global _thread, _statuses
    with _lock:
        thread, loop, stop = _thread, _loop, _stop
        _thread = None
        _statuses = []
    if thread is None or not thread.is_alive() or loop is None or stop is None:
        return
    loop.call_soon_threadsafe(stop.set)
    thread.join(timeout)
