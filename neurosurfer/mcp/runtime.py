"""Process-lifetime MCP host for non-CLI contexts (Phase 6, per-server since S6).

The CLI owns its own :class:`McpManager` on its main event loop. Everything else —
the workflow runner (synchronous), the Architect agent, the server — needs MCP
tools without owning an event loop that outlives the call. This module hosts them
on a dedicated daemon thread's loop.

**One manager per server.** Each configured server gets its own
:class:`McpManager` driven by its own long-lived task:
``connect → wait for stop → aclose``. That shape is required (the anyio stdio
transports demand same-task setup/teardown) *and* it is what makes independent
start/stop possible: stopping one server no longer tears down the others, which
the studio's per-server controls depend on.

The host loop stays alive for the process, so :class:`McpTool`'s cross-loop
marshalling (``run_coroutine_threadsafe`` back to its home loop) works from any
thread or loop — graph-executor threads included.

This module — not the individual managers — owns publishing to the global live
tool registry, because that registry is a single replaceable list and each
manager only knows about its own tools. Every start/stop republishes the union.

``ensure_mcp_tools()`` is sync, idempotent, and safe to call opportunistically:
no configured servers → no-op; already running → returns the current statuses.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from neurosurfer.config.mcp import McpServerConfig, McpStore

    from .manager import McpManager, ServerStatus

logger = logging.getLogger(__name__)

__all__ = [
    "ensure_mcp_tools",
    "shutdown_mcp",
    "mcp_statuses",
    "start_server",
    "stop_server",
    "server_status",
    "is_running",
    "running_names",
]

_lock = threading.RLock()
_thread: threading.Thread | None = None
_loop: asyncio.AbstractEventLoop | None = None
_loop_ready = threading.Event()


@dataclass
class _Running:
    """One connected server: its manager, its stop switch, its last status."""

    name: str
    manager: McpManager
    stop: asyncio.Event
    status: ServerStatus
    tools: list[Any] = field(default_factory=list)


_running: dict[str, _Running] = {}


# ── shared host loop ─────────────────────────────────────────────────────────

def _ensure_host_loop() -> asyncio.AbstractEventLoop:
    """Start (once) the daemon thread whose loop hosts every MCP session."""
    global _thread, _loop
    with _lock:
        if _thread is not None and _thread.is_alive() and _loop is not None:
            return _loop

        _loop_ready.clear()

        def _run() -> None:
            global _loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            _loop = loop
            _loop_ready.set()
            try:
                loop.run_forever()
            finally:
                loop.close()
                _loop = None

        _thread = threading.Thread(target=_run, name="mcp-host", daemon=True)
        _thread.start()

    if not _loop_ready.wait(10.0) or _loop is None:
        raise TimeoutError("MCP host loop failed to start")
    return _loop


def _republish() -> None:
    """Publish the union of every running server's tools to the live registry.

    Name clashes across servers are resolved first-wins and logged — a per-server
    manager can only de-duplicate within itself, so cross-server collisions have
    to be settled here. `tool_prefix` in a server's config avoids them entirely.
    """
    from neurosurfer.tools.registry import set_live_tools

    merged: list[Any] = []
    seen: set[str] = set()
    for entry in _running.values():
        for tool in entry.tools:
            if tool.name in seen:
                logger.warning(
                    "MCP tool name clash: '%s' from server '%s' is shadowed; "
                    "set a tool_prefix on one of the servers",
                    tool.name, entry.name,
                )
                continue
            seen.add(tool.name)
            merged.append(tool)
    set_live_tools(merged)


# ── per-server lifecycle ─────────────────────────────────────────────────────

async def _serve(cfg: McpServerConfig, ready: threading.Event, box: dict[str, Any]) -> None:
    """One task per server: connect, signal, wait for stop, close."""
    from .manager import McpManager

    manager = McpManager([cfg])
    stop = asyncio.Event()
    try:
        # publish=False: this module owns the global registry (see _republish).
        statuses = await manager.connect_all(publish=False)
    except Exception as e:  # noqa: BLE001 - surface to the waiting caller
        box["error"] = e
        ready.set()
        return

    status = statuses[0] if statuses else None
    box["status"] = status
    if status is not None and status.connected:
        with _lock:
            _running[cfg.name] = _Running(
                name=cfg.name, manager=manager, stop=stop,
                status=status, tools=list(manager.tools()),
            )
            _republish()
    ready.set()

    if status is None or not status.connected:
        await manager.aclose()
        return

    try:
        await stop.wait()
    finally:
        await manager.aclose()
        with _lock:
            _running.pop(cfg.name, None)
            _republish()


def start_server(cfg: McpServerConfig, *, timeout: float = 30.0) -> ServerStatus:
    """Connect one server and publish its tools. Idempotent per server name."""
    from .manager import ServerStatus

    with _lock:
        existing = _running.get(cfg.name)
        if existing is not None:
            return existing.status

    loop = _ensure_host_loop()
    ready = threading.Event()
    box: dict[str, Any] = {}
    asyncio.run_coroutine_threadsafe(_serve(cfg, ready, box), loop)

    if not ready.wait(timeout):
        return ServerStatus(
            cfg.name, connected=False,
            error=f"did not connect within {timeout}s",
        )
    if "error" in box:
        err = box["error"]
        return ServerStatus(cfg.name, connected=False, error=f"{type(err).__name__}: {err}")
    status = box.get("status")
    if status is None:
        return ServerStatus(cfg.name, connected=False, error="server reported no status")
    return status


def stop_server(name: str, *, timeout: float = 10.0) -> bool:
    """Disconnect one server. Returns False if it wasn't running."""
    with _lock:
        entry = _running.get(name)
        loop = _loop
    if entry is None or loop is None:
        return False

    done = threading.Event()

    def _set() -> None:
        entry.stop.set()
        done.set()

    loop.call_soon_threadsafe(_set)
    done.wait(timeout)
    # The task's finally-block removes it from _running; give it a moment.
    deadline = threading.Event()
    for _ in range(int(timeout * 20)):
        with _lock:
            if name not in _running:
                return True
        deadline.wait(0.05)
    return name not in _running


def is_running(name: str) -> bool:
    with _lock:
        return name in _running


def running_names() -> list[str]:
    with _lock:
        return sorted(_running)


def server_status(name: str) -> ServerStatus | None:
    with _lock:
        entry = _running.get(name)
        return entry.status if entry else None


# ── bulk helpers (the pre-existing public surface) ───────────────────────────

def ensure_mcp_tools(
    store: McpStore | None = None,
    *,
    timeout: float = 30.0,
    secrets: Mapping[str, str] | None = None,
) -> list[ServerStatus]:
    """Connect every enabled configured server (once) and return their statuses.

    Publishes discovered tools to the live registry as a side effect. Returns []
    when nothing is configured. Servers already running are left alone.

    *secrets* fills the config's `${VAR}` references, defaulting to whatever the
    calling context has bound. It is account-scoped and this module is host-level,
    so it cannot be read from here directly; without it a workflow reconnecting
    mid-run would resolve against the process environment alone and get an empty
    password where the user had set one.
    """
    from neurosurfer.config.mcp import McpStore as _Store

    if secrets is None:
        from .credentials import known_credentials

        secrets = known_credentials()

    servers = [s for s in (store or _Store.default()).list() if s.enabled]
    if not servers:
        return []

    statuses: list[ServerStatus] = []
    for cfg in servers:
        statuses.append(start_server(cfg.with_secrets(secrets), timeout=timeout))
    return statuses


def mcp_statuses() -> list[Any]:
    """Statuses of every currently-running server (may be empty)."""
    with _lock:
        return [entry.status for entry in _running.values()]


def shutdown_mcp(timeout: float = 10.0) -> None:
    """Disconnect every server (mainly for tests / clean shutdown).

    The host loop itself is left running — it is cheap, and restarting it on the
    next connect would race with tools still marshalling back to it.
    """
    for name in running_names():
        stop_server(name, timeout=timeout)
