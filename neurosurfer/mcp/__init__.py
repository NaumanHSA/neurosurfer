"""MCP (Model Context Protocol) integration.

Phase 1 — **client**: connect to external MCP servers and surface their tools as
neurosurfer :class:`~neurosurfer.tools.base.Tool` objects.

- :class:`~neurosurfer.mcp.manager.McpManager` — connection lifecycle + discovery.
- :class:`~neurosurfer.mcp.tool.McpTool` — one remote tool wrapped as a Tool.

Configuration lives in :mod:`neurosurfer.config.mcp` (``McpStore`` / ``McpServerConfig``).
The ``mcp`` SDK is an optional dependency (``pip install neurosurfer[mcp]``); importing
this package does not import it — that happens lazily when a manager actually connects.
"""

from __future__ import annotations

from .manager import McpManager, ServerStatus
from .tool import McpTool

__all__ = ["McpManager", "ServerStatus", "McpTool"]
