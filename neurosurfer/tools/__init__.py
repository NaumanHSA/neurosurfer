"""The curated tool pool.

A Task narrows this pool via its ``tools:`` allow-list. Because the pool is small,
all selected schemas are sent every turn (no deferred/ToolSearch indirection).
"""

from __future__ import annotations

from .base import (
    AutoApproveIOHandler,
    BaseIOHandler,
    FileState,
    IOHandler,
    TerminalIOHandler,
    Tool,
    ToolContext,
    ToolPool,
    ToolResult,
)
from .registry import all_tools, build_pool, default_pool

__all__ = [
    "Tool",
    "ToolPool",
    "ToolResult",
    "ToolContext",
    "IOHandler",
    "BaseIOHandler",
    "AutoApproveIOHandler",
    "TerminalIOHandler",
    "FileState",
    "all_tools",
    "default_pool",
    "build_pool",
]
