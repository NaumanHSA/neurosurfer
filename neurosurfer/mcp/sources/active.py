"""Which discovery engine the work happening *right now* should use.

The chooser in settings is per account, and the code that searches — capability
resolution during an Architect build — sits several frames below any request. It
is a library function about capabilities, not about engines, and threading a
`source` argument down to it would put a discovery parameter on every caller in
between purely so the bottom one can read it.

So the engine is ambient, and this module is the only place that ambience lives.
`active_source()` answers "the engine chosen for this work", falling back to the
official registry, which is always a correct answer to "where do I look for MCP
servers".

**A `ContextVar` does not cross a raw thread.** `threading.Thread` starts with an
empty context, and an Architect build runs on exactly that. So a caller that
hands work to a thread must re-bind inside it — see `ArchitectBuildManager.start`,
which captures the account's engine on the request thread and enters `use_source`
in the worker. Setting it once in the request and expecting the build to inherit
it is the mistake this docstring exists to prevent.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

__all__ = ["active_source", "set_active_source", "use_source"]

_ACTIVE: ContextVar[Any | None] = ContextVar("mcp_active_source", default=None)


def active_source() -> Any:
    """The engine chosen for this context, or the official registry."""
    source = _ACTIVE.get()
    if source is not None:
        return source
    from . import get_source

    return get_source()


def set_active_source(source: Any | None) -> Any:
    """Bind *source* for this context. Returns the token to reset with."""
    return _ACTIVE.set(source)


@contextmanager
def use_source(source: Any | None) -> Iterator[None]:
    """Bind *source* for the duration of the block.

    `None` is a no-op rather than an error: a caller that could not resolve an
    engine wants the default, not a failure on a line that has nothing to do with
    what it was trying to accomplish.
    """
    if source is None:
        yield
        return
    token = _ACTIVE.set(source)
    try:
        yield
    finally:
        _ACTIVE.reset(token)
