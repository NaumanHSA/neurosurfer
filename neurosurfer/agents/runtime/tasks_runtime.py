"""Background task runtime — lightweight asyncio job manager.

A :class:`TaskHandle` wraps a coroutine and exposes its lifecycle (status /
start+finish times / result / error). :class:`TasksRuntime` submits handles under
two guards:

* **no-overlap** — at most one *live* handle per ``name``, and
* **global cap** — at most ``max_concurrent`` handles running at once.
"""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from dataclasses import dataclass, field
from datetime import UTC, datetime
from functools import partial
from typing import Any, Literal

HandleStatus = Literal["running", "done", "error", "cancelled"]


def _now() -> datetime:
    return datetime.now(tz=UTC)


@dataclass
class TaskHandle:
    """A running or completed background task."""

    task_id: str
    description: str
    _task: asyncio.Task[Any] = field(repr=False, compare=False)
    name: str = ""
    started_at: datetime = field(default_factory=_now)
    finished_at: datetime | None = field(default=None, compare=False)

    @property
    def done(self) -> bool:
        return self._task.done()

    @property
    def status(self) -> HandleStatus:
        if not self._task.done():
            return "running"
        if self._task.cancelled():
            return "cancelled"
        return "error" if self._task.exception() is not None else "done"

    @property
    def error(self) -> BaseException | None:
        if self._task.done() and not self._task.cancelled():
            return self._task.exception()
        return None

    @property
    def result_value(self) -> Any:
        """The return value if the task finished cleanly, else ``None``."""
        return self._task.result() if self.status == "done" else None

    async def result(self) -> Any:
        return await self._task

    def cancel(self) -> None:
        self._task.cancel()


class TasksRuntime:
    """Foreground async job manager with no-overlap + concurrency guards."""

    def __init__(self, max_concurrent: int = 8) -> None:
        self._handles: dict[str, TaskHandle] = {}
        self._counter = 0
        self.max_concurrent = max(1, max_concurrent)

    def submit(
        self,
        coro: Coroutine[Any, Any, Any],
        *,
        name: str | None = None,
        description: str = "",
    ) -> TaskHandle | None:
        """Schedule ``coro``. Returns the handle, or ``None`` if a guard blocked it.

        Blocked when the global concurrency cap is reached, or when ``name`` is
        already live (no-overlap). A blocked coroutine is closed so it does not
        leak an "never awaited" warning.
        """
        if len(self.active()) >= self.max_concurrent:
            coro.close()
            return None
        if name is not None and self.is_live(name):
            coro.close()
            return None

        self._counter += 1
        task_id = f"task-{self._counter}"
        asyncio_task = asyncio.create_task(coro, name=task_id)
        handle = TaskHandle(
            task_id=task_id,
            description=description,
            _task=asyncio_task,
            name=name or task_id,
        )
        asyncio_task.add_done_callback(partial(_mark_finished, handle))
        self._handles[task_id] = handle
        return handle

    def active(self) -> list[TaskHandle]:
        return [h for h in self._handles.values() if not h.done]

    def all(self) -> list[TaskHandle]:
        return list(self._handles.values())

    def is_live(self, name: str) -> bool:
        return any(h.name == name and not h.done for h in self._handles.values())

    def get_live(self, name: str) -> TaskHandle | None:
        for h in self._handles.values():
            if h.name == name and not h.done:
                return h
        return None

    def cancel(self, name: str) -> bool:
        """Cancel the live handle for ``name`` (if any). Returns whether it acted."""
        handle = self.get_live(name)
        if handle is None:
            return False
        handle.cancel()
        return True

    async def shutdown(self) -> None:
        """Cancel every live handle and wait for them to unwind."""
        live = self.active()
        for h in live:
            h.cancel()
        if live:
            await asyncio.gather(*(h._task for h in live), return_exceptions=True)


def _mark_finished(handle: TaskHandle, _task: asyncio.Task[Any]) -> None:
    handle.finished_at = _now()
