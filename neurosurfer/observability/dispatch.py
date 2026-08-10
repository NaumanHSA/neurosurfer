"""Trace export runs on one background thread. The agent's thread only enqueues.

## Why this exists

Tracing is a side channel. It is allowed to be slow, to fail, and to lose data;
it is **not** allowed to make a run slower. Before this module every exporter
hook ran inline on whatever thread the agent was on, and `flush()` ran there
too — and `flush()` is not the cheap thing its name suggests. OpenTelemetry's
`BatchSpanProcessor` already owns a queue and a worker thread; `force_flush`
exists to *bypass* them and drain on the caller. Calling it at every run finish
meant the agent paid the full network cost of export, synchronously, per node.

With no collector listening that bill was measured at **8.2s per flush** on
Windows (a closed port is not refused instantly there, and `localhost` is tried
on both `::1` and `127.0.0.1`, then retried). Tutorial 03's router cell spent 74
seconds waiting on a socket nobody answered, against a model that replied in
under two. On Linux the same misconfiguration cost ~0ms — which is exactly what
makes an inline call dangerous: it is free on the machine you wrote it on.

So no exporter call happens on the agent's thread any more. Hooks and flushes
are submitted here and the caller returns immediately.

## What this guarantees, and what it does not

**Order is preserved.** One worker draining a FIFO means `on_run_start` really
does reach an exporter before the `on_turn` that follows it. Exporters keep
per-run dicts keyed by span id and would corrupt under reordering.

**Exporter state stops being racy.** A `map` node runs its body concurrently, so
several agent threads used to call into the same exporter's `_runs` dict at
once. Everything now lands on one thread, serialised.

**Delivery is best-effort.** The queue is bounded: past `_MAX_QUEUE` pending
items telemetry is dropped rather than allowed to grow without limit, because a
process that dies of its own monitoring is worse than one missing spans. The
thread is a daemon, so it never holds up interpreter exit; an `atexit` hook
drains it with a short deadline so a script that ends right after a run still
ships what it has.

**Arguments are captured, not copied.** A hook's kwargs are read on the worker
thread, so a caller that mutates a dict it already handed over may be traced
with the later value. Callers pass freshly built summaries today, which is why
this is a note rather than a defensive copy on every event.
"""

from __future__ import annotations

import atexit
import logging
import queue
import threading
from collections.abc import Callable

logger = logging.getLogger("neurosurfer.observability")

#: Pending exporter calls held before telemetry starts being dropped. Large
#: enough that no realistic run reaches it, small enough to bound memory.
_MAX_QUEUE = 10_000

#: How long `atexit` waits for the backlog at interpreter shutdown.
_EXIT_DRAIN_SECONDS = 3.0


class _ExportThread:
    """A daemon worker draining a FIFO of exporter callbacks."""

    def __init__(self, max_queue: int = _MAX_QUEUE) -> None:
        self._queue: queue.Queue[Callable[[], None]] = queue.Queue(maxsize=max_queue)
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._dropped = 0

    # ── worker ──────────────────────────────────────────────────────────────
    def _worker(self) -> None:
        while True:
            task = self._queue.get()
            try:
                task()
            except Exception:  # noqa: BLE001 — telemetry must never take the process
                logger.debug("trace export task failed", exc_info=True)
            finally:
                self._queue.task_done()

    def _ensure_started(self) -> None:
        # Started on first use, so importing neurosurfer costs no thread, and
        # restarted if it is missing — which is also what a forked child needs.
        if self._thread is not None and self._thread.is_alive():
            return
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._thread = threading.Thread(
                target=self._worker, name="neurosurfer-trace-export", daemon=True
            )
            self._thread.start()

    # ── api ─────────────────────────────────────────────────────────────────
    def submit(self, task: Callable[[], None]) -> None:
        """Queue *task* and return. Never raises, never blocks."""
        self._ensure_started()
        try:
            self._queue.put_nowait(task)
        except queue.Full:
            self._dropped += 1
            if self._dropped == 1:
                logger.warning(
                    "Trace export queue is full (%d pending); dropping telemetry for "
                    "the rest of this session. The run is unaffected — the backend is "
                    "not keeping up, or is unreachable.",
                    _MAX_QUEUE,
                )

    def drain(self, timeout: float = 5.0) -> bool:
        """Block until everything queued *so far* has run. Returns False on timeout.

        For tests and shutdown. A run must never call this — waiting for the
        exporter is the thing this module exists to stop doing.
        """
        if self._thread is None or not self._thread.is_alive():
            return True
        done = threading.Event()
        try:
            self._queue.put_nowait(done.set)   # FIFO: runs after all current work
        except queue.Full:
            return False
        return done.wait(timeout)

    @property
    def dropped(self) -> int:
        return self._dropped


_DISPATCH = _ExportThread()


def submit(task: Callable[[], None]) -> None:
    """Run *task* on the trace-export thread."""
    _DISPATCH.submit(task)


def drain(timeout: float = 5.0) -> bool:
    """Wait for queued trace work to finish. Returns False if it did not."""
    return _DISPATCH.drain(timeout)


def dropped() -> int:
    """How many exporter calls were discarded because the queue was full."""
    return _DISPATCH.dropped


@atexit.register
def _drain_at_exit() -> None:
    _DISPATCH.drain(_EXIT_DRAIN_SECONDS)
