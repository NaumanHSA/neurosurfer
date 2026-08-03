"""Human-in-the-loop gates for a running Architect build (S5).

A build runs in a worker thread. Two points in it need a person: the
requirement-gathering conversation asks clarifying questions, and an authored
tool wants approval before it is registered. Both are *async callbacks that
block until a human answers*, so one primitive serves both.

The flow is a rendezvous between two threads:

    build thread                          HTTP request
    ────────────                          ────────────
    gate.ask(...)                 →  record a pending interaction, emit an event
    (parks on a threading.Event)     …studio renders it, user clicks…
                                     POST …/respond  →  gate.respond(id, value)
    returns the value             ←  the Event is set

Parking is deliberate: the agent has no way to proceed without the answer, and a
build waiting on a person is not a stalled build. What it must never do is wait
*forever* — an abandoned browser tab would pin a thread for the process's life —
so every wait has a timeout, after which the gate applies a stated default and
the build carries on. `on_timeout` says which way that default falls: a
clarifying question resolves to "no preference", while an unapproved tool is
rejected, because the safe default for "should I install this generated code?"
is no.

Cancellation also comes through here. `cancel()` wakes every waiter with
`Cancelled`, so a build parked on a question dies as promptly as one mid-LLM
call rather than hanging until its timeout.
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

#: How long a build parks on a human before applying `on_timeout`. Long enough to
#: read a tool's source and decide; short enough that an abandoned tab frees the
#: thread the same afternoon.
DEFAULT_TIMEOUT_S = 30 * 60


class Cancelled(BaseException):
    """Raised inside the build thread when the build is cancelled.

    Deliberately a ``BaseException``, for the same reason ``KeyboardInterrupt``
    is one: this is a control-flow signal that must unwind the whole build, and
    the agent's tool runner catches ``Exception`` to turn tool failures into
    results the model can react to. As an ordinary error, a cancel was swallowed
    there and fed back as a failed tool call — the agent then carried on and
    finished as *blocked* rather than *cancelled*.
    """


@dataclass
class PendingInteraction:
    """One question awaiting an answer, as the studio sees it."""

    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    #: "question" (clarifying) or "tool_approval".
    kind: str = "question"
    prompt: str = ""
    #: Suggested answers. Free text is always allowed for questions.
    choices: list[str] = field(default_factory=list)
    #: Renderable payload — for a tool approval, the draft's source and tests.
    detail: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    expires_at: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "prompt": self.prompt,
            "choices": self.choices,
            "detail": self.detail,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
        }


class InteractionGate:
    """Rendezvous between a build thread and the HTTP request that answers it.

    One gate per build. Only one interaction is pending at a time — the agent is
    sequential, so a second question cannot arise before the first is answered.
    """

    def __init__(self, on_pending: Any = None, on_resolved: Any = None) -> None:
        #: Called with the PendingInteraction when the build starts waiting, and
        #: with (interaction, value) when it stops. Used to emit SSE events.
        self._on_pending = on_pending
        self._on_resolved = on_resolved
        self._lock = threading.Lock()
        self._pending: PendingInteraction | None = None
        self._event = threading.Event()
        self._value: Any = None
        self._cancelled = False

    # ── read ──────────────────────────────────────────────────────────────
    @property
    def pending(self) -> PendingInteraction | None:
        with self._lock:
            return self._pending

    @property
    def cancelled(self) -> bool:
        return self._cancelled

    # ── build thread ──────────────────────────────────────────────────────
    def ask(
        self,
        *,
        kind: str,
        prompt: str,
        choices: list[str] | None = None,
        detail: dict[str, Any] | None = None,
        on_timeout: Any,
        timeout_s: float = DEFAULT_TIMEOUT_S,
    ) -> Any:
        """Park the calling thread until answered, cancelled, or timed out.

        Returns the answer, or *on_timeout* if nobody replied in time.
        """
        self.raise_if_cancelled()
        interaction = PendingInteraction(
            kind=kind,
            prompt=prompt,
            choices=list(choices or []),
            detail=dict(detail or {}),
            expires_at=time.time() + timeout_s,
        )
        with self._lock:
            self._pending = interaction
            self._value = None
            self._event.clear()
        if self._on_pending is not None:
            self._on_pending(interaction)

        answered = self._event.wait(timeout_s)
        with self._lock:
            self._pending = None
            value = self._value if answered else on_timeout
        self.raise_if_cancelled()

        if self._on_resolved is not None:
            self._on_resolved(interaction, value, answered)
        return value

    def raise_if_cancelled(self) -> None:
        if self._cancelled:
            raise Cancelled("build cancelled")

    # ── HTTP thread ───────────────────────────────────────────────────────
    def respond(self, interaction_id: str, value: Any) -> bool:
        """Answer the pending interaction. False if it isn't the one waiting.

        The id must match: a stale tab answering a question the build has already
        moved past would otherwise feed the wrong answer into the next one.
        """
        with self._lock:
            if self._pending is None or self._pending.id != interaction_id:
                return False
            self._value = value
        self._event.set()
        return True

    def cancel(self) -> None:
        """Mark cancelled and wake anything parked."""
        self._cancelled = True
        with self._lock:
            self._pending = None
        self._event.set()
