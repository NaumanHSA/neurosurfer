"""Capture — turn a finished run's decisions into low-salience candidate memories.

Rule-based, no extra LLM call: at run end we read ``DurableState.decisions`` and
emit one low-salience ``decision`` memory per durable decision. ``MemoryStore.add``
already dedups, so re-running the same task doesn't flood the store. Distillation is
opt-out via config so headless/automation runs can disable it.

Decisions are agent-behaviour artefacts, so they land in the **agent** scope when an
agent name is known (else global). Salience is deliberately low — these are weak
signals that earn their place only if retrieval keeps surfacing them (``record_use``).
"""

from __future__ import annotations

from typing import cast

from ..agents.context.durable_state import DurableState
from .models import MemoryEntry, MemoryScope
from .store import MemoryStore

_CANDIDATE_SALIENCE = 0.4
_MAX_PER_RUN = 10
_MIN_LEN = 8


def distill_run(
    store: MemoryStore,
    durable: DurableState | None,
    *,
    agent: str | None,
    session_id: str | None = None,
) -> list[MemoryEntry]:
    """Persist candidate memories from a run's decisions. Returns what was added."""
    if durable is None or not durable.decisions:
        return []
    scope: MemoryScope = cast(MemoryScope, "agent" if agent else "global")
    scope_key = agent or ""
    added: list[MemoryEntry] = []
    seen: set[str] = set()
    for raw in durable.decisions[:_MAX_PER_RUN]:
        text = raw.strip()
        key = text.lower()
        if len(text) < _MIN_LEN or key in seen:
            continue
        seen.add(key)
        entry = store.add(
            MemoryEntry(
                scope=scope,
                scope_key=scope_key,
                kind="decision",
                text=text,
                source="distill",
                session_id=session_id,
                salience=_CANDIDATE_SALIENCE,
            )
        )
        added.append(entry)
    return added
