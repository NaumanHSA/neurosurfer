"""SessionStore — one JSON metadata file + one history file per session.

Layout under root (= ~/.neurosurfer/sessions/):
  <task>/
    <session_id>.json        — SessionRecord metadata
    <session_id>.hist.json   — message history JSON array
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from .history import load_history, save_history
from .models import SessionRecord

if TYPE_CHECKING:
    from ..llm.types import Message


def _safe_name(task: str) -> str:
    """Convert a task name to a filesystem-safe directory component."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", task) or "default"


class SessionStore:
    def __init__(self, root: Path) -> None:
        self._root = root
        self._root.mkdir(parents=True, exist_ok=True)

    # ── directories ────────────────────────────────────────────────────────────

    def _task_dir(self, task: str) -> Path:
        d = self._root / _safe_name(task)
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _meta_path(self, task: str, session_id: str) -> Path:
        return self._task_dir(task) / f"{session_id}.json"

    def _hist_path(self, task: str, session_id: str) -> Path:
        return self._task_dir(task) / f"{session_id}.hist.json"

    # ── create ────────────────────────────────────────────────────────────────

    def create(self, task: str, cwd: str = "") -> SessionRecord:
        rec = SessionRecord(task=task, cwd=cwd)
        self._write_meta(rec)
        return rec

    # ── read ──────────────────────────────────────────────────────────────────

    def get(self, task: str, session_id: str) -> SessionRecord | None:
        path = self._meta_path(task, session_id)
        if not path.exists():
            return None
        try:
            return SessionRecord.model_validate_json(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return None

    def list_for_task(self, task: str, *, limit: int = 20) -> list[SessionRecord]:
        """Return sessions for a task, sorted by updated_at descending."""
        td = self._task_dir(task)
        records: list[SessionRecord] = []
        for p in td.glob("*.json"):
            if p.name.endswith(".hist.json"):
                continue
            try:
                r = SessionRecord.model_validate_json(p.read_text(encoding="utf-8"))
                records.append(r)
            except Exception:  # noqa: BLE001
                pass
        records.sort(key=lambda r: r.updated_at, reverse=True)
        return records[:limit]

    # ── update ────────────────────────────────────────────────────────────────

    def update(self, rec: SessionRecord) -> None:
        rec.updated_at = datetime.utcnow()
        self._write_meta(rec)

    def close(self, rec: SessionRecord) -> None:
        rec.status = "closed"
        self.update(rec)

    def interrupt(self, rec: SessionRecord) -> None:
        rec.status = "interrupted"
        self.update(rec)

    def purge_empty(self, task: str) -> int:
        """Delete sessions that have no title and no messages. Returns count removed."""
        removed = 0
        for rec in self.list_for_task(task, limit=200):
            if not rec.title and rec.turn_count == 0 and rec.message_count == 0:
                self.delete(task, rec.id)
                removed += 1
        return removed

    def delete(self, task: str, session_id: str) -> bool:
        meta = self._meta_path(task, session_id)
        hist = self._hist_path(task, session_id)
        if not meta.exists():
            return False
        meta.unlink(missing_ok=True)
        hist.unlink(missing_ok=True)
        return True

    # ── history ───────────────────────────────────────────────────────────────

    def save_history(self, rec: SessionRecord, messages: list[Message]) -> None:
        save_history(self._hist_path(rec.task, rec.id), messages)

    def load_history(self, task: str, session_id: str) -> list[Message]:
        return load_history(self._hist_path(task, session_id))

    # ── internal ──────────────────────────────────────────────────────────────

    def _write_meta(self, rec: SessionRecord) -> None:
        self._meta_path(rec.task, rec.id).write_text(
            rec.model_dump_json(indent=2), encoding="utf-8"
        )
