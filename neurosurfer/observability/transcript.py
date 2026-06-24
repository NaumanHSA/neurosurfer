"""Per-run event transcript.

Every run writes one JSONL file under ``.neurosurfer/transcripts/<run_id>.jsonl``.
Each line is a JSON object with ``ts`` (ISO-8601), ``type``, and payload fields.
The transcript is append-only during a run; it is never compacted.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _now() -> str:
    return datetime.now(tz=UTC).isoformat()


class EventTranscript:
    """Append-only JSONL event log for one agent run."""

    def __init__(self, run_id: str, transcripts_dir: Path) -> None:
        self.run_id = run_id
        transcripts_dir.mkdir(parents=True, exist_ok=True)
        self._path = transcripts_dir / f"{run_id}.jsonl"
        self._fh = self._path.open("a", encoding="utf-8")

    # ── write ─────────────────────────────────────────────────────────────────

    def record(self, event_type: str, **payload: Any) -> None:
        entry = {"ts": _now(), "type": event_type, **payload}
        self._fh.write(json.dumps(entry, default=str) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()

    def __enter__(self) -> EventTranscript:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ── read (for tests / inspection) ────────────────────────────────────────

    @property
    def path(self) -> Path:
        return self._path

    def read_all(self) -> list[dict[str, Any]]:
        if not self._path.exists():
            return []
        lines = self._path.read_text(encoding="utf-8").splitlines()
        return [json.loads(line) for line in lines if line.strip()]


def new_run_id() -> str:
    return uuid.uuid4().hex[:12]
