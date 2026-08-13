"""Re-ingesting only what changed.

`RAGIngestor` deduplicates chunks by content hash within a run, so it never
writes the same chunk twice. What it has no notion of is a *previous* run: point
it at a directory of a thousand files with one edited and it reads, chunks and
embeds all thousand again. Embedding is the expensive step, and for a corpus
under active editing almost all of it is repeated work.

This is the missing half — a manifest of what each source looked like last time,
so a run can ask three questions before doing anything:

* which sources are **new**,
* which have **changed**,
* which have **gone**, and whose chunks should therefore be deleted.

Deliberately a plain JSON file. It records hashes, not content; it is safe to
delete (the next run simply re-ingests everything); and it needs no service.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["IngestManifest", "SourceDelta"]


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SourceDelta:
    """What changed since the last run."""

    new: list[str] = field(default_factory=list)
    changed: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)

    @property
    def to_ingest(self) -> list[str]:
        """Sources that need reading, chunking and embedding."""
        return [*self.new, *self.changed]

    @property
    def is_empty(self) -> bool:
        return not (self.new or self.changed or self.removed)

    def __str__(self) -> str:
        return (
            f"{len(self.new)} new, {len(self.changed)} changed, "
            f"{len(self.unchanged)} unchanged, {len(self.removed)} removed"
        )


@dataclass
class IngestManifest:
    """What each source hashed to last time, and which chunk ids it produced.

    The chunk ids matter as much as the hashes: when a source changes, its *old*
    chunks have to be deleted, and without a record of which ones they were the
    only options are leaving stale text in the index or clearing the whole
    collection. Both are what this exists to avoid.
    """

    path: Path
    sources: dict[str, str] = field(default_factory=dict)
    chunks: dict[str, list[str]] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path) -> IngestManifest:
        """Read the manifest, or start an empty one.

        Never raises on a damaged file: a corrupt manifest should cost a full
        re-ingest, not a failed run.
        """
        p = Path(path)
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            return cls(path=p, sources=dict(data.get("sources") or {}),
                       chunks={k: list(v) for k, v in (data.get("chunks") or {}).items()})
        except Exception:  # noqa: BLE001
            return cls(path=p)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps({"sources": self.sources, "chunks": self.chunks}, indent=1),
            encoding="utf-8",
        )

    def diff(self, current: dict[str, str]) -> SourceDelta:
        """Compare `{source_id: text}` against what was recorded."""
        new, changed, unchanged = [], [], []
        for source_id, text in current.items():
            digest = _hash(text)
            previous = self.sources.get(source_id)
            if previous is None:
                new.append(source_id)
            elif previous != digest:
                changed.append(source_id)
            else:
                unchanged.append(source_id)
        removed = [s for s in self.sources if s not in current]
        return SourceDelta(
            new=sorted(new),
            changed=sorted(changed),
            unchanged=sorted(unchanged),
            removed=sorted(removed),
        )

    def stale_chunk_ids(self, delta: SourceDelta) -> list[str]:
        """Chunk ids to delete: everything belonging to a changed or gone source."""
        out: list[str] = []
        for source_id in [*delta.changed, *delta.removed]:
            out.extend(self.chunks.get(source_id, []))
        return out

    def record(self, source_id: str, text: str, chunk_ids: list[str]) -> None:
        self.sources[source_id] = _hash(text)
        self.chunks[source_id] = list(chunk_ids)

    def forget(self, source_ids: list[str]) -> None:
        for source_id in source_ids:
            self.sources.pop(source_id, None)
            self.chunks.pop(source_id, None)
