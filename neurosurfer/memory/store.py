"""Memory store — JSONL-backed, global + per-agent.

Mirrors :class:`TaskRegistry`'s shape: a directory of plain files, one line per
entry. ``global.jsonl`` holds user-wide facts; ``agents/<agent>.jsonl`` holds one
agent's domain facts. ``all_in_scope(agent)`` returns the union an agent recalls.

Writes are append-or-rewrite of a single small file; we never need a DB. ``add``
does lexical dedup/supersede so the same fact recorded twice doesn't pile up.
"""

from __future__ import annotations

import re
from pathlib import Path

from ..observability.logging import get_logger
from .models import MemoryEntry

log = get_logger("memory.store")

_WORD_RE = re.compile(r"[a-z0-9]+")


def _norm(text: str) -> frozenset[str]:
    return frozenset(_WORD_RE.findall(text.lower()))


def _similar(a: str, b: str, threshold: float = 0.82) -> bool:
    """Jaccard token overlap — cheap near-duplicate detection (no embeddings)."""
    ta, tb = _norm(a), _norm(b)
    if not ta or not tb:
        return a.strip().lower() == b.strip().lower()
    inter = len(ta & tb)
    union = len(ta | tb)
    return union > 0 and inter / union >= threshold


class MemoryStore:
    """File-backed memory keyed by ``global`` + per-agent scopes."""

    def __init__(self, root: Path) -> None:
        self._root = root
        self._root.mkdir(parents=True, exist_ok=True)
        (self._root / "agents").mkdir(parents=True, exist_ok=True)

    # ── paths ─────────────────────────────────────────────────────────────────
    def _path(self, scope: str, scope_key: str) -> Path:
        if scope == "global":
            return self._root / "global.jsonl"
        safe = re.sub(r"[^A-Za-z0-9._-]", "_", scope_key or "_unknown")
        return self._root / "agents" / f"{safe}.jsonl"

    def _path_for(self, entry: MemoryEntry) -> Path:
        return self._path(entry.scope, entry.scope_key)

    # ── read ──────────────────────────────────────────────────────────────────
    def _read(self, path: Path) -> list[MemoryEntry]:
        if not path.exists():
            return []
        out: list[MemoryEntry] = []
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line:
                continue
            try:
                out.append(MemoryEntry.model_validate_json(line))
            except Exception as e:  # noqa: BLE001 - a bad line must not break recall
                log.warning("skipping corrupt memory line in %s: %s", path.name, e)
        return out

    def _write(self, path: Path, entries: list[MemoryEntry]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        body = "\n".join(e.model_dump_json() for e in entries)
        path.write_text(body + "\n" if body else "", encoding="utf-8")

    def list_scope(self, scope: str, scope_key: str = "") -> list[MemoryEntry]:
        return self._read(self._path(scope, scope_key))

    def all_in_scope(self, agent: str | None) -> list[MemoryEntry]:
        """Everything an ``agent`` recalls: global ∪ that agent's entries."""
        entries = self.list_scope("global")
        if agent:
            entries = entries + self.list_scope("agent", agent)
        return entries

    def list_all(self) -> list[MemoryEntry]:
        out = self.list_scope("global")
        agents_dir = self._root / "agents"
        if agents_dir.is_dir():
            for p in sorted(agents_dir.glob("*.jsonl")):
                out.extend(self._read(p))
        return out

    # ── write ─────────────────────────────────────────────────────────────────
    def add(self, entry: MemoryEntry) -> MemoryEntry:
        """Add a fact, deduping/superseding a near-identical existing one.

        If a similar entry already exists in the same file we bump its salience
        and refresh its text/provenance instead of appending a duplicate.
        """
        path = self._path_for(entry)
        entries = self._read(path)
        for i, existing in enumerate(entries):
            if existing.id == entry.id or _similar(existing.text, entry.text):
                entry.salience = max(entry.salience, existing.salience + 0.5)
                entry.uses = existing.uses
                entry.supersedes = sorted({*existing.supersedes, *entry.supersedes})
                entries[i] = entry
                self._write(path, entries)
                return entry
        entries.append(entry)
        self._write(path, entries)
        return entry

    def forget(self, entry_id: str) -> bool:
        """Remove an entry by id from whichever scope holds it."""
        for path in self._all_paths():
            entries = self._read(path)
            kept = [e for e in entries if e.id != entry_id]
            if len(kept) != len(entries):
                self._write(path, kept)
                return True
        return False

    def record_use(self, entry_ids: list[str]) -> None:
        """Increment ``uses`` for the given entries (retrieval feedback)."""
        if not entry_ids:
            return
        wanted = set(entry_ids)
        for path in self._all_paths():
            entries = self._read(path)
            changed = False
            for e in entries:
                if e.id in wanted:
                    e.uses += 1
                    changed = True
            if changed:
                self._write(path, entries)

    def promote(self, entry_id: str, salience: float) -> bool:
        """Set an entry's salience (e.g. user-curated importance)."""
        for path in self._all_paths():
            entries = self._read(path)
            for e in entries:
                if e.id == entry_id:
                    e.salience = salience
                    self._write(path, entries)
                    return True
        return False

    # ── helpers ───────────────────────────────────────────────────────────────
    def _all_paths(self) -> list[Path]:
        paths = [self._root / "global.jsonl"]
        agents_dir = self._root / "agents"
        if agents_dir.is_dir():
            paths.extend(sorted(agents_dir.glob("*.jsonl")))
        return [p for p in paths if p.exists()]
