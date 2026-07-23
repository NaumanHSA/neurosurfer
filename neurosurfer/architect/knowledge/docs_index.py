"""Lightweight retrieval over the project docs (Phase 3).

Splits every markdown file under ``docs/`` into heading-level sections and ranks
them for a query — BM25 when ``rank_bm25`` is installed (the ``search`` extra),
otherwise a plain term-frequency overlap score. Deliberately dependency-light: no
embeddings, no vector store, works offline and instantly, and is good enough for
"pull the authoritative paragraph about X" agent retrieval. Swap in the ``rag``
module later if semantic recall proves insufficient.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["DocSection", "DocsIndex"]

_HEADING_RE = re.compile(r"^(#{1,4})\s+(.*)$")
_TOKEN_RE = re.compile(r"[a-z0-9_]+")


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def default_docs_dir() -> Path:
    """The repo's docs/ directory (…/neurosurfer/docs relative to the package)."""
    return Path(__file__).resolve().parents[3] / "docs"


@dataclass
class DocSection:
    path: str          # docs-relative file path
    heading: str       # section heading (file title for the preamble)
    text: str          # section body (includes the heading line)
    _toks: list[str] = field(default_factory=list, repr=False)

    def snippet(self, limit: int = 700) -> str:
        t = self.text.strip()
        return t if len(t) <= limit else t[:limit] + "…"


class DocsIndex:
    """Heading-level markdown search over a docs directory."""

    def __init__(self, docs_dir: Path | None = None) -> None:
        self.docs_dir = Path(docs_dir) if docs_dir else default_docs_dir()
        self.sections: list[DocSection] = []
        self._bm25 = None
        self._load()

    # ── indexing ────────────────────────────────────────────────────────────
    def _load(self) -> None:
        if not self.docs_dir.is_dir():
            return
        for md in sorted(self.docs_dir.rglob("*.md")):
            rel = str(md.relative_to(self.docs_dir))
            try:
                content = md.read_text(encoding="utf-8")
            except OSError:
                continue
            self.sections.extend(self._split(rel, content))
        for sec in self.sections:
            sec._toks = _tokens(f"{sec.path} {sec.heading} {sec.text}")
        try:
            from rank_bm25 import BM25Okapi

            corpus = [s._toks for s in self.sections]
            if corpus:
                self._bm25 = BM25Okapi(corpus)
        except ImportError:
            self._bm25 = None  # plain scoring fallback below

    @staticmethod
    def _split(rel_path: str, content: str) -> list[DocSection]:
        sections: list[DocSection] = []
        current_heading = rel_path
        buf: list[str] = []

        def flush() -> None:
            body = "\n".join(buf).strip()
            if body:
                sections.append(DocSection(path=rel_path, heading=current_heading, text=body))

        for line in content.splitlines():
            m = _HEADING_RE.match(line)
            if m:
                flush()
                buf = [line]
                current_heading = m.group(2).strip()
            else:
                buf.append(line)
        flush()
        return sections

    # ── search ──────────────────────────────────────────────────────────────
    def search(self, query: str, k: int = 5) -> list[DocSection]:
        q = _tokens(query)
        if not q or not self.sections:
            return []
        if self._bm25 is not None:
            scores = self._bm25.get_scores(q)
        else:
            qset = set(q)
            scores = [
                sum(1.0 for t in s._toks if t in qset) / (len(s._toks) ** 0.5 or 1.0)
                for s in self.sections
            ]
        ranked = sorted(zip(scores, range(len(self.sections)), strict=True), reverse=True)
        return [self.sections[i] for score, i in ranked[:k] if score > 0]
