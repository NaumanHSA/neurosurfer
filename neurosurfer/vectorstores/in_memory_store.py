"""The reference implementation, and the one every test can use.

It was exported, documented in `docs/guides/rag.md` as *"handy for tests and
demos"*, and **could not be instantiated** — it never implemented
`delete_documents`, so the abstract base refused to construct it. Nothing in the
package or the tests ever built one, which is exactly why that survived.

It was also non-conformant in three quieter ways once you could: `add_documents`
appended rather than upserting, so re-ingesting a corpus duplicated every row,
and both `similarity_search` and `list_all_documents` accepted a
`metadata_filter` and ignored it.

Now it is the yardstick. It supports every :class:`StoreCapability`, its filter
semantics come from :func:`neurosurfer.vectorstores.filters.matches` — the same
function the conformance suite checks the others against — and it needs no
dependency at all.
"""

from __future__ import annotations

import math
from typing import Any

from .base import BaseVectorDB, Doc, StoreCapability


class InMemoryVectorStore(BaseVectorDB):
    """Ephemeral, dependency-free, and complete.

    ``dim`` is optional: given, it is enforced on write, which turns "these
    vectors came from a different embedding model" into an error at the point of
    ingestion rather than nonsense similarity scores at query time. Left out, the
    first document's width becomes the store's.
    """

    #: Everything except persistence — it is a dict.
    capabilities = StoreCapability.ALL - {StoreCapability.PERSISTENT}

    def __init__(self, dim: int | None = None) -> None:
        self.dim = dim
        self._docs: dict[str, Doc] = {}

    # ── writing ──────────────────────────────────────────────────────────────

    def add_documents(self, docs: list[Doc]) -> None:
        for d in self._prepare(docs):
            if self.dim is None:
                self.dim = len(d.embedding or [])
            elif len(d.embedding or []) != self.dim:
                raise ValueError(
                    f"embedding is {len(d.embedding or [])} wide, store is {self.dim}. "
                    f"A collection holds vectors from one embedding model; this "
                    f"looks like two."
                )
            self._docs[d.id] = d

    def delete_documents(self, ids: list[str]) -> None:
        for i in ids:
            self._docs.pop(i, None)

    def clear_collection(self) -> None:
        self._docs.clear()

    def delete_collection(self) -> None:
        self._docs.clear()

    # ── reading ──────────────────────────────────────────────────────────────

    def similarity_search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        metadata_filter: dict[str, Any] | None = None,
        similarity_threshold: float | None = None,
    ) -> list[tuple[Doc, float]]:
        candidates = self._filter_locally(list(self._docs.values()), metadata_filter)

        scored = [
            (d, _cosine(query_embedding, d.embedding or []))
            for d in candidates
        ]
        if similarity_threshold is not None:
            scored = [(d, s) for d, s in scored if s >= similarity_threshold]

        # Ties keep insertion order, so a result set is reproducible run to run.
        scored.sort(key=lambda ds: -ds[1])
        return scored[:top_k]

    def list_all_documents(
        self, metadata_filter: dict[str, Any] | None = None
    ) -> list[Doc]:
        return self._filter_locally(list(self._docs.values()), metadata_filter)

    def count(self) -> int:
        return len(self._docs)


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity, with a zero vector scoring 0 rather than dividing by it."""
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)
