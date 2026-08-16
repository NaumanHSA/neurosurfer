"""Chroma backend.

Three things here are not obvious and were wrong before:

**Chroma's default space is L2, and this store's contract is cosine.** A
collection created without `hnsw:space` returns squared-L2 distance — 0.0 for
identical vectors and 2.0 for orthogonal ones — and the old code returned
`1.0 - distance` as though it were cosine. Orthogonal vectors therefore scored
**-1.0** instead of 0.0, and with unnormalised embeddings the number was
arbitrary. `similarity_threshold` was applied to that same wrong scale.

New collections are created as cosine. Existing ones are read for the space they
were actually built with and converted accordingly, because silently returning
different numbers for the same store would be its own bug — and a store built
before this fix is exactly the one whose scores need correcting.

**Ids round-trip.** `list_all_documents` used to mint a fresh `uuid4()` for every
row it returned, so the ids it handed back matched nothing and
`delete_documents(...)` on them was a no-op.

**The filter is translated, not passed through.** Chroma's `where` is close to
the canonical grammar but not identical: it has no `$not`, and it requires an
explicit `$and` where the grammar allows two predicates on one field.
"""

from __future__ import annotations

import os
from typing import Any

import chromadb

from ..observability.logging import get_logger
from .base import BaseVectorDB, Doc, StoreCapability
from .filters import UnsupportedFilter, normalize

log = get_logger("vectorstores.chroma")

#: distance → cosine similarity, per space. Chroma reports squared L2 for `l2`.
_SPACES = {
    "cosine": lambda d: 1.0 - d,
    "ip": lambda d: -d,
    "l2": None,  # handled separately: needs the vectors' norms, which we lack
}


class ChromaVectorStore(BaseVectorDB):
    """Persistent, disk-backed, and the default for `RAGAgent`."""

    capabilities = frozenset(
        {
            StoreCapability.RANGE_FILTERS,
            StoreCapability.BOOLEAN_FILTERS,
            StoreCapability.NATIVE_UPSERT,
            StoreCapability.PERSISTENT,
            # No NEGATION: Chroma's `where` has `$ne`/`$nin` but no `$not`, and
            # declaring half a capability is how a caller ends up with unfiltered
            # rows. `$not` is refused; see `_translate`.
        }
    )

    def __init__(
        self,
        collection_name: str,
        clear_collection: bool = False,
        persist_directory: str = "chroma_storage",
    ) -> None:
        os.makedirs(persist_directory, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection_name = collection_name
        self.collection = self.client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )
        if clear_collection:
            self.clear_collection()
        self._space = self._detect_space()
        log.debug(
            "Chroma collection %r ready (space=%s, %d docs)",
            collection_name,
            self._space,
            self.collection.count(),
        )

    def _detect_space(self) -> str:
        """The space this collection was *actually* built with.

        `get_or_create_collection` keeps an existing collection's configuration,
        so asking for cosine does not convert one that already exists. Reading it
        back is the difference between correct scores on an old store and
        confidently wrong ones.
        """
        meta = getattr(self.collection, "metadata", None) or {}
        space = str(meta.get("hnsw:space", "")).lower()
        if space in _SPACES:
            return space
        cfg = getattr(self.collection, "configuration_json", None) or {}
        inner = (cfg.get("hnsw") or cfg.get("spann") or {}) if isinstance(cfg, dict) else {}
        space = str((inner or {}).get("space", "")).lower()
        return space if space in _SPACES else "l2"

    # ── writing ──────────────────────────────────────────────────────────────

    def add_documents(self, docs: list[Doc]) -> None:
        prepared = self._prepare(docs)
        if not prepared:
            return
        self.collection.upsert(
            ids=[d.id for d in prepared],
            documents=[d.text for d in prepared],
            embeddings=[d.embedding for d in prepared],
            # Chroma rejects an empty metadata dict on some versions; a single
            # placeholder key is cheaper than branching per document.
            metadatas=[d.metadata or {"_": ""} for d in prepared],
        )

    def delete_documents(self, ids: list[str]) -> None:
        if ids:
            self.collection.delete(ids=list(ids))

    def clear_collection(self) -> None:
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name, metadata={"hnsw:space": "cosine"}
        )
        self._space = self._detect_space()

    def delete_collection(self) -> None:
        self.client.delete_collection(self.collection_name)
        self.collection = None

    # ── reading ──────────────────────────────────────────────────────────────

    def similarity_search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        metadata_filter: dict[str, Any] | None = None,
        similarity_threshold: float | None = None,
    ) -> list[tuple[Doc, float]]:
        flt = normalize(metadata_filter)
        self._check_supported(flt)

        args: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": max(1, top_k),
            "include": ["documents", "metadatas", "distances", "embeddings"],
        }
        where = _translate(flt)
        if where:
            args["where"] = where

        res = self.collection.query(**args)
        ids = (res.get("ids") or [[]])[0]
        texts = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        embs = (res.get("embeddings") or [[]])
        embs = embs[0] if len(embs) else []

        out: list[tuple[Doc, float]] = []
        for i, (rid, txt, meta, dist) in enumerate(
            zip(ids, texts, metas, dists, strict=False)
        ):
            emb = list(embs[i]) if i < len(embs) and embs[i] is not None else None
            score = self._to_similarity(dist, query_embedding, emb)
            if similarity_threshold is not None and score < similarity_threshold:
                continue
            out.append((Doc(id=rid, text=txt, metadata=_clean(meta), embedding=emb), score))
        return out[:top_k]

    def _to_similarity(
        self, distance: float, query: list[float], stored: list[float] | None
    ) -> float:
        """Chroma's distance → cosine similarity, for whichever space this is."""
        convert = _SPACES.get(self._space)
        if convert is not None:
            return float(convert(distance))
        # `l2`: the distance carries no angle, so recompute from the vectors —
        # which is why `embeddings` is in the query's `include`. An old
        # collection is the only place this runs.
        if stored:
            from .in_memory_store import _cosine

            return _cosine(query, stored)
        return 1.0 - float(distance)

    def list_all_documents(
        self, metadata_filter: dict[str, Any] | None = None
    ) -> list[Doc]:
        flt = normalize(metadata_filter)
        self._check_supported(flt)
        where = _translate(flt)
        res = self.collection.get(
            where=where or None, include=["documents", "metadatas", "embeddings"]
        )
        ids = res.get("ids") or []
        embs = res.get("embeddings")
        embs = list(embs) if embs is not None else []
        return [
            Doc(
                id=rid,
                text=txt,
                metadata=_clean(meta),
                embedding=list(embs[i]) if i < len(embs) and embs[i] is not None else None,
            )
            for i, (rid, txt, meta) in enumerate(
                zip(ids, res.get("documents") or [], res.get("metadatas") or [], strict=False)
            )
        ]

    def count(self) -> int:
        return self.collection.count()

    # ── which model wrote these vectors ──────────────────────────────────────
    #
    # Chroma has a durable place to put it — the collection's own metadata —
    # so the check survives the process that ingested, which is the case that
    # matters: querying an existing store months later with a different model.

    def embedding_identity(self) -> tuple[str | None, int | None]:
        meta = getattr(self.collection, "metadata", None) or {}
        dim = meta.get("neurosurfer:embedding_dim")
        return meta.get("neurosurfer:embedding_model"), int(dim) if dim else None

    def set_embedding_identity(self, model: str, dimensions: int | None) -> None:
        # `modify` replaces metadata wholesale **and refuses any payload carrying
        # `hnsw:space`** — "changing the distance function is not supported" —
        # even when the value is unchanged. Dropping the `hnsw:` keys is safe:
        # the space is stored in the collection's `configuration_json`, which is
        # what `_detect_space` reads second and what survives this call.
        meta = {
            k: v
            for k, v in (getattr(self.collection, "metadata", None) or {}).items()
            if not k.startswith("hnsw:")
        }
        meta["neurosurfer:embedding_model"] = model
        if dimensions:
            meta["neurosurfer:embedding_dim"] = int(dimensions)
        try:
            self.collection.modify(metadata=meta)
        except Exception as e:  # noqa: BLE001
            # Recording provenance must never fail an ingest that is otherwise
            # fine; the in-process default still catches same-run mistakes.
            log.debug("could not persist embedding identity: %s", e)
            super().set_embedding_identity(model, dimensions)


def _clean(meta: dict[str, Any] | None) -> dict[str, Any]:
    """Drop the placeholder `add_documents` writes for metadata-less docs."""
    m = dict(meta or {})
    if m.get("_") == "":
        m.pop("_", None)
    return m


def _translate(flt: dict[str, Any] | None) -> dict[str, Any] | None:
    """Canonical filter → Chroma `where`.

    Chroma wants exactly one operator per dict, so a field carrying two
    predicates and a filter naming two fields both become an explicit `$and`.
    """
    if not flt:
        return None

    clauses: list[dict[str, Any]] = []
    for key, value in flt.items():
        if key in ("$and", "$or"):
            inner = [c for c in (_translate(v) for v in value) if c]
            if inner:
                clauses.append({key: inner} if len(inner) > 1 else inner[0])
        elif key == "$not":
            # Normally unreachable: the declared capabilities refuse `$not`
            # before a query is formed, which is what gives every backend
            # lacking the same thing the same message. Kept as a guard for a
            # subclass that widens `capabilities` without widening this.
            raise UnsupportedFilter(
                "Chroma's query language has no `$not`. Express it as `$ne` or "
                "`$nin` on the field, or use a store that declares NEGATION."
            )
        else:
            clauses.extend({key: {op: v}} for op, v in value.items())

    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}
