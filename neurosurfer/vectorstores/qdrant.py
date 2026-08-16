"""Qdrant backend — the second implementation, which is what makes the first a contract.

Chosen over the alternatives because it has the strongest filtering story, so it
pushes hardest on the grammar Phase 1 declared: it is the backend that can
express every capability, including the `$not` Chroma has to refuse.

It also runs **in-process** with `QdrantClient(":memory:")` or against a path,
so the conformance suite exercises a real translation of the filter grammar into
a real query language without a server anywhere.

Two things are specific to Qdrant and worth knowing:

**Point ids are integers or UUIDs**, and ours are arbitrary strings like
`install:0`. Each id is mapped through `uuid5` — deterministic, so the same
document id always lands on the same point and upsert stays upsert — with the
original kept in the payload as the thing round-tripped back to callers.

**`must_not` matches a document that lacks the field entirely**, which disagrees
with the reference semantics in `filters.matches`: there, a field the metadata
does not carry never matches, including under `$ne`. So a negated condition
carries an explicit "and the field exists" alongside it.
"""

from __future__ import annotations

import uuid
from typing import Any

from qdrant_client import QdrantClient, models

from .base import BaseVectorDB, Doc, StoreCapability
from .filters import normalize

#: Namespace for deriving a point UUID from a document id. Fixed forever: change
#: it and every existing point in every collection becomes unreachable.
_ID_NAMESPACE = uuid.UUID("6f9619ff-8b86-d011-b42d-00c04fc964ff")

_TEXT_KEY = "_ns_text"
_ID_KEY = "_ns_id"


def _point_id(doc_id: str) -> str:
    return str(uuid.uuid5(_ID_NAMESPACE, doc_id))


class QdrantVectorStore(BaseVectorDB):
    """Qdrant, in-memory or on disk or over the network.

    ``location`` follows the client's own convention: ``":memory:"`` for an
    in-process store, a filesystem path for an embedded one, or a URL for a
    server. ``dim`` is required because Qdrant sizes a collection at creation.
    """

    #: Everything. Qdrant is the backend that can express the whole grammar.
    capabilities = StoreCapability.ALL

    def __init__(
        self,
        collection_name: str,
        dim: int,
        *,
        location: str = ":memory:",
        clear_collection: bool = False,
        client: QdrantClient | None = None,
    ) -> None:
        self.collection_name = collection_name
        self.dim = dim
        if client is not None:
            self.client = client
        elif location.startswith(("http://", "https://")):
            self.client = QdrantClient(url=location)
        elif location == ":memory:":
            self.client = QdrantClient(":memory:")
        else:
            self.client = QdrantClient(path=location)

        if clear_collection and self._exists():
            self.client.delete_collection(collection_name)
        if not self._exists():
            self.client.create_collection(
                collection_name,
                vectors_config=models.VectorParams(
                    size=dim, distance=models.Distance.COSINE
                ),
            )

    def _exists(self) -> bool:
        try:
            return self.client.collection_exists(self.collection_name)
        except Exception:  # noqa: BLE001 — older clients have no such call
            return self.collection_name in {
                c.name for c in self.client.get_collections().collections
            }

    # ── writing ──────────────────────────────────────────────────────────────

    def add_documents(self, docs: list[Doc]) -> None:
        prepared = self._prepare(docs)
        if not prepared:
            return
        for d in prepared:
            if len(d.embedding or []) != self.dim:
                raise ValueError(
                    f"embedding is {len(d.embedding or [])} wide, collection is "
                    f"{self.dim}. A collection holds vectors from one embedding "
                    f"model; this looks like two."
                )
        self.client.upsert(
            self.collection_name,
            points=[
                models.PointStruct(
                    id=_point_id(d.id),
                    vector=list(d.embedding or []),
                    payload={**(d.metadata or {}), _TEXT_KEY: d.text, _ID_KEY: d.id},
                )
                for d in prepared
            ],
        )

    def delete_documents(self, ids: list[str]) -> None:
        if ids:
            self.client.delete(
                self.collection_name,
                points_selector=models.PointIdsList(points=[_point_id(i) for i in ids]),
            )

    def clear_collection(self) -> None:
        self.client.delete_collection(self.collection_name)
        self.client.create_collection(
            self.collection_name,
            vectors_config=models.VectorParams(
                size=self.dim, distance=models.Distance.COSINE
            ),
        )

    def delete_collection(self) -> None:
        self.client.delete_collection(self.collection_name)

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

        hits = self.client.query_points(
            self.collection_name,
            query=list(query_embedding),
            limit=max(1, top_k),
            query_filter=_translate(flt),
            score_threshold=similarity_threshold,
            with_payload=True,
            with_vectors=True,
        ).points
        return [(_to_doc(h.payload, h.vector), float(h.score)) for h in hits]

    def list_all_documents(
        self, metadata_filter: dict[str, Any] | None = None
    ) -> list[Doc]:
        flt = normalize(metadata_filter)
        self._check_supported(flt)

        out: list[Doc] = []
        offset = None
        while True:
            points, offset = self.client.scroll(
                self.collection_name,
                scroll_filter=_translate(flt),
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=True,
            )
            out.extend(_to_doc(p.payload, p.vector) for p in points)
            if offset is None:
                return out

    def count(self) -> int:
        return int(self.client.count(self.collection_name, exact=True).count)

    # ── which model wrote these vectors ──────────────────────────────────────
    #
    # Qdrant has no collection-level metadata to hang this on, so it lives in the
    # in-process default from `BaseVectorDB`. Same-run mistakes are caught; a
    # cross-run one is not, which is a real limitation and better stated than
    # papered over with a magic point.


def _to_doc(payload: dict[str, Any] | None, vector: Any) -> Doc:
    meta = dict(payload or {})
    text = meta.pop(_TEXT_KEY, "")
    doc_id = meta.pop(_ID_KEY, "")
    return Doc(
        id=doc_id,
        text=text,
        metadata=meta,
        embedding=list(vector) if isinstance(vector, (list, tuple)) else None,
    )


def _exists_condition(field: str) -> models.Filter:
    """"…and the field is present", which `must_not` alone does not imply."""
    return models.Filter(
        must_not=[models.IsEmptyCondition(is_empty=models.PayloadField(key=field))]
    )


def _translate(flt: dict[str, Any] | None) -> models.Filter | None:
    """Canonical filter → a Qdrant `Filter`."""
    if not flt:
        return None

    must: list[Any] = []
    for key, value in flt.items():
        if key == "$and":
            must.extend(c for c in (_translate(v) for v in value) if c is not None)
        elif key == "$or":
            clauses = [c for c in (_translate(v) for v in value) if c is not None]
            if clauses:
                must.append(models.Filter(should=clauses))
        elif key == "$not":
            inner = _translate(value)
            if inner is not None:
                must.append(models.Filter(must_not=[inner]))
        else:
            must.extend(_field_conditions(key, value))

    return models.Filter(must=must) if must else None


def _field_conditions(field: str, predicate: dict[str, Any]) -> list[Any]:
    """One field's predicates → Qdrant conditions, ANDed by the caller."""
    out: list[Any] = []
    rng: dict[str, Any] = {}

    for op, want in predicate.items():
        if op == "$eq":
            out.append(
                models.FieldCondition(key=field, match=models.MatchValue(value=want))
            )
        elif op == "$in":
            out.append(
                models.FieldCondition(key=field, match=models.MatchAny(any=list(want)))
            )
        elif op == "$ne":
            # Present, and not equal — `must_not` alone would also match a
            # document that has no such field at all.
            out.append(_exists_condition(field))
            out.append(
                models.Filter(
                    must_not=[
                        models.FieldCondition(
                            key=field, match=models.MatchValue(value=want)
                        )
                    ]
                )
            )
        elif op == "$nin":
            out.append(_exists_condition(field))
            out.append(
                models.Filter(
                    must_not=[
                        models.FieldCondition(
                            key=field, match=models.MatchAny(any=list(want))
                        )
                    ]
                )
            )
        elif op in ("$gt", "$gte", "$lt", "$lte"):
            rng[op.lstrip("$")] = want

    if rng:
        out.append(models.FieldCondition(key=field, range=models.Range(**rng)))
    return out
