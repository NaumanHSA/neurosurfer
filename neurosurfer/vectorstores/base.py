"""The vector-store contract, and what a backend may decline to implement.

`BaseVectorDB` had one working implementation, and an interface with one
implementation is a description of that implementation rather than a contract.
It showed: `delete_documents` was declared taking `list[Doc]` and implemented
taking `list[str]`, `metadata_filter` meant something different in each backend,
and `upsert` was reached for with `hasattr` rather than promised.

Three things make it a contract instead:

* **A declared filter grammar** — :mod:`neurosurfer.vectorstores.filters`, so a
  backend translates from a canonical form rather than re-deriving shorthand.
* **Capability flags** — :class:`StoreCapability`, following the same rule as
  ``mcp/sources``: a caller *asks* what a backend can do rather than inferring it
  from an empty result. A backend that cannot express a filter says so and
  raises, instead of quietly returning rows it did not filter.
* **A conformance suite** — ``tests/vectorstores/conformance.py``. "Implements
  ``BaseVectorDB``" means "passes that suite"; adding a backend is writing one
  class and one three-line test module.

**Similarity is cosine, and higher is better.** Every backend returns scores on
that scale whatever its native distance metric is, because a caller comparing
scores across two stores — or applying `similarity_threshold` — has no way to
know what it is holding otherwise.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from .filters import UnsupportedFilter, matches, normalize

__all__ = ["BaseVectorDB", "Doc", "StoreCapability", "UnsupportedFilter"]


# --------------------------
# Data structures
# --------------------------


@dataclass
class Doc:
    """One stored chunk: its text, its vector, and whatever else is known about it.

    ``id`` is the identity the store upserts and deletes on. Leave it empty and
    :meth:`BaseVectorDB.stable_id` derives one from the content, so re-ingesting
    an unchanged document overwrites its row rather than duplicating it.
    """

    id: str
    text: str
    embedding: list[float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class StoreCapability:
    """What a backend can do, so a caller can ask instead of finding out.

    A missing capability is not a failure and not a bug — it decides whether a
    query is worth forming. Offering a range filter against a store that ignores
    ranges is worse than not offering it, because the results look right.
    """

    RANGE_FILTERS = "range_filters"
    """`$gt` / `$gte` / `$lt` / `$lte` are honoured."""

    BOOLEAN_FILTERS = "boolean_filters"
    """`$and` / `$or` compose filters."""

    NEGATION = "negation"
    """`$not` and `$nin` are honoured."""

    NATIVE_UPSERT = "native_upsert"
    """The store replaces by id itself, rather than delete-then-add."""

    PERSISTENT = "persistent"
    """Data survives the process."""

    #: Everything the in-memory reference implementation supports. A backend that
    #: manages less is declaring a real limitation, not lagging behind.
    ALL = frozenset(
        {RANGE_FILTERS, BOOLEAN_FILTERS, NEGATION, NATIVE_UPSERT, PERSISTENT}
    )


class BaseVectorDB(ABC):
    """A store of embedded documents that can be searched by vector similarity."""

    #: Subset of :class:`StoreCapability`. Declared per class; a backend whose
    #: support depends on how it was constructed may override it per instance.
    capabilities: frozenset[str] = frozenset()

    # ── writing ──────────────────────────────────────────────────────────────

    @abstractmethod
    def add_documents(self, docs: list[Doc]) -> None:
        """Insert or replace *docs*, keyed on ``Doc.id``.

        **This is an upsert.** Adding a document whose id is already present
        replaces it; it does not duplicate it and does not raise. That is what
        makes re-ingesting a corpus idempotent, which every ingestion path here
        assumes.

        A doc with no ``id`` is assigned :meth:`stable_id`, derived from content,
        so the same chunk lands on the same row across runs.

        A doc with no ``embedding`` is an error — the store does not embed.
        """

    @abstractmethod
    def delete_documents(self, ids: list[str]) -> None:
        """Delete by id. Unknown ids are ignored rather than raising.

        **Ids, not documents.** Every backend's delete API takes ids, and
        requiring a whole :class:`Doc` to remove one means reading it back first
        just to throw it away. :meth:`delete_docs` is the convenience for when a
        caller does happen to hold the documents.
        """

    def delete_docs(self, docs: list[Doc]) -> None:
        """Delete the given documents, by their ids."""
        self.delete_documents([d.id for d in docs if d.id])

    @abstractmethod
    def clear_collection(self) -> None:
        """Remove every document, leaving the collection usable."""

    @abstractmethod
    def delete_collection(self) -> None:
        """Drop the collection entirely. The store is unusable afterwards."""

    # ── reading ──────────────────────────────────────────────────────────────

    @abstractmethod
    def similarity_search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        metadata_filter: dict[str, Any] | None = None,
        similarity_threshold: float | None = None,
    ) -> list[tuple[Doc, float]]:
        """The *top_k* documents closest to *query_embedding*, best first.

        Scores are **cosine similarity in [-1, 1], higher is better**, whatever
        the backend's native metric. `similarity_threshold` drops anything below
        it, and is applied on that same scale.

        `metadata_filter` follows :mod:`neurosurfer.vectorstores.filters`. A
        backend that cannot express part of it raises
        :class:`~neurosurfer.vectorstores.filters.UnsupportedFilter` rather than
        returning rows it did not filter.
        """

    @abstractmethod
    def list_all_documents(
        self, metadata_filter: dict[str, Any] | None = None
    ) -> list[Doc]:
        """Every document, optionally filtered. Ids round-trip: what comes back
        here can be handed to :meth:`delete_documents`."""

    @abstractmethod
    def count(self) -> int:
        """How many documents are stored."""

    # ── which model wrote these vectors ──────────────────────────────────────

    def embedding_identity(self) -> tuple[str | None, int | None]:
        """The ``(model, dimensions)`` this collection was embedded with.

        ``(None, None)`` when unknown — an empty collection, or one written
        before this was recorded.
        """
        return getattr(self, "_embedding_model", None), getattr(self, "_embedding_dim", None)

    def set_embedding_identity(self, model: str, dimensions: int | None) -> None:
        """Record which model wrote this collection's vectors.

        Backends with somewhere durable to put it override this; the default
        keeps it for the life of the object, which is still enough to catch the
        mistake inside one process.
        """
        self._embedding_model = model
        self._embedding_dim = dimensions

    def check_embedding_identity(self, model: str, dimensions: int | None) -> None:
        """Refuse a query embedded by a different model than the collection holds.

        Nothing recorded this before, so pointing a differently-embedded query at
        an existing store returned confident nonsense: the vectors are the right
        width and the arithmetic succeeds, the neighbours are just meaningless.
        An empty or unlabelled collection adopts the identity instead of
        refusing, so this never blocks a first ingest or an upgrade.
        """
        known_model, known_dim = self.embedding_identity()
        if known_model is None or self.count() == 0:
            self.set_embedding_identity(model, dimensions)
            return
        if known_model != model:
            raise ValueError(
                f"This collection was embedded with {known_model!r} and the query "
                f"was embedded with {model!r}. Their vectors are not comparable — "
                f"re-ingest the collection with one model, or point at another."
            )
        if dimensions and known_dim and dimensions != known_dim:
            raise ValueError(
                f"This collection holds {known_dim}-wide vectors and {model!r} "
                f"produces {dimensions}-wide ones."
            )

    # ── helpers for implementations ──────────────────────────────────────────

    @staticmethod
    def stable_id(doc: Doc) -> str:
        """A deterministic id for *doc*, from its content.

        The ingestor sets ``content_hash`` when it has one; otherwise the text is
        hashed. Suffixed with the chunk index so two chunks of one document do
        not collide.
        """
        h = doc.metadata.get("content_hash") or hashlib.sha256(
            (doc.text or "").encode("utf-8")
        ).hexdigest()
        return f"{h[:32]}:{doc.metadata.get('chunk_idx', 0)}"

    #: Kept as the old private name — `chroma.py` and any downstream subclass
    #: called it. It is the same function.
    _stable_id = stable_id

    def _prepare(self, docs: list[Doc]) -> list[Doc]:
        """Validate and id-fill *docs* on the way into a backend."""
        out: list[Doc] = []
        for d in docs:
            if d.embedding is None:
                raise ValueError(
                    f"Doc {d.id or '(no id)'!r} has no embedding; a vector store "
                    f"stores vectors, it does not make them. Embed first."
                )
            out.append(
                d if d.id else Doc(self.stable_id(d), d.text, d.embedding, d.metadata)
            )
        return out

    def _check_supported(self, flt: dict[str, Any] | None) -> None:
        """Raise `UnsupportedFilter` if *flt* needs a capability this store lacks."""
        if not flt:
            return
        needed = _capabilities_required(flt)
        missing = needed - set(self.capabilities)
        if missing:
            raise UnsupportedFilter(
                f"{type(self).__name__} cannot express this filter: it needs "
                f"{sorted(missing)}. Declared capabilities: "
                f"{sorted(self.capabilities) or 'none'}."
            )

    def _filter_locally(
        self, docs: list[Doc], metadata_filter: dict[str, Any] | None
    ) -> list[Doc]:
        """Apply a canonical filter in Python — for backends with no query language."""
        flt = normalize(metadata_filter)
        self._check_supported(flt)
        return [d for d in docs if matches(d.metadata, flt)]


_RANGE = frozenset({"$gt", "$gte", "$lt", "$lte"})
_NEGATING = frozenset({"$ne", "$nin"})


def _capabilities_required(flt: dict[str, Any]) -> set[str]:
    """Which :class:`StoreCapability` flags a canonical filter depends on."""
    needed: set[str] = set()
    for key, value in flt.items():
        if key in ("$and", "$or"):
            needed.add(StoreCapability.BOOLEAN_FILTERS)
            for clause in value:
                needed |= _capabilities_required(clause)
        elif key == "$not":
            needed.add(StoreCapability.NEGATION)
            needed |= _capabilities_required(value)
        else:
            ops = set(value)
            if ops & _RANGE:
                needed.add(StoreCapability.RANGE_FILTERS)
            if ops & _NEGATING:
                needed.add(StoreCapability.NEGATION)
    return needed
