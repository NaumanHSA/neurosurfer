"""One suite every vector store must pass. Adding a backend is a three-line module.

`BaseVectorDB` had one working implementation for its whole life, and an
interface with one implementation is a description of that implementation. This
is what makes it a contract: subclass :class:`VectorStoreConformance`, implement
``make_store``, and the backend is held to the same behaviour as every other —
including the reference in-memory one, whose semantics come from
``filters.matches``.

**Capability-gated, not lowest-common-denominator.** A backend that declares
`RANGE_FILTERS` is tested on ranges; one that does not is tested that it *refuses*
them with `UnsupportedFilter` rather than quietly returning unfiltered rows.
Skipping a test because a backend is weaker would let a store return wrong rows
and stay green, which is the failure this suite exists to catch.
"""

from __future__ import annotations

import pytest

from neurosurfer.vectorstores.base import BaseVectorDB, Doc, StoreCapability
from neurosurfer.vectorstores.filters import UnsupportedFilter

# Two-dimensional and hand-picked so every expected ordering is obvious by eye:
# `east` and `north` are orthogonal, `northeast` sits between them.
EAST = [1.0, 0.0]
NORTH = [0.0, 1.0]
NORTHEAST = [0.7071, 0.7071]


def _doc(id_: str, text: str, emb: list[float], **meta) -> Doc:
    return Doc(id=id_, text=text, embedding=emb, metadata=meta)


CORPUS = [
    _doc("a", "alpha", EAST, lang="py", kind="src", score=10),
    _doc("b", "beta", NORTH, lang="rs", kind="src", score=20),
    _doc("c", "gamma", NORTHEAST, lang="py", kind="test", score=30),
]


class VectorStoreConformance:
    """Subclass and implement :meth:`make_store`."""

    def make_store(self, request) -> BaseVectorDB:  # pragma: no cover - overridden
        raise NotImplementedError

    # ── fixtures ─────────────────────────────────────────────────────────────

    @pytest.fixture
    def store(self, request) -> BaseVectorDB:
        return self.make_store(request)

    @pytest.fixture
    def loaded(self, store) -> BaseVectorDB:
        store.add_documents(list(CORPUS))
        return store

    def _supports(self, store, cap: str) -> bool:
        return cap in store.capabilities

    # ── the contract: writing ────────────────────────────────────────────────

    def test_documents_come_back_after_adding(self, loaded):
        assert loaded.count() == 3
        assert {d.id for d in loaded.list_all_documents()} == {"a", "b", "c"}

    def test_adding_the_same_id_replaces_rather_than_duplicates(self, loaded):
        """Upsert is the contract — re-ingesting a corpus must be idempotent."""
        loaded.add_documents([_doc("a", "alpha rewritten", EAST, lang="py", kind="src", score=10)])

        assert loaded.count() == 3
        texts = {d.id: d.text for d in loaded.list_all_documents()}
        assert texts["a"] == "alpha rewritten"

    def test_a_document_with_no_id_gets_a_stable_one(self, store):
        """Same content, same row — twice, with no id supplied either time."""
        store.add_documents([Doc(id="", text="repeatable", embedding=EAST)])
        store.add_documents([Doc(id="", text="repeatable", embedding=EAST)])

        assert store.count() == 1

    def test_a_document_without_an_embedding_is_refused(self, store):
        """A vector store stores vectors; it does not make them."""
        with pytest.raises(ValueError, match="embedding"):
            store.add_documents([Doc(id="x", text="no vector", embedding=None)])

    # ── the contract: deleting ───────────────────────────────────────────────

    def test_delete_by_id(self, loaded):
        loaded.delete_documents(["a"])

        assert loaded.count() == 2
        assert {d.id for d in loaded.list_all_documents()} == {"b", "c"}

    def test_deleting_an_unknown_id_is_not_an_error(self, loaded):
        loaded.delete_documents(["nope"])
        assert loaded.count() == 3

    def test_ids_round_trip_from_listing_into_deleting(self, loaded):
        """`list_all_documents` used to mint fresh uuids, so what it returned
        could not be deleted. The round trip is the point of an id."""
        docs = loaded.list_all_documents()
        loaded.delete_docs(docs)

        assert loaded.count() == 0

    def test_clearing_leaves_the_collection_usable(self, loaded):
        loaded.clear_collection()
        assert loaded.count() == 0

        loaded.add_documents([_doc("z", "after", EAST)])
        assert loaded.count() == 1

    # ── the contract: searching ──────────────────────────────────────────────

    def test_search_ranks_by_similarity(self, loaded):
        hits = loaded.similarity_search(EAST, top_k=3)

        assert [d.id for d, _ in hits] == ["a", "c", "b"]

    def test_scores_are_cosine_similarity(self, loaded):
        """Every backend reports on one scale whatever its native metric is —
        otherwise a threshold means something different per store."""
        by_id = {d.id: s for d, s in loaded.similarity_search(EAST, top_k=3)}

        assert by_id["a"] == pytest.approx(1.0, abs=1e-3)   # identical
        assert by_id["c"] == pytest.approx(0.7071, abs=1e-3)  # 45 degrees
        assert by_id["b"] == pytest.approx(0.0, abs=1e-3)   # orthogonal

    def test_top_k_limits_the_result(self, loaded):
        assert len(loaded.similarity_search(EAST, top_k=2)) == 2

    def test_similarity_threshold_drops_the_weak(self, loaded):
        hits = loaded.similarity_search(EAST, top_k=10, similarity_threshold=0.5)

        assert [d.id for d, _ in hits] == ["a", "c"]

    def test_searching_an_empty_store_is_empty_not_an_error(self, store):
        assert store.similarity_search(EAST, top_k=5) == []

    # ── the contract: filtering ──────────────────────────────────────────────

    def test_filter_by_equality(self, loaded):
        hits = loaded.similarity_search(EAST, top_k=10, metadata_filter={"lang": "py"})

        assert {d.id for d, _ in hits} == {"a", "c"}

    def test_filter_shorthand_list_means_membership(self, loaded):
        hits = loaded.similarity_search(
            EAST, top_k=10, metadata_filter={"lang": ["rs", "py"]}
        )

        assert {d.id for d, _ in hits} == {"a", "b", "c"}

    def test_filter_applies_to_listing_too(self, loaded):
        docs = loaded.list_all_documents({"kind": "test"})

        assert [d.id for d in docs] == ["c"]

    def test_two_fields_are_anded(self, loaded):
        hits = loaded.similarity_search(
            EAST, top_k=10, metadata_filter={"lang": "py", "kind": "src"}
        )

        assert {d.id for d, _ in hits} == {"a"}

    def test_a_filter_matching_nothing_returns_nothing(self, loaded):
        assert loaded.similarity_search(EAST, top_k=10, metadata_filter={"lang": "go"}) == []

    def test_range_filters(self, loaded):
        want = {"score": {"$gte": 20}}
        if not self._supports(loaded, StoreCapability.RANGE_FILTERS):
            with pytest.raises(UnsupportedFilter):
                loaded.similarity_search(EAST, top_k=10, metadata_filter=want)
            return

        hits = loaded.similarity_search(EAST, top_k=10, metadata_filter=want)
        assert {d.id for d, _ in hits} == {"b", "c"}

    def test_boolean_composition(self, loaded):
        want = {"$or": [{"lang": "rs"}, {"kind": "test"}]}
        if not self._supports(loaded, StoreCapability.BOOLEAN_FILTERS):
            with pytest.raises(UnsupportedFilter):
                loaded.similarity_search(EAST, top_k=10, metadata_filter=want)
            return

        hits = loaded.similarity_search(EAST, top_k=10, metadata_filter=want)
        assert {d.id for d, _ in hits} == {"b", "c"}

    def test_negation(self, loaded):
        want = {"$not": {"lang": "py"}}
        if not self._supports(loaded, StoreCapability.NEGATION):
            with pytest.raises(UnsupportedFilter):
                loaded.similarity_search(EAST, top_k=10, metadata_filter=want)
            return

        hits = loaded.similarity_search(EAST, top_k=10, metadata_filter=want)
        assert {d.id for d, _ in hits} == {"b"}

    def test_an_unsupported_filter_raises_rather_than_returning_unfiltered_rows(
        self, loaded
    ):
        """The whole point of the capability flags. A store that cannot express a
        filter must say so — returning everything looks like a working query."""
        for cap, flt in (
            (StoreCapability.RANGE_FILTERS, {"score": {"$gt": 15}}),
            (StoreCapability.NEGATION, {"$not": {"lang": "py"}}),
            (StoreCapability.BOOLEAN_FILTERS, {"$or": [{"lang": "py"}, {"lang": "rs"}]}),
        ):
            if self._supports(loaded, cap):
                continue
            with pytest.raises(UnsupportedFilter):
                loaded.similarity_search(EAST, top_k=10, metadata_filter=flt)

    def test_a_malformed_filter_is_rejected(self, loaded):
        with pytest.raises(ValueError):
            loaded.similarity_search(EAST, top_k=10, metadata_filter={"a": {"$bogus": 1}})
