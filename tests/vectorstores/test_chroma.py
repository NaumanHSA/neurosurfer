"""Chroma against the same suite, plus the two defects that were specific to it."""

from __future__ import annotations

import pytest

from .conformance import EAST, NORTH, VectorStoreConformance

chromadb = pytest.importorskip("chromadb", reason="needs the `rag` extra")

from neurosurfer.vectorstores import ChromaVectorStore  # noqa: E402


def _store(tmp_path, name="conformance-probe"):
    return ChromaVectorStore(
        collection_name=name, persist_directory=str(tmp_path), clear_collection=True
    )


class TestChromaVectorStore(VectorStoreConformance):
    def make_store(self, request):
        return _store(request.getfixturevalue("tmp_path"))


# ── the defects that were Chroma's own ──────────────────────────────────────


def test_new_collections_are_cosine(tmp_path):
    """The contract is cosine similarity. Chroma's default is squared L2, and the
    old code returned `1.0 - distance` regardless — so orthogonal vectors scored
    **-1.0** and any `similarity_threshold` was applied to the wrong scale."""
    store = _store(tmp_path)

    assert store._space == "cosine"


def test_an_orthogonal_vector_scores_zero_not_minus_one(tmp_path):
    from neurosurfer.vectorstores.base import Doc

    store = _store(tmp_path)
    store.add_documents([Doc(id="n", text="north", embedding=NORTH)])

    [(_, score)] = store.similarity_search(EAST, top_k=1)
    assert score == pytest.approx(0.0, abs=1e-3)


def test_scores_are_corrected_on_a_collection_built_before_the_fix(tmp_path):
    """An existing L2 collection is the one whose scores most need correcting, so
    the space is read back rather than assumed — `get_or_create_collection` keeps
    the configuration a collection already has."""
    from neurosurfer.vectorstores.base import Doc

    client = chromadb.PersistentClient(path=str(tmp_path))
    client.get_or_create_collection(name="legacy-l2")  # no hnsw:space → l2

    store = ChromaVectorStore("legacy-l2", persist_directory=str(tmp_path))
    assert store._space == "l2"

    store.add_documents(
        [Doc(id="e", text="east", embedding=EAST), Doc(id="n", text="north", embedding=NORTH)]
    )
    by_id = {d.id: s for d, s in store.similarity_search(EAST, top_k=2)}

    assert by_id["e"] == pytest.approx(1.0, abs=1e-3)
    assert by_id["n"] == pytest.approx(0.0, abs=1e-3)


def test_listing_returns_the_real_ids(tmp_path):
    """`list_all_documents` minted a fresh uuid4 per row, so the ids it returned
    matched nothing and deleting by them silently did nothing."""
    from neurosurfer.vectorstores.base import Doc

    store = _store(tmp_path)
    store.add_documents([Doc(id="known", text="x", embedding=EAST)])

    assert [d.id for d in store.list_all_documents()] == ["known"]


def test_not_is_refused_rather_than_silently_dropped(tmp_path):
    """Chroma's `where` has no `$not`. Half a capability is how a caller ends up
    with rows they asked to exclude.

    The refusal comes from the declared capability rather than from Chroma's
    translator, which is the point of declaring one: the same message for every
    backend that lacks the same thing, before any query is formed.
    """
    from neurosurfer.vectorstores.filters import UnsupportedFilter

    store = _store(tmp_path)

    with pytest.raises(UnsupportedFilter, match="needs \\['negation'\\]"):
        store.similarity_search(EAST, top_k=5, metadata_filter={"$not": {"lang": "py"}})
