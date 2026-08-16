"""Qdrant against the same suite. The whole test module is four lines of setup.

That is the deliverable of Phase 1 being proved: adding a backend is writing one
class and pointing the existing suite at it. Anything the suite needed changed to
accommodate Qdrant would have been a finding about the contract, not about Qdrant.
"""

from __future__ import annotations

import pytest

from .conformance import VectorStoreConformance

pytest.importorskip("qdrant_client", reason="needs the `qdrant` extra")

from neurosurfer.vectorstores import QdrantVectorStore  # noqa: E402


class TestQdrantVectorStore(VectorStoreConformance):
    def make_store(self, request):
        return QdrantVectorStore("conformance-probe", dim=2, location=":memory:")


# ── behaviour specific to this backend ──────────────────────────────────────


def test_it_declares_every_capability():
    """The reason Qdrant was chosen first: it is the backend that can express the
    whole grammar, so it is the one that proves the grammar is expressible."""
    from neurosurfer.vectorstores import StoreCapability

    store = QdrantVectorStore("caps-probe", dim=2)
    assert store.capabilities == StoreCapability.ALL


def test_string_ids_survive_the_uuid_mapping():
    """Qdrant point ids are ints or UUIDs; ours are strings like `install:0`."""
    from neurosurfer.vectorstores import Doc

    store = QdrantVectorStore("ids-probe", dim=2)
    store.add_documents([Doc(id="install:0", text="x", embedding=[1.0, 0.0])])

    [doc] = store.list_all_documents()
    assert doc.id == "install:0"

    store.delete_documents(["install:0"])
    assert store.count() == 0


def test_the_id_mapping_is_deterministic():
    """Same document id, same point — otherwise every re-ingest duplicates."""
    from neurosurfer.vectorstores.qdrant import _point_id

    assert _point_id("a") == _point_id("a")
    assert _point_id("a") != _point_id("b")


def test_ne_does_not_match_a_document_missing_the_field():
    """Qdrant's `must_not` matches a document that lacks the field entirely,
    which disagrees with `filters.matches`. The translation carries an explicit
    existence condition so both answer the same question."""
    from neurosurfer.vectorstores import Doc

    store = QdrantVectorStore("ne-probe", dim=2)
    store.add_documents(
        [
            Doc(id="has", text="x", embedding=[1.0, 0.0], metadata={"lang": "py"}),
            Doc(id="has-other", text="y", embedding=[1.0, 0.0], metadata={"lang": "rs"}),
            Doc(id="missing", text="z", embedding=[1.0, 0.0], metadata={"kind": "src"}),
        ]
    )

    got = {d.id for d in store.list_all_documents({"lang": {"$ne": "py"}})}

    assert got == {"has-other"}, "a document with no `lang` must not satisfy `lang != py`"


def test_it_agrees_with_the_reference_implementation():
    """Cross-checked against `filters.matches` rather than against my reading of
    Qdrant's docs — the reference is what the contract actually means."""
    from neurosurfer.vectorstores import Doc
    from neurosurfer.vectorstores.filters import matches, normalize

    rows = [
        {"lang": "py", "kind": "src", "score": 10},
        {"lang": "rs", "kind": "src", "score": 20},
        {"lang": "py", "kind": "test", "score": 30},
        {"kind": "test"},
    ]
    store = QdrantVectorStore("agree-probe", dim=2)
    store.add_documents(
        [
            Doc(id=str(i), text=str(i), embedding=[1.0, 0.0], metadata=m)
            for i, m in enumerate(rows)
        ]
    )

    filters = [
        {"lang": "py"},
        {"lang": ["py", "rs"]},
        {"score": {"$gte": 20}},
        {"score": {"$gt": 10, "$lt": 30}},
        {"lang": "py", "kind": "test"},
        {"$or": [{"lang": "rs"}, {"kind": "test"}]},
        {"$not": {"lang": "py"}},
        {"lang": {"$ne": "py"}},
        {"lang": {"$nin": ["py"]}},
    ]

    for flt in filters:
        expected = {
            str(i) for i, m in enumerate(rows) if matches(m, normalize(flt))
        }
        actual = {d.id for d in store.list_all_documents(flt)}
        assert actual == expected, f"disagreement on {flt}: {actual} != {expected}"
