"""The in-memory store against the conformance suite — and it is the yardstick.

Two of these would have failed before this phase for a reason worth remembering:
the class could not be constructed at all, and once it could, it appended instead
of upserting and ignored every filter it was handed.
"""

from __future__ import annotations

from neurosurfer.vectorstores import InMemoryVectorStore

from .conformance import EAST, NORTH, VectorStoreConformance


class TestInMemoryVectorStore(VectorStoreConformance):
    def make_store(self, request):
        return InMemoryVectorStore()


# ── behaviour specific to this backend ──────────────────────────────────────


def test_dimension_is_enforced_once_known():
    """Vectors of two widths in one collection means two embedding models, which
    is a silently wrong search rather than an obviously broken one."""
    import pytest

    from neurosurfer.vectorstores.base import Doc

    store = InMemoryVectorStore()
    store.add_documents([Doc(id="a", text="x", embedding=EAST)])

    with pytest.raises(ValueError, match="one embedding model"):
        store.add_documents([Doc(id="b", text="y", embedding=[1.0, 2.0, 3.0])])


def test_a_declared_dimension_is_enforced_from_the_start():
    import pytest

    from neurosurfer.vectorstores.base import Doc

    store = InMemoryVectorStore(dim=3)

    with pytest.raises(ValueError, match="wide"):
        store.add_documents([Doc(id="a", text="x", embedding=EAST)])


def test_a_zero_vector_scores_zero_rather_than_dividing_by_zero():
    from neurosurfer.vectorstores.base import Doc

    store = InMemoryVectorStore()
    store.add_documents([Doc(id="zero", text="z", embedding=[0.0, 0.0])])

    [(_, score)] = store.similarity_search(NORTH, top_k=1)
    assert score == 0.0
