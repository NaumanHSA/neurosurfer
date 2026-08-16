"""Parent-document retrieval — embed small, return large.

Chunk size is a trade with no good answer. Small chunks embed precisely, because
one vector describes one idea; but the model then receives a fragment with no
surrounding argument. Large chunks read well and retrieve badly, because one
vector averaged over eight ideas is close to none of them.

Parent-document retrieval refuses the trade: index the small chunks, and when one
is retrieved, hand back the section it belongs to. Precision from the child,
context from the parent.

The parent store is deliberately a plain mapping. A parent is fetched by id and
never searched, so putting it in a vector store would mean embedding text nothing
ever embeds against — cost with no purpose.
"""

from __future__ import annotations

from collections.abc import Sequence

from ...vectorstores.base import Doc

__all__ = ["PARENT_ID_KEY", "ParentDocumentRetriever", "split_parent_child"]

PARENT_ID_KEY = "parent_id"


def split_parent_child(
    parents: Sequence[tuple[str, str]],
    child_of: callable,
) -> tuple[dict[str, str], list[tuple[str, str, str]]]:
    """`[(parent_id, text)]` → the parent map and `[(child_id, parent_id, text)]`.

    ``child_of(text) -> list[str]`` is any chunker; the children are numbered
    within their parent so their ids are stable across runs.
    """
    parent_map: dict[str, str] = {}
    children: list[tuple[str, str, str]] = []
    for parent_id, text in parents:
        parent_map[parent_id] = text
        for i, child in enumerate(child_of(text)):
            children.append((f"{parent_id}#{i}", parent_id, child))
    return parent_map, children


class ParentDocumentRetriever:
    """Wraps a child-level retrieval and returns the parents instead.

    ``parents`` maps parent id → text. Anything with ``__getitem__`` and ``get``
    will do — a dict, a shelf, a row lookup.
    """

    def __init__(self, parents: dict[str, str]) -> None:
        self.parents = parents

    def expand(self, children: Sequence[Doc], *, top_k: int | None = None) -> list[Doc]:
        """Child hits → their parent documents, de-duplicated, order preserved.

        **De-duplication is the point, not a detail.** Several children of one
        section routinely retrieve together — that is what a well-chunked section
        looks like — and returning the same parent three times would fill the
        context window with one passage repeated.
        """
        out: list[Doc] = []
        seen: set[str] = set()
        for child in children:
            parent_id = (child.metadata or {}).get(PARENT_ID_KEY)
            if not parent_id or parent_id in seen:
                continue
            seen.add(parent_id)
            text = self.parents.get(parent_id)
            if text is None:
                # A child whose parent is gone still carries its own text, which
                # is a worse answer than the parent and a much better one than
                # dropping the hit entirely.
                out.append(child)
                continue
            out.append(
                Doc(
                    id=parent_id,
                    text=text,
                    embedding=None,
                    metadata={
                        **(child.metadata or {}),
                        "retrieved_via": child.id,
                        "is_parent": True,
                    },
                )
            )
            if top_k is not None and len(out) >= top_k:
                break
        return out
