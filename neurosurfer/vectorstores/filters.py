"""The metadata filter grammar every vector store speaks.

`BaseVectorDB.similarity_search` took a `dict[str, Any]` and left each backend to
decide what it meant. Chroma read a list as `$in` and anything else as equality
(`chroma.py`, before this module); the in-memory store ignored the argument
entirely. So "filter by metadata" meant three things depending on what you were
holding, and a range or a negation had nowhere to go at all — which is precisely
what people reach for Qdrant and pgvector to get.

**The grammar is deliberately small: what two backends can both express.**
Equality, membership, ordering, and boolean composition. Anything richer is a
promise the in-memory store cannot keep, and a `StoreCapability` is the honest
way for a backend to say it does more.

The shape is Mongo-flavoured because Chroma's `where` already is, so the most
common backend translates almost one-for-one:

    {"lang": "py"}                          # shorthand: equals
    {"lang": ["py", "rs"]}                   # shorthand: one of
    {"lang": {"$eq": "py"}}                  # explicit
    {"score": {"$gte": 0.5, "$lt": 0.9}}     # two predicates on one field, ANDed
    {"lang": "py", "kind": "src"}            # two fields, ANDed
    {"$or": [{"lang": "py"}, {"lang": "rs"}]}
    {"$not": {"lang": "py"}}

`normalize` turns any of those into one canonical form, so a backend translates
from a shape it can trust rather than re-deriving the shorthand rules. `matches`
is the reference implementation of what the canonical form *means*, and it is
what the conformance suite checks every backend against.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "COMPARISON_OPS",
    "LOGICAL_OPS",
    "UnsupportedFilter",
    "matches",
    "normalize",
]

#: Operators over one field's value.
COMPARISON_OPS = frozenset({"$eq", "$ne", "$in", "$nin", "$gt", "$gte", "$lt", "$lte"})

#: Operators over whole filters.
LOGICAL_OPS = frozenset({"$and", "$or", "$not"})

#: Operators whose value is a list of filters rather than one filter.
_LIST_LOGICAL = frozenset({"$and", "$or"})


class UnsupportedFilter(ValueError):
    """A filter this backend cannot express.

    Distinct from a malformed one. `normalize` raises `ValueError` for a filter
    that is wrong under the grammar; a backend raises this for a filter that is
    valid and beyond it — `$not` on a store whose query language has no negation.
    The difference matters to the caller: the first is a bug in their filter, the
    second is a reason to pick another backend or another query.
    """


def normalize(raw: Any) -> dict[str, Any] | None:
    """Canonicalise *raw* into the explicit form, or ``None`` for "no filter".

    Canonical means: every field predicate is a dict of `$op` → value, and every
    logical operator holds already-canonical filters. Shorthand is expanded —
    a bare scalar becomes `$eq`, a bare list becomes `$in`.

    Raises `ValueError` on anything the grammar does not describe, because a
    filter that is silently ignored is how a query comes back with the wrong rows
    and nobody notices.
    """
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError(f"filter must be a dict, got {type(raw).__name__}")
    if not raw:
        return None

    out: dict[str, Any] = {}
    for key, value in raw.items():
        if key in _LIST_LOGICAL:
            if not isinstance(value, (list, tuple)) or not value:
                raise ValueError(f"{key} takes a non-empty list of filters")
            clauses = [normalize(v) for v in value]
            out[key] = [c for c in clauses if c is not None]
        elif key == "$not":
            inner = normalize(value)
            if inner is None:
                raise ValueError("$not takes a filter")
            out[key] = inner
        elif key.startswith("$"):
            raise ValueError(
                f"unknown operator {key!r} at filter top level; "
                f"expected a field name or one of {sorted(LOGICAL_OPS)}"
            )
        else:
            out[key] = _normalize_predicate(key, value)
    return out or None


def _normalize_predicate(field: str, value: Any) -> dict[str, Any]:
    """One field's predicate, with the two shorthands expanded."""
    if isinstance(value, dict) and any(k.startswith("$") for k in value):
        unknown = {k for k in value if k not in COMPARISON_OPS}
        if unknown:
            raise ValueError(
                f"unknown operator(s) {sorted(unknown)} on field {field!r}; "
                f"expected one of {sorted(COMPARISON_OPS)}"
            )
        for op in ("$in", "$nin"):
            if op in value and not isinstance(value[op], (list, tuple, set)):
                raise ValueError(f"{op} on field {field!r} takes a list")
        return {k: (list(v) if k in ("$in", "$nin") else v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return {"$in": list(value)}
    return {"$eq": value}


def matches(metadata: dict[str, Any] | None, flt: dict[str, Any] | None) -> bool:
    """Does *metadata* satisfy the **canonical** filter *flt*?

    The reference semantics. A backend is conformant when its own query language
    agrees with this function on every case in the conformance suite.

    A field the metadata does not carry never matches — including under `$ne`
    and `$nin`. That is the choice worth stating: "not equal to py" reads as
    though a document with no `lang` should pass, but a store that returns rows
    it knows nothing about when asked to exclude something is a store you cannot
    filter with. Absent is absent, and `$or` is how you ask for either.
    """
    if flt is None:
        return True
    meta = metadata or {}

    for key, value in flt.items():
        if key == "$and":
            if not all(matches(meta, c) for c in value):
                return False
        elif key == "$or":
            if not any(matches(meta, c) for c in value):
                return False
        elif key == "$not":
            if matches(meta, value):
                return False
        else:
            if key not in meta:
                return False
            if not _predicate_matches(meta[key], value):
                return False
    return True


def _predicate_matches(actual: Any, predicate: dict[str, Any]) -> bool:
    for op, want in predicate.items():
        if op == "$eq":
            if actual != want:
                return False
        elif op == "$ne":
            if actual == want:
                return False
        elif op == "$in":
            if actual not in want:
                return False
        elif op == "$nin":
            if actual in want:
                return False
        else:
            # Ordering against a value it cannot be ordered against is a
            # non-match, not a crash: metadata is heterogeneous by nature and one
            # odd row should not fail a query over ten thousand good ones.
            try:
                if op == "$gt" and not actual > want:
                    return False
                if op == "$gte" and not actual >= want:
                    return False
                if op == "$lt" and not actual < want:
                    return False
                if op == "$lte" and not actual <= want:
                    return False
            except TypeError:
                return False
    return True
