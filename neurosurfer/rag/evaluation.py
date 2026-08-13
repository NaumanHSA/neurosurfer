"""Measuring retrieval, so a change to it is a number rather than a conviction.

Everything in this phase — hybrid search, reranking, diversity — is a claim about
quality. Plan 01 spent a section admitting it had built mechanisms with no way to
measure them; this exists so that is not repeated one directory over.

The metrics are the standard three, and each answers a different question:

* **recall@k** — did the right chunk make it into the window at all? The one that
  matters most, because nothing downstream can recover from a miss.
* **MRR** — how far down was the first right answer? Sensitive to ordering in a
  way recall is not.
* **nDCG@k** — ordering again, but crediting several relevant results rather than
  only the first.

Usage::

    from neurosurfer.rag.evaluation import EvalCase, evaluate

    cases = [EvalCase(query="how do I install", relevant_ids={"readme:0"})]
    report = evaluate(cases, lambda q, k: my_retriever(q, k))
    print(report)
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

__all__ = ["EvalCase", "EvalReport", "compare", "evaluate"]


@dataclass(frozen=True)
class EvalCase:
    """One query and the chunk ids that should come back for it."""

    query: str
    relevant_ids: frozenset[str] | set[str]
    #: Free-text note for a reader of a failing run — why this case exists.
    note: str = ""


@dataclass
class EvalReport:
    """What a retriever scored, and on how many cases."""

    k: int
    cases: int
    recall: float
    mrr: float
    ndcg: float
    #: Per-case recall, so a regression can be traced to the query that caused it.
    per_case: dict[str, float] = field(default_factory=dict)

    def __str__(self) -> str:
        return (
            f"recall@{self.k}={self.recall:.3f}  "
            f"mrr={self.mrr:.3f}  ndcg@{self.k}={self.ndcg:.3f}  "
            f"({self.cases} cases)"
        )

    def better_than(self, other: EvalReport, *, margin: float = 0.0) -> bool:
        """Is this report an improvement on *other* by more than *margin*?

        Recall first: a chunk that never arrives cannot be reordered into place.
        """
        if abs(self.recall - other.recall) > margin:
            return self.recall > other.recall
        return self.ndcg > other.ndcg + margin


#: `(query, k) -> ranked ids, best first`. Deliberately not a `Retriever` type:
#: the harness should be able to measure anything, including a lambda in a test.
Retrieve = Callable[[str, int], Sequence[str]]


def evaluate(cases: Sequence[EvalCase], retrieve: Retrieve, k: int = 5) -> EvalReport:
    """Run *retrieve* over *cases* and score it."""
    if not cases:
        return EvalReport(k=k, cases=0, recall=0.0, mrr=0.0, ndcg=0.0)

    recalls: list[float] = []
    rrs: list[float] = []
    ndcgs: list[float] = []
    per_case: dict[str, float] = {}

    for case in cases:
        got = list(retrieve(case.query, k))[:k]
        relevant = set(case.relevant_ids)

        hits = [i for i, doc_id in enumerate(got) if doc_id in relevant]

        recall = len(set(got) & relevant) / len(relevant) if relevant else 0.0
        recalls.append(recall)
        per_case[case.query] = recall

        rrs.append(1.0 / (hits[0] + 1) if hits else 0.0)
        ndcgs.append(_ndcg(hits, min(len(relevant), k)))

    n = len(cases)
    return EvalReport(
        k=k,
        cases=n,
        recall=sum(recalls) / n,
        mrr=sum(rrs) / n,
        ndcg=sum(ndcgs) / n,
        per_case=per_case,
    )


def _ndcg(hit_positions: list[int], ideal_hits: int) -> float:
    """Binary-relevance nDCG from the 0-based ranks that were relevant."""
    if not ideal_hits:
        return 0.0
    dcg = sum(1.0 / math.log2(pos + 2) for pos in hit_positions)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg else 0.0


def compare(
    cases: Sequence[EvalCase], variants: dict[str, Retrieve], k: int = 5
) -> dict[str, EvalReport]:
    """Score several retrievers over the same cases.

    The shape an A/B actually takes: one corpus, one question set, several
    configurations, printed together so the delta is visible rather than
    remembered from the last run.
    """
    return {name: evaluate(cases, fn, k) for name, fn in variants.items()}
