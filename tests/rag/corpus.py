"""A fixture corpus with a deterministic "embedding", so retrieval is measurable.

The point of a fixture is that the *retrieval* is under test, not an embedding
model's opinion. So the vectors here come from a hash-based bag-of-words
embedder: two texts sharing vocabulary get similar vectors, and a query sharing
no vocabulary with a document gets a near-orthogonal one.

That gives dense search a realistic weakness on purpose. `ENAMETOOLONG` appears
in exactly one chunk and in no query's ordinary vocabulary, so a bag-of-words
dense model cannot find it by meaning — which is precisely the case BM25 handles
and the reason hybrid retrieval exists. A fixture where dense search already wins
everything would measure nothing.
"""

from __future__ import annotations

import hashlib
import math

from neurosurfer.rag.evaluation import EvalCase
from neurosurfer.vectorstores import Doc

DIM = 64


class HashingEmbedder:
    """Deterministic bag-of-words vectors. No model, no network, no randomness."""

    id = "hashing"
    model = "hashing-test-embedder"
    dimensions = DIM
    max_batch = 512

    def available(self) -> bool:
        return True

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._one(t) for t in texts]

    def _one(self, text: str) -> list[float]:
        vec = [0.0] * DIM
        for token in text.lower().split():
            word = "".join(ch for ch in token if ch.isalnum())
            if not word:
                continue
            slot = int(hashlib.md5(word.encode()).hexdigest(), 16) % DIM
            vec[slot] += 1.0
        norm = math.sqrt(sum(v * v for v in vec))
        return [v / norm for v in vec] if norm else vec


#: (id, text). Prose about a handful of topics, with the two hard cases seeded:
#: an error code that appears once, and a topic split across three near-identical
#: chunks so a diversity-free top-k fills itself with one paragraph.
DOCUMENTS: list[tuple[str, str]] = [
    ("install:0", "Install the package with pip install neurosurfer to get started quickly."),
    ("install:1", "Installation requires Python 3.11 or later and a recent pip release."),
    ("install:2", "After installing you can verify the install by importing the package."),
    ("agents:0", "An agent loop calls tools repeatedly until the task is finished."),
    ("agents:1", "The react agent parses thought and action text for models without tool APIs."),
    ("graph:0", "A graph node depends on other nodes and the engine runs independent nodes together."),
    ("graph:1", "Router nodes branch a workflow on a classification of the input."),
    ("errors:0", "The scan failed with ENAMETOOLONG because the candidate path exceeded NAME_MAX."),
    ("errors:1", "A retryable error backs off exponentially and honours the retry-after header."),
    ("vector:0", "Cosine similarity ranks documents by the angle between their embedding vectors."),
    ("vector:1", "Chroma stores vectors on disk and returns a distance for each neighbour."),
    ("tokens:0", "Token budgets decide how much retrieved context fits into one prompt."),
]

#: Deliberately three near-duplicates, to make redundancy measurable.
DOCUMENTS += [
    ("dup:0", "Prompt caching reduces cost by reusing an identical system prompt across calls."),
    ("dup:1", "Prompt caching reduces cost by reusing the identical system prompt between calls."),
    ("dup:2", "Prompt caching cuts cost by reusing an identical system prompt on repeat calls."),
]

CASES: list[EvalCase] = [
    EvalCase(
        query="how do I install the package",
        relevant_ids={"install:0", "install:1", "install:2"},
        note="ordinary topical query — dense should handle this",
    ),
    EvalCase(
        query="ENAMETOOLONG",
        relevant_ids={"errors:0"},
        note="a rare literal token; the case a bag-of-words dense model cannot do",
    ),
    EvalCase(
        query="NAME_MAX exceeded candidate path",
        relevant_ids={"errors:0"},
        note="identifier-heavy, the shape of a real error-message search",
    ),
    EvalCase(
        query="branch a workflow on a classification",
        relevant_ids={"graph:1"},
        note="topical, phrased differently from the document",
    ),
    EvalCase(
        query="how are independent nodes executed",
        relevant_ids={"graph:0"},
        note="topical",
    ),
    EvalCase(
        query="angle between embedding vectors",
        relevant_ids={"vector:0"},
        note="topical with shared vocabulary",
    ),
    EvalCase(
        query="retry-after header backoff",
        relevant_ids={"errors:1"},
        note="mixed literal and topical",
    ),
]


def build_docs(embedder: HashingEmbedder) -> list[Doc]:
    texts = [t for _, t in DOCUMENTS]
    vectors = embedder.embed(texts)
    return [
        Doc(id=doc_id, text=text, embedding=vec, metadata={"topic": doc_id.split(":")[0]})
        for (doc_id, text), vec in zip(DOCUMENTS, vectors, strict=True)
    ]
