"""What an embedding backend is, and the two ways it can fail to be one.

This layer was 69 lines with a three-branch `if`, and it is what every retrieval
path depends on. There was exactly one way to produce a vector — sentence-transformers,
so `torch` — even for a user whose whole stack is a local OpenAI-compatible
server that already exposes `/v1/embeddings`.

The shape here follows ``mcp/sources``, which solved the same problem for
registries: a Protocol, a metadata triple a caller can render, and `available()`
so a chooser can *ask* rather than construct one and see what happens.

**The two failures are different and must stay different.**

* :class:`EmbedderUnavailable` — *not configured*. The optional dependency is not
  installed, or no key is set. Degrading to lexical search is a reasonable answer.
* :class:`EmbeddingError` — *configured and broken*. The key is expired, the
  server is down, the model name is wrong. Degrading here turns "your credentials
  expired" into "retrieval quietly got worse", which is the class of silent
  plausible success the graph engine was rebuilt to eliminate.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

__all__ = [
    "EmbeddingBackend",
    "EmbeddingError",
    "EmbedderUnavailable",
    "Embedder",
    "NullEmbedder",
]


class EmbedderUnavailable(RuntimeError):
    """The backend is not usable as configured — a missing dep or an absent key.

    Nothing went wrong; it simply is not set up. `get_embedder` turns this into
    ``None`` so a caller can fall back to lexical search.
    """


class EmbeddingError(RuntimeError):
    """The backend was configured and the call failed anyway.

    Raised, never swallowed. An expired key or an unreachable server must not
    read as "no embeddings configured".
    """


@runtime_checkable
class Embedder(Protocol):
    """The minimum: texts in, vectors out, deterministically.

    Kept as the historical name and the narrowest possible surface, so anything
    with an ``embed`` method — including a test double — is still an embedder.
    """

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class EmbeddingBackend(Embedder, Protocol):
    """A named, describable embedder that a chooser can render and check.

    Implementations live in this package and are registered in ``registry.py``.
    """

    #: Stable identifier used in a spec string and stored in settings.
    id: str
    #: Shown in a UI.
    label: str
    #: The specific model this instance produces vectors with. Recorded on a
    #: collection, because a store holding two models' vectors returns confident
    #: nonsense rather than an error.
    model: str
    #: Vector width, or ``None`` when it is only knowable after the first call.
    dimensions: int | None
    #: How many texts to send per request.
    max_batch: int

    def available(self) -> bool:
        """False when this backend cannot be used as configured."""
        ...


class NullEmbedder:
    """No-op backend — signals 'use BM25/lexical'."""

    id = "null"
    label = "No embeddings"
    model = ""
    dimensions = None
    max_batch = 1

    def available(self) -> bool:
        return True

    def embed(self, texts: list[str]) -> list[list[float]]:
        return []
