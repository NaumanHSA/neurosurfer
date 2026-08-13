"""sentence-transformers, in-process. The original backend, behind the Protocol."""

from __future__ import annotations

from .base import EmbedderUnavailable, EmbeddingError

DEFAULT_MODEL = "all-MiniLM-L6-v2"


class LocalEmbedder:
    """Embeds in-process with `sentence-transformers`.

    Needs the `rag` extra (which brings `torch`), so `available()` is a real
    question rather than a formality — it is the heaviest dependency in the
    project and the one an OpenAI-compatible server exists to avoid.
    """

    id = "local"
    label = "sentence-transformers (in-process)"
    max_batch = 256

    def __init__(self, model: str = DEFAULT_MODEL) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise EmbedderUnavailable(
                "sentence-transformers is not installed — "
                "pip install 'neurosurfer[rag]', or use an OpenAI-compatible "
                "endpoint with `openai-compat:<model>@<base_url>`."
            ) from e

        self.model = model
        try:
            self._model = SentenceTransformer(model)
        except Exception as e:  # noqa: BLE001
            # A model name that does not resolve is *configured and broken*, not
            # unconfigured — the user named it, so they hear about it.
            raise EmbeddingError(f"could not load embedding model {model!r}: {e}") from e

    @property
    def dimensions(self) -> int | None:
        try:
            return int(self._model.get_sentence_embedding_dimension())
        except Exception:  # noqa: BLE001
            return None

    def available(self) -> bool:
        return True

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            vecs = self._model.encode(texts, normalize_embeddings=True)
        except Exception as e:  # noqa: BLE001
            raise EmbeddingError(f"{self.model} failed to embed: {e}") from e
        return [list(map(float, v)) for v in vecs]
