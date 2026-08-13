"""`/v1/embeddings` — hosted OpenAI, and any server that speaks its protocol.

The backend that unblocks a fully local stack. Neurosurfer already talks to LM
Studio, Ollama, vLLM and llama.cpp for chat; those same servers expose
`/v1/embeddings`, and until now using one for retrieval still meant installing
`torch` and `sentence-transformers` to embed in-process.

Two classes because there are two questions, not two protocols:
:class:`OpenAIEmbedder` defaults to `api.openai.com` and needs a key;
:class:`OpenAICompatEmbedder` needs a `base_url` and treats the key as optional,
because local servers ignore it.
"""

from __future__ import annotations

import os
import random
import time

import httpx

from ..llm.retry import is_retryable_error, retry_after_seconds
from ..observability.logging import get_logger
from .base import EmbedderUnavailable, EmbeddingError

log = get_logger("embeddings.openai")

DEFAULT_MODEL = "text-embedding-3-small"
OPENAI_BASE_URL = "https://api.openai.com/v1"

#: Conservative: the request limit is on total tokens, not texts, and a caller
#: chunking a book hits it long before 2048 items.
DEFAULT_MAX_BATCH = 128


class OpenAICompatEmbedder:
    """Any server implementing `POST {base_url}/embeddings`."""

    id = "openai-compat"
    label = "OpenAI-compatible endpoint"

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str | None = None,
        *,
        dimensions: int | None = None,
        max_batch: int = DEFAULT_MAX_BATCH,
        timeout: float = 60.0,
        max_attempts: int = 4,
    ) -> None:
        if not model:
            raise EmbedderUnavailable(f"{self.id}: no model named")
        if not base_url:
            raise EmbedderUnavailable(
                f"{self.id}: no base_url. Use "
                f"`openai-compat:<model>@http://localhost:1234/v1`."
            )
        self.model = model
        self.base_url = base_url.rstrip("/")
        # Local servers ignore it but some reject an absent header outright.
        self.api_key = api_key or "not-needed"
        self.max_batch = max(1, max_batch)
        self._timeout = timeout
        self._max_attempts = max(1, max_attempts)
        self._dimensions = dimensions

    @property
    def dimensions(self) -> int | None:
        """Known only after the first call, unless it was declared up front."""
        return self._dimensions

    def available(self) -> bool:
        return True

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        out: list[list[float]] = []
        for i in range(0, len(texts), self.max_batch):
            out.extend(self._embed_batch(texts[i : i + self.max_batch]))
        if out and self._dimensions is None:
            self._dimensions = len(out[0])
        return out

    def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        """One request, retried on the errors worth retrying.

        Sync on purpose — `Embedder.embed` is sync and every caller is. The
        retryable-error predicates come from `llm/retry.py` so an embedding
        request and a chat request agree on what a 429 means; only the sleep
        differs, because there is no event loop to await on here.
        """
        payload = {"model": self.model, "input": batch}
        headers = {"Authorization": f"Bearer {self.api_key}"}
        url = f"{self.base_url}/embeddings"

        attempt = 0
        while True:
            try:
                resp = httpx.post(
                    url, json=payload, headers=headers, timeout=self._timeout
                )
                resp.raise_for_status()
                return self._parse(resp.json())
            except Exception as err:  # noqa: BLE001
                attempt += 1
                if attempt >= self._max_attempts or not is_retryable_error(err):
                    raise EmbeddingError(
                        f"{self.model} at {self.base_url} failed to embed "
                        f"{len(batch)} text(s): {type(err).__name__}: {err}"
                    ) from err
                delay = retry_after_seconds(err)
                if delay is None:
                    delay = min(30.0, 1.0 * (2 ** (attempt - 1)))
                    delay += random.uniform(0, delay * 0.25)
                log.warning(
                    "embeddings retryable error (attempt %d/%d): %s — backing off %.1fs",
                    attempt,
                    self._max_attempts,
                    type(err).__name__,
                    delay,
                )
                time.sleep(delay)

    def _parse(self, body: dict) -> list[list[float]]:
        data = body.get("data")
        if not isinstance(data, list):
            raise EmbeddingError(
                f"{self.base_url} did not return an embeddings payload; got keys "
                f"{sorted(body)[:6]}. Is this an OpenAI-compatible endpoint?"
            )
        # `index` is authoritative: the spec permits any order, and a server that
        # returns them shuffled would otherwise silently mis-pair text to vector.
        rows = sorted(data, key=lambda d: d.get("index", 0))
        return [[float(x) for x in row["embedding"]] for row in rows]


class OpenAIEmbedder(OpenAICompatEmbedder):
    """Hosted OpenAI. Same protocol, different defaults and a required key."""

    id = "openai"
    label = "OpenAI"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        base_url: str = OPENAI_BASE_URL,
        **kw,
    ) -> None:
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise EmbedderUnavailable(
                "OPENAI_API_KEY is not set. Set it, or point at a local server "
                "with `openai-compat:<model>@<base_url>`."
            )
        super().__init__(model or DEFAULT_MODEL, base_url, key, **kw)
