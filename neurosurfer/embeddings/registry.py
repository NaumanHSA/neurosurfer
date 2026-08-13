"""Turning a spec string into a backend, and deciding when `None` is honest.

    none | bm25 | off                     → None, use lexical search
    local                                 → sentence-transformers, default model
    intfloat/e5-small-v2                  → sentence-transformers, that model
    openai                                → hosted OpenAI, default model
    openai:text-embedding-3-large         → hosted OpenAI, that model
    openai-compat:nomic-embed@http://…/v1 → any OpenAI-compatible server

**A bare string with no known prefix still means sentence-transformers**, which
is what it meant before this package existed. `intfloat/e5-small-v2` is in
`RAGAgentConfig` today and must keep working.
"""

from __future__ import annotations

from ..observability.logging import get_logger
from .base import Embedder, EmbedderUnavailable, EmbeddingError
from .local import LocalEmbedder
from .openai_compat import OpenAICompatEmbedder, OpenAIEmbedder

log = get_logger("embeddings")

__all__ = ["BACKENDS", "get_embedder", "parse_spec"]

#: Sentinels meaning "no embeddings" rather than a backend that failed.
_OFF = frozenset({"", "none", "null", "off", "bm25", "lexical"})

BACKENDS: dict[str, type] = {
    "local": LocalEmbedder,
    "sentence-transformers": LocalEmbedder,
    "st": LocalEmbedder,
    "openai": OpenAIEmbedder,
    "openai-compat": OpenAICompatEmbedder,
}


def parse_spec(spec: str) -> tuple[str, dict]:
    """`spec` → (backend id, kwargs). Raises `ValueError` on a malformed spec."""
    text = spec.strip()
    head, _, rest = text.partition(":")
    head_l = head.strip().lower()

    # No recognised prefix: the whole string is a sentence-transformers model.
    # `intfloat/e5-small-v2` has no colon; a bare `local` has no rest.
    if head_l not in BACKENDS:
        return "local", {"model": text}
    if not rest:
        return head_l, {}

    if head_l == "openai-compat":
        model, sep, base_url = rest.rpartition("@")
        if not sep:
            raise ValueError(
                f"openai-compat needs a base URL: "
                f"`openai-compat:<model>@<base_url>`, got {spec!r}"
            )
        return head_l, {"model": model.strip(), "base_url": base_url.strip()}

    return head_l, {"model": rest.strip()}


def get_embedder(
    backend: str | None, *, degrade: bool = False
) -> Embedder | None:
    """Build the embedder named by *backend*, or ``None`` for lexical search.

    ``None`` is returned when the spec explicitly asks for no embeddings, and
    when a backend is *not configured* — a missing optional dependency, an absent
    key. Both are reasonable states with a reasonable fallback.

    **A backend that is configured and broken raises.** A wrong model name or an
    expired key used to return ``None`` here, which meant a user got lexical
    search, slightly worse answers, and no indication anywhere that the thing
    they configured had never once worked. That is the failure mode this
    framework spends most of its validation budget eliminating elsewhere.

    ``degrade=True`` restores the old never-raises behaviour for a caller who
    genuinely wants it — a background re-index that should limp rather than stop.
    """
    name = (backend or "none").strip()
    if name.lower() in _OFF:
        return None

    try:
        backend_id, kwargs = parse_spec(name)
    except ValueError:
        if degrade:
            log.warning("embeddings spec %r is malformed — using lexical search", backend)
            return None
        raise

    cls = BACKENDS[backend_id]
    try:
        return cls(**kwargs)
    except EmbedderUnavailable as e:
        # Not configured. Degrading is the documented, sensible answer.
        log.warning("embeddings backend %r unavailable: %s", backend, e)
        return None
    except EmbeddingError:
        if degrade:
            log.warning("embeddings backend %r is broken — using lexical search", backend)
            return None
        raise
