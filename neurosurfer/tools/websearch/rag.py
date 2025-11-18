# neurosurfer/tools/websearch/rag.py
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from neurosurfer.models.embedders import BaseEmbedder
from neurosurfer.models.embedders.sentence_transformer import SentenceTransformerEmbedder
from neurosurfer.models.chat_models import BaseChatModel
from .config import WebSearchRAGConfig

try:
    # Optional: for more accurate token-based trimming if available
    from neurosurfer.agents.rag.token_utils import TokenCounter  # type: ignore
except Exception:  # pragma: no cover
    TokenCounter = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


@dataclass
class WebSearchRAGResult:
    """
    Return payload from RAG refinement of web search results.
    """
    query: str
    context: str                            # trimmed text context for the LLM
    results: List[Dict[str, Any]]           # filtered + enriched results
    context_tokens_used: int
    meta: Dict[str, Any] = field(default_factory=dict)
    llm_summary: Optional[Any] = None       # whatever BaseChatModel.ask returns


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #

def _build_text_for_embedding(
    result: Dict[str, Any],
    cfg: WebSearchRAGConfig,
) -> str:
    """
    Build a single text snippet from a normalized web result suitable for
    embedding and retrieval.

    Expects keys like: title, url, snippet, content.
    """
    parts: List[str] = []

    title = result.get("title") or ""
    snippet = result.get("snippet") or ""
    content = result.get("content") or ""

    if cfg.include_title_in_text and title:
        parts.append(f"Title: {title}")

    if cfg.include_snippet_in_text and snippet:
        parts.append(f"Snippet: {snippet}")

    if content:
        parts.append(str(content))

    full = "\n\n".join(p for p in parts if p)

    if not full:
        return ""

    # Hard per-result trim (char-based, approximated by per_result_max_tokens)
    max_chars = cfg.per_result_max_tokens * cfg.approx_chars_per_token
    if len(full) > max_chars:
        full = full[:max_chars]

    return full


def _build_segment_text(
    rank: int,
    score: float,
    result: Dict[str, Any],
    text: str,
) -> str:
    """
    Human-readable segment for concatenated context.
    """
    title = result.get("title") or "Untitled"
    url = result.get("url") or ""
    snippet = result.get("snippet") or ""

    header_lines = [
        f"[{rank}] {title}",
        f"Score: {score:.4f}",
    ]
    if url:
        header_lines.append(f"URL: {url}")
    if snippet:
        header_lines.append(f"Snippet: {snippet}")

    header = "\n".join(header_lines)
    return f"{header}\n\n{text}"


def _cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """
    Simple cosine similarity implementation that works with list/tuple/ndarray-like
    vectors without requiring numpy as a hard dependency.
    """
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a, b):
        dot += x * y
        norm_a += x * x
        norm_b += y * y

    if norm_a <= 0.0 or norm_b <= 0.0:
        return 0.0

    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))


def _trim_context(
    context: str,
    cfg: WebSearchRAGConfig,
    llm: Optional[BaseChatModel],
) -> Tuple[str, int]:
    """
    Trim a long context string to fit within cfg.max_context_tokens.
    Uses TokenCounter if available and llm is provided; otherwise a simple
    char-based approximation.
    """
    if not context:
        return "", 0

    # Try to use TokenCounter if available and LLM is provided
    if TokenCounter is not None and llm is not None:
        try:
            counter = TokenCounter(llm, chars_per_token=cfg.approx_chars_per_token)
            trimmed, used = counter.trim_to_tokens(
                context,
                cfg.max_context_tokens,
            )
            return trimmed, used
        except Exception:
            logger.exception(
                "WebSearchRAG: TokenCounter trimming failed; "
                "falling back to char-based approximation."
            )

    # Fallback: approximate via length / approx_chars_per_token
    max_chars = cfg.max_context_tokens * cfg.approx_chars_per_token
    trimmed = context[:max_chars]
    used_tokens = len(trimmed) // cfg.approx_chars_per_token
    return trimmed, used_tokens


def _summarize_context_with_llm(
    llm: BaseChatModel,
    query: str,
    context: str,
    cfg: WebSearchRAGConfig,
) -> Any:
    """
    Simple question-focused summarization over the already-trimmed context.
    The exact return type depends on your BaseChatModel implementation.
    """
    system_prompt = (
        "You are a careful research assistant. You receive a user question and "
        "a set of extracted web snippets that have already been filtered for relevance. "
        "Synthesize a clear, concise answer grounded only in the provided information. "
        "If something is uncertain or not supported by the snippets, say so explicitly."
    )

    user_prompt = f"""
User question:
{query}

Web context:
\"\"\"{context}\"\"\"

Write a focused answer that cites concrete facts from the context. 
Do not invent information that is not supported by the snippets.
"""

    # BaseChatModel.ask signature should match what you use in RAGAgent
    return llm.ask(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        stream=False,
        max_new_tokens=cfg.summary_max_new_tokens,
        temperature=cfg.summary_temperature,
    )


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def run_web_rag(
    query: str,
    results: Sequence[Dict[str, Any]],
    *,
    llm: Optional[BaseChatModel] = None,
    embedder: Optional[BaseEmbedder] = None,
    config: Optional[WebSearchRAGConfig] = None,
) -> WebSearchRAGResult:
    """
    RAG-style refinement over normalized web search results.

    - Re-scores each result embedding vs. the query embedding.
    - Selects top_k results.
    - Builds a single concatenated context string and trims it to
      `max_context_tokens`.
    - Optionally runs an LLM summarization over that context.

    This is *ephemeral RAG* — no persistent vectorstore, just in-memory
    embeddings for the current tool call.
    """
    cfg = config or WebSearchRAGConfig()

    if embedder is None:
        logger.info(
            "WebSearchRAG: No embedder provided, "
            "initializing SentenceTransformerEmbedder(%s)",
            cfg.embedding_model,
        )
        embedder = SentenceTransformerEmbedder(cfg.embedding_model)

    # 1) Build candidate texts
    candidate_texts: List[str] = []
    candidate_results: List[Tuple[int, Dict[str, Any], str]] = []

    for idx, r in enumerate(results):
        text = _build_text_for_embedding(r, cfg)
        if not text:
            continue
        candidate_texts.append(text)
        candidate_results.append((idx, r, text))

    if not candidate_texts:
        logger.warning("WebSearchRAG: No usable content in search results.")
        return WebSearchRAGResult(
            query=query,
            context="",
            results=[],
            context_tokens_used=0,
            meta={"reason": "no_content"},
        )

    # 2) Embed query and documents
    q_emb = embedder.embed(
        query=[query],
        normalize_embeddings=cfg.normalize_embeddings,
    )[0]

    doc_embs = embedder.embed(
        query=candidate_texts,
        normalize_embeddings=cfg.normalize_embeddings,
    )

    # 3) Compute cosine scores and select top_k
    scored: List[Tuple[float, Tuple[int, Dict[str, Any], str]]] = []
    for emb, triple in zip(doc_embs, candidate_results):
        score = _cosine_similarity(q_emb, emb)
        scored.append((score, triple))

    scored.sort(key=lambda s: s[0], reverse=True)
    top = scored[: cfg.top_k]

    selected_results: List[Dict[str, Any]] = []
    segments: List[str] = []

    for rank, (score, (idx, r, text)) in enumerate(top, start=1):
        segment = _build_segment_text(rank, score, r, text)
        segments.append(segment)

        # Copy original result and enrich with RAG metadata + trimmed content
        new_r = dict(r)
        new_r["rag_score"] = float(score)
        new_r["rag_rank"] = rank
        # Overwrite content with the trimmed text we actually used
        new_r["content"] = text
        selected_results.append(new_r)

    untrimmed_context = "\n\n---\n\n".join(segments)

    # 4) Trim to context budget
    trimmed_context, used_tokens = _trim_context(
        untrimmed_context,
        cfg,
        llm=llm,
    )

    rag_result = WebSearchRAGResult(
        query=query,
        context=trimmed_context,
        results=selected_results,
        context_tokens_used=used_tokens,
        meta={
            "top_k": cfg.top_k,
            "max_context_tokens": cfg.max_context_tokens,
            "num_candidates": len(candidate_texts),
            "num_selected": len(selected_results),
        },
    )

    # 5) Optional LLM summarization over the RAG context
    if cfg.use_llm_summarization and llm is not None and trimmed_context:
        try:
            rag_result.llm_summary = _summarize_context_with_llm(
                llm=llm,
                query=query,
                context=trimmed_context,
                cfg=cfg,
            )
        except Exception:
            logger.exception("WebSearchRAG: LLM summarization failed.")

    return rag_result


def run_web_rag_on_results_dict(
    results_dict: Dict[str, Any],
    *,
    llm: Optional[BaseChatModel] = None,
    embedder: Optional[BaseEmbedder] = None,
    config: Optional[WebSearchRAGConfig] = None,
) -> WebSearchRAGResult:
    """
    Convenience wrapper when you already have the full `results_dict`
    produced by WebSearchTool.
    """
    query = results_dict.get("query", "")
    results = results_dict.get("results", []) or []
    return run_web_rag(
        query=query,
        results=results,
        llm=llm,
        embedder=embedder,
        config=config,
    )
