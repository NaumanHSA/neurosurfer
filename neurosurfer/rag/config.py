from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Literal
from neurosurfer.vectorstores.base import Doc
from pydantic import Field
from pydantic_settings import BaseSettings
import os


class ChunkerConfig(BaseSettings):
    """
    Configuration for the GenericCodeChunker.
    Controls chunking sizes, overlaps, and fallback behavior
    for both line-based (code-friendly) and char-based (generic text) splitting.
    """
    # ----- Line-based chunking (good for code) -----
    fallback_chunk_size: int = Field(default=25)
    """Number of lines per chunk when using line-based splitting."""

    overlap_lines: int = Field(default=3)
    """Number of overlapping lines between consecutive line-based chunks to preserve context."""

    max_chunk_lines: int = Field(default=1000)
    """Hard safety cap on lines per chunk (prevents giant chunks for huge files)."""

    comment_block_threshold: int = Field(default=4)
    """Minimum number of consecutive comment-only lines to treat as a 'comment block'
    for possible filtering/skipping during chunking."""

    # ----- Character-based chunking (good for prose / unknown formats) -----
    char_chunk_size: int = Field(default=1000)
    """Number of characters per chunk when using char-based splitting."""

    char_overlap: int = Field(default=150)
    """Number of overlapping characters between consecutive char-based chunks."""

    # ----- Special-case formats -----
    readme_max_lines: int = Field(default=30)
    """Max number of lines per chunk for README/Markdown files."""

    json_chunk_size: int = Field(default=1000)
    """Max number of characters per JSON chunk (used when pretty-printing large JSONs)."""

    # ----- Fallback policy -----
    fallback_mode: str = Field(default="char")
    """What to do when file extension has no registered strategy:
       - "char": Always fall back to character-based chunking.
       - "line": Always fall back to line-based chunking.
       - "auto": Detect if content looks like code → line-based; else → char-based."""

    # Safety caps for custom handlers
    max_returned_chunks: int = Field(default=500)
    """Hard limit on number of chunks a handler may return (post-sanitize)."""
    max_total_output_chars: int = Field(default=1_000_000)
    """Hard limit on total chars across all chunks (post-sanitize)."""
    min_chunk_non_ws_chars: int = Field(default=1)
    """Drop chunks that have fewer than this many non-whitespace characters."""


RetrievalScope = Literal["small", "medium", "wide", "full"]
AnswerBreadth = Literal["single_fact", "short_list", "long_list", "aggregation", "summary"]
RetrievalMode = Literal["classic", "smart"]
RETRIEVAL_SCOPE_BASE_K: Dict[str, int] = {
    "small": 5,
    "medium": 10,
    "wide": 20,
    "full": 40,
}
ANSWER_BREADTH_MULTIPLIER: Dict[str, int] = {
    "single_fact": 1,
    "short_list": 2,
    "long_list": 4,
    "aggregation": 3,
    "summary": 3,
}

@dataclass
class RetrievalPlan:
    """
    Plan for how much to retrieve and how to shape the context.

    This is intentionally simple and can be created by:
      - RAGGate (UI layer) or
      - RAGAgent itself via an internal LLM call.
    """
    mode: RetrievalMode = "classic"
    scope: Optional[RetrievalScope] = None
    answer_breadth: Optional[AnswerBreadth] = None
    optimized_query: Optional[str] = None
    top_k: Optional[int] = None
    notes: Optional[str] = None
    extra: Dict[str, Any] = None

@dataclass
class RAGIngestorConfig:
    batch_size: int = 64
    max_workers: int = max(4, os.cpu_count() or 4)
    deduplicate: bool = True
    normalize_embeddings: bool = True
    default_metadata: Optional[Dict[str, Any]] = None
    tmp_dir: Optional[str] = "./tmp"

@dataclass
class RAGAgentConfig:
    # default embedding model; only used if no embedder is provided
    embedding_model: Optional[str] = "intfloat/e5-small-v2"           # Embedding model to use for RAG

    # vectorstore
    persist_directory: Optional[str] = "./rag-storage"                # Directory to persist vectorstore
    collection_name: Optional[str] = "neurosurfer-rag-agent"          # Name of the vectorstore collection
    clear_collection_on_init: Optional[bool] = True                   # Whether to clear the collection on init
    top_k: int = 5                                                     # Number of chunks to return from vectorstore
    similarity_threshold: Optional[float] = None                       # Optional similarity threshold for retrieval

    # Output budgeting
    fixed_max_new_tokens: Optional[int] = None                         # Fixed max new tokens for output
    auto_output_ratio: float = 0.25                                    # Auto output ratio
    min_output_tokens: int = 64                                        # Minimum output tokens
    safety_margin_tokens: int = 128                                    # Safety margin tokens

    # Context formatting
    include_metadata_in_context: bool = True                           # Whether to include metadata in context
    context_separator: str = "\n\n---\n\n"                              # Separator for context items
    context_item_header_fmt: str = "Source: {source}"                   # Format for context item headers
    normalize_embeddings: bool = True                                  # Whether to normalize embeddings

    # Tokenizer fallbacks (for OpenAI-style or unknown tokenizers)
    # Approx: ~4 chars/token (very rough), tune if you prefer 3.5–4.5
    approx_chars_per_token: float = 4.0

    # Number of tokens to reserve for system + user + history when trimming db context
    prompt_token_buffer: int = 500                                     # Number of tokens to reserve for system + user + history

    # Generation
    temperature: float = 0.7                                           # LLM sampling temperature
    max_new_tokens: int = 8192                                         # Default token budget for generation
