from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Literal
from neurosurfer.vectorstores.base import Doc
import os

@dataclass
class RetrieveResult:
    context: str                          # trimmed context
    max_new_tokens: int                   # dynamic value after budget calc
    base_tokens: int                      # tokens for system+history+user (no ctx)
    context_tokens_used: int              # tokens used by trimmed context
    token_budget: int                     # model window
    generation_budget: int                # remaining tokens for output
    docs: List[Doc] = field(default_factory=list)
    distances: List[Optional[float]] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)  # debug/trace info


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

    # LLM parameters
    temperature: float = 0.5
