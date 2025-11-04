from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from neurosurfer.vectorstores.base import Doc


@dataclass
class RetrieveResult:
    base_system_prompt: str
    base_user_prompt: str                 # template BEFORE injecting context
    context: str                          # trimmed context
    max_new_tokens: int                   # dynamic value after budget calc
    base_tokens: int                      # tokens for system+history+user (no ctx)
    context_tokens_used: int              # tokens used by trimmed context
    token_budget: int                     # model window
    generation_budget: int                # remaining tokens for output
    docs: List[Doc] = field(default_factory=list)
    distances: List[Optional[float]] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)  # debug/trace info


@dataclass
class RAGAgentConfig:
    # Retrieval
    top_k: int = 5
    similarity_threshold: Optional[float] = None

    # Output budgeting
    fixed_max_new_tokens: Optional[int] = None
    auto_output_ratio: float = 0.25
    min_output_tokens: int = 32
    safety_margin_tokens: int = 32

    # Context formatting
    include_metadata_in_context: bool = True
    context_separator: str = "\n\n---\n\n"
    context_item_header_fmt: str = "Source: {source}"
    normalize_embeddings: bool = True

    # Tokenizer fallbacks (for OpenAI-style or unknown tokenizers)
    # Approx: ~4 chars/token (very rough), tune if you prefer 3.5–4.5
    approx_chars_per_token: float = 4.0
