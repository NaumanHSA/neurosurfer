from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from neurosurfer.llm.types import CanonicalResponse
from neurosurfer.vectorstores.base import Doc


@dataclass
class RetrieveResult:
    context: str                          # trimmed context
    max_new_tokens: int                   # dynamic value after budget calc
    base_tokens: int                      # tokens for system+history+user (no ctx)
    context_tokens_used: int              # tokens used by trimmed context
    token_budget: int                     # model window
    generation_budget: int                # remaining tokens for output
    docs: list[Doc] = field(default_factory=list)
    distances: list[float | None] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)  # debug/trace info


@dataclass
class RAGAgentResponse:
    rag_retrieval: RetrieveResult
    agent_response: CanonicalResponse
