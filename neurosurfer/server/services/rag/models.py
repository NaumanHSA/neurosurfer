# neurosurfer/services/rag/models.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class RAGResult:
    """
    Outcome of a RAG orchestration step.
    """
    used: bool
    augmented_query: str
    meta: Dict[str, Any]


@dataclass
class GateDecision:
    """
    Result from the RAG routing (gate) LLM.
    """
    rag: bool
    related_files: List[str]
    optimized_query: Optional[str] = None
    raw_response: Optional[str] = None
    reason: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GateDecision":
        return cls(
            rag=bool(data.get("rag", False)),
            related_files=list(data.get("related_files") or []),
            optimized_query=data.get("optimized_query"),
            raw_response=data.get("raw_response"),
            reason=data.get("reason"),
        )
