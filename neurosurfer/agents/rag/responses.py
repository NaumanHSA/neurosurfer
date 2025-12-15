from __future__ import annotations
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Literal, Union
from neurosurfer.vectorstores.base import Doc
from pydantic import BaseModel as PydModel

from neurosurfer.tracing import TraceResult
from neurosurfer.agents.agent.responses import AgentResponse


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


@dataclass
class RAGAgentResponse:
    rag_retrieval: RetrieveResult
    agent_response: AgentResponse
