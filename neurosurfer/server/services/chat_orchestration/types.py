from dataclasses import dataclass, field
from typing import List, Optional, Literal, Any, Dict, Generator
from neurosurfer.tracing import TraceResult

from .code_context_service import CodeAgentContextResult
from .rag_context_service import RAGContextResult

RouteType = Literal["direct", "rag", "code", "clarify"]

@dataclass
class GateDecision:
    route: RouteType
    optimized_query: str
    use_files: List[str]
    clarification_question: Optional[str]
    reason: str
    raw_response: str  # for debugging/logging

    def pretty_str(self) -> str:
        return f"Route: {self.route}\n" \
               f"Optimized Query: {self.optimized_query}\n" \
               f"Use Files: {self.use_files}\n" \
               f"Clarification Question: {self.clarification_question}\n" \
               f"Reason: {self.reason}\n"        

@dataclass
class MainWorkflowResult:
    """
    High-level result of a single user query through the workflow.

    - route: which path was taken ("direct" | "rag" | "code" | "clarify").
    - gate_decision: full decision from GateLLM.
    - final_answer_stream: generator of text chunks (if streaming).
    - final_answer_text: full answer if non-streaming.
    - rag_context: RAGContextResult if RAG was used.
    - code_context: CodeAgentContextResult if code agent was used.
    """
    route: Literal["direct", "rag", "code", "clarify"]
    gate_decision: GateDecision

    final_answer_stream: Optional[Generator[str, None, None]] = None
    final_answer_text: Optional[str] = None

    rag_context: Optional[RAGContextResult] = None
    code_context: Optional[CodeAgentContextResult] = None

    # Convenience flag
    needs_clarification: bool = False

    # Arbitrary extras for telemetry / debugging
    extras: Dict[str, Any] = field(default_factory=dict)

    # Traces for debugging / self-repair
    traces: Optional[TraceResult] = None