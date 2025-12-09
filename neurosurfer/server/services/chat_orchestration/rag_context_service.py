from __future__ import annotations

import logging
import json
from dataclasses import dataclass
from typing import Any, Dict, Optional
from neurosurfer.tracing import Tracer, TracerConfig
from neurosurfer.server.services.rag.orchestrator import RAGOrchestrator, RAGResult

LOGGER = logging.getLogger(__name__)


@dataclass
class RAGContextResult:
    """
    High-level result from a RAG retrieval run.

    - context_block: text to feed to FinalAnswerGenerator.
    - rag_used: whether gate decided to use RAG.
    - gate_meta: full decision from gate (rag, related_files, scope, etc.).
    - collection: collection name in vector store (if any).
    - retrieval_meta: token usage and other meta.
    """
    context_block: str
    rag_used: bool
    gate_meta: Dict[str, Any]
    collection: Optional[str]
    retrieval_meta: Dict[str, Any]


class RAGService:
    """
    Service façade around RAGOrchestrator.

    Exposes a simple `.retrieve_for_context(...)` that returns a
    RAGContextResult ready for FinalAnswerGenerator.
    """

    def __init__(
        self,
        rag_orchestrator: RAGOrchestrator,
        *,
        tracer: Optional[Tracer] = None,
        log_traces: Optional[bool] = True,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.rag_orchestrator = rag_orchestrator
        self.logger = logger or LOGGER
        self.log_traces = log_traces
        # Base tracer that actually records and log steps (RichTracer by default).
        self.tracer: Tracer = tracer or Tracer(
            config=TracerConfig(log_steps=self.log_traces),
            meta={
                "agent_type": "RAGService",
                "model": rag_orchestrator.gate_llm.model_name if rag_orchestrator.gate_llm else "unknown",
                "log_steps": self.log_traces,
            },
            logger_=logger,
        )

    def retrieve_for_context(
        self,
        *,
        user_query: str,
        user_id: int,
        thread_id: int,
        stream: bool = False,
    ) -> RAGContextResult:
        if not user_id or not thread_id:
            ctx = "[RAG] No user_id/thread_id provided; skipping retrieval."
            return RAGContextResult(
                context_block=ctx,
                rag_used=False,
                gate_meta={"reason": "missing_ids"},
                collection=None,
                retrieval_meta={},
            )

        self.logger.info(
            "[RAGService] Retrieving context for user_id=%s thread_id=%s query=%r",
            user_id,
            thread_id,
            user_query,
        )

        result: RAGResult = self.rag_orchestrator.retrieve(
            user_id=user_id,
            thread_id=thread_id,
            user_query=user_query,
            files=[],   # files already ingested; we're just retrieving
        )

        gate_meta = result.meta or {}
        rag_used = bool(gate_meta.get("rag", True))

        # result.context is usually a text block with retrieved chunks
        rag_context_text = result.context or ""
        augmented_query = result.augmented_query or user_query

        context_block = self._build_context_block(
            user_query=user_query,
            augmented_query=augmented_query,
            rag_context=rag_context_text,
            gate_meta=gate_meta,
        )

        retrieval_meta = {
            "base_tokens": gate_meta.get("base_tokens"),
            "context_tokens_used": gate_meta.get("context_tokens_used"),
            "token_budget": gate_meta.get("token_budget"),
        }

        collection = gate_meta.get("collection")

        return RAGContextResult(
            context_block=context_block,
            rag_used=rag_used,
            gate_meta=gate_meta,
            collection=collection,
            retrieval_meta=retrieval_meta,
        )

    # ---------- Internals ----------

    def _build_context_block(
        self,
        *,
        user_query: str,
        augmented_query: str,
        rag_context: str,
        gate_meta: Dict[str, Any],
    ) -> str:
        """
        Construct a context block that clearly marks RAG content and
        optionally includes gate decisions for the final generator.
        """
        parts = []

        parts.append("RAG ROUTER DECISION (for debugging / clarity):")
        try:
            pretty_gate = json.dumps(gate_meta, indent=2, default=str)
        except Exception:
            pretty_gate = str(gate_meta)
        parts.append(pretty_gate)
        parts.append("")

        parts.append("USER QUERY:")
        parts.append(user_query.strip())
        parts.append("")

        parts.append("AUGMENTED QUERY USED FOR RETRIEVAL:")
        parts.append(augmented_query.strip())
        parts.append("")

        if rag_context.strip():
            parts.append("[RAG CONTEXT START]")
            parts.append(rag_context.strip())
            parts.append("[RAG CONTEXT END]")
        else:
            parts.append("[RAG CONTEXT] (no context retrieved)")

        return "\n".join(parts)
