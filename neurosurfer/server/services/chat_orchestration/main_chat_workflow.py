from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, Optional, List, Literal
import json

from neurosurfer.models.chat_models.base import BaseChatModel
from neurosurfer.tools import Toolkit
from neurosurfer.server.utils import build_files_context
from neurosurfer.server.services.rag.orchestrator import RAGOrchestrator
from neurosurfer.agents.code.agent import CodeAgentConfig
from neurosurfer.tracing import Tracer, TracerConfig, TraceStepContext
from neurosurfer.agents.common.utils import normalize_response, rprint

from .code_context_service import CodeAgentService, CodeAgentContextResult
from .rag_context_service import RAGService, RAGContextResult
from .gate import GateLLM, GateDecision
from .final_answer_generator import FinalAnswerGenerator
from .config import MainWorkflowConfig
from .types import MainWorkflowResult


LOGGER = logging.getLogger(__name__)


class MainChatWorkflow:
    """
    Orchestrates the full chat workflow for a single user query:

    1) Ingest new files (if any) via RAGOrchestrator.
    2) Build file context & summaries.
    3) Use GateLLM to decide route: "code" | "rag" | "direct" (or "clarify").
    4) Run the chosen sub-pipeline:
       - CodeAgentService -> code context_block
       - RAGService       -> rag context_block
       - Direct           -> generic context_block
    5) Call FinalAnswerGenerator to produce the user-facing answer.

    This class is intentionally *not* an Agent. It's a deterministic pipeline
    with one LLM routing call up front and one LLM final-answer call at the end.
    """

    def __init__(
        self,
        *,
        reasoning_llm: BaseChatModel,      # for GateLLM
        final_answer_llm: BaseChatModel,   # for FinalAnswerGenerator
        rag_orchestrator: RAGOrchestrator,
        config: Optional[MainWorkflowConfig] = None,
        code_agent_config: Optional[CodeAgentConfig] = None,
        tracer: Optional[Tracer] = None,
        log_traces: Optional[bool] = True,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.config = config or MainWorkflowConfig()
        self.logger = logger or LOGGER
        self.log_traces = log_traces
        # Base tracer that actually records and log steps (RichTracer by default).
        self.tracer: Tracer = tracer or Tracer(
            config=TracerConfig(log_steps=self.log_traces),
            meta={
                "agent_type": "MainChatAgent",
                "agent_config": self.config,
                "model": final_answer_llm.model_name,
                "toolkit": False,
                "log_steps": self.log_traces,
            },
            logger_=logger,
        )

        # Router (LLM gate)
        self.gate = GateLLM(
            llm=reasoning_llm,
            temperature=config.temperature,
            max_new_tokens=config.max_new_tokens,
            tracer=self.tracer,
            log_traces=self.log_traces,
            logger=self.logger,
        )

        # RAG & Code services
        self.rag_service = RAGService(
            rag_orchestrator=rag_orchestrator,
            logger=self.logger,
            log_traces=self.log_traces,
            tracer=self.tracer,
        )
        self.code_service = CodeAgentService(
            llm=reasoning_llm,
            config=code_agent_config,
            tracer=self.tracer,
            log_traces=self.log_traces,
            logger=self.logger,
        )

        # Final answer generator
        self.final_answer = FinalAnswerGenerator(
            llm=final_answer_llm,
            default_language=self.config.default_language,
            default_answer_length=self.config.default_answer_length,
            max_context_chars=self.config.max_context_chars,
            temperature=self.config.temperature,
            max_new_tokens=self.config.max_new_tokens,
            base_system_instructions=None,
            logger=self.logger,
        )

    def run(
        self,
        *,
        user_id: int,
        thread_id: int,
        message_id: int,
        user_query: str,
        has_files_message: bool,
        chat_history: Optional[List[Dict[str, Any]]] = None,
        stream: bool = True,
        reset_tracer: bool = True
    ) -> MainWorkflowResult:
        """
        Run the full workflow for a single user query.

        - user_id / thread_id / message_id: DB identifiers.
        - user_query: raw user text.
        - has_files_message: whether this specific message included file uploads.
        - chat_history_block: plain-text representation of prior conversation
          (user+assistant) for the gate + final answer to see.
        - stream: if True, returns a streaming generator; else returns a full string.
        """
        if reset_tracer:
            self.tracer.reset()
        
        # parse chat history block into a string
        chat_history_block = ""
        if chat_history:
            chat_history_block = self._chat_history_to_string(chat_history)
            print(f"\nChat History:\n{chat_history_block}\n\n")

        rprint("🧠 Thinking...", color="yellow")
        if self.log_traces:
            rprint(f"\n\\[{self.config.id}] Tracing Start!")

        main_tracer = self.tracer(
            agent_id=self.config.id,
            kind="main_chat_workflow",
            label="main_chat_workflow.run",
            inputs={
                "agent_type": type(self).__name__,
                "response_type": "streaming" if stream else "non-streaming",
                "has_files_message": has_files_message,
            },
        ).start()

        # Ingest files (if any) and build file context
        ingestion_summaries, files, files_summaries_block = self._ingest_and_build_files(
            user_id=user_id,
            thread_id=thread_id,
            message_id=message_id,
            has_files_message=has_files_message,
        )
        # print("ingestion_summaries:", ingestion_summaries)
        if ingestion_summaries:
            main_tracer.log(message=f"Files ingested for user_id={user_id} thread_id={thread_id}: {len(files)} files", type="info")
        files_context = build_files_context(user_id=user_id, thread_id=thread_id)

        # Route with GateLLM
        gate_decision = self.gate.decide(
            user_query=user_query,
            files_summaries_block=files_summaries_block,
            chat_history_block=chat_history_block,
        )
        needs_clarification = gate_decision.needs_clarification
        main_tracer.log(message=gate_decision.pretty_str(), type="thought", type_keyword=False)

        # If the gate says we need clarification, short-circuit.
        if needs_clarification:
            main_tracer.log(message="More clarification is needed on the query from the user", type="warning")
            result = MainWorkflowResult(
                route="clarify",
                gate_decision=gate_decision,
                needs_clarification=True,
                final_answer_text="Please clarify your request.",
                traces=self.tracer.results,
            )
            # For streaming case, wrap the clarification in a trivial generator
            if stream:
                result.final_answer_stream = self._single_chunk_stream("Please clarify your request.")
            main_tracer.close()
            return result

        # Execute chosen pipeline to obtain `context_block`
        route = gate_decision.route or "direct"
        context_block = ""
        rag_context: Optional[RAGContextResult] = None
        code_context: Optional[CodeAgentContextResult] = None

        optimized_query = user_query
        if gate_decision.optimized_query:
            optimized_query += f"\nOptimized Query: {gate_decision.optimized_query}\n"

        if route == "code" and self.config.enable_code:
            """Call CodeAgentService to build a context block."""
            main_tracer.log(message="Running code pipeline", type="info")
            files_hint = gate_decision.use_files or []
            code_context: CodeAgentContextResult = self.code_service.run_for_context(
                query=optimized_query,
                files_context=files_context,
                files_hint=files_hint or None,
                workdir=None,
                post_process="none",
                stream=stream,
                reset_tracer=False,
            )
            context_block = code_context.context_block
            main_tracer.log(message=f"Code pipeline completed with retrieved context of size {len(context_block)} chars.", type="info")

        elif route == "rag" and self.config.enable_rag:
            """Call RAGService to build a context block."""
            main_tracer.log(message="Running RAG pipeline", type="info")
            rag_context = self.rag_service.retrieve_for_context(
                user_query=optimized_query,
                user_id=user_id,
                thread_id=thread_id,
                stream=stream,
            )
            context_block = rag_context.context_block
            main_tracer.log(message=f"RAG pipeline completed with retrieved context of size {len(context_block)} chars.", type="info")
            
        # Optional: truncate context to avoid huge prompts
        if self.config.max_context_chars > 0 and len(context_block) > self.config.max_context_chars:
            context_block = (
                context_block[: self.config.max_context_chars] + "...\n[Context truncated due to length; most recent / most relevant parts kept.]"
            )

        # Final answer generation (streaming or not)
        if stream:
            main_tracer.log(message="Generating final answer (streaming)", type="info")
            answer_stream = self.final_answer.generate(
                user_query=optimized_query,
                route=route,
                context_block=context_block,
                files_summaries_block=files_summaries_block,
                chat_history_block=chat_history_block,
                target_language=self.config.default_language,
                answer_length=self.config.default_answer_length,
                route_reason=gate_decision.reason,
                extra_instructions="",
            )
            final_text = None
        else:
            main_tracer.log(message="Generating final answer (non-streaming)", type="info")
            chunks: List[str] = list(
                self.final_answer.generate(
                    user_query=optimized_query,
                    context_block=context_block,
                    files_summaries_block=files_summaries_block,
                    chat_history_block=chat_history_block,
                    target_language=self.config.default_language,
                    answer_length=self.config.default_answer_length,
                    extra_instructions="",
                )
            )
            final_text = "".join(chunks)
            answer_stream = None
        
        main_tracer.close()
        return MainWorkflowResult(
            route=route,  # "code" | "rag" | "direct"
            gate_decision=gate_decision,
            final_answer_stream=answer_stream,
            final_answer_text=final_text,
            rag_context=rag_context,
            code_context=code_context,
            extras={
                "files_ingested": len(files_context),
                "ingestion_summaries": ingestion_summaries,
            },
            traces=self.tracer.results,
        )

    def _ingest_and_build_files(
        self,
        user_id: int,
        thread_id: int,
        message_id: int,
        has_files_message: bool,
    ):
        """
        Wrapper around RAGOrchestrator.ingest(...) to keep workflow code clean.
        Returns:
            ingestion_summaries, files, files_summaries_block
        """
        ingestion_summaries, files, files_summaries_block = self._rag_orchestrator.ingest(
            user_id=user_id,
            thread_id=thread_id,
            message_id=message_id,
            has_files_message=has_files_message,
        )
        return ingestion_summaries, files, files_summaries_block

    @property
    def _rag_orchestrator(self) -> RAGOrchestrator:
        # Small helper so type checkers are happy.
        return self.rag_service.rag_orchestrator

    @staticmethod
    def _single_chunk_stream(text: str) -> Generator[str, None, None]:
        """Wrap a single text value into a generator, for unified streaming API."""
        yield text

    def _chat_history_to_string(self, chat_history: List[Dict[str, Any]]) -> str:
        """Convert chat history into a compact, readable string"""
        limit_assistant = 4000
        limit_user = 4000
        limit_fallback = 4000
        def _clip(text: str, limit: int) -> str:
            text = (text or "").strip().replace("\r\n", "\n")
            if len(text) <= limit:
                return text
            return text[:limit].rstrip() + "..."

        lines: List[str] = []
        for message in chat_history or []:
            role = (message.get("role") or "unknown").strip().lower()
            if role == "system":
                continue

            content = message.get("content", "")
            if role == "assistant":
                content = _clip(content, limit_assistant)
            elif role == "user":
                content = _clip(content, limit_user)
            else:
                content = _clip(content, limit_fallback)  # fallback for tool/other roles
            if content:
                lines.append(f"{role.upper()}: {content}")
        return "\n".join(lines)

    def _reduce_files_summaries_block(self, files_summaries_block: str, use_files: List[str]) -> str:
        """Reduces the files summaries block to only the files that are used in the gate decision."""
        try:
            b: Dict[str, Any] = json.loads(files_summaries_block) or {}
            for key in list(b.keys()):
                if key not in use_files:
                    del b[key]
            return json.dumps(b, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to reduce files summaries block: {e}")
        return files_summaries_block