# neurosurfer/services/rag/orchestrator.py
from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy.orm import Session

from neurosurfer.agents.rag.agent import RAGAgent, RAGAgentConfig, RAGIngestorConfig
from neurosurfer.models.embedders.base import BaseEmbedder
from neurosurfer.models.chat_models.base import BaseChatModel

from neurosurfer.server.services.rag.models import RAGResult
from neurosurfer.server.services.rag.summarizer import FileSummarizer
from neurosurfer.server.services.rag.ingestor import FileIngestor
from neurosurfer.server.services.rag.gate import RAGGate
from neurosurfer.server.db.models import ChatThread, Message, NSFile, User
from neurosurfer.server.db.db import SessionLocal
from neurosurfer.server.services.rag.metadata_filter import build_metadata_filter_from_related_files
from neurosurfer.server.config import APP_DATA_PATH

import logging

LOGGER = logging.getLogger(__name__)


class RAGOrchestrator:
    """
    Thin coordinator that wires:
      - File ingestion (SQL + vectorstore)
      - File summaries
      - RAG routing (gate LLM)
      - Retrieval via RAGAgent
    """

    _ALLOWED_NAME = re.compile(r"[^a-zA-Z0-9._-]+")

    def __init__(
        self,
        *,
        embedder: BaseEmbedder,
        gate_llm: Optional[BaseChatModel] = None,
        max_context_tokens: int = 1024,
        top_k: int = 8,
        similarity_threshold: Optional[float] = None,
        verbose: bool = False,  # Log more details
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.logger = logger or LOGGER
        # self.verbose = verbose
        self.verbose = True
        self.persist_directory = os.path.join(APP_DATA_PATH, "rag-storage")
        os.makedirs(self.persist_directory, exist_ok=True)
        # Core RAG agent (no LLM here; main LLM lives in chat handler)

        self.rag_config = RAGAgentConfig(
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            fixed_max_new_tokens=max_context_tokens,
            clear_collection_on_init=True,
            persist_directory=self.persist_directory,
        )
        self.rag_agent = RAGAgent(
            llm=gate_llm,
            embedder=embedder,
            config=self.rag_config,
            ingestor_config=RAGIngestorConfig(),
            logger=self.logger,
        )
        self.logger.info(f"Number of documents left: {self.rag_agent.vectorstore.count()}")

        # Sub-services
        # upload_root = os.path.join(self.persist_directory, "uploads")
        summarizer = FileSummarizer(rag_agent=self.rag_agent, llm=gate_llm, verbose=verbose)
        self.ingestion = FileIngestor(
            rag_agent=self.rag_agent,
            summarizer=summarizer,
            verbose=verbose,
        )
        self.gate = RAGGate(
            llm=gate_llm,
            verbose=verbose,
            logger=self.logger,
            temperature=self.rag_config.temperature,
        )
        self.top_k = top_k


    # -------- public API --------
    def retrieve(
        self,
        *,
        user_id: int,
        thread_id: int,
        user_query: str,
        files: List[Dict[str, Any]],
    ) -> RAGResult:
        """
        Main entry point.
        - user_id, thread_id: scope for both SQL and vectorstore collection
        - user_query: latest user message text
        - files: base64-encoded upload metadata from request (if any)
        """
        if not user_id or not thread_id:
            return RAGResult(used=False, augmented_query=user_query, meta={"used": False, "reason": "no_files"})
        
        # switch the path to the user's thread
        self._set_path(user_id, thread_id)
        collection_name = self._collection_for(user_id, thread_id)
        self.rag_agent.set_collection(collection_name)

        # db: SQLAlchemy Session (injected from FastAPI)
        db = SessionLocal()
        # Check if this thread has files
        has_files_thread = (
            db.query(NSFile)
            .filter(
                NSFile.user_id == user_id,
                NSFile.thread_id == thread_id,
                NSFile.collection == collection_name,
                NSFile.ingested == True,
            )
            .count()
            > 0
        )
        if not has_files_thread:
            return RAGResult(
                used=False,
                augmented_query=user_query,
                meta={"used": False, "reason": "no_files"},
            )

        # Gate: decide if query is about files & which ones
        decision = self.gate.decide(
            db,
            user_id=user_id,
            thread_id=thread_id,
            collection=collection_name,
            user_query=user_query,
            message_files=files,
        )
        if not decision.rag:
            return RAGResult(
                used=False,
                augmented_query=decision.optimized_query,
                meta={"used": False, "reason": decision.reason or "gate_false"},
            )
        metadata_filter = build_metadata_filter_from_related_files(
            db,
            user_id=user_id,
            thread_id=thread_id,
            collection=collection_name,
            related_files=decision.related_files,
        )
        if self.verbose:
            self.logger.info(f"Related files for thread {user_id}/{thread_id}: {decision.related_files}")
            self.logger.info(f"Metadata filter: {metadata_filter}")
            self.logger.info(f"Optimized query: {decision.optimized_query}")
            self.logger.info(f"Retrieval scope: {decision.retrieval_scope}")
            self.logger.info(f"Answer breadth: {decision.answer_breadth}")
            self.logger.info(f"Gate reason: {decision.reason}")

        # Retrieve context
        retrieved = self.rag_agent.retrieve(
            user_query=decision.optimized_query,
            top_k=self.top_k,
            metadata_filter=metadata_filter,
            similarity_threshold=None,
            retrieval_mode="smart",
            retrieval_scope=decision.retrieval_scope,
            answer_breadth=decision.answer_breadth,
        )
        ctx_text = (retrieved.context or "").strip()
        
        if self.verbose:
            self.logger.info(f"User Query: {decision.optimized_query}")
            self.logger.info(f"Documents in collection: {self.rag_agent._get_collection_docs_count()}")
            self.logger.info(f"Retrieved context: {len(ctx_text)}")

        if not ctx_text:
            return RAGResult(
                used=False,
                augmented_query=decision.optimized_query,
                meta={
                    "used": False,
                    "reason": "empty_context",
                    "gate": decision.__dict__,
                },
            )
        augmented = f"{decision.optimized_query}\n\n[CONTEXT]\n{ctx_text}\n[/CONTEXT]"
        # self.logger.info(f"\n\nFINAL USER PROMPT:\n{augmented}\n\n")
        db.close()
        return RAGResult(
            used=True,
            context=ctx_text,
            augmented_query=augmented,
            meta={
                "used": True,
                "reason": "ok",
                "gate": decision.__dict__,
            },
        )


    # -------- public API --------
    def ingest(
        self,
        *,
        user_id: int,
        thread_id: int,
        message_id: int,
        has_files_message: bool = False,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Main entry point.
        - user_id, thread_id: scope for both SQL and vectorstore collection
        - message_id: latest user message text
        - has_files_message: whether the message contains files
        """
        summaries, files = [], []
        if not user_id or not thread_id:
            return summaries, files
        
        # switch the path to the user's thread
        self._set_path(user_id, thread_id)
        collection_name = self._collection_for(user_id, thread_id)
        self.rag_agent.set_collection(collection_name)

        # db: SQLAlchemy Session (injected from FastAPI)
        db = SessionLocal()
        # Ingest files if provided
        if has_files_message and message_id:
            files = (
                db.query(NSFile)
                .filter(
                    NSFile.user_id == user_id,
                    NSFile.thread_id == thread_id,
                    NSFile.message_id == message_id,
                    NSFile.collection == collection_name,
                    NSFile.ingested == False,
                )
                .all()
            )
            if files:
                if self.verbose:
                    self.logger.info(f"Ingesting files for thread {user_id}/{thread_id}, num files: {len(files)}")

                summaries = self.ingestion.ingest_files(
                    db,
                    user_id=user_id,
                    thread_id=thread_id,
                    collection_name=collection_name,
                    files=files,
                )
                if self.verbose:
                    self.logger.info(f"Ingested files for thread {user_id}/{thread_id}")
                    if summaries:
                        self.logger.info(f"Ingested files summaries:")
                        for summary in summaries:
                            self.logger.info(summary)
        return summaries, files
        
    # -------- public API --------
    def apply(
        self,
        *,
        user_id: int,
        thread_id: int,
        user_query: str,
        message_id: int,
        has_files_message: bool = False,
    ) -> RAGResult:
        """
        Main entry point.
        - user_id, thread_id: scope for both SQL and vectorstore collection
        - user_query: latest user message text
        - files: base64-encoded upload metadata from request (if any)
        """
        # ingest files if any
        summaries, files = self.ingest(
            user_id=user_id,
            thread_id=thread_id,
            message_id=message_id,
            has_files_message=has_files_message,
        )
        # retrieve context based on the user query
        return self.retrieve(
            user_id=user_id,
            thread_id=thread_id,
            user_query=user_query,
            files=files,
        )

    # -------- internals --------
    def _set_path(self, user_id: int, thread_id: int) -> None:
        from neurosurfer.server.utils import ApplicationPaths
        self.rag_agent.cfg.persist_directory = str(ApplicationPaths.rag_storage_path(user_id, thread_id))

    def _collection_for(self, user_id: int, thread_id: int) -> str:
        return self._safe_collection_name(f"ns_vdb_u{user_id}_t{thread_id}")

    def _safe_collection_name(self, s: str) -> str:
        s = self._ALLOWED_NAME.sub("_", s).strip("._-")
        return s or "ns_vdb_default"
