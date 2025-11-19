# neurosurfer/services/rag/orchestrator.py
from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from neurosurfer.agents.rag.agent import RAGAgent, RAGAgentConfig, RAGIngestorConfig
from neurosurfer.models.embedders.base import BaseEmbedder
from neurosurfer.models.chat_models.base import BaseChatModel

from neurosurfer.server.services.rag.models import RAGResult
from neurosurfer.server.services.rag.summarizer import FileSummarizer
from neurosurfer.server.services.rag.ingestor import FileIngestor
from neurosurfer.server.services.rag.gate import RAGGate
from neurosurfer.server.db.models import ChatThread, Message, NMFile, User
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
        self.verbose = verbose

        # REMOVE THIS IN PRODUCTION
        self._reset_db()

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
            llm=None,
            embedder=embedder,
            config=self.rag_config,
            ingestor_config=RAGIngestorConfig(),
            logger=self.logger,
        )

        # Sub-services
        upload_root = os.path.join(self.persist_directory, "uploads")
        summarizer = FileSummarizer(rag_agent=self.rag_agent, llm=gate_llm, verbose=verbose)
        self.ingestion = FileIngestor(
            rag_agent=self.rag_agent,
            summarizer=summarizer,
            upload_root=upload_root,
            verbose=verbose,
        )
        self.gate = RAGGate(
            llm=gate_llm,
            verbose=verbose,
            logger=self.logger,
            temperature=self.rag_config.temperature,
        )
        self.top_k = top_k

    def _reset_db(self):
        """Clear all chats and threads from the database. Only leave Users Information."""
        db = SessionLocal()
        db.query(ChatThread).delete()
        db.query(Message).delete()
        db.query(NMFile).delete()
        db.commit()

        # check if there are any users left
        self.logger.info(f"Number of users left: {db.query(User).count()}")
        self.logger.info(f"Number of files left: {db.query(NMFile).count()}")
        self.logger.info(f"Number of threads left: {db.query(ChatThread).count()}")
        self.logger.info(f"Number of messages left: {db.query(Message).count()}")
        db.close()


    # -------- public API --------
    def apply(
        self,
        *,
        actor_id: int,
        thread_id: int,
        user_query: str,
        files: Optional[List[Dict[str, Any]]] = None,
    ) -> RAGResult:
        """
        Main entry point.
        - actor_id, thread_id: scope for both SQL and vectorstore collection
        - user_query: latest user message text
        - files: base64-encoded upload metadata from request (if any)
        """
        collection_name = self._collection_for(actor_id, thread_id)
        self.rag_agent.set_collection(collection_name)

        # db: SQLAlchemy Session (injected from FastAPI)
        db = SessionLocal()
        # Ingest files if provided
        if files:
            if self.verbose:
                self.logger.info(f"Ingesting files for thread {actor_id}/{thread_id}, num files: {len(files)}")

            summaries = self.ingestion.ingest_files(
                db,
                user_id=actor_id,
                thread_id=thread_id,
                collection_name=collection_name,
                files=files,
            )
            if self.verbose:
                self.logger.info(f"Ingested files for thread {actor_id}/{thread_id}")
                if summaries:
                    self.logger.info(f"Ingested files summaries:")
                    for summary in summaries:
                        self.logger.info(summary)

        # Check if this thread has files
        has_files = (
            db.query(NMFile)
            .filter(
                NMFile.user_id == actor_id,
                NMFile.thread_id == thread_id,
                NMFile.collection == collection_name,
            )
            .count()
            > 0
        )
        if self.verbose:
            self.logger.info(f"Thread {actor_id}/{thread_id} has files: {has_files}")
        if not has_files:
            return RAGResult(
                used=False,
                augmented_query=user_query,
                meta={"used": False, "reason": "no_files"},
            )

        # Gate: decide if query is about files & which ones
        decision = self.gate.decide(
            db,
            user_id=actor_id,
            thread_id=thread_id,
            collection=collection_name,
            user_query=user_query,
        )
        if not decision.rag:
            return RAGResult(
                used=False,
                augmented_query=user_query,
                meta={"used": False, "reason": decision.reason or "gate_false"},
            )

        metadata_filter = build_metadata_filter_from_related_files(
            db,
            user_id=actor_id,
            thread_id=thread_id,
            collection=collection_name,
            related_files=decision.related_files,
        )
        if self.verbose:
            self.logger.info(f"Metadata filter: {metadata_filter}")

        # Retrieve context
        retrieved = self.rag_agent.retrieve(
            user_query=user_query,
            top_k=self.top_k,
            metadata_filter=metadata_filter,
            similarity_threshold=None,
        )
        ctx_text = (retrieved.context or "").strip()
        if self.verbose:
            self.logger.info("User Query: ")
            self.logger.info(user_query)
            self.logger.info(f"Documents in collection: {self.rag_agent._get_collection_docs_count()}")
            self.logger.info("Retrieved context: ")
            self.logger.info(ctx_text)

        if not ctx_text:
            return RAGResult(
                used=False,
                augmented_query=user_query,
                meta={
                    "used": False,
                    "reason": "empty_context",
                    "gate": decision.__dict__,
                },
            )
        augmented = f"{user_query}\n\n[CONTEXT]\n{ctx_text}\n[/CONTEXT]"
        db.close()
        return RAGResult(
            used=True,
            augmented_query=augmented,
            meta={
                "used": True,
                "reason": "ok",
                "gate": decision.__dict__,
            },
        )

    # -------- internals --------
    def _collection_for(self, user_id: int, thread_id: int) -> str:
        return self._safe_collection_name(f"nm_u{user_id}_t{thread_id}")

    def _safe_collection_name(self, s: str) -> str:
        s = self._ALLOWED_NAME.sub("_", s).strip("._-")
        return s or "nm_default"
