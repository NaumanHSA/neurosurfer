from __future__ import annotations

import os
from typing import Any, Dict, List

from sqlalchemy.orm import Session

from neurosurfer.agents.rag.agent import RAGAgent
from neurosurfer.server.services.rag.summarizer import FileSummarizer
from neurosurfer.server.db.models import NSFile

import logging

LOGGER = logging.getLogger(__name__)


class FileIngestor:
    """
    RAG File Ingestor

    Takes already-persisted NSFile records (one per physical file on disk)
    and ingests them into the vectorstore.

    It assumes:
      - Files are already stored on disk at nsfile.stored_path
      - Zips have already been expanded at the API layer (one NSFile per
        extracted inner file)
    """

    def __init__(
        self,
        rag_agent: RAGAgent,
        summarizer: FileSummarizer,
        verbose: bool = False,
    ) -> None:
        self.rag_agent = rag_agent
        self.summarizer = summarizer
        self.verbose = verbose

    def ingest_files(
        self,
        db: Session,
        *,
        user_id: int,
        thread_id: int,
        collection_name: str,
        files: List[NSFile],
        reset_state: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Ingest existing NSFile rows into the vectorstore.

        Args:
            db: SQLAlchemy session
            user_id, thread_id: scope sanity-checks
            collection_name: vectorstore collection name
            files: NSFile records to ingest (e.g. new files for the last message)
            reset_state: passed only to the first ingest call
        Returns:
            List of ingestion summaries.
        """
        summaries: List[Dict[str, Any]] = []
        first = reset_state

        for nsfile in files:
            if nsfile.user_id != user_id or nsfile.thread_id != thread_id:
                LOGGER.warning(
                    "Skipping NSFile %s: user/thread mismatch (user_id=%s, thread_id=%s)",
                    nsfile.id,
                    nsfile.user_id,
                    nsfile.thread_id,
                )
                continue

            path = nsfile.stored_path
            if not path or not os.path.exists(path):
                LOGGER.warning(
                    "Skipping NSFile %s: missing stored_path (%s)",
                    nsfile.id,
                    path,
                )
                continue

            if self.verbose:
                LOGGER.info(
                    "RAG ingest: file_id=%s filename=%s size=%s mime=%s reset_state=%s",
                    nsfile.id,
                    nsfile.filename,
                    nsfile.size,
                    nsfile.mime,
                    first,
                )

            # Summarize (if not already summarized)
            if not nsfile.summary:
                try:
                    nsfile.summary = self.summarizer.summarize_path(path, is_zip_member=False)
                except Exception:
                    LOGGER.exception("Failed to summarize file %s", nsfile.id)

            res = self.rag_agent.ingest(
                sources=[path],
                extra_metadata={
                    "file_id": nsfile.id,
                    "filename": nsfile.filename,
                    "thread_id": nsfile.thread_id,
                    "user_id": nsfile.user_id,
                    "collection": collection_name,
                },
                reset_state=first,
            )
            first = False

            res["file_id"] = nsfile.id
            res["file_summary"] = nsfile.summary
            summaries.append(res)

            # If you use an `ingested` flag, you can mark it here:
            nsfile.ingested = True

        db.commit()
        return summaries
