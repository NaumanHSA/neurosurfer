from __future__ import annotations

import os
from typing import Any, Dict, List

from sqlalchemy.orm import Session

from neurosurfer.agents.rag.agent import RAGAgent
from neurosurfer.server.services.rag.summarizer import FileSummarizer
from neurosurfer.server.db.models import NMFile

import logging

LOGGER = logging.getLogger(__name__)


class FileIngestor:
    """
    RAG File Ingestor

    Takes already-persisted NMFile records (one per physical file on disk)
    and ingests them into the vectorstore.

    It assumes:
      - Files are already stored on disk at nmfile.stored_path
      - Zips have already been expanded at the API layer (one NMFile per
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
        files: List[NMFile],
        reset_state: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Ingest existing NMFile rows into the vectorstore.

        Args:
            db: SQLAlchemy session
            user_id, thread_id: scope sanity-checks
            collection_name: vectorstore collection name
            files: NMFile records to ingest (e.g. new files for the last message)
            reset_state: passed only to the first ingest call
        Returns:
            List of ingestion summaries.
        """
        summaries: List[Dict[str, Any]] = []
        first = reset_state

        for nmfile in files:
            if nmfile.user_id != user_id or nmfile.thread_id != thread_id:
                LOGGER.warning(
                    "Skipping NMFile %s: user/thread mismatch (user_id=%s, thread_id=%s)",
                    nmfile.id,
                    nmfile.user_id,
                    nmfile.thread_id,
                )
                continue

            path = nmfile.stored_path
            if not path or not os.path.exists(path):
                LOGGER.warning(
                    "Skipping NMFile %s: missing stored_path (%s)",
                    nmfile.id,
                    path,
                )
                continue

            if self.verbose:
                LOGGER.info(
                    "RAG ingest: file_id=%s filename=%s size=%s mime=%s reset_state=%s",
                    nmfile.id,
                    nmfile.filename,
                    nmfile.size,
                    nmfile.mime,
                    first,
                )

            # Summarize (if not already summarized)
            if not nmfile.summary:
                try:
                    nmfile.summary = self.summarizer.summarize_path(path, is_zip_member=False)
                except Exception:
                    LOGGER.exception("Failed to summarize file %s", nmfile.id)

            res = self.rag_agent.ingest(
                sources=[path],
                extra_metadata={
                    "file_id": nmfile.id,
                    "filename": nmfile.filename,
                    "thread_id": nmfile.thread_id,
                    "user_id": nmfile.user_id,
                    "collection": collection_name,
                },
                reset_state=first,
            )
            first = False

            res["file_id"] = nmfile.id
            res["file_summary"] = nmfile.summary
            summaries.append(res)

            # If you use an `ingested` flag, you can mark it here:
            nmfile.ingested = True

        db.commit()
        return summaries
