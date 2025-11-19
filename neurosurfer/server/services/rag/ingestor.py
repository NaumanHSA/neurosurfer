from __future__ import annotations

import base64
import os
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Dict, List

from sqlalchemy.orm import Session

from neurosurfer.agents.rag.agent import RAGAgent
from neurosurfer.agents.rag.constants import supported_file_types
from neurosurfer.server.services.rag.summarizer import FileSummarizer
from neurosurfer.server.db.models import NMFile
from uuid import uuid4
import logging

LOGGER = logging.getLogger(__name__)


class FileIngestor:
    """
    Handles persistence of uploaded files + ingestion into the vectorstore.
    """

    def __init__(
        self,
        rag_agent: RAGAgent,
        summarizer: FileSummarizer,
        upload_root: str,
        verbose: bool = False,
    ) -> None:
        self.rag_agent = rag_agent
        self.summarizer = summarizer
        self.upload_root = upload_root
        self.verbose = verbose
        os.makedirs(self.upload_root, exist_ok=True)

    # -------- public API --------
    def ingest_files(
        self,
        db: Session,
        *,
        user_id: int,
        thread_id: int,
        collection_name: str,
        files: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Persist uploaded files and ingest them into the vectorstore.

        `files` are dicts from Pydantic's .model_dump():
            - name
            - content (base64)
            - type (MIME, optional)
        """
        first_ingest = True

        for idx, f in enumerate(files):
            name = f.get("name") or f"upload_{idx}.bin"
            content_b64 = f.get("content") or ""
            mime = f.get("type") or None

            if not content_b64:
                continue

            raw_bytes = base64.b64decode(content_b64)
            # Save original upload
            original_filename = f"{uuid4()}__{name}"
            original_path = os.path.join(self.upload_root, original_filename)
            with open(original_path, "wb") as out:
                out.write(raw_bytes)

            if self.verbose:
                LOGGER.info(f"Ingesting file: {name}")
                LOGGER.info(f"File size: {os.path.getsize(original_path)} bytes")
                LOGGER.info(f"File MIME: {mime}")
                LOGGER.info(f"File path: {original_path}")
                LOGGER.info(f"File content length: {len(raw_bytes)} bytes")

            if name.lower().endswith(".zip"):
                summaries = self._ingest_zip(
                    db,
                    user_id=user_id,
                    thread_id=thread_id,
                    collection_name=collection_name,
                    zip_name=name,
                    zip_path=original_path,
                    reset_state_flag=lambda r: self._reset_flag_update(r, lambda v: self._set_flag(v, first_ingest)),
                )
                first_ingest = False
                return summaries
            else:
                summaries = self._ingest_single_file(
                    db,
                    user_id=user_id,
                    thread_id=thread_id,
                    collection_name=collection_name,
                    upload_name=name,
                    stored_path=original_path,
                    mime=mime,
                    reset_state=first_ingest,
                )
                first_ingest = False
                return summaries

    # -------- internals --------
    def _ingest_single_file(
        self,
        db: Session,
        *,
        user_id: int,
        thread_id: int,
        collection_name: str,
        upload_name: str,
        stored_path: str,
        mime: str | None,
        reset_state: bool,
    ) -> List[Dict[str, Any]]:
        LOGGER.info(f"[RAGIngest] Handling file: {upload_name}")
        summary = self.summarizer.summarize_path(stored_path, is_zip_member=False)
        nmfile = NMFile(
            id=str(uuid4()),
            user_id=user_id,
            thread_id=thread_id,
            filename=upload_name,
            summary=summary,
            stored_path=stored_path,
            mime=mime,
            size=os.path.getsize(stored_path),
            collection=collection_name,
        )
        db.add(nmfile)
        db.commit()
        db.refresh(nmfile)

        ingestion_summary = self.rag_agent.ingest(
            sources=[stored_path],
            extra_metadata={
                "file_id": nmfile.id,
                "filename": nmfile.filename,
                "thread_id": thread_id,
                "user_id": user_id,
            },
            reset_state=reset_state,
        )
        ingestion_summary["file_summary"] = summary
        return [ingestion_summary]

    def _ingest_zip(
        self,
        db: Session,
        *,
        user_id: int,
        thread_id: int,
        collection_name: str,
        zip_name: str,
        zip_path: str,
        reset_state_flag,
    ) -> List[Dict[str, Any]]:
        """
        Extract a zip to a temp dir, ingest each supported file inside.
        """
        LOGGER.info(f"[RAGIngest] Handling zip archive: {zip_name}")

        with tempfile.TemporaryDirectory(prefix="nm_zip_") as tmpdir:
            tmp_root = os.path.abspath(tmpdir)

            # Safe extraction
            with zipfile.ZipFile(zip_path, "r") as zf:
                for info in zf.infolist():
                    target_path = os.path.abspath(os.path.join(tmp_root, info.filename))
                    if not target_path.startswith(tmp_root):
                        continue  # zip-slip protection

                    if info.is_dir():
                        os.makedirs(target_path, exist_ok=True)
                        continue

                    os.makedirs(os.path.dirname(target_path), exist_ok=True)
                    with zf.open(info, "r") as src, open(target_path, "wb") as dst:
                        shutil.copyfileobj(src, dst)

            # Walk extracted tree
            first_ingest = reset_state_flag  # we handle via closure
            ingestion_summaries = []
            for root, dirs, filenames in os.walk(tmp_root):
                for fname in filenames:
                    ext = os.path.splitext(fname)[1].lower()
                    if supported_file_types and ext not in supported_file_types:
                        continue

                    extracted_path = os.path.join(root, fname)
                    rel_inside_zip = os.path.relpath(extracted_path, tmp_root)

                    safe_rel = rel_inside_zip.replace(os.sep, "__")
                    final_name = f"{uuid4()}__{safe_rel}"
                    final_path = os.path.join(self.upload_root, final_name)
                    os.makedirs(os.path.dirname(final_path), exist_ok=True)
                    shutil.copy2(extracted_path, final_path)
                    summary = self.summarizer.summarize_path(final_path, is_zip_member=True)

                    nmfile = NMFile(
                        id=str(uuid4()),
                        user_id=user_id,
                        thread_id=thread_id,
                        filename=rel_inside_zip,
                        summary=summary,
                        stored_path=final_path,
                        mime=None,
                        size=os.path.getsize(final_path),
                        collection=collection_name,
                    )
                    db.add(nmfile)
                    db.commit()
                    db.refresh(nmfile)

                    ingestion_summary = self.rag_agent.ingest(
                        sources=[final_path],
                        extra_metadata={
                            "file_id": nmfile.id,
                            "filename": nmfile.filename,
                            "thread_id": thread_id,
                            "user_id": user_id,
                            "source_zip": zip_name,
                        },
                        reset_state=first_ingest,
                    )
                    first_ingest = False
                    ingestion_summary["file_summary"] = summary
                    ingestion_summaries.append(ingestion_summary)
            return ingestion_summaries

    # tiny helpers to keep type-checkers happy (can simplify if you want)
    @staticmethod
    def _reset_flag_update(reset_state_flag, updater):
        return updater(reset_state_flag)

    @staticmethod
    def _set_flag(value, flag):
        return value and flag
