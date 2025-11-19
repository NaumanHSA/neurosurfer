# neurosurfer/services/rag/file_summarizer.py
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional
import logging

from neurosurfer.models.chat_models.base import BaseChatModel
from neurosurfer.agents.rag.agent import RAGAgent

LOGGER = logging.getLogger(__name__)

FILE_SUMMARY_SYSTEM_PROMPT = """You summarize files for a Retrieval-Augmented Generation (RAG) system.

Your goal is to describe WHAT the file is about and its main topics in plain language.
Do not copy large chunks of text. Be concise and informative."""


class FileSummarizer:
    """
    Responsible for turning file content into a short textual summary,
    used by the routing LLM (gate) and stored in NMFile.summary.
    """

    def __init__(
        self,
        rag_agent: RAGAgent,
        llm: Optional[BaseChatModel] = None,
        verbose: bool = False,
    ) -> None:
        self.rag_agent = rag_agent
        self.llm = llm  # can be None: falls back to simple excerpt
        self.verbose = verbose

    # -------- public API --------

    def summarize_path(self, path: str, *, is_zip_member: bool) -> str:
        """
        Build a summary for the file at `path` using actual content if possible.

        - Reads via RAGAgent's FileReader.
        - Samples text (start/middle/end) with a budget.
        - Uses LLM if available:
            - zip member -> ~2 sentences
            - single file -> ~1 short paragraph
        """
        basename = os.path.basename(path)
        raw_text = self._read_text(path)
        if not raw_text:
            return f"User-uploaded file named '{basename}'."

        excerpt = self._sample_text(raw_text, is_zip_member=is_zip_member)
        if not excerpt:
            return f"User-uploaded file named '{basename}'."

        if not self.llm:
            return self._fallback_excerpt(basename, excerpt, is_zip_member=is_zip_member)

        style = (
            "two concise sentences"
            if is_zip_member
            else "one short paragraph of 3-4 sentences"
        )
        user_prompt = (
            f"Summarize the following file content in {style}. "
            "Focus on what the file is about and its main topics.\n\n"
            "CONTENT START\n"
            f"{excerpt}\n"
            "CONTENT END"
        )
        try:
            summary = self.llm.ask(
                user_prompt=user_prompt,
                system_prompt=FILE_SUMMARY_SYSTEM_PROMPT,
                temperature=0.5,
                max_new_tokens=1024,
                stream=False,
            ).choices[0].message.content
            return summary.strip()
        except Exception:
            # Log outside if needed
            pass
        return self._fallback_excerpt(basename, excerpt, is_zip_member=is_zip_member)

    # -------- internals --------
    def _read_text(self, path: str) -> str:
        try:
            return self.rag_agent.file_reader.read(Path(path)) or ""
        except Exception:
            return ""

    def _sample_text(self, text: str, *, is_zip_member: bool) -> str:
        text = re.sub(r"\s+", " ", text or "").strip()
        if not text:
            return ""
        words = text.split(" ")
        max_words = 200 if is_zip_member else 600
        if len(words) <= max_words:
            return " ".join(words)

        seg_count = 3
        per_seg = max(max_words // seg_count, 1)
        first = words[:per_seg]
        mid_start = max((len(words) // 2) - (per_seg // 2), 0)
        middle = words[mid_start : mid_start + per_seg]
        last = words[-per_seg:]
        return " ".join(first) + " ... " + " ".join(middle) + " ... " + " ".join(last)

    def _fallback_excerpt(self, basename: str, excerpt: str, *, is_zip_member: bool) -> str:
        max_chars = 320 if is_zip_member else 640
        truncated = excerpt[:max_chars] + ("..." if len(excerpt) > max_chars else "")
        return f"User-uploaded file named '{basename}' about: {truncated}"

