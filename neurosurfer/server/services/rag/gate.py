from __future__ import annotations

import json
from typing import Any, Optional
import logging
from sqlalchemy.orm import Session

from neurosurfer.models.chat_models.base import BaseChatModel
from neurosurfer.agents.common.utils import extract_and_repair_json
from neurosurfer.server.db.models import NMFile
from neurosurfer.server.services.rag.models import GateDecision

DEFAULT_RAG_GATE_SYSTEM_PROMPT = """
You are a routing and query-optimization assistant for a Retrieval-Augmented
Generation (RAG) system.

For each user query, you must:

1. Decide whether the question requires document-based retrieval ("rag": true/false).
2. Identify which uploaded files (if any) are relevant ("related_files": [...] exact names).
4. Decide the retrieval scope ("retrieval_scope": "small" | "medium" | "wide" | "full").

You are given:
- The user's current query.
- A list of uploaded files (file name + short description/summary).

You MUST output a single JSON object with the following structure:

{
  "rag": true | false,
  "related_files": ["file-A.ext", "file-B.ext"],
  "retrieval_scope": "small" | "medium" | "wide" | "full"
}

**Rules for deciding rag:**
- If the question clearly depends on the content of one or more uploaded files,
  set "rag": true and include ONLY the relevant file names.
- If the question is general or does not require file content,
  set "rag": false and "related_files": [].

**Rules for retrieval_scope:**
- small: the question concerns a specific fact, formula, definition, or short section.
- medium: the question spans multiple concepts or sections but not the entire file.
- wide: the question requires broad coverage, comparisons, or multiple far-apart sections.
- full: the question requires full-file understanding, full summary, or complete content.

**Important:**
- Output ONLY valid JSON.
- No comments, explanations, or trailing commas.
"""



class RAGGate:
    def __init__(
        self,
        llm: BaseChatModel | None,
        system_prompt: str = DEFAULT_RAG_GATE_SYSTEM_PROMPT,
        temperature: float = 0.5,
        max_new_tokens: int = 256,
        verbose: bool = False,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.llm = llm
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.verbose = verbose
        self.logger = logger or logging.getLogger(__name__)

    def decide(
        self,
        db: Session,
        *,
        user_id: int,
        thread_id: int,
        collection: str,
        user_query: str,
    ) -> GateDecision:
        """
        Decide whether to use RAG, and which files are relevant.
        """
        if not self.llm:
            # No router model configured -> always use RAG when files exist.
            return GateDecision(rag=True, optimized_query=user_query, related_files=[], retrieval_scope="medium", reason="no_gate_llm")

        files = (
            db.query(NMFile)
            .filter(
                NMFile.user_id == user_id,
                NMFile.thread_id == thread_id,
                NMFile.collection == collection,
            )
            .order_by(NMFile.created_at.asc())
            .all()
        )
        if not files:
            return GateDecision(rag=False, optimized_query=user_query, related_files=[], retrieval_scope="medium", reason="no_files_for_thread")

        files_block = "\n".join(
            f"Filename: {f.filename}\nSummary:\n{f.summary}\n\n" for f in files
        )
        self.logger.info(f"\n\nUSERQUERY: {user_query}\n\nFILES_BLOCK:\n{files_block}\n\n")

        user_prompt = (
            "# User query:\n"
            f"{user_query}\n\n"
            "# Uploaded files and their summaries:\n"
            f"{files_block}\n"
        )
        try:
            raw_text = self.llm.ask(
                user_prompt=user_prompt,
                system_prompt=self.system_prompt,
                temperature=self.temperature,
                max_new_tokens=self.max_new_tokens,
                stream=False,
            ).choices[0].message.content
        except Exception:
            return GateDecision(rag=True, optimized_query=user_query, related_files=[], retrieval_scope="medium", reason="gate_llm_error")

        try:
            gate_obj = extract_and_repair_json(raw_text)
            self.logger.info(f"GATE DECISION -------------- {gate_obj}")
            return GateDecision(
                rag=bool(gate_obj.get("rag", False)),
                related_files=list(gate_obj.get("related_files") or []),
                optimized_query=user_query,
                retrieval_scope=gate_obj.get("retrieval_scope", "medium"),
                raw_response=raw_text,
            )
        except Exception:
            return GateDecision(
                rag=False,
                related_files=[],
                optimized_query=user_query,
                retrieval_scope="medium",
                raw_response=raw_text,
                reason="json_parse_error",
            )