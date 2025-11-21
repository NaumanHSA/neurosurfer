from __future__ import annotations

import json
from typing import Any, Optional, List
import logging
from sqlalchemy.orm import Session

from neurosurfer.models.chat_models.base import BaseChatModel
from neurosurfer.agents.common.utils import extract_and_repair_json
from neurosurfer.server.db.models import NSFile
from neurosurfer.server.services.rag.models import GateDecision

DEFAULT_RAG_GATE_SYSTEM_PROMPT = """
You are a routing assistant for a Retrieval-Augmented Generation (RAG) system.
Based on the user query and the list of uploaded files, you must decide whether the
question requires RAG and, if so, which files are relevant.

For each user query, you must:
1. Decide whether the question requires document-based retrieval ("rag": true/false).
2. Identify which uploaded files (if any) are relevant ("related_files": [...] exact names).
3. Decide the retrieval scope ("retrieval_scope": "small" | "medium" | "wide" | "full").

You are given:
- The user's current query.
- A list of uploaded files for this thread. Some of them may be newly attached
  to the current message ("attached_to_current_message: yes/no").

You MUST output a single JSON object with the following structure:

{
  "rag": true | false,
  "related_files": ["file-A.ext", "file-B.ext"],
  "retrieval_scope": "small" | "medium" | "wide" | "full"
}

Rules for deciding rag:
- If the question clearly depends on the content of one or more uploaded files,
  set "rag": true and include ONLY the relevant file names in "related_files".
- If the question is general or does not require file content,
  set "rag": false and "related_files": [].

Rules for using attached files:
- Files marked as "attached_to_current_message: yes" are strongly likely to be
  relevant to the current question.
- The user may attach these files as the primary source OR as additional context.
- You may also include other older files from the thread if they clearly help
  answer the question, but do not add unrelated files.

Rules for retrieval_scope:
- small: the question concerns a specific fact, formula, definition, or short section.
- medium: the question spans multiple concepts or sections but not the entire file.
- wide: the question requires broad coverage, comparisons, or multiple far-apart sections.
- full: the question requires full-file understanding, full summary, or complete content.

Important:
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
        message_files: List[NSFile] = [],
    ) -> GateDecision:
        """
        Decide whether to use RAG, which files are relevant, and the retrieval scope.

        Behavior:
        - If there are files attached to the current message:
            - They are ALWAYS included in related_files (pinned).
            - If there are other files in the thread and a gate LLM is configured,
              we ask the LLM to select any additional relevant files from the rest.
        - If there are no message_files:
            - We fall back to normal gate behavior over all thread files.
        """
        # No router model configured -> simple fallback behavior.
        if not self.llm:
            if message_files:
                # Always use RAG if the user attached files in this message.
                return GateDecision(
                    rag=True,
                    optimized_query=user_query,
                    related_files=[f.filename for f in message_files],
                    retrieval_scope="medium",
                    reason="no_gate_llm_has_message_files",
                )
            else:
                # Without files and without gate LLM, default to no RAG.
                return GateDecision(
                    rag=False,
                    optimized_query=user_query,
                    related_files=[],
                    retrieval_scope="medium",
                    reason="no_gate_llm_no_files",
                )

        # Fetch ALL files for this thread/collection
        all_files = (
            db.query(NSFile)
            .filter(
                NSFile.user_id == user_id,
                NSFile.thread_id == thread_id,
                NSFile.collection == collection,
            )
            .order_by(NSFile.created_at.asc())
            .all()
        )

        if not all_files and not message_files:
            return GateDecision(
                rag=False,
                optimized_query=user_query,
                related_files=[],
                retrieval_scope="medium",
                reason="no_files_for_thread",
            )

        # Separate attached files vs older files
        attached_ids = {f.id for f in message_files}
        attached_files = list(message_files)
        older_files = [f for f in all_files if f.id not in attached_ids]

        attached_names = [f.filename for f in attached_files]

        # If we have ONLY attached files and nothing else in the thread,
        # we don't need the gate LLM: just use them.
        if attached_files and not older_files:
            return GateDecision(
                rag=True,
                optimized_query=user_query,
                related_files=attached_names,
                retrieval_scope="medium",
                reason="only_attached_files",
            )

        # Build the files block for the LLM:
        # we include BOTH attached and older files, marking which are attached now.
        def _fmt_file(f: NSFile, attached: bool) -> str:
            flag = "yes" if attached else "no"
            return (
                f"Filename: {f.filename}\n"
                f"Attached_to_current_message: {flag}\n"
                f"Summary:\n{f.summary or '(no summary)'}\n"
            )

        files_block_lines: List[str] = []
        for f in attached_files:
            files_block_lines.append(_fmt_file(f, attached=True))
        for f in older_files:
            files_block_lines.append(_fmt_file(f, attached=False))

        files_block = "\n\n".join(files_block_lines)

        if self.verbose:
            self.logger.info(
                f"\n\nUSERQUERY: {user_query}\n\nFILES_BLOCK:\n{files_block}\n\n"
            )

        # Construct user prompt for gate LLM
        user_prompt = (
            "# User query:\n"
            f"{user_query}\n\n"
            "# Uploaded files for this thread (with summaries):\n"
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
            # If the gate LLM fails, still use attached files if present.
            if attached_files:
                return GateDecision(
                    rag=True,
                    optimized_query=user_query,
                    related_files=attached_names,
                    retrieval_scope="medium",
                    reason="gate_llm_error_with_attached_files",
                )
            return GateDecision(
                rag=False,
                optimized_query=user_query,
                related_files=[],
                retrieval_scope="medium",
                reason="gate_llm_error_no_attached_files",
            )

        try:
            gate_obj = extract_and_repair_json(raw_text)
            self.logger.info(f"GATE DECISION -------------- {gate_obj}")
            gate_rag = bool(gate_obj.get("rag", False))
            gate_related = list(gate_obj.get("related_files") or [])
            scope = gate_obj.get("retrieval_scope", "medium")

            # Final related_files = attached_files UNION gate-selected files
            final_related = sorted(set(attached_names) | set(gate_related))

            # If the LLM says rag=false but we have attached files, override to true.
            if attached_files:
                rag_flag = True
            else:
                rag_flag = gate_rag and bool(final_related)

            return GateDecision(
                rag=rag_flag,
                related_files=final_related,
                optimized_query=user_query,
                retrieval_scope=scope,
                raw_response=raw_text,
            )
        except Exception:
            # If JSON parsing fails, still fall back gracefully.
            if attached_files:
                return GateDecision(
                    rag=True,
                    related_files=attached_names,
                    optimized_query=user_query,
                    retrieval_scope="medium",
                    raw_response=raw_text,
                    reason="json_parse_error_with_attached_files",
                )
            return GateDecision(
                rag=False,
                related_files=[],
                optimized_query=user_query,
                retrieval_scope="medium",
                raw_response=raw_text,
                reason="json_parse_error_no_attached_files",
            )
