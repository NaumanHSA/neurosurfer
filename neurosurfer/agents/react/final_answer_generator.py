from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Generator
import logging

from neurosurfer.models.chat_models.base import BaseChatModel
from neurosurfer.agents.common.utils import normalize_response
from neurosurfer.agents.common import AgentMemory
from neurosurfer.tracing import Tracer, TracerConfig, TraceStepContext

LOGGER = logging.getLogger(__name__)


FINAL_ANSWER_SYSTEM_PROMPT = """
You are the FINAL ANSWER WRITER for a multi-step agent system.

You receive:
- The user's original query.
- Uploaded files and their summaries and keywords.
- A textual history of previous internal steps: thoughts, tool calls, and tool results.

Your job:
- Produce a single, clear final answer for the user.
- Use the tool results in the history as your primary source of truth.
- Do NOT reveal or describe the internal chain-of-thought or JSON actions explicitly.
- Do NOT mention that you saw "history", "thoughts", or "tools" — just answer naturally.
- Do NOT suggest new internal steps or tools; assume the computation is already done.

Language and length:
- You must respect the TARGET_LANGUAGE and ANSWER_LENGTH given in the prompt:
  - TARGET_LANGUAGE:
      - "english": answer entirely in English.
      - "arabic": answer entirely in Modern Standard Arabic.
      - "auto": if present, infer the most appropriate language from the user query.
  - ANSWER_LENGTH:
      - "short": 1-3 concise sentences.
      - "medium": a few short paragraphs.
      - "detailed": a thorough, step-by-step explanation with clear structure.

If tools in the history clearly computed a numeric or factual result (e.g. a count),
you MUST state that result explicitly and accurately.

If the history shows errors, missing libraries, missing files, or other limitations:
- Explain briefly what went wrong.
- Provide any partial results that are still valid.
- Suggest what would be needed to fully answer the question (e.g. installing a package,
  providing a missing file, or changing the environment).
""".strip()


FINAL_ANSWER_USER_PROMPT_TEMPLATE = """
Original user query:
{user_query}

Uploaded Files and their summaries and keywords:
{files_summaries_block}

History of internal steps for this query:
{history_block}

TARGET_LANGUAGE: {target_language}
ANSWER_LENGTH: {answer_length}

Additional instructions for this answer (if any):
{extra_instructions}

Instructions for this answer:
- Base your answer on the information in the history above.
- Do not invent specific numbers or file contents that are not shown there.
- You may use general world knowledge for explanation/clarification,
  but not to fabricate concrete results.
- Write the final answer only, with no preambles about your reasoning process.
- Remember, you must answer in `{target_language}` language.
""".strip()


class FinalAnswerGenerator:
    """
    Component that turns an agent's internal reasoning + tool history
    into a single user-facing final answer in a requested language
    and length.

    This is NOT a tool; it is owned and called by the ReActAgent (or its wrapper).

    Typical usage:
    - Called as the LAST step by a ReAct agent.
    - The agent passes:
        - user_query: original question.
        - history: history.to_prompt() string (thoughts, actions, tool results).
        - files_summaries_block: optional text block with file summaries.
        - target_language: "english" | "arabic" | "auto".
        - answer_length: "short" | "medium" | "detailed".
        - extra_instructions: per-answer tweaks (user preferences, etc.).
    """

    def __init__(
        self,
        llm: BaseChatModel,
        language: str = "english",        # "english" | "arabic"
        answer_length: str = "detailed",  # "short" | "medium" | "detailed"
        max_history_chars: int = 12000,
        temperature: float = 0.3,
        max_new_tokens: int = 1024,
        tracer: Optional[Tracer] = None,
        log_traces: Optional[bool] = True,
        logger: Optional[logging.Logger] = None,
        base_system_instructions: Optional[str] = None,
    ) -> None:
        if llm is None:
            raise ValueError("FinalAnswerGenerator requires an llm")
        self.llm = llm
        self.language = language
        self.answer_length = answer_length
        self.max_history_chars = max_history_chars
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.logger = logger or LOGGER

        # Allow extending the system prompt for app-wide style instructions
        self.system_prompt = FINAL_ANSWER_SYSTEM_PROMPT
        if base_system_instructions:
            self.system_prompt += (
                "\n\nAdditional global style instructions:\n" + base_system_instructions.strip()
            )

        self.log_traces = log_traces
        self.tracer: Tracer = tracer or Tracer(
            config=TracerConfig(log_steps=self.log_traces),
            meta={
                "agent_type": "react_agent",
                "agent_id": "FinalAnswerGenerator",
                "model": self.llm.model_name,
                "target_language": self.language,
                "answer_context": self.answer_length,
                "log_steps": self.log_traces,
            },
            logger_=logger,
        )

    # --------- Public API ---------
    def generate(
        self,
        *,
        user_query: str,
        history: str,
        memory: AgentMemory,
        target_language: Optional[str] = None,
        answer_length: Optional[str] = None,
        extra_instructions: Optional[str] = None,
    ) -> Generator[str, None, None]:
        """
        Generate the final answer text from the query + history.

        Returns a Generator[str] for streaming. If you want a full string,
        just join the chunks.
        """
        files_summaries_block = memory.get_persistent("files_summaries_block")
        files_block = (files_summaries_block or "").strip()
        lang = self._normalize_language(target_language)
        length = self._normalize_length(answer_length)
        history_block = self._truncate_history(history or "")
        extra_instr = (extra_instructions or "").strip() or "(none)"

        user_prompt = FINAL_ANSWER_USER_PROMPT_TEMPLATE.format(
            user_query=(user_query or "").strip(),
            history_block=history_block.strip() or "(no internal history available)",
            files_summaries_block=files_block or "(no files summaries available)",
            target_language=lang,
            answer_length=length,
            extra_instructions=extra_instr,
        )
        streaming_response = self.llm.ask(
            system_prompt=self.system_prompt,
            user_prompt=user_prompt,
            chat_history=[],
            temperature=self.temperature,
            max_new_tokens=self.max_new_tokens,
            stream=True,
        )

        # normalize_response will yield strings for each chunk
        return normalize_response(streaming_response)

    # --------- Internals ---------
    def _normalize_language(self, lang: Optional[str]) -> str:
        if not lang:
            return self.language

        v = lang.strip().lower()
        if v in {"en", "english"}:
            return "english"
        if v in {"ar", "arabic", "arab"}:
            return "arabic"
        if v in {"auto", "auto-detect", "detect"}:
            return "auto"

        return self.language

    def _normalize_length(self, length: Optional[str]) -> str:
        if not length:
            return self.answer_length

        v = length.strip().lower()
        if v in {"short", "medium", "detailed"}:
            return v

        return self.answer_length

    def _truncate_history(self, history: str) -> str:
        """
        Truncate history if it exceeds max_history_chars.
        We keep the *end* since it usually contains the latest, most relevant tool results.
        """
        max_len = max(0, int(self.max_history_chars))
        if max_len == 0 or len(history) <= max_len:
            return history

        tail = history[-max_len:]
        return "[History truncated; showing the most recent steps only]\n" + tail
