from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Generator
import logging

from neurosurfer.agents.react import final_answer_generator
from neurosurfer.models.chat_models.base import BaseChatModel
from neurosurfer.agents.common.utils import normalize_response
from neurosurfer.tracing import Tracer, TracerConfig

from .templates import FINAL_ANSWER_SYSTEM_PROMPT, FINAL_ANSWER_USER_PROMPT_TEMPLATE

LOGGER = logging.getLogger(__name__)


class FinalAnswerGenerator:
    """
    Component that turns RAG / code / other tool context into a single
    user-facing final answer in a requested language and length.

    This is NOT a tool; it is called by the main orchestrator after the
    router + pipeline (RAG or CodeAgent or direct) have produced their context.

    Typical usage:

        context_block = rag_context_block | code_agent_summary | "...etc..."
        answer_stream = final_answer_generator.generate(
            user_query=user_query,
            context_block=context_block,
            files_summaries_block=files_summaries_block,
            chat_history_block=chat_history_block,
            target_language="english",
            answer_length="detailed",
            extra_instructions="Be friendly but concise.",
        )
        for chunk in answer_stream:
            yield chunk
    """

    def __init__(
        self,
        llm: BaseChatModel,
        default_language: str = "english",        # "english" | "arabic"
        default_answer_length: str = "detailed",  # "short" | "medium" | "detailed"
        max_context_chars: int = 12000,
        temperature: float = 0.3,
        max_new_tokens: int = 1024,
        log_traces: Optional[bool] = True,
        tracer: Optional[Tracer] = None,
        logger: Optional[logging.Logger] = None,
        base_system_instructions: Optional[str] = None,
    ) -> None:
        if llm is None:
            raise ValueError("FinalAnswerGenerator requires an llm")

        self.llm = llm
        self.default_language = default_language
        self.default_answer_length = default_answer_length
        self.max_context_chars = max_context_chars
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.log_traces = log_traces
        self.logger = logger or LOGGER

        # Base tracer that actually records and log steps (RichTracer by default).
        self.tracer: Tracer = tracer or Tracer(
            config=TracerConfig(log_steps=self.log_traces),
            meta={
                "agent_type": "FinalAnswerGenerator",
                "model": llm.model_name,
                "log_steps": self.log_traces,
            },
            logger_=logger,
        )

        # Allow extending the system prompt for app-wide style instructions
        system_prompt = FINAL_ANSWER_SYSTEM_PROMPT
        if base_system_instructions:
            system_prompt += (
                "\n\nAdditional global style instructions:\n"
                + base_system_instructions.strip()
            )
        self.system_prompt = system_prompt

    # --------- Public API ---------
    def generate(
        self,
        *,
        user_query: str,
        route: str,
        context_block: str,
        files_summaries_block: Optional[str] = None,
        chat_history_block: Optional[str] = None,
        target_language: Optional[str] = None,
        answer_length: Optional[str] = None,
        extra_instructions: Optional[str] = None,
        route_reason: Optional[str] = None,
    ) -> Generator[str, None, None]:
        """
        Generate the final answer text from the query + context.

        Returns a streaming Generator[str]. If you want a full string,
        just `''.join(generator)`.
        """
        with self.tracer(
            agent_id="final_answer_generator",
            kind="llm.call",
            label="final_answer_generator.llm.call",
            inputs={
                "user_query": user_query,
                "target_language": target_language,
                "answer_length": answer_length,
                "extra_instructions": extra_instructions,
                "files_summaries_block": files_summaries_block,
            },
        ) as tracer:
            lang = self._normalize_language(target_language)
            length = self._normalize_length(answer_length)

            if route == "reject":
                user_prompt = (
                    f"The user asked for a response that was rejected by the router. "
                    f"Reason: {route_reason}."
                    f"\n\nPlease provide a helpful response based on the user's original query in {lang.upper()} language."
                )
            else:
                files_block = (files_summaries_block or "").strip() or "(no files summaries available)"
                ctx_block = self._truncate_context(context_block or "")
                history_block = (chat_history_block or "").strip() or "(not provided)"
                extra_instr = (extra_instructions or "").strip() or "(none)"
                user_prompt = FINAL_ANSWER_USER_PROMPT_TEMPLATE.format(
                    user_query=(user_query or "").strip(),
                    files_summaries_block=files_block,
                    context_block=ctx_block.strip() or "(no additional context provided)",
                    chat_history_block=history_block,
                    target_language=lang,
                    answer_length=length,
                    extra_instructions=extra_instr,
                )
            print(f"\n\nUser Prompt for Final Answer Generator:\n\n{user_prompt}\n\n")

            tracer.inputs(user_prompt=user_prompt, user_prompt_len=len(user_prompt))
            streaming_response = self.llm.ask(
                system_prompt=self.system_prompt,
                user_prompt=user_prompt,
                chat_history=[],
                temperature=self.temperature,
                max_new_tokens=self.max_new_tokens,
                stream=True,
            )
            final_answer = ""
            for chunk in streaming_response:
                chunk = chunk.choices[0].delta.content or ""
                tracer.stream(chunk, type="whiteb")
                final_answer += chunk
                yield chunk
            tracer.stream("\n\n", type="whiteb")
            # normalize_response will yield strings for each chunk
            tracer.outputs(final_answer=final_answer)

    # --------- Internals ---------
    def _normalize_language(self, lang: Optional[str]) -> str:
        if not lang:
            return self.default_language

        v = lang.strip().lower()
        if v in {"en", "english"}:
            return "english"
        if v in {"ar", "arabic", "arab"}:
            return "arabic"
        if v in {"auto", "auto-detect", "detect"}:
            return "auto"

        return self.default_language

    def _normalize_length(self, length: Optional[str]) -> str:
        if not length:
            return self.default_answer_length

        v = length.strip().lower()
        if v in {"short", "medium", "detailed"}:
            return v

        return self.default_answer_length

    def _truncate_context(self, context: str) -> str:
        """
        Truncate context if it exceeds max_context_chars.
        We keep the *end* since it usually contains the latest, most relevant results.
        """
        max_len = max(0, int(self.max_context_chars))
        if max_len == 0 or len(context) <= max_len:
            return context

        tail = context[-max_len:]
        return "[Context truncated; showing the most recent part only]\n" + tail