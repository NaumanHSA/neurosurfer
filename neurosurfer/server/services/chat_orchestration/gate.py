from dataclasses import dataclass
from typing import List, Optional, Literal, Any, Dict
import json
import logging

from neurosurfer.models.chat_models.base import BaseChatModel
from neurosurfer.agents.common.utils import extract_and_repair_json
from neurosurfer.tracing import Tracer, TracerConfig

from .types import GateDecision, RouteType
from .templates import MAIN_AGENT_GATE_SYSTEM_PROMPT, MAIN_AGENT_GATE_USER_PROMPT_TEMPLATE

class GateLLM:
    """
    LLM-based router that decides whether to:
    - answer directly,
    - call the RAG pipeline,
    - call the CodeAgent,
    - or ask the user for clarification.

    Inputs:
        - user_query: raw text from the user.
        - files_summaries_block: JSON-like string describing uploaded files.
        - chat_history_block: formatted recent chat history.

    Output:
        - MainAgentGateDecision with route, optimized_query, use_files, etc.
    """

    def __init__(
        self,
        llm: BaseChatModel,
        temperature: float = 0.1,
        max_new_tokens: int = 1024,
        tracer: Optional[Tracer] = None,
        log_traces: Optional[bool] = True,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        if llm is None:
            raise ValueError("MainAgentRouter requires a BaseChatModel llm")

        self.llm = llm
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.logger = logger or logging.getLogger(__name__)
        self.log_traces = log_traces
        # Base tracer that actually records and log steps (RichTracer by default).
        self.tracer: Tracer = tracer or Tracer(
            config=TracerConfig(log_steps=self.log_traces),
            meta={
                "agent_type": "MainAgentRouter",
                "model": self.llm.model_name,
                "log_steps": self.log_traces,
            },
            logger_=logger,
        )

    def decide(
        self,
        user_query: str,
        files_summaries_block: str,
        chat_history_block: str = "",
    ) -> GateDecision:
        """
        Run the routing LLM and parse its decision. Falls back to a safe default
        (route="direct") if parsing fails.
        """
        with self.tracer(
            agent_id="gate_llm",
            kind="llm.call",
            label="gate.decide",
            inputs={
                "agent_type": type(self).__name__,
                "query": user_query
            },
        ) as gate_tracer:
            user_prompt = MAIN_AGENT_GATE_USER_PROMPT_TEMPLATE.format(
                user_query=user_query.strip(),
                files_summaries_block=(files_summaries_block or "(no uploaded files)"),
                chat_history_block=(chat_history_block or "(no recent history)"),
            )
            # print(f"\n\nuser_prompt:{user_prompt}\n\n")
            gate_tracer.inputs(user_prompt=user_prompt, user_prompt_len=len(user_prompt))
            resp = self.llm.ask(
                system_prompt=MAIN_AGENT_GATE_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                chat_history=[],
                temperature=self.temperature,
                max_new_tokens=self.max_new_tokens,
                stream=False,
            )
            raw = (resp.choices[0].message.content or "").strip()
            try:
                obj = self._parse_json_safely(raw)
                gate_tracer.outputs(decision=obj)
                decision = self._from_json_obj(obj, raw_response=raw)
                return decision
            except Exception as e:
                self.logger.error(
                    "[MainAgentRouter] Failed to parse routing response (%s). "
                    "Falling back to direct answer. Raw: %r",
                    e,
                    raw,
                )
                # Safe fallback: answer directly with original query
                gate_tracer.log(message=f"Failed to parse routing response with error {e}. Falling back to direct answer.", type="error")
                gate_tracer.outputs(decision=f"Error: Failed to decide on route. Defaulting to `direct`")
                return GateDecision(
                    route="direct",
                    optimized_query=user_query,
                    query_language_detected="unknown",
                    use_files=[],
                    clarification_question=None,
                    reason="Fallback: could not parse routing JSON",
                    raw_response=raw,
                )

    def _parse_json_safely(self, text: str) -> Dict[str, Any]:
        """
        Try to parse as JSON. If that fails, fall back to your
        extract_and_repair_json helper (if available).
        """
        try:
            return json.loads(text)
        except Exception:
            # Optional: use your existing robust JSON repair helper
            try:
                return extract_and_repair_json(text, return_dict=True)
            except Exception as e:
                raise ValueError(f"Could not parse routing JSON: {e}") from e

    def _from_json_obj(self, obj: Dict[str, Any], raw_response: str) -> GateDecision:
        """
        Normalize JSON fields and enforce defaults.
        """
        route_raw = str(obj.get("route", "direct")).strip().lower()
        if route_raw not in {"direct", "rag", "code", "clarify"}:
            route: RouteType = "direct"
        else:
            route = route_raw  # type: ignore[assignment]

        optimized_query = str(obj.get("optimized_query") or "").strip()
        if not optimized_query:
            # If the gate didn't provide an optimized query, fall back to original.
            optimized_query = "<missing optimized_query – fallback to user_query>"

        query_language_detected = str(obj.get("query_language_detected") or "").strip()
        if not query_language_detected:
            query_language_detected = "unknown"

        use_files_raw = obj.get("use_files") or []
        if not isinstance(use_files_raw, list):
            use_files_raw = []
        use_files = [str(k) for k in use_files_raw]

        clarification_question_raw = obj.get("clarification_question", None)
        clarification_question: Optional[str]
        if route == "clarify":
            clarification_question = (
                str(clarification_question_raw).strip()
                if clarification_question_raw
                else "Could you please clarify your request?"
            )
        else:
            clarification_question = None

        reason = str(obj.get("reason") or "").strip()
        if not reason:
            reason = f"Router chose route={route}"

        return GateDecision(
            route=route,
            optimized_query=optimized_query,
            query_language_detected=query_language_detected,
            use_files=use_files,
            clarification_question=clarification_question,
            reason=reason,
            raw_response=raw_response,
        )
