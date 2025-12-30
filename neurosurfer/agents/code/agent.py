# neurosurfer/agents/code/agent.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Literal, Generator
import logging

from neurosurfer.models.chat_models.base import BaseChatModel
from neurosurfer.tools import Toolkit
from neurosurfer.tools.base_tool import ToolResponse
from neurosurfer.tools.code_execution import (
    PythonExecTool,
    PythonExecToolConfig,
)
from neurosurfer.agents.react.agent import ReActAgent
from neurosurfer.agents.react.types import ReactAgentResponse
from neurosurfer.tracing import Tracer, TracerConfig

from .config import CodeAgentConfig
from .scratchpad import CODE_AGENT_SPECIFIC_INSTRUCTIONS


logger = logging.getLogger(__name__)


class CodeAgent(ReActAgent):
    """
    CodeAgent: a specialized ReAct agent for multi-step Python code execution.

    - Uses the generic ReActAgent core loop (Actions + tool calls + Final Answer).
    - Comes with a default toolkit containing `python_execute`.
    - Encourages multi-step workflows: inspect -> compute -> explain.
    - Supports streaming and non-streaming responses.
    - Allows passing `files_context` and `workdir` directly at run-time.
    """

    def __init__(
        self,
        *,
        llm: BaseChatModel,
        toolkit: Optional[Toolkit] = None,
        config: Optional[CodeAgentConfig] = None,
        tracer: Optional[Tracer] = None,
        log_traces: bool = True,
        logger_: Optional[logging.Logger] = None,
    ) -> None:
        if llm is None:
            raise ValueError("CodeAgent requires an llm")

        self.code_config = config or CodeAgentConfig(skip_special_tokens=True)
        print(f"Code Agent Cofig:\n{self.code_config}\n")
        logger_local = logger_ or logger

        # Default toolkit: just PythonExecTool for now, but can be extended
        if toolkit is None:
            toolkit = self._build_default_toolkit(llm=llm)

        # Tracer (reuse your global tracer infra)
        tracer = tracer or Tracer(
            config=TracerConfig(log_steps=log_traces),
            meta={
                "agent_type": "code_agent",
                "agent_config": self.code_config,
                "model": llm.model_name,
                "toolkit": bool(toolkit),
                "logging": "full" if self.code_config.log_internal_thoughts else "basic",
            },
            logger_=logger_local,
        )
        super().__init__(
            id=self.code_config.agent_name,
            llm=llm,
            toolkit=toolkit,
            specific_instructions=CODE_AGENT_SPECIFIC_INSTRUCTIONS,
            config=self.code_config,
            logger=logger_local,
            tracer=tracer,
            log_traces=log_traces,
        )

    # ---------- Toolkit wiring ----------
    def _build_default_toolkit(self, llm: BaseChatModel) -> Toolkit:
        """
        Build a default Toolkit for CodeAgent.

        Currently:
        - Registers only the PythonExecTool.
        - You can later extend this to include a RAG retrieval tool, etc.
        """
        tk = Toolkit()
        py_cfg = PythonExecToolConfig(
            max_code_retries=3,
            include_code_in_answer=False,
            max_table_rows=20,
        )
        py_tool = PythonExecTool(llm=llm, config=py_cfg, logger=logger)
        tk.register_tool(py_tool)
        return tk

    def run(
        self,
        *,
        query: str,
        stream: Optional[bool] = None,
        temperature: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
        files_context: Optional[Dict[str, Dict[str, Any]]] = None,
        workdir: Optional[str] = None,
        post_process: Literal["none", "summarize"] = "none",
        specific_instructions: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        _route_extra_instructions: str = "",
        final_target_language: Optional[str] = None,
        final_answer_length: Optional[str] = None,
        final_answer_instructions: Optional[str] = None,
        reset_tracer: bool = True,
    ) -> ReactAgentResponse:
        """
        Run the CodeAgent on a natural-language query.

        Args:
            query:
                The user question/task, e.g.
                "Analyze the GPA distribution from the uploaded CSV and show top 10 students."
            stream:
                If True, returns a streaming ReactAgentResponse (chunks via generator).
                If False, returns a ReactAgentResponse with the full final string.
            temperature:
                Overrides config.temperature if provided.
            max_new_tokens:
                Overrides config.max_new_tokens if provided.
            files_context:
                Mapping of filename -> {path, mime, size} for the current chat/thread.
                This is injected into the code tools via persistent memory.
            workdir:
                Directory where code execution should occur (plots, temp files).
                If None, uses CodeAgentConfig.default_workdir.
            post_process:
                - "none" (default): use the agent's / tools' output as-is.
                - "summarize": after the ReAct run completes (non-streaming), do one
                  extra LLM call to rewrite the raw answer into a cleaner explanation.
                  Ignored when stream=True.
            mode:
                - "delegate_final": CodeAgent is talking directly to the end user.
                  It should produce a complete, user-facing final answer.
                - "analysis_only": CodeAgent is being used as a sub-tool by another
                  agent. It should focus on correct computation and memory/extras,
                  and keep the final answer short and technical.
            reset_tracer:
                If True, resets tracing results before this run.

        Returns:
            ReactAgentResponse: includes final answer text and trace results.
        """
        # Inject runtime context into persistent memory so tools can see it.
        if files_context is not None:
            self.set_persistent_memory(files_context=files_context)
        if workdir is not None:
            self.set_persistent_memory(workdir=workdir)
        elif self.code_config.default_workdir is not None:
            self.set_persistent_memory(workdir=self.code_config.default_workdir)

        # Use base ReActAgent.run to handle streaming + core loop
        base_resp = super().run(
            query=query,
            stream=stream,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            specific_instructions=specific_instructions,
            context=context,
            _route_extra_instructions=_route_extra_instructions,
            final_target_language=final_target_language,
            final_answer_length=final_answer_length,
            final_answer_instructions=final_answer_instructions,
            reset_tracer=reset_tracer,
        )

        # If streaming, we can't easily post-process; just return.
        if stream:
            return base_resp

        # Optional final summarization pass (non-streaming only).
        # NOTE: we keep this independent of `mode` for now; you can later decide
        # to disable summarization for `analysis_only` if you want super-short outputs.
        if (
            post_process == "summarize"
            and self.code_config.enable_post_processing
            and base_resp.response
        ):
            summarized = self._summarize_final_answer(base_resp.response)
            return ReactAgentResponse(
                response=summarized,
                traces=base_resp.traces,
            )
        return base_resp

    # ---------- Optional post-processing ----------
    def _summarize_final_answer(self, raw_answer: str) -> str:
        """
        One extra LLM call to clean up / simplify the raw code-oriented answer.
        This is optional; used only when post_process='summarize'.
        """
        try:
            resp = self.llm.ask(
                system_prompt=(
                    "You are a senior data analyst. "
                    "Given the following technical/code-oriented answer, "
                    "rewrite it as a clear, concise explanation for a user. "
                    "Preserve all important numbers and conclusions."
                ),
                user_prompt=raw_answer,
                chat_history=[],
                temperature=max(0.2, float(self.code_config.temperature) - 0.3),
                max_new_tokens=min(int(self.code_config.max_new_tokens * 1.5), 2048),
                stream=False,
            )
            return resp.choices[0].message.content or raw_answer
        except Exception as e:
            logger.warning(f"[CodeAgent] Post-processing summarization failed: {e}")
            return raw_answer

    def _update_memory_from_extras(self, extras: Dict[str, Any], scope: str = "ephemeral", created_by: Optional[str] = None) -> None:
        """
        CodeAgent-specific adapter for ToolResponse.extras.

        - Normalizes known code-related extras (e.g. generated_plots).
        - Keeps any rich slots ({value, description, visible_to_llm, ...}) as-is.
        - Delegates actual memory writing to ReActAgent._update_memory_from_extras.
        """

        if not extras:
            return

        normalized: Dict[str, Any] = {}
        for key, raw in extras.items():
            # Special handling for common runtime-only extras
            if key == "generated_plots" and not (
                isinstance(raw, dict) and "value" in raw
            ):
                normalized[key] = {
                    "value": raw,
                    "description": "List of plot filenames generated by the last code execution.",
                    "visible_to_llm": False,
                    "scope": scope,
                    "created_by": created_by or "python_execute",
                }
                continue

            # If tool already provided a rich slot, respect it
            if isinstance(raw, dict) and "value" in raw:
                wrapped = dict(raw)
                wrapped.setdefault("scope", scope)
                wrapped.setdefault("created_by", created_by or "python_execute")
                normalized[key] = wrapped
            else:
                # Generic runtime-only extra
                normalized[key] = {
                    "value": raw,
                    "description": "",
                    "visible_to_llm": False,
                    "scope": scope,
                    "created_by": created_by or "python_execute",
                }
        ReActAgent._update_memory_from_extras(
            self,
            normalized,
            scope=scope,
            created_by=created_by,
        )