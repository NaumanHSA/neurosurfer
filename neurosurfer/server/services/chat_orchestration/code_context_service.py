from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Literal, Generator

from neurosurfer.models.chat_models import BaseChatModel
from neurosurfer.tools import Toolkit
from neurosurfer.agents.code.agent import CodeAgent, CodeAgentConfig
from neurosurfer.agents.react.types import ReactAgentResponse, ToolCall
from neurosurfer.tracing import Tracer


LOGGER = logging.getLogger(__name__)


@dataclass
class CodeAgentContextResult:
    """
    High-level result from running the CodeAgent as a service.

    - context_block: text you can feed directly into FinalAnswerGenerator.
    - mode: CodeAgent mode ("analysis_only" or "delegate_final").
    - tool_calls: structured internal tool calls for debugging / memory.
    - raw_answer: whatever CodeAgent returned as its `.response`.
    - extras: additional metadata you may want to persist or log.
    """
    context_block: str
    mode: str
    tool_calls: List[ToolCall]
    raw_answer: str
    extras: Dict[str, Any]


class CodeAgentService:
    """
    Service façade around CodeAgent (no BaseTool / ToolSpec).

    It exposes a simple `.run_for_context(...)` API that returns a
    CodeAgentContextResult, which you can pass to FinalAnswerGenerator.
    """

    def __init__(
        self,
        *,
        llm: BaseChatModel,
        toolkit: Optional[Toolkit] = None,
        config: Optional[CodeAgentConfig] = None,
        tracer: Optional[Tracer] = None,
        log_traces: bool = False,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.logger = logger or LOGGER

        # Default config when used as a sub-pipeline
        if config is None:
            config = CodeAgentConfig(skip_special_tokens=True)
        if getattr(config, "mode", None) is None:
            # When used under the main chat agent we usually want analysis_only
            config.mode = "analysis_only"

        self.code_agent = CodeAgent(
            llm=llm,
            toolkit=toolkit,
            config=config,
            tracer=tracer,
            log_traces=log_traces,
            logger_=self.logger,
        )

    # ------------- Public API -------------

    def run_for_context(
        self,
        *,
        query: str,
        files_context: Optional[Dict[str, Dict[str, Any]]] = None,
        files_hint: Optional[List[str]] = None,
        workdir: Optional[str] = None,
        post_process: Literal["none", "summarize"] = "none",
        stream: bool = False,
    ) -> CodeAgentContextResult:
        """
        Run CodeAgent and return a context block suitable for FinalAnswerGenerator.

        - query: natural language instruction for CodeAgent.
        - files_context: mapping filename -> {path, mime, size, ...}.
        - files_hint: optional list of filenames to narrow files_context.
        - workdir: optional cwd for python_execute.
        - post_process: optional mode for CodeAgent itself.
        """
        files_context = files_context or {}
        files_hint = files_hint or []

        # Optionally narrow the files_context based on files_hint
        if files_hint:
            filtered: Dict[str, Dict[str, Any]] = {
                name: meta for name, meta in files_context.items() if name in files_hint
            }
            if filtered:
                files_context = filtered
        agent_result: ReactAgentResponse = self.code_agent.run(
            query=query,
            files_context=files_context,
            workdir=workdir,
            post_process=post_process,
            stream=stream,
        )
        
        agent_response = ""
        if isinstance(agent_result.response, Generator):
            for chunk in agent_result.response:
                agent_response += chunk
        else:
            agent_response = str(agent_result.response)

        # Tool calls
        tool_calls: List[ToolCall] = getattr(agent_result, "tool_calls", []) or []

        # Build a human-readable context block from internal steps
        context_block = self._build_context_block(raw_answer=agent_response, tool_calls=tool_calls)

        agent_mode = getattr(self.code_agent.code_config, "mode", "delegate_final")

        extras: Dict[str, Any] = {
            "code_agent_mode": agent_mode,
            "code_agent_tool_calls_count": len(tool_calls),
        }

        return CodeAgentContextResult(
            context_block=context_block,
            mode=agent_mode,
            tool_calls=tool_calls,
            raw_answer=agent_response,
            extras=extras,
        )

    # ------------- Internals -------------

    def _build_context_block(
        self,
        *,
        raw_answer: str,
        tool_calls: List[ToolCall],
        max_output_chars: Optional[int] = 8000,
    ) -> str:
        """
        Build a context block that FinalAnswerGenerator can consume.

        We reuse the "Tool Execution Steps" formatting from CodeAgentTool,
        but treat it purely as CONTEXT instead of user-facing text.
        """
        steps_str = self._format_tool_steps(tool_calls, max_output_chars=max_output_chars)
        raw_answer = (raw_answer or "").strip()

        parts: List[str] = []
        if steps_str:
            parts.append("CODE AGENT EXECUTION LOG:")
            parts.append(steps_str)

        if raw_answer:
            parts.append("")
            parts.append("CODE AGENT FINAL SNIPPET:")
            parts.append(raw_answer)

        context_block = "\n".join(parts).strip()
        if not context_block:
            context_block = "(CodeAgent did not produce any visible output.)"

        return context_block

    def _format_tool_steps(
        self,
        tool_calls: List[ToolCall],
        max_output_chars: Optional[int] = None,
    ) -> str:
        if not tool_calls:
            return ""

        lines: List[str] = []
        for idx, call in enumerate(tool_calls, start=1):
            lines.append(f"Step {idx}: Tool Call")
            lines.append(f"  Tool: {call.tool}")

            task = call.inputs.get("task") or call.inputs.get("query") or ""
            if task:
                lines.append(f"  Task: {task}")

            if call.output:
                out = str(call.output).strip()
                if max_output_chars is not None and len(out) > max_output_chars:
                    out = out[:max_output_chars] + "... [truncated]"
                lines.append("  Output:")
                indented = "    " + out.replace("\n", "\n    ")
                lines.append(indented)

            lines.append("")  # blank line between steps

        return "\n".join(lines).rstrip()
