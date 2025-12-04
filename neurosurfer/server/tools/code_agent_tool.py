# neurosurfer/tools/code_agent_tool.py

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Literal

from neurosurfer.models.chat_models import BaseChatModel
from neurosurfer.tools.base_tool import BaseTool, ToolResponse
from neurosurfer.tools.tool_spec import ToolSpec, ToolParam, ToolReturn
from neurosurfer.tools import Toolkit
from neurosurfer.agents.code.agent import CodeAgent, CodeAgentConfig
from neurosurfer.agents.react.types import ReactAgentResponse, ToolCall
from neurosurfer.tracing import Tracer


class CodeAgentTool(BaseTool):
    """
    Tool wrapper around the CodeAgent.

    This lets a higher-level ReAct agent treat the entire CodeAgent
    (python execution, files_context, working memory, etc.) as a single tool.

    Behavior by mode (CodeAgent.config.mode):

    - delegate_final:
        CodeAgent is acting as a primary agent.
        -> This tool returns ONLY the CodeAgent's final answer as text
           (no internal steps in the main text), but internal tool_calls
           are still available in ToolResponse.extras["code_agent_tool_calls"].

    - analysis_only (default when used as a tool):
        CodeAgent is acting as a sub-agent.
        -> This tool returns a textual "Tool Execution Steps" log plus
           a short final snippet from the CodeAgent.
        -> Full structured tool_calls are still in extras.
    """

    spec = ToolSpec(
        name="code_agent_run",
        description=(
            "Use the CodeAgent to answer questions that require Python-based "
            "analysis or computation over uploaded files (e.g. CSV, text, etc.). "
            "The CodeAgent can inspect file structure, compute statistics, create plots, "
            "and return summarized results."
        ),
        when_to_use=(
            "Use this tool whenever the query clearly needs precise computation or "
            "data manipulation that is hard to do in your head, especially when the "
            "user has uploaded structured data (CSV, tables, logs). "
            "Examples: counting rows, aggregating metrics, computing correlations, "
            "or generating plots from uploaded files. "
            "Do NOT use this tool for general Q&A that does not depend on code or files."
        ),
        inputs=[
            ToolParam(
                name="query",
                type="string",
                description=(
                    "Natural-language description of what the CodeAgent should do. "
                    "You can mention uploaded files by name and describe the "
                    "analysis or computation needed."
                ),
                required=True,
                llm=True,
            ),
            ToolParam(
                name="files_hint",
                type="array",
                description=(
                    "Optional list of filenames that are especially relevant. "
                    "If omitted, the runtime may pass all available files_context."
                ),
                required=False,
                llm=True,
            ),
            ToolParam(
                name="files_context",
                type="object",
                description=(
                    "Mapping of available filenames to their metadata and absolute paths. "
                    "Injected by the backend, NOT by the LLM. Example:\n"
                    "{\n"
                    '  \"Student Degree College Data.csv\": {\n'
                    '    \"path\": \"/abs/path/Student Degree College Data.csv\",\n'
                    '    \"mime\": \"text/csv\",\n'
                    '    \"size\": 422368\n'
                    "  }\n"
                    "}"
                ),
                required=False,
                llm=False,
            ),
            ToolParam(
                name="workdir",
                type="string",
                description=(
                    "Optional working directory where the CodeAgent (and underlying "
                    "python_execute tool) can write temporary outputs such as plots."
                ),
                required=False,
                llm=False,
            ),
            ToolParam(
                name="post_process",
                type="string",
                description=(
                    "Post-processing mode for the CodeAgent's answer. "
                    "If omitted, defaults to 'none'."
                ),
                required=False,
                llm=False,
            ),
        ],
        returns=ToolReturn(
            type="string",
            description=(
                "A human-readable answer produced by the CodeAgent. "
                "In delegate_final mode: the full user-facing answer. "
                "In analysis_only mode: a log of tool execution steps plus a short "
                "final snippet from the CodeAgent. "
                "Structured extras (e.g. internal tool_calls) are attached in "
                "ToolResponse.extras['code_agent_tool_calls']."
            ),
        ),
    )

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
        """
        Args:
            llm: BaseChatModel instance to use for the CodeAgent.
            toolkit: Optional Toolkit to use for the CodeAgent.
            config: Optional CodeAgentConfig to use for the CodeAgent.
            tracer: Optional Tracer to use for the CodeAgent.
            log_traces: Whether to log traces for the CodeAgent.
            logger: Optional logger.
        """
        super().__init__()
        self.logger = logger or logging.getLogger(__name__)

        # Default config for CodeAgent when used as a tool:
        # - skip_special_tokens: True (common for agents)
        # - mode: "analysis_only" (sub-agent behavior)
        if config is None:
            config = CodeAgentConfig(skip_special_tokens=True)
        if getattr(config, "mode", None) is None:
            # Only override if caller didn't explicitly choose a mode
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
    def __call__(
        self,
        query: str,
        files_hint: Optional[List[str]] = None,
        files_context: Optional[Dict[str, Dict[str, Any]]] = None,
        workdir: Optional[str] = None,
        post_process: Literal["none", "summarize"] = "none",
        **kwargs: Any,
    ) -> ToolResponse:
        """
        Run the CodeAgent as a tool.

        The ReAct agent supplies `query` (and optionally files_hint).
        The runtime supplies `files_context`, `workdir`, and `post_process`.

        We forward these to CodeAgent.run(...) and adapt its result
        into a ToolResponse, including a structured tool_calls list in extras.
        """
        final_answer_flag: bool = bool(kwargs.get("final_answer", False))

        files_context = files_context or {}
        files_hint = files_hint or []

        # Optionally narrow the files_context based on files_hint
        if files_hint:
            filtered: Dict[str, Dict[str, Any]] = {
                name: meta
                for name, meta in files_context.items()
                if name in files_hint
            }
            if filtered:
                files_context = filtered

        # Call the underlying CodeAgent.
        agent_result: ReactAgentResponse = self.code_agent.run(
            query=query,
            files_context=files_context,
            workdir=workdir,
            post_process=post_process,
            stream=False,
        )

        # Base text answer from CodeAgent
        if hasattr(agent_result, "response"):
            text_answer = str(agent_result.response)
        else:
            text_answer = str(agent_result)

        # Extract tool_calls and serialize them
        tool_calls: List[ToolCall] = getattr(agent_result, "tool_calls", []) or []

        # Structured tool_calls for extras
        tool_calls_payload: List[Dict[str, Any]] = []
        for idx, call in enumerate(tool_calls, start=1):
            tool_calls_payload.append(
                {
                    "index": idx,
                    "tool": call.tool,
                    "inputs": call.inputs,
                    "final_answer": call.final_answer,
                    "memory_keys": call.memory_keys,
                    "rationale": call.rationale,
                    "output": call.output,
                }
            )

        # Decide what to put in results based on mode
        agent_mode = getattr(self.code_agent.code_config, "mode", "delegate_final")
        if agent_mode == "analysis_only":
            # Pretty string for steps (for analysis_only mode)
            steps_str = self._format_tool_steps(tool_calls)
            # Sub-agent behavior: expose internal steps + short final snippet
            pieces: List[str] = []

            if steps_str:
                pieces.append("Tool Execution Steps (CodeAgent internal):")
                pieces.append(steps_str)

            if text_answer.strip():
                pieces.append("")
                pieces.append("CodeAgent final snippet:")
                pieces.append(text_answer.strip())

            combined_answer = "\n".join(pieces).strip() or text_answer
        else:
            # delegate_final: CodeAgent is the one talking to the end user
            combined_answer = text_answer

        extras: Dict[str, Any] = {
            "code_agent_mode": agent_mode,
            "code_agent_tool_calls": tool_calls_payload,
        }

        return ToolResponse(
            final_answer=final_answer_flag,
            results=combined_answer,
            extras=extras,
        )

    # ------------- Internals -------------

    def _format_tool_steps(self, tool_calls: List[ToolCall], max_output_chars: int = None) -> str:
        """
        Create a readable textual history of CodeAgent's internal tool calls.

        Example:
        Tool Execution Steps (CodeAgent internal):
        Step 1: Tool Call
          Tool: python_execute
          Task: Load the 'Student Degree College Data.csv' file and display the first few rows...
          Output:
            Here are the results from executing Python code...
        """
        if not tool_calls:
            return ""

        lines: List[str] = []
        for idx, call in enumerate(tool_calls, start=1):
            lines.append(f"Step {idx}: Tool Call")
            lines.append(f"  Tool: {call.tool}")

            task = (
                call.inputs.get("task")
                or call.inputs.get("query")
                or ""
            )
            if task:
                lines.append(f"  Task: {task}")

            # if call.memory_keys:
            #     lines.append(f"  memory_keys: {call.memory_keys}")

            if call.output:
                out = str(call.output).strip()
                if max_output_chars is not None and len(out) > max_output_chars:
                    out = out[:max_output_chars] + "... [truncated]"
                lines.append("  Output:")
                # indent multi-line output
                indented = "    " + out.replace("\n", "\n    ")
                lines.append(indented)

            lines.append("")  # blank line between steps

        return "\n".join(lines).rstrip()
