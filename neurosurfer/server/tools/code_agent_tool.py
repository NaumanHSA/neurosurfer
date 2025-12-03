from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Literal

from neurosurfer.models.chat_models import BaseChatModel
from neurosurfer.tools import BaseTool, Toolkit, ToolResponse, ToolSpec, ToolParam, ToolReturn
from neurosurfer.agents.code.agent import CodeAgent, CodeAgentConfig
from neurosurfer.tracing import Tracer


class CodeAgentTool(BaseTool):
    """
    Tool wrapper around the CodeAgent.

    This lets a higher-level ReAct agent treat the entire CodeAgent
    (python execution, files_context, working memory, etc.) as a single tool.

    Typical use cases:
    - "Using the uploaded CSV, compute X / plot Y / transform Z"
    - "Run code to analyze the uploaded data and summarize the results"
    - Any query where *programmatic* reasoning over user files is needed.
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
            # LLM-provided
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
            # Runtime-provided
            ToolParam(
                name="files_context",
                type="object",
                description=(
                    "Mapping of available filenames to their metadata and absolute paths. "
                    "Injected by the backend, NOT by the LLM. Example:\n"
                    "{\n"
                    '  "Student Degree College Data.csv": {\n'
                    '    "path": "/abs/path/Student Degree College Data.csv",\n'
                    '    "mime": "text/csv",\n'
                    '    "size": 422368\n'
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
                "A human-readable answer produced by the CodeAgent, possibly including "
                "summaries of computations, small tables, and references to generated plots. "
                "Structured extras (e.g. python_last_result_summary, python_last_error, "
                "generated_plots) are attached in the ToolResponse.extras for memory."
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
        self.code_agent = CodeAgent(
            llm=llm, 
            toolkit=toolkit,
            config=config, 
            tracer=tracer, 
            log_traces=log_traces,
            logger_=self.logger
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
        into a ToolResponse.
        """
        final_answer_flag: bool = bool(kwargs.get("final_answer", False))

        files_context = files_context or {}
        files_hint = files_hint or []

        # Optionally narrow the files_context based on files_hint
        if files_hint:
            filtered: Dict[str, Dict[str, Any]] = {
                name: meta for name, meta in files_context.items() if name in files_hint
            }
            # Only shrink if we actually matched something
            if filtered:
                files_context = filtered

        # Call the underlying CodeAgent.
        # Your CodeAgent.run signature (from your example):
        #   run(query, files_context=None, workdir=None, post_process="none", stream=False)
        agent_result = self.code_agent.run(
            query=query,
            files_context=files_context,
            workdir=workdir,
            post_process=post_process,
            stream=False,
        )

        # Try to extract a nice text answer + extras generically.
        # We assume CodeAgent returns an object with `.response` (str-like)
        # and possibly `.extras` (dict). We keep this defensive.
        text_answer: str
        extras: Dict[str, Any]

        if hasattr(agent_result, "response"):
            text_answer = str(getattr(agent_result, "response"))
        else:
            # Fallback: best-effort stringification
            text_answer = str(agent_result)

        extras_obj: Any = getattr(agent_result, "extras", None)
        if isinstance(extras_obj, dict):
            extras = extras_obj
        else:
            extras = {}
        return ToolResponse(
            final_answer=final_answer_flag,
            results=text_answer,
            extras=extras,
        )
