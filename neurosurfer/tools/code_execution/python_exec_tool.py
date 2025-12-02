# neurosurfer/tools/python_exec_tool.py

from __future__ import annotations

import os, io
import traceback
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import logging

from neurosurfer.models.chat_models.base import BaseChatModel
from neurosurfer.tools.base_tool import BaseTool, ToolResponse
from neurosurfer.tools.tool_spec import ToolSpec, ToolParam, ToolReturn

from .templates import PYTHON_EXEC_SYSTEM_PROMPT
from .utils import (
    build_error_extras, 
    build_memory_extras_for_result, 
    format_result, 
    format_files_listing,
    build_user_prompt,
    extract_code_block
)


@dataclass
class PythonExecToolConfig:
    max_code_retries: int = 3
    include_code_in_answer: bool = True
    max_table_rows: int = 20   # for DataFrame/Series pretty printing


class PythonExecTool(BaseTool):
    """
    Generic Python execution tool.

    It asks the LLM to generate Python code for a given task, with access to
    a limited set of libraries and a `files` mapping (uploaded files in the chat).
    Then it executes that code with retries.

    Use cases:
    - tabular stats (CSV via pandas)
    - light data munging / filtering
    - simple numerical analysis (numpy/statistics)
    - basic plotting (matplotlib)
    """

    spec = ToolSpec(
        name="python_execute",
        description=(
            "Execute Python code to solve tasks that require precise computation, "
            "data analysis, or plotting, using common libraries (numpy, pandas, matplotlib)."
        ),
        when_to_use=(
            "Use this tool whenever the question clearly requires programmatic computation "
            "or data manipulation beyond what you can reliably do in your head. "
            "Examples: computing statistics from uploaded CSVs, filtering rows, "
            "aggregating values, generating plots, or transforming data."
            "Do NOT use this tool for heavy machine learning training or complex ML pipelines."
        ),
        inputs=[
            # Natural-language specification of the task (LLM-provided).
            ToolParam(
                name="task",
                type="string",
                description="Natural-language description of what should be computed with Python.",
                required=True,
                llm=True,
            ),
            # Optional hint of which filenames are relevant (LLM can choose).
            ToolParam(
                name="file_names",
                type="array",
                description="Optional list of filenames from the uploaded files that are relevant.",
                required=False,
                llm=False,
            ),
            # Runtime-only: mapping filename -> {path, mime, size}, injected by backend.
            ToolParam(
                name="files_context",
                type="object",
                description=(
                    "Mapping of available filenames to their metadata and absolute paths. "
                    "This is provided by the runtime, not by the LLM."
                ),
                required=False,
                llm=False,
            ),
            # Runtime-only: working directory path for code execution.
            ToolParam(
                name="workdir",
                type="string",
                description="Working directory where code will be executed and plots saved.",
                required=False,
                llm=False,
            ),
        ],
        returns=ToolReturn(
            type="string",
            description=(
                "A human-readable answer describing the results of the Python execution. "
                "It may include small tables of values and references to generated plot filenames."
            ),
        ),
    )

    def __init__(
        self,
        llm: BaseChatModel,
        config: Optional[PythonExecToolConfig] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        super().__init__()
        self.llm = llm
        self.config = config or PythonExecToolConfig()
        self.logger = logger or logging.getLogger(__name__)

    def __call__(
        self,
        task: str,
        file_names: Optional[List[str]] = None,
        files_context: Optional[Dict[str, Dict[str, Any]]] = None,
        workdir: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> ToolResponse:
        """
        Execute the Python task.

        files_context format:
            {
                "students.csv": {"path": "/abs/path/students.csv", "mime": "text/csv", "size": 12345},
                ...
            }
        """
        final_answer = kwargs.get("final_answer", False)
        files_context = files_context or {}
        file_names = file_names or list(files_context.keys())

        # Build a human-readable listing for the prompt
        files_listing = format_files_listing(files_context, file_names)
        code, result, generated_plots, error, error_extras, error_category = self._run_code_with_retries(
            task=task,
            files_context=files_context,
            files_listing_for_prompt=files_listing,
            workdir=workdir,
            context=context,
        )

        if error is not None:
            answer = (
                "I attempted to solve this using Python code, but the code kept failing.\n\n"
                f"Last error:\n{error}"
            )
            # If it's a missing dependency, treat this as a hard stop.
            if error_category == "missing_dependency":
                # Extract the first line for a concise message
                first_line = error.splitlines()[0] if error else ""
                answer = (
                    "I tried to run Python code for your task, but the environment is missing "
                    f"a required library (error: `{first_line}`).\n\n"
                    "This sandbox cannot install new packages (no pip/conda), and libraries are limited."
                )
            
            if code and self.config.include_code_in_answer:
                answer += "\n\nGenerated code (may be buggy):\n```python\n"
                answer += code
                answer += "\n```"

            extras: Dict[str, Any] = {"generated_plots": generated_plots or []}
            extras.update(error_extras)
            return ToolResponse(final_answer=final_answer, results=answer, extras=extras)
        
        # Success
        result_text = format_result(result, max_table_rows=self.config.max_table_rows)
        answer = f"Here are the results from executing Python code for your task:\n\n{result_text}"

        if generated_plots:
            answer += "\n\nGenerated plots (saved in this session's working directory):\n"
            for p in generated_plots:
                answer += f"- {p}\n"

        if self.config.include_code_in_answer and code:
            answer += "\n\nPython code used:\n```python\n"
            answer += code
            answer += "\n```"

        # build extras that can be used as memory for the next tool call
        extras: Dict[str, Any] = {"generated_plots": generated_plots or []}
        extras.update(build_memory_extras_for_result(result))

        # add code to the extras
        extras["python_last_code"] = {
            "value": code,
            "description": "Python code used in the last python_execute call.",
            "visible_to_llm": False,
        }
        return ToolResponse(final_answer=final_answer, results=answer, extras=extras)

    def _ask_for_code(
        self,
        task: str,
        files_listing_for_prompt: str,
        previous_error: Optional[str] = None,
        previous_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        user_prompt = build_user_prompt(task, files_listing_for_prompt, context)
        if previous_error:
            repair_note = (
                "\n\nThe previous code attempt failed with this error:\n"
                f"{previous_error}\n\n"
                "Please FIX the code so that it runs without errors in the given environment. "
                "You may reuse parts of the previous code if helpful."
            )
            if previous_code:
                repair_note += "\n\nPrevious code:\n```python\n" + previous_code + "\n```"
            user_prompt += repair_note

        resp = self.llm.ask(
            system_prompt=PYTHON_EXEC_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=0.1,
            max_new_tokens=512,
            stream=False,
        )
        raw = resp.choices[0].message.content or ""
        return extract_code_block(raw)

    def _run_code_with_retries(
        self,
        task: str,
        files_context: Dict[str, Dict[str, Any]],
        files_listing_for_prompt: str,
        workdir: Optional[str],
        context: Optional[Dict[str, Any]] = None,
    ) -> tuple[Optional[str], Any, List[str], Optional[str], Dict[str, Any], Optional[str]]:
        last_error: Optional[str] = None
        last_code: Optional[str] = None
        last_category: Optional[str] = None
        last_plots: List[str] = []
        error_extras: Dict[str, Any] = {}

        for attempt in range(self.config.max_code_retries):
            code = self._ask_for_code(
                task=task,
                files_listing_for_prompt=files_listing_for_prompt,
                context=context,
                previous_error=last_error,
                previous_code=last_code,
            )
            last_code = code

            try:
                result, plots = self._exec_code(
                    code=code,
                    files_context=files_context,
                    workdir=workdir,
                )
                return code, result, plots, None, {}, None

            except Exception as e:
                tb = traceback.format_exc()
                last_category = self._classify_error(e)
                self.logger.error(
                    f"[python_execute] Code execution failed on attempt {attempt + 1}: {e}\n{tb}"
                )
                last_error = f"{e}\n\nTraceback:\n{tb}"

                # Generic extras for ReAct
                error_extras = build_error_extras(e, tb)

                # Optionally: early exit for clearly non-code issues
                if isinstance(e, (FileNotFoundError, ImportError)):
                    break

                # For hard environment errors, retrying code is pointless
                if last_category == "missing_dependency":
                    break

        return last_code, None, last_plots, last_error, error_extras, last_category


    def _exec_code(
        self,
        code: str,
        files_context: Dict[str, Dict[str, Any]],
        workdir: Optional[str],
    ) -> tuple[Any, List[str]]:
        """
        Execute LLM-generated code with a constrained namespace.

        - Exposes: math, statistics, json, numpy, pandas, matplotlib.pyplot
        - Exposes: files (mapping filename -> meta dict)
        - If workdir is provided, we chdir into it for the duration.
        """
        import math
        import statistics
        import json
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import os
        import contextlib

        cwd = os.getcwd()
        if workdir:
            os.makedirs(workdir, exist_ok=True)
            os.chdir(workdir)

        try:
            # Single shared sandbox for both globals and locals.
            # This avoids NameError issues for functions used inside
            # list comprehensions, generators, etc.
            sandbox: Dict[str, Any] = {
                "math": math,
                "statistics": statistics,
                "json": json,
                "np": np,
                "pd": pd,
                "plt": plt,
                "files": files_context,
            }
            # Explicitly expose builtins (optional but clearer)
            sandbox["__builtins__"] = __builtins__

            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                exec(  # nosec - you control the environment; see note above
                    code,
                    sandbox,
                    sandbox,
                )

            # Prefer an explicit `result` variable if present
            if "result" in sandbox:
                result = sandbox["result"]
            else:
                # Fall back to stdout if no result var
                output = buf.getvalue().strip()
                if not output:
                    raise RuntimeError(
                        "Code executed successfully but did not define `result` "
                        "and printed nothing."
                    )
                result = output

            generated_plots: List[str] = []
            gp = sandbox.get("generated_plots")
            if isinstance(gp, list):
                generated_plots = [str(x) for x in gp]

            return result, generated_plots
        finally:
            os.chdir(cwd)

    def _classify_error(self, e: Exception) -> str:
        """
        Rough error categories, used to decide whether retrying makes sense.
        """
        msg = str(e)
        if isinstance(e, ModuleNotFoundError) or "No module named" in msg:
            return "missing_dependency"
        if "not available and cannot be installed in this environment" in msg:
            return "missing_dependency"
        return "generic"