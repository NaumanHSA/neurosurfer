# neurosurfer/tools/python_exec_tool.py

from __future__ import annotations

import os, io
import traceback
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Literal, Generator

import logging

from neurosurfer.models.chat_models.base import BaseChatModel
from neurosurfer.tools.base_tool import BaseTool, ToolResponse
from neurosurfer.tools.tool_spec import ToolSpec, ToolParam, ToolReturn
from neurosurfer.server.schemas import ChatCompletionChunk, ChatCompletionRequest
from neurosurfer.utils.generator import consume, iterate_with_return

from .templates import PYTHON_EXEC_SYSTEM_PROMPT
from .utils import (
    build_error_extras, 
    build_memory_extras_for_result, 
    format_result, 
    format_files_listing,
    build_user_prompt,
    extract_code_block
)
from .config import PythonExecToolConfig

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
        name="python_code_generate_and_execute",
        description=(
            "Solve computational or data-analysis tasks by taking a plain-English query, "
            "automatically generating Python code, executing it, and returning the results. "
            "The caller must NEVER provide Python code."
        ),
        when_to_use=(
            "Use when precise computation, data manipulation, statistics, or plotting is required. "
            "Pass ONLY the goal in natural language; this tool handles code generation and execution."
        ),
        inputs=[
            ToolParam(
                name="task",
                type="string",
                description="Plain-English description of what should be computed (NO code, NO pseudo-code).",
                required=True,
                llm=True,
            ),
            ToolParam(
                name="file_names",
                type="array",
                description="Optional list of relevant uploaded filenames.",
                required=False,
                llm=False,
            ),
            ToolParam(
                name="files_context",
                type="object",
                description="Runtime-injected file metadata and paths.",
                required=False,
                llm=False,
            ),
            ToolParam(
                name="workdir",
                type="string",
                description="Runtime-injected working directory for execution.",
                required=False,
                llm=False,
            ),
        ],
        returns=ToolReturn(
            type="string",
            description="Human-readable summary of results and generated outputs (e.g., tables or plots).",
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

        - Returns ToolResponse (unchanged API)
        - ToolResponse.results is a *generator* of strings that:
            1) streams code between <special_token> markers
            2) then yields the final human-readable answer
        """
        final_answer_flag = kwargs.get("final_answer", False)
        files_context = files_context or {}
        file_names = file_names or list(files_context.keys())

        files_listing = format_files_listing(files_context, file_names)

        # Shared, mutable extras dict that the generator will fill in lazily
        extras: Dict[str, Any] = {
            "generated_plots": [],
        }

        def result_gen():
            """
            Lazy generator:
            - calls _run_code_with_retries only when first iterated
            - streams LLM code chunks
            - then yields formatted answer
            - fills `extras` dict as a side-effect
            """
            inner_gen = self._run_code_with_retries(
                task=task,
                files_context=files_context,
                files_listing_for_prompt=files_listing,
                workdir=workdir,
                context=context,
            )

            code: Optional[str] = None
            result: Any = None
            generated_plots: List[str] = []
            error: Optional[str] = None
            error_extras: Dict[str, Any] = {}
            error_category: Optional[str] = None

            first_code_chunk = True

            # 1) Forward LLM code chunks as they arrive
            for event in iterate_with_return(inner_gen):
                if event.kind == "yield":
                    chunk = event.value  # raw code delta (str)

                    # Start code block on first chunk
                    if self.config.include_code_in_answer and first_code_chunk:
                        first_code_chunk = False
                        yield "\n"

                    # Stream the chunk itself
                    if self.config.include_code_in_answer:
                        yield chunk

                else:
                    # Final result from _run_code_with_retries
                    (
                        code,
                        result,
                        generated_plots,
                        error,
                        error_extras,
                        error_category,
                    ) = event.value

            # 2) Close code block if we ever opened it
            if self.config.include_code_in_answer and not first_code_chunk:
                yield "\n"

            # 3) Error case
            if error is not None:
                if error_category == "missing_dependency":
                    first_line = error.splitlines()[0] if error else ""
                    answer = (
                        "I tried to run Python code for your task, but the environment is missing "
                        f"a required library (error: `{first_line}`).\n\n"
                        "This sandbox cannot install new packages (no pip/conda), and libraries are limited."
                    )
                else:
                    answer = (
                        "I attempted to solve this using Python code, but the code kept failing.\n\n"
                        f"Last error:\n{error}"
                    )

                # Yield final answer text
                yield answer

                # Fill extras lazily
                extras["generated_plots"] = generated_plots or []
                extras.update(error_extras)

                return  # end generator

            # 4) Success case
            result_text = format_result(
                result, max_table_rows=self.config.max_table_rows
            )
            answer = f"Here are the results from executing Python code for your task:\n\n{result_text}"

            if generated_plots:
                answer += "\n\nGenerated plots (saved in this session's working directory):\n"
                for p in generated_plots:
                    answer += f"- {p}\n"

            # Yield final natural-language answer
            yield answer

            # Fill extras lazily
            results_extras = build_memory_extras_for_result(result, style=self.config.memory_style)
            extras["generated_plots"] = generated_plots or []
            extras.update(results_extras)
            extras["python_last_code"] = {
                "value": code,
                "description": "Python code used in the last python_execute call.",
                "visible_to_llm": False,
            }

        # Return ToolResponse immediately.
        # Heavy work only starts when result_gen() is iterated.
        return ToolResponse(
            final_answer=final_answer_flag,
            results=result_gen(),
            extras=extras,
        )
            
    def _ask_for_code(
        self,
        task: str,
        files_listing_for_prompt: str,
        previous_error: Optional[str] = None,
        previous_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Generator[ChatCompletionChunk, None, None]:
        user_prompt = build_user_prompt(task, files_listing_for_prompt, context, max_snippet_len=self.config.max_context_length)
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
        
        # print(f"\n\n####################################\nUser Prompt for Code GEneration:\n{user_prompt}\n####################################\n\n")
        # Stream the code
        return self.llm.ask(
            system_prompt=PYTHON_EXEC_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=self.config.temperature,
            max_new_tokens=self.config.max_new_tokens,
            stream=True,
        )

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
            code_stream = self._ask_for_code(
                task=task,
                files_listing_for_prompt=files_listing_for_prompt,
                context=context,
                previous_error=last_error,
                previous_code=last_code,
            )
            last_code = ""
            yield self.config.soc + "\n"
            for chunk in code_stream:
                chunk = chunk.choices[0].delta.content or ""
                yield chunk
                last_code += chunk
            yield "\n" + self.config.eoc + "\n"
            last_code = extract_code_block(last_code)
            try:
                result, plots = self._exec_code(
                    code=last_code,
                    files_context=files_context,
                    workdir=workdir,
                )
                return last_code, result, plots, None, {}, None

            except Exception as e:
                tb = traceback.format_exc()
                last_category = self._classify_error(e)
                # self.logger.error(f"[python_execute] Code execution failed on attempt {attempt + 1}: {e}\n{tb}")
                last_error = f"{e}\n\nTraceback:\n{tb}"

                # Generic extras for ReAct
                error_extras = build_error_extras(e, tb)

                # Optionally: early exit for clearly non-code issues
                if isinstance(e, (FileNotFoundError, ImportError, KeyError)):
                    break

                # For hard environment errors, retrying code is pointless
                if last_category == "missing_dependency" or last_category == "missing_file" or last_category == "key_error":
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
        if isinstance(e, ModuleNotFoundError) or isinstance(e, ImportError) or "No module named" in msg:
            return "missing_dependency"
        if "not available and cannot be installed in this environment" in msg:
            return "missing_dependency"
        if isinstance(e, FileNotFoundError) or "No such file or directory" in msg:
            return "missing_file"
        if isinstance(e, KeyError):
            return "key_error"
        return "generic"