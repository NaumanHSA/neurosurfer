# neurosurfer/tools/python_exec_tool.py

from __future__ import annotations

import io
import textwrap
import traceback
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import logging

from neurosurfer.models.chat_models.base import BaseChatModel
from neurosurfer.tools.base_tool import BaseTool, ToolResponse
from neurosurfer.tools.tool_spec import ToolSpec, ToolParam, ToolReturn


PYTHON_EXEC_SYSTEM_PROMPT = """You are a careful Python 3 assistant.

You write minimal, correct Python code to solve the user's task.
The code will be executed in a controlled environment with:

- Standard Python 3
- Allowed libraries:
    - math
    - statistics
    - json
    - numpy as np
    - pandas as pd
    - matplotlib.pyplot as plt

You MUST respect these constraints:

- Do NOT import or use any other libraries (no torch, no sklearn, no http clients, no os / subprocess for shell access).
- Do NOT access the network.
- Do NOT read or write arbitrary files. You may ONLY access paths provided in the `files` mapping.

Available objects:

- `files`: a dict mapping filename (string) to an object:
    {
        "filename": {
            "path": "<absolute-path>",
            "mime": "<mime-type>",
            "size": <int bytes>
        },
        ...
    }

You may:
- Use `pd.read_csv(files["students.csv"]["path"])` for CSV.
- Open text files by path in read-only mode if absolutely necessary.

Your job:

1. Use Python to EXACTLY compute the answer to the user's task.
2. Store the final answer in a variable named `result`.

`result` can be:
- int, float, str
- list or dict
- pandas.Series or pandas.DataFrame
- or another simple printable object

If you create plots, you MUST:
- Use matplotlib (plt).
- Save them to PNG files in the current working directory with filenames like:
      "plot_1.png", "plot_2.png", ...
- Also store a list of created filenames in a variable called `generated_plots`,
  e.g.:
      generated_plots = ["plot_1.png", "plot_2.png"]

Output format:

- Respond with ONLY a single Python code block.
- No explanations outside the code block.
"""

PYTHON_EXEC_USER_PROMPT_TEMPLATE = """
User task:
{task}

Available files (from chat session):
{files_listing}

Guidelines:
- If you need a file, reference it via the `files` dict using its filename key.
- Prefer pandas for CSV, numpy/math/statistics for numeric work, and matplotlib for plots.
- Keep code as short and clear as possible.
"""


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

    # ------------- Public API -------------
    def __call__(
        self,
        task: str,
        file_names: Optional[List[str]] = None,
        files_context: Optional[Dict[str, Dict[str, Any]]] = None,
        workdir: Optional[str] = None,
        **_: Any,
    ) -> ToolResponse:
        """
        Execute the Python task.

        files_context format:
            {
                "students.csv": {"path": "/abs/path/students.csv", "mime": "text/csv", "size": 12345},
                ...
            }
        """
        files_context = files_context or {}
        file_names = file_names or list(files_context.keys())

        # Build a human-readable listing for the prompt
        files_listing = self._format_files_listing(files_context, file_names)

        code, result, generated_plots, error = self._run_code_with_retries(
            task=task,
            files_context=files_context,
            files_listing_for_prompt=files_listing,
            workdir=workdir,
        )

        if error is not None:
            answer = (
                "I attempted to solve this using Python code, but the code kept failing.\n\n"
                f"Last error:\n{error}"
            )
            if code and self.config.include_code_in_answer:
                answer += "\n\nGenerated code (may be buggy):\n```python\n"
                answer += code
                answer += "\n```"
            return ToolResponse(
                final_answer=True,
                results=answer,
                extras={"generated_plots": generated_plots or []},
            )

        # Success
        result_text = self._format_result(result)
        answer = f"Here are the results from executing Python code for your task:\n\n{result_text}"

        if generated_plots:
            answer += "\n\nGenerated plots (saved in this session's working directory):\n"
            for p in generated_plots:
                answer += f"- {p}\n"

        if self.config.include_code_in_answer and code:
            answer += "\n\nPython code used:\n```python\n"
            answer += code
            answer += "\n```"

        return ToolResponse(
            final_answer=True,
            results=answer,
            extras={"generated_plots": generated_plots or []},
        )

    # ------------- Internals -------------

    def _format_files_listing(
        self,
        files_context: Dict[str, Dict[str, Any]],
        file_names: List[str],
    ) -> str:
        if not files_context:
            return "(no files available)"

        lines: List[str] = []
        for name in file_names:
            meta = files_context.get(name)
            if not meta:
                continue
            size = meta.get("size")
            mime = meta.get("mime")
            path = meta.get("path")
            lines.append(f"- {name} (mime={mime}, size={size} bytes, path='{path}')")
        if not lines:
            return "(no matching files in context)"
        return "\n".join(lines)

    def _build_user_prompt(
        self,
        task: str,
        files_listing: str,
    ) -> str:
        tpl = PYTHON_EXEC_USER_PROMPT_TEMPLATE.format(
            task=task,
            files_listing=files_listing,
        )
        return textwrap.dedent(tpl).strip()

    def _ask_for_code(
        self,
        task: str,
        files_listing_for_prompt: str,
        previous_error: Optional[str] = None,
        previous_code: Optional[str] = None,
    ) -> str:
        user_prompt = self._build_user_prompt(task, files_listing_for_prompt)

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
        return self._extract_code_block(raw)

    @staticmethod
    def _extract_code_block(text: str) -> str:
        import re

        pattern = r"```(?:python)?\s*(.*?)```"
        m = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if m:
            return m.group(1).strip()
        return text.strip()

    def _run_code_with_retries(
        self,
        task: str,
        files_context: Dict[str, Dict[str, Any]],
        files_listing_for_prompt: str,
        workdir: Optional[str],
    ) -> tuple[Optional[str], Any, List[str], Optional[str]]:
        last_error: Optional[str] = None
        last_code: Optional[str] = None
        last_plots: List[str] = []

        for attempt in range(self.config.max_code_retries):
            code = self._ask_for_code(
                task=task,
                files_listing_for_prompt=files_listing_for_prompt,
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
                return code, result, plots, None
            except Exception as e:
                tb = traceback.format_exc()
                self.logger.error(
                    f"[python_execute] Code execution failed on attempt {attempt+1}: {e}\n{tb}"
                )
                last_error = f"{e}\n\nTraceback:\n{tb}"

        return last_code, None, last_plots, last_error

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
            local_ns: Dict[str, Any] = {}
            global_ns: Dict[str, Any] = {
                "math": math,
                "statistics": statistics,
                "json": json,
                "np": np,
                "pd": pd,
                "plt": plt,
                "files": files_context,
            }

            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                exec(  # nosec - you control the environment; see note below
                    code,
                    global_ns,
                    local_ns,
                )

            result = None
            if "result" in local_ns:
                result = local_ns["result"]
            elif "result" in global_ns:
                result = global_ns["result"]
            else:
                # Fall back to stdout if no result var
                output = buf.getvalue().strip()
                if not output:
                    raise RuntimeError(
                        "Code executed successfully but did not define `result` "
                        "and printed nothing."
                    )
                result = output

            generated_plots = []
            gp = local_ns.get("generated_plots") or global_ns.get("generated_plots")
            if isinstance(gp, list):
                generated_plots = [str(x) for x in gp]

            return result, generated_plots
        finally:
            os.chdir(cwd)

    def _format_result(self, result: Any) -> str:
        # Import here to avoid hard dependency at module import time
        try:
            import pandas as pd
        except Exception:  # pragma: no cover
            pd = None

        if pd is not None:
            import pandas as _pd

            if isinstance(result, _pd.DataFrame):
                if len(result) > self.config.max_table_rows:
                    result = result.head(self.config.max_table_rows)
                return result.to_markdown(index=False)
            if isinstance(result, _pd.Series):
                if len(result) > self.config.max_table_rows:
                    result = result.head(self.config.max_table_rows)
                return result.to_markdown()

        if isinstance(result, (list, dict)):
            import json

            return json.dumps(result, indent=2, default=str)

        return str(result)
