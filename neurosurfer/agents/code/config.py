# neurosurfer/agents/code/config.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Literal, Dict, List

from neurosurfer.agents.react.config import ReActConfig


@dataclass
class CodeAgentConfig(ReActConfig):
    """
    Configuration for CodeAgent.

    Inherits all ReActConfig knobs (temperature, max_new_tokens, retries, etc.)
    and adds a few code-specific ones.
    """

    # Name to show in tracing / logs
    agent_name: str = "CodeAgent"

    # If True, CodeAgent.run(..., post_process='summarize') will
    # summarize the raw tool/agent output with one extra LLM call.
    enable_post_processing: bool = False

    # Default working directory for python_execute; can be overridden per-run.
    default_workdir: Optional[str] = None

    # When True, CodeAgent will try to be "inspection-first":
    #   - For ambiguous data tasks, it is encouraged (via prompt) to:
    #       1) inspect files/columns
    #       2) then run more complex code.
    encourage_multistep_planning: bool = True

    # Future extension point: default to returning "raw" code tool output
    # vs natural-language answers from the tool.
    default_return_raw: bool = False

    # Forcing memory keys for specific tools
    forced_memory_keys: Dict[str, List[str]] = field(default_factory=lambda: {
        "python_execute": [
            "python_last_result_summary", 
            "python_last_error"
        ]
    })

