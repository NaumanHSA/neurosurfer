from dataclasses import dataclass, field
from neurosurfer.config import config
from .retry import RetryPolicy
from typing import Literal

@dataclass
class ReActConfig:
    # Mode to run the CodeAgent in
    # "delegate_final": CodeAgent is talking directly to the end user. It should produce a complete, user-facing final answer.
    # "analysis_only": CodeAgent is being used as a sub-tool by another agent. It should focus on correct computation and memory/extras, and keep the final answer short and technical.
    mode: Literal["delegate_final", "analysis_only"] = "delegate_final"

    temperature: float = config.base_model.temperature
    max_new_tokens: int = config.base_model.max_new_tokens
    allow_input_pruning: bool = True     # drop extra inputs not in ToolSpec
    repair_with_llm: bool = True         # ask LLM to repair invalid Action
    retry: RetryPolicy = field(default_factory=RetryPolicy)
    skip_special_tokens: bool = False
    return_stream_by_default: bool = False
    log_internal_thoughts: bool = True
    return_internal_thoughts: bool = False