from dataclasses import dataclass, field
from neurosurfer.config import config
from .retry import RetryPolicy

@dataclass
class ReActConfig:
    temperature: float = config.base_model.temperature
    max_new_tokens: int = config.base_model.max_new_tokens
    allow_input_pruning: bool = True     # drop extra inputs not in ToolSpec
    repair_with_llm: bool = True         # ask LLM to repair invalid Action
    retry: RetryPolicy = field(default_factory=RetryPolicy)
    skip_special_tokens: bool = False
    return_stream_by_default: bool = False
    log_internal_thoughts: bool = True
    return_internal_thoughts: bool = False