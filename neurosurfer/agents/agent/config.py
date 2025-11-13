from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict

@dataclass
class RouterRetryPolicy:
    """Retry tuning for routing + tool execution."""
    max_route_retries: int = 2
    max_tool_retries: int = 1
    backoff_sec: float = 0.7  # linear backoff

@dataclass
class AgentConfig:
    """
    Top-level configuration for the Agent.
    """
    # Routing:
    allow_input_pruning: bool = True    # drop unknown inputs not in ToolSpec
    repair_with_llm: bool = True        # ask LLM to repair invalid routing/inputs
    strict_tool_call: bool = False      # router must output JSON; else can answer in plain text
    # synonyms: Dict[str, Dict[str, str]] = field(default_factory=dict)  # field -> {from: to}

    # LLM defaults:
    temperature: float = 0.7
    max_new_tokens: int = 512
    return_stream_by_default: bool = False

    # Retries:
    retry: RouterRetryPolicy = field(default_factory=RouterRetryPolicy)

    # Structured-output options:
    strict_json: bool = True                  # enforce RFC 8259 JSON
    max_repair_attempts: int = 1              # for malformed JSON repairs
