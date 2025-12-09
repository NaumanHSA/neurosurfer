from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, Optional, List, Literal

# Config & Result Types
@dataclass
class MainWorkflowConfig:
    """
    Configuration for the main chat workflow.

    - default_language / default_answer_length: fallbacks if the gate doesn't
      specify them.
    - enable_rag / enable_code: feature switches (useful for environments
      without vector store or python execution).
    - log_traces: whether to log high-level workflow decisions.
    """
    id: str = "main_chat_workflow"
    default_language: str = "english"        # "english" | "arabic"
    default_answer_length: str = "detailed"  # "short" | "medium" | "detailed"

    enable_rag: bool = True
    enable_code: bool = True

    # If True, we log routing decisions and context summaries.
    log_traces: bool = True

    # Optional: max chars for context passed to FinalAnswerGenerator
    max_context_chars: int = 16000

    max_history_chars: int = 12000
    temperature: float = 0.7
    max_new_tokens: int = 1024