"""Provider-neutral LLM layer.

Public surface: canonical types, the ``Provider`` protocol, capabilities, retry
helpers, token math, and the registry that builds the active provider.
"""

from .base import Provider
from .capabilities import (
    ProviderCapabilities,
    anthropic_capabilities,
    openai_capabilities,
)
from .registry import build_provider
from .retry import is_context_overflow_error, is_retryable_error, with_retry
from .tokens import (
    auto_compact_threshold,
    effective_window,
    estimate_messages_tokens,
)
from .types import (
    CanonicalResponse,
    ContentBlock,
    Done,
    GenerationConfig,
    Message,
    StreamEvent,
    TextBlock,
    TextDelta,
    ThinkingBlock,
    ThinkingDelta,
    ToolResultBlock,
    ToolSchema,
    ToolUseArgsDelta,
    ToolUseBlock,
    ToolUseStart,
    Usage,
)

__all__ = [
    "Provider",
    "ProviderCapabilities",
    "anthropic_capabilities",
    "openai_capabilities",
    "build_provider",
    "with_retry",
    "is_retryable_error",
    "is_context_overflow_error",
    "auto_compact_threshold",
    "effective_window",
    "estimate_messages_tokens",
    "CanonicalResponse",
    "ContentBlock",
    "Done",
    "GenerationConfig",
    "Message",
    "StreamEvent",
    "TextBlock",
    "TextDelta",
    "ThinkingBlock",
    "ThinkingDelta",
    "ToolResultBlock",
    "ToolSchema",
    "ToolUseArgsDelta",
    "ToolUseBlock",
    "ToolUseStart",
    "Usage",
]
