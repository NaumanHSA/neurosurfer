"""LLM connection settings — which provider/model the engine talks to."""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_ANTHROPIC_MODEL = "claude-opus-4-8"
# Anthropic 200k window is the engine default; local servers override via CONTEXT_WINDOW.
DEFAULT_CONTEXT_WINDOW = 200_000


@dataclass
class LLMConfig:
    provider: str = "anthropic"
    model: str = DEFAULT_ANTHROPIC_MODEL

    anthropic_api_key: str | None = None

    openai_base_url: str = "http://localhost:1234/v1"
    openai_api_key: str = "not-needed"

    # Used for compaction thresholds. Anthropic resolves the real window from
    # capabilities; OpenAI-compatible servers have no count endpoint so this is
    # authoritative for local models.
    context_window: int = DEFAULT_CONTEXT_WINDOW

    @property
    def is_anthropic(self) -> bool:
        return self.provider == "anthropic"

    @property
    def is_openai(self) -> bool:
        return self.provider == "openai"
