"""Configuration loading for neurosurfer.

Settings come from environment variables (optionally seeded from a local ``.env``
file). The engine is provider-neutral; ``LLM_PROVIDER`` selects which adapter the
registry builds, and the rest of the values configure that adapter plus paths.

``Config`` is a composition of one dataclass per concern (``llm``,
``observability``) — see the sibling modules.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from .base import env_int, load_dotenv
from .llm import DEFAULT_ANTHROPIC_MODEL, DEFAULT_CONTEXT_WINDOW, LLMConfig
from .observability import ObservabilityConfig
from .projects import ProjectsConfig

__all__ = [
    "Config",
    "load_config",
    "LLMConfig",
    "ObservabilityConfig",
    "ProjectsConfig",
    "DEFAULT_ANTHROPIC_MODEL",
    "DEFAULT_CONTEXT_WINDOW",
]


@dataclass
class Config:
    """Resolved runtime configuration, namespaced by concern."""

    llm: LLMConfig = field(default_factory=LLMConfig)
    observability: ObservabilityConfig = field(default_factory=ObservabilityConfig)
    projects: ProjectsConfig = field(default_factory=ProjectsConfig)

    @property
    def home_dir(self) -> Path:
        """Root neurosurfer config directory (~/.neurosurfer)."""
        return Path.home() / ".neurosurfer"

    def ensure_dirs(self) -> None:
        self.home_dir.mkdir(parents=True, exist_ok=True)
        self.observability.transcripts_dir().mkdir(parents=True, exist_ok=True)
        self.projects.dir.mkdir(parents=True, exist_ok=True)

    def redacted(self) -> dict[str, object]:
        """Config snapshot safe to print (keys masked)."""

        def mask(v: str | None) -> str:
            if not v:
                return "<unset>"
            return v[:6] + "…" if len(v) > 8 else "<set>"

        return {
            "provider": self.llm.provider,
            "model": self.llm.model,
            "anthropic_api_key": mask(self.llm.anthropic_api_key),
            "openai_base_url": self.llm.openai_base_url,
            "openai_api_key": mask(self.llm.openai_api_key),
            "context_window": self.llm.context_window,
            "state_dir": str(self.observability.state_dir),
            "log_level": self.observability.log_level,
        }


def load_config(env_file: Path | None = None) -> Config:
    """Build a :class:`Config` from the environment (+ optional .env file)."""
    load_dotenv(env_file or Path.cwd() / ".env")

    provider = os.environ.get("LLM_PROVIDER", "anthropic").strip().lower()
    if provider not in ("anthropic", "openai"):
        provider = "anthropic"

    default_model = DEFAULT_ANTHROPIC_MODEL if provider == "anthropic" else "local-model"
    model = os.environ.get("MODEL", default_model).strip()

    state_dir = os.environ.get("NEUROSURFER_STATE_DIR")

    cfg = Config(
        llm=LLMConfig(
            provider=provider,
            model=model,
            anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY"),
            openai_base_url=os.environ.get("OPENAI_BASE_URL", "http://localhost:1234/v1").strip(),
            openai_api_key=os.environ.get("OPENAI_API_KEY", "not-needed").strip() or "not-needed",
            context_window=env_int("CONTEXT_WINDOW", DEFAULT_CONTEXT_WINDOW),
        ),
        observability=ObservabilityConfig(
            log_level=os.environ.get("NEUROSURFER_LOG_LEVEL", "INFO").strip().upper(),
        ),
    )
    if state_dir:
        cfg.observability.state_dir = Path(state_dir).expanduser()
    return cfg
