"""Build the active :class:`Provider` from configuration or a provider profile.

Resolution order (highest first):
  1. an explicitly passed ``ProviderProfile``;
  2. the active profile in the ``ProviderStore`` (if any profiles are configured);
  3. the ``Config`` (i.e. ``.env`` / environment).
"""

from __future__ import annotations

from ..config import Config
from ..config.profiles import ProviderProfile, ProviderStore
from ..config.task_providers import TaskProviderStore
from .base import Provider


def build_provider(cfg: Config, model_override: str | None = None) -> Provider:
    """Build a provider from .env-style :class:`Config`."""
    model = model_override or cfg.llm.model
    if cfg.llm.is_anthropic:
        if not cfg.llm.anthropic_api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set (LLM_PROVIDER=anthropic)")
        from .providers.anthropic import AnthropicProvider

        return AnthropicProvider(api_key=cfg.llm.anthropic_api_key, model=model)

    from .providers.openai import OpenAICompatProvider

    return OpenAICompatProvider(
        base_url=cfg.llm.openai_base_url,
        api_key=cfg.llm.openai_api_key,
        model=model,
        context_window=cfg.llm.context_window,
    )


def build_provider_from_profile(
    profile: ProviderProfile, model_override: str | None = None
) -> Provider:
    """Build a provider from a named :class:`ProviderProfile`."""
    model = model_override or profile.model
    if profile.kind == "anthropic":
        if not profile.api_key:
            raise RuntimeError(f"Provider profile '{profile.name}' has no API key.")
        from .providers.anthropic import AnthropicProvider

        return AnthropicProvider(api_key=profile.api_key, model=model)

    from .providers.openai import OpenAICompatProvider

    return OpenAICompatProvider(
        base_url=profile.base_url or "http://localhost:1234/v1",
        api_key=profile.api_key or "not-needed",
        model=model,
        context_window=profile.context_window,
        max_output_tokens=profile.max_output_tokens,
    )


def resolve_provider(
    cfg: Config,
    store: ProviderStore | None = None,
    model_override: str | None = None,
) -> Provider:
    """Prefer the active provider profile; fall back to .env Config."""
    store = store or ProviderStore.default()
    active = store.get_active()
    if active is not None:
        return build_provider_from_profile(active, model_override)
    return build_provider(cfg, model_override)


def resolve_provider_for_task(
    cfg: Config,
    store: ProviderStore,
    task_providers: TaskProviderStore,
    task_name: str | None,
    model_override: str | None = None,
) -> Provider:
    """Prefer a Task's pinned provider (``/task provider``); else the usual default.

    Lets different Tasks run on different models/providers (e.g. a fast local
    model for one Task, Claude for another) without disturbing the single
    "active" profile that everything else still falls back to.
    """
    if task_name:
        pinned_name = task_providers.get(task_name)
        if pinned_name:
            profile = store.get(pinned_name)
            if profile is None:
                raise RuntimeError(
                    f"Task '{task_name}' is pinned to provider '{pinned_name}', which no "
                    f"longer exists. Fix it with /task provider {task_name} <name>."
                )
            return build_provider_from_profile(profile, model_override)
    return resolve_provider(cfg, store, model_override)
