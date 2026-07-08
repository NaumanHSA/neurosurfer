"""Build the active :class:`Provider` from configuration or a provider profile.

Resolution order (highest first):
  1. an explicitly passed ``ProviderProfile``;
  2. the active profile in the ``ProviderStore`` (if any profiles are configured);
  3. the ``Config`` (i.e. ``.env`` / environment).
"""

from __future__ import annotations

from ..config import Config
from ..config.profiles import ProviderProfile, ProviderStore
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
        supports_vision=cfg.llm.supports_vision,
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

    if profile.kind == "openai_native":
        if not profile.api_key:
            raise RuntimeError(f"Provider profile '{profile.name}' has no API key.")
        from .providers.openai import OpenAIProvider

        return OpenAIProvider(
            api_key=profile.api_key,
            model=model,
            max_output_tokens=profile.max_output_tokens,
        )

    from .providers.openai import OpenAICompatProvider

    return OpenAICompatProvider(
        base_url=profile.base_url or "http://localhost:1234/v1",
        api_key=profile.api_key or "not-needed",
        model=model,
        context_window=profile.context_window,
        max_output_tokens=profile.max_output_tokens,
        supports_vision=profile.supports_vision,
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


