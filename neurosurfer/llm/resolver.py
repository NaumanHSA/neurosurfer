"""Which provider runs a given node.

Until this existed, a graph ran entirely on the single ``provider`` handed to
:class:`~neurosurfer.graph.engine.executor.GraphExecutor`. ``GraphNode.model`` was
read in exactly one place — to *label the trace* — so a graph that set it produced
traces naming a model nothing had called. That is worse than an unsupported field:
it reports a falsehood in the one place you look to check.

A resolver turns the node's declaration into a real client, in this order:

1. ``node.provider`` — a named profile, for "classify on the cheap model, write on
   the strong one, keep this one local";
2. the graph's default (whatever the executor was constructed with);
3. ``node.model`` alone rebinds the chosen provider to that model.

Clients are cached per ``(profile, model)``: a 20-node graph must not open 20
connections to say the same thing, and provider construction reads credentials.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Mapping
from typing import Any

from .base import Provider

__all__ = ["ProviderResolver", "UnknownProviderError"]


class UnknownProviderError(RuntimeError):
    """A node named a provider profile that is not configured."""

    def __init__(self, name: str, known: list[str]) -> None:
        self.name = name
        self.known = known
        known_text = ", ".join(sorted(known)) if known else "none configured"
        super().__init__(
            f"Node requests provider '{name}', which is not configured "
            f"(available: {known_text})."
        )


class ProviderResolver:
    """Resolves ``(provider_name, model)`` to a cached :class:`Provider`.

    *default* is the fallback client — what every node used to get unconditionally.
    *profiles* maps a name to something the factory can build from; *factory* turns
    one of those plus an optional model override into a Provider. Both are optional,
    so an embedded caller with a single provider keeps working and simply gains
    ``node.model``.
    """

    def __init__(
        self,
        default: Provider | None = None,
        *,
        profiles: Mapping[str, Any] | None = None,
        factory: Callable[[Any, str | None], Provider] | None = None,
    ) -> None:
        self.default = default
        self._profiles = dict(profiles or {})
        self._factory = factory
        self._cache: dict[tuple[str | None, str | None], Provider] = {}
        self._lock = threading.RLock()

    @property
    def names(self) -> list[str]:
        return sorted(self._profiles)

    def knows(self, name: str) -> bool:
        return name in self._profiles

    def resolve(self, provider_name: str | None = None, model: str | None = None) -> Provider:
        """The client a node should run on. Raises if it names an unknown profile."""
        if provider_name is None and model is None:
            if self.default is None:
                raise RuntimeError("No provider is configured for this run.")
            return self.default

        key = (provider_name, model)
        with self._lock:
            hit = self._cache.get(key)
            if hit is not None:
                return hit

        built = self._build(provider_name, model)
        with self._lock:
            # Another thread may have won the race; either instance is equivalent,
            # so keep whichever landed first rather than replacing a live client.
            return self._cache.setdefault(key, built)

    def _build(self, provider_name: str | None, model: str | None) -> Provider:
        if provider_name is not None:
            if provider_name not in self._profiles:
                raise UnknownProviderError(provider_name, self.names)
            if self._factory is None:
                raise RuntimeError(
                    f"Provider '{provider_name}' is configured but this run was given "
                    f"no way to build it."
                )
            return self._factory(self._profiles[provider_name], model)

        # Model-only override: keep the default's connection, change the model.
        if self.default is None:
            raise RuntimeError("No provider is configured for this run.")
        return rebind_model(self.default, model)


def rebind_model(provider: Provider, model: str | None) -> Provider:
    """A copy of *provider* speaking to *model*.

    Providers hold a connection plus a model name, and the model is per-call
    everywhere we support. Copying rather than mutating matters because the original
    is shared by every other node in the graph — setting ``.model`` in place would
    make one node's override leak into all of them, which is the sort of bug that
    only shows up as a surprising bill.
    """
    if not model or getattr(provider, "model", None) == model:
        return provider
    import copy

    clone = copy.copy(provider)
    clone.model = model
    return clone
