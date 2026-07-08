from __future__ import annotations

from dataclasses import dataclass

from .backends.base import Backend


@dataclass
class RouteTarget:
    backend: Backend
    upstream_model: str | None = None


class ModelRouter:
    """Routes model IDs to their backends."""

    def __init__(self):
        self._models: dict[str, RouteTarget] = {}
        self._default_backend: Backend | None = None

    def set_default_backend(self, backend: Backend) -> None:
        self._default_backend = backend

    def register_model(self, model_id: str, target: RouteTarget) -> None:
        self._models[model_id] = target

    def resolve(self, model_id: str) -> RouteTarget:
        if model_id in self._models:
            return self._models[model_id]
        if self._default_backend is not None:
            return RouteTarget(backend=self._default_backend, upstream_model=model_id)
        raise KeyError(f"Unknown model: {model_id!r}")

    def all_models(self) -> dict[str, RouteTarget]:
        return dict(self._models)
