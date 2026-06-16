from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional
from .backends.base import Backend

@dataclass
class RouteTarget:
    backend: Backend
    upstream_model: Optional[str] = None

class ModelRouter:
    def __init__(self):
        self._models: Dict[str, RouteTarget] = {}
        self._default_backend: Optional[Backend] = None

    def set_default_backend(self, backend: Backend) -> None:
        self._default_backend = backend

    def register_model(self, model_id: str, target: RouteTarget) -> None:
        self._models[model_id] = target

    def resolve(self, model_id: str) -> RouteTarget:
        if model_id in self._models:
            return self._models[model_id]
        if self._default_backend is not None:
            return RouteTarget(backend=self._default_backend, upstream_model=model_id)
        raise KeyError(f"Unknown model: {model_id}")

    def all_models(self) -> Dict[str, RouteTarget]:
        return dict(self._models)
