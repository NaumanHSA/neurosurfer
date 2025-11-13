from __future__ import annotations

from typing import Any, Dict


class ArtifactStore:
    """
    Minimal artifact store for graph runs.

    Currently in-memory dict keyed by string IDs.
    Later you can plug in disk / DB / S3 implementations.
    """

    def __init__(self) -> None:
        self._data: Dict[str, Any] = {}

    def put(self, key: str, value: Any) -> None:
        self._data[key] = value

    def get(self, key: str) -> Any:
        if key not in self._data:
            raise KeyError(f"Artifact not found: {key}")
        return self._data[key]

    def as_dict(self) -> Dict[str, Any]:
        return dict(self._data)
