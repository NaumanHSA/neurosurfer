"""Per-Task provider pinning — run a specific Task on a specific provider profile.

Decoupled from the Task registry (and its readonly/system protection) so a
built-in Task like ``code`` can be pinned to a provider profile without ever
writing to its packaged YAML. Persists to ``~/.neurosurfer/task_providers.json``
as a flat ``{task_name: provider_profile_name}`` map.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TaskProviderStore:
    path: Path

    @classmethod
    def default(cls, state_home: Path | None = None) -> TaskProviderStore:
        home = state_home or (Path.home() / ".neurosurfer")
        return cls(path=home / "task_providers.json")

    def _read(self) -> dict[str, str]:
        if not self.path.exists():
            return {}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 - corrupt file → start fresh
            return {}
        return data if isinstance(data, dict) else {}

    def _write(self, data: dict[str, str]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")

    def all(self) -> dict[str, str]:
        return self._read()

    def get(self, task_name: str) -> str | None:
        return self._read().get(task_name)

    def set(self, task_name: str, provider_name: str) -> None:
        data = self._read()
        data[task_name] = provider_name
        self._write(data)

    def unset(self, task_name: str) -> None:
        data = self._read()
        if data.pop(task_name, None) is not None:
            self._write(data)
