"""Provider profiles — named, persisted LLM connection settings.

A profile captures the connection (kind / base URL / model / api key / context
window); one profile is **active** and drives the client. Profiles persist to
``~/.neurosurfer/providers.json`` with mode ``0600`` (it holds API keys), and
secrets are masked for display.

This supersedes single-value ``.env`` config for connection selection: if profiles
exist, the active one wins; otherwise the engine falls back to ``Config`` (.env).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

ProviderKind = Literal["anthropic", "openai", "openai_native"]


def mask_secret(value: str | None) -> str:
    if not value:
        return "no key"
    if len(value) <= 8:
        return "•" * len(value)
    return value[:4] + "…" + value[-2:]


class ProviderProfile(BaseModel):
    name: str
    kind: ProviderKind = "openai"
    base_url: str | None = None  # OpenAI-compatible only
    model: str = ""
    api_key: str | None = None
    context_window: int = 200_000
    # Hard cap on tokens generated per model call. Reasoning/thinking models
    # can consume this budget for internal chain-of-thought; lower values
    # keep individual turns snappy on local hardware.
    max_output_tokens: int = 8192

    def endpoint(self) -> str:
        if self.kind == "anthropic":
            return "Anthropic API"
        if self.kind == "openai_native":
            return "OpenAI API"
        return self.base_url or "(no base URL)"

    def summary(self, *, active: bool = False) -> str:
        _KIND_LABEL = {
            "anthropic":    "Anthropic",
            "openai_native": "OpenAI",
            "openai":       "OpenAI-compatible",
        }
        kind = _KIND_LABEL.get(self.kind, self.kind)
        tail = "  (active)" if active else ""
        mot = f" · max_out={self.max_output_tokens}" if self.kind == "openai" else ""
        return (
            f"{self.name}: {kind} · {self.endpoint()} · {self.model or '(no model)'}"
            f"{mot} · {mask_secret(self.api_key)}{tail}"
        )


# Aliases bound while `list` still means the builtin — the store has a `.list()`
# method that would otherwise shadow `list[...]` inside method annotations.
_Profiles = list[ProviderProfile]
_Names = list[str]


class _StoreFile(BaseModel):
    active: str | None = None
    profiles: list[ProviderProfile] = Field(default_factory=list)
    # Set once the user has explicitly picked a default among 2+ profiles
    # (via the startup prompt or /provider use) — gates re-asking at every launch.
    confirmed_default: bool = False


@dataclass
class ProviderStore:
    """CRUD + active-selection over the persisted profiles file."""

    path: Path

    @classmethod
    def default(cls, state_home: Path | None = None) -> ProviderStore:
        home = state_home or (Path.home() / ".neurosurfer")
        return cls(path=home / "providers.json")

    # ── load / save ────────────────────────────────────────────────────────────
    def _read(self) -> _StoreFile:
        if not self.path.exists():
            return _StoreFile()
        try:
            return _StoreFile.model_validate_json(self.path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 - corrupt file → start fresh
            return _StoreFile()

    def _write(self, data: _StoreFile) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(data.model_dump_json(indent=2), encoding="utf-8")
        try:
            os.chmod(self.path, 0o600)  # secrets live here
        except OSError:
            pass

    # ── queries ────────────────────────────────────────────────────────────────
    def list(self) -> _Profiles:
        return self._read().profiles

    def names(self) -> _Names:
        return [p.name for p in self._read().profiles]

    def get(self, name: str) -> ProviderProfile | None:
        return next((p for p in self._read().profiles if p.name == name), None)

    def active_name(self) -> str | None:
        return self._read().active

    def get_active(self) -> ProviderProfile | None:
        data = self._read()
        if data.active is None:
            return None
        return next((p for p in data.profiles if p.name == data.active), None)

    def is_default_confirmed(self) -> bool:
        """Whether the user has settled on a default among 2+ profiles.

        With 0-1 profiles there's nothing to choose between, so this is
        trivially true and the startup prompt never fires.
        """
        data = self._read()
        return len(data.profiles) <= 1 or data.confirmed_default

    def confirm_default(self, name: str) -> None:
        """Record an explicit default-provider choice (the startup prompt)."""
        data = self._read()
        if not any(p.name == name for p in data.profiles):
            raise KeyError(f"No profile named '{name}'.")
        data.active = name
        data.confirmed_default = True
        self._write(data)

    # ── mutations ──────────────────────────────────────────────────────────────
    def add(self, profile: ProviderProfile, *, make_active: bool = True) -> None:
        data = self._read()
        if any(p.name == profile.name for p in data.profiles):
            raise ValueError(f"A profile named '{profile.name}' already exists.")
        data.profiles.append(profile)
        if make_active or data.active is None:
            data.active = profile.name
        self._write(data)

    def update(self, name: str, **changes: object) -> ProviderProfile:
        data = self._read()
        for i, p in enumerate(data.profiles):
            if p.name == name:
                updated = p.model_copy(update={k: v for k, v in changes.items() if v is not None})
                data.profiles[i] = updated
                self._write(data)
                return updated
        raise KeyError(f"No profile named '{name}'.")

    def delete(self, name: str) -> None:
        data = self._read()
        if not any(p.name == name for p in data.profiles):
            raise KeyError(f"No profile named '{name}'.")
        data.profiles = [p for p in data.profiles if p.name != name]
        if data.active == name:
            data.active = data.profiles[0].name if data.profiles else None
        self._write(data)

    def set_active(self, name: str) -> None:
        data = self._read()
        if not any(p.name == name for p in data.profiles):
            raise KeyError(f"No profile named '{name}'.")
        data.active = name
        data.confirmed_default = True  # an explicit /provider use counts as confirming
        self._write(data)
