"""Phase 10 — provider profiles, provider resolution, slash commands, completer."""

from __future__ import annotations

import stat

import pytest

from neurosurfer.app.cli.commands import build_registry
from neurosurfer.config import Config, LLMConfig
from neurosurfer.config.profiles import ProviderProfile, ProviderStore, mask_secret
from neurosurfer.llm.registry import build_provider_from_profile, resolve_provider


def store(tmp_path) -> ProviderStore:
    return ProviderStore(path=tmp_path / "providers.json")


# ── ProviderStore ─────────────────────────────────────────────────────────────
def test_add_makes_active_and_persists(tmp_path):
    s = store(tmp_path)
    s.add(ProviderProfile(name="lmstudio", kind="openai", base_url="http://x/v1", model="gemma"))
    assert s.names() == ["lmstudio"]
    assert s.active_name() == "lmstudio"
    assert (tmp_path / "providers.json").exists()
    # Reload from disk → same data.
    assert store(tmp_path).get_active().model == "gemma"


def test_duplicate_name_rejected(tmp_path):
    s = store(tmp_path)
    s.add(ProviderProfile(name="a"))
    with pytest.raises(ValueError):
        s.add(ProviderProfile(name="a"))


def test_file_permissions_are_owner_only(tmp_path):
    s = store(tmp_path)
    s.add(ProviderProfile(name="a", api_key="secret"))
    mode = stat.S_IMODE((tmp_path / "providers.json").stat().st_mode)
    assert mode == 0o600


def test_mask_secret():
    assert mask_secret(None) == "no key"
    assert mask_secret("sk-ant-abcdefghij").startswith("sk-a")
    assert "abcdefgh" not in mask_secret("sk-ant-abcdefghij")


# ── provider resolution ───────────────────────────────────────────────────────
def test_build_provider_from_openai_profile():
    p = ProviderProfile(name="x", kind="openai", base_url="http://h/v1", model="m", context_window=4096)
    provider = build_provider_from_profile(p)
    assert provider.model == "m"
    assert provider.capabilities.context_window == 4096


def test_build_provider_from_anthropic_profile_requires_key():
    with pytest.raises(RuntimeError):
        build_provider_from_profile(ProviderProfile(name="x", kind="anthropic", model="claude-opus-4-8"))


def test_resolve_prefers_active_profile(tmp_path):
    s = store(tmp_path)
    s.add(ProviderProfile(name="local", kind="openai", base_url="http://h/v1", model="m"))
    cfg = Config(llm=LLMConfig(provider="anthropic", anthropic_api_key=None))  # would fail if used
    provider = resolve_provider(cfg, s)
    assert provider.capabilities.tool_call_style == "openai"


# ── command registry ───────────────────────────────────────────────────────────
def test_registry_aliases_and_matches():
    reg = build_registry()
    assert reg.get("provider") is reg.get("pro")  # alias
    assert reg.get("/workflow") is reg.get("wf")   # alias
    names = {c.name for c in reg.matches("pro")}
    assert "provider" in names


# ── default-provider confirmation ─────────────────────────────────────────────
def test_confirm_default_sets_active_and_flag(tmp_path):
    s = store(tmp_path)
    s.add(ProviderProfile(name="a", model="m1"))
    s.add(ProviderProfile(name="b", model="m2"), make_active=False)
    s.confirm_default("b")
    assert s.active_name() == "b"
    assert s.is_default_confirmed() is True
    # Persists across instances.
    assert store(tmp_path).is_default_confirmed() is True
