"""Phase 10 — provider profiles, provider resolution, slash commands, completer."""

from __future__ import annotations

import stat

import pytest

from neurosurfer.app.cli.commands import build_registry
from neurosurfer.app.cli.commands.provider import op_delete as prov_delete
from neurosurfer.app.cli.commands.provider import op_use, provider_table_rows
from neurosurfer.app.cli.completer import SlashCompleter
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


def test_second_add_non_active_keeps_active(tmp_path):
    s = store(tmp_path)
    s.add(ProviderProfile(name="a", model="m1"))
    s.add(ProviderProfile(name="b", model="m2"), make_active=False)
    assert s.active_name() == "a"
    assert set(s.names()) == {"a", "b"}


def test_set_active_update_delete_reassigns(tmp_path):
    s = store(tmp_path)
    s.add(ProviderProfile(name="a", model="m1"))
    s.add(ProviderProfile(name="b", model="m2"), make_active=False)
    s.set_active("b")
    assert s.active_name() == "b"
    s.update("b", model="m2-new")
    assert s.get("b").model == "m2-new"
    s.delete("b")  # active deleted → reassigns to remaining
    assert s.active_name() == "a"
    assert s.names() == ["a"]


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


def test_resolve_falls_back_to_config(tmp_path):
    cfg = Config(llm=LLMConfig(provider="openai", model="m", openai_base_url="http://h/v1", context_window=8192))
    provider = resolve_provider(cfg, store(tmp_path))  # empty store
    assert provider.capabilities.tool_call_style == "openai"
    assert provider.capabilities.context_window == 8192


# ── command registry + completer ──────────────────────────────────────────────
def test_registry_aliases_and_matches():
    reg = build_registry()
    assert reg.get("provider") is reg.get("pro")  # alias
    assert reg.get("/workflow") is reg.get("wf")   # alias
    names = {c.name for c in reg.matches("pro")}
    assert "provider" in names


def _completions(text: str):
    from prompt_toolkit.document import Document

    reg = build_registry()
    comp = SlashCompleter(reg)
    return [c.text for c in comp.get_completions(Document(text, len(text)), None)]


def test_completer_command_names():
    assert "provider" in _completions("/pro")
    assert set(_completions("/")) >= {"provider", "workflow", "help"}
    # task and run are removed from the registry
    assert "task" not in _completions("/")
    # Non-slash input yields nothing.
    assert _completions("hello") == []


def test_completer_subcommands():
    subs = _completions("/provider ")
    assert {"list", "use", "add", "delete"} <= set(subs)
    assert "build" in _completions("/workflow b")


# ── provider command ops ──────────────────────────────────────────────────────
def test_provider_ops(tmp_path):
    s = store(tmp_path)
    s.add(ProviderProfile(name="a", model="m1"))
    s.add(ProviderProfile(name="b", model="m2"), make_active=False)
    assert "now 'b'" in op_use(s, "b")
    assert s.active_name() == "b"
    rows = provider_table_rows(s)
    assert any(marker == "active" for _, marker in rows)
    assert "Deleted" in prov_delete(s, "a")
    assert "a" not in s.names()


# ── default-provider confirmation ─────────────────────────────────────────────
def test_single_profile_is_confirmed_by_default(tmp_path):
    s = store(tmp_path)
    s.add(ProviderProfile(name="a", model="m1"))
    assert s.is_default_confirmed() is True  # nothing to choose between


def test_two_profiles_unconfirmed_until_chosen(tmp_path):
    s = store(tmp_path)
    s.add(ProviderProfile(name="a", model="m1"))
    s.add(ProviderProfile(name="b", model="m2"), make_active=False)
    assert s.is_default_confirmed() is False


def test_confirm_default_sets_active_and_flag(tmp_path):
    s = store(tmp_path)
    s.add(ProviderProfile(name="a", model="m1"))
    s.add(ProviderProfile(name="b", model="m2"), make_active=False)
    s.confirm_default("b")
    assert s.active_name() == "b"
    assert s.is_default_confirmed() is True
    # Persists across instances.
    assert store(tmp_path).is_default_confirmed() is True


def test_confirm_default_unknown_profile_raises(tmp_path):
    s = store(tmp_path)
    s.add(ProviderProfile(name="a", model="m1"))
    with pytest.raises(KeyError):
        s.confirm_default("nope")


def test_set_active_marks_default_confirmed(tmp_path):
    s = store(tmp_path)
    s.add(ProviderProfile(name="a", model="m1"))
    s.add(ProviderProfile(name="b", model="m2"), make_active=False)
    assert s.is_default_confirmed() is False
    s.set_active("b")
    assert s.is_default_confirmed() is True


