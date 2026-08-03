"""Where the gateway's long-lived objects come from — the registry, the run
manager, the settings store, the Architect manager.

## Single-tenant, on purpose

The studio branch made every one of these **per account**: each signed-in user
got its own artifact space under `<home>/users/<id>/`, because otherwise a
workflow one account created was visible and runnable by all of them.

This line has no accounts. So the workspace dimension is gone and every function
here answers for the one installation — but the *signatures* keep their `user`
parameter, and that is deliberate rather than lazy:

- the routes are ported unchanged from a branch that passes a user, and rewriting
  nine call sites to drop an argument is nine chances to drop the wrong one;
- `user` is the seam. If accounts ever arrive, this module is the only file that
  has to learn about them, which is exactly the property the per-account version
  had and is worth keeping for free.

`user` is therefore accepted and ignored. There is one registry, one run store,
one settings store.

## Managers are cached, not rebuilt

A `RunManager` owns the in-memory run index and the threads of live runs, so
building a new one per request would lose track of everything already running.
That reasoning is unchanged from the per-account version; only the key is gone.
"""

from __future__ import annotations

import threading
from typing import Any

from neurosurfer.config.paths import generated_tools_dir, runs_dir, workflows_dir

_lock = threading.Lock()
_run_manager: Any = None
_architect_manager: Any = None
_settings_store: Any = None

#: The owner column every settings row is written under. The store is still
#: account-shaped (it came across whole), so it needs *a* key; with no accounts
#: there is exactly one, and naming it here keeps that fact in one place.
_OWNER = "default"


def workspace_key(user=None) -> str:  # noqa: ARG001 - see the module docstring
    """The single workspace. Kept as a function because the routes call it."""
    return _OWNER


def settings_store_for(server) -> Any:
    """The gateway's `SettingsStore` — one SQLite file.

    An injected `server.settings_store` wins, so tests and embedders that supply
    their own keep working.
    """
    injected = getattr(server, "settings_store", None)
    if injected is not None:
        return injected

    global _settings_store
    with _lock:
        if _settings_store is None:
            from .settings_store import SettingsStore

            _settings_store = SettingsStore()
            # One-off: adopt whatever the CLI had in providers.json, so somebody
            # who configured providers there does not start to an empty list.
            _settings_store.import_json_profiles()
        return _settings_store


def discovery_source_for(server, user=None) -> Any:
    """The MCP discovery engine, from settings.

    A settings failure degrades to the default rather than raising — a bad stored
    value should not take discovery down, and the official registry is always a
    correct answer to "where do I look for MCP servers".
    """
    from neurosurfer.mcp.sources import get_source

    chosen, key = "official", None
    try:
        settings = settings_store_for(server).get_settings(workspace_key(user)) or {}
        chosen = str(settings.get("mcp_source") or "official")
        key = settings.get("smithery_api_key") or None
    except Exception:  # noqa: BLE001 - discovery must survive a settings problem
        pass
    return get_source(chosen, smithery_key=key)


def known_credentials_for(server, user=None) -> dict[str, str]:
    """Credential values already held, by the name entries use.

    A registry entry states its requirement as a `{placeholder}` naming the value
    — `Bearer {smithery_api_key}` — so the keys here are placeholder names, not
    header names. Read on the request thread and bound for the work, because an
    Architect build runs on a raw thread with no context of its own.
    """
    out: dict[str, str] = {}
    owner = workspace_key(user)
    store = settings_store_for(server)
    try:
        out.update(store.secret_values(owner))
    except Exception:  # noqa: BLE001 - a store fault must not block a build
        pass
    try:
        settings = store.get_settings(owner) or {}
    except Exception:  # noqa: BLE001
        return out
    for key in ("smithery_api_key",):
        value = settings.get(key)
        if value:
            out.setdefault(key, str(value))
    return out


def secret_saver_for(server, user=None):
    """A callable that stores a secret, bound on the request thread.

    Deliberately write-only. A build asks for a credential and gets back nothing
    but "it is set" — nothing that can put a value into a prompt, a node or a
    transcript ever holds one.
    """
    owner = workspace_key(user)
    store = settings_store_for(server)

    def _save(name: str, value: str) -> None:
        store.put_secret(name, value, description="supplied during a build",
                         owner=owner)

    return _save


def registry_for(user=None):
    """The `WorkflowRegistry`."""
    from neurosurfer.graph.workflow.registry import WorkflowRegistry

    return WorkflowRegistry(workflows_dir())


def generated_tools_for(user=None):  # noqa: ARG001
    """The generated-tool store.

    Host-level, and that was a correctness point even with accounts:
    `tools/registry.py` loads generated tools with no workspace argument, so a
    tool written into a per-account directory is never loaded by the engine — a
    node could not call the thing that was just authored.
    """
    from neurosurfer.tools.generated import GeneratedToolsConfig

    return GeneratedToolsConfig(dir=generated_tools_dir())


def invalidate_run_manager(user=None) -> None:  # noqa: ARG001
    """Forget the cached managers so the next run rebuilds its providers.

    Providers are resolved once and cached with the manager; without this, a key
    or default changed in Settings would not take effect until a restart.
    """
    global _run_manager, _architect_manager
    with _lock:
        _run_manager = None
        _architect_manager = None


def run_manager_for(server, user=None):
    """The `RunManager`, built once and reused.

    An injected `server.run_manager` still wins, so tests and embedders that
    supply their own keep working.
    """
    injected = getattr(server, "run_manager", None)
    if injected is not None:
        return injected

    global _run_manager
    with _lock:
        if _run_manager is not None:
            return _run_manager

    from .workflow_runs.manager import RunManager
    from .workflow_runs.store import RunStore

    provider, resolver = providers_for(server, user)
    manager = RunManager(
        provider,
        registry=registry_for(user),
        store=RunStore(runs_dir()),
        provider_resolver=resolver,
        # A callable, not a snapshot: this manager is cached for the process's
        # lifetime, and a secret set after it was built must reach the next run.
        secrets_provider=lambda: known_credentials_for(server, user),
    )
    with _lock:
        if _run_manager is None:
            _run_manager = manager
        return _run_manager


def providers_for(server, user=None):
    """`(default_provider, resolver)`.

    The default is the profile marked default in Settings, falling back to `.env`
    `Config` when nothing is configured — so a fresh install still runs, and a
    configured one never silently uses the wrong key. The resolver carries every
    profile by name, which is what lets a node say which one it wants.
    """
    from neurosurfer.config import load_config
    from neurosurfer.llm.registry import build_provider_from_profile, resolve_provider
    from neurosurfer.llm.resolver import ProviderResolver

    owner = workspace_key(user)
    rows = []
    try:
        rows = settings_store_for(server).list_providers(owner)
    except Exception:  # noqa: BLE001 - a settings problem must not stop runs entirely
        pass

    profiles = {r.name: r.profile() for r in rows}
    default_row = next((r for r in rows if r.is_default), None)
    if default_row is not None:
        default = build_provider_from_profile(default_row.profile())
    else:
        # load_config (not Config()) so LLM_PROVIDER/MODEL/.env are honoured.
        default = resolve_provider(load_config())

    return default, ProviderResolver(
        default,
        profiles=profiles,
        factory=lambda profile, model: build_provider_from_profile(profile, model),
    )


def architect_manager_for(server, user=None):
    """The `ArchitectManager`, writing into the registry."""
    injected = getattr(server, "architect_manager", None)
    if injected is not None:
        return injected

    global _architect_manager
    with _lock:
        if _architect_manager is not None:
            return _architect_manager

    from .architect_builds.manager import ArchitectManager

    run_mgr = run_manager_for(server, user)
    manager = ArchitectManager(run_mgr.provider, registry=run_mgr.registry)
    with _lock:
        if _architect_manager is None:
            _architect_manager = manager
        return _architect_manager


def reset_cache() -> None:
    """Drop cached managers (tests, and after `NEUROSURFER_HOME` changes).

    The settings store goes too. It holds an open SQLite handle to a path derived
    from `NEUROSURFER_HOME`, so a cache that survived a home change would keep
    writing to the previous one — the same shape of bug as a module-scoped
    manifest that once ran against the developer's real `~/.neurosurfer`.
    """
    global _settings_store, _run_manager, _architect_manager
    with _lock:
        _run_manager = None
        _architect_manager = None
        if _settings_store is not None:
            try:
                _settings_store.close()
            except Exception:  # noqa: BLE001 - a closed/broken handle is still discarded
                pass
            _settings_store = None
