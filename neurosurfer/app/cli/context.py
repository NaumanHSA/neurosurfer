"""CLIContext — shared state passed to slash-command handlers and the app.

Holds the resolved config, the Rich console, the provider store, the task
registry, and mutable session state (the active task for this session).

``active_task`` is a property: setting it auto-saves the value to a small
JSON file so the next session restores where you left off.  On first launch
it defaults to the 'general' built-in task if one is registered.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from neurosurfer.config import Config
from neurosurfer.config.profiles import ProviderStore
from neurosurfer.config.task_providers import TaskProviderStore
from neurosurfer.tasks.registry import TaskRegistry

if TYPE_CHECKING:
    from rich.console import Console

    from neurosurfer.sessions.store import SessionStore


class CLIContext:
    def __init__(
        self,
        cfg: Config,
        console: Console,
        providers: ProviderStore,
        registry: TaskRegistry,
        active_task: str | None = None,
        task_providers: TaskProviderStore | None = None,
    ) -> None:
        self.cfg = cfg
        self.console = console
        self.providers = providers
        self.registry = registry
        self.task_providers = task_providers or TaskProviderStore.default(cfg.tasks.dir.parent)
        self._active_task = active_task
        self.should_exit: bool = False
        self._extra: dict = {}
        # Session store (lazy-init via init_session_store)
        self._session_store: SessionStore | None = None
        self.active_session_id: str | None = None

    # ── active_task property: auto-persists on every change ───────────────────
    @property
    def active_task(self) -> str | None:
        return self._active_task

    @active_task.setter
    def active_task(self, value: str | None) -> None:
        self._active_task = value
        self._save_session()

    # ── session persistence ───────────────────────────────────────────────────
    def _session_file(self) -> Path:
        return self.cfg.tasks.dir.parent / "session.json"

    def _save_session(self) -> None:
        from . import theme as _theme

        path = self._session_file()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "active_task": self._active_task,
            "theme": _theme.current().name,
        }))

    def load_session(self) -> None:
        """Restore last active task and theme, or defaults on first run."""
        from . import theme as _theme

        data: dict = {}
        session_file = self._session_file()
        if session_file.exists():
            try:
                data = json.loads(session_file.read_text())
            except Exception:  # noqa: BLE001
                pass

        # Restore theme first so the banner prints in the right color.
        saved_theme = data.get("theme", _theme.DEFAULT_THEME)
        if saved_theme in _theme.THEMES:
            _theme.set_theme(saved_theme)

        saved_task = data.get("active_task")
        available = self.registry.list()
        if saved_task and saved_task in available:
            self._active_task = saved_task
        elif "general" in available:
            self._active_task = "general"
        # else leave as None (no built-in general registered)

    # ── session store ─────────────────────────────────────────────────────────
    @property
    def session_store(self) -> SessionStore | None:
        return self._session_store

    def init_session_store(self) -> None:
        """Lazy-init the SessionStore.  Call once from run_repl() after create()."""
        from neurosurfer.sessions.store import SessionStore
        self._session_store = SessionStore(self.cfg.sessions.dir)

    # ── factory ───────────────────────────────────────────────────────────────
    @classmethod
    def create(cls, cfg: Config) -> CLIContext:
        from rich.console import Console

        cfg.ensure_dirs()
        providers = ProviderStore.default(cfg.tasks.dir.parent)
        task_providers = TaskProviderStore.default(cfg.tasks.dir.parent)
        registry = TaskRegistry(cfg.tasks.dir)
        ctx = cls(
            cfg=cfg, console=Console(), providers=providers, registry=registry,
            task_providers=task_providers,
        )
        ctx.load_session()
        return ctx
