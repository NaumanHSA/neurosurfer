"""CLIContext — shared state passed to slash-command handlers and the app."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from neurosurfer.config import Config
from neurosurfer.config.profiles import ProviderStore

if TYPE_CHECKING:
    from rich.console import Console


class CLIContext:
    def __init__(
        self,
        cfg: Config,
        console: Console,
        providers: ProviderStore,
    ) -> None:
        self.cfg = cfg
        self.console = console
        self.providers = providers
        self.should_exit: bool = False
        self._extra: dict = {}

    # ── CLI state persistence (theme) ─────────────────────────────────────────
    def _state_file(self) -> Path:
        return self.cfg.home_dir / "cli_state.json"

    def _save_state(self) -> None:
        from . import theme as _theme

        path = self._state_file()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"theme": _theme.current().name}))

    def load_state(self) -> None:
        """Restore last theme, or default on first run."""
        from . import theme as _theme

        data: dict = {}
        state_file = self._state_file()
        if state_file.exists():
            try:
                data = json.loads(state_file.read_text())
            except Exception:  # noqa: BLE001
                pass

        saved_theme = data.get("theme", _theme.DEFAULT_THEME)
        if saved_theme in _theme.THEMES:
            _theme.set_theme(saved_theme)

    # ── factory ───────────────────────────────────────────────────────────────
    @classmethod
    def create(cls, cfg: Config) -> CLIContext:
        from rich.console import Console

        cfg.ensure_dirs()
        providers = ProviderStore.default(cfg.home_dir)
        ctx = cls(cfg=cfg, console=Console(), providers=providers)
        ctx.load_state()
        return ctx
