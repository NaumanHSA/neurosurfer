"""CLIContext — shared state passed to slash-command handlers and the app."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from neurosurfer.config import Config
from neurosurfer.config.mcp import McpStore
from neurosurfer.config.profiles import ProviderStore

if TYPE_CHECKING:
    from rich.console import Console

    from neurosurfer.mcp import McpManager


class CLIContext:
    def __init__(
        self,
        cfg: Config,
        console: Console,
        providers: ProviderStore,
        mcp_store: McpStore,
    ) -> None:
        self.cfg = cfg
        self.console = console
        self.providers = providers
        self.mcp_store = mcp_store
        self.mcp: McpManager | None = None
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

    # ── MCP lifecycle ──────────────────────────────────────────────────────────
    async def setup_mcp(self) -> None:
        """Connect to enabled MCP servers and publish their tools (best-effort).

        Must run inside the REPL's async task — the transports own anyio cancel
        scopes, so connect and :meth:`close_mcp` have to share a task. A server
        that is down is reported and skipped; it never blocks startup.
        """
        from . import theme

        servers = self.mcp_store.enabled()
        if not servers:
            return
        try:
            from neurosurfer.mcp import McpManager
        except ImportError:
            self.console.print(
                f"[{theme.WARN}]MCP servers configured but the 'mcp' extra is not "
                f"installed — skipping. Install with: pip install neurosurfer[mcp][/{theme.WARN}]"
            )
            return

        self.mcp = McpManager(servers)
        statuses = await self.mcp.connect_all()
        ok = [s for s in statuses if s.connected]
        bad = [s for s in statuses if not s.connected]
        if ok:
            total = sum(s.tool_count for s in ok)
            names = ", ".join(s.name for s in ok)
            self.console.print(
                f"[{theme.OK}]✓[/{theme.OK}] MCP: {len(ok)} server(s) connected "
                f"({total} tools) — {names}"
            )
        for s in bad:
            self.console.print(f"[{theme.WARN}]MCP server '{s.name}' unavailable: {s.error}[/{theme.WARN}]")

    async def close_mcp(self) -> None:
        if self.mcp is not None:
            await self.mcp.aclose()
            self.mcp = None

    # ── factory ───────────────────────────────────────────────────────────────
    @classmethod
    def create(cls, cfg: Config) -> CLIContext:
        from rich.console import Console

        cfg.ensure_dirs()
        providers = ProviderStore.default(cfg.home_dir)
        mcp_store = McpStore.default(cfg.home_dir)
        ctx = cls(
            cfg=cfg,
            console=Console(),
            providers=providers,
            mcp_store=mcp_store,
        )
        ctx.load_state()
        return ctx
