"""`doctor` — config + active-connection reachability check."""

from __future__ import annotations

from . import theme
from .context import CLIContext


async def check_reachable(ctx: CLIContext) -> tuple[bool, str]:
    """Ping the active connection (active provider profile, else .env Config)."""
    active = ctx.providers.get_active()
    if active is not None:
        kind = active.kind
        base_url = active.base_url or "http://localhost:1234/v1"
        api_key = active.api_key
        model = active.model
        label = f"profile '{active.name}'"
    else:
        cfg = ctx.cfg.llm
        kind = "anthropic" if cfg.is_anthropic else "openai"
        base_url = cfg.openai_base_url
        api_key = cfg.anthropic_api_key if cfg.is_anthropic else cfg.openai_api_key
        model = cfg.model
        label = ".env config"

    if kind == "anthropic":
        if not api_key:
            return False, f"No API key for {label}."
        try:
            import anthropic

            client = anthropic.AsyncAnthropic(api_key=api_key)
            await client.messages.count_tokens(
                model=model, messages=[{"role": "user", "content": "ping"}]
            )
            return True, f"Anthropic reachable ({label}); model={model}"
        except Exception as e:  # noqa: BLE001
            return False, f"Anthropic check failed ({label}): {e}"
    else:
        try:
            import httpx

            url = base_url.rstrip("/") + "/models"
            async with httpx.AsyncClient(timeout=5.0) as http_client:
                resp = await http_client.get(
                    url, headers={"Authorization": f"Bearer {api_key or 'not-needed'}"}
                )
            if resp.status_code == 200:
                return True, f"OpenAI-compatible server reachable at {base_url} ({label})"
            return False, f"Server returned HTTP {resp.status_code} at {url}"
        except Exception as e:  # noqa: BLE001
            return False, f"OpenAI-compatible server unreachable ({label}): {e}"


async def check_mcp(ctx: CLIContext) -> list[tuple[bool, str]]:
    """Connect to each enabled MCP server and report tool counts (best-effort)."""
    servers = ctx.mcp_store.enabled()
    if not servers:
        return []
    try:
        from neurosurfer.mcp import McpManager
    except ImportError:
        return [(False, "MCP servers configured but the 'mcp' extra is not installed.")]

    mgr = McpManager(servers)
    try:
        statuses = await mgr.connect_all()
    finally:
        await mgr.aclose()
    return [
        (
            s.connected,
            f"MCP '{s.name}': {s.tool_count} tools"
            if s.connected
            else f"MCP '{s.name}' unreachable: {s.error}",
        )
        for s in statuses
    ]


def _report_managed_env(ctx: CLIContext) -> None:
    """Report the managed python_exec venv's status (read-only; never provisions)."""
    from neurosurfer.tools.builtin.python_exec.managed_env import (
        installed_packages,
        is_provisioned,
        managed_python,
        managed_venv_dir,
    )

    console = ctx.console
    venv_dir = managed_venv_dir()
    if is_provisioned():
        interp = managed_python()
        pkgs = installed_packages(interp) if interp else []
        console.print(
            f"[{theme.OK}]✓[/{theme.OK}] Managed Python env ready at {venv_dir} "
            f"({len(pkgs)} packages)"
        )
    else:
        console.print(
            f"[{theme.WARN}]![/{theme.WARN}] Managed Python env not provisioned "
            f"(will provision on first python_exec use, or run `neurosurfer setup`)"
        )


def cmd_doctor(ctx: CLIContext) -> int:
    import asyncio

    from rich.table import Table

    console = ctx.console
    console.print("[bold]neurosurfer doctor[/bold]\n")

    table = Table(show_header=False, box=None, pad_edge=False)
    for k, v in ctx.cfg.redacted().items():
        table.add_row(f"[{theme.ACCENT_DIM}]{k}[/{theme.ACCENT_DIM}]", str(v))
    active = ctx.providers.active_name()
    table.add_row(f"[{theme.ACCENT_DIM}]active_profile[/{theme.ACCENT_DIM}]", active or "(none — using .env)")
    console.print(table)
    console.print()

    ok, msg = asyncio.run(check_reachable(ctx))
    if ok:
        console.print(f"[{theme.OK}]✓[/{theme.OK}] {msg}")
    else:
        console.print(f"[{theme.ERR}]✗[/{theme.ERR}] {msg}")

    for mcp_ok, mcp_msg in asyncio.run(check_mcp(ctx)):
        glyph = f"[{theme.OK}]✓[/{theme.OK}]" if mcp_ok else f"[{theme.WARN}]![/{theme.WARN}]"
        console.print(f"{glyph} {mcp_msg}")

    _report_managed_env(ctx)

    return 0 if ok else 1


def cmd_setup(ctx: CLIContext) -> int:
    """Provision the managed python_exec venv up front (curated packages)."""
    from neurosurfer.tools.builtin.python_exec.managed_env import ensure_managed_venv

    console = ctx.console
    console.print("[bold]neurosurfer setup[/bold]\n")
    try:
        interpreter = ensure_managed_venv(notify=console.print)
    except RuntimeError as e:
        console.print(f"[{theme.ERR}]✗[/{theme.ERR}] {e}")
        return 1
    console.print(f"[{theme.OK}]✓[/{theme.OK}] Managed interpreter: {interpreter}")
    return 0
