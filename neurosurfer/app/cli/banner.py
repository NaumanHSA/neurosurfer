"""CLI banner — provider status appended below the import-time startup banner."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

from . import theme
from .context import CLIContext

if TYPE_CHECKING:
    pass

_TIPS: list[str] = [
    "just type what you want — I can read/write files, run commands, search the web",
    "summarise this folder  — or: find all TODO comments / run the test suite",
    "/workflow build  — design a reusable pipeline for recurring or multi-stage jobs",
    "/workflow run    — execute a registered workflow",
    "/workflow list   — see all your registered workflows",
    "/provider add    — connect an Anthropic, LM Studio, or Ollama model",
    "/help            — see every available command",
]


# ── Provider connectivity probe ───────────────────────────────────────────────

async def probe_provider(ctx: CLIContext) -> tuple[bool, str]:
    """Ping the active provider. Returns (reachable, status_message)."""
    active = ctx.providers.get_active()

    if active is not None:
        kind = active.kind
        base_url = active.base_url or ""
        api_key = active.api_key or ""
        model = active.model
    else:
        cfg = ctx.cfg.llm
        if cfg.is_anthropic and cfg.anthropic_api_key:
            kind = "anthropic"
            base_url = ""
            api_key = cfg.anthropic_api_key
            model = cfg.model
        elif cfg.is_openai:
            kind = "openai"
            base_url = cfg.openai_base_url
            api_key = cfg.openai_api_key
            model = cfg.model
        else:
            return False, "not configured"

    if kind == "openai":
        url = (base_url or "http://localhost:1234/v1").rstrip("/")
        try:
            import httpx
            headers = {"Authorization": f"Bearer {api_key or 'not-needed'}"}
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{url}/models", headers=headers)
            if resp.status_code < 500:
                return True, f"connected · {model or url}"
            return False, f"HTTP {resp.status_code} · {url}"
        except httpx.ConnectError:
            return False, f"unable to connect · {url}"
        except httpx.TimeoutException:
            return False, f"timed out · {url}"
        except Exception as e:  # noqa: BLE001
            return False, str(e)[:80]
    else:  # anthropic
        if not api_key:
            return False, "API key not set — use /provider add"
        if api_key.startswith("sk-ant-"):
            return True, f"key configured · {model}"
        return False, "key present but format unexpected"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _resolve_provider_status(ctx: CLIContext) -> tuple[str, str, bool]:
    """Return (label, model, configured)."""
    active = ctx.providers.get_active()
    if active is not None:
        return active.name, active.model or "(no model)", True
    cfg = ctx.cfg.llm
    if cfg.is_anthropic and cfg.anthropic_api_key:
        return "Anthropic (.env)", cfg.model, True
    if cfg.is_openai and cfg.openai_base_url:
        return f"OpenAI-compat (.env) {cfg.openai_base_url}", cfg.model, True
    return "not configured", "", False


# ── Public ────────────────────────────────────────────────────────────────────

async def print_banner(ctx: CLIContext) -> None:
    """Append provider status + tips below the import-time startup banner."""
    from rich.table import Table

    console = ctx.console

    reachable, probe_msg = await probe_provider(ctx)
    label, model, configured = _resolve_provider_status(ctx)

    # Match the import-time banner grid exactly: cyan-dim right label, 16 chars wide.
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="cyan dim", justify="right", min_width=16)
    grid.add_column()

    dot = "●" if (configured and reachable) else "○"
    dot_style = "bright_cyan" if (configured and reachable) else "red"
    prov_val = f"[{dot_style}]{dot} {label}[/{dot_style}]"
    if model:
        prov_val += f"  [dim]{model}[/dim]"
    if probe_msg and probe_msg not in ("not configured",):
        msg_style = "dim" if reachable else "red dim"
        prov_val += f"  [{msg_style}]({probe_msg})[/{msg_style}]"

    grid.add_row("provider", prov_val)
    console.print(grid)
    console.print()

    # Capability card — one dim line for general automation, one accented line for
    # the workflow headline so it stands out clearly as the differentiating feature.
    console.print(
        f"  [{theme.DIM}]Files · shell & code · web search"
        f" — ask me to do almost anything.[/{theme.DIM}]"
    )
    console.print(
        f"  [{theme.ACCENT}]✦[/{theme.ACCENT}]"
        f"  [{theme.ACCENT_DIM}]Design & register"
        f" [bold]reusable workflow pipelines[/bold]"
        f" for recurring or multi-stage jobs.[/{theme.ACCENT_DIM}]"
    )
    console.print()

    tips = random.sample(_TIPS, k=min(2, len(_TIPS)))
    for tip in tips:
        console.print(f"  [{theme.DIM}]tip: {tip}[/{theme.DIM}]")
    console.print()


def print_status(ctx: CLIContext) -> None:
    """Sync status re-print used after /provider commands."""
    from rich.table import Table

    label, model, configured = _resolve_provider_status(ctx)
    dot = "●" if configured else "○"
    dot_style = theme.OK_DIM if configured else theme.ERR

    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="cyan dim", justify="right", min_width=16)
    grid.add_column()
    prov_val = f"[{dot_style}]{dot} {label}[/{dot_style}]"
    if model:
        prov_val += f"  [dim]{model}[/dim]"
    grid.add_row("provider", prov_val)
    ctx.console.print(grid)


def status_summary(ctx: CLIContext) -> str:
    """Compact one-line status for the bottom toolbar."""
    label, model, _ = _resolve_provider_status(ctx)
    prov = f"{label} · {model}" if model else label
    from .commands.workflow import _registry as _wf_registry
    try:
        count = len(_wf_registry(ctx).list())
        wf_status = f"{count} workflow{'s' if count != 1 else ''}"
    except Exception:  # noqa: BLE001
        wf_status = "workflows"
    return f"{prov}  |  {wf_status}"
