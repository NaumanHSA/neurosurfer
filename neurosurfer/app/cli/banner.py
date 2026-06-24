"""Startup banner + provider status line for the Workflow Architect."""

from __future__ import annotations

from typing import TYPE_CHECKING

from neurosurfer import __version__
from . import theme
from .context import CLIContext

if TYPE_CHECKING:
    pass

_LOGO = r"""
  ███╗   ██╗███████╗██╗   ██╗██████╗  ██████╗
  ████╗  ██║██╔════╝██║   ██║██╔══██╗██╔═══██╗
  ██╔██╗ ██║█████╗  ██║   ██║██████╔╝██║   ██║
  ██║╚██╗██║██╔══╝  ██║   ██║██╔══██╗██║   ██║
  ██║ ╚████║███████╗╚██████╔╝██║  ██║╚██████╔╝
  ╚═╝  ╚═══╝╚══════╝ ╚═════╝ ╚═╝  ╚═╝ ╚═════╝
   ██████╗ ██╗   ██╗██████╗ ███████╗███████╗██████╗
  ██╔════╝ ██║   ██║██╔══██╗██╔════╝██╔════╝██╔══██╗
  ╚█████╗  ██║   ██║██████╔╝█████╗  █████╗  ██████╔╝
   ╚════██╗██║   ██║██╔══██╗██╔══╝  ██╔══╝  ██╔══██╗
  ██████╔╝ ╚██████╔╝██║  ██║██║     ███████╗██║  ██║
  ╚═════╝   ╚═════╝ ╚═╝  ╚═╝╚═╝     ╚══════╝╚═╝  ╚═╝
"""

_TAGLINE = "AI Agent Framework  ·  lean, local, production-ready"

_HOW_IT_WORKS = """\
  How it works:
    1  You describe what you want to automate in plain English.
    2  The Architect researches your problem on the web.
    3  It asks you a few focused questions (3 choices each).
    4  It designs a multi-step workflow and builds every node.
    5  The workflow is registered and ready to run — on any model.

  Python API:  from neurosurfer.graph.workflow.package import load_package
               from neurosurfer.graph.workflow.runner  import WorkflowRunner\
"""

_TIPS: list[str] = [
    "/workflow build  — describe a workflow and let the Architect build it",
    "/workflow list   — see all your registered workflows",
    "/workflow run    — execute a registered workflow",
    "/workflow show   — inspect a workflow's nodes and graph",
    "/provider add    — connect an Anthropic, LM Studio, or Ollama model",
    "/help            — see every available command",
]


# ──────────────────────────────────────────────────────────────────────────────
# Provider connectivity probe
# ──────────────────────────────────────────────────────────────────────────────

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
            return False, "API key not set — set ANTHROPIC_API_KEY or use /provider add"
        if api_key.startswith("sk-ant-"):
            return True, f"key configured · {model}"
        return False, "key present but format unexpected"


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

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


# ──────────────────────────────────────────────────────────────────────────────
# Public
# ──────────────────────────────────────────────────────────────────────────────

async def print_banner(ctx: CLIContext) -> None:
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    console = ctx.console
    console.print()

    for line in _LOGO.strip("\n").splitlines():
        console.print(f"[{theme.ACCENT}]{line}[/{theme.ACCENT}]")
    console.print(f"     [{theme.DIM}]{_TAGLINE}[/{theme.DIM}]")
    console.print()

    # ── connectivity probe ─────────────────────────────────────────────────
    reachable, probe_msg = await probe_provider(ctx)

    # ── provider status panel ─────────────────────────────────────────────
    label, model, configured = _resolve_provider_status(ctx)
    prov_color = theme.OK_DIM if (configured and reachable) else theme.ERR
    conn_icon = "●" if (configured and reachable) else "○"
    conn_label = "connected" if reachable else "unreachable"
    conn_color = theme.OK_DIM if reachable else theme.ERR

    prov_text = Text()
    prov_text.append(f"{conn_icon} {label}", style=prov_color)
    if model:
        prov_text.append(" · ", style=theme.DIM)
        prov_text.append(model, style=theme.ACCENT_DIM)
    prov_text.append("\n")
    prov_text.append(f"  [{conn_label}]", style=conn_color)
    if probe_msg and probe_msg not in ("not configured",):
        prov_text.append(f"  {probe_msg}", style=theme.DIM)

    grid = Table.grid(padding=(0, 2))
    grid.add_column(justify="right", style=theme.DIM)
    grid.add_column()
    grid.add_row("provider", prov_text)
    console.print(Panel(grid, border_style=theme.ACCENT_DIM, expand=False))

    # ── how it works panel ────────────────────────────────────────────────
    console.print()
    for line in _HOW_IT_WORKS.splitlines():
        console.print(f"[{theme.DIM}]{line}[/{theme.DIM}]")
    console.print()

    # ── tips (pick 2) ─────────────────────────────────────────────────────
    import random
    tips = random.sample(_TIPS, k=min(2, len(_TIPS)))
    for tip in tips:
        console.print(f"  [{theme.DIM}]tip: {tip}[/{theme.DIM}]")

    console.print()
    console.print(
        f"[{theme.DIM}]neurosurfer v{__version__}  ·  "
        f"type [{theme.DIM}][{theme.ACCENT}]/help[/{theme.ACCENT}][/{theme.DIM}]"
        f"[{theme.DIM}] or describe what you want to build[/{theme.DIM}]"
    )
    console.print()


def print_status(ctx: CLIContext) -> None:
    """Sync status re-print used after /provider commands."""
    from rich.panel import Panel
    from rich.table import Table

    label, model, configured = _resolve_provider_status(ctx)
    prov_color = theme.OK_DIM if configured else theme.ERR
    prov_dot = "●" if configured else "○"
    prov_text = f"[{prov_color}]{prov_dot} {label}[/{prov_color}]"
    if model:
        prov_text += f" [{theme.DIM}]·[/{theme.DIM}] [{theme.ACCENT_DIM}]{model}[/{theme.ACCENT_DIM}]"

    grid = Table.grid(padding=(0, 2))
    grid.add_column(justify="right", style=theme.DIM)
    grid.add_column()
    grid.add_row("provider", prov_text)
    ctx.console.print(Panel(grid, border_style=theme.ACCENT_DIM, expand=False))


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
