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

# Error-message markers that mean "the model actually ran but we capped its
# output" — a token/length limit hit. That still proves the round-trip works
# (auth + reachability + valid model), so we treat it as connected. Reasoning
# models (GPT-5, o-series) spend the budget on hidden reasoning and routinely
# trip this on a tiny probe.
_TOKEN_LIMIT_MARKERS = (
    "max_tokens",
    "max_completion_tokens",
    "output limit",
    "output_limit",
    "finish the message",
    "length limit",
)


async def probe_provider(ctx: CLIContext) -> tuple[bool, str]:
    """Send a real minimal completion to verify the provider actually works.

    Returns (reachable, status_message).
    """
    import asyncio

    from neurosurfer.llm.registry import resolve_provider
    from neurosurfer.llm.types import GenerationConfig, Message, TextBlock

    try:
        provider = resolve_provider(ctx.cfg, ctx.providers)
    except RuntimeError:
        return False, "not configured"

    active = ctx.providers.get_active()
    model = (active.model if active else ctx.cfg.llm.model) or "unknown"

    messages = [Message(role="user", content=[TextBlock(text="hi")])]
    # 16 tokens: enough for a trivial reply on non-reasoning models; reasoning
    # models will still hit their cap, which we treat as a successful probe.
    config = GenerationConfig(max_tokens=16, temperature=0.0)

    async def _drain() -> None:
        async for _ in provider.stream(messages, None, [], config):
            pass

    try:
        await asyncio.wait_for(_drain(), timeout=15.0)
        return True, f"connected · {model}"
    except TimeoutError:
        return False, f"timed out · {model}"
    except Exception as e:  # noqa: BLE001
        msg = str(e).lower()
        if any(marker in msg for marker in _TOKEN_LIMIT_MARKERS):
            # The model ran and hit our token cap — connection is fine.
            return True, f"connected · {model}"
        brief = str(e)[:80].replace("\n", " ").strip()
        label = type(e).__name__
        return False, f"{label}: {brief}" if brief else label


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
