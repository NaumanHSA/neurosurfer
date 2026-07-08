"""/provider — manage provider profiles (add / set-active / edit / delete).

All interaction happens inline in the REPL — no full-screen dialog overlays.
Pure operations (unit-tested) are separated from rendering.
"""

from __future__ import annotations

from typing import cast

from neurosurfer.config.profiles import ProviderKind, ProviderProfile, ProviderStore

from .. import theme
from ..context import CLIContext
from ..io import _ainput, select_menu
from .base import SlashCommand


# ── pure operations (unit-tested) ─────────────────────────────────────────────
def provider_table_rows(store: ProviderStore) -> list[tuple[str, str]]:
    """(summary, marker) rows for display/testing."""
    active = store.active_name()
    return [(p.summary(active=p.name == active), "active" if p.name == active else "") for p in store.list()]


def op_use(store: ProviderStore, name: str) -> str:
    store.set_active(name)
    return f"Active provider is now '{name}'."


def op_delete(store: ProviderStore, name: str) -> str:
    store.delete(name)
    return f"Deleted provider profile '{name}'."


# ── inline prompt helpers ──────────────────────────────────────────────────────
async def _prompt(label: str, default: str = "", *, is_password: bool = False) -> str:
    """Single-line inline prompt; returns default if user enters nothing."""
    suffix = f" [{default}]" if default else ""
    raw = await _ainput(f"  {label}{suffix}: ", is_password=is_password)
    return raw.strip() or default


async def _radio(ctx: CLIContext, title: str, values: list[tuple]) -> str | None:
    """Arrow-key (↑/↓ + Enter) selector; returns None if the user cancels."""
    return await select_menu(ctx.console, title, list(values))


async def _prompt_vision(ctx: CLIContext, current: bool | None) -> bool | None:
    """Tri-state vision prompt for OpenAI-compatible profiles. Returns None
    (auto-detect), True (force on), or False (force off); keeps *current* on cancel."""
    choice = await _radio(ctx, "Image input (vision)", [
        ("auto", "Auto-detect",   "Enable only for known vision model names"),
        ("yes",  "Yes, supported", "Force-enable — for local vision models (Qwen-VL, LLaVA, …)"),
        ("no",   "No",            "Force-disable image input"),
    ])
    return {"auto": None, "yes": True, "no": False}.get(choice, current)


# ── rendering ─────────────────────────────────────────────────────────────────
def _print_list(ctx: CLIContext) -> None:
    profiles = ctx.providers.list()
    if not profiles:
        ctx.console.print(f"[{theme.DIM}]No provider profiles yet. Use /provider add.[/{theme.DIM}]")
        return
    active = ctx.providers.active_name()
    ctx.console.print(f"[bold {theme.ACCENT}]Provider profiles[/bold {theme.ACCENT}]")
    for p in profiles:
        is_active = p.name == active
        color = theme.OK_DIM if is_active else theme.DIM
        ctx.console.print(f"  [{color}]{'●' if is_active else '○'} {p.summary(active=is_active)}[/{color}]")


# ── interactive menu (arrow keys via inline prompts) ─────────────────────────
async def _add_profile(ctx: CLIContext) -> None:
    ctx.console.print(f"\n[bold {theme.ACCENT}]Add provider[/bold {theme.ACCENT}]")

    kind_raw = await _radio(ctx, "Provider kind", [
        ("openai_native", "OpenAI",                "api.openai.com — just a model ID and API key"),
        ("openai",        "OpenAI-compatible",      "LM Studio / Ollama / vLLM — requires base URL"),
        ("anthropic",     "Anthropic",              "api.anthropic.com"),
    ])
    if kind_raw is None:
        return
    kind = cast(ProviderKind, kind_raw)

    name = await _prompt("Profile name")
    if not name:
        return

    base_url: str | None = None
    context_window = 32_768
    max_output_tokens = 8192
    supports_vision: bool | None = None
    if kind == "openai":
        base_url = await _prompt("Base URL", "http://localhost:1234/v1") or None
        cw_raw = await _prompt("Context window (tokens)", "32768")
        try:
            context_window = int(cw_raw) if cw_raw else 32768
        except ValueError:
            context_window = 32768
        mot_raw = await _prompt("Max output tokens per turn", "8192")
        try:
            max_output_tokens = int(mot_raw) if mot_raw else 8192
        except ValueError:
            max_output_tokens = 8192
        supports_vision = await _prompt_vision(ctx, None)
    elif kind == "openai_native":
        mot_raw = await _prompt("Max output tokens per turn", "16384")
        try:
            max_output_tokens = int(mot_raw) if mot_raw else 16384
        except ValueError:
            max_output_tokens = 16384

    model = await _prompt("Model id")
    api_key_raw = await _prompt("API key (blank if none)", is_password=True)

    profile = ProviderProfile(
        name=name, kind=kind, base_url=base_url, model=model,
        api_key=api_key_raw or None, context_window=context_window,
        max_output_tokens=max_output_tokens, supports_vision=supports_vision,
    )
    try:
        ctx.providers.add(profile)
        ctx.console.print(f"[{theme.OK}]✓[/{theme.OK}] Added and activated profile '{name}'.")
    except ValueError as e:
        ctx.console.print(f"[{theme.ERR}]{e}[/{theme.ERR}]")


async def _pick_profile(ctx: CLIContext, title: str) -> str | None:
    names = ctx.providers.names()
    if not names:
        ctx.console.print(f"[{theme.DIM}]No profiles to choose from.[/{theme.DIM}]")
        return None
    return await _radio(ctx, title, [(n, n) for n in names])


async def _interactive_menu(ctx: CLIContext) -> None:
    from ..io import InputCancelled

    while True:
        _print_list(ctx)
        choice = await _radio(ctx, "Provider manager", [
            ("add",    "Add provider",    "Configure a new LM Studio / Ollama / Anthropic connection"),
            ("use",    "Set active",      "Switch which provider the agent uses"),
            ("edit",   "Edit provider",   "Update model, URL, or API key for an existing profile"),
            ("delete", "Delete provider", "Remove a provider profile permanently"),
            ("done",   "Done",            "Return to the chat prompt"),
        ])
        if choice in (None, "done"):
            return
        try:
            if choice == "add":
                await _add_profile(ctx)
            elif choice == "use":
                name = await _pick_profile(ctx, "Set active provider")
                if name:
                    ctx.console.print(f"[{theme.OK}]✓[/{theme.OK}] {op_use(ctx.providers, name)}")
            elif choice == "edit":
                name = await _pick_profile(ctx, "Edit provider")
                if name:
                    await _edit_profile(ctx, name)
            elif choice == "delete":
                name = await _pick_profile(ctx, "Delete provider")
                if name:
                    ctx.console.print(f"[{theme.OK}]✓[/{theme.OK}] {op_delete(ctx.providers, name)}")
        except InputCancelled:
            ctx.console.print(f"[{theme.DIM}]  ↩  cancelled[/{theme.DIM}]")


async def _edit_profile(ctx: CLIContext, name: str) -> None:
    p = ctx.providers.get(name)
    if p is None:
        return
    ctx.console.print(f"\n[bold {theme.ACCENT}]Edit '{name}'[/bold {theme.ACCENT}]")
    model = await _prompt("Model id", p.model)
    base_url = p.base_url
    max_output_tokens = p.max_output_tokens
    supports_vision = p.supports_vision
    if p.kind == "openai":
        base_url = (await _prompt("Base URL", p.base_url or "")) or None
        mot_raw = await _prompt("Max output tokens per turn", str(p.max_output_tokens))
        try:
            max_output_tokens = int(mot_raw) if mot_raw else p.max_output_tokens
        except ValueError:
            max_output_tokens = p.max_output_tokens
        supports_vision = await _prompt_vision(ctx, p.supports_vision)
    elif p.kind == "openai_native":
        mot_raw = await _prompt("Max output tokens per turn", str(p.max_output_tokens))
        try:
            max_output_tokens = int(mot_raw) if mot_raw else p.max_output_tokens
        except ValueError:
            max_output_tokens = p.max_output_tokens
    api_key_raw = await _prompt("API key (blank = keep existing)", is_password=True)
    changes: dict = {
        "model": model,
        "base_url": base_url,
        "max_output_tokens": max_output_tokens,
        "supports_vision": supports_vision,
    }
    if api_key_raw:
        changes["api_key"] = api_key_raw
    ctx.providers.update(name, **changes)
    ctx.console.print(f"[{theme.OK}]✓[/{theme.OK}] Updated profile '{name}'.")


# ── command handler ───────────────────────────────────────────────────────────
async def handle(ctx: CLIContext, args: list[str]) -> None:
    if not args:
        await _interactive_menu(ctx)
        return
    sub, *rest = args
    try:
        if sub == "list":
            _print_list(ctx)
        elif sub == "use" and rest:
            ctx.console.print(f"[{theme.OK}]✓[/{theme.OK}] {op_use(ctx.providers, rest[0])}")
        elif sub == "delete" and rest:
            ctx.console.print(f"[{theme.OK}]✓[/{theme.OK}] {op_delete(ctx.providers, rest[0])}")
        elif sub == "add":
            await _add_profile(ctx)
        else:
            ctx.console.print(
                f"[{theme.DIM}]Usage: /provider "
                r"\[list|use <name>|delete <name>|add]"
                f"[/{theme.DIM}]"
            )
    except (KeyError, ValueError) as e:
        ctx.console.print(f"[{theme.ERR}]{e}[/{theme.ERR}]")


COMMAND = SlashCommand(
    name="provider",
    summary="Manage provider profiles (add / set-active / edit / delete)",
    handler=handle,
    aliases=["providers", "pro"],
)
