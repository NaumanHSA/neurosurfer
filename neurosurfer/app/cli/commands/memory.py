"""/memory — inspect and curate long-term memory.

  memory list [agent]      list global (+ an agent's) memories
  memory search <query>    rank memories against a query (BM25)
  memory add <text>        add a global fact
  memory forget <id>       remove a memory by id
  memory curate            review/clean memories by conversation (memory_curator)

Mirrors the /task command shape: a single ``handle`` dispatches subcommands, and a
``COMMAND`` binds it for the REPL.
"""

from __future__ import annotations

from .. import theme
from ..context import CLIContext
from .base import SlashCommand


def _store(ctx: CLIContext):
    from neurosurfer.memory.store import MemoryStore

    return MemoryStore(ctx.cfg.memory.dir)


def _print_entries(ctx: CLIContext, entries) -> None:
    if not entries:
        ctx.console.print(f"[{theme.DIM}]No memories.[/{theme.DIM}]")
        return
    for e in entries:
        scope = "global" if e.scope == "global" else f"agent:{e.scope_key}"
        ctx.console.print(
            f"  [{theme.ACCENT}]{e.id}[/{theme.ACCENT}] "
            f"[{theme.DIM}]{scope} · {e.kind}[/{theme.DIM}]  {e.text}"
        )


async def handle_memory(ctx: CLIContext, args: list[str]) -> None:
    if not ctx.cfg.memory.enabled:
        ctx.console.print(f"[{theme.WARN}]Memory is disabled (NEUROSURFER_MEMORY=0).[/{theme.WARN}]")
        return

    sub = args[0] if args else "list"
    rest = args[1:]
    store = _store(ctx)

    if sub == "list":
        agent = rest[0] if rest else None
        entries = store.all_in_scope(agent) if agent else store.list_scope("global")
        title = f"global + agent:{agent}" if agent else "global"
        ctx.console.print(f"[{theme.ACCENT_DIM}]Memories ({title}):[/{theme.ACCENT_DIM}]")
        _print_entries(ctx, entries)
        if not agent:
            ctx.console.print(f"[{theme.DIM}]Add an agent name to include its scope: /memory list code[/{theme.DIM}]")
        return

    if sub == "search":
        if not rest:
            ctx.console.print(f"[{theme.ERR}]Usage: /memory search <query>[/{theme.ERR}]")
            return
        from neurosurfer.memory.embeddings import get_embedder
        from neurosurfer.memory.retrieval import retrieve

        query = " ".join(rest)
        result = retrieve(
            store.list_all(), query,
            budget_tokens=ctx.cfg.memory.token_budget,
            embedder=get_embedder(ctx.cfg.memory.embeddings_backend),
        )
        if not result.block:
            ctx.console.print(f"[{theme.DIM}]No matching memories.[/{theme.DIM}]")
            return
        # markup=False: the block contains "[kind]" tags that Rich would parse as style markup.
        ctx.console.print(result.block, markup=False)
        return

    if sub == "add":
        if not rest:
            ctx.console.print(f"[{theme.ERR}]Usage: /memory add <text>[/{theme.ERR}]")
            return
        from neurosurfer.memory.models import MemoryEntry

        entry = store.add(MemoryEntry(scope="global", kind="fact", text=" ".join(rest), source="user"))
        ctx.console.print(f"[{theme.OK}]Saved global memory[/{theme.OK}] [{theme.ACCENT}]{entry.id}[/{theme.ACCENT}].")
        return

    if sub == "forget":
        if not rest:
            ctx.console.print(f"[{theme.ERR}]Usage: /memory forget <id>[/{theme.ERR}]")
            return
        ok = store.forget(rest[0])
        msg = (f"[{theme.OK}]Forgot {rest[0]}.[/{theme.OK}]" if ok
               else f"[{theme.ERR}]No memory '{rest[0]}'.[/{theme.ERR}]")
        ctx.console.print(msg)
        return

    if sub == "curate":
        await _run_curator(ctx)
        return

    ctx.console.print(f"[{theme.ERR}]Unknown subcommand '{sub}'.[/{theme.ERR}] Try list/search/add/forget/curate.")


async def _run_curator(ctx: CLIContext) -> None:
    """Launch the memory_curator task, seeding it with the current memories."""
    from ..run import run_task

    store = _store(ctx)
    entries = store.list_all()
    if not entries:
        ctx.console.print(f"[{theme.DIM}]Nothing to curate yet — no memories saved.[/{theme.DIM}]")
        return
    rendered = "\n".join(
        f"- id={e.id} | {'global' if e.scope == 'global' else 'agent:' + e.scope_key} "
        f"| {e.kind} | {e.text}"
        for e in entries
    )
    try:
        task = ctx.registry.get("memory_curator")
    except Exception as exc:  # noqa: BLE001
        ctx.console.print(f"[{theme.ERR}]memory_curator unavailable: {exc}[/{theme.ERR}]")
        return
    await run_task(ctx, task, provided={"current_memories": rendered})


COMMAND = SlashCommand(
    name="memory",
    summary="Inspect or curate long-term memory (list/search/add/forget/curate)",
    handler=handle_memory,
    aliases=["mem"],
)
