"""Sub-agent spawning — child agents with forked context.

- Child gets its own empty history (not a fork of the parent's).
- Child shares the parent's provider but has its own tool pool (filtered by
  the SubAgentDefinition's allow/deny lists).
- Only the child's final report string returns to the parent.
- Parallel spawning is via ``asyncio.gather`` (foreground only; background deferred).
- Hard caps: ``MAX_DEPTH`` (absolute ceiling) and ``max_concurrent_subagents``
  from the active Guardrails.

Usage::

    runner = SubAgentRunner(full_pool, provider, io=io, cwd=cwd, guardrails=g)
    spawn_fn = runner.make_spawn_fn(parent_depth=0)
    # Wire spawn_fn into the parent Agent's ToolContext.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from neurosurfer.agents.runtime.permissions import Guardrails
from neurosurfer.llm.base import Provider
from neurosurfer.tools.base import IOHandler, SpawnFn, ToolPool

# Personas are registered by the product layer (neurosurfer.app), not the engine.
# The engine only reads the registry — it ships no personas of its own.
from .defs import get_agent

if TYPE_CHECKING:
    pass

log = logging.getLogger(__name__)

# Absolute ceiling regardless of guardrails config.
MAX_DEPTH = 3


class SubAgentRunner:
    """Factory that creates bound ``SpawnFn`` closures for parent agents.

    One ``SubAgentRunner`` per session; multiple parent agents share the same
    instance so the concurrency counter is session-scoped.
    """

    def __init__(
        self,
        full_pool: ToolPool,
        provider: Provider,
        *,
        io: IOHandler,
        cwd: Path,
        guardrails: Guardrails,
    ) -> None:
        self._full_pool = full_pool
        self._provider = provider
        self._io = io
        self._cwd = cwd
        self._guardrails = guardrails
        self._semaphore = asyncio.Semaphore(guardrails.max_concurrent_subagents)

    # ── public API ────────────────────────────────────────────────────────────

    def make_spawn_fn(self, parent_depth: int) -> SpawnFn:
        """Return a ``SpawnFn`` bound to ``parent_depth + 1``."""
        child_depth = parent_depth + 1

        async def _spawn(agent_type: str, prompt: str) -> str:
            return await self.spawn(agent_type, prompt, depth=child_depth)

        return _spawn

    async def spawn(self, agent_type: str, prompt: str, *, depth: int = 1) -> str:
        """Spawn a single sub-agent and return its final report."""
        if depth > MAX_DEPTH:
            return (
                f"[Sub-agent depth limit reached ({MAX_DEPTH}). "
                f"Cannot spawn '{agent_type}' at depth {depth}.]"
            )
        if depth > self._guardrails.max_subagent_depth:
            return (
                f"[Guardrails: max_subagent_depth={self._guardrails.max_subagent_depth} "
                f"exceeded at depth {depth}. Cannot spawn '{agent_type}'.]"
            )

        defn = get_agent(agent_type)
        if defn is None:
            return f"[Unknown sub-agent type: '{agent_type}'. Check agents registry.]"

        async with self._semaphore:
            return await self._run_child(defn, prompt, depth=depth)

    async def spawn_parallel(
        self, tasks: list[tuple[str, str]], *, depth: int = 1
    ) -> list[str]:
        """Spawn multiple sub-agents concurrently and return all reports."""
        return list(
            await asyncio.gather(
                *[self.spawn(agent_type, prompt, depth=depth) for agent_type, prompt in tasks]
            )
        )

    # ── internal ──────────────────────────────────────────────────────────────

    async def _run_child(self, defn: SubAgentDefinition, prompt: str, *, depth: int) -> str:  # type: ignore[name-defined]  # noqa: F821
        # Import here to avoid a circular import (Agent → subagent → Agent).
        from neurosurfer.agents.agentic_loop import AgenticLoop
        from neurosurfer.agents.conversation import events as ev_mod

        from .defs import SubAgentDefinition  # noqa: F401 (used in type comment)

        # Build the child's tool pool filtered by the agent definition.
        pool_names = self._full_pool.names()
        allowed_names = defn.resolve_tools(pool_names)
        child_pool = self._full_pool.select(allowed_names)

        # Children cannot spawn further agents (flat model for now).
        child_guardrails = Guardrails(
            write_scope=self._guardrails.write_scope,
            shell_policy=self._guardrails.shell_policy,
            path_allow=self._guardrails.path_allow,
            path_deny=self._guardrails.path_deny,
            max_turns=min(50, self._guardrails.max_turns),
            max_subagent_depth=0,
            max_concurrent_subagents=0,
        )

        child = AgenticLoop(
            provider=self._provider,
            tools=child_pool,
            system_prompt=defn.get_system_prompt(),
            guardrails=child_guardrails,
            io=self._io,
            cwd=self._cwd,
            depth=depth,
            # No spawn fn — child agents are leaf nodes (flat model).
            spawn=None,
        )

        log.debug("spawn depth=%d type=%s", depth, defn.agent_type)

        # Stream the child's events so tool activity is surfaced to the user
        # via io.notify(), which writes to the parent's console.  This mirrors
        # run_collect() but emits a notification for each tool call the child makes.
        text_parts: list[str] = []
        report = ""
        async for ev in child.run(prompt):
            if isinstance(ev, ev_mod.TextDelta):
                text_parts.append(ev.text)
            elif isinstance(ev, ev_mod.RunFinished):
                report = ev.report
            elif isinstance(ev, ev_mod.ToolStarted):
                msg = _child_tool_label(ev.name, ev.args)
                if msg:
                    self._io.notify(f"↳ {msg}")

        final_text = "".join(text_parts)
        return report or final_text or f"[{defn.agent_type} completed with no output]"


def _child_tool_label(name: str, args: dict) -> str:
    """Plain-text activity label for a sub-agent tool call (no Rich markup)."""
    from pathlib import Path
    if name == "read_file" and "path" in args:
        return f"Reading {Path(str(args['path'])).name}"
    if name == "list_dir":
        p = str(args.get("path") or ".").rstrip("/") or "."
        return f"Scouting {p}"
    if name == "search" and "pattern" in args:
        pat = str(args["pattern"])
        return f"Searching {pat!r}" if len(pat) < 40 else "Searching"
    if name == "web_search" and "query" in args:
        q = str(args["query"])
        return f"Web search: {q[:50]}"
    if name == "run_command" and "command" in args:
        cmd = str(args["command"])[:60]
        return f"Running: {cmd}"
    if name == "write_file" and "path" in args:
        return f"Writing {Path(str(args['path'])).name}"
    if name == "apply_edit" and "path" in args:
        return f"Editing {Path(str(args['path'])).name}"
    if name in ("finish", "ask_user", "present_plan", "todo"):
        return ""  # skip meta tools — too noisy
    return name
