"""Permission + guardrail enforcement.

Guardrails are **dual-homed**: described in the Task's system prompt *and* enforced
here. Prose-only guardrails are advisory; these are real. The loop calls
:meth:`Permissions.check` before every tool execution.
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from neurosurfer.tools.base import ToolContext
from neurosurfer.tools.utils import resolve_path

PermissionMode = Literal["plan", "default", "accept_edits", "bypass"]
ShellPolicy = Literal["gated", "readonly", "denied"]
# Network egress for the http/browse tools: gated = ask per request (default),
# open = always allow, denied = no network access at all.
NetworkPolicy = Literal["gated", "open", "denied"]
# External MCP-tool calls: gated = ask per non-read-only call (default; read-only
# tools pass), open = always allow, denied = no MCP calls at all.
McpPolicy = Literal["gated", "open", "denied"]

READ_TOOLS = {"read_file", "list_dir", "search", "data"}
WRITE_TOOLS = {"write_file", "apply_edit"}
NETWORK_TOOLS = {"http", "browse"}
CONTROL_TOOLS = {"ask_user", "present_plan", "propose_workflow", "todo", "finish", "spawn_agent"}


class Guardrails(BaseModel):
    """Enforced limits. Shared by the engine (here) and Tasks (Phase 6)."""

    write_scope: list[str] = Field(default_factory=lambda: ["**"])
    shell_policy: ShellPolicy = "gated"
    network_policy: NetworkPolicy = "gated"
    mcp_policy: McpPolicy = "gated"
    path_allow: list[str] = Field(default_factory=lambda: ["**"])
    path_deny: list[str] = Field(
        default_factory=lambda: [".env", "**/.env", "**/secrets/**", ".git/**", "**/.git/**"]
    )
    max_turns: int = 200
    max_subagent_depth: int = 2
    max_concurrent_subagents: int = 4


@dataclass
class Decision:
    allow: bool
    reason: str = ""


def _relposix(cwd: Path, raw: str) -> tuple[str, bool]:
    """Return (path-string-for-matching, is_inside_cwd)."""
    p = resolve_path(cwd, raw)
    try:
        rel = p.resolve().relative_to(cwd.resolve())
        return rel.as_posix(), True
    except ValueError:
        return p.as_posix(), False


def _match_any(globs: list[str], relpath: str) -> bool:
    base = relpath.rsplit("/", 1)[-1]
    for g in globs:
        if fnmatch.fnmatch(relpath, g) or fnmatch.fnmatch(base, g):
            return True
    return False


def _glob_scope_match(cwd: Path, pattern: str, target: Path) -> bool:
    if fnmatch.fnmatch(target.as_posix(), pattern):
        return True
    try:
        rel = target.relative_to(cwd).as_posix()
    except ValueError:
        return False
    return fnmatch.fnmatch(rel, pattern) or fnmatch.fnmatch(rel.rsplit("/", 1)[-1], pattern)


def _in_write_scope(cwd: Path, scope: list[str], target: Path) -> bool:
    """True if absolute ``target`` falls under any scope entry.

    Entries may be relative (resolved against ``cwd``), absolute (e.g. a folder
    the user approved with "always"), ``**``/``*``/``.`` (everything), or globs.
    """
    if not scope:
        return False
    cwd_r = cwd.resolve()
    for raw in scope:
        s = raw.strip()
        if s in ("**", "*", ".", "./", ""):
            return True
        if any(ch in s for ch in "*?["):
            if _glob_scope_match(cwd_r, s, target):
                return True
            continue
        base = Path(s).expanduser()
        if not base.is_absolute():
            base = cwd_r / base
        base = base.resolve()
        if target == base or base in target.parents:
            return True
    return False


class Permissions:
    def __init__(self, guardrails: Guardrails, cwd: Path):
        self.guardrails = guardrails
        self.cwd = cwd

    async def check(
        self,
        tool_name: str,
        args: BaseModel,
        ctx: ToolContext,
        mode: PermissionMode,
        tool: object | None = None,
    ) -> Decision:
        if mode == "bypass":
            return Decision(True)

        # External MCP tools are gated by their own policy (read-only calls pass).
        if getattr(tool, "is_mcp", False):
            return await self._check_mcp(tool, tool_name, args, ctx)

        if tool_name in READ_TOOLS:
            return self._check_read(args)
        if tool_name == "run_command":
            return await self._check_shell(args, ctx, mode)
        if tool_name in WRITE_TOOLS:
            return await self._check_write(args, ctx, mode)
        if tool_name in NETWORK_TOOLS:
            return await self._check_network(args, ctx, mode)
        if tool_name in CONTROL_TOOLS:
            return Decision(True)
        # Unknown tools: allow (the loop's own guards still apply).
        return Decision(True)

    # ── MCP ──────────────────────────────────────────────────────────────────
    async def _check_mcp(
        self, tool: object, tool_name: str, args: BaseModel, ctx: ToolContext
    ) -> Decision:
        policy = self.guardrails.mcp_policy
        if policy == "denied":
            return Decision(False, "MCP tool calls are disabled for this task.")
        if policy == "open":
            return Decision(True)
        # gated: read-only tools pass; anything that may mutate is asked per call.
        try:
            read_only = bool(tool.is_read_only(args))  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001 - conservative: treat as not read-only
            read_only = False
        if read_only:
            return Decision(True)
        approved = await ctx.io.request_shell_approval(tool_name, "call an MCP tool")
        if approved:
            return Decision(True)
        return Decision(False, f"User declined the MCP tool call '{tool_name}'.")

    # ── reads ────────────────────────────────────────────────────────────────
    def _check_read(self, args: BaseModel) -> Decision:
        raw = getattr(args, "path", None) or "."
        rel, _ = _relposix(self.cwd, str(raw))
        if _match_any(self.guardrails.path_deny, rel):
            return Decision(False, f"Reading '{raw}' is denied by guardrails.")
        return Decision(True)

    # ── shell ────────────────────────────────────────────────────────────────
    async def _check_shell(
        self, args: BaseModel, ctx: ToolContext, mode: PermissionMode
    ) -> Decision:
        policy = self.guardrails.shell_policy
        if policy == "denied":
            return Decision(False, "Shell commands are disabled for this task.")
        command = getattr(args, "command", "")
        description = getattr(args, "description", "") or "run a shell command"
        if policy == "readonly":
            return Decision(True)
        # gated: ask per command (unless accept_edits/bypass already trusted shell —
        # we still gate shell even in accept_edits, which only auto-approves writes).
        approved = await ctx.io.request_shell_approval(command, description)
        if approved:
            return Decision(True)
        return Decision(False, "User declined to run the shell command.")

    # ── network ──────────────────────────────────────────────────────────────
    async def _check_network(
        self, args: BaseModel, ctx: ToolContext, mode: PermissionMode
    ) -> Decision:
        policy = self.guardrails.network_policy
        if policy == "denied":
            return Decision(False, "Network access is disabled for this task.")
        if policy == "open":
            return Decision(True)
        # gated: approve per request. Reuse the shell-approval channel so a
        # non-interactive (headless) IO denies network exactly as it denies shell.
        url = str(getattr(args, "url", "") or "")
        method = str(getattr(args, "method", "GET"))
        label = f"{method} {url}".strip()
        approved = await ctx.io.request_shell_approval(label, "make a network request")
        if approved:
            return Decision(True)
        return Decision(False, "User declined the network request.")

    # ── writes ───────────────────────────────────────────────────────────────
    async def _check_write(
        self, args: BaseModel, ctx: ToolContext, mode: PermissionMode
    ) -> Decision:
        if mode == "plan":
            return Decision(
                False,
                "In plan mode: present a plan with present_plan and get approval "
                "before writing any files.",
            )
        raw = getattr(args, "path", "")
        rel, _inside = _relposix(self.cwd, str(raw))
        target = resolve_path(self.cwd, str(raw)).resolve()
        # path_deny is a hard boundary (secrets, .git, …) — never negotiable.
        if _match_any(self.guardrails.path_deny, rel):
            return Decision(False, f"Writing '{raw}' is denied by guardrails.")
        if _in_write_scope(self.cwd, self.guardrails.write_scope, target):
            return Decision(True)
        # Out of the Task's declared write scope. Rather than hard-deny, escalate
        # to the user (the scope is the agent's *default* boundary; the human can
        # consciously allow a location). Three answers:
        #   always → allow + persist the folder to the task's write scope
        #   once   → allow just this write, scope unchanged
        #   deny   → refuse
        scope = ", ".join(self.guardrails.write_scope) or "(none)"
        summary = f"This task is normally allowed to write only to: {scope}."
        choice = await ctx.io.request_write_approval(str(raw), summary)
        if choice == "deny":
            return Decision(
                False,
                f"Writing '{raw}' was declined (outside the task's write scope: {scope}).",
            )
        if choice == "always":
            self._widen_scope(target.parent, ctx)
        return Decision(True)

    def _widen_scope(self, directory: Path, ctx: ToolContext) -> None:
        """Add ``directory`` to the live write scope and persist it if possible."""
        entry = directory.as_posix().rstrip("/") + "/"
        if entry not in self.guardrails.write_scope:
            self.guardrails.write_scope.append(entry)  # session-wide
        if ctx.persist_scope is not None:
            ctx.persist_scope(entry)  # cross-session (runner/REPL handle errors)


def initial_mode(plan_required: bool) -> PermissionMode:
    return "plan" if plan_required else "default"
