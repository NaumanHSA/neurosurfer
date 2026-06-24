"""Tool dispatch for one assistant turn.

The assistant's ``tool_use`` blocks are partitioned into batches of **consecutive**
concurrency-safe tools, with every non-concurrency-safe tool as its own singleton
batch. Batches run **in listed order**; a concurrency-safe batch runs its members
in parallel (``asyncio.gather``). This preserves ordering between writes/shell and
the reads around them (a write never gets reordered before a later read in the same
turn), which the simpler all-reads-first split got wrong.

Permission gating happens at execution time, inside each batch and in order.
Tool errors and permission denials both come back as ``tool_result(is_error=True)``
so the model self-corrects next turn.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from pydantic import BaseModel, ValidationError

from neurosurfer.llm.types import ToolUseBlock
from neurosurfer.tools.base import Tool, ToolContext, ToolPool, ToolResult

from .permissions import PermissionMode, Permissions


@dataclass
class ToolOutcome:
    id: str
    name: str
    args: dict
    result: ToolResult


@dataclass
class _Planned:
    tu: ToolUseBlock
    tool: Tool | None
    args: BaseModel | None
    parse_error: str | None
    concurrency_safe: bool


@dataclass
class _Batch:
    concurrency_safe: bool
    items: list[_Planned] = field(default_factory=list)


def _plan(tool_uses: list[ToolUseBlock], tools: ToolPool) -> list[_Planned]:
    """Parse args and decide concurrency-safety for each call (no gating yet).

    A call is concurrency-safe only if its args parse *and* the tool reports it
    safe; any failure is conservatively unsafe.
    """
    planned: list[_Planned] = []
    for tu in tool_uses:
        tool = tools.get(tu.name)
        if tool is None:
            planned.append(_Planned(tu, None, None, f"Unknown tool: {tu.name}", False))
            continue
        try:
            args = tool.parse_args(tu.input)
        except ValidationError as e:
            planned.append(_Planned(tu, tool, None, f"Invalid arguments: {e}", False))
            continue
        try:
            safe = bool(tool.is_concurrency_safe(args))
        except Exception:  # noqa: BLE001 - conservative: treat as unsafe
            safe = False
        planned.append(_Planned(tu, tool, args, None, safe))
    return planned


def _partition(planned: list[_Planned]) -> list[_Batch]:
    """Group consecutive concurrency-safe calls; each unsafe call is its own batch."""
    batches: list[_Batch] = []
    for p in planned:
        if p.concurrency_safe and batches and batches[-1].concurrency_safe:
            batches[-1].items.append(p)
        else:
            batches.append(_Batch(concurrency_safe=p.concurrency_safe, items=[p]))
    return batches


async def execute_tool_uses(
    tool_uses: list[ToolUseBlock],
    *,
    tools: ToolPool,
    ctx: ToolContext,
    permissions: Permissions,
    mode: PermissionMode,
) -> list[ToolOutcome]:
    planned = _plan(tool_uses, tools)

    async def run_one(p: _Planned) -> tuple[str, ToolResult]:
        if p.tool is None:
            return p.tu.id, ToolResult.error(p.parse_error or "Unknown tool.")
        if p.args is None:
            return p.tu.id, ToolResult.error(p.parse_error or "Invalid arguments.")
        decision = await permissions.check(p.tu.name, p.args, ctx, mode)
        if not decision.allow:
            return p.tu.id, ToolResult.error(decision.reason or "Not permitted.")
        result = await p.tool.run(p.tu.input, ctx)  # re-validates + guards exceptions
        return p.tu.id, result

    results: dict[str, ToolResult] = {}
    for batch in _partition(planned):
        if batch.concurrency_safe and len(batch.items) > 1:
            for tid, res in await asyncio.gather(*(run_one(p) for p in batch.items)):
                results[tid] = res
        else:
            for p in batch.items:
                tid, res = await run_one(p)
                results[tid] = res

    return [
        ToolOutcome(id=tu.id, name=tu.name, args=tu.input, result=results[tu.id])
        for tu in tool_uses
    ]
