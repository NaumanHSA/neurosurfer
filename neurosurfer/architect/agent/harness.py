"""A/B evaluation harness: ReAct architect vs. legacy staged pipeline (Phase 4d).

Runs a fixed suite of intents through named builder callables and compares
outcome quality: success rate, validation status of what was registered, node
count (design depth), and wall time. This is the evidence that decides whether
the ReAct agent replaces the legacy pipeline (plan §8) — no pre-commitment.

Builders are plain async callables ``(intent) -> registered_path`` so anything
can compete (the agent, the legacy pipeline, future variants), and tests can
inject stubs. ``default_builders(provider)`` wires the two real contenders.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "EvalCase", "run_harness", "render_report", "default_builders",
    "SUITE_BASIC", "SUITE_CONTROL_FLOW",
]

Builder = Callable[[str], Awaitable[str]]

# Canonical intent suites for A/B runs. BASIC = linear pipelines; CONTROL_FLOW =
# intents that *warrant* routers/loops/maps — grading whether a builder reaches
# for the Phase 1 constructs when the task calls for them.
SUITE_BASIC: list[str] = [
    "Build a workflow that takes a short article text as input, summarises it in "
    "3 sentences, and then writes a catchy title for the summary.",
    "Build a workflow that takes a product name as input and writes a one-paragraph "
    "marketing blurb for it, ending with a call to action.",
    "Build a workflow that takes meeting notes as input and produces a list of "
    "action items with owners.",
]

SUITE_CONTROL_FLOW: list[str] = [
    # → router
    "Build a workflow that takes a customer support ticket as input, decides "
    "whether it is urgent or routine, and drafts an escalation notice for urgent "
    "tickets but a polite standard reply for routine ones.",
    # → loop
    "Build a workflow that takes a topic as input and drafts a short poem about "
    "it, then reviews its own draft and redrafts until the review says it is good "
    "(at most 3 attempts).",
    # → map
    "Build a workflow that takes a list of city names as input and, for each "
    "city, writes a one-line travel tip, returning all tips together.",
]


@dataclass
class EvalCase:
    """Outcome of one (builder, intent) evaluation."""

    builder: str
    intent: str
    ok: bool = False
    registered_path: str | None = None
    error: str = ""
    seconds: float = 0.0
    node_count: int = 0
    kinds: list[str] = field(default_factory=list)
    validation_ok: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "builder": self.builder, "intent": self.intent, "ok": self.ok,
            "registered_path": self.registered_path, "error": self.error,
            "seconds": round(self.seconds, 2), "node_count": self.node_count,
            "kinds": self.kinds, "validation_ok": self.validation_ok,
        }


def _inspect_package(path: str) -> tuple[int, list[str], bool]:
    """(node_count, kinds, validation_ok) for a registered package path."""
    from neurosurfer.graph.workflow.package import load_package
    from neurosurfer.graph.workflow.validate import validate_package

    pkg = load_package(Path(path))
    kinds = [n.kind for n in pkg.graph.nodes]
    return len(pkg.graph.nodes), kinds, validate_package(pkg).ok


async def run_harness(
    intents: list[str],
    builders: dict[str, Builder],
    *,
    notify: Callable[[str], None] | None = None,
) -> list[EvalCase]:
    """Run every intent through every builder; never let one failure stop the sweep."""
    say = notify or (lambda _m: None)
    results: list[EvalCase] = []
    for name, build in builders.items():
        for intent in intents:
            say(f"[{name}] {intent[:70]}…")
            case = EvalCase(builder=name, intent=intent)
            start = time.time()
            try:
                path = await build(intent)
                case.seconds = time.time() - start
                case.registered_path = str(path)
                case.node_count, case.kinds, case.validation_ok = _inspect_package(path)
                case.ok = True
            except Exception as e:  # noqa: BLE001 - a failed build is a data point
                case.seconds = time.time() - start
                case.error = f"{type(e).__name__}: {e}"
            results.append(case)
            say(f"  → {'ok' if case.ok else 'FAILED'} in {case.seconds:.1f}s")
    return results


def render_report(results: list[EvalCase]) -> str:
    """Markdown comparison: per-case table + per-builder summary."""
    lines = ["# Architect A/B report", "", "| builder | intent | ok | nodes | kinds | valid | seconds |",
             "|---|---|---|---|---|---|---|"]
    for c in results:
        intent = c.intent if len(c.intent) <= 60 else c.intent[:60] + "…"
        lines.append(
            f"| {c.builder} | {intent} | {'✅' if c.ok else '❌'} | {c.node_count or ''} "
            f"| {','.join(c.kinds)} | "
            f"{'' if c.validation_ok is None else ('✅' if c.validation_ok else '❌')} "
            f"| {c.seconds:.1f} |"
        )
    lines += ["", "## Summary", "", "| builder | ok rate | avg nodes | avg seconds |",
              "|---|---|---|---|"]
    by_builder: dict[str, list[EvalCase]] = {}
    for c in results:
        by_builder.setdefault(c.builder, []).append(c)
    for name, cases in by_builder.items():
        oks = [c for c in cases if c.ok]
        ok_rate = f"{len(oks)}/{len(cases)}"
        avg_nodes = (sum(c.node_count for c in oks) / len(oks)) if oks else 0
        avg_secs = sum(c.seconds for c in cases) / len(cases)
        lines.append(f"| {name} | {ok_rate} | {avg_nodes:.1f} | {avg_secs:.1f} |")
    failed = [c for c in results if not c.ok]
    if failed:
        lines += ["", "## Failures", ""]
        lines += [f"- **{c.builder}** / {c.intent[:60]}: {c.error}" for c in failed]
    return "\n".join(lines)


def default_builders(
    provider: Any,
    *,
    registry: Any = None,
    staging_root: Path | None = None,
    approve_tool: Any = None,
) -> dict[str, Builder]:
    """The two real contenders: the ReAct agent and the legacy staged pipeline."""
    from ..build import ArchitectBuilder
    from .agent import ArchitectAgent

    async def react_agent(intent: str) -> str:
        agent = ArchitectAgent(
            provider, registry=registry, staging_root=staging_root,
            approve_tool=approve_tool,
        )
        return await agent.build(intent)

    async def legacy_pipeline(intent: str) -> str:
        # Non-empty answers skip the clarify node's interactive input() — the
        # harness must run headless.
        return await ArchitectBuilder(provider).run(
            intent,
            answers={"clarifications": "none; build from the intent exactly as stated"},
            approve_tool=approve_tool,
        )

    return {"react_agent": react_agent, "legacy_pipeline": legacy_pipeline}
