"""The review pass — does this graph answer the question it was asked?

Every other gate in V3 checks something *decidable*. A `react` node either has
tools or it doesn't; a plan step either has a node or it doesn't; a capability
either resolves or it doesn't. Those are errors, and they refuse.

This one is different. It reads the intent, the plan and the finished graph and
asks whether the design actually does what was asked — a node summarising when it
should be extracting, a branch that never fires, a prompt that quietly answers a
narrower question than the user's. Verification (Phase 5) cannot catch these: the
workflow runs green and produces a plausible answer to the wrong question.

**It warns; it does not refuse.** The whole thesis of this plan is that gates beat
instructions, so the exception needs a reason. Blocking a build on one weak model's
judgement of another weak model's output is how V2's `verify="required"` stalled a
9B model — an unfalsifiable opinion, repeated. Judgements belong in warnings the
builder and the user both see; refusals are reserved for facts. The mode is
switchable for deployments that want it stricter.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from neurosurfer.llm.types import GenerationConfig, Message

logger = logging.getLogger(__name__)

__all__ = ["ReviewIssue", "ReviewReport", "review_workflow"]

_MAX_ISSUES = 6


@dataclass
class ReviewIssue:
    node: str = ""
    problem: str = ""
    fix: str = ""

    def render(self) -> str:
        where = f"[{self.node}] " if self.node else ""
        return f"{where}{self.problem}" + (f" → {self.fix}" if self.fix else "")

    def to_dict(self) -> dict[str, Any]:
        return {"node": self.node, "problem": self.problem, "fix": self.fix}


@dataclass
class ReviewReport:
    ok: bool = True
    issues: list[ReviewIssue] = field(default_factory=list)
    summary: str = ""
    #: True when the reviewer could not be reached or did not answer usably. A
    #: review that failed is not a review that passed, and saying so keeps a
    #: silent outage from reading as approval.
    inconclusive: bool = False

    def render(self) -> str:
        if self.inconclusive:
            return "REVIEW INCONCLUSIVE — the reviewer did not return a usable answer."
        if self.ok and not self.issues:
            return f"REVIEW OK — {self.summary}" if self.summary else "REVIEW OK."
        lines = [f"REVIEW found {len(self.issues)} issue(s):"]
        lines += [f"  • {i.render()}" for i in self.issues]
        if self.summary:
            lines.append(self.summary)
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "inconclusive": self.inconclusive,
            "summary": self.summary,
            "issues": [i.to_dict() for i in self.issues],
        }


_SYSTEM = """\
You review a workflow design against the request it was built for. You are the
last reader before it ships, and the only one asking whether it answers the RIGHT
question — something that cannot be caught by running it, because a workflow that
answers the wrong question runs perfectly well.

Output STRICT JSON only, no prose, no code fences:
  {"ok": true|false,
   "summary": "<one sentence: what this workflow does>",
   "issues": [{"node": "<node id or ''>", "problem": "<what is wrong>",
               "fix": "<the concrete change>"}]}

Report ONLY things that would make a user say "that's not what I asked for":
- a step that does something different from what the request needs;
- a clause of the request that no node addresses;
- a node whose prompt answers a narrower or broader question than intended;
- wiring that means a node cannot see data it needs;
- a branch, guard or loop condition that can never fire.

Do NOT report: style, naming, wording preferences, missing error handling,
missing validation, or anything the user did not ask for. An extra safeguard you
would personally add is NOT an issue. If the design does what was asked, return
{"ok": true, "issues": []} — that is the expected answer for a good build, and
inventing issues to seem thorough is worse than saying nothing.
Maximum 6 issues, most serious first.
"""


async def review_workflow(
    provider: Any,
    *,
    intent: str,
    graph_yaml: str,
    plan: Any = None,
    gen_config: GenerationConfig | None = None,
) -> ReviewReport:
    """One call: intent + plan + graph → issues worth fixing before registering."""
    from .agent.jsonio import parse_json

    parts = [f"The user asked for:\n{intent.strip()}"]
    if plan is not None and getattr(plan, "steps", None):
        parts.append(f"The plan it was built from:\n{plan.render()}")
    parts.append(f"The workflow as built:\n{graph_yaml}")
    parts.append("Does this do what was asked? Return the JSON verdict.")

    try:
        response = await provider.complete(
            messages=[Message.user_text("\n\n".join(parts))],
            system=_SYSTEM,
            tools=[],
            config=gen_config or GenerationConfig(stream=False),
        )
    except Exception as e:  # noqa: BLE001 - a reviewer outage must not fail a build
        logger.debug("review call failed: %s", e)
        return ReviewReport(ok=True, inconclusive=True, summary=str(e)[:200])

    data = parse_json(response.text() or "", want="ok")
    if not isinstance(data, dict):
        return ReviewReport(ok=True, inconclusive=True)

    issues = [
        ReviewIssue(
            node=str(raw.get("node") or ""),
            problem=str(raw.get("problem") or "").strip(),
            fix=str(raw.get("fix") or "").strip(),
        )
        for raw in (data.get("issues") or [])[:_MAX_ISSUES]
        if isinstance(raw, dict) and str(raw.get("problem") or "").strip()
    ]
    # `ok` and a non-empty issue list contradict each other often enough to need a
    # rule: the issues are the evidence, the flag is a summary of it.
    return ReviewReport(
        ok=bool(data.get("ok", True)) and not issues,
        issues=issues,
        summary=str(data.get("summary") or "").strip(),
    )
