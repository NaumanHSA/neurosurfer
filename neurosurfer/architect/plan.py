"""The workflow plan — a reviewable artifact, produced before any node exists.

Phase 1 made an ungrounded workflow unregisterable. Phase 2 gave the agent a way to
find what a step needs. Neither changed *when* the question gets asked: the builder
still decided a node's kind and tools in the same breath as writing its prompt,
inside a ReAct stream that also held requirements, wiring, validation and
verification. A small model drops things from the middle of that.

A plan pulls one question out and asks it on its own, per step, before anything is
built: **does this step reach outside the model, and if so, what provides it?**

That is what :attr:`PlanStep.is_external` is for, and it is the field that makes the
original bug structurally impossible. A step declared external cannot become a node
without a resolved capability behind it; the lexical check in
``graph/workflow/capability.py`` stops being the only thing standing between a
plausible sentence and a workflow that invents its results.

The plan is data, not prose. It is rendered for the model, shown to the user,
gated on when requirement-gathering is on, and carried on the session so the
builder, the review pass and verification all read the same artifact.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

__all__ = ["PlanStep", "WorkflowPlan", "PlanInput"]

# Node kinds a plan step may propose. The planner picks the *shape* of a step; the
# builder writes it. Anything outside this is normalised to `base`, because a wrong
# kind that parses is worse than an obvious default.
KIND_HINTS = ("base", "tool", "react", "router", "loop", "map")


class PlanInput(BaseModel):
    """One value the finished workflow will be given when it runs."""

    name: str
    type: str = "string"
    required: bool = True
    description: str = ""

    def to_node_input(self) -> dict[str, Any]:
        return {"name": self.name, "type": self.type,
                "required": self.required, "description": self.description}


class PlanStep(BaseModel):
    """One step of the intended workflow, and what it needs to be real."""

    id: str = Field(description="snake_case identifier, becomes the node id.")
    intent: str = Field(description="What this step does, in one sentence.")
    kind_hint: str = Field(default="base", description=f"One of {KIND_HINTS}.")
    depends_on: list[str] = Field(default_factory=list)
    produces: str = Field(default="", description="What it hands downstream.")

    # The question the whole plan exists to ask.
    is_external: bool = Field(
        default=False,
        description="True if this step must reach outside the model — read a file, "
                    "call an API, touch an inbox, send a message, run a command.",
    )
    needed_capability: str = Field(
        default="",
        description="When is_external: the capability in plain English, e.g. "
                    "'read a file from disk'.",
    )
    secrets: list[str] = Field(
        default_factory=list,
        description="Names of stored values this step needs (DB_PASSWORD, "
                    "API_TOKEN). A credential is declared here and NEVER as a "
                    "graph input — inputs are in the interpolation scope, so one "
                    "declared there is reachable from every node's prompt.",
    )

    # Filled by the resolution pass; never by the model.
    resolution: dict[str, Any] | None = None

    @property
    def resolved(self) -> bool:
        """Can this step be built **right now**? Internal steps always can."""
        if not self.is_external:
            return True
        status = (self.resolution or {}).get("status")
        return status in {"have", "not_external"}

    @property
    def reachable(self) -> bool:
        """Can this step be built, possibly after acquiring something?

        The distinction `resolved` could not make. A step whose capability lives
        on an MCP server nobody has installed is not *blocked* — it is one
        approved install away, and the agent has a tool for exactly that. Treating
        it as fatal is why a build was declared infeasible with the server it
        needed sitting in the checklist beside the refusal.

        `none` is reachable too when the taxonomy has *not* declared a curated gap:
        nothing off-the-shelf provides it, but it may still be authorable in
        Python — and blocking at the plan is what made `author_tool` unreachable
        for external steps despite the resolver telling the model to use it.
        """
        if self.resolved:
            return True
        resolution = self.resolution or {}
        if resolution.get("status") == "installable":
            return True
        return resolution.get("status") == "none" and not resolution.get("curated_gap")

    @property
    def tool_names(self) -> list[str]:
        return [t["name"] for t in (self.resolution or {}).get("tools", [])]

    def render(self) -> str:
        bits = [f"  {self.id} ({self.kind_hint}) — {self.intent}"]
        if self.depends_on:
            bits.append(f"      after: {', '.join(self.depends_on)}")
        if self.is_external:
            status = (self.resolution or {}).get("status", "unresolved")
            line = f"      needs: {self.needed_capability or '?'} [{status}]"
            tools = self.tool_names
            if tools:
                line += f" → use {', '.join(f'`{t}`' for t in tools)}"
            bits.append(line)
        if self.secrets:
            # The builder reads this render. Without the line the declaration
            # stops at the plan and the node is written without `secrets:`,
            # which is how the credential ends up interpolated instead.
            bits.append(
                f"      secrets: {', '.join(self.secrets)} "
                f"— put these in tool_args as ${{NAME}}, never in a prompt"
            )
        return "\n".join(bits)


class WorkflowPlan(BaseModel):
    """The whole intended design, before a single node is written."""

    name: str = Field(default="", description="snake_case workflow name.")
    description: str = ""
    inputs: list[PlanInput] = Field(default_factory=list)
    steps: list[PlanStep] = Field(default_factory=list)
    outputs: list[str] = Field(default_factory=list,
                               description="Step ids whose output is the result.")
    open_questions: list[str] = Field(default_factory=list)

    # ── views ───────────────────────────────────────────────────────────────
    @property
    def external_steps(self) -> list[PlanStep]:
        return [s for s in self.steps if s.is_external]

    @property
    def unresolved_steps(self) -> list[PlanStep]:
        """Steps that cannot be built as things stand — including acquirable ones."""
        return [s for s in self.steps if not s.resolved]

    @property
    def acquirable_steps(self) -> list[PlanStep]:
        """Steps one approved acquisition away — an install, or an authored tool."""
        return [s for s in self.steps if not s.resolved and s.reachable]

    @property
    def impossible_steps(self) -> list[PlanStep]:
        """Steps nothing here can provide. **These** are what a plan blocks on."""
        return [s for s in self.steps if not s.reachable]

    def step(self, step_id: str) -> PlanStep | None:
        return next((s for s in self.steps if s.id == step_id), None)

    def render(self) -> str:
        """The plan as the model and the user both read it."""
        lines = [f"PLAN: {self.name or '(unnamed)'}"]
        if self.description:
            lines.append(self.description)
        if self.inputs:
            lines.append("Inputs: " + ", ".join(
                f"{i.name}: {i.type}{'' if i.required else ' (optional)'}"
                for i in self.inputs
            ))
        lines.append(f"Steps ({len(self.steps)}):")
        lines += [s.render() for s in self.steps]
        if self.outputs:
            lines.append(f"Result: {', '.join(self.outputs)}")
        if self.open_questions:
            lines.append("Open questions:")
            lines += [f"  - {q}" for q in self.open_questions]
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    # ── normalisation ───────────────────────────────────────────────────────
    def normalised(self) -> WorkflowPlan:
        """Repair what a small model reliably gets slightly wrong.

        None of this is defensive programming for its own sake — each rule is a
        shape that came back from a real model and would otherwise have produced a
        plan the builder could not follow.
        """
        import re

        seen: set[str] = set()
        steps: list[PlanStep] = []
        for step in self.steps:
            sid = re.sub(r"[^a-z0-9_]+", "_", (step.id or "").strip().lower()).strip("_")
            if not sid or sid in seen:
                sid = f"step_{len(steps) + 1}"
            seen.add(sid)
            kind = step.kind_hint if step.kind_hint in KIND_HINTS else "base"
            # A step that names a capability is external whatever the flag says;
            # models set the prose and forget the boolean far more often than the
            # reverse.
            external = bool(step.is_external or step.needed_capability.strip())
            # `tool` and `react` exist to call tools. On a step that reaches
            # nothing they produce a node with an empty toolbelt — which the
            # Phase 1 gate rejects outright for `react`. Demote rather than let
            # the plan describe something unbuildable.
            if not external and kind in {"tool", "react"}:
                kind = "base"
            steps.append(step.model_copy(update={
                "id": sid, "kind_hint": kind, "is_external": external,
            }))

        ids = {s.id for s in steps}
        for step in steps:
            step.depends_on = [d for d in step.depends_on if d in ids and d != step.id]

        name = re.sub(r"[^a-z0-9_]+", "_", (self.name or "").strip().lower()).strip("_")
        outputs = [o for o in self.outputs if o in ids]
        if not outputs and steps:
            # No declared result is not a plan; the last step is the honest guess.
            outputs = [steps[-1].id]
        return self.model_copy(update={
            "name": name or "generated_workflow", "steps": steps, "outputs": outputs,
        })
