"""Rules for the kinds that call a model — `base` and `react`.

Everything here is about the call itself: what the step is told to do, what it
runs on, and what shape it must answer in. Wiring is elsewhere; so is whether the
tools it names exist.
"""

from __future__ import annotations

from neurosurfer.graph.engine.kinds import node_kind_spec
from neurosurfer.graph.engine.templates import node_instruction

from ..models import Severity, ValidationIssue
from ..registry import node_rule

#: The kinds that make a model call. `router` classifies with one too, but its
#: prompt is a fixed-format question the engine writes, so the rules about *your*
#: prompt do not apply to it.
AGENT_KINDS = ("base", "react")


@node_rule(kinds=AGENT_KINDS, severity=Severity.WARNING)
def agent_has_instructions(node, ctx, report) -> None:
    """A step that calls a model with nothing to do."""
    if (node_instruction(node) or "").strip():
        return
    report.add(ValidationIssue(
        severity=Severity.WARNING,
        kind="agent.no_instructions",
        node_id=node.id,
        message="This step has no instructions, so the model is not told what to do.",
        suggestion="Say what this step should do in its instructions.",
    ))


def _environment_can_supply_a_model() -> bool:
    """Can the gateway's own environment answer a model call?

    This is the ``.env`` fallback the studio shows as *Local · .env fallback*: no
    provider profile is configured, but ``LLM_PROVIDER`` and a key (or a local
    base URL) are, and runs work. Distinguishing it from *nothing at all* is the
    difference between a nudge and a refusal.
    """
    import os  # noqa: PLC0415 - read at call time; a test may set these

    return any(
        (os.environ.get(name) or "").strip()
        for name in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "OPENAI_BASE_URL")
    )


@node_rule(kinds=AGENT_KINDS, severity=Severity.ERROR)
def agent_has_a_model(node, ctx, report) -> None:
    """A step that calls a model, and what it will run on.

    Two outcomes, because there are two situations and only one of them is
    broken:

    * **Nothing to run on** — no profile configured and no environment fallback.
      The call cannot be made, so this is an error and registration stops.
    * **Something to fall back to** — a default profile, or the gateway's own
      environment. The step runs, so this is a warning that encourages naming a
      model rather than inheriting whatever the run happens to start with.

    Skipped entirely when the caller does not say which profiles exist: most
    callers genuinely do not know, and guessing would reject valid graphs on a
    machine that simply has not been set up yet.
    """
    if getattr(node, "provider", None):
        return  # names one; whether it exists is `provider_is_configured`
    if ctx.known_providers is None:
        return  # not told — see the docstring

    if ctx.known_providers or _environment_can_supply_a_model():
        report.add(ValidationIssue(
            severity=Severity.WARNING,
            kind="agent.no_model",
            node_id=node.id,
            message=(
                "This step has no chat model of its own, so it runs on whatever "
                "the workflow was started with."
            ),
            suggestion="Attach a chat model to say which one it should use.",
        ))
        return

    report.add(ValidationIssue(
        severity=Severity.ERROR,
        kind="agent.no_model_available",
        node_id=node.id,
        message=(
            "This step needs a chat model and there is none set up, so it "
            "cannot run."
        ),
        suggestion="Add a chat model, then attach it to this step.",
    ))


@node_rule(kinds=("base",), severity=Severity.ERROR)
def structured_output_has_a_shape(node, ctx, report) -> None:
    """Asking for a structured answer without saying what shape it takes.

    Without a schema the engine has nothing to validate against, so a node asked
    for an object hands back JSON as a string and every step downstream of it
    receives text where it expected fields.
    """
    mode = str(getattr(node, "mode", "") or "")
    if mode != "structured":
        return
    if getattr(node, "output_schema", None):
        return
    report.add(ValidationIssue(
        severity=Severity.ERROR,
        kind="agent.structured_without_schema",
        node_id=node.id,
        message=(
            "This step is set to return structured output but no shape is "
            "described, so it would return plain text instead."
        ),
        suggestion="Describe the shape it should return, or set it back to text.",
        detail="mode='structured' requires output_schema",
    ))


@node_rule(kinds=("base",), severity=Severity.ERROR)
def a_shaped_answer_and_tools_do_not_combine(node, ctx, report) -> None:
    """Asking for both a described answer and tools, where only one can happen.

    `run_base_node` is explicit: *"``output_schema`` and tools do not combine —
    the structured path answers in one shot by construction and never reaches the
    tool loop. A node asking for both gets the schema."* So the tools are
    accepted, drawn on the canvas, and never offered to the model.

    That is silent in the worst way — the step returns a well-formed object, so
    nothing downstream complains; it is simply an object the model invented
    without ever looking anything up. An error rather than a warning because both
    settings were deliberate and only one of them is going to happen.

    The way to have both is two steps: one that uses the tools, and one that
    shapes the result.
    """
    if not getattr(node, "output_schema", None):
        return
    if not (getattr(node, "tools", None) or []):
        return
    report.add(ValidationIssue(
        severity=Severity.ERROR,
        kind="agent.shape_disables_tools",
        node_id=node.id,
        message=(
            "This step both describes the answer it returns and has tools "
            "attached. Only the description takes effect — the tools would "
            "never be used."
        ),
        suggestion=(
            "Drop the tools here, or split it: one step that uses the tools, "
            "and one that shapes its result."
        ),
        detail="output_schema wins over tools in run_base_node's one-shot path",
    ))


@node_rule(kinds=("base",), severity=Severity.WARNING)
def enough_tool_rounds_for_the_tools_attached(node, ctx, report) -> None:
    """More tools than the kind has rounds to call them in.

    A `base` node gets **one** round of tool calls — now declared as
    `tool_rounds` on its spec rather than buried as a literal in `run_base_node`,
    which is what makes this rule expressible at all. One round means the model
    may call tools once, sees the results, and answers. It cannot use one tool's
    result to decide the next.

    So two or more attached tools is a step that *may* be a sequence, and a
    sequence is the one thing this kind cannot do. It is not certainly wrong —
    two tools can be independent lookups answered in a single parallel round,
    which works perfectly — so this warns rather than refuses.

    Getting it wrong used to be silent: the round was spent on the first tool,
    the second was refused, and the node returned an empty string reported as
    success. That specific silence is fixed, but the failure still only arrives
    at run time, after the model call is paid for. This says it before.
    """
    tools = list(getattr(node, "tools", None) or [])
    if len(tools) < 2:
        return
    # A shaped answer disables tools entirely; that is the other rule's finding,
    # and reporting both would describe one node as two different mistakes.
    if getattr(node, "output_schema", None):
        return

    rounds = node_kind_spec("base").tool_rounds if node_kind_spec("base") else 1
    if rounds is None or len(tools) <= rounds:
        return

    report.add(ValidationIssue(
        severity=Severity.WARNING,
        kind="agent.tools_exceed_rounds",
        node_id=node.id,
        message=(
            f"This step has {len(tools)} tools attached but gets "
            f"{rounds} round of tool calls, so it cannot use one tool's result "
            f"to choose the next."
        ),
        suggestion=(
            "Fine if the tools are independent lookups. If one feeds the next, "
            "make this a ReAct Agent step, which loops until it is done."
        ),
        detail=f"kind `base` declares tool_rounds={rounds}; {len(tools)} tools attached",
    ))
