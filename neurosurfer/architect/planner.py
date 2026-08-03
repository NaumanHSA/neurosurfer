"""The Planner — one structured call that decides the shape of the workflow.

Deliberately **not** a ReAct agent. The builder is one, and holding requirements,
design, capability sourcing, wiring, validation and verification in a single stream
is what a small model drops things from. Planning is a single question with a
single schema-shaped answer, which is the form a weak model is most reliable at.

Two stages, and the second is the one that matters:

1. :class:`PlannerAgent` turns an intent into a :class:`WorkflowPlan` — steps, and
   for each, *does this reach outside the model?*
2. :func:`resolve_plan` takes every step that said yes to the Phase 2 ladder,
   **in code**, and writes the answer back onto the step.

By the time the builder sees the plan, a step needing a file has `read_file`
attached to it. The builder is told which tool to use rather than asked to
remember one — which is the single biggest difference between this working on
gpt-4o-mini and not.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from neurosurfer.llm.types import GenerationConfig, Message

from .plan import KIND_HINTS, WorkflowPlan

logger = logging.getLogger(__name__)

__all__ = ["PlannerAgent", "resolve_plan", "plan_and_resolve"]


_SYSTEM = """\
You plan neurosurfer workflows. You do NOT build them — you decide the shape, and
another agent writes the nodes.

Output STRICT JSON only, no prose, no code fences:
{"name": "<snake_case_workflow_name>",
 "description": "<one sentence>",
 "inputs": [{"name": "<snake_case>", "type": "string|integer|float|boolean|object|array",
             "required": true, "description": "<what it is>"}],
 "steps": [{"id": "<snake_case>",
            "intent": "<what this step does, one sentence>",
            "kind_hint": "base|tool|react|router|loop|map",
            "depends_on": ["<earlier step id>"],
            "produces": "<what it hands downstream>",
            "is_external": true|false,
            "needed_capability": "<plain English, ONLY when is_external>",
            "secrets": ["<STORED_VALUE_NAME>"]}],
 "outputs": ["<step id whose output is the result>"],
 "open_questions": ["<anything genuinely ambiguous in the request>"]}

# THE QUESTION THAT MATTERS: is_external
For EVERY step ask: can a language model do this with only its prompt?
- NO — it must read a file, list a directory, call an API or URL, touch an email
  inbox, send a message or SMS, query a database, run a command, or reach ANY
  system outside itself → `is_external: true`, and write `needed_capability` in
  plain English ("read a file from disk", "read an email inbox", "send a text
  message"). Do NOT name a tool; you do not know what is installed.
- YES — writing, summarising, classifying, analysing, rewriting, drafting,
  judging, extracting from text it was given → `is_external: false`.

A step that says "read the file" and is not marked external produces a workflow
that INVENTS the file's contents. This is the single most important field here.

# CREDENTIALS ARE NEVER INPUTS
A database password, connection string, API key or token is NOT an input. Declare
it in that step's `secrets` list, by name: `"secrets": ["SALES_DB_URL"]`.

`inputs` are in the interpolation scope, which means anything declared there is
readable from EVERY node's prompt — and from there the model's context, the trace,
and anything the trace is exported to. A step named `secrets` reaches the tool call
and nothing else. So:

    WRONG:  inputs: [{"name": "db_connection_config", "type": "object"}]
    RIGHT:  step "load_sales_data": {"secrets": ["SALES_DB_URL"]}

Inputs are the things a person would happily type in front of someone else: a
topic, a date range, a file to process, a row limit. If you would not put it on a
screen, it is a secret.

**And getting hold of a credential is NEVER a step.** Do not plan "load the
credentials", "read the connection details from the environment", or "validate
the config" — there is nothing to run. The engine substitutes a declared secret
into the tool call itself, so the step that USES the database simply names what it
needs and connects. A step whose whole job is fetching a credential resolves to a
capability nothing provides, and blocks a build that was otherwise fine.

    WRONG:  step "load_db_credentials" → needs "read environment variables"
            step "query_sales"         → depends_on load_db_credentials
    RIGHT:  step "query_sales"         → secrets: ["SALES_DB_URL"]

# Choosing kind_hint
- `base`   — one LLM call over text it was given. The default, and the ONLY
             choice for a step with `is_external: false`.
- `tool`   — ONE external call whose arguments are ALL known in advance: a
             constant, or something an earlier step produced. "Read the file at
             the path the user gave" is a `tool` step.
- `react`  — external work where a model must WORK OUT what to send: writing the
             SQL for a query, the phrase for a search, the body for a request.
             Also anything needing several calls or a choice between them.
`tool` and `react` exist to CALL tools, so both require `is_external: true`.

The `tool`/`react` line is about the ARGUMENTS, not the number of calls. Running
one SQL query is a single call, but its query text has to be written by a model —
so it is `react`. A `tool` step has no model in it at all, and nothing to compose
what it sends.

**A `react` step can use a credential.** Declare it in `secrets` exactly as a
`tool` step would: the builder binds it to the right parameter and the model
composes the rest of the call without ever seeing the value. So "query the sales
database" is one `react` step with `secrets: ["SALES_DB_URL"]` — not a `tool`
step for connecting followed by a `react` step for querying, which cannot pass a
connection from one to the other.

# Control flow — decide it here, the builder will not add it later
Read the request for these shapes and use them when they are there:
- **different handling per category** ("urgent vs routine", "by language",
  "depending on the type") → a `router` step, with one step per branch
  listing the router in its `depends_on`.
  The router IS the classifier: it decides AND routes in one step. Do NOT plan a
  separate "classify" step feeding a router — that is two steps doing one job.
- **repeat until it is good enough** ("refine until", "at most 3 attempts")
  → a `loop` step.
- **the same processing for every item of a list** ("for each city", "per file")
  → a `map` step.
- **a step that only sometimes applies** → still its own step; say so in `intent`.
A plain linear request needs NONE of these. Do not force branching onto a task
that has one path.

# Decomposition
One step per distinct capability or decision. A step that needs a different tool,
or makes a different decision, is a different step. Do not pad the plan with
validation or formatting steps the user did not ask for, and do not collapse two
genuinely different capabilities into one step.

Every step except the first should list `depends_on` — data flows only along those
edges. `inputs` are what the USER supplies when running the workflow, not what a
step produces.

# open_questions — only what a person must answer
A question belongs here ONLY if the design changes depending on the answer and
nobody but the user can give it. Two things do not belong:

- **Anything the workflow can find out at run time.** "What are the table and
  column names?" is not a question — it is the first thing the step does, with
  the schema tools it already has. Plan the discovery instead of asking.
- **Detail that has an obvious default.** "Should the report include a title?"
  Pick one and get on with it.

Everything you put here is shown to the user as something they need to resolve.
A list of things they cannot answer, or should not have to, is noise on top of a
design that was ready to build.
"""

_RETRY = (
    "That was not valid JSON matching the schema. Reply with ONLY the JSON object, "
    "starting with { and ending with }. No explanation, no code fence."
)


class PlannerAgent:
    """``await PlannerAgent(provider).plan(intent)`` → a :class:`WorkflowPlan`."""

    def __init__(
        self,
        provider: Any,
        *,
        notify: Callable[[str], None] | None = None,
        gen_config: GenerationConfig | None = None,
    ) -> None:
        self.provider = provider
        self._notify = notify or (lambda _m: None)
        self._gen_config = gen_config

    async def plan(
        self,
        intent: str,
        *,
        answers: dict[str, str] | None = None,
        existing_graph: str | None = None,
    ) -> WorkflowPlan:
        """Plan *intent*. Never raises — a plan we can't parse degrades to one step.

        Degrading rather than failing is deliberate: an unparseable plan should
        cost the build its planning advantage, not its life. The builder can still
        work the way it did before this existed.
        """
        prompt = self._render_prompt(intent, answers, existing_graph)
        self._notify("planning the workflow")

        text = await self._ask(prompt)
        plan = self._parse(text)
        if plan is None:
            # One retry with the complaint restated — small models usually comply
            # the second time, and a second call is far cheaper than a bad build.
            self._notify("plan was not valid JSON — retrying once")
            text = await self._ask(prompt, retry_of=text)
            plan = self._parse(text)

        if plan is None or not plan.steps:
            self._notify("planning failed — the builder will design as it goes")
            return WorkflowPlan(
                name="", description=intent[:200], steps=[], outputs=[],
            )

        plan = plan.normalised()
        self._notify(
            f"planned {len(plan.steps)} step(s), "
            f"{len(plan.external_steps)} needing an external capability"
        )
        return plan

    # ── internals ───────────────────────────────────────────────────────────
    async def _ask(self, prompt: str, *, retry_of: str | None = None) -> str:
        messages = [Message.user_text(prompt)]
        if retry_of is not None:
            messages += [Message.assistant_text(retry_of or "(empty)"),
                         Message.user_text(_RETRY)]
        response = await self.provider.complete(
            messages=messages,
            system=_SYSTEM,
            tools=[],
            config=self._gen_config or GenerationConfig(stream=False),
        )
        return response.text() or ""

    @staticmethod
    def _parse(text: str) -> WorkflowPlan | None:
        from .agent.jsonio import parse_json

        data = parse_json(text, want="steps")
        if not isinstance(data, dict):
            return None
        try:
            return WorkflowPlan.model_validate(data)
        except Exception as e:  # noqa: BLE001 - a malformed plan is a retry, not a crash
            logger.debug("plan did not validate: %s", e)
            return None

    @staticmethod
    def _render_prompt(
        intent: str, answers: dict[str, str] | None, existing_graph: str | None
    ) -> str:
        parts = [f"Plan a workflow for this request:\n\n{intent.strip()}"]
        if answers:
            qa = "\n".join(f"- {q}: {a}" for q, a in answers.items())
            parts.append(f"Clarifying answers already collected:\n{qa}")
        if existing_graph:
            parts.append(
                "This MODIFIES an existing workflow. Plan the workflow as it should "
                "be AFTER the change — keep the steps that still apply, with their "
                f"ids:\n\n{existing_graph}"
            )
        parts.append(
            f"kind_hint must be one of {', '.join(KIND_HINTS)}. "
            "Mark every step that reaches outside the model as is_external."
        )
        return "\n\n".join(parts)


def resolve_plan(
    plan: WorkflowPlan, *, notify: Callable[[str], None] | None = None
) -> WorkflowPlan:
    """Run every external step through the Phase 2 ladder, in code.

    Synchronous and blocking (the registry leg is HTTP); callers on an event loop
    should hand this to a thread. Never raises: a step whose resolution fails is
    left unresolved, which the plan gate then reports.
    """
    say = notify or (lambda _m: None)
    from .capability import resolve_capability

    external = plan.external_steps
    if not external:
        return plan

    say(f"resolving {len(external)} external capabilit"
        f"{'y' if len(external) == 1 else 'ies'}")
    for step in external:
        need = step.needed_capability.strip() or step.intent
        try:
            resolution = resolve_capability(need)
        except Exception as e:  # noqa: BLE001 - a failed lookup is not a failed build
            logger.debug("resolution failed for %r: %s", need, e)
            step.resolution = {"need": need, "status": "none", "error": str(e)}
            continue
        step.resolution = resolution.to_dict()
        # A capability the ladder recognises as composition was never external;
        # correcting it here keeps the plan honest and spares the builder a step
        # it would have had to argue its way out of.
        if resolution.status == "not_external":
            step.is_external = False
            step.needed_capability = ""
        say(f"  {step.id}: {need} → {resolution.status}")
    return plan


async def plan_and_resolve(
    provider: Any,
    intent: str,
    *,
    answers: dict[str, str] | None = None,
    existing_graph: str | None = None,
    notify: Callable[[str], None] | None = None,
    gen_config: GenerationConfig | None = None,
) -> WorkflowPlan:
    """Plan, then ground every external step. The whole Phase 3 pipeline."""
    import asyncio

    planner = PlannerAgent(provider, notify=notify, gen_config=gen_config)
    plan = await planner.plan(intent, answers=answers, existing_graph=existing_graph)
    if not plan.steps:
        return plan
    return await asyncio.to_thread(resolve_plan, plan, notify=notify)
