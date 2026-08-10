"""ArchitectAgent — the ReAct workflow architect (Phase 4a/4c).

One planner agent with a toolbelt replaces the fixed 8-node pipeline: it reads its
auto-derived self-knowledge (Phase 3), builds the graph incrementally with
``add_node``/``update_node``, checks itself with ``validate_workflow``, authors
missing tools through the sandbox+approval flow, and either registers a valid
workflow or declares the request blocked with a clear reason.

Terminal contract (enforced after the loop ends):
- ``session.registered_path`` set  → return it.
- ``session.blocked_reason`` set   → raise :class:`WorkflowInfeasible`.
- neither                          → ``RuntimeError`` with the agent's last text.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from neurosurfer.llm.base import Provider
from neurosurfer.llm.types import GenerationConfig

from ..build import WorkflowInfeasible
from .session import BuildSession
from .tools import architect_tools

logger = logging.getLogger(__name__)

__all__ = ["ArchitectAgent"]


_SYSTEM_PROMPT = """\
You are the Neurosurfer Architect — an agent that designs and builds runnable
workflow packages from a user's plain-English intent.

# Operating procedure
1. BUILD. `set_workflow` first (name, description, declared inputs), then
   `add_node` one node at a time — read every warning and fix it.
   - **With a plan** (usually — it is in your first message): build it. One node
     per step, keeping the step ids. The tools are already chosen for you; use
     exactly those. If a step's id differs from your node's, pass `plan_step_id`.
     Every planned step must be built or explicitly `drop_plan_step`ped.
   - **Without a plan**: design as you go. One node per distinct capability or
     decision; let the work set the size, not a target count. Map each node to a
     clause the user actually asked for — do NOT add validation, formatting or
     "nice-to-have" steps they did not request.
2. GROUND anything the plan didn't. For each node ask: *does this reach outside
   the model?* Reading a file, fetching a URL, calling an API, touching an inbox,
   sending a message, running a command — none of that can be done by an LLM step,
   however well you word it. Such a node MUST have a tool; `find_capability` finds
   it. See "Capability grounding" below; the validator enforces it.
3. VERIFY structure: `validate_workflow`, fix every reported issue, repeat until
   VALID.
4. PROVE it works: `test_workflow` actually runs your workflow on realistic
   derived inputs and judges the outputs against the user's intent. If it FAILS,
   read the diagnosis and FIX THE SMALLEST THING FIRST: sharpen the failing
   node's `purpose`/`goal`/`expected_result`, correct its `depends_on` wiring, or
   swap a tool. Only add a new node or a control-flow construct if the intent
   truly needs a step that is missing — do NOT escalate a working `router` into a
   `loop` (or bolt on extra nodes) to patch what is really a prompt bug. Never
   leave a failing verification unaddressed.
5. FINISH: declare the result node(s) with `set_outputs` (never `update_node`),
   then `register_workflow` once valid (and tested), and stop with a 2–3
   sentence summary of what the workflow does and its inputs. If the request is
   impossible as described (needs credentials/resources the user didn't provide,
   or an unsafe capability), call `declare_blocked` with a precise reason instead.

When unsure how a construct works, use `describe_capability` — the node kinds,
their required fields and worked shapes are yours already, above. `neurosurfer_docs`
is for project background (configuration, the CLI, providers), NOT for authoring:
those docs are written for people setting neurosurfer up and lag the code, so they
answer build questions with setup guides. `web_search` only for domain research,
never for neurosurfer questions.

# CRITICAL — keep going until done
- Drive the whole build yourself in ONE session. After every tool result,
  IMMEDIATELY make the next tool call. Do NOT stop to narrate progress or ask the
  user anything — the intent is already given.
- You are finished ONLY after `register_workflow` succeeds or `declare_blocked` is
  called. A turn with only text and no tool call before then is a mistake.

# Node-authoring rules
- Every base/react node needs a clear `purpose` (its system prompt), a `goal`, and
  an `expected_result`. Reference declared workflow inputs as `{input_name}`.
- WIRING IS MANDATORY: data flows ONLY along `depends_on` edges. Any node that
  uses another node's result MUST list that node in `depends_on` (e.g. a
  title-writing step that uses the summary MUST depend on the summarise step).
  A multi-node workflow with no `depends_on` edges is wrong — it is a bag of
  parallel nodes, not a pipeline.
- NO ORPHAN NODES: every node's output must be consumed — either declared in
  `set_outputs` or listed in a downstream node's `depends_on`. If nothing uses a
  node's result, delete the node; it is dead weight, not a feature.
- Guards over LLM text: prefer `contains(lower(nodes.x), 'label')` — never exact
  equality against raw model output.
- Router case targets (and `on_error` targets) must list the router/node in their
  `depends_on`.
- Loops always need `max_iterations`; maps always need `over`.

# Capability grounding — THE most common way a build fails
An LLM node only ever sees its prompt. Wording a `base` node as "Read the contents
of {file_path}" does not make it read anything — it makes it INVENT a plausible
file. The plan has usually decided this for you; where it hasn't, ask per node:
**does this reach outside the model?**
- NO (write, summarise, classify, rewrite, judge, reason over text it was given)
  → `base`, no tools.
- YES, one definite call → `tool`. YES, several steps or a choice between them →
  `react` WITH tools. A tool-less `react` is refused by the validator and engine.

**NEVER guess a tool name.** `find_capability("<the capability in plain English>")`
searches the catalog then the MCP registry and says what each option costs. Then
take exactly one of these, never papering over it:
  1. **assign a tool it found** — `update_node` with the right kind and `tools`;
  2. `author_tool` — buildable from plain Python plus a value the user can paste,
     sandbox-tested and human-approved. Reach for this as readily as for an
     install; it is often quicker and always safer;
  3. `install_mcp_server` — when the capability needs a vendor account, a browser
     sign-in or an SDK that a Python file cannot reproduce;
  4. `declare_blocked` — name the server and the exact variables it wants, so the
     user gets a checklist rather than an apology. A clear blocker beats a workflow
     that pretends. But NEVER block because a step needs analysis, summarising,
     classifying or writing — that is what a `base` node is for;
  5. `acknowledge_capability` — only when the check is wrong and the step needs no
     tool (e.g. its data already arrives from an upstream node).

## Acquiring a capability — the loop, in order
A step marked INSTALLABLE is **not** a blocker. Getting it is your job, not the
user's, and there are always TWO ways to get it:

  1. `find_capability` — read the shortlist. Prefer a server whose description
     names the thing you actually need over one that merely shares a word. **A
     shortlist of servers that do not match is the same as an empty one** — being
     offered a flood-data server for "read environment variables" means nothing
     here provides it, not that you should install a flood-data server.
  2. **Decide which route the capability needs.**
     - `author_tool` when it is plain Python plus a value the user can paste: a
       database query, a file conversion, a REST call, a calculation. Three or
       four small tools usually cover a whole integration. This route is also
       safer — your tool is sandbox-tested and approved here, where an install
       runs someone else's code.
     - `install_mcp_server` when it needs a vendor account, a browser sign-in, an
       OAuth grant, or a proprietary SDK — things a Python file cannot reproduce.
  3. **Check what a server exposes before installing it.** Where the shortlist
     shows a server's tools, read them; otherwise install and then
     `list_mcp_tools`. A server whose description sounded right and whose tools
     are wrong is the common failure, and it costs a whole build to find late.
  4. `install_mcp_server` **asks the user** before running anything; you do not
     need permission to try. Credentials already configured here are filled in for
     you, so supply only values the user gave you in this conversation, and never
     invent one.
  5. If an install is declined or needs a value nobody gave you, do not stop at the
     next server — ask whether you could simply **write** the tool.
  6. Acquiring settles the steps that were waiting on it. **Then look again** at
     what is still unresolved and repeat, until every step has something real
     behind it. Only then start writing nodes.

`declare_blocked` is for what neither route reaches: a private system, a
credential nobody can supply, a capability that needs a human. It is not the
answer to "the registry had nothing useful".

Tool node (one direct call, no LLM — the right shape for "read this file"):
  {"id": "load_file", "kind": "tool", "tools": ["read_file"],
   "tool_args": {"path": "{file_path}"}, "writes": "file_text"}

# Control-flow cookbook (add_node `node` argument — copy these shapes)
WHEN to use what: different handling per category → `router`; retry/refine until
good → `loop`; same processing for every item of a list → `map`; a step that only
sometimes applies → a `when:` guard; a risky step needing a fallback → `on_error`.
A plain linear pipeline needs NONE of these — don't force control flow.
If the plan asked for one of these, build that shape — do not flatten it.

Router (the router ITSELF classifies — no separate classify node needed; every
target depends_on the router):
  {"id": "route", "kind": "router",
   "goal": "Route this support ticket by urgency: {ticket}",
   "routes": {"urgent": "escalate", "routine": "archive"},
   "default": "archive"}
(Advanced: deterministic routing on a prior node's output uses
 "cases": [{"when": "contains(lower(nodes.check), 'yes')", "to": "…"}] instead.)

Loop (iterate a nested body until good; state the stop condition in PLAIN ENGLISH
via `until` — an internal judge decides stop/continue each iteration and its
reason reaches the next attempt as {feedback}):
  {"id": "refine", "kind": "loop", "max_iterations": 3,
   "until": "the review approves the draft",
   "body": [{"id": "draft", "kind": "base",
             "goal": "Draft it. Reviewer feedback from last attempt: {feedback}"},
            {"id": "review", "kind": "base", "depends_on": ["draft"],
             "goal": "Review the draft critically."}]}
(`until` is the only stop condition. For a check that code can make — budgets,
 cursors, counts, thresholds — write it as a function in the graph's
 `functions:` file and set "until": "<function_name>": it is free and exact,
 where plain English costs one LLM call per iteration. Plain English is for
 judgements only a reader can make. Either way it must describe what the body
 actually produces: a condition about a different subject stops the loop.)

Map (fan out over a list; the node's output is the ordered per-item results):
  {"id": "per_item", "kind": "map", "over": "inputs.items", "as": "item",
   "body": [{"id": "handle", "kind": "base",
             "purpose": "Process one item: {item}", "goal": "…"}]}

# Your capabilities (auto-derived, version-pinned)
{knowledge}
"""


class ArchitectAgent:
    """Drive one ReAct build: ``await ArchitectAgent(provider).build(intent)``."""

    def __init__(
        self,
        provider: Provider,
        *,
        registry: Any = None,
        staging_root: Path | None = None,
        knowledge: Any = None,
        approve_tool: Any = None,
        approve_mcp: Any = None,
        request_authorization: Any = None,
        request_secrets: Any = None,
        ask_question: Any = None,
        node_event: Any = None,
        notify: Callable[[str], None] | None = None,
        max_turns: int = 40,
        gen_config: GenerationConfig | None = None,
        # V3 Phase 5c. `encouraged` is prompt-only, and R6 is exactly what that
        # buys on a weak model: both bug-report transcripts went validate → ok →
        # register with no test_workflow. The V2 reason for not defaulting to
        # `required` was that a 9B model stalled under it; the Phase 3/4 gates and
        # prescriptive errors exist to fix the stalling, so it is now affordable.
        # Lowering it to `encouraged`/`off` is an explicit caller decision.
        verify: str = "required",
        review: str = "warn",
        plan: bool = True,
        approve_plan: Any = None,
    ) -> None:
        if verify not in {"off", "encouraged", "required"}:
            raise ValueError("verify must be 'off', 'encouraged', or 'required'")
        if review not in {"off", "warn", "required"}:
            raise ValueError("review must be 'off', 'warn', or 'required'")
        self.provider = provider
        self._registry = registry
        self._staging_root = staging_root
        self._knowledge = knowledge
        self._approve_tool = approve_tool
        self._approve_mcp = approve_mcp
        self._request_authorization = request_authorization
        self._request_secrets = request_secrets
        self._ask_question = ask_question
        self._node_event = node_event
        self._notify = notify or (lambda _m: None)
        self._max_turns = max_turns
        self._gen_config = gen_config
        self._verify = verify
        self._review = review
        # Plan first by default. It costs one call and grounds every external step
        # before a node exists, which is worth far more than it costs — but it is
        # switchable, because a caller driving the builder with a scripted provider
        # is testing the builder, not the planner.
        self._plan = plan
        self._approve_plan = approve_plan
        # Set at the start of build(); lets external observers snapshot the graph.
        self.session: Any = None
        # Continuation rounds after a premature text-only stop (see _nudge).
        # Small models narrate mid-build and stall the loop; nudges recover it.
        self._max_nudges = 6

    # ── public ──────────────────────────────────────────────────────────────
    async def build(
        self,
        intent: str,
        *,
        answers: dict[str, str] | None = None,
        refines: str | None = None,
        plan: Any = None,
    ) -> str:
        """Design, validate, and register a workflow for *intent*.

        Returns the registered package path, raises :class:`WorkflowInfeasible`
        when the agent declares the request blocked.

        With *refines*, the session starts from that registered workflow instead
        of an empty canvas, so the intent is read as a change to an existing
        design rather than a fresh build.

        With *plan* (a :class:`WorkflowPlan` or its dict form), the planning call
        is skipped and that plan is built — how a plan reviewed in one request
        gets built by the next one without being re-invented in between.
        """
        from neurosurfer.agents.agentic_loop import AgenticLoop
        from neurosurfer.agents.runtime.permissions import Guardrails
        from neurosurfer.tools.base import AutoApproveIOHandler, ToolPool

        session = self._make_session(intent)
        # Expose the live session so external observers (e.g. the studio's build
        # stream) can snapshot the staged graph as it is assembled.
        self.session = session

        if refines:
            try:
                session.load_from(session.registry.get(refines))
            except Exception as e:  # noqa: BLE001 - a missing target is the caller's error
                raise ValueError(
                    f"Cannot refine '{refines}': {e}"
                ) from e
            self._notify(
                f"loaded '{session.name}' for modification "
                f"({len(session.nodes)} node(s))"
            )

        # Best-effort MCP: connect configured servers so their tools join the
        # workflow-usable catalog (and the knowledge context) before designing.
        import asyncio as _asyncio

        try:
            from neurosurfer.mcp.runtime import ensure_mcp_tools

            statuses = await _asyncio.to_thread(ensure_mcp_tools)
            if statuses:
                ok = [s for s in statuses if getattr(s, "connected", False)]
                self._notify(
                    f"MCP: {len(ok)}/{len(statuses)} server(s) connected, "
                    f"{sum(len(getattr(s, 'tools', []) or []) for s in ok)} tool(s) available"
                )
                session.knowledge.refresh()
        except Exception as e:  # noqa: BLE001 - MCP is optional; never block a build
            self._notify(f"MCP unavailable: {e}")

        # Plan, and ground every external step, BEFORE any node exists. Runs after
        # MCP so a connected server's tools are in the catalog the ladder searches.
        if plan is not None:
            await self._adopt_plan(session, plan)
        elif self._plan:
            await self._plan_build(session, intent, answers)
        if session.blocked_reason:
            raise WorkflowInfeasible(
                session.blocked_reason, session.blocking_requirements()
            )

        # replace(), not format(): the prompt legitimately contains literal {braces}
        # (e.g. the `{input_name}` templating example). Workflow-only tool catalog:
        # a focused list beats a complete one for design work.
        system = _SYSTEM_PROMPT.replace(
            "{knowledge}",
            session.knowledge.render_context(workflow_tools_only=True),
        )

        agent = AgenticLoop(
            provider=self.provider,
            tools=ToolPool(architect_tools(session)),
            system_prompt=system,
            guardrails=Guardrails(
                max_turns=self._max_turns,
                shell_policy="gated",
                network_policy="open",
                write_scope=["**"],
            ),
            io=AutoApproveIOHandler(),
            cwd=session.staging_root,
            gen_config=self._gen_config,
            mode="bypass",
            verbose=False,
            show_environment=False,
            trace_name="ArchitectAgent.build",
        )

        prompt = self._render_prompt(
            intent, answers, session.plan,
            plan_answers=getattr(session, "plan_answers", ""),
        )
        result = await agent.run_collect(prompt)

        # Small models sometimes end a turn with pure narration ("let me build…"),
        # which the loop treats as a final answer. Nudge the SAME conversation
        # (history persists on the agent) back to work until a terminal state.
        rounds = 0
        while (
            session.registered_path is None
            and session.blocked_reason is None
            and rounds < self._max_nudges
        ):
            rounds += 1
            # A nudge is pointless without turn budget: if the loop already burned
            # max_turns (flailing), run_collect would return instantly. Grant a
            # small allowance per nudge so the recovery attempt is real.
            agent.guardrails.max_turns = max(
                agent.guardrails.max_turns, agent.turns + 4
            )
            self._notify(f"agent stopped early — nudging to continue ({rounds})")
            result = await agent.run_collect(self._nudge(session))

        final_text = (getattr(result, "final_text", "") or "").strip()

        if session.registered_path:
            return session.registered_path
        if session.blocked_reason:
            raise WorkflowInfeasible(
                session.blocked_reason, session.blocking_requirements()
            )
        # Non-convergence. If a workflow was built but couldn't pass required
        # verification, say so with the last report — that's the actionable truth,
        # not "it did nothing".
        if session.verification_mode == "required" and session.last_verification and not session.last_verification[0]:
            raise RuntimeError(
                "Built a workflow but it did not pass verification, and the agent "
                "could not fix it within the step budget. Last verification:\n"
                + session.last_verification[1]
            )
        raise RuntimeError(
            "The architect agent finished without registering a workflow or "
            "declaring the request blocked."
            + (f" Its last message: {final_text}" if final_text else "")
        )

    async def _adopt_plan(self, session: BuildSession, plan: Any) -> None:
        """Build a plan somebody else already produced (and possibly edited).

        Always re-resolved rather than trusted: the caller may have changed a
        step's capability, and a resolution that came back over HTTP describes
        whatever the catalog looked like when it was made, not now.
        """
        import asyncio as _aio

        from ..plan import WorkflowPlan
        from ..planner import resolve_plan

        if isinstance(plan, dict):
            try:
                plan = WorkflowPlan.model_validate(plan)
            except Exception as e:  # noqa: BLE001 - a bad plan is the caller's error
                raise ValueError(f"Not a usable plan: {e}") from e
        if not isinstance(plan, WorkflowPlan) or not plan.steps:
            raise ValueError("A plan must have at least one step.")

        plan = plan.normalised()
        plan = await _aio.to_thread(resolve_plan, plan, notify=self._notify)
        session.plan = plan
        self._notify(plan.render())

    async def _plan_build(
        self, session: BuildSession, intent: str, answers: dict[str, str] | None
    ) -> None:
        """Plan the workflow, resolve its external steps, and gate on the result.

        Sets ``session.blocked_reason`` when a step needs something nothing can
        provide — the cheapest possible place to discover a build cannot happen,
        since not a single node has been written or paid for yet.
        """
        from ..planner import plan_and_resolve

        try:
            plan = await plan_and_resolve(
                self.provider, intent,
                answers=answers,
                existing_graph=session.to_yaml() if session.nodes else None,
                notify=self._notify,
                gen_config=self._gen_config,
            )
        except Exception as e:  # noqa: BLE001 - planning is an advantage, not a gate
            self._notify(f"planning failed ({type(e).__name__}) — building without a plan")
            return
        if not plan.steps:
            return

        session.plan = plan
        self._notify(plan.render())
        await self._ask_open_questions(session, plan)

        # Review, when there is somebody to review it. The plan is the cheapest
        # thing to change — a wrong step costs one edit here and a whole rebuild
        # once it is nodes — so the gate covers every plan, not only broken ones.
        if self._approve_plan is not None:
            decided = await self._review_plan(plan)
            if decided is None:
                session.blocked_reason = (
                    "The plan was not accepted, so nothing was built. Describe the "
                    "change you want and I'll plan it again."
                )
                self._notify("plan rejected — no nodes were built")
                return
            if decided is not plan:
                import asyncio as _aio

                from ..planner import resolve_plan

                self._notify("plan edited — re-resolving capabilities")
                decided = await _aio.to_thread(resolve_plan, decided, notify=self._notify)
                # Set before narrating: observers snapshot the session on every
                # notify, so the render and the snapshot must describe the same
                # plan or the record keeps the version the user replaced.
                session.plan = decided
                self._notify(decided.render())
            session.plan = plan = decided
            # Accepting a plan with unresolved steps is an explicit override; the
            # reviewer was shown exactly which steps and what they'd need.
            return

        # Only what nothing here can provide. A step whose capability sits on an
        # MCP server nobody installed is *acquirable*, and the agent has a tool for
        # that — blocking it at the plan declared a build infeasible with the
        # server it needed listed in the same refusal.
        acquirable = plan.acquirable_steps
        if acquirable:
            self._notify(
                f"{len(acquirable)} step(s) need something installed first — "
                f"the build will ask before installing anything"
            )

        impossible = plan.impossible_steps
        if impossible:
            lines = [
                "This workflow needs capabilities that are not available here:",
                "",
            ]
            for step in impossible:
                need = step.needed_capability or step.intent
                lines.append(f"  • {step.id} — {need}")
                for server in (step.resolution or {}).get("servers", [])[:2]:
                    # What is *missing*, named as the user would supply it. A
                    # credential already on file is not a reason to block, and a
                    # templated one is asked for by its placeholder — listing
                    # `Authorization` asks for a value nobody has under that name.
                    creds = [
                        c.get("ask_for") or c["name"]
                        for c in server.get("credentials", [])
                        if c.get("required") and not c.get("satisfied_from")
                    ]
                    lines.append(
                        f"      available via `{server['name']}`"
                        + (f", which needs {', '.join(creds)}" if creds else "")
                    )
            lines += [
                "",
                "Install a server for these from the list above, or set any value "
                "it needs in Settings → Secrets, then run this again — or ask for a "
                "workflow that does not need them.",
            ]
            session.blocked_reason = "\n".join(lines)
            self._notify("blocked at the plan — no nodes were built")

    async def _ask_open_questions(self, session: Any, plan: Any) -> None:
        """Put the planner's `open_questions` to the user, and keep the answers.

        They used to be rendered at the foot of the plan card and never asked —
        so a decision only the user could make was shown to them as a footnote
        and then guessed at. The planner's own rule is now that a question
        belongs there ONLY when the design changes on the answer and nobody else
        can give it; if that holds, it is worth one round trip.

        Answers go to the builder as context rather than back through planning:
        the plan is already resolved by this point, and re-planning would throw
        away the capability work to change a sentence.
        """
        questions = [q for q in (getattr(plan, "open_questions", None) or []) if q.strip()]
        if not questions or session.ask_question is None:
            return
        answered: list[str] = []
        for question in questions[:3]:  # a plan with ten is a planning problem
            try:
                answer = session.ask_question(question, [])
                if hasattr(answer, "__await__"):
                    answer = await answer
            except Exception:  # noqa: BLE001 - an unanswered question never blocks
                return
            text = str(answer or "").strip()
            # The gate's timeout default. Silence means "you decide", which is
            # what the build would have done anyway.
            if text and text.lower() not in {"no preference", "skip"}:
                answered.append(f"- {question}\n  → {text}")
        if answered:
            session.plan_answers = "\n".join(answered)
            self._notify(f"{len(answered)} open question(s) answered")

    async def _review_plan(self, plan: Any) -> Any:
        """Show the plan to whoever is reviewing. Returns what to build, or None.

        The reviewer may hand back a *different* plan — the point of reviewing a
        design is being able to change it, not only to veto it. A returned plan is
        re-resolved by the caller, since an edited step may name a capability
        nothing has looked up yet.
        """
        from ..plan import WorkflowPlan

        result = self._approve_plan(plan)
        if hasattr(result, "__await__"):
            result = await result
        if result is None or result is False:
            return None
        if isinstance(result, WorkflowPlan):
            return result
        if isinstance(result, dict):
            try:
                return WorkflowPlan.model_validate(result).normalised()
            except Exception as e:  # noqa: BLE001 - a malformed edit is not a build failure
                self._notify(f"edited plan was not usable ({e}) — building the original")
                return plan
        # Anything else truthy means "yes, as planned".
        return plan

    def _nudge(self, session: BuildSession) -> str:
        """Status-grounded, prescriptive continuation after a premature text stop.

        Names the exact next tool to call, so a weaker model has one obvious move
        instead of an open-ended "continue".
        """
        status: list[str] = []
        if not session.name:
            status.append("workflow meta NOT set")
        else:
            status.append(f"name={session.name!r}, nodes={session.node_ids()}, "
                          f"outputs={session.outputs}")

        # Decide the single next action from the current state.
        if not session.name:
            nxt = "Call set_workflow (name, description, inputs)."
        elif not session.nodes:
            nxt = "Call add_node to add the first node."
        else:
            ok, _, report = session._validate()
            ungrounded = session.unresolved_capabilities(report) if ok else []
            if not ok:
                nxt = "Call validate_workflow, then fix each reported issue with add_node/update_node."
            elif ungrounded:
                # register would refuse on these, so sending it there would just
                # burn a turn and land back here with the same state.
                first = ungrounded[0]
                status.append(
                    f"{len(ungrounded)} node(s) need a tool they don't have: "
                    + ", ".join(str(i.node_id) for i in ungrounded)
                )
                nxt = (
                    f"Node '{first.node_id}' {first.message}. "
                    f"Call find_capability('{first.subject or 'the capability it needs'}') "
                    "to see what provides it, then update_node to attach the tool — "
                    "or declare_blocked if nothing can."
                )
            elif session.verification_mode == "required" and (
                session.last_verification is None or not session.last_verification[0]
            ):
                if session.last_verification is None:
                    status.append("not yet tested")
                    nxt = "Call test_workflow to prove it works."
                else:
                    status.append("last verification FAILED")
                    nxt = "Fix the design per the last diagnosis, then call test_workflow again."
            else:
                nxt = "Call register_workflow to finish."

        return (
            "You stopped mid-build without a tool call — nothing is registered or "
            "blocked yet. Reply with ONLY the next tool call, no prose.\n"
            f"Status: {'; '.join(status)}.\n"
            f"NEXT ACTION: {nxt}\n"
            "If the workflow genuinely cannot be built, call declare_blocked instead."
        )

    # ── internals ───────────────────────────────────────────────────────────
    def _make_session(self, intent: str) -> BuildSession:
        from neurosurfer.architect.knowledge import KnowledgeBase
        from neurosurfer.config.paths import projects_dir
        from neurosurfer.graph.workflow.registry import WorkflowRegistry

        return BuildSession(
            intent=intent,
            staging_root=self._staging_root or projects_dir(),
            registry=self._registry or WorkflowRegistry(),
            knowledge=self._knowledge or KnowledgeBase(),
            provider=self.provider,
            approve_tool=self._approve_tool,
            approve_mcp=self._approve_mcp,
            request_authorization=self._request_authorization,
            request_secrets=self._request_secrets,
            ask_question=self._ask_question,
            node_event=self._node_event,
            notify=self._notify,
            verification_mode=self._verify,
            review_mode=self._review,
        )

    @staticmethod
    def _render_prompt(
        intent: str, answers: dict[str, str] | None, plan: Any = None,
        *, plan_answers: str = "",
    ) -> str:
        parts = [f"Build a workflow for this request:\n\n{intent.strip()}"]
        if answers:
            qa = "\n".join(f"- {q}: {a}" for q, a in answers.items())
            parts.append(f"Clarifying answers already collected:\n{qa}")

        if plan is not None and getattr(plan, "steps", None):
            # Naming the tool per step is the single biggest weak-model lever in
            # this design: the builder is TOLD which tool to attach instead of
            # being asked to remember one, which is where it used to invent names
            # or quietly leave a node toolless.
            parts.append(
                "A plan has already been made and every step that reaches outside "
                "the model has been resolved against the real catalog. BUILD THIS "
                "PLAN — one node per step, keeping the ids:\n\n" + plan.render()
            )
            # What the user said when the plan's open questions were put to them.
            # These are decisions only they could make, so they outrank anything
            # the plan assumed.
            if plan_answers:
                parts.append(
                    "The user answered the plan's open questions. These are "
                    "decisions, not suggestions — build to them:\n" + plan_answers
                )
            assignments = [
                f"- node `{s.id}`: use tools {s.tool_names} "
                f"(kind `{s.kind_hint}`) — it needs to {s.needed_capability}"
                for s in plan.external_steps if s.tool_names
            ]
            if assignments:
                parts.append(
                    "Tool assignments already decided for you — use exactly these, "
                    "do not substitute or invent:\n" + "\n".join(assignments)
                )
            parts.append(
                "Steps NOT marked as needing an external capability are plain `base` "
                "nodes with no tools. Do not add tools to them and do not block on "
                "them. If the plan is wrong about something, fix it as you build and "
                "say so — but start from it."
            )
        parts.append(
            "Design it, add the nodes, validate until clean, then register. "
            "If it cannot be built as described, declare it blocked with the reason."
        )
        return "\n\n".join(parts)
