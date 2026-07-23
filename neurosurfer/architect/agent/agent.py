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
1. UNDERSTAND the intent (clarifying answers, if any, are included). Use
   `neurosurfer_docs` / `describe_capability` when unsure how a construct works;
   use `web_search` only for domain research, not for neurosurfer questions.
2. DESIGN the graph: decompose into the FEWEST focused nodes that satisfy the
   intent (usually 2–4). Map each node to a clause the user actually asked for —
   do NOT add validation, formatting, or "nice-to-have" steps they did not
   request. Use the right kind per node; wire `depends_on` edges to form a DAG.
   Use control flow when the intent needs it: `router` for branching, `loop` for
   iterate-until, `map` for per-item fan-out, `when:` guards for conditional
   steps, `on_error:` for fallbacks. Don't force control flow onto a linear task.
3. BUILD incrementally: `set_workflow` first (name, description, declared inputs),
   then `add_node` one node at a time — read every warning and fix it.
4. Assign ONLY tools that exist in the catalog (workflow-usable, marked ✓ in your
   capabilities). If a required capability has NO tool, `author_tool` it (it will
   be sandbox-tested and needs human approval) — or `declare_blocked` if it cannot
   be built safely.
5. VERIFY structure: `validate_workflow`, fix every reported issue, repeat until
   VALID.
6. PROVE it works: `test_workflow` actually runs your workflow on realistic
   derived inputs and judges the outputs against the user's intent. If it FAILS,
   read the diagnosis and FIX THE SMALLEST THING FIRST: sharpen the failing
   node's `purpose`/`goal`/`expected_result`, correct its `depends_on` wiring, or
   swap a tool. Only add a new node or a control-flow construct if the intent
   truly needs a step that is missing — do NOT escalate a working `router` into a
   `loop` (or bolt on extra nodes) to patch what is really a prompt bug. Never
   leave a failing verification unaddressed.
7. FINISH: declare the result node(s) with `set_outputs` (never `update_node`),
   then `register_workflow` once valid (and tested), and stop with a 2–3
   sentence summary of what the workflow does and its inputs. If the request is
   impossible as described (needs credentials/resources the user didn't provide,
   or an unsafe capability), call `declare_blocked` with a precise reason instead.

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
- `react` nodes MUST list their `tools`; `base` nodes must not have tools.
- Guards over LLM text: prefer `contains(lower(nodes.x), 'label')` — never exact
  equality against raw model output.
- Router case targets (and `on_error` targets) must list the router/node in their
  `depends_on`.
- Loops always need `max_iterations`; maps always need `over`.

# Control-flow cookbook (add_node `node` argument — copy these shapes)
WHEN to use what: different handling per category → `router`; retry/refine until
good → `loop`; same processing for every item of a list → `map`; a step that only
sometimes applies → a `when:` guard; a risky step needing a fallback → `on_error`.
A plain linear pipeline needs NONE of these — don't force control flow.

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
(Deterministic loops — budgets, cursors, index checks — use
 "break_when": "<expression>" instead of `until`.)

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
        notify: Callable[[str], None] | None = None,
        max_turns: int = 40,
        gen_config: GenerationConfig | None = None,
        verify: str = "encouraged",
    ) -> None:
        if verify not in {"off", "encouraged", "required"}:
            raise ValueError("verify must be 'off', 'encouraged', or 'required'")
        self.provider = provider
        self._registry = registry
        self._staging_root = staging_root
        self._knowledge = knowledge
        self._approve_tool = approve_tool
        self._notify = notify or (lambda _m: None)
        self._max_turns = max_turns
        self._gen_config = gen_config
        self._verify = verify
        # Continuation rounds after a premature text-only stop (see _nudge).
        # Small models narrate mid-build and stall the loop; nudges recover it.
        self._max_nudges = 6

    # ── public ──────────────────────────────────────────────────────────────
    async def build(self, intent: str, *, answers: dict[str, str] | None = None) -> str:
        """Design, validate, and register a workflow for *intent*.

        Returns the registered package path, raises :class:`WorkflowInfeasible`
        when the agent declares the request blocked.
        """
        from neurosurfer.agents.agentic_loop import AgenticLoop
        from neurosurfer.agents.runtime.permissions import Guardrails
        from neurosurfer.tools.base import AutoApproveIOHandler, ToolPool

        session = self._make_session(intent)

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

        prompt = self._render_prompt(intent, answers)
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
            raise WorkflowInfeasible(session.blocked_reason)
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
            ok, _ = session.validate()
            if not ok:
                nxt = "Call validate_workflow, then fix each reported issue with add_node/update_node."
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
            notify=self._notify,
            verification_mode=self._verify,
        )

    @staticmethod
    def _render_prompt(intent: str, answers: dict[str, str] | None) -> str:
        parts = [f"Build a workflow for this request:\n\n{intent.strip()}"]
        if answers:
            qa = "\n".join(f"- {q}: {a}" for q, a in answers.items())
            parts.append(f"Clarifying answers already collected:\n{qa}")
        parts.append(
            "Design it, add the nodes, validate until clean, then register. "
            "If it cannot be built as described, declare it blocked with the reason."
        )
        return "\n\n".join(parts)
