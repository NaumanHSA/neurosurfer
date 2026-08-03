"""ArchitectManager (S5) — runs Architect builds in the background and records
their step log + staged-graph snapshots for the studio to stream.

Each build runs the ReAct ``ArchitectAgent`` in a worker thread. The agent's
``notify`` callback appends a ``log`` event and snapshots the live staged graph
(``agent.session.graph_dict()``) so the studio canvas can animate the build.
The terminal outcome is a registered workflow (``succeeded``), a
``WorkflowInfeasible`` (``blocked``), an error (``failed``), or ``cancelled``.

A build can also stop and ask. With ``clarify=True`` it first runs
:class:`ArchitectConversation` to gather requirements, and authored tools can
require approval before registration. Both park the worker thread on an
:class:`InteractionGate` until an HTTP request answers — see ``interaction.py``.

``agent_factory`` is injectable so tests can drive the plumbing without an LLM.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .interaction import Cancelled, InteractionGate
from .store import BuildRecord

logger = logging.getLogger("neurosurfer.server.architect")


def _clarify_mode(value: bool | str | None) -> str:
    """Normalise the clarify setting to `auto` | `always` | `off`.

    `auto` is the default and asks only where the answer would change the design.
    Booleans are still accepted because that is what this took before, and an API
    caller sending `true`/`false` means exactly what it always did.

    An unrecognised value falls back to `auto` rather than raising: this arrives
    from a request body, and a typo should get the sensible behaviour, not a 500.
    """
    if value is True:
        return "always"
    if value is False:
        return "off"
    mode = str(value or "auto").strip().lower()
    return mode if mode in {"auto", "always", "off"} else "auto"


class ArchitectManager:
    def __init__(
        self,
        provider: Any,
        *,
        registry: Any = None,
        staging_root: Path | None = None,
        agent_factory: Callable[..., Any] | None = None,
        # Tracks `ArchitectAgent`'s own default (V3 Phase 5c). Left at
        # "encouraged" this would quietly undo it for every build the gateway
        # runs, which is every build the studio makes.
        verify_default: str = "required",
    ) -> None:
        self.provider = provider
        if registry is None:
            from neurosurfer.graph.workflow.registry import WorkflowRegistry

            registry = WorkflowRegistry()
        self.registry = registry
        self.staging_root = staging_root
        self._agent_factory = agent_factory
        self._verify_default = verify_default
        self._builds: dict[str, BuildRecord] = {}
        self._gates: dict[str, InteractionGate] = {}
        self._lock = threading.Lock()

    # ── read ──────────────────────────────────────────────────────────────
    def get(self, build_id: str) -> BuildRecord | None:
        return self._builds.get(build_id)

    def list(self) -> list[BuildRecord]:
        return sorted(self._builds.values(), key=lambda r: r.created_at, reverse=True)

    # ── interaction ───────────────────────────────────────────────────────
    def respond(self, build_id: str, interaction_id: str, value: Any) -> bool:
        """Answer whatever the build is parked on. False if it isn't waiting on that."""
        gate = self._gates.get(build_id)
        return gate.respond(interaction_id, value) if gate is not None else False

    def cancel(self, build_id: str) -> bool:
        """Ask a running build to stop. False if it is already terminal.

        Cooperative: the worker thread notices at its next ``notify`` (the agent
        narrates constantly, so that is soon) and anything parked on the gate
        wakes immediately. A thread cannot be killed outright, and interrupting
        an LLM call mid-flight would leak the staged project directory.
        """
        rec = self._builds.get(build_id)
        if rec is None or rec.status != "running":
            return False
        gate = self._gates.get(build_id)
        if gate is not None:
            gate.cancel()
        rec.add_event("log", message="cancelling…")
        return True

    # ── lifecycle ─────────────────────────────────────────────────────────
    def start(
        self,
        intent: str,
        *,
        verify: str | None = None,
        clarify: bool | str = "auto",
        approve_tools: bool = False,
        review_plan: bool = False,
        refines: str | None = None,
        plan: dict[str, Any] | None = None,
        discovery: Any = None,
        credentials: dict[str, str] | None = None,
        save_secret: Any = None,
        history: list[dict[str, str]] | None = None,
    ) -> BuildRecord:
        """Start a build.

        `discovery` is the MCP engine this account chose and `credentials` the
        values it already holds. Both are passed in already resolved because they
        must be read on the *calling* thread: the build runs on a raw
        `threading.Thread`, which starts with an empty context, so anything ambient
        set during the request is gone by the time the agent searches.

        `review_plan` shows the plan for approval (and editing) before any node is
        written — its own flag rather than a rider on `approve_tools`, because
        reviewing a design and vetting generated code are different decisions.
        `plan` builds a plan the caller already has, skipping the planning call.

        `clarify` is `auto` (ask only where the answer would change the design),
        `always` (2-5 questions whatever the request) or `off` (never ask).
        **Auto is the default**, so an ordinary build now costs one extra model
        call before designing — it buys a sharpened intent even when it asks
        nothing, which is the common case. `approve_tools` gates each authored
        tool on a human; `refines` seeds the build with an existing workflow so
        the intent reads as a change to it rather than a fresh start.
        """
        rec = BuildRecord(intent=intent, refines=refines)
        gate = InteractionGate(
            on_pending=lambda i: self._pending(rec, i),
            on_resolved=lambda i, v, answered: self._resolved(rec, i, v, answered),
        )
        with self._lock:
            self._builds[rec.id] = rec
            self._gates[rec.id] = gate
        verify_mode = verify or self._verify_default
        clarify_mode = _clarify_mode(clarify)

        def _snapshot(agent: Any) -> None:
            try:
                sess = getattr(agent, "session", None)
                g = sess.graph_dict() if sess is not None else None
            except Exception:  # noqa: BLE001 - snapshot is best-effort
                g = None
            if g and g.get("nodes") and g != rec.graph:
                rec.graph = g
                rec.add_event("graph", graph=g)

        def _work() -> None:
            from neurosurfer.mcp.credentials import use_credentials
            from neurosurfer.mcp.sources import use_source

            with use_source(discovery), use_credentials(credentials):
                _build()

        def _build() -> None:
            holder: dict[str, Any] = {}
            rec.add_event("build", status="running")

            def notify(msg: str) -> None:
                # The cancel check rides on notify because the agent narrates
                # every step, making this the densest cooperative checkpoint we
                # have without threading a flag through the whole agent.
                gate.raise_if_cancelled()
                rec.add_event("log", message=msg)
                self._capture_verification(rec, holder.get("agent"), msg)
                self._capture_plan(rec, holder.get("agent"))
                self._capture_review(rec, holder.get("agent"))
                self._capture_coverage(rec, holder.get("agent"))
                _snapshot(holder.get("agent"))

            try:
                import asyncio

                build_intent = intent
                build_kwargs: dict[str, Any] = {}
                answers: dict[str, str] | None = None
                if clarify_mode != "off":
                    try:
                        build_intent, answers = asyncio.run(
                            self._gather_requirements(intent, gate, notify,
                                                      mode=clarify_mode,
                                                      history=history)
                        )
                        rec.add_event("intent", intent=build_intent, answers=answers)
                    except Cancelled:
                        raise
                    except Exception as e:  # noqa: BLE001
                        # The clarifier sharpens a request; it is not the request.
                        # Since `auto` made it run on *every* build, a failure here
                        # would newly take down builds that used to work — so it
                        # degrades to what the user actually typed and says so.
                        logger.warning("clarification failed, building as written",
                                       exc_info=True)
                        notify(f"could not refine the request ({e}) — building as written")
                        build_intent, answers = intent, None

                if refines:
                    build_intent = self._refine_intent(refines, build_intent)
                    build_kwargs["refines"] = refines

                # The gate always goes in: installing a third-party server has to
                # be able to ask, whatever the other switches say.
                agent = self._make_agent(
                    notify, verify_mode, gate,
                    review_plan=review_plan, approve_tools=approve_tools,
                    save_secret=save_secret,
                    # Verification is the longest phase of a build and showed
                    # nothing but "running verification" — while executing the
                    # real graph through the real runner, which already emits
                    # exactly the events a registered run uses to light the
                    # canvas. They were never forwarded.
                    node_event=lambda nid, status, *_a: rec.add_event(
                        "node", node_id=str(nid), status=str(status),
                        phase="verify",
                    ),
                )
                holder["agent"] = agent
                if answers:
                    build_kwargs["answers"] = answers
                if plan:
                    build_kwargs["plan"] = plan
                # Keywords are passed only when they carry something, so an agent
                # implementing the plain `build(intent)` contract keeps working.
                path = asyncio.run(agent.build(build_intent, **build_kwargs))

                _snapshot(agent)
                self._capture_verification(rec, agent, None)
                self._capture_coverage(rec, agent)
                name = getattr(getattr(agent, "session", None), "name", None)
                name = name or Path(path).name
                rec.workflow = name
                rec.path = path
                # Authoritative final graph from the registry, if available.
                try:
                    pkg = self.registry.get(name)
                    rec.graph = pkg.graph.model_dump(mode="json")
                    rec.add_event("graph", graph=rec.graph)
                except Exception:  # noqa: BLE001 - fall back to the last snapshot
                    pass
                rec.status = "succeeded"
                rec.add_event("build", status="succeeded", workflow=name, path=path)
            except Cancelled:
                rec.status = "cancelled"
                rec.add_event("build", status="cancelled")
            except Exception as e:  # noqa: BLE001 - map outcomes to terminal states
                from neurosurfer.architect.agent.agent import WorkflowInfeasible

                if gate.cancelled:
                    # A cancel that got converted into an ordinary error on its
                    # way out still means the user asked to stop.
                    rec.status = "cancelled"
                    rec.add_event("build", status="cancelled")
                elif isinstance(e, WorkflowInfeasible):
                    rec.status = "blocked"
                    rec.error = str(e)
                    # The structured half of the same answer (Phase 2c). Discarding
                    # it left the studio with a paragraph where it could have had a
                    # checklist of servers and the credentials each wants.
                    rec.requirements = list(getattr(e, "requirements", None) or [])
                    rec.add_event(
                        "build", status="blocked", error=str(e),
                        requirements=rec.requirements,
                    )
                else:
                    rec.status = "failed"
                    rec.error = str(e)
                    rec.add_event("build", status="failed", error=str(e))
            finally:
                rec.pending = None

        threading.Thread(target=_work, name=f"arch-{rec.id[:8]}", daemon=True).start()
        return rec

    # ── requirement gathering ─────────────────────────────────────────────
    async def _gather_requirements(
        self, initial: str, gate: InteractionGate, notify: Callable[[str], None],
        *, mode: str = "auto", history: list[dict[str, str]] | None = None,
    ) -> tuple[str, dict[str, str]]:
        """Run the clarifying conversation, asking the studio each question."""
        import asyncio

        from neurosurfer.architect.conversation import ArchitectConversation

        async def ask(question: str, choices: list[str]) -> str:
            # `gate.ask` blocks, so it goes to a thread rather than stalling this
            # build's event loop.
            return await asyncio.to_thread(
                gate.ask,
                kind="question",
                prompt=question,
                choices=list(choices or []),
                # A question left unanswered shouldn't dead-end the build; "no
                # preference" lets the model choose and keeps going.
                on_timeout="no preference",
            )

        notify("reading the request…" if mode == "auto" else "gathering requirements…")
        conversation = ArchitectConversation(self.provider)
        intent, answers = await conversation.run(
            initial, ask=ask, say=notify, mode=mode, history=history,
        )
        # "nothing to ask" is the expected outcome in auto and worth saying, so a
        # silent pass does not read as a step that failed to happen.
        notify(
            f"requirements settled ({len(answers)} answered)" if answers
            else "the request was clear enough to build"
        )
        return intent, answers

    @staticmethod
    def _refine_intent(workflow: str, instruction: str) -> str:
        # The session is seeded with the real graph before the loop starts, so
        # this tells the model what it is looking at rather than asking it to
        # fetch anything.
        return (
            f"The workflow '{workflow}' is already loaded and staged — view it "
            f"first, then make ONLY the change described below. Keep everything "
            f"else as it is, re-verify, and register it under the same name.\n\n"
            f"Requested change:\n{instruction}"
        )

    # ── verification ──────────────────────────────────────────────────────
    @staticmethod
    def _capture_plan(rec: BuildRecord, agent: Any) -> None:
        """Mirror the agent's plan onto the record whenever it changes.

        The plan is the only artifact that exists before any node does, so it is
        what the studio has to show while a build is still deciding what to make —
        and, when a build blocks at the plan, the whole explanation of why.

        Emitted on every change rather than once: a reviewed plan can be *edited*,
        and a record still showing the version the model proposed would attribute
        the user's design to the agent and hide the step they added.
        """
        session = getattr(agent, "session", None)
        plan = getattr(session, "plan", None) if session is not None else None
        if plan is None:
            return
        try:
            snapshot = plan.to_dict()
        except Exception:  # noqa: BLE001 - a snapshot must never fail a build
            return
        if snapshot == rec.plan:
            return
        rec.plan = snapshot
        rec.add_event("plan", plan=snapshot, rendered=plan.render())

    @staticmethod
    def _capture_coverage(rec: BuildRecord, agent: Any) -> None:
        """Which planned steps became nodes, and which were explicitly dropped.

        Coverage refuses registration (Phase 4b), so by the time a build succeeds
        nothing is missing — what is worth seeing is what got *dropped*, and any
        node no step asked for. Both are silent in the graph alone.
        """
        session = getattr(agent, "session", None)
        if session is None or getattr(session, "plan", None) is None:
            return
        # A blocked build did not build the rest of the plan *by definition* —
        # listing every step it never reached reads as a second, larger failure
        # underneath the first. One build blocked at step 2 of 12 and the card
        # named the other ten as "not built", which is true and useless. The
        # block already says what is missing; nothing is added by enumerating
        # the work it prevented. (V4 Phase 5.)
        if getattr(session, "blocked_reason", None):
            return
        try:
            missing, extra = session.plan_coverage()
        except Exception:  # noqa: BLE001 - a snapshot must never break a build
            return
        coverage = {
            "missing": [s.id for s in missing],
            "unplanned": list(extra),
            "dropped": dict(getattr(session, "dropped_steps", {}) or {}),
        }
        # An empty coverage used to return here, which looks harmless and is not:
        # the studio renders coverage as ONE card, replaced in place, so with no
        # clearing event the card froze at the last non-empty snapshot. A build
        # that went on to build everything still showed "not built: <the last
        # node it added>" — a complaint about work that had just been completed.
        #
        # So an empty state is emitted, but only as a *retraction* of something
        # already said. Announcing "nothing is missing" before any node exists
        # would put an empty card at the top of every transcript.
        if not any(coverage.values()) and not rec.coverage:
            return
        if coverage != rec.coverage:
            rec.coverage = coverage
            rec.add_event("coverage", coverage=coverage)

    @staticmethod
    def _capture_review(rec: BuildRecord, agent: Any) -> None:
        """Mirror the design review onto the record as its own event.

        A review that only reaches the step log is a review nobody reads — it is
        the one signal here that can say "this built fine and answers the wrong
        question", which is exactly the thing a green build hides.
        """
        session = getattr(agent, "session", None)
        report = getattr(session, "last_review", None) if session is not None else None
        if report is None:
            return
        try:
            snapshot = report.to_dict()
        except Exception:  # noqa: BLE001 - a snapshot must never fail a build
            return
        if snapshot == rec.review:
            return
        rec.review = snapshot
        rec.add_event("review", review=snapshot, rendered=report.render())

    def _capture_verification(self, rec: BuildRecord, agent: Any, msg: str | None) -> None:
        """Mirror the agent's verification result onto the record, structured.

        The agent renders its report into the step log as text; the studio needs
        the parts (criteria, per-criterion verdicts, branch cases, coverage gaps)
        to show them as UI rather than a wall of prose.
        """
        session = getattr(agent, "session", None)
        if session is None:
            return
        # The **raw** stored verification, not `last_report`. That property is
        # fingerprint-gated — it answers "does this still describe the graph as it
        # stands", which is the right question for *reusing* a result and the
        # wrong one for *reporting* it. A failed verification is always followed
        # by edits, so the fingerprint always moved, and the verdicts were dropped
        # every single time: a build that ran three verifications showed the user
        # three criteria and no outcomes.
        stored = getattr(session, "verification", None)
        report = getattr(stored, "report", None) if stored else None
        if report is None:
            # A session that keeps no record, or one whose verification still
            # matches — either way `last_report` is the same object.
            report = getattr(session, "last_report", None)
        plan = getattr(session, "acceptance_plan", None)
        if report is None and plan is None:
            return

        payload: dict[str, Any] = {}
        if plan is not None:
            payload["criteria"] = [
                {"id": c.id, "description": c.description}
                for c in getattr(plan, "criteria", []) or []
            ]
            payload["test_inputs"] = getattr(plan, "test_inputs", {}) or {}
            # The fixture the harness built for the run (Phase 5a). A PASSED whose
            # fixture turns out to have been one empty file is a verdict about
            # nothing, and the report is the only place that shows it.
            fixtures = getattr(plan, "fixtures", None)
            if fixtures:
                payload["fixtures"] = {
                    "setup": getattr(fixtures, "setup", "") or "",
                    "creates": list(getattr(fixtures, "creates", []) or []),
                }
        if report is not None:
            payload.update(
                {
                    "passed": bool(getattr(report, "passed", False)),
                    "run_ok": bool(getattr(report, "run_ok", False)),
                    "verdicts": list(getattr(report, "verdicts", []) or []),
                    "case_results": list(getattr(report, "case_results", []) or []),
                    "coverage_gaps": list(getattr(report, "coverage_gaps", []) or []),
                    "diagnosis": getattr(report, "diagnosis", "") or "",
                    "suggestions": getattr(report, "suggestions", "") or "",
                    # Full graph executions this verification cost — the expensive
                    # half of a build, so it belongs on screen and not just in a log.
                    "graph_runs": int(getattr(report, "graph_runs", 0) or 0),
                    # Phase 5a: what the run actually had on disk, and any input the
                    # harness repointed at a fixture file.
                    "fixtures_created": list(getattr(report, "fixtures_created", []) or []),
                    "paired_inputs": dict(getattr(report, "paired_inputs", {}) or {}),
                    # Phase 5b: steps that ran against a stub. A PASSED that quietly
                    # tested half the graph is worse than a FAILED, so this has to
                    # reach the badge and not only the text.
                    "not_exercised": list(getattr(report, "not_exercised", []) or []),
                    "stubbed_tools": list(getattr(report, "stubbed_tools", []) or []),
                    "partial": bool(getattr(report, "partial", False)),
                    # Phase 5d: the test rig failed rather than the workflow, so
                    # this is "unverified", not "verified and bad".
                    "fixture_problem": bool(getattr(report, "fixture_problem", False)),
                    "missing_fixture_for": list(
                        getattr(report, "missing_fixture_for", []) or []
                    ),
                }
            )
        if payload and payload != rec.verification:
            rec.verification = payload
            rec.add_event("verification", verification=payload)

    # ── event plumbing ────────────────────────────────────────────────────
    @staticmethod
    def _pending(rec: BuildRecord, interaction: Any) -> None:
        rec.pending = interaction.to_dict()
        rec.add_event("pending", interaction=rec.pending)

    @staticmethod
    def _resolved(rec: BuildRecord, interaction: Any, value: Any, answered: bool) -> None:
        rec.pending = None
        # A `secrets_request` answer carries credential values. The build record is
        # streamed to the browser, replayed on reconnect and kept for the life of
        # the build — everything a secret must never enter. Only the names survive.
        if getattr(interaction, "kind", "") == "secrets_request":
            value = (
                {"supplied": sorted(str(k) for k in value)}
                if isinstance(value, dict) else {"supplied": []}
            )
        rec.add_event(
            "resolved",
            interaction_id=interaction.id,
            value=value,
            # Distinguishes "the user chose this" from "nobody replied, so we
            # applied the default" — they read identically in the log otherwise.
            answered=answered,
        )

    # ── agent construction (overridable for tests) ────────────────────────
    def _make_agent(
        self,
        notify: Callable[[str], None],
        verify: str,
        gate: InteractionGate | None = None,
        *,
        review_plan: bool = False,
        approve_tools: bool = False,
        save_secret: Any = None,
        node_event: Any = None,
    ) -> Any:
        if self._agent_factory is not None:
            # The gate is offered as a third argument so a test double can drive
            # the interactive paths; factories that only want (notify, verify)
            # keep working. Decided by signature rather than by catching
            # TypeError, which would also swallow one raised inside the factory.
            import inspect

            try:
                takes_gate = len(inspect.signature(self._agent_factory).parameters) >= 3
            except (TypeError, ValueError):  # builtins and C callables have no signature
                takes_gate = False
            if takes_gate:
                return self._agent_factory(notify, verify, gate)
            return self._agent_factory(notify, verify)
        from neurosurfer.architect import ArchitectAgent

        async def _auto_approve(draft: Any, _res: Any) -> bool:
            notify(f"authored tool: {getattr(draft, 'name', '?')} (auto-approved)")
            return True

        async def _ask_approval(draft: Any, res: Any) -> bool:
            import asyncio

            name = getattr(draft, "name", "?")
            answer = await asyncio.to_thread(
                gate.ask,
                kind="tool_approval",
                prompt=f"Register the authored tool '{name}'?",
                choices=["approve", "reject"],
                detail=_tool_detail(draft, res),
                # An unattended build must not silently install generated code.
                on_timeout="reject",
            )
            approved = str(answer).strip().lower() in {"approve", "approved", "yes", "true"}
            notify(f"authored tool: {name} ({'approved' if approved else 'rejected'})")
            return approved

        async def _ask_mcp_install(cfg: Any, entry: dict[str, Any]) -> bool:
            import asyncio

            name = getattr(cfg, "name", "?")
            answer = await asyncio.to_thread(
                gate.ask,
                kind="mcp_install_approval",
                prompt=f"Install and run the MCP server '{name}'?",
                choices=["approve", "reject"],
                detail=_mcp_detail(cfg, entry),
                # Unattended, this would start a third-party process on the
                # user's machine. Silence is not consent.
                on_timeout="reject",
            )
            approved = str(answer).strip().lower() in {"approve", "approved", "yes", "true"}
            notify(f"MCP server '{name}' ({'approved' if approved else 'rejected'})")
            return approved

        async def _ask_for_secrets(missing: list[dict[str, str]]) -> bool:
            """Ask for stored values, save them, and report only *whether* any came.

            The values are written straight to the account's store and never
            returned to the caller. That is deliberate: the agent asks for a
            credential and is told "yes, it is set" — it never holds one, so it
            cannot put one in a prompt, a node, or its own summary. The rule the
            whole secrets design serves is that a credential reaches the tool and
            nothing else, and an agent that could see one to pass it along would
            be a hole straight through it.
            """
            import asyncio

            names = [m["name"] for m in missing if m.get("name")]
            answer = await asyncio.to_thread(
                gate.ask,
                kind="secrets_request",
                prompt=(
                    "This workflow needs "
                    + (f"{names[0]}" if len(names) == 1 else f"{len(names)} stored values")
                    + " before it can be tested."
                ),
                choices=["save", "skip"],
                detail={"missing": list(missing)},
                # Typing several credentials from a password manager is not a
                # ten-second decision; matched to the browser-consent timeout.
                timeout_s=600.0,
                # Silence means the run goes ahead unsatisfied and fails honestly,
                # rather than parking a build nobody is watching.
                on_timeout="skip",
            )
            values: dict[str, str] = {}
            if isinstance(answer, dict):
                values = {
                    str(k): str(v) for k, v in answer.items()
                    if k in set(names) and str(v or "").strip()
                }
            if not values or save_secret is None:
                return False
            try:
                from neurosurfer.mcp.credentials import remember_credential

                for name, value in values.items():
                    save_secret(name, value)
                    # Also into the context the build is running under. The
                    # account snapshot was taken before the build started, so
                    # without this the next requirement check cannot see what
                    # was just saved and asks for all of it again.
                    remember_credential(name, value)
            except Exception:  # noqa: BLE001 - a failed save is "not supplied"
                logger.exception("could not store supplied secrets")
                return False
            # Names only. A value must never reach a log line or the transcript.
            notify(f"stored {', '.join(sorted(values))}")
            return True

        async def _ask_plan_question(question: str, _choices: Any = None) -> str:
            import asyncio

            return await asyncio.to_thread(
                gate.ask,
                kind="question",
                prompt=question,
                choices=[],
                # Silence means "you decide", which is what the build would have
                # done anyway — this is a question worth asking, not one worth
                # stalling on.
                on_timeout="no preference",
            )

        async def _await_authorization(server: str, setup_url: str) -> bool:
            import asyncio

            answer = await asyncio.to_thread(
                gate.ask,
                kind="mcp_authorization",
                prompt=f"'{server}' needs you to grant access in your browser.",
                choices=["authorized", "skip"],
                detail={"server": server, "setup_url": setup_url},
                # Longer than the rest: this one sends you to a third-party consent
                # screen, and a sign-in plus an OAuth grant is not a ten-second
                # decision like approving a diff.
                timeout_s=600.0,
                # Silence means "I did not do it" — retrying the install would just
                # fail again against the same un-granted account.
                on_timeout="skip",
            )
            granted = str(answer).strip().lower() in {
                "authorized", "authorised", "done", "yes", "true", "approve"
            }
            notify(f"'{server}' {'authorised' if granted else 'not authorised'}")
            return granted

        async def _ask_plan(plan: Any) -> Any:
            """Show the plan and return what to build — possibly an edited one."""
            import asyncio

            unresolved = [s.id for s in plan.unresolved_steps]
            prompt = (
                "Review the plan before I build it."
                if not unresolved else
                f"These steps need capabilities that aren't available: "
                f"{', '.join(unresolved)}. Review the plan before I build it."
            )
            answer = await asyncio.to_thread(
                gate.ask,
                kind="plan_approval",
                prompt=prompt,
                choices=["build", "cancel"],
                detail={"plan": plan.to_dict(), "rendered": plan.render(),
                        "unresolved": unresolved},
                # Silence means stop. A plan nobody looked at is not an approved
                # plan, and an unresolved step becomes a node that cannot do its
                # job — the bug this whole phase exists for.
                on_timeout="cancel",
            )
            # The answer is either a verdict string or a whole edited plan.
            if isinstance(answer, dict):
                edited = answer.get("plan") if "plan" in answer else answer
                action = str(answer.get("action", "build")).strip().lower()
                if action in {"cancel", "stop", "reject", "no"}:
                    notify("plan cancelled")
                    return None
                if isinstance(edited, dict):
                    notify("plan accepted with edits")
                    return edited
                notify("plan accepted")
                return plan
            approved = str(answer).strip().lower() in {
                "build", "build anyway", "approve", "yes", "true"
            }
            notify(f"plan {'accepted' if approved else 'cancelled'}")
            return plan if approved else None

        return ArchitectAgent(
            self.provider,
            registry=self.registry,
            staging_root=self.staging_root,
            notify=notify,
            verify=verify,
            # Only when asked for: an unreviewed build must not park on a plan
            # nobody is watching for.
            approve_plan=_ask_plan if (gate is not None and review_plan) else None,
            # Authoring is auto-approved unless asked for: it is code we generated
            # and sandbox-tested.
            approve_tool=(
                _ask_approval if (gate is not None and approve_tools) else _auto_approve
            ),
            # Installing is *always* asked, whenever there is anyone to ask. It
            # runs code from a public index, which is a different decision from
            # approving our own — and gating it behind `approve_tools` meant the
            # default configuration could not install anything at all, while the
            # resolver went on telling the model to. No auto-approve counterpart:
            # without a person, the answer is no.
            approve_mcp=_ask_mcp_install if gate is not None else None,
            # Same channel, same rule: with nobody to ask, a consent screen cannot
            # be visited, so the install fails as it did before.
            # The planner's `open_questions` — decisions only the user can make,
            # which were rendered at the foot of the plan card and never asked.
            # Same channel and same shape as a clarifying question.
            ask_question=_ask_plan_question if gate is not None else None,
            node_event=node_event,
            request_authorization=_await_authorization if gate is not None else None,
            # Needs both a person to ask and somewhere to put the answer. Without
            # either, verification runs unsatisfied and fails honestly — which is
            # the old behaviour, now reached deliberately instead of by omission.
            request_secrets=(
                _ask_for_secrets
                if (gate is not None and save_secret is not None) else None
            ),
        )


def _mcp_detail(cfg: Any, entry: dict[str, Any]) -> dict[str, Any]:
    """What a person needs in order to judge an MCP install: who published it,
    what will actually run here, and what it will be handed."""
    from neurosurfer.mcp.registry import credential_requirements

    transport = getattr(cfg, "transport", None)
    return {
        "name": getattr(cfg, "name", None),
        "registry_name": entry.get("name"),
        "description": entry.get("description"),
        "repository": (entry.get("repository") or {}).get("url"),
        "version": entry.get("version"),
        "transport": transport,
        # For stdio this is the command line that will execute locally — the
        # single most important thing to show before saying yes.
        "command": (
            " ".join([getattr(cfg, "command", "") or "", *(getattr(cfg, "args", None) or [])]).strip()
            if transport == "stdio" else None
        ),
        "url": getattr(cfg, "url", None),
        # Names only. A credential's *value* must never travel to the UI.
        "credentials": [c.to_dict() for c in credential_requirements(entry)],
        "supplied_keys": sorted(getattr(cfg, "env", None) or {}),
    }


def _tool_detail(draft: Any, res: Any) -> dict[str, Any]:
    """What a person needs in order to judge an authored tool: what it claims to
    do, the code that will run, and how its sandbox run went."""
    detail: dict[str, Any] = {
        "name": getattr(draft, "name", None),
        "description": getattr(draft, "description", None),
        "source": getattr(draft, "source", None) or getattr(draft, "code", None),
        "tests": getattr(draft, "tests", None),
        "input_schema": getattr(draft, "input_schema", None),
    }
    if res is not None:
        detail["sandbox"] = {
            "ok": bool(getattr(res, "ok", getattr(res, "passed", False))),
            "output": str(getattr(res, "output", "") or "")[:4000],
            "error": str(getattr(res, "error", "") or "")[:2000],
        }
    return {k: v for k, v in detail.items() if v is not None}
