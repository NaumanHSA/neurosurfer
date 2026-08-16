"""BuildSession — the staged workflow one Architect-agent run is constructing (4a).

All architect tools operate on this shared object: nodes are added/edited in
memory, staged to disk as a real WorkflowPackage for validation, and registered
only when the validation gate passes. The session also records the terminal
outcome (registered path / blocked reason) that :class:`ArchitectAgent` reads
after the loop ends.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

__all__ = ["BuildSession"]


@dataclass
class VerificationRecord:
    """A verification, and exactly what it was a verification *of*.

    Verifying re-runs the whole graph — by some distance the most expensive thing a
    build does. Storing *what* was verified, rather than a bare "verified" flag, is
    what lets a repeat `test_workflow` on an unchanged design be answered from here
    instead of paid for again.
    """

    fingerprint: str          # graph + authored tools at the time of the run
    inputs_key: str           # the test inputs actually used
    passed: bool
    rendered: str
    report: Any               # the structured VerificationReport


@dataclass
class BuildSession:
    """Mutable state for one architect build."""

    intent: str
    staging_root: Path
    registry: Any                      # WorkflowRegistry
    knowledge: Any                     # KnowledgeBase
    provider: Any = None               # for author_tool
    approve_tool: Any = None           # async (ToolDraft, SandboxResult) -> bool
    # async (McpServerConfig, registry_entry) -> bool. Installing an MCP server
    # runs third-party code on the user's machine, so it is gated exactly like
    # authoring a tool. None means no channel, and installs are refused.
    approve_mcp: Any = None
    #: (server, setup_url) -> bool. Parks the build while the user grants
    #: access in a browser; None means nobody is there to ask.
    request_authorization: Any = None
    #: (question, choices) -> str. Puts one question to the user mid-build.
    #: Used for the planner's `open_questions` — which were rendered at the
    #: bottom of the plan card and never asked, so a design decision only the
    #: user could make was shown to them as a footnote and then guessed at.
    ask_question: Any = None
    #: (node_id, status) -> None. Node-level progress from a verification run,
    #: so the canvas lights up while a test executes — the same events a
    #: registered run emits. Verification is the longest phase of a build and
    #: showed nothing but "running verification" until this was wired.
    node_event: Any = None
    #: (missing) -> bool. Parks the build while the user supplies stored values
    #: the workflow needs to run. The callback stores them and reports whether
    #: anything arrived; **the values never come back through here**, so a
    #: credential cannot reach the session, the transcript or the model.
    request_secrets: Any = None
    notify: Callable[[str], None] = lambda _m: None

    # staged workflow
    name: str = ""
    description: str = ""
    inputs: list[dict[str, Any]] = field(default_factory=list)
    nodes: list[dict[str, Any]] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)

    # terminal outcome
    registered_path: str | None = None
    #: The graph fingerprint at the moment `register()` wrote the package. What
    #: makes "the registry copy is stale" a question that can be asked — see
    #: :meth:`sync_registration`.
    registered_fingerprint: str = ""
    blocked_reason: str | None = None
    authored_tools: list[str] = field(default_factory=list)
    # Node ids whose capability warning the agent has explicitly argued past, with
    # the reason it gave. The lexical capability check reads prompt text, so it can
    # be wrong; overriding it must be possible, but never silent — this is what the
    # transcript and the studio show.
    acknowledged_capabilities: dict[str, str] = field(default_factory=dict)
    # Phase 2: what the resolution ladder was asked and what it found, plus any
    # MCP servers installed during this build. Both are build history the studio
    # renders — "why did it pick that tool" is otherwise unanswerable.
    resolutions: list[Any] = field(default_factory=list)
    installed_servers: list[str] = field(default_factory=list)
    # Phase 3: the plan this build is executing, with every external step already
    # resolved. The builder, the register gate and the studio all read this one
    # artifact rather than each forming their own idea of what was intended.
    plan: Any = None
    # The user's answers to the plan's `open_questions`, for the builder.
    plan_answers: str = ""
    # node id → plan step id, for the nodes whose mapping isn't just their name.
    # The builder is told to keep the plan's ids, so most nodes need no entry;
    # this holds the ones it renamed or split.
    node_steps: dict[str, str] = field(default_factory=dict)
    # Plan steps the agent deliberately did not build, and why. A dropped step is
    # a decision — sometimes a right one, as when the router turns out to be the
    # classifier — but it must be a stated decision rather than a silent gap.
    dropped_steps: dict[str, str] = field(default_factory=dict)

    # closed-loop verification (Phase 5)
    # "off": no test tooling; "encouraged": test available + prompted;
    # "required": register refuses without a passing, non-stale verification.
    # Required by default since V3 Phase 5c — a workflow nobody ran is the thing
    # this whole rebuild exists to stop registering.
    verification_mode: str = "required"
    # Phase 4 design review: "off" | "warn" (surface issues, register anyway) |
    # "required" (refuse until clean). Warn by default — see review.py for why a
    # judgement is not allowed to be a gate.
    review_mode: str = "warn"
    review: Any = None                          # (fingerprint, ReviewReport)
    acceptance_plan: Any = None                 # cached AcceptancePlan
    # One re-derivation of the acceptance plan per build when the test rig fails
    # (Phase 5b/5d). Tracked here rather than inferred from the failure, so a
    # second rig problem cannot re-enter the loop.
    fixture_retry_used: bool = False
    verification: VerificationRecord | None = None
    # How many full graph executions this build has paid for, so the cost of
    # verification is a number in the log rather than an impression.
    graph_runs: int = 0

    def load_from(self, package: Any) -> None:
        """Seed this session with an already-registered workflow.

        Modifying an existing workflow means starting from its real graph. The
        design tools (view / update_node / remove_node) all read this state, so
        without seeding, a "refine" build would begin on an empty canvas and
        rebuild from scratch — quietly replacing the workflow instead of
        changing it.
        """
        graph = package.graph
        dump = graph.model_dump(mode="json", exclude_none=True)
        self.name = dump.get("name") or getattr(package, "name", "") or self.name
        self.description = dump.get("description") or ""
        self.inputs = list(dump.get("inputs") or [])
        self.nodes = list(dump.get("nodes") or [])
        self.outputs = list(dump.get("outputs") or [])
        # The loaded graph has not been verified *by this session*, and its
        # design is about to change anyway.
        self.invalidate_verification()

    def invalidate_verification(self) -> None:
        """Drop the stored verification outright.

        Edits do not need this — :meth:`current_verification` compares fingerprints,
        so a changed graph stales itself and an edit that changes nothing does not.
        Reserved for wholesale replacement of the session's subject, i.e.
        :meth:`load_from`.
        """
        self.verification = None

    def graph_fingerprint(self) -> str:
        """A stable digest of everything a verification's result depends on.

        The graph, plus the tools authored so far: a node can name a tool before it
        exists, so authoring one changes what the same graph text *does*.
        """
        payload = json.dumps(
            {"graph": self.graph_dict(), "tools": sorted(self.authored_tools)},
            sort_keys=True, ensure_ascii=False, default=str,
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    @staticmethod
    def inputs_key(test_inputs: dict[str, Any] | None) -> str:
        return json.dumps(test_inputs or {}, sort_keys=True, ensure_ascii=False, default=str)

    def current_verification(self) -> VerificationRecord | None:
        """The stored verification, if it still describes the graph as it stands."""
        rec = self.verification
        if rec is None or rec.fingerprint != self.graph_fingerprint():
            return None
        return rec

    def cached_verification(self, test_inputs: dict[str, Any] | None) -> VerificationRecord | None:
        """A stored verification of *this* graph on *these* inputs, if there is one."""
        rec = self.current_verification()
        if rec is None or rec.inputs_key != self.inputs_key(test_inputs):
            return None
        return rec

    def record_verification(
        self, *, passed: bool, rendered: str, report: Any, test_inputs: dict[str, Any] | None
    ) -> None:
        self.verification = VerificationRecord(
            fingerprint=self.graph_fingerprint(),
            inputs_key=self.inputs_key(test_inputs),
            passed=passed,
            rendered=rendered,
            report=report,
        )

    # Read-only views kept for the agent loop, the register gate and the studio's
    # verification panel. They answer `None` once the graph has moved on, which is
    # what "stale" now means.
    @property
    def last_verification(self) -> tuple[bool, str] | None:
        rec = self.current_verification()
        return (rec.passed, rec.rendered) if rec else None

    @property
    def last_report(self) -> Any:
        rec = self.current_verification()
        return rec.report if rec else None

    # ── capability resolution (Phase 2) ─────────────────────────────────────
    def record_resolution(self, resolution: Any) -> None:
        """Keep what the ladder was asked and what came back, newest last."""
        self.resolutions.append(resolution)

    def blocking_requirements(self) -> list[dict[str, Any]]:
        """What the user would have to supply for the unresolved capabilities.

        Attached to :class:`WorkflowInfeasible` so a blocked build hands back a
        checklist — the server that provides the missing capability and the
        credentials it wants — rather than a sentence about it being impossible.
        """
        # Keyed by capability, because the same one arrives twice: once from the
        # plan step that needs it and again from the `find_capability` the agent
        # ran for that step. Rendered as two identical checklist items, which
        # reads as two problems and doubles a list nobody wants to be long.
        out: dict[str, dict[str, Any]] = {}

        def add(capability: str, servers: list[Any], step: str | None = None) -> None:
            key = (capability or "").strip().lower()
            if not key:
                return
            existing = out.get(key)
            if existing is None:
                out[key] = {"capability": capability, "servers": servers}
                if step:
                    out[key]["step"] = step
                return
            # A later mention may know the step, or may have found servers the
            # earlier one had not; keep whichever half is more useful.
            if step and "step" not in existing:
                existing["step"] = step
            if servers and not existing.get("servers"):
                existing["servers"] = servers

        # A plan's unresolved external steps are blockers too, and they are the
        # ones found *before* any node was written — the cheapest place to learn
        # a build cannot happen.
        for step in getattr(self.plan, "unresolved_steps", []) or []:
            res = step.resolution or {}
            add(step.needed_capability or step.intent, res.get("servers", []), step.id)
        # `resolutions` is an append-only log of what the ladder was *asked* and
        # what it answered **at the time**. Plan steps get re-resolved after an
        # install; these never do, so a capability answered `installable` and then
        # acquired keeps that verdict forever — and a build that installed exactly
        # the server it needed still ended by telling the user to go and install
        # it. Seen live, with the server connected and serving seven nodes.
        for res in self.resolutions:
            # "have" is satisfied; "not_external" was never a capability to begin
            # with (it is writing). Neither belongs on a list of things the user
            # must go and obtain.
            if getattr(res, "status", "have") in {"have", "not_external"}:
                continue
            if self._now_satisfied(res.need):
                continue
            add(res.need, [s.to_dict() for s in getattr(res, "servers", [])])
        return list(out.values())

    def _now_satisfied(self, capability: str) -> bool:
        """Has this capability been acquired since the ladder last answered?

        Only asked for entries from the resolution log, which is the stale half —
        re-checking the plan's steps would repeat a registry round trip for
        capabilities that were settled long ago.

        Best-effort: a resolver fault must never delete a genuine blocker from the
        checklist, so anything unexpected keeps the entry.
        """
        if not capability:
            return False
        try:
            from ..capability import resolve_capability

            # Local catalog only. The question is "has this been *acquired*", and
            # acquiring means the tool is here now — a registry hit would answer
            # `installable`, which is precisely not acquired. Searching spends a
            # multi-second network round trip, once per blocker, on an answer it
            # cannot give.
            resolution = resolve_capability(capability, include_registry=False)
            return getattr(resolution, "status", "") in {"have", "not_external"}
        except Exception:  # noqa: BLE001 - keep the blocker when unsure
            return False

    async def await_authorization(self, server: str, setup_url: str) -> bool:
        """Ask the user to grant access in a browser, and wait for them to say done.

        A consent screen is the one credential nobody can hand over in a form, so
        the previous behaviour was to give up and print the URL in a refusal — a
        link the user then had to find, visit, and re-run the whole build behind.
        The build is already parked whenever it asks anything; parking here too
        turns a dead end into a pause.

        False when there is no channel to ask on (headless), or the user declined
        — the caller then tries another server or blocks, as before.
        """
        if self.request_authorization is None:
            return False
        result = self.request_authorization(server, setup_url)
        if hasattr(result, "__await__"):
            result = await result
        return bool(result)

    async def await_secrets(self, missing: list[dict[str, str]]) -> bool:
        """Ask for stored values the staged workflow needs, and wait.

        A workflow that reaches a real database needs a real connection string,
        and verification runs it for real. Without this the run fails on the
        first node with whatever the service says about an empty password, the
        judge reports every criterion unmet, and the model reads a workflow it
        built correctly as broken — then edits a working design to chase it. It
        is the same shape as the browser-consent case: the build already parks
        whenever it asks anything, so it parks here rather than failing.

        Returns True when values arrived and the run is worth attempting again.
        False when there is nobody to ask (headless) or the user skipped — the
        caller verifies anyway, so an unsatisfiable requirement still surfaces
        as a real failure rather than a silent skip.
        """
        if self.request_secrets is None or not missing:
            return False
        result = self.request_secrets(missing)
        if hasattr(result, "__await__"):
            result = await result
        return bool(result)

    async def approve_mcp_install(self, cfg: Any, entry: dict[str, Any]) -> bool:
        """Ask the human whether to install and run this server.

        Refuses when there is no approval channel rather than defaulting to yes:
        the alternative is a headless build silently starting a third-party
        process on someone's machine.
        """
        if self.approve_mcp is None:
            return False
        result = self.approve_mcp(cfg, entry)
        if hasattr(result, "__await__"):
            result = await result
        return bool(result)

    # ── design review (Phase 4) ─────────────────────────────────────────────
    @property
    def last_review(self) -> Any:
        """The stored review, if it still describes the graph as it stands."""
        if self.review is None or self.review[0] != self.graph_fingerprint():
            return None
        return self.review[1]

    async def ensure_review(self) -> Any:
        """Review this graph, or reuse the review of this exact graph.

        Keyed to the same fingerprint verification uses, so an unchanged design is
        never reviewed twice — the reviewer would only be asked to re-derive an
        opinion we already hold, at the price of another call.
        """
        cached = self.last_review
        if cached is not None:
            return cached
        from ..review import review_workflow

        self.notify("reviewing the design against the request")
        report = await review_workflow(
            self.provider, intent=self.intent, graph_yaml=self.to_yaml(),
            plan=self.plan,
        )
        self.review = (self.graph_fingerprint(), report)
        self.notify(report.render())
        return report

    # ── plan ↔ graph coverage (Phase 4) ─────────────────────────────────────
    def step_for_node(self, node_id: str) -> str | None:
        """Which plan step this node implements, if any.

        An explicit citation wins; otherwise the id itself, because the builder is
        told to keep the plan's ids and overwhelmingly does. Inferring the common
        case means the model only has to say something when it did something
        unusual — the alternative is a required field it will forget.
        """
        cited = self.node_steps.get(node_id)
        if cited:
            return cited
        if self.plan is not None and self.plan.step(node_id) is not None:
            return node_id
        return None

    def plan_coverage(self) -> tuple[list[Any], list[str]]:
        """(plan steps nothing implements, top-level nodes no step asked for).

        Body nodes are excluded: a `map` step becomes one container node whose
        body is an implementation detail the plan never described.
        """
        if self.plan is None or not self.plan.steps:
            return [], []
        covered = {
            step for n in self.nodes
            if (step := self.step_for_node(n.get("id", ""))) is not None
        }
        missing = [
            s for s in self.plan.steps
            if s.id not in covered and s.id not in self.dropped_steps
        ]
        extra = [
            nid for n in self.nodes
            if (nid := n.get("id", "")) and self.step_for_node(nid) is None
        ]
        return missing, extra

    # ── graph assembly ──────────────────────────────────────────────────────
    def node_ids(self) -> list[str]:
        return [n.get("id", "") for n in self.nodes]

    def get_node(self, node_id: str) -> dict[str, Any] | None:
        return next((n for n in self.nodes if n.get("id") == node_id), None)

    def graph_dict(self) -> dict[str, Any]:
        return {
            "name": self.name or "unnamed_workflow",
            "description": self.description,
            "inputs": self.inputs,
            "nodes": self.nodes,
            "outputs": self.outputs or ([self.nodes[-1]["id"]] if self.nodes else []),
        }

    def to_yaml(self) -> str:
        return yaml.dump(self.graph_dict(), sort_keys=False, allow_unicode=True)

    # ── staging / validation / registration ─────────────────────────────────
    def stage(self) -> Path:
        """Write the staged package (workflow.yaml + graph.yaml) and return its dir."""
        import datetime

        name = self.name or "unnamed_workflow"
        pkg_dir = self.staging_root / name
        pkg_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "name": name,
            "version": "0.1.0",
            "description": self.description,
            "entrypoint": "graph.yaml",
            "created_by": "architect-agent",
            "created_at": datetime.datetime.now(tz=datetime.UTC).isoformat(),
            "tags": ["generated"],
        }
        (pkg_dir / "workflow.yaml").write_text(
            yaml.dump(manifest, sort_keys=False, allow_unicode=True), encoding="utf-8"
        )
        (pkg_dir / "graph.yaml").write_text(self.to_yaml(), encoding="utf-8")
        return pkg_dir

    def _validate(self) -> tuple[bool, str, Any | None]:
        """Stage and validate. Returns (ok, human-readable text, structured report).

        The structured report is ``None`` when staging never got far enough to
        produce one; :meth:`register` needs it to inspect capability warnings, which
        are not visible in the rendered text as anything it can act on selectively.
        """
        if not self.nodes:
            return False, "The workflow has no nodes yet.", None
        if not self.name:
            return False, "The workflow has no name — call set_workflow first.", None
        from neurosurfer.graph.engine.errors import GraphConfigurationError
        from neurosurfer.graph.engine.loader import load_graph_from_dict

        try:
            load_graph_from_dict(self.graph_dict())
        except GraphConfigurationError as e:
            return False, f"Graph structure invalid:\n{e}", None

        from neurosurfer.graph.workflow.package import PackageLoadError, load_package
        from neurosurfer.graph.workflow.validate import validate_package

        try:
            pkg = load_package(self.stage())
        except PackageLoadError as e:
            return False, f"Staged package failed to load: {e}", None
        report = validate_package(pkg)
        if report.ok:
            summary = report.summary()
            return True, ("VALID." if summary == "Package is valid."
                          else f"VALID with warnings:\n{summary}"), report
        return False, report.summary(), report

    def validate(self) -> tuple[bool, str]:
        """Stage and validate the package. Returns (ok, human-readable report)."""
        ok, text, _ = self._validate()
        return ok, text

    def unresolved_capabilities(self, report: Any) -> list[Any]:
        """Capability warnings on *report* that nobody has answered for.

        The check is a warning at the package level — a hand-written package must
        not be rejected on a string match. A *build* is different: the agent is
        right there and can attach a tool, author one, block, or say why the check
        is wrong. So this is where it becomes a refusal.
        """
        if report is None:
            return []
        return [
            w for w in report.warnings
            if w.kind == "capability_gap"
            and w.node_id not in self.acknowledged_capabilities
        ]

    def pre_register(self) -> tuple[bool, str]:
        """Every refusal that costs nothing to decide. Returns (ok, message).

        Split out from :meth:`register` so the caller can run these *before* the
        design review, which costs a model call. Reviewing a graph that is about
        to be refused for a missing tool is money spent on an answer nobody will
        read — and the refusal the agent needs to see is the cheap one anyway.
        """
        ok, report_text, report = self._validate()
        if not ok:
            return False, f"Refusing to register — validation failed:\n{report_text}"

        ungrounded = self.unresolved_capabilities(report)
        if ungrounded:
            lines = [
                "Refusing to register — these nodes describe work an LLM step "
                "cannot do, and hold no tool. A workflow like this runs green and "
                "invents its results.",
                "",
            ]
            for issue in ungrounded:
                lines.append(f"  • node '{issue.node_id}': {issue.message}")
                if issue.suggestion:
                    lines.append(f"      → {issue.suggestion}")
            lines += [
                "",
                "Fix each one with exactly one of:",
                "  - update_node(id, {'kind': 'tool', 'tools': ['<tool>'], "
                "'tool_args': {...}})  ← one direct call",
                "  - update_node(id, {'kind': 'react', 'tools': ['<tool>', ...]})"
                "  ← several steps",
                "  - author_tool(...)         if no catalog tool provides it",
                "  - declare_blocked(reason)  if it needs a credential or "
                "integration you don't have",
                "  - acknowledge_capability(node_id, reason)  if the check is wrong "
                "and the node truly needs no tool",
            ]
            return False, "\n".join(lines)

        # Every planned step must be built or explicitly dropped. A step that
        # quietly vanishes is a piece of what the user asked for going missing,
        # and it is invisible in a graph that otherwise validates.
        missing, _extra = self.plan_coverage()
        if missing:
            lines = [
                "Refusing to register — the plan has steps this workflow does not "
                "implement. The user asked for them:",
                "",
            ]
            for step in missing:
                lines.append(f"  • {step.id} — {step.intent}")
            lines += [
                "",
                "Either build each one with add_node (cite it with "
                "plan_step_id if you name the node differently), or call "
                "drop_plan_step(step_id, reason) if it genuinely should not exist.",
            ]
            return False, "\n".join(lines)

        if self.verification_mode == "required":
            if self.last_verification is None:
                return False, (
                    "Refusing to register — this build requires a passing "
                    "test_workflow verification of the CURRENT graph first. "
                    "Call test_workflow."
                )
            if not self.last_verification[0] and not self.verification_unavailable:
                return False, (
                    "Refusing to register — the last verification FAILED. Fix the "
                    "design and test_workflow again:\n" + self.last_verification[1]
                )
        return True, ""

    @property
    def verification_unavailable(self) -> bool:
        """The test rig failed, so there is no verdict — not a verdict of "bad".

        Refusing here would stall the build permanently on a problem the graph
        cannot cause and the builder cannot fix, which is the same trap that made
        a partial verification registerable (Phase 5b). So it registers, and
        :meth:`verification_caveat` says loudly that nobody ever ran it.
        """
        rec = self.current_verification()
        return bool(getattr(getattr(rec, "report", None), "fixture_problem", False))

    def verification_caveat(self) -> str:
        """What a reader of "registered" still needs to know about the proof.

        A partial verification (Phase 5b) is a real pass over the part of the
        graph this machine can run, and blocking on it would make a workflow
        unregisterable on the very machine that cannot exercise it. So it
        registers — and says which steps nobody has ever seen work.
        """
        rec = self.current_verification()
        report = getattr(rec, "report", None)
        if getattr(report, "fixture_problem", False):
            return (
                " WARNING: this workflow is UNVERIFIED. The test harness could not "
                "be set up (see the last test_workflow report), so nothing has ever "
                "run it end to end. Run it yourself against real inputs before "
                "relying on it."
            )
        not_exercised = list(getattr(report, "not_exercised", []) or [])
        if not not_exercised:
            return ""
        return (
            " NOTE: this was a PARTIAL verification — "
            + ", ".join(not_exercised)
            + " could not be exercised on this machine and have never been proven "
            "to work. Connect the tools they need and test again before relying "
            "on them."
        )

    def register(self) -> tuple[bool, str]:
        """Run every free gate, then register into the registry."""
        ok, msg = self.pre_register()
        if not ok:
            return False, msg
        from neurosurfer.graph.workflow.package import load_package

        pkg = load_package(self.staging_root / self.name)
        dest = self.registry.save(pkg)
        self.registered_path = str(dest)
        self.registered_fingerprint = self.graph_fingerprint()
        caveat = self.verification_caveat()
        note = ""
        if caveat:
            note = " (UNVERIFIED)" if self.verification_unavailable else " (partial verification)"
        self.notify(f"Workflow '{self.name}' registered at {dest}{note}")
        return True, (
            f"Registered at {dest}. The build is complete — you may finish now."
            + caveat
        )

    def registration_is_stale(self) -> bool:
        """Has the design changed since the package was written to the registry?"""
        return bool(
            self.registered_path
            and self.graph_fingerprint() != self.registered_fingerprint
        )

    def sync_registration(self) -> tuple[bool, str]:
        """Re-save the registered package when the session has moved on. **The
        guarantee that the registry copy is the design the build ended with.**

        `register()` snapshots to disk, so anything edited afterwards lived only
        here. That is not a hypothetical: with `review_mode="warn"` the reviewer
        reports its findings *after* the package is written, the tool invites the
        model to fix them, the model does — and the fix reached nothing. The
        registered artifact is the deliverable, so it tracks the session or the
        review is advice nobody can act on.

        Returns ``(changed, message)``. Unchanged is the overwhelmingly common
        case and writes nothing, so this is safe to call on every terminal path.
        A re-save runs the same gates as the first one; if the edit broke
        something, the earlier good copy is left in place and the caller is told,
        because a stale-but-valid package beats a fresh broken one.
        """
        if not self.registration_is_stale():
            return False, ""
        ok, msg = self.register()
        if not ok:
            self.notify(f"edits after registering were NOT saved — {msg}")
            return False, msg
        self.notify("re-registered: the design changed after it was first saved")
        return True, msg
