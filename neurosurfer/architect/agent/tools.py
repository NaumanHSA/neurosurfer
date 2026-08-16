"""The Architect agent's toolbelt (Phase 4b).

Every tool operates on the shared :class:`BuildSession`: graph construction is
incremental and in-memory, validation stages a real package through the same gate
the YAML loader uses, and registration is refused until validation passes. Tool
errors flow back as results so the model can self-correct.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field

from neurosurfer.tools.base import Tool, ToolContext, ToolResult

from .session import BuildSession

__all__ = ["architect_tools"]


def _dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=1, default=str)


# ── workflow meta ───────────────────────────────────────────────────────────────

class SetWorkflowArgs(BaseModel):
    name: str = Field(description="Workflow name in snake_case (e.g. 'ticket_triage').")
    description: str = Field(default="", description="One-sentence description.")
    inputs: list[dict] | None = Field(
        default=None,
        description="Declared graph inputs: [{name, type, required, description}]. "
                    "type ∈ string|integer|float|boolean|object|array.",
    )
    outputs: list[str] | None = Field(
        default=None,
        description="Node ids whose outputs are the workflow result.",
    )


class SetWorkflowTool(Tool):
    name = "set_workflow"
    description = (
        "Set the workflow's name, description, declared inputs, and output node ids. "
        "Call this first; call again anytime to update (e.g. set outputs at the end)."
    )
    input_model = SetWorkflowArgs

    def __init__(self, session: BuildSession) -> None:
        self.session = session

    def is_read_only(self, args: BaseModel) -> bool:
        return False

    async def call(self, args: SetWorkflowArgs, ctx: ToolContext) -> ToolResult:
        import re

        s = self.session
        name = re.sub(r"[^a-z0-9_]", "_", args.name.strip().lower())
        name = re.sub(r"_+", "_", name).strip("_")
        if not name:
            return ToolResult.error("Workflow name must not be empty.")
        s.name = name
        if args.description:
            s.description = args.description
        if args.inputs is not None:
            s.inputs = args.inputs
        if args.outputs is not None:
            s.outputs = args.outputs
        s.notify(f"workflow meta set: {name}")
        return ToolResult.ok(f"Workflow meta set. name={name!r}, "
                             f"inputs={[i.get('name') for i in s.inputs]}, "
                             f"outputs={s.outputs}")


class SetOutputsArgs(BaseModel):
    outputs: list[str] = Field(
        description="Node ids whose outputs are the workflow result (usually the terminal nodes)."
    )


class SetOutputsTool(Tool):
    name = "set_outputs"
    description = (
        "Set the workflow's output node ids — the graph's final result. Use THIS "
        "(not update_node) to declare or change outputs."
    )
    input_model = SetOutputsArgs

    def __init__(self, session: BuildSession) -> None:
        self.session = session

    def is_read_only(self, args: BaseModel) -> bool:
        return False

    async def call(self, args: SetOutputsArgs, ctx: ToolContext) -> ToolResult:
        s = self.session
        unknown = [o for o in args.outputs if o not in set(s.node_ids())]
        if unknown:
            return ToolResult.error(
                f"Unknown output node ids: {unknown}. Existing nodes: {s.node_ids()}"
            )
        s.outputs = list(args.outputs)
        s.notify(f"outputs set: {s.outputs}")
        return ToolResult.ok(f"Workflow outputs set to {s.outputs}.")


# ── node construction ───────────────────────────────────────────────────────────

class AddNodeArgs(BaseModel):
    node: dict = Field(
        description="Complete node spec: {id, kind, purpose/goal, depends_on, tools, "
                    "when, writes, cases/default (router), body/max_iterations/"
                    "until (loop), body/over/as (map), callable (function), …}. "
                    "Same schema as a graph.yaml node."
    )
    plan_step_id: str | None = Field(
        default=None,
        description="The plan step this node implements. Only needed when the node "
                    "id differs from the step id — matching ids are paired "
                    "automatically.",
    )


class AddNodeTool(Tool):
    name = "add_node"
    description = (
        "Add ONE node to the workflow graph. The spec is validated immediately — "
        "fix and retry on error. Node ids must be unique; use update_node to change "
        "an existing node."
    )
    input_model = AddNodeArgs

    def __init__(self, session: BuildSession) -> None:
        self.session = session

    async def call(self, args: AddNodeArgs, ctx: ToolContext) -> ToolResult:
        return _put_node(self.session, args.node, replace=False,
                         plan_step_id=args.plan_step_id)


class UpdateNodeArgs(BaseModel):
    id: str = Field(description="Id of the node to update.")
    patch: dict = Field(description="Fields to change (merged over the current spec).")


class UpdateNodeTool(Tool):
    name = "update_node"
    description = "Update fields of an existing node (merge patch, then re-validate)."
    input_model = UpdateNodeArgs

    def __init__(self, session: BuildSession) -> None:
        self.session = session

    async def call(self, args: UpdateNodeArgs, ctx: ToolContext) -> ToolResult:
        current = self.session.get_node(args.id)
        if current is None:
            hint = ""
            if "outputs" in args.patch or "output" in args.id.lower():
                hint = " To set the workflow's output node ids, call set_outputs (not update_node)."
            return ToolResult.error(
                f"No node '{args.id}'. Existing: {self.session.node_ids()}.{hint}"
            )
        merged = {**current, **args.patch, "id": args.id}
        return _put_node(self.session, merged, replace=True)


class RemoveNodeArgs(BaseModel):
    id: str = Field(description="Id of the node to remove.")


class RemoveNodeTool(Tool):
    name = "remove_node"
    description = "Remove a node from the graph (dependents keep their depends_on — fix them)."
    input_model = RemoveNodeArgs

    def __init__(self, session: BuildSession) -> None:
        self.session = session

    async def call(self, args: RemoveNodeArgs, ctx: ToolContext) -> ToolResult:
        before = len(self.session.nodes)
        self.session.nodes = [n for n in self.session.nodes if n.get("id") != args.id]
        if len(self.session.nodes) == before:
            return ToolResult.error(f"No node '{args.id}' to remove.")
        # Every other mutating tool narrates; this one did not, so a node that was
        # added, removed and added again read as "added twice" in the transcript —
        # a model thrashing and a model repeating itself look identical there.
        self.session.notify(f"node removed: {args.id} [{len(self.session.nodes)} left]")
        return ToolResult.ok(f"Removed '{args.id}'. Remaining: {self.session.node_ids()}")


def _put_node(
    session: BuildSession, spec: dict, *, replace: bool, plan_step_id: str | None = None
) -> ToolResult:
    """Validate a single node spec and insert/replace it in the session."""
    from neurosurfer.graph.engine.schema import GraphNode

    try:
        node = GraphNode.model_validate(spec)
    except Exception as e:  # noqa: BLE001 - validation feedback goes to the model
        return ToolResult.error(f"Invalid node spec: {e}")

    nid = node.id
    exists = session.get_node(nid) is not None
    if exists and not replace:
        return ToolResult.error(
            f"Node '{nid}' already exists — use update_node to change it."
        )

    warnings: list[str] = []
    # Unknown depends_on targets (may be added later — warn, don't block).
    known = set(session.node_ids()) | {nid}
    for dep in node.depends_on or []:
        if dep not in known:
            warnings.append(f"depends_on '{dep}' does not exist yet")
    # Tool names must be real (or authored) — this is the top failure mode.
    if node.tools:
        from neurosurfer.tools.registry import all_tools

        registered = {t.name for t in all_tools()}
        for t in node.tools:
            if t not in registered:
                warnings.append(
                    f"tool '{t}' is not registered — use a catalog tool or author_tool"
                )

    clean = node.model_dump(mode="json", exclude_none=True, exclude_defaults=True)
    clean["id"] = nid
    clean["kind"] = node.kind
    if exists:
        session.nodes = [clean if n.get("id") == nid else n for n in session.nodes]
    else:
        session.nodes.append(clean)

    if plan_step_id:
        step = session.plan.step(plan_step_id) if session.plan is not None else None
        if step is None:
            warnings.append(
                f"plan_step_id '{plan_step_id}' is not a step in the plan — ignored"
            )
        else:
            session.node_steps[nid] = plan_step_id

    # Progress against the plan, so a long build reads as movement rather than a
    # list of node names.
    progress = ""
    if session.plan is not None and session.plan.steps:
        missing, _ = session.plan_coverage()
        done = len(session.plan.steps) - len(missing)
        progress = f" [{done}/{len(session.plan.steps)} steps]"
    session.notify(
        f"node {'updated' if exists else 'added'}: {nid} ({node.kind}){progress}"
    )

    msg = f"Node '{nid}' ({node.kind}) {'updated' if exists else 'added'}. Graph: {session.node_ids()}"
    if warnings:
        msg += "\nWARNINGS:\n- " + "\n- ".join(warnings)
    return ToolResult.ok(msg)


# ── inspection / validation / registration ─────────────────────────────────────

class _NoArgs(BaseModel):
    pass


class ViewWorkflowTool(Tool):
    name = "view_workflow"
    description = "Show the current staged workflow (graph.yaml as it stands now)."
    input_model = _NoArgs

    def __init__(self, session: BuildSession) -> None:
        self.session = session

    def is_read_only(self, args: BaseModel) -> bool:
        return True

    async def call(self, args: _NoArgs, ctx: ToolContext) -> ToolResult:
        return ToolResult.ok(self.session.to_yaml())


class ValidateWorkflowTool(Tool):
    name = "validate_workflow"
    description = (
        "Validate the staged workflow through the full gate (structure, DAG, control "
        "flow, tool names, schemas). Returns VALID or the exact issues to fix."
    )
    input_model = _NoArgs

    def __init__(self, session: BuildSession) -> None:
        self.session = session

    def is_read_only(self, args: BaseModel) -> bool:
        return True

    async def call(self, args: _NoArgs, ctx: ToolContext) -> ToolResult:
        ok, report = self.session.validate()
        self.session.notify(f"validate: {'ok' if ok else 'issues found'}")
        return ToolResult.ok(report) if ok else ToolResult.error(report)


class RegisterWorkflowTool(Tool):
    name = "register_workflow"
    description = (
        "Validate and, if clean, register the workflow so it can be run. Refuses "
        "while validation fails. Call once the design is complete and valid."
    )
    input_model = _NoArgs

    def __init__(self, session: BuildSession) -> None:
        self.session = session

    async def call(self, args: _NoArgs, ctx: ToolContext) -> ToolResult:
        s = self.session

        # Every free refusal first: no point paying a reviewer to read a graph
        # that is about to be turned away for a missing tool or an unbuilt step.
        ok, msg = s.pre_register()
        if not ok:
            return ToolResult.error(msg)

        review = None
        if s.review_mode != "off" and s.provider is not None:
            review = await s.ensure_review()
            if review.issues and s.review_mode == "required":
                return ToolResult.error(
                    "Refusing to register — the design review found problems with "
                    "what this workflow does:\n" + review.render() +
                    "\n\nFix these, then register again."
                )

        ok, msg = s.register()
        if not ok:
            return ToolResult.error(msg)
        if review is not None and review.issues:
            # Registered, but the reviewer disagreed. Saying so on the success
            # channel keeps it in the transcript instead of only in the log.
            #
            # **The "you may finish" half of `register`'s message is dropped.**
            # Sent whole, this said the build was complete *and* asked for a fix
            # in the same breath, and a small model takes the shorter road: a
            # real transcript has it patch the node the reviewer named and then
            # stop, leaving the flaw in the registered copy. One instruction.
            # (`sync_registration` is what makes the fix land either way.)
            msg = msg.split(" The build is complete")[0].rstrip()
            msg += (
                "\n\nThe design review flagged the following. Nothing blocks the "
                "build, but fix these now — your edits are saved automatically, so "
                "correcting a node here is the last thing to do:\n" + review.render()
            )
        return ToolResult.ok(msg)


# ── knowledge / research ────────────────────────────────────────────────────────

class DocsArgs(BaseModel):
    query: str = Field(description="What to look up in the neurosurfer docs.")
    k: int = Field(default=4, description="Max sections to return.")


class NeurosurferDocsTool(Tool):
    name = "neurosurfer_docs"
    # The docs are written for people setting the project up — CLI flags, provider
    # profiles, what a workflow package is. They lag the code and say little about
    # authoring a node. Everything needed to BUILD is already in the system prompt
    # (node kinds, build rules, worked shapes) or in `describe_node_kind`; pointing
    # the model here for those questions returns setup guides and release notes.
    description = (
        "Search the user-facing neurosurfer docs for background on the project — "
        "configuration, the CLI, providers, what a workflow package is. NOT the "
        "place to learn how to author a node: node kinds, their required fields "
        "and worked examples are in your system prompt and in describe(name)."
    )
    input_model = DocsArgs

    def __init__(self, session: BuildSession) -> None:
        self.session = session

    def is_read_only(self, args: BaseModel) -> bool:
        return True

    async def call(self, args: DocsArgs, ctx: ToolContext) -> ToolResult:
        hits = self.session.knowledge.search_docs(args.query, k=args.k)
        if not hits:
            return ToolResult.ok("No matching docs sections.")
        return ToolResult.ok(_dumps(hits))


class DescribeArgs(BaseModel):
    name: str = Field(description="A node kind (e.g. 'router', 'loop') or a tool name "
                                  "(e.g. 'read_file') to get full details for.")


class DescribeCapabilityTool(Tool):
    name = "describe_capability"
    description = (
        "Get full details for one node kind (fields, requirements) or one tool "
        "(description + input schema)."
    )
    input_model = DescribeArgs

    def __init__(self, session: BuildSession) -> None:
        self.session = session

    def is_read_only(self, args: BaseModel) -> bool:
        return True

    async def call(self, args: DescribeArgs, ctx: ToolContext) -> ToolResult:
        kb = self.session.knowledge
        kind = kb.describe_node_kind(args.name)
        if kind is not None:
            return ToolResult.ok(_dumps({"node_kind": args.name, **kind}))
        tool = kb.describe_tool(args.name)
        if tool is not None:
            return ToolResult.ok(_dumps(tool))
        return ToolResult.error(
            f"'{args.name}' is neither a node kind nor a registered tool."
        )


# ── tool authoring ──────────────────────────────────────────────────────────────

class AuthorToolArgs(BaseModel):
    name: str = Field(description="Distinctive snake_case name for the NEW tool.")
    purpose: str = Field(description="One sentence: what the tool does.")
    inputs: list[str] = Field(
        default_factory=list,
        description="Input fields, e.g. 'db_path: path to the SQLite file'.",
    )
    signature_hint: str = Field(default="", description="How call() should behave.")
    test_setup: str = Field(
        default="",
        description="Self-contained stdlib Python that builds test fixtures in cwd "
                    "(may define ARGS dict to supply call arguments).",
    )
    test_args: dict = Field(default_factory=dict,
                            description="Concrete sample args to functionally test with.")
    expected_behavior: str = Field(default="",
                                   description="What a successful test looks like.")


class AuthorToolTool(Tool):
    name = "author_tool"
    description = (
        "Author a brand-new tool when NO catalog tool provides a needed capability. "
        "The tool is generated, sandbox-tested by actually running it, and requires "
        "human approval before it is registered. Prefer existing tools."
    )
    input_model = AuthorToolArgs

    def __init__(self, session: BuildSession) -> None:
        self.session = session

    async def call(self, args: AuthorToolArgs, ctx: ToolContext) -> ToolResult:
        s = self.session
        if s.approve_tool is None:
            return ToolResult.error(
                "No tool-approval channel available in this run — compose existing "
                "catalog tools instead, or declare_blocked explaining the gap."
            )
        if s.provider is None:
            return ToolResult.error("No provider available for tool authoring.")
        from ..tool_author import ToolAuthor, ToolGapSpec

        spec = ToolGapSpec(
            name=args.name,
            purpose=args.purpose,
            inputs=list(args.inputs),
            context=args.signature_hint,
            source_workflow=s.name or None,
            test_setup=args.test_setup,
            test_args=dict(args.test_args),
            expected_behavior=args.expected_behavior,
        )
        author = ToolAuthor(s.provider)
        meta = await author.author(spec, approve=s.approve_tool, notify=s.notify)
        if meta is None:
            reason = getattr(author, "last_failure", "unknown")
            return ToolResult.error(f"Tool '{args.name}' was not registered: {reason}")
        s.authored_tools.append(args.name)
        s.knowledge.refresh()  # the new tool is now part of the capability set
        return ToolResult.ok(
            f"Tool '{args.name}' authored, sandbox-tested, approved, and registered. "
            f"You may now assign it to nodes."
        )


# ── closed-loop verification (Phase 5) ─────────────────────────────────────────

class TestWorkflowArgs(BaseModel):
    test_inputs: dict | None = Field(
        default=None,
        description="Optional concrete inputs to test with. Omit to auto-derive "
                    "realistic test inputs (and acceptance criteria) from the intent.",
    )


async def _satisfy_requirements(session: BuildSession) -> None:
    """Park and ask for any stored value the staged workflow needs but lacks.

    Best-effort throughout: a package that will not load, or a store that is
    unavailable, must not stop a verification that might otherwise have run —
    this exists to *improve* a run's chances, never to gate it.
    """
    from pathlib import Path

    from neurosurfer.graph.workflow.package import load_package
    from neurosurfer.graph.workflow.requirements import workflow_requirements

    try:
        pkg = load_package(Path(session.staging_root) / session.name)
        missing = [r for r in workflow_requirements(pkg) if not r.satisfied]
    except Exception:  # noqa: BLE001 - never block a run on the checklist
        return
    if not missing:
        return

    names = ", ".join(r.name for r in missing)
    if session.request_secrets is None:
        # Headless: say what is missing, so the failure that follows is legible
        # as a missing credential rather than a mysterious node error.
        session.notify(f"missing stored values ({names}) — verification will run without them")
        return

    session.notify(f"verification needs {names} — asking")
    await session.await_secrets([
        {
            "name": r.name,
            "source": r.source,
            # Why a value that IS set cannot be used, and how to obtain a good
            # one. Asking again without saying why is the worst version of this:
            # a stale hostname sat in a secret called `…_SQLSERVER_URL` and the
            # only visible symptom was a driver error three nodes later.
            "problem": getattr(r, "problem", "") or "",
            "help": getattr(r, "help", "") or "",
        }
        for r in missing
    ])

    # Re-derive rather than trust the answer. Reporting "supplied" for everything
    # asked for was a lie the first time it ran: one of seven fields was left
    # blank, six were stored, and the log claimed all seven — while the next
    # check quietly disagreed. What matters is what is *still* missing.
    try:
        pkg = load_package(Path(session.staging_root) / session.name)
        still = [r.name for r in workflow_requirements(pkg) if not r.satisfied]
    except Exception:  # noqa: BLE001
        return
    session.notify(
        f"still missing {', '.join(still)} — testing without"
        if still else "all required values are set"
    )


class TestWorkflowTool(Tool):
    name = "test_workflow"
    description = (
        "Prove the staged workflow works: derives acceptance criteria + test inputs "
        "from the user's intent, RUNS the workflow on them, and judges the outputs "
        "per criterion. Returns PASSED, or the failures with a diagnosis and "
        "suggested design changes. Changing the design stales the result — re-test "
        "after changes. Calling it again without changing anything returns the same "
        "verdict without re-running, so act on the diagnosis rather than retrying. "
        "Requires the workflow to be structurally valid first."
    )
    input_model = TestWorkflowArgs

    def __init__(self, session: BuildSession) -> None:
        self.session = session

    async def call(self, args: TestWorkflowArgs, ctx: ToolContext) -> ToolResult:
        s = self.session
        if s.provider is None:
            return ToolResult.error("No provider available to run the verification.")
        ok, report = s.validate()
        if not ok:
            return ToolResult.error(
                "Cannot test — the workflow is not structurally valid yet:\n" + report
            )

        from .verify import derive_acceptance, verify_workflow

        # Answer from the last verification when nothing it depended on has moved.
        # Re-running would re-execute every node for an answer we already hold — and
        # on a failure, a fresh roll of the same dice invites the model to retry its
        # way past a verdict instead of fixing the design.
        cached = s.cached_verification(dict(args.test_inputs) if args.test_inputs else None)
        if cached is not None:
            s.notify("verification reused — graph unchanged since the last test")
            note = (
                "(Not re-run: the graph and test inputs are unchanged since this "
                "verification. Change the design, then test again.)\n\n"
            )
            body = note + cached.rendered
            return ToolResult.ok(body) if cached.passed else ToolResult.error(body)

        # Derive (and cache) the acceptance plan; explicit inputs override.
        if s.acceptance_plan is None:
            s.notify("deriving acceptance criteria + test inputs")
            s.acceptance_plan = await derive_acceptance(
                s.provider, s.intent, s.to_yaml(), declared_inputs=s.inputs
            )
        plan = s.acceptance_plan
        if args.test_inputs:
            plan = plan.model_copy(update={"test_inputs": dict(args.test_inputs)})

        # Verification RUNS the workflow, so a step that reaches a real database
        # needs a real connection string. Ask before running rather than after
        # failing: an unmet credential otherwise arrives as every criterion
        # failing at once, which reads like a broken design and is not one.
        await _satisfy_requirements(s)

        s.notify(f"running verification: {s.name}")
        result = await verify_workflow(
            s.provider,
            intent=s.intent,
            package_dir=s.staging_root / s.name,
            plan=plan,
            declared_inputs=s.inputs,
            on_node_event=s.node_event,
        )

        # The test rig failed, not the workflow: no fixture where one was needed,
        # or a fixture script that did not work. Repair it here rather than
        # reporting it — the builder's tools edit graphs, so nothing it can do
        # will help, and a weak model handed an unactionable failure does one of
        # two bad things (both seen on gpt-4o-mini): declares the whole build
        # blocked, or starts adding `setup_fixtures` nodes to the workflow and
        # loops on the validation errors those cause.
        #
        # Once per build, tracked on the session rather than by the shape of this
        # failure, so a second fixture problem cannot re-enter the loop.
        if result.fixture_problem and not s.fixture_retry_used:
            s.fixture_retry_used = True
            s.notify(
                "the test fixtures did not work ("
                + (", ".join(result.missing_fixture_for) or "setup failed")
                + ") — re-deriving the acceptance plan"
            )
            s.acceptance_plan = await derive_acceptance(
                s.provider, s.intent, s.to_yaml(), declared_inputs=s.inputs,
                require_fixture_for=result.missing_fixture_for or [
                    spec["name"] for spec in s.inputs if spec.get("name")
                ],
            )
            plan = s.acceptance_plan
            if args.test_inputs:
                plan = plan.model_copy(update={"test_inputs": dict(args.test_inputs)})
            result = await verify_workflow(
                s.provider,
                intent=s.intent,
                package_dir=s.staging_root / s.name,
                plan=plan,
                declared_inputs=s.inputs,
                on_node_event=s.node_event,
            )

        # Still the rig. Say so unmistakably: the last thing that must happen here
        # is the model deciding the graph is at fault.
        if result.fixture_problem:
            rendered = plan.render() + "\n\n" + result.render()
            s.record_verification(
                passed=False, rendered=rendered, report=result,
                test_inputs=dict(args.test_inputs) if args.test_inputs else None,
            )
            s.notify("verification blocked on the test fixtures, not the workflow")
            return ToolResult.error(
                "The TEST HARNESS could not be set up — this is not a fault in the "
                "workflow, and no change to the graph will fix it. Do NOT add nodes "
                "to create test files, and do NOT redesign the workflow.\n\n"
                + rendered
                + "\n\nThe design is not blocked by this. If the graph is right, "
                "call register_workflow — it will register and record the workflow "
                "as UNVERIFIED. Otherwise call test_workflow again with explicit "
                "`test_inputs` pointing at a file that already exists."
            )

        s.graph_runs += result.graph_runs
        rendered = plan.render() + "\n\n" + result.render()
        s.record_verification(
            passed=result.passed, rendered=rendered, report=result,
            test_inputs=dict(args.test_inputs) if args.test_inputs else None,
        )
        s.notify(
            f"verification {'PASSED' if result.passed else 'FAILED'} "
            f"({result.graph_runs} graph run{'s' if result.graph_runs != 1 else ''}; "
            f"{s.graph_runs} this build)"
        )
        if result.passed:
            return ToolResult.ok(rendered)

        # A judged failure, which is the only kind that counts against the repair
        # budget — a rig problem returned above and is not a verdict on the design.
        s.failed_verifications += 1
        # A failed verification goes back on the error channel so the model treats
        # it as something to fix, not a success to summarise.
        if s.verification_exhausted:
            # Out of attempts. Say what happens next, because the alternative — a
            # model that has run out of road and is not told so — is the state
            # that produced a two-node summariser declared infeasible.
            return ToolResult.error(
                rendered
                + f"\n\nThis is attempt {s.failed_verifications} of "
                f"{s.max_verification_attempts}, and the budget is spent. Do NOT "
                f"declare this blocked: an LLM step cannot be proved free of "
                f"invention, and if a criterion demands a guarantee rather than an "
                f"observable result then the criterion is wrong, not the workflow. "
                f"Call register_workflow — it will register and record the failure "
                f"as a caveat the user can see."
            )
        return ToolResult.error(rendered)


# ── capability resolution + MCP discovery (Phase 2) ─────────────────────────────

class FindCapabilityArgs(BaseModel):
    capability: str = Field(
        description="The capability a step needs, in plain English — e.g. 'read a "
                    "file from disk', 'read an email inbox', 'send a text message'."
    )


class FindCapabilityTool(Tool):
    name = "find_capability"
    description = (
        "Find something that can provide a capability a node needs. Searches the "
        "live tool catalog first, then the official MCP registry, and reports what "
        "each option would cost you (an install, a credential). ALWAYS call this "
        "before assuming a capability is unavailable — do not guess a tool name."
    )
    input_model = FindCapabilityArgs

    def __init__(self, session: BuildSession) -> None:
        self.session = session

    def is_read_only(self, args: BaseModel) -> bool:
        return True

    async def call(self, args: FindCapabilityArgs, ctx: ToolContext) -> ToolResult:
        import asyncio

        from ..capability import resolve_capability

        need = args.capability.strip()
        if not need:
            return ToolResult.error("Say what capability you are looking for.")
        self.session.notify(f"resolving capability: {need}")
        # The registry leg is a blocking HTTP call; keep the loop free.
        resolution = await asyncio.to_thread(resolve_capability, need)
        self.session.record_resolution(resolution)
        self.session.notify(f"  → {resolution.status}")
        return ToolResult.ok(resolution.render())


class ListMcpToolsTool(Tool):
    name = "list_mcp_tools"
    description = (
        "List the tools of every MCP server currently connected. These are "
        "assignable to nodes exactly like built-in tools."
    )
    input_model = _NoArgs

    def __init__(self, session: BuildSession) -> None:
        self.session = session

    def is_read_only(self, args: BaseModel) -> bool:
        return True

    async def call(self, args: _NoArgs, ctx: ToolContext) -> ToolResult:
        import asyncio

        from neurosurfer.mcp import runtime

        statuses = await asyncio.to_thread(runtime.mcp_statuses)
        if not statuses:
            return ToolResult.ok(
                "No MCP server is connected. Use `find_capability` to search the "
                "registry for one, then `install_mcp_server`."
            )
        lines = []
        for status in statuses:
            lines.append(f"- {status.name}: {', '.join(status.tools) or '(no tools)'}")
            if status.error:
                lines.append(f"    error: {status.error}")
        return ToolResult.ok("\n".join(lines))


class InstallMcpServerArgs(BaseModel):
    name: str = Field(
        description="The registry name from find_capability, e.g. "
                    "'io.github.codespar/mcp-twilio'."
    )
    env: dict | None = Field(
        default=None,
        description="Environment variables the server needs (credentials). Only "
                    "supply values the user actually gave you — never invent one.",
    )
    headers: dict | None = Field(
        default=None, description="HTTP headers, for a remote (http) server."
    )


class InstallMcpServerTool(Tool):
    name = "install_mcp_server"
    description = (
        "Install an MCP server from the registry and connect it, so its tools "
        "become assignable. Requires human approval — this runs third-party code "
        "on the user's machine. If the server needs credentials you were not "
        "given, do NOT install it: call declare_blocked and say what is missing."
    )
    input_model = InstallMcpServerArgs

    def __init__(self, session: BuildSession) -> None:
        self.session = session

    async def call(self, args: InstallMcpServerArgs, ctx: ToolContext) -> ToolResult:
        import asyncio

        from neurosurfer.mcp.credentials import blocking, supply_for
        from neurosurfer.mcp.registry import McpRegistryError, credential_requirements
        from neurosurfer.mcp.sources import (
            AuthorizationRequired,
            SourceUnavailable,
            active_source,
        )

        s = self.session
        name = args.name.strip()
        if not name:
            return ToolResult.error("Give the registry name of the server to install.")

        # Already here? Say so and move on. The model asked twice for the same
        # server on a real build — once per capability that needed it — and each
        # ask was a second approval prompt for something already running.
        already = _already_installed(name)
        if already is not None:
            settled = await asyncio.to_thread(_reresolve_waiting_steps, s)
            note = f" That settles: {', '.join(settled)}." if settled else ""
            return ToolResult.ok(
                f"'{already[0]}' is already installed and connected — nothing to "
                f"do. Its tools: {already[1]}.{note}"
            )

        # Through the engine that *found* it. This imported the official client
        # directly, so a server discovered on Smithery 404'd the moment the agent
        # tried to install it — the search was fixed and this was not.
        source = active_source()
        try:
            detail = await asyncio.to_thread(source.detail, name)
        except SourceUnavailable as e:
            return ToolResult.error(f"Cannot install '{name}': {e}")
        except McpRegistryError as e:
            return ToolResult.error(f"Cannot install '{name}': {e}")

        server = detail.get("server", detail) or {}
        # Through the source, not the official reader. The two describe
        # requirements in entirely different shapes — `environmentVariables` /
        # `headers` arrays versus a per-connection JSON Schema — so reading a
        # Smithery entry with the official reader finds nothing required, installs
        # a server with no connection details, and it fails to start with nobody
        # having been asked for anything.
        try:
            required = source.credentials(server)
        except Exception:  # noqa: BLE001 - fall back rather than fail the install
            required = credential_requirements(server)
        supplied = set(args.env or {}) | set(args.headers or {})
        # A credential already on file is not missing. `blocking` consults the
        # account's stored values and the environment, so the agent is not asked
        # to hand over a key the machine is already holding.
        missing = [c for c in blocking(required) if c.name not in supplied]
        if missing:
            # Installing something that cannot start is worse than not installing
            # it: the failure surfaces mid-run as a connection error.
            return ToolResult.error(
                f"'{name}' needs credentials you have not supplied:\n"
                + "\n".join(f"  - {c.render()}" for c in missing)
                + "\n\nIf the user gave you these values, pass them in `env`. "
                "Otherwise call declare_blocked and list exactly these."
            )

        # Anything held is filled in automatically, by the placeholder name the
        # entry actually substitutes on.
        held = supply_for(required)
        body = {
            "env": {**held.get("env", {}), **(args.env or {})},
            "headers": {**held.get("headers", {}), **(args.headers or {})},
        }
        cfg = None
        for attempt in (1, 2):
            try:
                cfg = await asyncio.to_thread(source.install_config, server, body)
                break
            except AuthorizationRequired as e:
                # The third kind of credential, and the only one nobody can hand
                # over here: a consent screen has to be *visited*. Printing the URL
                # in a refusal made the user find it, visit it, and then re-run the
                # whole build behind it. The build already parks whenever it asks
                # anything — so park here too, and retry once they say they are done.
                setup = getattr(e, "setup_url", "") or ""
                if attempt == 2 or not setup:
                    s.notify(f"'{name}' is still not authorised")
                    return ToolResult.error(
                        f"'{name}' needs authorising in a browser and still is not."
                        + (f" The link is {setup}." if setup else "")
                        + " Try another server from the shortlist, or declare_blocked "
                        "including that link."
                    )
                s.notify(f"'{name}' needs authorising in a browser — waiting for you")
                granted = await s.await_authorization(name, setup)
                if not granted:
                    return ToolResult.error(
                        f"'{name}' was not authorised. Pick another server from the "
                        f"shortlist, author a tool, or declare_blocked and include "
                        f"this link so the user can grant access later: {setup}"
                    )
                s.notify(f"'{name}' authorised — retrying the install")
            except McpRegistryError as e:
                return ToolResult.error(f"Cannot install '{name}': {e}")
        if cfg is None:  # pragma: no cover - the loop always sets or returns
            return ToolResult.error(f"Cannot install '{name}'.")

        approved = await s.approve_mcp_install(cfg, server)
        if not approved:
            return ToolResult.error(
                f"Installing '{name}' was declined. Choose another server, "
                "author a tool, or declare_blocked."
            )

        result = await asyncio.to_thread(_install_and_start, cfg)
        if not result[0]:
            # Say so out loud. Only the success path narrated, so a server that was
            # approved and then failed left "(approved)" as the last word on it —
            # indistinguishable in the transcript from one that worked.
            s.notify(f"MCP server '{cfg.name}' would not start: {result[1]}")
            return ToolResult.error(
                f"'{name}' installed but would not start: {result[1]}\n"
                "Usually this means it needs configuration nobody supplied. Try "
                "another server from the shortlist, or declare_blocked naming what "
                "this one wants."
            )

        s.installed_servers.append(cfg.name)
        s.knowledge.refresh()
        s.notify(f"MCP server '{cfg.name}' installed and connected")

        # Close the loop: the steps that were waiting on this are now answerable,
        # and re-asking here means the plan the model reads next says so. Without
        # it the plan still reports `installable` for a capability that has just
        # arrived, and the model either re-installs or gives up.
        settled = await asyncio.to_thread(_reresolve_waiting_steps, s)
        note = f"\nThat settles: {', '.join(settled)}." if settled else ""

        return ToolResult.ok(
            f"Installed and connected '{cfg.name}'. Its tools are now assignable:\n"
            f"  {result[1]}{note}\n"
            "Assign them to nodes with add_node/update_node."
        )


def _already_installed(registry_name: str) -> tuple[str, str] | None:
    """``(local_name, tools)`` if this registry entry is installed and running.

    Matched on the local name an install derives — the last path segment, made
    filesystem-safe — because that is what ends up in `mcp.json`, not the
    qualified registry name the model is holding.
    """
    import re

    from neurosurfer.mcp import runtime

    leaf = registry_name.split("/")[-1]
    candidates = {
        registry_name,
        leaf,
        re.sub(r"[^A-Za-z0-9_.-]+", "-", registry_name).strip("-"),
        re.sub(r"[^A-Za-z0-9_.-]+", "-", leaf).strip("-"),
    }
    for status in runtime.mcp_statuses():
        if status.name in candidates and status.connected:
            return status.name, ", ".join(status.tools) or "(none reported)"
    return None


def _reresolve_waiting_steps(session: BuildSession) -> list[str]:
    """Re-run the ladder for plan steps that were not yet buildable.

    Only the unresolved ones: re-resolving the whole plan would re-search the
    registry for capabilities that were settled long ago, and each of those is a
    multi-second round trip in the middle of a build.

    Returns the step ids that became buildable, for the model to read.
    """
    plan = getattr(session, "plan", None)
    if plan is None:
        return []

    from ..capability import resolve_capability

    settled: list[str] = []
    for step in list(plan.unresolved_steps):
        need = (step.needed_capability or "").strip() or step.intent
        try:
            resolution = resolve_capability(need)
        except Exception:  # noqa: BLE001 - a failed re-check leaves it as it was
            continue
        step.resolution = resolution.to_dict()
        if step.resolved:
            settled.append(step.id)
    return settled


def _install_and_start(cfg) -> tuple[bool, str]:
    """Persist the config, connect, and report the tools. Runs off the loop."""
    from neurosurfer.config.mcp import McpStore
    from neurosurfer.mcp import runtime

    store = McpStore.default()
    fresh = store.get(cfg.name) is None
    if fresh:
        store.add(cfg)
    status = runtime.start_server(cfg)
    if not status.connected:
        # Roll back. Persisting a server that could not start leaves a permanently
        # broken row in `mcp.json` that reads as "installed, stopped" — the user
        # has to work out that it never worked, and delete it by hand.
        if fresh:
            try:
                store.delete(cfg.name)
            except Exception:  # noqa: BLE001 - the start error is the real news
                pass
        return False, status.error or "connection failed"
    # A server that connected is one this deployment now depends on; leaving it
    # disabled would mean the registered workflow cannot reconnect on its own.
    store.set_enabled(cfg.name, True)
    return True, ", ".join(status.tools) or "(the server exposed no tools)"


# ── capability override ─────────────────────────────────────────────────────────

class AcknowledgeCapabilityArgs(BaseModel):
    node_id: str = Field(description="The node the capability warning is about.")
    reason: str = Field(
        description="WHY this node needs no tool — e.g. the data already arrives "
                    "from an upstream node, or the phrasing describes its input "
                    "rather than an action it performs."
    )


class AcknowledgeCapabilityTool(Tool):
    name = "acknowledge_capability"
    description = (
        "Argue past a capability warning on ONE node, when the node genuinely needs "
        "no tool (usually: it processes data an upstream node already fetched). Use "
        "this ONLY when the warning is wrong — if the node really must reach outside "
        "the model, attach a tool, author one, or declare_blocked instead."
    )
    input_model = AcknowledgeCapabilityArgs

    def __init__(self, session: BuildSession) -> None:
        self.session = session

    async def call(self, args: AcknowledgeCapabilityArgs, ctx: ToolContext) -> ToolResult:
        s = self.session
        if s.get_node(args.node_id) is None:
            return ToolResult.error(
                f"No node '{args.node_id}'. Existing: {s.node_ids()}"
            )
        reason = args.reason.strip()
        if not reason:
            return ToolResult.error(
                "A reason is required — this override is recorded and shown to the user."
            )
        s.acknowledged_capabilities[args.node_id] = reason
        s.notify(f"capability warning acknowledged on '{args.node_id}': {reason}")
        return ToolResult.ok(
            f"Recorded. '{args.node_id}' will not block registration. "
            "If it turns out the node cannot produce its result without a tool, "
            "the verification run will show it."
        )


class DropPlanStepArgs(BaseModel):
    step_id: str = Field(description="The plan step you are not building.")
    reason: str = Field(
        description="WHY it should not exist — e.g. another node already covers it, "
                    "or the plan asked for something the user did not."
    )


class DropPlanStepTool(Tool):
    name = "drop_plan_step"
    description = (
        "Record that a planned step will NOT be built, and why. Use this when the "
        "plan was wrong — a step another node already covers, or one the request "
        "never asked for. Every planned step must be built or dropped; a step that "
        "silently vanishes is part of the user's request going missing."
    )
    input_model = DropPlanStepArgs

    def __init__(self, session: BuildSession) -> None:
        self.session = session

    async def call(self, args: DropPlanStepArgs, ctx: ToolContext) -> ToolResult:
        s = self.session
        if s.plan is None or s.plan.step(args.step_id) is None:
            known = [st.id for st in (s.plan.steps if s.plan else [])]
            return ToolResult.error(
                f"No plan step '{args.step_id}'. Steps: {known}"
            )
        reason = args.reason.strip()
        if not reason:
            return ToolResult.error(
                "A reason is required — dropping a planned step is shown to the user."
            )
        s.dropped_steps[args.step_id] = reason
        s.notify(f"plan step dropped: {args.step_id} — {reason}")
        return ToolResult.ok(f"Recorded. '{args.step_id}' will not block registration.")


# ── terminal: blocked ───────────────────────────────────────────────────────────

class BlockedArgs(BaseModel):
    reason: str = Field(
        description="WHY the workflow cannot be built as described, and exactly what "
                    "the user must provide or change to make it possible."
    )


class DeclareBlockedTool(Tool):
    name = "declare_blocked"
    description = (
        "Declare the requested workflow infeasible as described (missing external "
        "resource, capability that cannot be built safely, contradictory request). "
        "This ends the build with a clear report instead of a broken workflow. "
        "NOT for a workflow that merely fails its own verification — a complete, "
        "grounded design is never blocked, and this refuses one."
    )
    input_model = BlockedArgs

    def __init__(self, session: BuildSession) -> None:
        self.session = session

    async def call(self, args: BlockedArgs, ctx: ToolContext) -> ToolResult:
        s = self.session
        # **The gate that keeps a finished design from being thrown away.** See
        # `BuildSession.blocking_is_justified`: this rule lived in the system
        # prompt, and a model that has run out of ideas is exactly the model that
        # stops reading prompts.
        justified, why_not = s.blocking_is_justified()
        if not justified:
            s.notify("refused a declare_blocked — the design is complete and grounded")
            return ToolResult.error(why_not)
        s.blocked_reason = args.reason.strip() or "infeasible (no reason given)"
        s.notify("build declared blocked")
        return ToolResult.ok(
            "Recorded as blocked. Stop now — reply with a short summary for the user."
        )


# ── assembly ────────────────────────────────────────────────────────────────────

def architect_tools(session: BuildSession) -> list[Tool]:
    """The full architect toolbelt bound to *session* (+ web_search if available)."""
    tools: list[Tool] = [
        SetWorkflowTool(session),
        SetOutputsTool(session),
        AddNodeTool(session),
        UpdateNodeTool(session),
        RemoveNodeTool(session),
        ViewWorkflowTool(session),
        ValidateWorkflowTool(session),
        RegisterWorkflowTool(session),
        NeurosurferDocsTool(session),
        DescribeCapabilityTool(session),
        FindCapabilityTool(session),
        ListMcpToolsTool(session),
        InstallMcpServerTool(session),
        AuthorToolTool(session),
        AcknowledgeCapabilityTool(session),
        DropPlanStepTool(session),
        DeclareBlockedTool(session),
    ]
    if session.verification_mode != "off":
        tools.append(TestWorkflowTool(session))
    try:
        from neurosurfer.tools.registry import all_tools

        web = next((t for t in all_tools() if t.name == "web_search"), None)
        if web is not None:
            tools.append(web)
    except Exception:  # noqa: BLE001 - research is optional
        pass
    return tools
