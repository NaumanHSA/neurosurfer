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
        s.invalidate_verification()
        s.notify(f"workflow meta set: {name}")
        return ToolResult.ok(f"Workflow meta set. name={name!r}, "
                             f"inputs={[i.get('name') for i in s.inputs]}, "
                             f"outputs={s.outputs}")


# ── node construction ───────────────────────────────────────────────────────────

class AddNodeArgs(BaseModel):
    node: dict = Field(
        description="Complete node spec: {id, kind, purpose/goal, depends_on, tools, "
                    "when, writes, cases/default (router), body/max_iterations/"
                    "break_when (loop), body/over/as (map), callable (function), …}. "
                    "Same schema as a graph.yaml node."
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
        return _put_node(self.session, args.node, replace=False)


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
            return ToolResult.error(
                f"No node '{args.id}'. Existing: {self.session.node_ids()}"
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
        self.session.invalidate_verification()
        return ToolResult.ok(f"Removed '{args.id}'. Remaining: {self.session.node_ids()}")


def _put_node(session: BuildSession, spec: dict, *, replace: bool) -> ToolResult:
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
    session.invalidate_verification()
    session.notify(f"node {'updated' if exists else 'added'}: {nid} ({node.kind})")

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
        ok, msg = self.session.register()
        return ToolResult.ok(msg) if ok else ToolResult.error(msg)


# ── knowledge / research ────────────────────────────────────────────────────────

class DocsArgs(BaseModel):
    query: str = Field(description="What to look up in the neurosurfer docs.")
    k: int = Field(default=4, description="Max sections to return.")


class NeurosurferDocsTool(Tool):
    name = "neurosurfer_docs"
    description = "Search the neurosurfer documentation for how a feature/construct works."
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


class TestWorkflowTool(Tool):
    name = "test_workflow"
    description = (
        "Prove the staged workflow works: derives acceptance criteria + test inputs "
        "from the user's intent, RUNS the workflow on them, and judges the outputs "
        "per criterion. Returns PASSED, or the failures with a diagnosis and "
        "suggested design changes. Any graph edit stales the result — re-test after "
        "changes. Requires the workflow to be structurally valid first."
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

        # Derive (and cache) the acceptance plan; explicit inputs override.
        if s.acceptance_plan is None:
            s.notify("deriving acceptance criteria + test inputs")
            s.acceptance_plan = await derive_acceptance(
                s.provider, s.intent, s.to_yaml(), declared_inputs=s.inputs
            )
        plan = s.acceptance_plan
        if args.test_inputs:
            plan = plan.model_copy(update={"test_inputs": dict(args.test_inputs)})

        s.notify(f"running verification: {s.name}")
        result = await verify_workflow(
            s.provider,
            intent=s.intent,
            package_dir=s.staging_root / s.name,
            plan=plan,
        )
        rendered = plan.render() + "\n\n" + result.render()
        s.last_verification = (result.passed, rendered)
        s.notify(f"verification {'PASSED' if result.passed else 'FAILED'}")
        if result.passed:
            return ToolResult.ok(rendered)
        # A failed verification goes back on the error channel so the model treats
        # it as something to fix, not a success to summarise.
        return ToolResult.error(rendered)


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
        "This ends the build with a clear report instead of a broken workflow."
    )
    input_model = BlockedArgs

    def __init__(self, session: BuildSession) -> None:
        self.session = session

    async def call(self, args: BlockedArgs, ctx: ToolContext) -> ToolResult:
        self.session.blocked_reason = args.reason.strip() or "infeasible (no reason given)"
        self.session.notify("build declared blocked")
        return ToolResult.ok(
            "Recorded as blocked. Stop now — reply with a short summary for the user."
        )


# ── assembly ────────────────────────────────────────────────────────────────────

def architect_tools(session: BuildSession) -> list[Tool]:
    """The full architect toolbelt bound to *session* (+ web_search if available)."""
    tools: list[Tool] = [
        SetWorkflowTool(session),
        AddNodeTool(session),
        UpdateNodeTool(session),
        RemoveNodeTool(session),
        ViewWorkflowTool(session),
        ValidateWorkflowTool(session),
        RegisterWorkflowTool(session),
        NeurosurferDocsTool(session),
        DescribeCapabilityTool(session),
        AuthorToolTool(session),
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
