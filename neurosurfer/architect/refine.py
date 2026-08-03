"""Runtime repair for registered workflows (V2 Phase E7, folded into V3 Phase 5d).

Phases 1–4 heal a workflow at *build* time, and Phase 5 proves it runs before it is
registered. This is the arm for the failure nobody could have tested: a workflow that
registered on passing evidence and then breaks against the real world — a renamed
file, an API that changed shape, a credential that expired.

**It proposes; it does not quietly patch.** The legacy E7 behaviour was to write
`agents/<id>.yaml` overrides straight into the registered package and re-run, which
edits something in production with nobody's approval. Every other V3 mechanism
produces a reviewable artifact and gates it, so this one does too:
:meth:`WorkflowRefiner.diagnose` returns a :class:`RepairProposal`, and applying it is
a separate, explicit call. `refine(apply=True)` keeps the old loop for the CLI, which
asked for healing and prints what it changed.

Two things it refuses to mistake for a design bug:

* **A failure the graph cannot fix.** A step whose MCP tool is missing or whose
  credentials are unset did not fail because its prompt is wrong. Rewriting the node
  would be worse than useless — it would obscure a missing secret. Such a node gets a
  diagnosis naming what to supply, and no patch.
* **A tool that does not exist.** The doctor is shown the catalog and told to use it;
  it is also *checked*, because "use only tools that exist" is exactly the kind of
  instruction Phase 1 established a model will not reliably follow.

Patches, when applied, go to the package's ``agents/<id>.yaml`` override layer (which
wins the merge in ``load_package``) and every patched package is re-validated through
the full gate — which since Phase 1 includes the capability checks.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import yaml

from neurosurfer.graph.workflow.package import WorkflowPackage, load_package
from neurosurfer.graph.workflow.validate import validate_package
from neurosurfer.llm.base import Provider
from neurosurfer.llm.types import GenerationConfig, Message

# The tolerant JSON extraction the planner and the verifier already share (V3
# Phase 3). This module carried a private near-copy until 5d.
from .agent.jsonio import parse_json as _parse_json

logger = logging.getLogger(__name__)

__all__ = [
    "NodeDiagnosis",
    "RefineResult",
    "RepairProposal",
    "WorkflowRefiner",
    "inert_patch_fields",
]

# Node fields the doctor is allowed to change. Keeps a hallucinated patch from
# injecting arbitrary keys; the validation gate still checks the result.
_PATCHABLE_FIELDS = {
    "instructions",
    "purpose",
    "goal",
    "expected_result",
    "tools",
    "mode",
    "output_schema",
    "depends_on",
    "tool_args",
}

_SYSTEM_PROMPT = """\
You are a workflow node doctor. A single node in a multi-node workflow failed at
runtime. Given the node's config and the error it produced, propose the MINIMAL patch
to its config that would fix the failure.

Output STRICT JSON only, no prose, no code fences:
  {"diagnosis": "<one sentence>", "patch": {<only the node fields you change>}}

You may change only these fields: instructions, purpose, goal, expected_result,
tools, mode, output_schema, depends_on, tool_args. Use ONLY tools that exist (listed below).

Return {"diagnosis": "...", "patch": {}} — a diagnosis and NO patch — when the config
is not what is wrong. Config cannot fix: a missing credential or API key, an MCP
server that is not installed or connected, a network outage, a file that the caller
must supply. Say what needs to be supplied instead of rewriting the node.
"""

Notify = Callable[[str], None]


@dataclass
class NodeDiagnosis:
    """What went wrong at one node, and what (if anything) to change about it."""

    node_id: str
    error: str
    kind: str = ""
    diagnosis: str = ""
    patch: dict[str, Any] = field(default_factory=dict)
    # Set when the failure is not the graph's fault — a missing credential, an
    # absent MCP server. `patch` is always empty in that case, deliberately.
    external_reason: str = ""
    # Tool names the doctor asked for that do not exist, dropped from the patch.
    rejected_tools: list[str] = field(default_factory=list)
    # Fields the doctor tried to change that this node's kind never reads, so
    # changing them could not have altered the outcome.
    inert_fields: list[str] = field(default_factory=list)

    @property
    def actionable(self) -> bool:
        return bool(self.patch)

    @property
    def answered(self) -> bool:
        """True when this node got a real verdict, patch or not.

        "Nothing about this node's config is wrong — the caller passed a path that
        does not exist" *is* an answer, and a useful one. Only a doctor that said
        nothing usable leaves the failure unanswered. Collapsing the two reads as
        the refiner having failed when in fact it diagnosed correctly, which is
        what the first live run of this phase looked like.
        """
        return bool(self.patch or self.external_reason or self.diagnosis)

    def render(self) -> str:
        lines = [f"  • [{self.node_id}] {self.error[:200]}"]
        if self.diagnosis:
            lines.append(f"      diagnosis: {self.diagnosis}")
        if self.external_reason:
            lines.append(f"      NOT a design problem: {self.external_reason}")
        if self.patch:
            lines.append(f"      proposed patch: {', '.join(sorted(self.patch))}")
        elif self.external_reason:
            pass          # already said what to do about it
        elif self.diagnosis:
            lines.append(
                "      no change proposed — the workflow's design is not the fault; "
                "fix the input or the environment it was given"
            )
        else:
            lines.append("      no diagnosis could be read from the doctor")
        if self.rejected_tools:
            lines.append(
                "      dropped (no such tool): " + ", ".join(self.rejected_tools)
            )
        if self.inert_fields:
            lines.append(
                f"      dropped (a `{self.kind}` node never reads these): "
                + ", ".join(self.inert_fields)
            )
        return "\n".join(lines)


@dataclass
class RepairProposal:
    """One run of a registered workflow, and what to do about how it went."""

    workflow: str
    ran: bool
    failed_nodes: list[NodeDiagnosis] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True when the workflow ran without a node failing."""
        return self.ran and not self.failed_nodes

    @property
    def actionable(self) -> bool:
        return any(d.actionable for d in self.failed_nodes)

    @property
    def blocked_on_external(self) -> list[NodeDiagnosis]:
        return [d for d in self.failed_nodes if d.external_reason]

    def render(self) -> str:
        if self.ok:
            return f"'{self.workflow}' ran cleanly — nothing to repair."
        lines = [
            f"'{self.workflow}' failed at {len(self.failed_nodes)} "
            f"node{'s' if len(self.failed_nodes) != 1 else ''}:"
        ]
        lines += [d.render() for d in self.failed_nodes]
        if self.blocked_on_external:
            lines.append(
                "\nSupply what these steps need — no graph change will fix them: "
                + ", ".join(d.node_id for d in self.blocked_on_external)
            )
        unanswered = [d for d in self.failed_nodes if not d.answered]
        if unanswered:
            lines.append(
                "\nNo diagnosis could be produced for: "
                + ", ".join(d.node_id for d in unanswered)
            )
        elif not self.actionable and not self.blocked_on_external:
            lines.append(
                "\nEvery failure was diagnosed and none of them is a design fault — "
                "the workflow is fine; what it was given is not."
            )
        return "\n".join(lines)


@dataclass
class RefineResult:
    ok: bool
    rounds: int
    patched_nodes: list[str] = field(default_factory=list)
    message: str = ""
    # The last proposal considered, so a caller that only wanted a diagnosis can
    # read one out of a healing run too.
    proposal: RepairProposal | None = None


class WorkflowRefiner:
    """Run a registered workflow and diagnose — or, on request, heal — its failures."""

    def __init__(
        self,
        provider: Provider,
        *,
        registry: Any = None,
        runner: Any = None,
        max_rounds: int = 3,
    ) -> None:
        self.provider = provider
        self._registry = registry
        self._runner = runner
        self.max_rounds = max_rounds

    # ── diagnosis (the default: look, do not touch) ───────────────────────────────
    async def diagnose(
        self,
        name: str,
        inputs: dict[str, Any],
        *,
        notify: Notify | None = None,
        pkg: WorkflowPackage | None = None,
    ) -> RepairProposal:
        """Run *name* once and report what failed and what to change. Writes nothing."""
        say = notify or (lambda _msg: None)
        pkg = pkg if pkg is not None else (self._registry or self._default_registry()).get(name)
        runner = self._runner or self._default_runner()

        say(f"running '{name}'…")
        try:
            result = runner.run(pkg, dict(inputs))
        except Exception as e:  # noqa: BLE001 - a workflow that cannot start is a finding
            return RepairProposal(
                workflow=name, ran=False,
                failed_nodes=[NodeDiagnosis(
                    node_id="(workflow)", error=str(e),
                    diagnosis="The workflow could not start at all.",
                )],
            )

        failed = {
            nid: nr for nid, nr in result.nodes.items()
            if nr.error and not getattr(nr, "skipped", False)
        }
        if not failed:
            say("ran cleanly.")
            return RepairProposal(workflow=name, ran=True)

        diagnoses = []
        for nid, nr in failed.items():
            node = next((n for n in pkg.graph.nodes if n.id == nid), None)
            error = str(nr.error)
            say(f"  node '{nid}' failed: {error[:120]}")
            if node is None:
                diagnoses.append(NodeDiagnosis(node_id=nid, error=error))
                continue
            diagnoses.append(await self._diagnose_node(pkg, node, error))
        return RepairProposal(workflow=name, ran=True, failed_nodes=diagnoses)

    # ── healing (opt-in: apply, re-validate, re-run) ──────────────────────────────
    async def refine(
        self,
        name: str,
        inputs: dict[str, Any],
        *,
        notify: Notify | None = None,
        apply: bool = True,
    ) -> RefineResult:
        """Diagnose and, when *apply*, patch failing nodes and re-run up to a cap.

        *apply=False* makes this a one-shot diagnosis wearing the old return type —
        useful for a caller that wants the legacy shape without the side effects.
        """
        say = notify or (lambda _msg: None)
        registry = self._registry or self._default_registry()
        pkg = registry.get(name)

        patched: list[str] = []
        proposal: RepairProposal | None = None

        for round_no in range(1, self.max_rounds + 1):
            say(f"Run {round_no}: executing '{name}'…")
            proposal = await self.diagnose(name, inputs, notify=say, pkg=pkg)
            if proposal.ok:
                return RefineResult(ok=True, rounds=round_no, patched_nodes=patched,
                                    proposal=proposal)
            if not apply:
                return RefineResult(
                    ok=False, rounds=round_no, patched_nodes=patched,
                    message=proposal.render(), proposal=proposal,
                )
            if round_no == self.max_rounds:
                break  # no point patching after the final allowed run

            if proposal.blocked_on_external:
                # Re-running will fail the same way, and no patch exists to try.
                return RefineResult(
                    ok=False, rounds=round_no, patched_nodes=patched,
                    message=proposal.render(), proposal=proposal,
                )
            if not proposal.actionable:
                return RefineResult(
                    ok=False, rounds=round_no, patched_nodes=patched,
                    message=proposal.render(), proposal=proposal,
                )

            for d in proposal.failed_nodes:
                if not d.patch:
                    continue
                self._apply_patch(pkg, d.node_id, d.patch)
                if d.node_id not in patched:
                    patched.append(d.node_id)
                say(f"  patched '{d.node_id}': {', '.join(sorted(d.patch))}")

            # Re-load with the new overrides and re-validate before re-running.
            pkg = load_package(pkg.path)
            report = validate_package(pkg)
            if not report.ok:
                return RefineResult(
                    ok=False, rounds=round_no, patched_nodes=patched,
                    message="Patch produced an invalid workflow:\n" + report.summary(),
                    proposal=proposal,
                )

        return RefineResult(
            ok=False, rounds=self.max_rounds, patched_nodes=patched,
            message=f"Still failing after {self.max_rounds} rounds."
                    + (f"\n{proposal.render()}" if proposal else ""),
            proposal=proposal,
        )

    def apply(self, proposal: RepairProposal, *, pkg: WorkflowPackage | None = None) -> list[str]:
        """Write an approved *proposal*'s patches into the package. Returns node ids."""
        if pkg is None:
            pkg = (self._registry or self._default_registry()).get(proposal.workflow)
        applied = []
        for d in proposal.failed_nodes:
            if d.patch:
                self._apply_patch(pkg, d.node_id, d.patch)
                applied.append(d.node_id)
        return applied

    # ── per-node diagnosis ───────────────────────────────────────────────────────
    async def _diagnose_node(
        self, pkg: WorkflowPackage, node: Any, error: str
    ) -> NodeDiagnosis:
        external = _external_failure(node, error)
        if external:
            # No model call: nothing about this node's config is the problem, and
            # asking for a patch invites one that hides a missing secret.
            return NodeDiagnosis(
                node_id=node.id, error=error, kind=str(node.kind),
                external_reason=external,
                diagnosis="This step could not reach what it needs; its "
                          "configuration is not the problem.",
            )

        from neurosurfer.tools.registry import all_tools, format_workflow_tool_catalog

        node_yaml = yaml.dump(_node_to_dict(node), sort_keys=False, allow_unicode=True)
        prompt = (
            f"Workflow: {pkg.name} — {pkg.description or ''}\n\n"
            f"Failing node id: {node.id} (kind: {node.kind})\n"
            f"Current config:\n{node_yaml}\n"
            f"Runtime error:\n{error}\n\n"
            f"Available tools:\n{format_workflow_tool_catalog()}"
        )
        response = await self.provider.complete(
            messages=[Message.user_text(prompt)],
            system=_SYSTEM_PROMPT,
            tools=[],
            config=GenerationConfig(stream=False),
        )
        data = _parse_json(response.text(), want="diagnosis")
        if not isinstance(data, dict):
            return NodeDiagnosis(node_id=node.id, error=error, kind=str(node.kind),
                                 diagnosis="The doctor's answer could not be read.")

        diagnosis = str(data.get("diagnosis") or "")
        raw = data.get("patch")
        patch = {k: v for k, v in raw.items() if k in _PATCHABLE_FIELDS} \
            if isinstance(raw, dict) else {}

        # A node states its job once. `instructions` supersedes the three fields
        # that came before it, and the engine prefers it outright — so a patch
        # that writes `instructions` onto a node that still carries a `goal`
        # produces a node whose YAML shows two versions of its job and whose run
        # silently uses one. Worse in the other direction: the doctor patching
        # `goal` on a node that already has `instructions` is a repair that
        # changes nothing and reports success.
        #
        # Whichever way the patch goes, it decides. The studio does the same on
        # its own edits — see `instructionsPatch` in studio/src/graph/binding.ts.
        _LEGACY_PROMPT_FIELDS = ("purpose", "goal", "expected_result")
        if patch.get("instructions"):
            for field in _LEGACY_PROMPT_FIELDS:
                patch[field] = None
        elif any(patch.get(f) for f in _LEGACY_PROMPT_FIELDS) and node.instructions:
            patch["instructions"] = None

        # "Use only tools that exist" is an instruction, so check it. A patch that
        # swaps a broken tool for an invented one converts a runtime error into a
        # validation error, one round later and further from the cause.
        rejected: list[str] = []
        if "tools" in patch:
            known = {t.name for t in all_tools()}
            wanted = patch["tools"] if isinstance(patch["tools"], list) else [patch["tools"]]
            keep = [t for t in wanted if t in known]
            rejected = [str(t) for t in wanted if t not in known]
            if keep:
                patch["tools"] = keep
            else:
                patch.pop("tools")

        # A patch to a field this kind never reads cannot have fixed anything, and
        # applying it would write a no-op override into a registered package while
        # reporting a repair.
        inert = inert_patch_fields(node.kind, patch)
        for f in inert:
            patch.pop(f, None)
        return NodeDiagnosis(node_id=node.id, error=error, kind=str(node.kind),
                             diagnosis=diagnosis, patch=patch,
                             rejected_tools=rejected, inert_fields=inert)

    # ── patch application ────────────────────────────────────────────────────────
    def _apply_patch(self, pkg: WorkflowPackage, nid: str, patch: dict) -> None:
        """Write the patch into ``agents/<nid>.yaml`` — the layer that wins the merge."""
        agents_dir = pkg.path / "agents"
        agents_dir.mkdir(parents=True, exist_ok=True)
        override_file = agents_dir / f"{nid}.yaml"

        existing: dict = {}
        if override_file.exists():
            try:
                existing = yaml.safe_load(override_file.read_text(encoding="utf-8")) or {}
            except yaml.YAMLError:
                existing = {}

        merged = {**existing, **patch, "id": nid}
        override_file.write_text(
            yaml.dump(merged, sort_keys=False, allow_unicode=True), encoding="utf-8"
        )

    # ── defaults (deferred imports avoid a heavy import at module load) ───────────
    def _default_registry(self) -> Any:
        from neurosurfer.graph.workflow.registry import WorkflowRegistry

        return WorkflowRegistry()

    def _default_runner(self) -> Any:
        from neurosurfer.graph.workflow.runner import WorkflowRunner

        return WorkflowRunner(self.provider)


# ── a patch that could not possibly have worked ───────────────────────────────────
#
# Fields a node kind never reads. `purpose` / `goal` / `expected_result` /
# `output_schema` are only consulted on the base/react path, which builds a system
# prompt and calls a model; a `tool` node invokes its tool with `tool_args` and a
# `function` node calls a callable, neither going near any of it.
#
# Observed on gpt-4o-mini: asked about `File not found: quarterly_report.txt` on a
# `kind: tool` node, it proposed `goal: "Load the contents of a valid file located at
# the provided file_path."` — prose, on a node that makes no LLM call, in answer to a
# caller having passed a path that does not exist. Applying it would have written an
# override into a registered package and claimed a repair that changes nothing.
_INERT_FIELDS: dict[str, frozenset[str]] = {
    "tool": frozenset({"purpose", "goal", "expected_result", "mode", "output_schema"}),
    "function": frozenset({"purpose", "goal", "expected_result", "mode",
                           "output_schema", "tools", "tool_args"}),
    "python": frozenset({"purpose", "goal", "expected_result", "mode",
                         "output_schema", "tools", "tool_args"}),
}


def inert_patch_fields(kind: str, patch: dict[str, Any]) -> list[str]:
    """Patched fields a *kind* node never reads, so changing them is a no-op."""
    return sorted(set(patch) & _INERT_FIELDS.get(str(kind), frozenset()))


# ── what config cannot fix ────────────────────────────────────────────────────────

# Runtime errors that mean "something outside this graph is missing", not "this
# node is misconfigured". Matched on the error text because that is all a failed
# run leaves behind; the MCP credential case is checked structurally below.
_EXTERNAL_ERROR_MARKS = (
    "not exercised",
    "api key",
    "apikey",
    "unauthorized",
    "401",
    "403",
    "authentication",
    "credential",
    "permission denied",
    "connection refused",
    "connection error",
    "temporary failure in name resolution",
    "timed out",
    "rate limit",
    "429",
)


def _external_failure(node: Any, error: str) -> str:
    """Why this failure is not the node's config, or "" if it might be.

    Reuses Phase 5b's credential check: if the node's MCP server has an unset
    declared secret, that is the answer regardless of how the error reads.
    """
    from neurosurfer.graph.workflow.runner import uncredentialed_reason
    from neurosurfer.tools.registry import all_tools

    tool_map = {t.name: t for t in all_tools()}
    for name in node.tools or []:
        tool = tool_map.get(name)
        if tool is None:
            return (
                f"the tool '{name}' is not available on this machine — install or "
                "connect whatever provides it"
            )
        reason = uncredentialed_reason(tool)
        if reason:
            return f"'{name}' cannot run: {reason}"

    low = error.lower()
    for mark in _EXTERNAL_ERROR_MARKS:
        if mark in low:
            return (
                f"the error reports '{mark}', which is about access or "
                "connectivity rather than this node's configuration"
            )
    return ""


# ── helpers ───────────────────────────────────────────────────────────────────────

def _node_to_dict(node: Any) -> dict:
    keep = (
        "id", "kind", "purpose", "goal", "expected_result",
        "tools", "depends_on", "mode", "output_schema", "callable",
    )
    out: dict[str, Any] = {}
    for k in keep:
        v = getattr(node, k, None)
        if v in (None, [], ""):
            continue
        out[k] = v.value if hasattr(v, "value") else v
    return out
