"""Self-healing refinement loop for registered workflows (Phase E7).

E1–E6 heal a workflow at *build* time (before it is registered). E7 heals a workflow
that registered fine but **fails when actually run**: it executes the workflow, reads
which node errored and why, asks a "node-doctor" LLM for a minimal patch to that node's
config, applies it, re-validates, and re-runs — up to a capped number of rounds.

Patches are written to the package's ``agents/<id>.yaml`` override layer (which wins the
merge in ``load_package``), so a fix always takes effect regardless of ``graph.yaml``.
Every patched package is re-validated through the E1 gate before it is re-run.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import yaml

from neurosurfer.graph.workflow.package import WorkflowPackage, load_package
from neurosurfer.graph.workflow.validate import validate_package
from neurosurfer.llm.base import Provider
from neurosurfer.llm.types import GenerationConfig, Message

logger = logging.getLogger(__name__)

# Node fields the doctor is allowed to change. Keeps a hallucinated patch from
# injecting arbitrary keys; the E1 gate still validates the result.
_PATCHABLE_FIELDS = {
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

You may change only these fields: purpose, goal, expected_result, tools, mode,
output_schema, depends_on, tool_args. Use ONLY tools that exist (listed below). If you
cannot fix it from the config alone, return {"diagnosis": "...", "patch": {}}.
"""

Notify = Callable[[str], None]


@dataclass
class RefineResult:
    ok: bool
    rounds: int
    patched_nodes: list[str] = field(default_factory=list)
    message: str = ""


class WorkflowRefiner:
    """Run a registered workflow and heal failing nodes from their run errors."""

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

    async def refine(
        self,
        name: str,
        inputs: dict[str, Any],
        *,
        notify: Notify | None = None,
    ) -> RefineResult:
        say = notify or (lambda _msg: None)

        registry = self._registry or self._default_registry()
        pkg = registry.get(name)
        runner = self._runner or self._default_runner()

        patched: list[str] = []
        for round_no in range(1, self.max_rounds + 1):
            say(f"Run {round_no}: executing '{name}'…")
            result = runner.run(pkg, inputs)
            failed = {
                nid: nr
                for nid, nr in result.nodes.items()
                if nr.error and not getattr(nr, "skipped", False)
            }
            if not failed:
                say("Workflow ran cleanly.")
                return RefineResult(ok=True, rounds=round_no, patched_nodes=patched)

            if round_no == self.max_rounds:
                break  # no point patching after the final allowed run

            # Heal each genuinely-failed node (skipped downstream nodes self-resolve).
            healed_any = False
            for nid, nr in failed.items():
                node = next((n for n in pkg.graph.nodes if n.id == nid), None)
                if node is None:
                    continue
                say(f"  node '{nid}' failed: {str(nr.error)[:120]}")
                patch = await self._diagnose(pkg, node, str(nr.error))
                if not patch:
                    say(f"  no patch proposed for '{nid}'.")
                    continue
                self._apply_patch(pkg, nid, patch)
                if nid not in patched:
                    patched.append(nid)
                healed_any = True
                say(f"  patched '{nid}': {', '.join(patch)}")

            if not healed_any:
                return RefineResult(
                    ok=False, rounds=round_no, patched_nodes=patched,
                    message="No node could be patched from its error.",
                )

            # Re-load with the new overrides and re-validate before re-running.
            pkg = load_package(pkg.path)
            report = validate_package(pkg)
            if not report.ok:
                return RefineResult(
                    ok=False, rounds=round_no, patched_nodes=patched,
                    message="Patch produced an invalid workflow:\n" + report.summary(),
                )

        return RefineResult(
            ok=False, rounds=self.max_rounds, patched_nodes=patched,
            message=f"Still failing after {self.max_rounds} rounds.",
        )

    # ── diagnosis ────────────────────────────────────────────────────────────────
    async def _diagnose(self, pkg: WorkflowPackage, node: Any, error: str) -> dict | None:
        from neurosurfer.tools.registry import format_workflow_tool_catalog

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
        data = _parse_json(response.text())
        if not isinstance(data, dict):
            return None
        patch = data.get("patch")
        if not isinstance(patch, dict):
            return None
        # Keep only known, changeable fields.
        clean = {k: v for k, v in patch.items() if k in _PATCHABLE_FIELDS}
        return clean or None

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


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_json(text: str) -> Any:
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = _JSON_RE.search(text)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    return None
