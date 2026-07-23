"""BuildSession — the staged workflow one Architect-agent run is constructing (4a).

All architect tools operate on this shared object: nodes are added/edited in
memory, staged to disk as a real WorkflowPackage for validation, and registered
only when the validation gate passes. The session also records the terminal
outcome (registered path / blocked reason) that :class:`ArchitectAgent` reads
after the loop ends.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

__all__ = ["BuildSession"]


@dataclass
class BuildSession:
    """Mutable state for one architect build."""

    intent: str
    staging_root: Path
    registry: Any                      # WorkflowRegistry
    knowledge: Any                     # KnowledgeBase
    provider: Any = None               # for author_tool
    approve_tool: Any = None           # async (ToolDraft, SandboxResult) -> bool
    notify: Callable[[str], None] = lambda _m: None

    # staged workflow
    name: str = ""
    description: str = ""
    inputs: list[dict[str, Any]] = field(default_factory=list)
    nodes: list[dict[str, Any]] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)

    # terminal outcome
    registered_path: str | None = None
    blocked_reason: str | None = None
    authored_tools: list[str] = field(default_factory=list)

    # closed-loop verification (Phase 5)
    # "off": no test tooling; "encouraged": test available + prompted;
    # "required": register refuses without a passing, non-stale verification.
    verification_mode: str = "encouraged"
    acceptance_plan: Any = None                 # cached AcceptancePlan
    last_verification: tuple[bool, str] | None = None  # (passed, rendered report)

    def invalidate_verification(self) -> None:
        """Any graph edit stales the last verification (it tested a different graph)."""
        self.last_verification = None

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

    def validate(self) -> tuple[bool, str]:
        """Stage and validate the package. Returns (ok, human-readable report)."""
        if not self.nodes:
            return False, "The workflow has no nodes yet."
        if not self.name:
            return False, "The workflow has no name — call set_workflow first."
        from neurosurfer.graph.engine.errors import GraphConfigurationError
        from neurosurfer.graph.engine.loader import load_graph_from_dict

        try:
            load_graph_from_dict(self.graph_dict())
        except GraphConfigurationError as e:
            return False, f"Graph structure invalid:\n{e}"

        from neurosurfer.graph.workflow.package import PackageLoadError, load_package
        from neurosurfer.graph.workflow.validate import validate_package

        try:
            pkg = load_package(self.stage())
        except PackageLoadError as e:
            return False, f"Staged package failed to load: {e}"
        report = validate_package(pkg)
        if report.ok:
            summary = report.summary()
            return True, ("VALID." if summary == "Package is valid."
                          else f"VALID with warnings:\n{summary}")
        return False, report.summary()

    def register(self) -> tuple[bool, str]:
        """Validate, then register into the registry. Returns (ok, message)."""
        ok, report = self.validate()
        if not ok:
            return False, f"Refusing to register — validation failed:\n{report}"
        if self.verification_mode == "required":
            if self.last_verification is None:
                return False, (
                    "Refusing to register — this build requires a passing "
                    "test_workflow verification of the CURRENT graph first."
                )
            if not self.last_verification[0]:
                return False, (
                    "Refusing to register — the last verification FAILED. Fix the "
                    "design and test_workflow again:\n" + self.last_verification[1]
                )
        from neurosurfer.graph.workflow.package import load_package

        pkg = load_package(self.staging_root / self.name)
        dest = self.registry.save(pkg)
        self.registered_path = str(dest)
        self.notify(f"Workflow '{self.name}' registered at {dest}")
        return True, f"Registered at {dest}. The build is complete — you may finish now."
