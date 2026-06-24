"""assemble — function node that builds the final workflow package.

Reads the :class:`WorkflowPlan` (from the ``plan`` node) and the staged agent
YAML files written by the ``write_nodes`` react-node, generates ``workflow.yaml``
and ``graph.yaml`` in the staging directory, validates the package via
:func:`load_package`, then registers it via :class:`WorkflowRegistry`.

Returns the path (as a string) where the registered package lives.

Called by GraphExecutor as a *function* node::

    assemble(user_intent=..., plan=<WorkflowPlan|dict>, write_nodes=...)
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any

import yaml

from neurosurfer.config.projects import ProjectsConfig
from neurosurfer.graph.workflow.package import PackageLoadError, load_package
from neurosurfer.graph.workflow.registry import WorkflowRegistry

from ..schemas import NodePlan, WorkflowPlan


def run(*, plan: Any, **_: Any) -> str:
    """Assemble and register a :class:`WorkflowPackage` from the architect's plan."""
    # ── 1. Normalise plan ────────────────────────────────────────────────────
    if isinstance(plan, dict):
        workflow_plan = WorkflowPlan.model_validate(plan)
    elif isinstance(plan, WorkflowPlan):
        workflow_plan = plan
    else:
        # Raw string output: attempt JSON parse
        import json
        try:
            workflow_plan = WorkflowPlan.model_validate(json.loads(str(plan)))
        except Exception as exc:
            raise ValueError(f"Cannot parse plan output as WorkflowPlan: {exc}") from exc

    # ── 2. Locate staging directory ──────────────────────────────────────────
    projects = ProjectsConfig()
    project_dir = projects.project_dir(workflow_plan.name)
    agents_dir = project_dir / "agents"
    agents_dir.mkdir(parents=True, exist_ok=True)

    # ── 3. Write workflow.yaml (manifest) ────────────────────────────────────
    manifest = {
        "name": workflow_plan.name,
        "version": "0.1.0",
        "description": workflow_plan.description,
        "entrypoint": "graph.yaml",
        "created_by": "architect",
        "created_at": datetime.datetime.now(tz=datetime.UTC).isoformat(),
        "tags": ["generated"],
    }
    (project_dir / "workflow.yaml").write_text(
        yaml.dump(manifest, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    # ── 4. Write graph.yaml ──────────────────────────────────────────────────
    nodes_yaml = _build_nodes_yaml(workflow_plan.nodes, agents_dir)
    graph: dict[str, Any] = {
        "name": workflow_plan.name,
        "description": workflow_plan.description,
        "inputs": [{"name": "query", "type": "string", "required": False}],
        "nodes": nodes_yaml,
        "outputs": workflow_plan.outputs,
    }
    (project_dir / "graph.yaml").write_text(
        yaml.dump(graph, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    # ── 5. Structural validation (loadable?) ─────────────────────────────────
    try:
        pkg = load_package(project_dir)
    except PackageLoadError as exc:
        raise RuntimeError(
            f"Assembled package failed to load: {exc}\n"
            f"Staging dir: {project_dir}"
        ) from exc

    # ── 6. Semantic validation gate (E1) — nothing registers unless it passes ─
    from neurosurfer.graph.workflow.validate import DEFER_MARKER, validate_package

    report = validate_package(pkg)
    if not report.ok:
        # Defer to the ArchitectBuilder, which has the provider + approval UI to author
        # missing tools (gaps) and renders clean errors (hard errors). Returning a
        # marker rather than raising keeps the executor from printing a traceback.
        return f"{DEFER_MARKER}{project_dir}"

    # ── 7. Register ──────────────────────────────────────────────────────────
    registry = WorkflowRegistry()
    dest = registry.save(pkg)
    print(f"\n[Architect] Workflow '{workflow_plan.name}' registered at {dest}")
    return str(dest)


def _build_nodes_yaml(node_plans: list[NodePlan], agents_dir: Path) -> list[dict]:
    """Convert NodePlan list → graph.yaml node dicts.

    If a staged agent YAML exists for a node, its content is merged in; otherwise
    the graph entry is generated directly from the plan fields.

    Tool validity is NOT patched here — unregistered tools are surfaced loudly by
    the validation gate (``validate_package``) so the build fails with an actionable
    report instead of silently dropping a capability a node depends on.
    """
    nodes: list[dict] = []
    for np in node_plans:
        node_dict: dict[str, Any] = {
            "id": np.id,
            "kind": np.kind,
            "depends_on": np.depends_on,
            "mode": np.mode,
        }
        if np.purpose:
            node_dict["purpose"] = np.purpose
        if np.goal:
            node_dict["goal"] = np.goal
        if np.expected_result:
            node_dict["expected_result"] = np.expected_result
        if np.tools:
            node_dict["tools"] = np.tools
        if np.output_schema:
            node_dict["output_schema"] = np.output_schema
        if np.callable:
            node_dict["callable"] = np.callable
        if np.tool_args:
            node_dict["tool_args"] = np.tool_args

        # Merge staged agent YAML if the architect wrote one
        staged = agents_dir / f"{np.id}.yaml"
        if staged.exists():
            try:
                override = yaml.safe_load(staged.read_text(encoding="utf-8")) or {}
                # staged content wins over plan defaults
                node_dict = {**node_dict, **override, "id": np.id}
            except yaml.YAMLError:
                pass  # use plan-derived dict as fallback

        # Normalise invented tool names to real tools (list_files → list_dir, etc.)
        # so common LLM inventions don't trip the gate or spawn redundant generated
        # tools. Genuinely-novel names are left for the validation gate / E6.
        if node_dict.get("tools"):
            from neurosurfer.tools.registry import normalize_tool_names

            tools = node_dict["tools"]
            if isinstance(tools, str):
                tools = [tools]
            normalized = normalize_tool_names(tools)
            if normalized:
                node_dict["tools"] = normalized
            else:
                node_dict.pop("tools", None)

        nodes.append(node_dict)
    return nodes
