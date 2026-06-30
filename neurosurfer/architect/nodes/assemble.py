"""assemble — function node that builds the final workflow package.

Reads the final :class:`WorkflowPlan` (from the ``critique`` node), the
:class:`CapabilityPlan` (from the ``tool_design`` node), and the staged agent
YAML files written by the ``write_nodes`` react-node. It then:

  • gates on feasibility — if tool_design judged any node infeasible, it returns
    the ``INFEASIBLE_MARKER`` instead of registering a broken workflow;
  • applies the CapabilityPlan's authoritative tool assignments + any user-supplied
    workflow inputs (e.g. a connection string);
  • generates ``workflow.yaml`` / ``graph.yaml``, validates via :func:`load_package`,
    and registers via :class:`WorkflowRegistry` (or defers to the builder to author
    any new tools the plan needs).

Returns the registered package path, or a marker the ArchitectBuilder finalizes.

Called by GraphExecutor as a *function* node::

    assemble(critique=<WorkflowPlan>, tool_design=<CapabilityPlan>, write_nodes=...)
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any

import yaml

from neurosurfer.config.projects import ProjectsConfig
from neurosurfer.graph.workflow.package import PackageLoadError, load_package
from neurosurfer.graph.workflow.registry import WorkflowRegistry

from ..schemas import CapabilityPlan, NodePlan, WorkflowPlan


def run(
    *, critique: Any = None, plan: Any = None, tool_design: Any = None, **_: Any
) -> str:
    """Assemble and register a :class:`WorkflowPackage` from the architect's plan.

    Prefers the ``critique`` kwarg (output of the critique node, which holds the
    final improved WorkflowPlan); falls back to ``plan`` for backwards compatibility.
    ``tool_design`` is the :class:`CapabilityPlan` (per-node tool decisions); when
    present it drives the feasibility gate, authoritative tool assignment, and any
    user-supplied workflow inputs.
    """
    raw_plan = critique if critique is not None else plan
    # ── 1. Normalise plan ────────────────────────────────────────────────────
    if isinstance(raw_plan, dict):
        workflow_plan = WorkflowPlan.model_validate(raw_plan)
    elif isinstance(raw_plan, WorkflowPlan):
        workflow_plan = raw_plan
    else:
        # Raw string output: attempt JSON parse
        import json
        try:
            workflow_plan = WorkflowPlan.model_validate(json.loads(str(raw_plan)))
        except Exception as exc:
            raise ValueError(f"Cannot parse plan output as WorkflowPlan: {exc}") from exc

    cap_plan = _parse_capability_plan(tool_design)

    # ── 1b. Feasibility gate ─────────────────────────────────────────────────
    # If tool_design judged any node infeasible, do NOT register a broken workflow.
    # Return a marker so the ArchitectBuilder renders a clean "not doable" message.
    if cap_plan is not None and not cap_plan.feasible:
        from neurosurfer.graph.workflow.validate import INFEASIBLE_MARKER
        return f"{INFEASIBLE_MARKER}{_render_blockers(cap_plan)}"

    # ── 1c. Apply capability decisions to the plan ───────────────────────────
    tool_overrides, declared_inputs, goal_suffixes = _capability_overrides(cap_plan)
    for np in workflow_plan.nodes:
        if np.id in goal_suffixes:
            np.goal = (np.goal or "") + goal_suffixes[np.id]

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
    nodes_yaml = _build_nodes_yaml(workflow_plan.nodes, agents_dir, tool_overrides)
    inputs: list[dict[str, Any]] = [{"name": "query", "type": "string", "required": False}]
    # User-supplied run-time inputs declared by authored tools (e.g. connection_string).
    for name in declared_inputs:
        if name != "query":
            inputs.append({"name": name, "type": "string", "required": True})
    graph: dict[str, Any] = {
        "name": workflow_plan.name,
        "description": workflow_plan.description,
        "inputs": inputs,
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


def _build_nodes_yaml(
    node_plans: list[NodePlan],
    agents_dir: Path,
    tool_overrides: dict[str, list[str]] | None = None,
) -> list[dict]:
    """Convert NodePlan list → graph.yaml node dicts.

    If a staged agent YAML exists for a node, its content is merged in; otherwise
    the graph entry is generated directly from the plan fields.

    Tool validity is NOT patched here — unregistered tools are surfaced loudly by
    the validation gate (``validate_package``) so the build fails with an actionable
    report instead of silently dropping a capability a node depends on.

    When ``tool_overrides`` is given (from the CapabilityPlan), it is the
    *authoritative* final word on a node's tools — applied AFTER the staged-YAML
    merge so the validated capability decision can't be undone by write_nodes.
    """
    tool_overrides = tool_overrides or {}
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

        # Authoritative override from the CapabilityPlan (applied last). These names
        # are already final — existing tools are pre-normalised by the caller and new
        # authored-tool names are intentionally left intact for the validation gate.
        if np.id in tool_overrides:
            final = tool_overrides[np.id]
            if final:
                node_dict["tools"] = final
            else:
                node_dict.pop("tools", None)

        # A node that ends up with tools MUST be 'react' — only react nodes call tools.
        # (critique/write_nodes may have left it 'base'; the capability decision wins.)
        if node_dict.get("tools") and node_dict.get("kind") == "base":
            node_dict["kind"] = "react"

        nodes.append(node_dict)
    return nodes


# ── capability-plan helpers ──────────────────────────────────────────────────────

def _parse_capability_plan(tool_design: Any) -> CapabilityPlan | None:
    """Coerce the tool_design node output into a :class:`CapabilityPlan` (or None)."""
    if tool_design is None:
        return None
    if isinstance(tool_design, CapabilityPlan):
        return tool_design
    if isinstance(tool_design, dict):
        try:
            return CapabilityPlan.model_validate(tool_design)
        except Exception:  # noqa: BLE001 - a malformed plan must not crash the build
            return None
    # Raw string (e.g. JSON-mode text output): best-effort parse.
    import json
    try:
        return CapabilityPlan.model_validate(json.loads(str(tool_design)))
    except Exception:  # noqa: BLE001
        return None


def _input_token(raw: str) -> str:
    """Extract a clean snake_case identifier from a 'name: description' input spec."""
    import re
    name = str(raw).split(":", 1)[0].strip()
    name = re.sub(r"[^a-z0-9_]", "_", name.lower())
    name = re.sub(r"_+", "_", name).strip("_")
    return name


def _capability_overrides(
    cap_plan: CapabilityPlan | None,
) -> tuple[dict[str, list[str]], list[str], dict[str, str]]:
    """Derive (tool_overrides, declared_inputs, goal_suffixes) from a CapabilityPlan.

    - tool_overrides: node_id → final authoritative tool list (existing + authored).
    - declared_inputs: de-duplicated user-supplied workflow input names.
    - goal_suffixes: node_id → text appended to the node goal so the run-time LLM
      knows which workflow inputs to pass to its authored tool (via ``{name}``).
    """
    tool_overrides: dict[str, list[str]] = {}
    declared_inputs: list[str] = []
    goal_suffixes: dict[str, str] = {}
    if cap_plan is None:
        return tool_overrides, declared_inputs, goal_suffixes

    from neurosurfer.tools.registry import normalize_tool_names

    for nc in cap_plan.nodes:
        if nc.decision == "infeasible":
            continue
        existing = normalize_tool_names(nc.assigned_tools) if nc.assigned_tools else []
        new_names = [s.name for s in nc.new_tools if s.name]
        final = list(existing)
        for n in new_names:
            if n not in final:
                final.append(n)
        if final:
            tool_overrides[nc.node_id] = final

        # Collect any user-supplied run-time inputs declared by this node's new tools.
        node_inputs: list[str] = []
        for spec in nc.new_tools:
            for wi in spec.workflow_inputs:
                token = _input_token(wi)
                if token and token != "query":
                    node_inputs.append(token)
                    if token not in declared_inputs:
                        declared_inputs.append(token)
        if node_inputs:
            uniq = list(dict.fromkeys(node_inputs))
            hint = ", ".join(f"{n} = {{{n}}}" for n in uniq)
            goal_suffixes[nc.node_id] = (
                f"\n\nRun-time inputs provided to this workflow (pass to the tool as "
                f"needed): {hint}."
            )

    return tool_overrides, declared_inputs, goal_suffixes


def _render_blockers(cap_plan: CapabilityPlan) -> str:
    """Render a human-readable feasibility report from a CapabilityPlan's blockers."""
    lines: list[str] = []
    for b in cap_plan.blockers:
        lines.append(f"  • {b}")
    if cap_plan.notes:
        lines.append("")
        lines.append(cap_plan.notes)
    return "\n".join(lines) if lines else "The requested workflow can't be built as described."
