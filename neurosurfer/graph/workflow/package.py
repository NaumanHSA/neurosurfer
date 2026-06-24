"""WorkflowPackage — multi-file workflow package loader and saver.

On-disk layout::

    <name>/
      workflow.yaml   ← manifest (name, version, description, entrypoint, …)
      graph.yaml      ← graph spec (nodes, edges, outputs)
      agents/<id>.yaml ← optional per-agent node overrides (merged into graph nodes)
      nodes/<id>.py   ← python callables for function nodes
      schemas.py      ← pydantic output-schema classes referenced by nodes
      README.md       ← auto-generated docs

The loader reads ``workflow.yaml`` + the graph file named by ``entrypoint``.
Per-agent overrides in ``agents/`` are merged into matching graph nodes at load time.
The result is an in-memory :class:`WorkflowPackage` that callers can pass to
:class:`~neurosurfer.graph.workflow.runner.WorkflowRunner` or the registry.
"""

from __future__ import annotations

import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from neurosurfer.graph.engine import load_graph_from_dict

from .schema import Graph, WorkflowManifest

__all__ = [
    "WorkflowPackage",
    "load_package",
    "save_package",
    "PackageLoadError",
]


class PackageLoadError(Exception):
    """Raised when a workflow package directory cannot be loaded."""


@dataclass
class WorkflowPackage:
    """In-memory representation of a multi-file workflow package."""

    manifest: WorkflowManifest
    graph: Graph
    path: Path

    @property
    def name(self) -> str:
        return self.manifest.name

    @property
    def description(self) -> str:
        return self.manifest.description

    @property
    def version(self) -> str:
        return self.manifest.version


# ── loader ────────────────────────────────────────────────────────────────────

def load_package(path: Path | str) -> WorkflowPackage:
    """Load a :class:`WorkflowPackage` from a directory on disk.

    Steps:
    1. Read ``workflow.yaml`` → :class:`WorkflowManifest`.
    2. Read the graph file named by ``manifest.entrypoint`` (default ``graph.yaml``).
    3. Merge any per-agent overrides from ``agents/<id>.yaml`` into matching nodes.
    4. Return a :class:`WorkflowPackage`.
    """
    path = Path(path)
    if not path.is_dir():
        raise PackageLoadError(f"Workflow package directory not found: {path}")

    # ── manifest ──────────────────────────────────────────────────────────────
    manifest_file = path / "workflow.yaml"
    if not manifest_file.exists():
        raise PackageLoadError(
            f"Missing 'workflow.yaml' in package directory: {path}"
        )
    try:
        raw_manifest: dict = yaml.safe_load(manifest_file.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise PackageLoadError(f"Cannot parse workflow.yaml: {exc}") from exc

    try:
        manifest = WorkflowManifest.from_dict(raw_manifest)
    except (KeyError, TypeError) as exc:
        raise PackageLoadError(f"Invalid workflow.yaml — {exc}") from exc

    # ── graph ─────────────────────────────────────────────────────────────────
    graph_file = path / manifest.entrypoint
    if not graph_file.exists():
        raise PackageLoadError(
            f"Graph file '{manifest.entrypoint}' not found in package: {path}"
        )
    try:
        raw_graph: dict = yaml.safe_load(graph_file.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise PackageLoadError(f"Cannot parse {manifest.entrypoint}: {exc}") from exc

    # ── per-agent overrides ───────────────────────────────────────────────────
    agents_dir = path / "agents"
    if agents_dir.is_dir():
        raw_graph = _merge_agent_overrides(raw_graph, agents_dir)

    try:
        graph = load_graph_from_dict(raw_graph)
    except Exception as exc:
        raise PackageLoadError(f"Invalid graph spec — {exc}") from exc

    return WorkflowPackage(manifest=manifest, graph=graph, path=path)


def _merge_agent_overrides(raw_graph: dict, agents_dir: Path) -> dict:
    """Merge per-agent YAML files into matching node dicts in the raw graph spec."""
    nodes: list[dict[str, Any]] = raw_graph.get("nodes", [])
    overrides: dict[str, dict] = {}
    for yaml_file in agents_dir.glob("*.yaml"):
        try:
            data = yaml.safe_load(yaml_file.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError:
            continue
        node_id = data.get("id") or yaml_file.stem
        overrides[node_id] = data

    merged_nodes = []
    for node in nodes:
        nid = node.get("id")
        if nid and nid in overrides:
            merged = {**node, **overrides[nid]}
        else:
            merged = node
        merged_nodes.append(merged)

    return {**raw_graph, "nodes": merged_nodes}


# ── saver ─────────────────────────────────────────────────────────────────────

def save_package(pkg: WorkflowPackage, dest: Path | str) -> Path:
    """Copy a :class:`WorkflowPackage` directory tree to *dest*.

    If *dest* already exists it is overwritten in place
    (``shutil.copytree`` with ``dirs_exist_ok=True``).
    Returns the destination path.
    """
    dest = Path(dest)
    if pkg.path.resolve() == dest.resolve():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(pkg.path, dest, dirs_exist_ok=True)
    return dest


# ── sys.path helper used by the runner ───────────────────────────────────────

class _PackagePathContext:
    """Context manager: temporarily prepend the package dir to ``sys.path``.

    This lets function nodes import from ``nodes/<id>.py`` via plain
    ``import nodes.<id>`` without requiring the user to install the package.
    """

    def __init__(self, pkg: WorkflowPackage) -> None:
        self._path = str(pkg.path)
        self._added = False

    def __enter__(self) -> _PackagePathContext:
        if self._path not in sys.path:
            sys.path.insert(0, self._path)
            self._added = True
        return self

    def __exit__(self, *_: object) -> None:
        if self._added and self._path in sys.path:
            sys.path.remove(self._path)
