"""Workflows: the multi-file Workflow *package* abstraction — a persisted,
versioned, runnable DAG layered on the DAG engine.

This package owns the package format, registry, runner, validation and schema
(``package``, ``registry``, ``runner``, ``validate``, ``schema``, ``node_tool``).
The conversational graph-builder is a separate feature: :mod:`neurosurfer.graph.builder`.
The DAG engine lives under :mod:`neurosurfer.graph`.

See ``RESTRUCTURE_PLAN.md`` at the repo root for the design and phased roadmap.
"""

__version__ = "0.1.1"
from . import node_tool  # noqa: F401 — registers write_workflow_node into the tool registry
from .package import WorkflowPackage, load_package, save_package
from .registry import WorkflowNotFoundError, WorkflowRegistry
from .schema import WorkflowManifest


def __getattr__(name: str):
    if name in {"WorkflowRunner", "run_workflow"}:
        import importlib
        mod = importlib.import_module(f"{__name__}.runner")
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "__version__",
    "WorkflowManifest",
    "WorkflowPackage",
    "WorkflowRegistry",
    "WorkflowNotFoundError",
    "WorkflowRunner",
    "load_package",
    "save_package",
    "run_workflow",
]
