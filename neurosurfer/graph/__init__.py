"""Graph orchestration domain — the framework's agentic-workflow subsystem.

Three layers, each a submodule:

- :mod:`~neurosurfer.graph.engine`   — the DAG engine (``Graph``, ``GraphExecutor``,
  ``GraphNode``, loader, errors, node runner). A standalone core primitive, the
  LangGraph analog; re-exported here so ``from neurosurfer.graph import Graph`` works.
- :mod:`~neurosurfer.graph.workflow` — the persisted **Workflow package** layer
  (load / validate / register / run a graph saved as a multi-file package).
- :mod:`~neurosurfer.graph.builder`  — the conversational **Architect** that
  generates workflow packages from plain-English intent.

The engine primitives are re-exported at this package top for convenience. The
``workflow`` and ``builder`` subpackages are **not** imported eagerly — importing the
engine must never pull in the LLM / conversation stack. Import them explicitly
(``from neurosurfer.graph.workflow import WorkflowRunner``) or via attribute access
(``neurosurfer.graph.builder``), which is resolved lazily below.
"""
import importlib
from typing import Any

from .engine import (  # noqa: F401
    _VALID_NODE_KINDS,
    AgentError,
    ArtifactStore,
    CodeExecutionError,
    Graph,
    GraphConfigurationError,
    GraphError,
    GraphExecutionError,
    GraphExecutionResult,
    GraphExecutor,
    GraphInput,
    GraphNode,
    InputValidationError,
    ManagerAgent,
    ManagerConfig,
    NeurosurferError,
    NodeError,
    NodeExecutionError,
    NodeExecutionResult,
    NodeFailedError,
    NodeMode,
    NodePolicy,
    NodeSkippedError,
    NodeTimeoutError,
    StructuredOutputError,
    ToolError,
    ToolExecutionError,
    ToolInputError,
    ToolNotFoundError,
    ValidationError,
    _topo_layers,
    import_string,
    load_graph,
    load_graph_from_dict,
    topo_sort,
)


def __getattr__(name: str) -> Any:
    # Lazy access to the feature subpackages so importing the engine stays cheap
    # and free of the LLM/conversation stack.
    if name in {"workflow", "builder", "engine"}:
        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # ── engine primitives (re-exported) ──
    "AgentError",
    "ArtifactStore",
    "CodeExecutionError",
    "Graph",
    "GraphConfigurationError",
    "GraphError",
    "GraphExecutionError",
    "GraphExecutionResult",
    "GraphExecutor",
    "GraphInput",
    "GraphNode",
    "InputValidationError",
    "ManagerAgent",
    "ManagerConfig",
    "NeurosurferError",
    "NodeError",
    "NodeExecutionError",
    "NodeExecutionResult",
    "NodeFailedError",
    "NodeMode",
    "NodePolicy",
    "NodeSkippedError",
    "NodeTimeoutError",
    "StructuredOutputError",
    "ToolError",
    "ToolExecutionError",
    "ToolInputError",
    "ToolNotFoundError",
    "ValidationError",
    "_VALID_NODE_KINDS",
    "_topo_layers",
    "import_string",
    "load_graph",
    "load_graph_from_dict",
    "topo_sort",
    # ── feature subpackages (lazy) ──
    "engine",
    "workflow",
    "builder",
]
