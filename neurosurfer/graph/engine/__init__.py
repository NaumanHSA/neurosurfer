"""The DAG graph engine — a standalone core primitive (the LangGraph analog).

Re-exported at :mod:`neurosurfer.graph` for convenience. Depends only on other
core primitives (agents / llm / tools / tracing); never on the workflow or builder
features layered on top of it.
"""
# ── errors ────────────────────────────────────────────────────────────────────
# ── misc ──────────────────────────────────────────────────────────────────────
from .artifacts import ArtifactStore  # noqa: F401

# ── control flow (Phase 1) ─────────────────────────────────────────────────────
from .builder import GraphBuilder  # noqa: F401
from .errors import (  # noqa: F401
    AgentError,
    CodeExecutionError,
    GraphConfigurationError,
    GraphError,
    GraphExecutionError,
    InputValidationError,
    NeurosurferError,
    NodeError,
    NodeExecutionError,
    NodeFailedError,
    NodeSkippedError,
    NodeTimeoutError,
    StructuredOutputError,
    ToolError,
    ToolExecutionError,
    ToolInputError,
    ToolNotFoundError,
    ValidationError,
)

# ── executor + topo helper ────────────────────────────────────────────────────
from .executor import (  # noqa: F401
    GraphExecutor,
    _topo_layers,
)
from .expressions import ExpressionError, evaluate, safe_bool  # noqa: F401

# ── loader ────────────────────────────────────────────────────────────────────
from .loader import (  # noqa: F401
    load_graph,
    load_graph_from_dict,
)
from .manager import ManagerAgent, ManagerConfig  # noqa: F401

# ── schema ────────────────────────────────────────────────────────────────────
from .schema import (  # noqa: F401
    _VALID_NODE_KINDS,
    Graph,
    GraphExecutionResult,
    GraphInput,
    GraphNode,
    NodeExecutionResult,
    NodeMode,
    NodePolicy,
    RouterCase,
)
from .state import WorkflowState  # noqa: F401
from .utils import import_string, topo_sort  # noqa: F401

__all__ = [
    # errors
    "AgentError",
    "CodeExecutionError",
    "GraphConfigurationError",
    "GraphError",
    "GraphExecutionError",
    "InputValidationError",
    "NeurosurferError",
    "NodeError",
    "NodeExecutionError",
    "NodeFailedError",
    "NodeSkippedError",
    "NodeTimeoutError",
    "StructuredOutputError",
    "ToolError",
    "ToolExecutionError",
    "ToolInputError",
    "ToolNotFoundError",
    "ValidationError",
    # schema
    "_VALID_NODE_KINDS",
    "Graph",
    "GraphExecutionResult",
    "GraphInput",
    "GraphNode",
    "NodeExecutionResult",
    "NodeMode",
    "NodePolicy",
    "RouterCase",
    # control flow
    "WorkflowState",
    "ExpressionError",
    "evaluate",
    "safe_bool",
    "GraphBuilder",
    # executor
    "GraphExecutor",
    "_topo_layers",
    # loader
    "load_graph",
    "load_graph_from_dict",
    # misc
    "ArtifactStore",
    "ManagerAgent",
    "ManagerConfig",
    "import_string",
    "topo_sort",
]
