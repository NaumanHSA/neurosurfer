from __future__ import annotations
from typing import Optional

class GraphError(Exception):
    """Base exception for graph-related issues."""


class GraphConfigurationError(GraphError):
    """Invalid graph spec, missing tools, cycles, etc."""


class GraphExecutionError(GraphError):
    """Errors that occur during graph execution."""


class NodeExecutionError(GraphExecutionError):
    """A single node failed in an unexpected way."""


class ValidationError(GraphError):
    pass

class PlanningError(GraphError):
    pass

class NodeError(GraphError):
    def __init__(self, node_id: str, message: str, *, cause: Optional[BaseException] = None):
        super().__init__(f"[{node_id}] {message}")
        self.node_id = node_id
        self.cause = cause
