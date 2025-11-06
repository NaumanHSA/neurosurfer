# neurosurfer/agents/graph/errors.py
from typing import Optional

class GraphError(Exception):
    pass

class ValidationError(GraphError):
    pass

class PlanningError(GraphError):
    pass

class NodeError(GraphError):
    def __init__(self, node_id: str, message: str, *, cause: Optional[BaseException] = None):
        super().__init__(f"[{node_id}] {message}")
        self.node_id = node_id
        self.cause = cause
