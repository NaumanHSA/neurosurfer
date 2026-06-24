"""Shared agent runtime: tool dispatch, permissions/guardrails, structured output,
background tasks."""
from .loop import ToolOutcome, execute_tool_uses  # noqa: F401
from .permissions import (  # noqa: F401
    Decision,
    Guardrails,
    PermissionMode,
    Permissions,
    initial_mode,
)
from .structured import StructuredCompletionError, structured_completion  # noqa: F401
from .tasks_runtime import TaskHandle, TasksRuntime  # noqa: F401
