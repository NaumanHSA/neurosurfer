from __future__ import annotations

from typing import Any

# ── Root ────────────────────────────────────────────────────────────────────────

class NeurosurferError(Exception):
    """Root of the neurosurfer exception tree.

    All library exceptions inherit from this so callers can catch a single type
    for coarse-grained handling, or a specific subclass for fine-grained handling
    (mirrors the LangChain approach of a single LangChainException root).
    """


# ── Graph-level ─────────────────────────────────────────────────────────────────

class GraphError(NeurosurferError):
    """Base exception for graph-related issues."""


class GraphConfigurationError(GraphError):
    """Invalid graph spec, missing tools, cycles, bad callable references, etc."""


class GraphExecutionError(GraphError):
    """Raised when the graph as a whole cannot continue (e.g. fail_fast mode)."""

    def __init__(self, message: str, *, failed_node: str = "") -> None:
        super().__init__(message)
        self.failed_node = failed_node


class InputValidationError(GraphError):
    """Raised when runtime inputs do not satisfy the graph's declared input spec."""


# ── Node-level ───────────────────────────────────────────────────────────────────

class NodeFailedError(GraphExecutionError):
    """A single node failed; carries structured context for trace / self-healing.

    LangChain analogue: ``ChainError`` carrying ``llm_output`` + ``observation``.
    """

    def __init__(
        self,
        node_id: str,
        message: str,
        *,
        cause: BaseException | None = None,
        attempt: int = 1,
        duration_ms: int = 0,
    ) -> None:
        super().__init__(f"[{node_id}] {message}", failed_node=node_id)
        self.node_id = node_id
        self.cause = cause
        self.attempt = attempt
        self.duration_ms = duration_ms

    def __str__(self) -> str:  # noqa: D105
        base = super().__str__()
        if self.cause:
            return f"{base} (caused by {type(self.cause).__name__}: {self.cause})"
        return base


class NodeSkippedError(NodeFailedError):
    """A node was skipped because an upstream dependency failed."""

    def __init__(self, node_id: str, upstream_id: str, upstream_error: str) -> None:
        super().__init__(
            node_id,
            f"Skipped: upstream node '{upstream_id}' failed: {upstream_error}",
        )
        self.upstream_id = upstream_id
        self.upstream_error = upstream_error


class NodeTimeoutError(NodeFailedError):
    """A node exceeded its configured timeout_s budget."""

    def __init__(self, node_id: str, timeout_s: float) -> None:
        super().__init__(node_id, f"Timed out after {timeout_s}s")
        self.timeout_s = timeout_s


# ── Agent / output-schema ─────────────────────────────────────────────────────

class AgentError(NeurosurferError):
    """Base for agent-level failures."""


class StructuredOutputError(AgentError, NodeFailedError):
    """Raised when a structured-output node exhausts all JSON repair attempts.

    LangChain analogue: ``OutputParserException`` from ``OutputFixingParser``
    after ``max_retries`` are exhausted.
    """

    def __init__(
        self,
        node_id: str,
        schema: type[Any],
        last_raw: str,
        *,
        attempts: int = 1,
    ) -> None:
        msg = (
            f"Structured output for schema '{schema.__name__}' failed after "
            f"{attempts} repair attempt(s). Last raw response: {last_raw[:200]!r}"
        )
        # Call AgentError (and transitively NeurosurferError) explicitly; avoid
        # MRO ambiguity by not calling both super().__init__() chains.
        NeurosurferError.__init__(self, f"[{node_id}] {msg}")
        self.node_id = node_id
        self.failed_node = node_id
        # Set NodeFailedError attrs not set via __init__ to keep __str__ working.
        self.cause = None
        self.attempt = 1
        self.duration_ms = 0
        self.schema = schema
        self.last_raw = last_raw
        self.attempts = attempts


# ── Tool-level ────────────────────────────────────────────────────────────────

class ToolError(NeurosurferError):
    """Base for tool-related failures."""

    def __init__(self, tool_name: str, message: str) -> None:
        super().__init__(f"[tool:{tool_name}] {message}")
        self.tool_name = tool_name


class ToolNotFoundError(ToolError):
    """Requested tool name is not registered in the Toolkit."""


class ToolInputError(ToolError):
    """Tool inputs failed schema validation before execution.

    LangChain analogue: ``ValidationError`` from ``StructuredTool`` arg parsing.
    """

    def __init__(self, tool_name: str, errors: Any) -> None:
        super().__init__(tool_name, f"Invalid inputs: {errors}")
        self.validation_errors = errors


class ToolExecutionError(ToolError):
    """Tool raised an exception during execution.

    LangChain analogue: ``ToolException`` raised inside ``BaseTool.run()``.
    """

    def __init__(self, tool_name: str, cause: BaseException) -> None:
        super().__init__(tool_name, f"Execution failed: {cause}")
        self.cause = cause


# ── Code execution ────────────────────────────────────────────────────────────

class CodeExecutionError(AgentError):
    """Raised when code execution returns non-zero exit code or times out."""

    def __init__(self, returncode: int, stderr: str, *, timeout: bool = False) -> None:
        if timeout:
            super().__init__("Code execution timed out")
        else:
            super().__init__(f"Code execution failed (rc={returncode}): {stderr[:500]}")
        self.returncode = returncode
        self.stderr = stderr
        self.timeout = timeout


# ── Legacy aliases (keep existing code working) ──────────────────────────────

# Pre-D1 names kept for backward compatibility
NodeExecutionError = NodeFailedError
NodeError = NodeFailedError
ValidationError = InputValidationError
PlanningError = GraphError
