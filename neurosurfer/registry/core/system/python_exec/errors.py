from __future__ import annotations


class CodeExecutionError(RuntimeError):
    """Raised when the Python subprocess cannot be launched (OS-level failure, not a code error)."""
