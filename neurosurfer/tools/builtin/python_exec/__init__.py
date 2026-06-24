"""Subprocess-sandboxed Python code execution tool.

Public surface::

    from neurosurfer.tools.builtin.python_exec import PythonExecTool, CodeExecutionError

``PythonExecTool`` is the native :class:`~neurosurfer.tools.base.Tool` to register
in an agent's tool pool.  ``CodeExecutionError`` is raised when the subprocess
cannot be launched (OS-level failure, not a code-level error).
"""

from .errors import CodeExecutionError
from .tool import PythonExecArgs, PythonExecTool

__all__ = ["CodeExecutionError", "PythonExecArgs", "PythonExecTool"]
