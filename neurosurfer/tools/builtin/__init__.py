"""Backwards-compatible re-exports; the tools themselves live in the registry.

They moved to :mod:`neurosurfer.registry.core`, grouped by what they touch —
`filesystem/`, `data/`, `database/`, `web/`, `system/`, `agent/` — because
eighteen files in one flat directory had stopped being scannable, and because an
imported or authored tool needs an obvious place to land beside the core one it
sits next to.

This module stays so that ``from neurosurfer.tools.builtin import ReadFileTool``
keeps working. An internal reorganisation should not become a migration for
anybody: nothing outside this package needs to know the files moved.
"""

from __future__ import annotations

from neurosurfer.registry.core.agent.ask_user import AskUserTool
from neurosurfer.registry.core.agent.finish import FinishTool
from neurosurfer.registry.core.agent.spawn_agent import SpawnAgentTool
from neurosurfer.registry.core.agent.todo import TodoTool
from neurosurfer.registry.core.data.data_tool import DataTool
from neurosurfer.registry.core.database.sql_tools import (
    SqlTool,
)
from neurosurfer.registry.core.filesystem.apply_edit import ApplyEditTool
from neurosurfer.registry.core.filesystem.list_dir import ListDirTool
from neurosurfer.registry.core.filesystem.read_file import ReadFileTool
from neurosurfer.registry.core.filesystem.search import SearchTool
from neurosurfer.registry.core.filesystem.write_file import WriteFileTool
from neurosurfer.registry.core.system.install_package import InstallPythonPackageTool
from neurosurfer.registry.core.system.python_exec import (
    CodeExecutionError,
    PythonExecTool,
)
from neurosurfer.registry.core.system.run_command import RunCommandTool
from neurosurfer.registry.core.system.set_python_env import SetPythonEnvTool
from neurosurfer.registry.core.web.browse import BrowseTool
from neurosurfer.registry.core.web.http_tool import HttpTool
from neurosurfer.registry.core.web.web_search import WebSearchTool

# Moved out of the framework in F6: present_plan + register_task →
# neurosurfer.app.tools; write_workflow_node → neurosurfer.graph.workflow.node_tool.

__all__ = [
    "ApplyEditTool",
    "AskUserTool",
    "BrowseTool",
    "CodeExecutionError",
    "DataTool",
    "FinishTool",
    "HttpTool",
    "InstallPythonPackageTool",
    "ListDirTool",
    "PythonExecTool",
    "ReadFileTool",
    "RunCommandTool",
    "SearchTool",
    "SetPythonEnvTool",
    "SpawnAgentTool",
    "SqlTool",
    "TodoTool",
    "WebSearchTool",
    "WriteFileTool",
]
