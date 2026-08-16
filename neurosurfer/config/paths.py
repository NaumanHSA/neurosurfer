"""Where neurosurfer writes workflow artifacts (projects, registry, generated tools).

For now these live **locally** under ``./.neurosurfer/`` in the current working
directory so generated workflows are easy to inspect while we iterate. Set the
``NEUROSURFER_HOME`` environment variable to point them elsewhere (e.g. back to
``~/.neurosurfer`` once the Architect is trusted).

Resolution is lazy (read at call time, not import time) so the active cwd / env var
is honoured per command.
"""

from __future__ import annotations

import os
from pathlib import Path


def artifacts_home() -> Path:
    """Base directory for all generated workflow artifacts.

    ``NEUROSURFER_HOME`` overrides; otherwise a local ``./.neurosurfer`` in the cwd
    (handy for debugging — the files sit right next to where you launched).
    """
    env = os.environ.get("NEUROSURFER_HOME")
    if env:
        return Path(env).expanduser()
    return Path.cwd() / ".neurosurfer"


def config_dir() -> Path:
    """Host-level configuration: MCP servers, provider profiles.

    Deliberately *not* per-workspace. An MCP server is a process the gateway
    spawns as its own OS user, so a per-account copy of `mcp.json` would look
    like isolation while providing none.
    """
    return artifacts_home() / "config"


def mcp_config_path() -> Path:
    return config_dir() / "mcp.json"


def projects_dir() -> Path:
    """Staging area for in-progress workflow builds."""
    return artifacts_home() / "projects"


def workflows_dir() -> Path:
    """Registry of finished, registered workflow packages."""
    return artifacts_home() / "workflows"


def generated_tools_dir() -> Path:
    """On-disk store for Architect-generated tools."""
    return artifacts_home() / "tools"


def runs_dir() -> Path:
    """Durable run records — one directory per run.

    Sibling of `workflows_dir()`: a workflow is the thing you registered, a run is
    one execution of it. Kept apart because a registry is edited and a run record
    is append-only.
    """
    return artifacts_home() / "runs"


def traces_dir() -> Path:
    """Where exported execution traces (JSON) are written for debugging."""
    return artifacts_home() / "traces"
