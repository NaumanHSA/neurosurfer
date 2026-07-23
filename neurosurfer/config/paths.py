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


def projects_dir() -> Path:
    """Staging area for in-progress workflow builds."""
    return artifacts_home() / "projects"


def workflows_dir() -> Path:
    """Registry of finished, registered workflow packages."""
    return artifacts_home() / "workflows"


def generated_tools_dir() -> Path:
    """On-disk store for Architect-generated tools."""
    return artifacts_home() / "tools"


def traces_dir() -> Path:
    """Where exported execution traces (JSON) are written for debugging."""
    return artifacts_home() / "traces"


def runs_dir() -> Path:
    """Durable per-run records for the workflow execution API (Phase 2)."""
    return artifacts_home() / "runs"
