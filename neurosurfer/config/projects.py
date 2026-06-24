"""Projects directory — staging area for in-progress workflow builds."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .paths import projects_dir


@dataclass
class ProjectsConfig:
    """Paths for in-progress workflow build staging.

    Each build gets its own subdirectory under the artifacts home (local
    ``./.neurosurfer/projects/`` by default; see :mod:`neurosurfer.config.paths`)::

        <home>/projects/<workflow_name>/
            agents/          ← per-node YAML written by write_workflow_node
            workflow.yaml    ← assembled manifest
            graph.yaml       ← assembled graph spec
    """

    dir: Path = field(default_factory=projects_dir)

    def project_dir(self, name: str) -> Path:
        """Return the staging directory for a named workflow build."""
        return self.dir / name
