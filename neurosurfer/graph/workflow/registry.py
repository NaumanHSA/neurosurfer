"""WorkflowRegistry — persistent store for multi-file workflow packages.

Packages live under the artifacts home's ``workflows/<name>/`` (local
``./.neurosurfer/workflows/`` by default; see :mod:`neurosurfer.config.paths`).
Each sub-directory must contain a ``workflow.yaml`` manifest (the presence of that
file is the canonical marker that makes a directory a recognised workflow package).
"""

from __future__ import annotations

import shutil
from pathlib import Path

from neurosurfer.config.paths import workflows_dir as _workflows_dir

from .package import WorkflowPackage, load_package, save_package

__all__ = [
    "WorkflowRegistry",
    "WorkflowNotFoundError",
]


class WorkflowNotFoundError(KeyError):
    """Raised when a requested workflow name is not in the registry."""


class WorkflowRegistry:
    """List, get, save, and delete :class:`WorkflowPackage` entries.

    Parameters
    ----------
    workflows_dir:
        Root directory for stored packages.  Defaults to the artifacts home's
        ``workflows/`` (local ``./.neurosurfer/workflows/`` unless
        ``NEUROSURFER_HOME`` is set).
    """

    def __init__(self, workflows_dir: Path | None = None) -> None:
        self._dir = Path(workflows_dir) if workflows_dir else _workflows_dir()
        self._dir.mkdir(parents=True, exist_ok=True)

    # ── read ──────────────────────────────────────────────────────────────────

    def list(self) -> list[str]:
        """Return sorted names of all registered workflows."""
        return sorted(
            p.name
            for p in self._dir.iterdir()
            if p.is_dir() and (p / "workflow.yaml").exists()
        )

    def path(self, name: str) -> Path:
        """Return the on-disk directory for *name* (may not exist yet)."""
        return self._dir / name

    def exists(self, name: str) -> bool:
        return (self._dir / name / "workflow.yaml").exists()

    def get(self, name: str) -> WorkflowPackage:
        """Load and return the named workflow package.

        Raises :class:`WorkflowNotFoundError` if not found,
        :class:`PackageLoadError` if the package is corrupt.
        """
        pkg_dir = self._dir / name
        if not (pkg_dir / "workflow.yaml").exists():
            raise WorkflowNotFoundError(
                f"No workflow named '{name}'. "
                f"Available: {self.list() or ['(none)']}"
            )
        return load_package(pkg_dir)

    # ── write ─────────────────────────────────────────────────────────────────

    def save(self, pkg: WorkflowPackage) -> Path:
        """Register (or overwrite) *pkg* in the registry.

        The package is copied to ``<workflows_dir>/<pkg.name>/``.
        Returns the destination path.
        """
        dest = self._dir / pkg.name
        return save_package(pkg, dest)

    def delete(self, name: str) -> None:
        """Delete a registered workflow by name.

        Raises :class:`WorkflowNotFoundError` if it does not exist.
        """
        pkg_dir = self._dir / name
        if not pkg_dir.is_dir():
            raise WorkflowNotFoundError(f"No workflow named '{name}'.")
        shutil.rmtree(pkg_dir)
