"""Provisioning for the managed ``python_exec`` virtual environment.

A stdlib ``venv`` at ``~/.neurosurfer/venv`` (overridable via ``NEUROSURFER_HOME``),
pre-populated with a curated set of packages covering the common "agent writes a
script that needs numpy/pandas/PDF/image libs" cases. Self-contained — it does not
depend on conda or on whatever interpreter launched the host process.
"""

from __future__ import annotations

import functools
import os
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

# Packages pre-installed into the managed venv so common tasks (data analysis,
# PDF/Office export, image processing, plotting, scraping) work out of the box.
CURATED_PACKAGES: tuple[str, ...] = (
    "numpy",
    "pandas",
    "matplotlib",
    "pillow",
    "reportlab",
    "fpdf2",
    "openpyxl",
    "python-docx",
    "beautifulsoup4",
    "requests",
    "tabulate",
    "lxml",
)

_MARKER_NAME = ".provisioned"


def _home_dir() -> Path:
    override = os.environ.get("NEUROSURFER_HOME")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".neurosurfer"


def managed_venv_dir() -> Path:
    """Root directory of the managed venv (``~/.neurosurfer/venv``)."""
    return _home_dir() / "venv"


def _bin_dir(venv_dir: Path) -> Path:
    return venv_dir / ("Scripts" if os.name == "nt" else "bin")


def managed_python() -> Path | None:
    """Path to the managed venv's interpreter, or ``None`` if not provisioned."""
    venv_dir = managed_venv_dir()
    exe_name = "python.exe" if os.name == "nt" else "python"
    interpreter = _bin_dir(venv_dir) / exe_name
    return interpreter if interpreter.exists() else None


def is_provisioned() -> bool:
    return (managed_venv_dir() / _MARKER_NAME).exists() and managed_python() is not None


def ensure_managed_venv(*, notify: Callable[[str], None] | None = None) -> Path:
    """Create the managed venv and install :data:`CURATED_PACKAGES` if needed.

    Idempotent: a no-op (fast path) once the ``.provisioned`` marker exists next to
    an interpreter. Returns the interpreter path. Raises ``RuntimeError`` on failure
    so callers can decide whether to fall back to ``sys.executable``.
    """
    def _notify(msg: str) -> None:
        if notify is not None:
            notify(msg)

    if is_provisioned():
        interp = managed_python()
        assert interp is not None
        return interp

    venv_dir = managed_venv_dir()
    venv_dir.parent.mkdir(parents=True, exist_ok=True)

    interpreter = _bin_dir(venv_dir) / ("python.exe" if os.name == "nt" else "python")
    if not interpreter.exists():
        _notify(f"Creating managed Python environment at {venv_dir}…")
        result = subprocess.run(
            [sys.executable, "-m", "venv", str(venv_dir)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Failed to create managed venv: {result.stderr.strip()}")

    _notify(f"Installing {len(CURATED_PACKAGES)} packages into the managed environment…")
    result = subprocess.run(
        [str(interpreter), "-m", "pip", "install", "--quiet", *CURATED_PACKAGES],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to install curated packages: {result.stderr.strip()}")

    (venv_dir / _MARKER_NAME).write_text("ok", encoding="utf-8")
    _notify(f"Managed Python environment ready at {venv_dir}.")
    return interpreter


@functools.lru_cache(maxsize=8)
def _pip_freeze(interpreter: str) -> tuple[str, ...]:
    """Cached ``pip list`` for ``interpreter`` — called on every prompt turn via
    :mod:`neurosurfer.prompts.environment`, so avoid re-shelling out each time.
    Invalidated by :func:`invalidate_package_cache` after an install."""
    try:
        result = subprocess.run(
            [interpreter, "-m", "pip", "list", "--format=freeze"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return ()
        return tuple(ln.strip() for ln in result.stdout.splitlines() if ln.strip())
    except Exception:
        return ()


def installed_packages(interpreter: Path | str, *, limit: int = 40) -> list[str]:
    """Best-effort ``pip list`` (name==version) for ``interpreter``, capped at ``limit``.

    Never raises — returns an empty list on any failure so prompt assembly stays safe.
    """
    return list(_pip_freeze(str(interpreter)))[:limit]


def invalidate_package_cache() -> None:
    """Clear the cached ``pip list`` output — call after installing a package."""
    _pip_freeze.cache_clear()
