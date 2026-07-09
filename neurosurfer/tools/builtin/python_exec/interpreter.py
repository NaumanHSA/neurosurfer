"""Resolves which Python interpreter ``python_exec`` (and the installer) should use.

Precedence, highest first:
1. An interpreter pinned on the session (``ToolContext.extra["python_interpreter"]``),
   set via the ``set_python_env`` tool / ``/pyenv use`` command.
2. The ``NEUROSURFER_PYENV`` environment variable (same spec grammar as ``set_python_env``).
3. The managed venv (``~/.neurosurfer/venv``), auto-provisioned on first use.
4. ``sys.executable`` — the interpreter that launched the host process, as a last resort.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from ...base import ToolContext
from .managed_env import ensure_managed_venv, managed_python, managed_venv_dir

SESSION_KEY = "python_interpreter"


class EnvResolutionError(RuntimeError):
    """Raised when a user-supplied env spec (path / conda name) cannot be resolved."""


def _pinned_or_env(ctx: ToolContext) -> str | None:
    """The two cheap, non-provisioning precedence steps shared by both resolvers."""
    pinned = ctx.extra.get(SESSION_KEY)
    if pinned:
        return str(pinned)

    from_env = os.environ.get("NEUROSURFER_PYENV")
    if from_env:
        try:
            return resolve_env_spec(from_env)
        except EnvResolutionError:
            pass  # fall through
    return None


def resolve_interpreter(ctx: ToolContext) -> str:
    """Return the interpreter path ``python_exec``/the installer should invoke for ``ctx``.

    May block to provision the managed venv on first use — callers on the event
    loop should run this via ``asyncio.to_thread``.
    """
    found = _pinned_or_env(ctx)
    if found is not None:
        return found

    managed = managed_python()
    if managed is not None:
        return str(managed)

    try:
        return str(ensure_managed_venv())
    except RuntimeError:
        return sys.executable


def describe_interpreter(ctx: ToolContext) -> tuple[str, bool]:
    """Non-provisioning peek at what ``resolve_interpreter`` would currently return.

    Returns ``(interpreter_path, ready)`` — ``ready=False`` means this is the managed
    venv's would-be path but it hasn't been provisioned yet. Used for prompt display
    (called every turn); never shells out to create the venv, so it can't add
    first-run provisioning latency to prompt assembly.
    """
    found = _pinned_or_env(ctx)
    if found is not None:
        return found, True

    managed = managed_python()
    if managed is not None:
        return str(managed), True

    return str(_bin_dir_python(managed_venv_dir())), False


def _conda_env_python(name: str) -> str:
    """Locate the interpreter for conda env ``name`` via ``conda env list --json``."""
    try:
        result = subprocess.run(
            ["conda", "env", "list", "--json"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as e:
        raise EnvResolutionError(f"Could not run conda: {e}") from e
    if result.returncode != 0:
        raise EnvResolutionError(f"conda env list failed: {result.stderr.strip()}")

    import json

    try:
        envs = json.loads(result.stdout).get("envs", [])
    except json.JSONDecodeError as e:
        raise EnvResolutionError(f"Could not parse conda env list: {e}") from e

    for env_path in envs:
        if Path(env_path).name == name:
            exe = Path(env_path) / ("Scripts" if os.name == "nt" else "bin") / (
                "python.exe" if os.name == "nt" else "python"
            )
            if exe.exists():
                return str(exe)
    raise EnvResolutionError(f"No conda environment named {name!r} found.")


def resolve_env_spec(spec: str) -> str:
    """Resolve a user-facing env spec to a concrete interpreter path.

    Accepted forms:
    - ``"managed"`` — the neurosurfer-managed venv (provisioning it if needed).
    - ``"conda:NAME"`` — a conda environment, located via ``conda env list``.
    - a path to a venv directory (its ``bin/python`` is used) or directly to an
      interpreter executable.
    """
    spec = spec.strip()
    if spec == "managed":
        return str(ensure_managed_venv())
    if spec.startswith("conda:"):
        name = spec.removeprefix("conda:").strip()
        if not name:
            raise EnvResolutionError("conda: spec requires an environment name, e.g. 'conda:ABC'.")
        return _conda_env_python(name)

    path = Path(spec).expanduser()
    if path.is_file():
        return str(path)
    if path.is_dir():
        exe = _bin_dir_python(path)
        if exe.exists():
            return str(exe)
        raise EnvResolutionError(f"No python interpreter found under {path}.")
    raise EnvResolutionError(
        f"'{spec}' is not 'managed', 'conda:<name>', or an existing path."
    )


def _bin_dir_python(venv_dir: Path) -> Path:
    return venv_dir / ("Scripts" if os.name == "nt" else "bin") / (
        "python.exe" if os.name == "nt" else "python"
    )
