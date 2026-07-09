"""The ``<environment>`` system-prompt section.

Tells the agent its actual runtime context — working directory, OS, git state, and
the active ``python_exec`` interpreter (plus what's installed in it) — instead of
leaving it to guess or to claim capabilities/packages it doesn't have. Appended as a
suffix (alongside durable state) so it never disturbs the static, cacheable prefix
in :mod:`neurosurfer.prompts.base_agent`.
"""

from __future__ import annotations

import platform
import subprocess
from pathlib import Path

from ..tools.base import ToolContext


def _git_status(cwd: Path) -> str | None:
    """Best-effort ``branch (clean|dirty)``; ``None`` outside a git repo or on any failure."""
    try:
        branch = subprocess.run(
            ["git", "-C", str(cwd), "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            timeout=3,
        )
        if branch.returncode != 0:
            return None
        dirty = subprocess.run(
            ["git", "-C", str(cwd), "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=3,
        )
        state = "dirty" if dirty.stdout.strip() else "clean"
        return f"{branch.stdout.strip()} ({state})"
    except Exception:
        return None


def environment_section(ctx: ToolContext) -> str:
    """Render the environment section for ``ctx``. Never raises."""
    from ..tools.builtin.python_exec.interpreter import describe_interpreter
    from ..tools.builtin.python_exec.managed_env import installed_packages

    lines = [
        "# Environment",
        f"- Working directory: {ctx.cwd}",
        f"- Platform: {platform.platform()}",
    ]

    git = _git_status(ctx.cwd)
    if git:
        lines.append(f"- Git: {git}")

    interpreter, ready = describe_interpreter(ctx)
    if ready:
        pkgs = installed_packages(interpreter, limit=12)
        names = [p.split("==")[0] for p in pkgs]
        pkg_note = f" — packages available: {', '.join(names)}" if names else ""
        lines.append(f"- python_exec interpreter: {interpreter}{pkg_note}")
    else:
        lines.append(
            f"- python_exec interpreter: not yet provisioned (will set up {interpreter} "
            "automatically on first use — common data/PDF/image packages included)"
        )
    lines.append(
        "  Use install_python_package for anything else missing; use set_python_env "
        "to switch environments (e.g. a conda env)."
    )

    return "\n".join(lines)
