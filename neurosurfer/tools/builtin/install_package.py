"""Gated pip-install tool for the active ``python_exec`` environment.

Reuses the existing shell-approval channel (:meth:`IOHandler.request_shell_approval`)
so installing a missing dependency goes through the same y/N (or free-text redirect)
gate as ``run_command`` — no new approval machinery. On approval, installs into
whichever interpreter ``python_exec`` would currently use (managed venv, pinned
session env, or a user-selected conda/venv — see ``python_exec/interpreter.py``),
so a subsequent ``python_exec`` call can immediately pick it up.
"""

from __future__ import annotations

import asyncio

from pydantic import BaseModel, Field

from ..base import Tool, ToolContext, ToolResult
from .python_exec.interpreter import resolve_interpreter
from .python_exec.managed_env import invalidate_package_cache

INSTALL_TIMEOUT = 300

# Packages whose distribution name differs from the module you `import`.
_IMPORT_NAME_OVERRIDES: dict[str, str] = {
    "pillow": "PIL",
    "beautifulsoup4": "bs4",
    "python-docx": "docx",
    "fpdf2": "fpdf",
    "pyyaml": "yaml",
    "scikit-learn": "sklearn",
    "opencv-python": "cv2",
}


def _import_name(package: str) -> str:
    base = package.split("==")[0].split(">=")[0].split("<=")[0].strip().lower()
    return _IMPORT_NAME_OVERRIDES.get(base, base.replace("-", "_"))


async def _is_importable(interpreter: str, package: str) -> bool:
    import_name = _import_name(package)
    probe = f"import importlib.util, sys; sys.exit(0 if importlib.util.find_spec({import_name!r}) else 1)"
    try:
        proc = await asyncio.create_subprocess_exec(
            interpreter,
            "-c",
            probe,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        code = await asyncio.wait_for(proc.wait(), timeout=15)
        return code == 0
    except Exception:
        return False


class InstallPythonPackageArgs(BaseModel):
    packages: list[str] = Field(
        description="Pip-installable package name(s), e.g. ['reportlab', 'fpdf2']."
    )
    reason: str = Field(
        default="",
        description="Why this package is needed — shown to the user in the approval prompt.",
    )


class InstallPythonPackageTool(Tool):
    name = "install_python_package"
    description = (
        "Install one or more Python packages (via pip) into the environment python_exec "
        "uses. Requires user approval before installing. Call this when python_exec fails "
        "with ModuleNotFoundError, then re-run the code — the package will be available. "
        "Already-importable packages are skipped automatically."
    )
    input_model = InstallPythonPackageArgs

    def is_read_only(self, args: BaseModel) -> bool:
        return False

    def is_destructive(self, args: BaseModel) -> bool:
        return True

    def progress_message(self, args: dict) -> str:
        pkgs = args.get("packages") or []
        return f"Installing {', '.join(pkgs)}…" if pkgs else "Installing package…"

    async def call(self, args: InstallPythonPackageArgs, ctx: ToolContext) -> ToolResult:  # type: ignore[override]
        if not args.packages:
            return ToolResult.error("No packages specified.")

        interpreter = await asyncio.to_thread(resolve_interpreter, ctx)

        missing = [p for p in args.packages if not await _is_importable(interpreter, p)]
        if not missing:
            return ToolResult.ok(
                f"Already available in {interpreter}: {', '.join(args.packages)}."
            )

        reason = args.reason.strip() or f"install {', '.join(missing)} for the current task"
        approval = await ctx.io.request_shell_approval(f"pip install {' '.join(missing)}", reason)
        if not approval.approved:
            return ToolResult.error(approval.feedback or "Installation declined by the user.")

        try:
            proc = await asyncio.create_subprocess_exec(
                interpreter,
                "-m",
                "pip",
                "install",
                "--quiet",
                *missing,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
        except OSError as e:
            return ToolResult.error(f"Failed to launch pip: {e}")

        try:
            out, _ = await asyncio.wait_for(proc.communicate(), timeout=INSTALL_TIMEOUT)
        except TimeoutError:
            proc.kill()
            await proc.wait()
            return ToolResult.error(f"pip install timed out after {INSTALL_TIMEOUT}s.")

        if proc.returncode != 0:
            text = out.decode("utf-8", errors="replace")
            return ToolResult.error(
                f"pip install failed (exit {proc.returncode}):\n{text[-2000:]}"
            )

        invalidate_package_cache()
        return ToolResult.ok(
            f"Installed {', '.join(missing)} into {interpreter}. "
            "python_exec is ready to use them — re-run your code."
        )
