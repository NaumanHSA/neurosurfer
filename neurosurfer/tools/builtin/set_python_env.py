"""Lets the user (or the agent, on request) pin ``python_exec`` to a specific
Python environment for the rest of the session — e.g. "use conda env ABC".

The resolved interpreter is stored on ``ToolContext.extra`` (read by
``python_exec`` / ``install_python_package`` via ``interpreter.resolve_interpreter``)
and mirrored into durable state so it is visible in the system prompt and survives
context compaction — the model won't re-ask or silently drift back to the managed venv.
"""

from __future__ import annotations

import asyncio

from pydantic import BaseModel, Field

from ..base import Tool, ToolContext, ToolResult
from .python_exec.interpreter import SESSION_KEY, EnvResolutionError, resolve_env_spec
from .python_exec.managed_env import installed_packages


class SetPythonEnvArgs(BaseModel):
    spec: str = Field(
        description=(
            "Which Python environment python_exec should use for the rest of the "
            "session: 'managed' (the neurosurfer-managed venv), 'conda:NAME' "
            "(a conda environment), or a path to a venv directory / interpreter."
        )
    )


class SetPythonEnvTool(Tool):
    name = "set_python_env"
    description = (
        "Pin python_exec (and install_python_package) to a specific Python "
        "environment for the rest of the session — e.g. spec='conda:myenv' or "
        "spec='managed' to go back to the default managed venv."
    )
    input_model = SetPythonEnvArgs

    def is_read_only(self, args: BaseModel) -> bool:
        return False

    async def call(self, args: SetPythonEnvArgs, ctx: ToolContext) -> ToolResult:  # type: ignore[override]
        try:
            interpreter = await asyncio.to_thread(resolve_env_spec, args.spec)
        except EnvResolutionError as e:
            return ToolResult.error(str(e))

        ctx.extra[SESSION_KEY] = interpreter
        if ctx.durable is not None:
            ctx.durable.set_python_env(interpreter)

        pkgs = await asyncio.to_thread(installed_packages, interpreter, limit=15)
        pkg_note = f" ({len(pkgs)} packages, e.g. {', '.join(pkgs[:5])})" if pkgs else ""
        return ToolResult.ok(f"python_exec now uses: {interpreter}{pkg_note}")
