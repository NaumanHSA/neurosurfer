"""/pyenv — inspect or pin the Python environment python_exec uses.

  /pyenv               show the interpreter python_exec currently resolves to
  /pyenv use <spec>    pin it for the rest of the session ('managed', 'conda:NAME',
                       or a venv/interpreter path)
  /pyenv list          list available conda environments (best-effort)

Mirrors the ``set_python_env`` tool so the user has a direct, non-agentic way to do
the same thing ("use conda env ABC" without going through the model).
"""

from __future__ import annotations

import asyncio
import os

from .. import assistant as _assistant
from ..context import CLIContext
from .base import SlashCommand


async def _pyenv_cmd(ctx: CLIContext, args: list[str]) -> None:
    from neurosurfer.tools.builtin.python_exec.interpreter import (
        SESSION_KEY,
        EnvResolutionError,
        describe_interpreter,
        resolve_env_spec,
    )

    console = ctx.console
    agent_io = ctx._extra.get(_assistant._ASSISTANT_KEY)

    if not args:
        if agent_io is not None:
            agent, _ = agent_io
            interpreter, ready = describe_interpreter(agent.tool_context)
        else:
            interpreter, ready = _peek_without_agent()
        state = "" if ready else " (not yet provisioned)"
        console.print(f"python_exec interpreter: {interpreter}{state}")
        return

    sub, *rest = args

    if sub == "list":
        console.print("managed  — the neurosurfer-managed venv (~/.neurosurfer/venv)")
        for name in await asyncio.to_thread(_conda_env_names):
            console.print(f"conda:{name}")
        return

    if sub == "use":
        if not rest:
            console.print("Usage: /pyenv use <managed|conda:NAME|/path/to/venv>")
            return
        spec = rest[0]
        try:
            interpreter = await asyncio.to_thread(resolve_env_spec, spec)
        except EnvResolutionError as e:
            console.print(f"✗ {e}")
            return

        if agent_io is not None:
            agent, _ = agent_io
            agent.tool_context.extra[SESSION_KEY] = interpreter
            if agent.durable is not None:
                agent.durable.set_python_env(interpreter)
        else:
            # No assistant agent constructed yet this session — set the env var so
            # the interpreter resolver's precedence picks it up once one is built.
            os.environ["NEUROSURFER_PYENV"] = spec

        console.print(f"✓ python_exec now uses: {interpreter}")
        return

    console.print(f"Unknown /pyenv subcommand '{sub}'. Try: (none), use <spec>, list")


def _peek_without_agent() -> tuple[str, bool]:
    from pathlib import Path

    from neurosurfer.tools.base import AutoApproveIOHandler, ToolContext
    from neurosurfer.tools.builtin.python_exec.interpreter import describe_interpreter

    dummy_ctx = ToolContext(cwd=Path.cwd(), io=AutoApproveIOHandler())
    return describe_interpreter(dummy_ctx)


def _conda_env_names() -> list[str]:
    import json
    import subprocess
    from pathlib import Path

    try:
        result = subprocess.run(
            ["conda", "env", "list", "--json"], capture_output=True, text=True, timeout=10
        )
        if result.returncode != 0:
            return []
        return [Path(p).name for p in json.loads(result.stdout).get("envs", [])]
    except Exception:
        return []


COMMAND = SlashCommand(
    name="pyenv",
    summary="Show or pin the Python environment python_exec uses",
    handler=_pyenv_cmd,
)
