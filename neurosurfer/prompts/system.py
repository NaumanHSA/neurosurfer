"""System-prompt assembly: static base + Task instructions + dynamic env.

The static base (identity/tone/tasks/tools/planning) comes first so it forms a
stable, cacheable prefix on Anthropic. Task-specific instructions and the
guardrail summary follow, then a small env block.
"""

from __future__ import annotations

import platform
from datetime import date
from pathlib import Path

from ..agents.runtime.permissions import Guardrails
from .base_agent import base_system_sections


def _env_section(cwd: Path, model: str) -> str:
    return (
        "# Environment\n"
        f"- Working directory: {cwd}\n"
        f"- Platform: {platform.system().lower()}\n"
        f"- Today's date: {date.today().isoformat()}\n"
        f"- Model: {model}"
    )


def _guardrail_section(g: Guardrails) -> str:
    return (
        "# Guardrails (enforced)\n"
        f"- Writable scope: {', '.join(g.write_scope) or '(none)'}\n"
        f"- Shell policy: {g.shell_policy}\n"
        f"- Off-limits paths: {', '.join(g.path_deny) or '(none)'}\n"
        "These limits are enforced by the engine and default your writes to the scope "
        "above. If the user explicitly asks you to write somewhere else, still attempt "
        "the write with write_file — the engine will ask the user to approve that "
        "location (always / once / deny). Do not refuse up front or paste file contents "
        "into the chat as a substitute for writing the file. Off-limits paths are a hard "
        "boundary and cannot be approved."
    )


def build_system_prompt(
    *,
    task_instructions: str,
    guardrails: Guardrails,
    cwd: Path,
    model: str,
    extra_sections: list[str] | None = None,
) -> str:
    sections = list(base_system_sections())
    if task_instructions.strip():
        sections.append("# Your task\n" + task_instructions.strip())
    sections.append(_guardrail_section(guardrails))
    sections.append(_env_section(cwd, model))
    for extra in extra_sections or []:
        if extra.strip():
            sections.append(extra.strip())
    return "\n\n".join(sections)
