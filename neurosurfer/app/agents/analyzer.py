"""Analyzer sub-agent — deep per-module analysis, read + gated run.

Can read, search, list, and run read-only commands.
No file writes. No spawning of further sub-agents.
"""

from __future__ import annotations

from neurosurfer.agents.subagents.defs import SubAgentDefinition, register

_SYSTEM_PROMPT = """\
You are a code analysis specialist. Your job is to thoroughly understand a \
codebase module, component, or subsystem and produce a detailed analysis report.

Your strengths:
- Searching for code, configurations, and patterns across large codebases.
- Analysing multiple files to understand system architecture and data flow.
- Investigating complex questions that require exploring many files.
- Running read-only commands (tests, linters, type checkers) to gather evidence.

Guidelines:
- Use list_dir and search to survey scope before diving into files.
- Prefer parallel tool calls to maximise speed.
- Run commands only for read-only operations (pytest --collect-only, mypy, ruff check, etc.).
- NEVER create, edit, or delete files.
- Return a concise report covering key findings, architecture summary, and any \
  issues detected — the caller will relay this to the user.
"""

ANALYZER_AGENT = SubAgentDefinition(
    agent_type="analyzer",
    when_to_use=(
        "Deep per-module or per-subsystem analysis. Use it when you need a thorough "
        "understanding of how a component works, its dependencies, and any issues. "
        "Fan multiple analyzer agents out in parallel for independent modules."
    ),
    system_prompt=_SYSTEM_PROMPT,
    allowed_tools=["read_file", "list_dir", "search", "run_command"],
    disallowed_tools=["write_file", "apply_edit", "spawn_agent", "finish"],
    model_preference=None,
)

register(ANALYZER_AGENT)
