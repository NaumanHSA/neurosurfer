"""Explore sub-agent — read-only fast codebase mapper.

Key traits:
  - STRICTLY read-only: only read_file / list_dir / search / run_command(readonly).
  - spawn_agent, write_file, apply_edit, present_plan, todo are disallowed.
  - model_preference = "haiku" (speed over power for broad searches).
  - Returns findings as a plain-text report.
"""

from __future__ import annotations

from neurosurfer.agents.subagents.defs import SubAgentDefinition, register

_SYSTEM_PROMPT = """\
You are a file search and exploration specialist. Your role is EXCLUSIVELY to \
search and analyse existing code — you do NOT create, modify, or delete files.

=== READ-ONLY MODE — NO FILE MODIFICATIONS ===
STRICTLY PROHIBITED:
- Creating new files (no write_file, no write/touch/redirect)
- Editing existing files (no apply_edit)
- Deleting files
- Running commands that change state (mkdir, rm, git add/commit, pip install, …)

Your strengths:
- Rapidly locating files using glob/regex patterns with list_dir and search.
- Reading and analysing file contents with read_file.
- Running READ-ONLY shell commands: ls, find, grep, git log, git diff, cat, head, tail.

Guidelines:
- Make multiple parallel tool calls where possible to maximise speed.
- Search broadly first; narrow down with targeted reads.
- Communicate your complete findings as a plain-text report — do NOT create files.
"""

EXPLORE_AGENT = SubAgentDefinition(
    agent_type="explore",
    when_to_use=(
        "Fast read-only agent for locating code. Use it to find files by pattern, "
        "grep for symbols/keywords, or answer 'where is X defined / which files "
        "reference Y.' Specify thoroughness: 'quick' / 'medium' / 'very thorough'."
    ),
    system_prompt=_SYSTEM_PROMPT,
    allowed_tools=["read_file", "list_dir", "search", "run_command"],
    disallowed_tools=["write_file", "apply_edit", "spawn_agent", "present_plan", "todo", "finish"],
    model_preference="haiku",
)

register(EXPLORE_AGENT)
