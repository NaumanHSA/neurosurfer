"""Writer sub-agent — generates one doc section or file.

Has read + targeted write access (write_file, apply_edit) but no shell or
spawn. Scoped to produce a single focused piece of written output.
"""

from __future__ import annotations

from neurosurfer.agents.subagents.defs import SubAgentDefinition, register

_SYSTEM_PROMPT = """\
You are a technical writer. You generate a single well-structured document \
section or file based on provided source material and instructions.

Guidelines:
- Read the relevant source files thoroughly before writing.
- Write clear, accurate, professional prose targeted at developers.
- Do not invent API details — derive everything from the actual code you read.
- Produce ONE coherent output file or section; do not create extra files.
- When done, report what you wrote and its location.
"""

WRITER_AGENT = SubAgentDefinition(
    agent_type="writer",
    when_to_use=(
        "Generate a single documentation section or file. Provide the writer with "
        "the target file path, any source files to read, and the desired format/style. "
        "Run multiple writers in parallel to cover independent sections."
    ),
    system_prompt=_SYSTEM_PROMPT,
    allowed_tools=["read_file", "list_dir", "search", "write_file", "apply_edit"],
    disallowed_tools=["run_command", "spawn_agent", "present_plan", "finish"],
    model_preference=None,
)

register(WRITER_AGENT)
