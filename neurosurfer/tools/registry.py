"""Assembles the curated tool pool from the built-in implementations.

A Task narrows this pool via its ``tools:`` allow-list. Because the pool is small,
all selected schemas are sent every turn (no deferred/ToolSearch indirection).
"""

from __future__ import annotations

from collections.abc import Callable

from .base import Tool, ToolPool
from .builtin import (
    ApplyEditTool,
    AskUserTool,
    BrowseTool,
    DataTool,
    FinishTool,
    HttpTool,
    InstallPythonPackageTool,
    ListDirTool,
    PythonExecTool,
    ReadFileTool,
    RunCommandTool,
    SearchTool,
    SetPythonEnvTool,
    SpawnAgentTool,
    TodoTool,
    WebSearchTool,
    WriteFileTool,
)

# Plugin hook: product/feature layers (app, workflows) register their own tool
# factories here so the framework registry never imports product/feature code.
# Each factory is a zero-arg callable returning a fresh Tool instance.
_REGISTERED_FACTORIES: list[Callable[[], Tool]] = []


def register_tool_factory(factory: Callable[[], Tool]) -> None:
    """Register a tool factory contributed by a non-framework layer.

    Idempotent per factory object so re-imports don't duplicate registrations.
    The tool becomes visible from :func:`all_tools` / :func:`default_pool` once
    the contributing layer (e.g. ``neurosurfer.app`` or ``neurosurfer.graph.workflow``)
    has been imported.
    """
    if factory not in _REGISTERED_FACTORIES:
        _REGISTERED_FACTORIES.append(factory)


# Live, session-scoped tool instances contributed at runtime — currently MCP tools
# discovered by an :class:`~neurosurfer.mcp.manager.McpManager` after it connects.
# Unlike factories/generated tools these are *already-constructed* instances bound to
# a live connection, so the manager sets the whole list and clears it on shutdown.
# They appear in :func:`all_tools` (so agents/sub-agents can call them) but NOT in
# :func:`workflow_node_tools` (the Architect must not compose graphs from ephemeral
# connections — see the Integrations plan, Phase 2).
_LIVE_TOOLS: list[Tool] = []


def set_live_tools(tools: list[Tool]) -> None:
    """Replace the live (MCP) tool set. Called by the MCP manager after connecting."""
    _LIVE_TOOLS[:] = list(tools)


def clear_live_tools() -> None:
    """Drop all live tools (called when the MCP manager shuts down)."""
    _LIVE_TOOLS.clear()


def live_tools() -> list[Tool]:
    return list(_LIVE_TOOLS)


def _builtin_tools() -> list[Tool]:
    """One fresh instance of every generic framework tool, plus any tools
    registered by product/feature layers via :func:`register_tool_factory`."""
    tools: list[Tool] = [
        ReadFileTool(),
        ListDirTool(),
        SearchTool(),
        DataTool(),
        RunCommandTool(),
        PythonExecTool(),
        InstallPythonPackageTool(),
        SetPythonEnvTool(),
        WebSearchTool(),
        HttpTool(),
        BrowseTool(),
        WriteFileTool(),
        ApplyEditTool(),
        AskUserTool(),
        TodoTool(),
        SpawnAgentTool(),
        FinishTool(),
    ]
    tools.extend(factory() for factory in _REGISTERED_FACTORIES)
    return tools


def all_tools() -> list[Tool]:
    """Every available tool: built-ins plus Architect-generated tools (Phase E3).

    Generated tools live on disk under ``~/.neurosurfer/tools/`` and are discovered
    at call time, so a tool authored during a build is immediately usable. A built-in
    always wins a name clash with a generated tool.
    """
    from .generated import load_generated_tools  # noqa: PLC0415 - avoid import cycle

    tools = _builtin_tools()
    known = {t.name for t in tools}
    for gen in load_generated_tools():
        if gen.name not in known:
            tools.append(gen)
            known.add(gen.name)
    # Live MCP tools last: a built-in or generated tool always wins a name clash.
    for live in live_tools():
        if live.name not in known:
            tools.append(live)
            known.add(live.name)
    return tools


# Built-in tools that make sense as building blocks inside a generated workflow node.
# Excludes agent-loop-control tools (finish / present_plan / todo / ask_user /
# spawn_agent) and the architect-only write_workflow_node. The Architect
# composes workflows from these plus any generated tools — it must not invent names.
_BUILTIN_WORKFLOW_NODE_TOOLS: frozenset[str] = frozenset({
    "read_file",
    "list_dir",
    "search",
    "data",
    "run_command",
    "web_search",
    "http",
    "browse",
    "write_file",
    "apply_edit",
})

# Back-compat alias (built-in worker set). Prefer workflow_node_tool_names() for the
# dynamic set that also includes generated tools.
WORKFLOW_NODE_TOOL_NAMES: frozenset[str] = _BUILTIN_WORKFLOW_NODE_TOOLS

# Common names the LLM invents for capabilities that an existing tool already covers.
# Mapped to the real tool before validation so we don't auto-generate redundant tools.
# A value of "" means "drop this tool" (the node is just an LLM step — e.g. 'llm_writer').
_TOOL_ALIASES: dict[str, str] = {
    # directory / listing / walking
    "ls": "list_dir",
    "ls_dir": "list_dir",
    "list_files": "list_dir",
    "list_directory": "list_dir",
    "list_dir_contents": "list_dir",
    "get_directory_structure": "list_dir",
    "directory_listing": "list_dir",
    "scan_directory": "list_dir",
    "scan_dir": "list_dir",
    "scan_files": "list_dir",
    "walk_directory": "list_dir",
    "walk_dir": "list_dir",
    "walk_files": "list_dir",
    "traverse_directory": "list_dir",
    "directory_walker": "list_dir",
    "os_walk": "list_dir",
    "file_tree": "list_dir",
    "get_file_tree": "list_dir",
    "glob": "list_dir",
    "glob_files": "list_dir",
    "find_files": "list_dir",
    "discover_files": "list_dir",
    "enumerate_files": "list_dir",
    # reading
    "get_file_content": "read_file",
    "read_file_content": "read_file",
    "file_reader": "read_file",
    "read_content": "read_file",
    "cat": "read_file",
    "open_file": "read_file",
    # writing
    "file_writer": "write_file",
    "save_file": "write_file",
    "create_file": "write_file",
    "write_to_file": "write_file",
    # editing
    "edit_file": "apply_edit",
    "modify_file": "apply_edit",
    # shell / fs ops that have no dedicated tool → run_command
    "create_directory": "run_command",
    "make_directory": "run_command",
    "mkdir": "run_command",
    "shell": "run_command",
    "execute_command": "run_command",
    "run_shell": "run_command",
    "code_parser": "run_command",
    "parse_code": "run_command",
    # search
    "grep": "search",
    "find_in_files": "search",
    "search_files": "search",
    # web
    "fetch_url": "http",
    "http_request": "http",
    "web_scraper": "browse",
    "scrape": "browse",
    "search_web": "web_search",
    # pure-LLM "tools" → the node is a base/react step; drop the tool
    "llm_writer": "",
    "llm": "",
    "text_generator": "",
    "summarizer": "",
    "summariser": "",
    "generator": "",
    "writer": "",
}


# Conservative keyword fallback for invented names the exact map misses. Only the
# unambiguous filesystem / shell clusters — anything else is left to the gate / E6 so
# we never silently mis-map a genuinely-novel capability.
_KEYWORD_FALLBACK: list[tuple[tuple[str, ...], str]] = [
    (("directory", "folder", "walk", "traverse", "file_tree", "filetree"), "list_dir"),
    (("read_file", "file_content", "file_read", "read_content", "get_content"), "read_file"),
    (("write_file", "save_file", "create_file", "file_write", "output_file"), "write_file"),
    (("run_command", "shell", "subprocess", "execute_shell", "bash", "command_line"), "run_command"),
]


def _keyword_fallback(name: str) -> str | None:
    low = name.lower()
    for keys, target in _KEYWORD_FALLBACK:
        if any(k in low for k in keys):
            return target
    return None


def normalize_tool_names(tools: list[str]) -> list[str]:
    """Map invented tool names to real tools (and drop pure-LLM pseudo-tools).

    Resolution order per name: exact alias → keyword fallback → leave as-is. Names with
    no mapping are kept so the validation gate still surfaces them (typo → suggestion,
    or genuine gap → E6). Returns a de-duplicated list.
    """
    out: list[str] = []
    for t in tools or []:
        if t in _TOOL_ALIASES:
            mapped = _TOOL_ALIASES[t]
        elif t in workflow_node_tool_names():
            mapped = t  # already a real tool
        else:
            mapped = _keyword_fallback(t) or t
        if mapped and mapped not in out:
            out.append(mapped)
    return out


def workflow_node_tools() -> list[Tool]:
    """Tools usable inside a generated workflow node: the built-in worker subset,
    every Architect-generated tool, and any live (MCP) tools.

    MCP tools are workflow-usable because their server configs persist in the
    ``McpStore``: the workflow runtime reconnects on demand
    (:func:`neurosurfer.mcp.runtime.ensure_mcp_tools`), so a registered workflow
    referencing an MCP tool keeps working across sessions — the connection is
    ephemeral, the capability is not."""
    from .generated import load_generated_tools  # noqa: PLC0415 - avoid import cycle

    tools = [t for t in _builtin_tools() if t.name in _BUILTIN_WORKFLOW_NODE_TOOLS]
    known = {t.name for t in tools}
    for gen in load_generated_tools():
        if gen.name not in known:
            tools.append(gen)
            known.add(gen.name)
    for live in live_tools():
        if live.name not in known:
            tools.append(live)
            known.add(live.name)
    return tools


def workflow_node_tool_names() -> set[str]:
    """Dynamic set of tool names valid inside a workflow node (built-ins + generated)."""
    return {t.name for t in workflow_node_tools()}


def format_workflow_tool_catalog() -> str:
    """A compact ``- name: one-line description`` catalog for architect prompts."""
    lines: list[str] = []
    for t in workflow_node_tools():
        desc = (t.description or "").strip().splitlines()[0] if t.description else ""
        lines.append(f"- {t.name}: {desc}")
    return "\n".join(lines)


def default_pool() -> ToolPool:
    return ToolPool(all_tools())


def build_pool(names: list[str]) -> ToolPool:
    """Build a pool limited to ``names`` (a Task's allow-list)."""
    return default_pool().select(names)
