# Tools Catalog

The built-in tools an agent can call. Assemble them with `default_pool()` (a sensible default set) or
`build_pool([...])` to pick specific ones. For how tools work and how to write your own, see the
[Tools guide](../guides/tools.md).

```python
from neurosurfer.tools import default_pool, build_pool
tools = default_pool()                     # the standard set
tools = build_pool(["read_file", "search", "run_command"])   # a curated subset
```

## Files & editing

| Tool | What it does |
|---|---|
| `read_file` | Read a file (optionally a line range); returns text, with binary detection. |
| `write_file` | Create or overwrite a file. Gated by `write_scope`. |
| `apply_edit` | Apply a targeted edit (find/replace style) to an existing file. |
| `list_dir` | List a directory's contents. |
| `search` | Search file contents / names across the workspace (grep-like). |

## Shell & execution

| Tool | What it does |
|---|---|
| `run_command` | Run a shell command. Gated by `shell_policy` (`gated` / `readonly` / `denied`). |
| `python_exec` | Execute Python in a sandboxed environment and capture output. |

## Web & network

| Tool | What it does | Needs |
|---|---|---|
| `web_search` | Search the web and return ranked results. | `search` extra (free DuckDuckGo) or `SERPAPI_API_KEY` |
| `http` | Make HTTP requests. Gated by `network_policy`. | — |
| `browse` | Drive a headless browser to load and read pages. | `browser` extra |

## Data

| Tool | What it does |
|---|---|
| `data` | Work with structured/tabular data (load, query, transform). |

## Interaction & control

| Tool | What it does |
|---|---|
| `ask_user` | Ask the user a question through the `io` handler. |
| `todo` | Maintain a task list the agent tracks across turns. |
| `finish` | Signal the run is complete (with a status/report). |
| `spawn_agent` | Spawn a scoped [sub-agent](../guides/subagents.md). Bounded by guardrails. |

## Name aliases

Models often invent tool names. The registry normalises common ones to the canonical tool, so a model
calling `grep`, `cat`, or `shell` still lands on the right tool:

| Model says | Runs |
|---|---|
| `grep`, `find` | `search` |
| `cat`, `open` | `read_file` |
| `shell`, `bash`, `exec` | `run_command` |

This keeps agents robust across models without you having to prompt exact tool names. See the
[Tools guide](../guides/tools.md) for custom tools, the `ToolPool`, and MCP tools.
