# Tools Catalog

The built-in tools an agent can call. Assemble them with `default_pool()` (a sensible default set)
or `build_pool([...])` to pick specific ones. For how tools work and how to write your own, see the
[Tools guide](tools.md); for how a *need* resolves to a tool, see the
[Tool Registry](tool-registry.md).

```python
from neurosurfer.tools import default_pool, build_pool
tools = default_pool()                     # the standard set
tools = build_pool(["read_file", "search", "run_command"])   # a curated subset
```

Tools live under `neurosurfer/registry/core/<domain>/`, grouped by what they are rather than by what
they need. The **capability** column is the tag a tool declares itself against — that is what a
workflow resolves against, not the description text.

## Files & editing

`neurosurfer.registry.core.filesystem`

| Tool | What it does | Capability |
|---|---|---|
| `read_file` | Read a file (optionally a line range); returns text, with binary detection. | `file.read` |
| `write_file` | Create or overwrite a file. Gated by `write_scope`. | `file.write` |
| `apply_edit` | Apply a targeted edit (find/replace style) to an existing file. | `file.edit` |
| `list_dir` | List a directory's contents. | `file.list` |
| `search` | Search file contents / names across the workspace (grep-like). | `file.search` |

## Shell & execution

`neurosurfer.registry.core.system`

| Tool | What it does | Capability |
|---|---|---|
| `run_command` | Run a shell command. Gated by `shell_policy` (`gated` / `readonly` / `denied`). | `system.shell` |
| `python_exec` | Execute Python in a sandboxed environment and capture output. | — |
| `install_python_package` | Install a package into the managed environment. | — |
| `set_python_env` | Point `python_exec` at a different interpreter / environment. | — |

!!! note "Why `python_exec` declares no capability"
    A workflow runs Python through a [`function` node](../graph/node-kinds.md#function), not through
    a tool. Listing `system.python` as an unprovided capability told the model it could not do
    something it can. The vocabulary covers what a *node* resolves against.

## Web & network

`neurosurfer.registry.core.web`

| Tool | What it does | Capability | Needs |
|---|---|---|---|
| `web_search` | Search the web and return ranked results. | `web.search` | `search` extra (free DuckDuckGo) or `SERPAPI_API_KEY` |
| `http` | Make HTTP requests. Gated by `network_policy`. | `web.request` | — |
| `browse` | Drive a headless browser to load and read pages. | `web.browse` | `browser` extra |

## Data & databases

`neurosurfer.registry.core.data` · `neurosurfer.registry.core.database`

| Tool | What it does | Capability |
|---|---|---|
| `data` | Inspect or query a structured data **file** (CSV, JSON, SQLite). | `data.inspect` |
| `sql` | Read-only access to a database **server**. | `db.connect`, `db.schema`, `db.query` |

A file and a server are different problems, which is why they are different tools.

### The `sql` tool has four operations

One tool, four things it does — a tool is a *type* of integration and an operation is a thing it
does. Each operation carries its own arguments, so configuring one shows you the arguments *that*
operation needs rather than the union of all four with everything optional.

| Operation | What it does | Capability |
|---|---|---|
| `test_connection` | Check the connection works and report the server version. | `db.connect` |
| `list_tables` | List tables as schema-qualified names. | `db.schema` |
| `table_schema` | Columns with types and nullability, primary key, foreign keys. | `db.schema` |
| `query` | Run one read-only query and return rows as a table. | `db.query` |

It speaks whatever SQLAlchemy speaks (SQL Server, PostgreSQL, MySQL) and **cannot modify data** —
only `SELECT` / `WITH` / `EXPLAIN` / `SHOW`. Work in the order above: a table name guessed from the
request is the most common reason a generated query fails.

Its `dsn` is a **secret input**, so it never travels through a prompt. See
[State & secrets](../graph/state.md#secrets) and the [SQL Agent guide](../tutorials/sql-agent.md).

## Interaction & control

`neurosurfer.registry.core.agent`

| Tool | What it does |
|---|---|
| `ask_user` | Ask the user a question through the `io` handler. |
| `todo` | Maintain a task list the agent tracks across turns. |
| `finish` | Signal the run is complete (with a status/report). |
| `spawn_agent` | Spawn a scoped [sub-agent](subagents.md). Bounded by guardrails. |

## Extras

Some tools require extras: `pip install "neurosurfer[search,browser]"`. See
[Installation](../getting-started/installation.md#extras).

## Name aliases

Models often invent tool names. The registry normalises a long list of them to the canonical tool,
so a model calling `grep`, `cat`, or `shell` still lands on the right place:

| Model says | Runs |
|---|---|
| `grep`, `find`, `find_files`, `glob` | `search` / `list_dir` |
| `cat`, `open_file`, `get_file_content` | `read_file` |
| `save_file`, `create_file`, `write_to_file` | `write_file` |
| `edit_file`, `modify_file` | `apply_edit` |
| `ls`, `walk_directory`, `file_tree` | `list_dir` |
| `shell`, `bash`, `exec` | `run_command` |

This keeps agents robust across models without you having to prompt exact tool names.

## Next

- [Tools guide](tools.md) — writing your own, and what a tool declares.
- [Tool Registry](tool-registry.md) — capabilities, manifests, and how a need finds a tool.
