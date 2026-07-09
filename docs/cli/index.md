# CLI Agent

Neurosurfer's interactive CLI agent is the fastest way to start using the framework — no Python
required. Installing Neurosurfer puts a `neurosurfer` command on your PATH. With **no subcommand**
it opens an interactive REPL; with a subcommand it runs a scriptable action.

```bash
neurosurfer            # interactive REPL
neurosurfer doctor     # check configuration + active connection
neurosurfer provider   # manage provider profiles
neurosurfer serve      # start the OpenAI-compatible gateway
```

Run `neurosurfer --help` (or `neurosurfer <command> --help`) for full flags.

## Interactive REPL

Running `neurosurfer` with no arguments starts a chat REPL with persistent history, a live
provider/status line, and slash commands. Type a message to chat; type `/` to see commands.

| Command | What it does |
|---|---|
| `/help` (`/h`, `/?`) | List available commands. |
| `/status` | Show provider + task status. |
| `/provider` | Manage provider profiles (add, switch, list). |
| `/mcp` | Manage MCP servers (list, add, remove, tools). |
| `/workflow` | Build and run workflows (drives the [Architect](../architect/index.md)). |
| `/doctor` | Check the active connection. |
| `/new` | Clear chat history and start a fresh session. |
| `/theme` | Change the color theme. |
| `/clear` (`/cls`) | Clear the screen. |
| `/exit` (`/quit`, `/q`) | Leave Neurosurfer. |

## Provider profiles

Profiles are named provider configurations stored at `~/.neurosurfer/providers.json` (file mode
`0600`, secrets masked on display). Manage them from the shell or the REPL's `/provider`:

```bash
neurosurfer provider list          # list profiles
neurosurfer provider add           # add a profile (interactive)
neurosurfer provider use <name>    # set the active profile
neurosurfer provider delete <name> # remove a profile
```

When more than one profile is configured and none is confirmed yet, the REPL asks once which is the
default; after that (or after `/provider use`) it's settled.

## doctor

`neurosurfer doctor` verifies your configuration and that the active provider (and any enabled MCP
servers) are reachable — a quick first check when something isn't connecting.

## serve — the gateway

`neurosurfer serve` starts the [OpenAI-compatible gateway](../server/index.md):

```bash
neurosurfer serve --host 0.0.0.0 --port 8000
# proxy an upstream OpenAI-compatible backend:
neurosurfer serve --upstream-url http://localhost:1234/v1
```

| Flag | Purpose |
|---|---|
| `--host`, `--port` | Bind address (default `0.0.0.0:8000`). |
| `--upstream-url` | Proxy this OpenAI-compatible base URL. |
| `--upstream-api-key` | API key for the upstream server. |
| `--log-level` | Uvicorn log level. |
| `--workers` | Number of uvicorn workers. |
| `--reload` | Auto-reload (development). |
| `--no-docs` | Disable the `/docs` UI. |

See the [Server guide](../server/index.md) for registering agents and workflows programmatically.
