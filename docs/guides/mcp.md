# MCP

The **Model Context Protocol** (MCP) is an open standard for exposing tools to LLM apps.
Neurosurfer's MCP client connects to external MCP servers and surfaces their tools as ordinary
Neurosurfer [`Tool`](tools.md) objects — so an agent can use them exactly like built-ins.

This page covers connecting a server you already know about. To **find** one — search the registry,
see what it needs, and install it — see [MCP Discovery](mcp-discovery.md).

Install the extra:

```bash
pip install "neurosurfer[mcp]"
```

Everything is in `neurosurfer.mcp`:

```python
from neurosurfer.mcp import McpManager, McpTool, ServerStatus
from neurosurfer.config.mcp import McpServerConfig
```

## Connect to a server

Describe each server with an `McpServerConfig` (stdio or HTTP transport), then open an `McpManager`.
Use it as an async context manager so connections are cleaned up:

```python
import sys
from neurosurfer.config.mcp import McpServerConfig
from neurosurfer.mcp import McpManager

cfg = McpServerConfig(
    name="tutorial",
    transport="stdio",
    command=sys.executable,      # how to launch the server process
    args=["./my_mcp_server.py"],
)

async with McpManager([cfg]) as mgr:
    for st in mgr.status():
        print(f"{st.name!r}: connected={st.connected}, tools={st.tools}")

    for t in mgr.tools():        # each remote tool wrapped as a Neurosurfer Tool
        print(f"  · {t.name} — {t.description.splitlines()[0]}")
```

HTTP servers are configured the same way with `transport="http"` and a `url` (plus optional
`headers` for auth):

```python
McpServerConfig(
    name="github",
    transport="http",
    url="https://api.githubcopilot.com/mcp/",
    headers={"Authorization": "Bearer ${GITHUB_TOKEN}"},
)
```

## Give MCP tools to an agent

`mgr.tools()` returns a list of `Tool`s — drop them into a `ToolPool` alongside whatever built-ins
you want (usually at least `finish`):

```python
from neurosurfer.agents import AgenticLoop, Guardrails
from neurosurfer.tools import ToolPool
from neurosurfer.registry.core.agent.finish import FinishTool

async with McpManager([cfg]) as mgr:
    pool = ToolPool([*mgr.tools(), FinishTool()])

    agent = AgenticLoop(
        provider=provider,
        tools=pool,
        system_prompt="Use the available tools to answer, then call finish.",
        guardrails=Guardrails(max_turns=8),
        io=AutoIO(),               # see the Agents guide
        cwd=".",
    )
    async for ev in agent.run("What's the weather in London?"):
        if hasattr(ev, "text"):
            print(ev.text, end="", flush=True)
```

MCP tool calls are gated by the agent's `mcp_policy` guardrail (`gated` / `open` / `denied`), and
each remote tool carries read-only/destructive flags from its MCP annotations — so the
[permission layer](agents.md#permissions-and-guardrails) treats them just like local tools.

## Persisting server configs

`McpStore` saves server configs to disk so the [CLI](../cli/index.md) (`/mcp` commands) and your apps can
share them:

```python
from neurosurfer.config.mcp import McpStore, McpServerConfig

store = McpStore.default()          # resolves the host-level config path for you
store.add(McpServerConfig(name="tutorial", transport="stdio",
                          command="python", args=["./my_mcp_server.py"]))
for s in store.list():
    print(s.summary())
```

`McpStore.default()` resolves to `config/mcp.json` under the single data root —
`./.neurosurfer/` unless `NEUROSURFER_HOME` says otherwise. See
[Storage layout](configuration.md#storage-layout). That file is **host-level and
deliberately shared**: an MCP server is a process the gateway spawns as its own OS
user, so a per-workspace copy would suggest an isolation that does not exist.

The `mcp` SDK is imported lazily — importing `neurosurfer.mcp` doesn't require it; the dependency is
only touched when a manager actually connects.

## Outside the CLI

The CLI owns its own `McpManager` on its main event loop. Everything else — the workflow runner,
the Architect, the server — uses the process-lifetime runtime instead:

```python
from neurosurfer.mcp.runtime import ensure_mcp_tools

ensure_mcp_tools()      # sync, idempotent; no configured servers → no-op
```

See [MCP Discovery](mcp-discovery.md#the-runtime) for how it hosts one manager per server.

## Next

- [MCP Discovery](mcp-discovery.md) — finding, vetting, and installing a server.
- [Tool Registry](tool-registry.md) — how imported tools rank against built-in ones.
- [Tools](tools.md) — the `Tool` interface an MCP tool is wrapped as.
