# MCP Servers

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NaumanHSA/neurosurfer/blob/main/tutorials/04_mcp_servers.ipynb)

Connect an external Model Context Protocol server and expose its tools to an agent.

## Minimal walkthrough

Install the extra, then connect a server through the MCP manager — its remote tools become ordinary
`Tool`s in your pool:

```bash
pip install -U "neurosurfer[mcp]"
```

```python
from neurosurfer.mcp import McpManager
from neurosurfer.tools import default_pool

manager = McpManager()
await manager.connect("filesystem", command="npx", args=["-y", "@modelcontextprotocol/server-filesystem", "/data"])

pool = default_pool()
pool.extend(manager.tools())     # MCP tools now callable by the agent

agent = AgenticLoop(provider=provider, tools=pool, system_prompt="…",
                    guardrails=Guardrails(), io=AutoIO(), cwd=Path.cwd())
```

MCP tool calls are gated by `mcp_policy` in [`Guardrails`](../learn/permissions.md), like any other
action.

## Full notebook

The [Colab notebook](https://colab.research.google.com/github/NaumanHSA/neurosurfer/blob/main/tutorials/04_mcp_servers.ipynb)
walks through connecting a server, listing its tools, and persisting server configs.

**Next:** [MCP guide](../guides/mcp.md) · [Tutorial 5 →](insight-engine.md)
