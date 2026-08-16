# MCP Discovery

When a workflow needs a capability nothing installed provides, the answer is not "I can't do that"
— it is *"install this server and set these two variables"*. Discovery is what turns a gap into an
instruction.

Three things, and they are the same three whether a person or the [Architect](../architect/index.md)
is asking: **search**, **what would this cost me**, and **install it**.

```python
from neurosurfer.mcp.registry import search_registry, registry_detail, install_config
from neurosurfer.mcp.registry import credential_requirements
```

This client is FastAPI-free on purpose. The logic used to live inside a route module and raise
`HTTPException`, which made it reachable from a browser and from nowhere else — while the Architect
runs in-process for CLI and library callers that have no web server at all. Both callers now get the
same answers from the same code; the route module maps `McpRegistryError` to HTTP.

## Search

```python
hits = search_registry("send an email", limit=10)
for h in hits:
    print(entry_summary(h))
```

Results are cached for 15 minutes. The registry API is slow enough — it sits on a ~14s plateau for
roughly a third of calls — that a short TTL is paid for in seconds of staring at a spinner.

## What it needs before it will run

This is the piece that did not exist before, and the difference between *"I can't do Gmail"* and
*"install `…/gmail-mcp` and set `GOOGLE_CLIENT_ID` and `GOOGLE_CLIENT_SECRET`"*.

```python
reqs = credential_requirements(entry)
for r in reqs:
    print(r.name, r.required, r.secret, r.description)
```

The registry declares what a server needs to run — `environmentVariables` on a package, `headers` on
a remote, each with `isRequired` and `isSecret` — and that was being thrown away.

### Ask for the placeholder, not the slot

A templated credential's **name** is where the value goes (`Authorization`); the thing a person can
actually supply is the **placeholder inside it** (`smithery_api_key`).

A checklist that says "provide `Authorization`" is asking for something nobody possesses under that
name. Conflating the two is how the Architect came to refuse a workflow over a header whose declared
value was `Bearer {smithery_api_key}` — a placeholder naming a key already stored in settings, with
the code to fill it already written.

### Declared ≠ blocked

A registry entry declaring its requirements is a statement **about the server**, not about whether
*you* can run it. Those are different questions and only the second should ever block a build.

```python
from neurosurfer.mcp.credentials import blocking, satisfied_from, use_credentials

with use_credentials(my_secrets):
    still_needed = blocking(reqs)      # what this account genuinely cannot supply
```

Availability is **ambient and account-scoped**: the credentials an account holds are resolved once
on the request thread and bound for the work, because an Architect build runs on a raw thread and
inherits no context of its own.

## Install

```python
cfg = install_config(entry)      # → an McpServerConfig ready to add to the store
```

`default_runtime(entry)` picks the runtime when an entry offers several.

## Discovery engines

Two sources, and which one is active is a setting.

```python
from neurosurfer.mcp.sources import active_source, set_active_source, use_source, all_sources
```

### Official — the default

The [official MCP registry](https://registry.modelcontextprotocol.io). Always available, no key,
keyword search.

### Smithery — opt-in, experimental

Faster and semantically stronger. Measured against the official registry on the same machine, six
identical repeated queries each: **median 0.68s vs 11.94s**, and consistent rather than erratic.

It also answers two questions the official index structurally cannot:

- **What a server exposes, before anything is installed** — `tools` with input and output schemas,
  plus `prompts` and `resources`.
- **How much a server is used** (`useCount`, observed 0 → 25,335) — the only popularity signal
  available anywhere in the ecosystem, and the one honest way to rank a search where thousands of
  entries share a keyword.

Its search is genuinely semantic: `"send a text message"` returns telnyx and two SMS gateways with
no literal token match.

**Why it stays opt-in:** it needs the user's own API key, it is closed source, and servers found
through it run on its infrastructure. Choosing it without a key is **refused** rather than silently
returning nothing.

!!! warning "`security.scanPassed` is not a health check"
    It exists in the Smithery schema and came back `null` on every entry sampled, authenticated and
    not. Do not read it as a security verdict.

## The runtime

Non-CLI callers — the workflow runner (synchronous), the Architect, the server — need MCP tools
without owning an event loop that outlives the call.

```python
from neurosurfer.mcp.runtime import ensure_mcp_tools
statuses = ensure_mcp_tools()      # sync, idempotent, safe to call opportunistically
```

No configured servers → no-op; already running → returns the current statuses.

**One manager per server.** Each configured server gets its own `McpManager` driven by its own
long-lived task: `connect → wait for stop → aclose`. That shape is required — the anyio stdio
transports demand same-task setup and teardown — *and* it is what makes independent start/stop
possible: stopping one server no longer tears down the others.

The host loop stays alive for the process, so a tool's cross-loop marshalling works from any thread
or loop, graph-executor threads included.

!!! note "Starting a run connects every enabled server"
    Which is why [workflow requirements](../graph/validation.md#what-a-workflow-requires-supplied)
    include enabled servers whether or not the workflow names one of their tools — one of them
    missing a value fails the run regardless.

## Next

- [MCP](mcp.md) — connecting a server you already know about.
- [Tool Registry](tool-registry.md) — how an imported server's tools rank against core ones.
- [The Architect](../architect/index.md) — the caller this exists for.
