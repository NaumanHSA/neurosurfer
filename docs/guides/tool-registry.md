# Tool Registry

The registry answers one question: **what is available, what can it do, and what does it need?**

It matters because the alternative was matching plain English against tool descriptions and picking
wrong — a database-migration tool for "generate chart images", a hosted gateway for a database on
`localhost`. Resolution here matches a **declared tag**, and where a tool cannot be reached the
manifest says so.

```python
from neurosurfer.registry import manifests, providers_of, unsatisfied_capabilities
```

## One lookup path, three kinds of backing

| Origin | Backed by |
|---|---|
| `core` | A Python class in `neurosurfer/registry/core/<domain>/`. |
| `imported` | A verified MCP server, plus the tool name it knows. |
| `authored` | Generated source, kept after it passed its sandbox. |

Callers ask for a **capability**, not a tool, and get back manifests. That is the whole point: it
lets resolution stop caring where a tool came from.

## The capability vocabulary

A closed, hand-written set of tags. A tag is a *capability*, not a tool and not an implementation —
`db.query` is "run a read-only query against a database server" whoever provides it.

| Group | Tags |
|---|---|
| Files | `file.read`, `file.write`, `file.edit`, `file.list`, `file.search` |
| Structured data | `data.inspect` |
| Databases | `db.connect`, `db.schema`, `db.query` |
| Network | `web.request`, `web.browse`, `web.search` |
| This machine | `system.shell` |
| Artifacts | `chart.render`, `pdf.render` |
| People | `message.send`, `inbox.read` |

```python
from neurosurfer.registry import CAPABILITIES, describe, is_known

describe("file.write")     # 'Create or overwrite a file at a given path.'
```

### Why it is closed

Letting each tool invent its own tags re-admits the matching problem one level up: two tools tagged
`db.query` and `sql.run` would need a scorer to reconcile, and **a scorer guessing at tags is what
this whole layer exists to replace.**

Growth is expected and cheap — add the constant, tag the tools, map the phrases. What is not cheap
is a tag that means two things, so each one carries a sentence saying exactly what it covers.

### Tags with no provider, on purpose

`chart.render` and `pdf.render` are declared before anything claims them. The honest answer to
"generate a chart" is *"no tool here has that capability"* — which is a job to author or import,
and a far better answer than whichever tool shared the most words with the request.

```python
unsatisfied_capabilities()     # ['chart.render', 'inbox.read', 'message.send', 'pdf.render']
```

**Naming the gap is what turns it into an authorable job.**

## Finding a provider

```python
providers_of("file.write")                          # best-known first
providers_of("db.query", reaches_localhost=True)    # only tools that can see this machine
```

Ranking is **verified first**, then `core` over `authored` over `imported` — a tool that has
demonstrably run beats one that merely claims the tag.

`reaches_localhost` filters on where the tool runs. Pass `True` when the workflow targets something
on this machine: a hosted service is a perfectly good tool that simply cannot see `localhost:1433`,
and offering one for a database in a local container wasted an afternoon before this field existed.

## Manifests

`manifests()` reads the **live catalog** rather than a stored list, so an MCP server started thirty
seconds ago is included — and included *as what it is*, with no capabilities and an unknown
credential story, rather than as an equal candidate to a core tool that has declared both.

A `ToolManifest` carries identity (`name`, `title`, `description`, `icon`), `origin`, `capabilities`,
`runtime`, `secret_inputs`, `credential_help`, `input_schema`, `settings_schema`, `read_only`, a
`verified` record, and per-operation `OperationManifest`s.

```python
from neurosurfer.registry import manifest_for
m = manifest_for(some_tool)
m.covers("db.query")     # does this tool declare the tag?
```

## Prose search is a tiebreak, not the signal

It remains, as a fallback for untagged tools — an MCP server nobody has tagged still has to be
findable. It is never the primary signal.

## Self-knowledge

On top of the registry sits a content-hash-versioned **capability manifest** and a `KnowledgeBase`,
which is what the [Architect](../architect/index.md) reads when deciding whether it can build
something. A need resolves against a declared tag rather than against words a description happens
to share.

Because it is versioned by content hash, adding or retagging a tool invalidates it automatically —
there is no cache to remember to clear.

## Icons

Every tool resolves to an icon, served to front-ends as a URL:

```python
from neurosurfer.registry import icon_slugs, resolve_icon, icon_bytes
```

Empty `icon` falls back to the tool's registry **domain**, so an MCP tool nobody has ever seen still
reads as *database* or *web* rather than as a generic box. Per-tool artwork stays optional instead
of being a prerequisite for appearing in a palette.

## What the registry holds

```python
from neurosurfer.registry import registry_report
registry_report()
# {'tools': 19, 'by_origin': {...}, 'untagged': [...], 'unsatisfied': [...]}
```

`untagged` is the maintenance list — a tool nobody can resolve against is a tool only prose search
can find.

## Next

- [Tools guide](tools.md) — declaring `capabilities` on your own tool.
- [Tools Catalog](tools-catalog.md) — what ships, and what each declares.
- [MCP Discovery](mcp-discovery.md) — importing a tool that does not exist yet.
- [Validation](../graph/validation.md#capability-grounding) — how a workflow is checked against this.
