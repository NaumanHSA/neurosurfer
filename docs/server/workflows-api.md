# Workflows API

Beyond the OpenAI-compatible chat routes, the gateway exposes the **workflow runtime** over HTTP:
list and manage registered [workflow packages](../graph/packages.md), start runs, and follow them
live.

Requires the `serve` extra, and is subject to the same `NS_API_KEYS` bearer auth as the rest of
`/v1/*`.

## Workflows

| Method | Route | Does |
|---|---|---|
| `GET` | `/v1/workflows` | List registered packages. |
| `GET` | `/v1/workflows/{name}` | One package — metadata and its graph. |
| `GET` | `/v1/workflows/{name}/requirements` | What must be supplied before it can run. |
| `POST` | `/v1/workflows` | Register a package. `201` |
| `PUT` | `/v1/workflows/{name}` | Replace one. |
| `DELETE` | `/v1/workflows/{name}` | Remove one. |
| `POST` | `/v1/workflows/validate` | Validate a graph **without** registering it. |

### Requirements before a run

```http
GET /v1/workflows/report_builder/requirements
```

Returns each required value, where it is asked from (`node:<id>` or `server:<name>`), whether it is
satisfied, and — when a value **is** set but unusable — the `problem` saying why.

Set is not the same as usable. See
[Validation](../graph/validation.md#what-a-workflow-requires-supplied).

### Validate a draft

```http
POST /v1/workflows/validate
{ "graph": { ... }, "name": "draft" }
```

The same gate registration uses, so a client can check a graph while it is being edited rather than
discovering the problem at registration.

## Runs

| Method | Route | Does |
|---|---|---|
| `POST` | `/v1/workflows/{name}/runs` | Start a run. `202` |
| `GET` | `/v1/runs` | List runs. |
| `GET` | `/v1/runs/{run_id}` | One run's status and result. |
| `GET` | `/v1/runs/{run_id}/nodes/{node_id}` | One node's output. |
| `GET` | `/v1/runs/{run_id}/trace` | The run's trace, groupable per node. |
| `GET` | `/v1/runs/{run_id}/events` | **SSE** — follow the run live. |
| `POST` | `/v1/runs/{run_id}/resume` | Supply values and continue. `202` |
| `DELETE` | `/v1/runs/{run_id}` | Delete a run record. |

### Starting a run

```http
POST /v1/workflows/report_builder/runs
{ "inputs": { "article": "…" } }
```

Returns `202` with a run id — runs are asynchronous and durable, not request-scoped.

**Pass the names the package declares.** A value no step reads logs a warning naming the key; see
[State](../graph/state.md#templates).

### Following a run

```
GET /v1/runs/{run_id}/events        → text/event-stream
```

### The trace

`GET /v1/runs/{run_id}/trace` returns the run's steps. Each step carries a `node_id`, which is what
lets a reader group a run into a **per-node tree** rather than a flat list — and step ids are
unique even under a concurrent `map`. Pass a node id to filter to one node.

### Resuming an `awaiting_input` run

A workflow containing an [`input` node](../graph/node-kinds.md#input) pauses rather than failing:
the run finishes `awaiting_input`, and a client supplies the values.

```http
POST /v1/runs/{run_id}/resume
{ "values": { "approved": "yes" } }
```

!!! warning "Resuming re-runs the whole graph"
    A node before the `input` node runs twice. Worth knowing before you put an expensive step
    upstream of a human decision.

## Next

- [Architect API](architect-api.md) — building workflows over HTTP.
- [Workflow packages](../graph/packages.md) — what is being registered and run.
- [Deployment](deployment.md) — auth, workers, and containers.
