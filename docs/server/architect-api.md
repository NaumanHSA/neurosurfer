# Architect API

Drives the [Architect](../architect/index.md) over HTTP: plan a workflow, build one, answer the
questions a build asks part-way through, and follow it live.

The interesting part is that a build is **long-running and interactive**. It can park to ask a
person for an approval, a credential, or a clarification, and resume when the answer arrives — so
the API is a build *resource* with an event stream, not a request that returns a workflow.

## Plans

| Method | Route | Does |
|---|---|---|
| `POST` | `/v1/architect/plans` | Produce a plan for an intent, without building it. |

```http
POST /v1/architect/plans
{ "intent": "Summarise a CSV and email the result", "answers": { ... } }
```

Planning separately is what lets a plan be **reviewed before it is built**. The plan you get back
can be handed straight to a build, which then skips its own planning call rather than
re-inventing it.

## Builds

| Method | Route | Does |
|---|---|---|
| `POST` | `/v1/architect/builds` | Start a build. `202` |
| `GET` | `/v1/architect/builds` | List builds. |
| `GET` | `/v1/architect/builds/{build_id}` | One build's state. |
| `GET` | `/v1/architect/builds/{build_id}/events` | **SSE** — follow it live. |
| `POST` | `/v1/architect/builds/{build_id}/respond` | Answer what the build is waiting on. |
| `POST` | `/v1/architect/builds/{build_id}/cancel` | Stop it. |

### Starting a build

```http
POST /v1/architect/builds
{
  "intent": "Summarise a CSV and write the result to a file",
  "verify": "required",
  "plan":   { ... },          // optional — a reviewed plan
  "refines": "old_workflow",  // optional — change an existing one
  "clarify": "auto",
  "approve_tools": true,
  "review_plan": false
}
```

Returns `202` with a build id.

| Field | Effect |
|---|---|
| `intent` | The plain-English request. |
| `verify` | `off` · `encouraged` · `required`. See [Verification](../architect/verification.md#gating). |
| `plan` | Build this plan instead of planning again. |
| `refines` | Read the intent as a change to an existing registered workflow. |
| `clarify` | Whether the build may ask clarifying questions. |
| `approve_tools` | Require approval before an authored tool is registered. |
| `review_plan` | Require plan approval before building. |
| `history` | Prior conversation turns. |

### A build that parks

When the build hits something only a person can answer — approve this authored tool, supply this
credential, which of these did you mean — it emits an **interaction** on the event stream and
waits.

```http
POST /v1/architect/builds/{build_id}/respond
{ "interaction_id": "…", "value": "…" }
```

The build resumes from where it stopped. This is the same callback set
[`ArchitectAgent`](../architect/agent.md#callbacks) exposes in-process — the HTTP layer turns each
callback into a parked interaction rather than a blocking prompt.

### Following a build

```
GET /v1/architect/builds/{build_id}/events    → text/event-stream
```

Carries progress, per-node events, interactions, and the terminal outcome.

## Repairs

| Method | Route | Does |
|---|---|---|
| `POST` | `/v1/architect/repairs` | Propose fixes for a workflow that failed. |
| `POST` | `/v1/architect/repairs/apply` | Apply proposed fixes. |

Proposal and application are separate calls so a repair can be **shown before it is made**.

## Outcomes

A build ends in one of the states the
[terminal contract](../architect/agent.md#the-terminal-contract) allows: a registered package path,
or a blocked reason. A build that produced neither is an error, not a silent success.

A blocked build is a [refusal](../architect/grounding.md#refusal-is-an-outcome-not-a-failure) with a
precise reason — treat it as information, not as a failure to retry blindly.

## Next

- [The Agent](../architect/agent.md) — the in-process equivalent of all of this.
- [Workflows API](workflows-api.md) — running what a build registers.
- [Deployment](deployment.md) — auth and hosting.
