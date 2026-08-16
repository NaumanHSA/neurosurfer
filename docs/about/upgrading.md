# Upgrading

What changes when you move a working setup onto this release. Ordered by how quietly it bites:
the first three produce **no error at all** — the run goes green and the result is wrong or the
configuration looks empty.

The [Changelog](changelog.md) is the full list. This page is only the things that need an action
from you.

!!! info "Tested on Linux"
    This release's suite, live tests and every tutorial were run on Linux. The code
    is cross-platform and the Windows-specific defect known at the time — a
    timed-out child surviving because `os.killpg` does not exist there — is fixed,
    but the Windows suite was not re-run for this release.

---

## 1. A node is told what it names — nothing ambient

**This is the change most likely to break a workflow you already have, and it fails silently.**

A node's turn is now **what its own task text names, plus the outputs of the steps it declared in
`depends_on`** — and that is all. The block that recited every graph input underneath each node's
instructions is gone.

A workflow written against the old behaviour **still validates, still runs, and still reports
success**. The model simply answers as though it had been handed nothing:

```
"Please provide the topic you'd like me to research."
```

### What to check

A step that needs a graph input must **interpolate it**:

```python
# before — worked only because the engine recited every input under the prompt
GraphNode(id="researcher", kind="base",
          goal="Research the topic and produce 5 bullet points.")

# after — the step names what it needs
GraphNode(id="researcher", kind="base",
          goal="Research {topic} and produce 5 bullet points.")
```

A step that reads its input from an upstream node needs nothing — the `depends_on` edge still
carries it.

`validate_package` reports a **declared** input that no step names. It cannot see an input the
graph never declared, so a graph that relied on the ambient block without declaring anything is
invisible to it — those you find by reading.

### Why

It could not be made correct, only less wrong. A `map` body was handed the whole collection it was
iterating, once per item, beside the single item it was working on. A value the instruction had
already interpolated was printed again underneath it. And any rule for "which inputs matter to this
node" is a worse version of the one the author already wrote, in the placeholders of the
instruction.

---

## 2. MCP server configs live somewhere else

`McpStore.default()` moved:

| | Path |
|---|---|
| Before | `~/.neurosurfer/mcp.json` |
| Now | `<NEUROSURFER_HOME>/config/mcp.json`, default `./.neurosurfer/config/mcp.json` |

Two consequences, both silent — nothing errors, the server list is just **empty**:

- **Your existing servers are not found.** The old file is still on disk; nothing reads it.
- **The default is now relative to your working directory**, not your home directory. Launch from
  a different folder and you get a different (empty) configuration.

### What to do

Move the file, and set `NEUROSURFER_HOME` if you want one host-wide configuration the way it
behaved before:

```bash
export NEUROSURFER_HOME="$HOME/.neurosurfer"
mkdir -p "$NEUROSURFER_HOME/config"
mv "$HOME/.neurosurfer/mcp.json" "$NEUROSURFER_HOME/config/mcp.json"
```

Without `NEUROSURFER_HOME`, everything Neurosurfer writes — workflows, runs, traces, authored
tools, MCP config — sits under `./.neurosurfer/` beside the work. That is deliberate for
debugging; it is probably not what you want for a daemon. See
[Storage layout](../guides/configuration.md#storage-layout).

---

## 3. `.env` wins the provider argument

`LLM_PROVIDER=openai` in your `.env` could previously be overridden by a stored provider profile,
so a script could demonstrate a model other than the one its own configuration named. The `.env`
value now wins.

**Check this if a run suddenly uses a different model than you expect** — the fix is usually to
delete a stale profile with `neurosurfer provider delete <name>`, or to stop setting
`LLM_PROVIDER` if the profile was the one you actually wanted.

---

## 4. A truncated `base` step now fails instead of reporting success

A `base` node gets **one round of tool calls**. Asked to fetch a page and then write a file, it
would spend the round on the fetch, be refused the second call, and return an empty answer that the
run recorded as a **success**.

It now **fails**, naming the tools it did call and pointing at `react`.

This can turn a previously-green run red. That is the point — the run was not doing what it
claimed. The fix is nearly always to change the node's kind:

```python
GraphNode(id="fetch_and_save", kind="react",     # was "base"
          tools=["http", "write_file", "finish"], ...)
```

A truncated step that still produced text keeps it and does not fail.

---

## 5. `break_when` is gone; a loop stops for one reason

A loop has **one** stop condition, `until`. `break_when` has been removed.

```yaml
# before
- id: polish
  kind: loop
  break_when: "len(nodes.draft) < 200"

# after — a name defined in the graph's functions: sidecar
functions: helpers.py
nodes:
  - id: polish
    kind: loop
    max_iterations: 4
    until: draft_is_short
```

`until` is read as whichever of two things it is:

- **a function** — a callable, or the name of one in the graph's `functions:` file. It receives a
  `LoopIteration` and returns `True` to stop, or `(stop, "reason")` to also set the next
  iteration's `{feedback}`. This subsumes every expression `break_when` could hold and goes well
  past it: it gets the body's whole `GraphExecutionResult` plus the iteration index, history and
  loop vars, and it is real Python rather than a restricted evaluator.
- **plain English** — judged by an internal LLM decision after each iteration.

Which one a string is, is a **lookup, not a heuristic**: a name the sidecar defines is the
function, anything else is prose. A graph that declares a sidecar and names something absent from
it is an **error**, not a prompt — so a typo cannot quietly become an English condition sent to a
model.

---

## 6. Importing a built-in tool by submodule path

Built-in tools moved from `neurosurfer/tools/builtin/` to
`neurosurfer/registry/core/<domain>/`.

**Package-level imports still work** — `neurosurfer.tools.builtin` re-exports every tool:

```python
from neurosurfer.tools.builtin import ReadFileTool     # ✅ still fine
```

**Submodule paths do not:**

```python
from neurosurfer.tools.builtin.search import SearchTool               # ❌ gone
from neurosurfer.registry.core.filesystem.search import SearchTool    # ✅
```

The domains are `agent`, `data`, `database`, `filesystem`, `system`, and `web`. See the
[Tools Catalog](../guides/tools-catalog.md).

---

## 7. The executor is a package

`neurosurfer/graph/engine/executor.py` is now `executor/` — split into scheduler, iteration,
routing, deterministic kinds, io, and llm.

```python
from neurosurfer.graph.engine.executor import GraphExecutor   # ✅ unchanged
```

Reaching into the old module's internals by path is not supported and will not resolve.

---

## 8. Named-but-unconfigured exporters are skipped

`NEUROSURFER_EXPORTERS=otel` with no `OTEL_EXPORTER_OTLP_ENDPOINT` used to build the exporter
anyway, and the OpenTelemetry SDK filled in its own `http://localhost:4318` — so an install that
had pointed at no collector still opened one and paid to find out nothing was there.

The explicit list now requires the same connection variables auto-detection requires, and **warns
what is missing** instead of building. Passing a constructed instance to `register_exporter` still
bypasses the check, since that is a deliberate choice by the caller.

If you relied on the implicit localhost default, set it explicitly:

```bash
export OTEL_EXPORTER_OTLP_ENDPOINT="http://localhost:4318"
```

---

## 9. A declared input no step reads now refuses to run

**This is the only change here that breaks something already on disk rather than in your source.**

A workflow that declares an input and names it nowhere accepts a parameter and ignores it. That was
a warning, so it registered and ran; it is an error now, so it does not.

```
The workflow asks for 'article' but no step uses it, so the value a caller passes is ignored.
  → Name it in a step's instructions as {article}, or drop it from the workflow's inputs.
```

### What to check

Run the validator over your registry **before** upgrading a running system:

```python
from neurosurfer.graph.workflow.package import load_package
from neurosurfer.graph.workflow.validation import validate_package

for path in registry_dir.iterdir():
    report = validate_package(load_package(path))
    if not report.ok:
        print(path.name, report.summary())
```

Either interpolate the input in the step that needs it, or drop it from `inputs`. The workflow was
ignoring it either way — the change is that you now find out at the door instead of from a strange
answer. See [Validation](../graph/validation.md#an-input-no-step-reads-is-an-error) for the reads
that count, and for when the rule downgrades itself to a warning.

`GraphExecutor(..., validate=False)` skips the gate for one run if you need to ship first and fix
after.

---

## 10. Cost accounting is gone; tokens remain

`neurosurfer.llm.pricing` is deleted, along with `RunResult.cost()`,
`GraphExecutionResult.total_cost()`, and the `model` field on `RunResult` and
`NodeExecutionResult` — which existed only to feed them.

```python
# before
result.cost()                      # ✗ gone
graph_result.total_cost()          # ✗ gone

# now
result.usage.input_tokens, result.usage.output_tokens
result.usage.cache_read_input_tokens, result.usage.cache_creation_input_tokens
graph_result.total_usage()
```

Nothing replaces it. This framework counts tokens; what they cost belongs to your trace backend.
Vendor rates vary by contract and region and go stale silently, and the exporters already receive
the model name alongside `Usage` — so Langfuse and OpenTelemetry attribute spend from tables they
maintain. See [Providers](../guides/providers.md#token-usage).

---

## Smaller behaviour changes

- **Validation runs on every run**, not only at registration. A graph that cannot run is refused
  before a model is called rather than partway through.
- **A `react` node that ends with `finish()` returns what it finished with**, rather than the
  loop's last assistant text.
- **A run handed an *undeclared* value no step reads logs a warning** naming the key and the fix.
  Quiet where it cannot know: `function`, `python` and `tool` nodes receive the whole mapping as
  keyword arguments, so their presence silences the check. A *declared* input nothing reads is the
  stricter case above — that one blocks.
- **The "fewer than three LLM nodes is under-designed" heuristic is gone.** A correct two-step
  workflow is no longer refused registration.
- **Trace export never runs on the agent's thread.** Delivery is best-effort over a bounded queue;
  call `neurosurfer.observability.dispatch.drain()` in tests that need to observe it.
- **A node's system prompt is identical for every node of every graph** — the task moved into the
  user turn, so prompt caching can fire.

---

## Next

- [Changelog](changelog.md) — the complete list.
- [Graph & Workflows](../graph/index.md) — the runtime these changes affect most.
- [Configuration](../guides/configuration.md) — every variable and the storage layout.
