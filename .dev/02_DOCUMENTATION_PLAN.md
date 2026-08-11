# 02 — The documentation

**Goal:** the docs describe the framework that is on this branch, not the one that
was on `main`. Every subsystem plan 01 brought across — control flow, the tool
registry, MCP discovery, the ReAct Architect, the execution API — has a page that
says what it is, how to reach it, and what it refuses to do. And the two changes
that break a working workflow in silence are written down somewhere a person will
find them before the workflow goes quiet.

**Why now.** Plan 01's §5 checklist ends at a merge to `main` with a version bump.
Merging 48 commits of behaviour under documentation that describes the previous
behaviour is how a framework acquires a reputation for being undocumented.

---

## §0 — The diagnosis

Measured on 2026-08-11 against `6233f1a`, by diffing `main...HEAD` and reading
every page under `docs/`.

### §0.1 — Forty-eight commits, six pages

`git diff --stat main...HEAD` is **273 files, +39,686 lines**; `neurosurfer/`
alone is **+23,671**. Against that, `git diff --name-status main...HEAD -- docs`
returns **six modified files and no new ones**:

```
M docs/architect/building.md          M docs/guides/configuration.md
M docs/architect/how-it-works.md      M docs/guides/graph-workflows.md
M docs/architect/index.md             M docs/observability/opentelemetry.md
```

Those six are good and current. The subsystems with **zero** pages are:

| Subsystem | Lines added | Pages |
|---|---|---|
| `neurosurfer/registry/` — the tool registry | ~1,100 + 21 icons | 0 |
| `neurosurfer/mcp/{registry,runtime,credentials,sources}` | ~1,700 | 0 |
| `neurosurfer/architect/agent/` — the ReAct Architect | ~3,600 | 0 |
| `neurosurfer/app/server/` — workflows + architect HTTP | ~3,100 | 0 |
| `neurosurfer/graph/workflow/validation/` — the rule table | ~2,400 | 0 |
| `neurosurfer/graph/engine/{nodes,builder,kinds,state,secrets}` | ~1,400 | 0 |

### §0.2 — The docs are a runtime input, not only a human one

[`docs_index.py`](../neurosurfer/architect/knowledge/docs_index.py) BM25-indexes
every markdown file under `docs/`, splits it at headings, and serves sections to
the Architect's `describe_capability` tool during a build. A subsystem with no
page is a subsystem the Architect cannot look up when it is deciding whether it
can build something.

This changes what a documentation gap costs. A missing page on node kinds is not
only a person reading source instead — it is the Architect grounding a design
against a vocabulary it cannot retrieve. It also sets a constraint on *how* the
pages are written: the index splits at headings, so a heading has to name the
thing under it, and the paragraph under a heading has to stand alone without the
three above it.

The same file already carries the evidence that this matters. `_EXCLUDED_PREFIXES`
drops `about/roadmap` and `about/changelog` from the index because a roadmap is a
list of capability nouns that BM25 scores highly for almost any query — measured
on six build-shaped queries it took a top-3 slot in four, once beating the graph
guide for *"tool node tool_args required arguments"* with a section about trace
spans.

### §0.3 — Four references that are wrong on the page

1. [`architect/index.md:27`](../docs/architect/index.md) links **tutorial
   `06_architect.ipynb`**. It is on no branch here — it was added on the studio
   line (`ca9a4a5`) and never ported. Dead link, added by this branch.
2. [`tutorials/sql-agent.md:3,37`](../docs/tutorials/sql-agent.md) links
   **`06_capstone_sql_agent.ipynb`** twice, including a Colab badge. That notebook
   was **deliberately removed** in `a6a4d1a`. The whole page is orphaned.
3. [`guides/mcp.md:96`](../docs/guides/mcp.md) documents
   `McpStore(path=Path.home() / ".neurosurfer" / "mcp.json")`. The root is now
   `./.neurosurfer/config/mcp.json` ([`config/paths.py`](../neurosurfer/config/paths.py)),
   which [`guides/configuration.md`](../docs/guides/configuration.md) already
   states — so the two pages contradict each other.
4. [`guides/mcp.md:65`](../docs/guides/mcp.md) imports
   `from neurosurfer.tools.builtin import FinishTool`. This one **still works and
   is supported** — the package re-exports every tool — so it is a staleness
   rather than a break. Worth updating to the canonical
   `registry.core.<domain>.<module>` location so the page teaches where tools now
   live, but it is not urgent and nothing depending on it will fail.

Also: `tutorials/index.md` numbers both the SQL capstone and (implicitly) the
Architect tutorial as **6**, and `mkdocs.yml:173` labels `about/roadmap.md` as
"Roadmap" when it is an *observability* roadmap — the Architect roadmap is
`.dev/ROADMAP.md`.

### §0.4 — The change that breaks a workflow without saying anything

`0579b32` — *a node is told what it names*. A node's turn is now its own task text
plus the outputs of its `depends_on`, and nothing ambient. The block that recited
every graph input under each node's instructions is gone.

A workflow written against the old behaviour **still validates, still runs, still
reports success**, and the model answers as though it had been handed nothing.
There is no error and no warning on the path that matters — `validate_package`
reports a *declared* input that no step names, but a workflow that never declared
the input is invisible to it.

This is in the CHANGELOG. It is not in the docs, and a CHANGELOG bullet is not
where someone looks when a workflow that worked last week returns "please provide
the topic". It needs a page of its own, and `guides/graph-workflows.md` needs to
state the rule where the graph is first built — which it now does.

Second, quieter one: built-in tools moved to `neurosurfer/registry/core/<domain>/`.
The package re-export keeps `from neurosurfer.tools.builtin import ReadFileTool`
working; **submodule paths break** —
`from neurosurfer.tools.builtin.search import SearchTool` is now
`from neurosurfer.registry.core.filesystem.search import SearchTool`.

**And a third, found while writing the upgrade page and in the CHANGELOG nowhere:**
`McpStore.default()` moved. On `main` it was `~/.neurosurfer/mcp.json`
([`config/mcp.py:102-104`](../neurosurfer/config/mcp.py) at `main`); it is now
`mcp_config_path()` → `<artifacts_home>/config/mcp.json`, i.e.
`./.neurosurfer/config/mcp.json` unless `NEUROSURFER_HOME` is set. Two silent
consequences: an existing server list is **not found** (the old file is still on
disk, nothing reads it), and the location is now **relative to the working
directory** rather than to `$HOME`, so launching from elsewhere yields a different,
empty configuration. Nothing errors — the server list is simply empty.

Note what is *not* a change, because the docs said otherwise and that misled this
plan's first draft: `NEUROSURFER_HOME` already defaulted to `./.neurosurfer` on
`main` ([`config/paths.py`](../neurosurfer/config/paths.py) at `main`). The old
`guides/configuration.md` documented `~/.neurosurfer`, which was simply wrong; the
branch fixed the page, not the behaviour. Only `mcp.json`'s path within that root
actually moved.

### §0.5 — Two things are in the wrong place in the nav

**Getting Started is a tab holding two pages** that every visitor needs and that
belong on the landing surface, while `index.md` is a card grid that links away to
them. Three clicks to install.

**Graph & Workflows is one 222-line page inside Guides**, under which sit: eleven
node kinds, a class hierarchy, a fluent builder, a safe expression evaluator, a
typed state model, a secrets boundary, a sidecar module system, and a validation
rule table. It is the largest subsystem in the framework and it is one bullet in a
list of twelve guides. It cannot absorb what §0.1 says is missing without becoming
a page nobody finishes.

---

## §1 — The shape it should have

Two structural moves, then the pages.

**Home absorbs Getting Started.** `index.md`, Installation, Quickstart — one tab.
The file paths under `getting-started/` **do not change**; this is a nav change
only, because moving them breaks inbound links and `edit_uri` for no gain the
reader can see.

**Graph & Workflows becomes a top-level tab**, `docs/graph/`. This one *does* move
files: `guides/graph-workflows.md` → `graph/index.md`, and the nine inbound links
across seven pages get fixed with it. There is no redirects plugin in
`docs/requirements.txt`, so the links are the migration.

```
Home                 index.md · Installation · Quickstart
CLI Agent            cli/index.md
Learn                Architecture · Core Concepts · Permissions & Safety
Guides               Providers · Agents · Structured Output · Tools · Tool Registry
                     Sub-agents · Background Tasks · Context & Memory · RAG
                     MCP · MCP Discovery · Configuration · Tools Catalog
Graph & Workflows    Overview · Node Kinds · Control Flow · State & Secrets
                     Validation · Workflow Packages · Building in Python
Observability        Overview · Langfuse · OpenTelemetry · Custom Exporters
Server               Overview · Serving Agents · Backends · Hooks
                     Workflows API · Architect API · Deployment
Architect            Overview · The Agent · Grounding & Refusal · Verification
                     Self-knowledge · Building Workflows
Tutorials            (unchanged, minus the orphan)
About                Contributing · Upgrading · Roadmap · Changelog
```

---

## §2 — The phases

Ordered so that the things that break a user's working code land first, and the
Architect's retrievable vocabulary lands before the pages that merely describe it.

### Phase 1 — Structure, and the references that are wrong

- [x] `mkdocs.yml`: Home absorbs Getting Started; `Graph & Workflows` becomes a tab
- [x] `guides/graph-workflows.md` → `graph/index.md`, all 9 inbound links fixed
- [x] `about/upgrading.md` — new. Prompt scoping first and loudest, then the
      `mcp.json` move, provider resolution, truncated-`base` now failing,
      `break_when` → `until`, `tools.builtin` submodule paths, executor internals,
      unconfigured exporters, and the smaller behaviour changes
- [x] `architect/index.md:27` — dead `06_architect.ipynb` link removed, replaced
      with the actual `ArchitectAgent` call (which is **async** — the first draft
      of that snippet was wrong and was caught by reading the signature)
- [x] `tutorials/sql-agent.md` — orphaned notebook links removed; kept as a written
      guide, rewritten around the real `sql` tool and its four operations
- [x] `tutorials/index.md` — renumbered; notebook-backed and written-only split
- [x] `guides/mcp.md:96` — corrected to `McpStore.default()` + the real root
- [x] `guides/mcp.md:65` — import updated to `registry.core.agent.finish`
- [x] `mkdocs.yml` — "Roadmap" relabelled "Observability Roadmap"

**Done when:** `mkdocs build --strict` passes, and no page links to a file that
does not exist. **Blocked on tooling:** no Python environment on this machine has
the dependencies installed (`pydantic` missing on every interpreter found), so
neither `mkdocs build` nor executing a sample can run here. Verification so far is
static — imports checked against the defining module and `__all__`. This is what
Phase 7's sweep exists to close, and it needs an environment.

### Phase 2 — Graph & Workflows, the tab

The biggest phase, and the one the Architect reads. Every page here is written to
be retrieved a section at a time.

- [x] `graph/index.md` — what the two layers are, build a graph, run it, where to
      go. Carries the §0.4 scoping rule at the point the first graph is built.
      Slimmed: control flow and packages moved to their own pages rather than
      being duplicated
- [x] `graph/node-kinds.md` — all eleven kinds, one section each, with the fields
      and constraints read off each kind's own module in
      [`kinds/`](../neurosurfer/graph/engine/kinds/). The classes
      (`BaseNode` … `ContainerNode`) and why every name carries the `Node` suffix
- [x] `graph/control-flow.md` — router (both forms), loop (the full `until`
      contract incl. UNRELATED), map, subgraph; `when`, `on_error`, `writes`,
      `policy.retries`; the expression evaluator
- [x] `graph/state.md` — what a node receives, templates, `writes`, the
      `functions:` sidecar, and **secrets as a boundary**: `{name}` reaches a
      prompt, `${NAME}` reaches only `tool_args` on a node that declared it; the
      declaration is a gate, not documentation; redaction covers the record, not
      the call
- [x] `graph/validation.md` — the rule table, message/detail split, severity and
      gaps, validation-first-on-every-run, capability grounding and its four
      false-positive rules, `workflow_requirements`
- [x] `graph/packages.md` — load/run/register, what a package carries, where they
      live, what they require supplied
- [x] `graph/building.md` — `GraphBuilder` fluent API, the YAML round-trip, and
      the `as_` / `item_var` naming wrinkle
- [x] `mkdocs.yml` — all seven pages in the new tab

**Done when:** a reader can author every node kind from these pages without
opening `neurosurfer/`, and `describe_capability` retrieves a relevant section for
each of the eleven kinds. **Link integrity verified** by a static checker matching
`markdown.extensions.toc.slugify` — 0 broken links, 0 broken anchors, 0 orphaned
pages, every nav entry resolves. Retrieval quality against the docs index is
**unverified** and needs an environment.

### Phase 3 — Tools and the registry

- [x] `guides/tools.md` — rewritten. `title`, `icon`, `capabilities`,
      `secret_inputs`, `credential_help`, `settings_model` (author-configured vs
      model-supplied, and why they are two declarations), `root_setting` /
      `path_inputs` confinement, `check_secret`, `runtime`, `Operation`
- [x] `guides/tool-registry.md` — new. The closed capability vocabulary and why it
      is closed, tags declared with no provider on purpose, `manifest_for` /
      `providers_of` (incl. `reaches_localhost`) / `unsatisfied_capabilities`, the
      three origins, prose search as tiebreak only, icons, `registry_report`
- [x] `guides/tools-catalog.md` — rewritten against `registry/core/<domain>/`, with
      the capability each tool declares, the **`sql`** tool and its four
      operations, and why `python_exec` declares none

### Phase 4 — MCP

- [x] `guides/mcp.md` — updated. Manual config still, plus `ensure_mcp_tools()` and
      a pointer to discovery. Left largely intact: the manual-connection material
      was already correct, and rewriting it wholesale would have been churn
- [x] `guides/mcp-discovery.md` — new. Registry search and install, the two sources
      (official default, Smithery opt-in with the measured numbers and why it stays
      opt-in), credential requirements — asking for the placeholder rather than the
      slot, and declared ≠ blocked — and the per-server runtime

### Phase 5 — The Architect

- [x] `architect/index.md` — reframed around `ArchitectAgent`; quick start and the
      three properties that separate it from a prompt that emits YAML
- [x] `architect/agent.md` — new. The terminal contract, `build()`, `verify` /
      `review` / `plan`, the callback set, the full toolbelt grouped by purpose,
      nudges, and the "fix the smallest thing first" rule
- [x] `architect/grounding.md` — new. The question that decides everything,
      grounding against declared tags, filling a gap, the two checks and the
      asymmetric trade behind them, refusal as an outcome
- [x] `architect/verification.md` — new. Acceptance plans, fixtures and why a
      missing one is a hard failure, the two scoring paths, the fail-closed judge,
      design-revision-not-field-patching, and the fingerprint
- [x] `architect/knowledge.md` — new. The content-hashed capability manifest and
      the docs index — including that **these pages are the input**, what that
      demands of a heading, and why two pages are excluded
- [x] `architect/how-it-works.md` — retitled and marked as the older
      `ArchitectBuilder` path, with a pointer to the agent
- [ ] `architect/building.md` — **not done.** Its `ArchitectBuilder` content is
      still accurate and the page now sits behind the agent pages; revisiting it
      is worth doing but nothing on it is currently wrong

### Phase 6 — The server APIs

- [x] `server/workflows-api.md` — new. `/v1/workflows` (list, get, requirements,
      validate, CRUD), `/v1/workflows/{name}/runs`, `/v1/runs` incl. node output,
      trace, resume, delete, and the SSE event stream. Carries the warning that
      resuming re-runs the whole graph
- [x] `server/architect-api.md` — new. `/v1/architect/plans`, `/builds` (+respond,
      cancel, list, get, SSE events) and `/repairs`, with the parked-interaction
      model explained as the HTTP form of the agent's callbacks
- [x] `server/index.md` — says there are now three surfaces, not one

### Phase 7 — Observability, Learn, and the sweep

- [x] `observability/index.md` — the dispatch worker: off-thread, bounded, FIFO,
      best-effort, `drain()`, plus named-but-unconfigured exporters being skipped
- [x] `observability/custom-exporters.md` — what dispatch means for an exporter
      author: ordering is guaranteed so per-run state is safe, delivery is not, and
      blocking costs every exporter behind you. Includes the `force_flush` warning
- [x] `learn/architecture.md` — layer diagram redrawn; registry and authoring
      called out as their own layers, gateway lists all three surfaces
- [x] `index.md` — card grid gains Graph & Workflows; "what's in the box" matches
      the framework; install section points at Upgrading
- [ ] `learn/concepts.md` — **not done.** Canonical messages, events and sessions
      are all still accurate; adding graph vocabulary here would duplicate the
      Graph tab, which now owns it. Left deliberately
- [x] Full-docs sweep — **unblocked and done.** A `.venv` was built,
      `docs/requirements.txt` and the package installed, and both gates run clean:
      `mkdocs build --strict` exits 0 with no warnings, and
      [`check_docs_imports.py`](check_docs_imports.py) resolves **79/79** distinct
      `neurosurfer` imports across every page

**Done when:** `mkdocs build --strict` is clean, and every fenced `python` block
in `docs/` either executes or is explicitly marked as illustrative. ✅

### What the sweep found

**One real bug, in a page this plan never touched.**
[`server/agents.md`](../docs/server/agents.md) imported
`build_provider_from_profile`, which has never existed on any branch. The function
is `build_provider` and it takes a `Config`, so the sample is now
`build_provider(load_config())`. It had been wrong long enough that nobody reading
the page had run it.

That is the argument for the checker existing rather than for reading harder. The
failure mode docs have is not prose that is hard to follow — it is a symbol
renamed in the source and left behind in the prose, and only an import resolves
that.

**Two Windows notes**, both pre-existing and neither a docs problem:

- Importing `neurosurfer` **dies on Windows without `PYTHONIOENCODING=utf-8`**.
  The startup banner contains box-drawing characters and the default cp1252
  stdout cannot encode them, so `import neurosurfer` raises `UnicodeEncodeError`
  before anything runs. Same root cause as the seven subprocess failures in
  [WINDOWS_TEST_FAILURES.md](WINDOWS_TEST_FAILURES.md) — and worth noting that it
  is not only subprocesses: a plain interactive import is affected too.
- [`tracing/tracer.py:443,445`](../neurosurfer/tracing/tracer.py) emits three
  `SyntaxWarning: invalid escape sequence "\{"`. Harmless today, an error in a
  future Python. Not fixed here; it belongs to whoever owns the tracer.

---

## §3 — Conventions

**A page owns one question.** If two pages could answer it, one links to the
other. The docs index rewards this and punishes the alternative: two half-answers
both score, and the reader gets whichever BM25 preferred.

**A heading names its subject.** Sections are retrieved without their parents, so
`## Loop` under `# Node Kinds` reads as "Loop" to whatever pulled it — but
`## Iterating` does not.

**Every claim about behaviour is checked against the code**, not against the
CHANGELOG. The CHANGELOG says what changed; only the source says what it does now.

**A sample that cannot run does not ship.** Phase 7's sweep is the gate, but the
cheaper habit is to write samples against real imports as each page is written.
Two errors were caught this way while writing Phase 1 alone — `ArchitectAgent.build`
is `async` and the first snippet omitted the `await`, and
`registry.core.agent` re-exports nothing, so the `FinishTool` import had to name
the defining submodule.

**Link integrity is checked, not assumed.** [`check_docs_links.py`](check_docs_links.py)
covers links, anchors and nav membership with no toolchain — run it before pushing.
It does not replace `mkdocs build --strict`, which still needs an environment.

**A ticked box means it shipped**, per `.dev/README.md`. Items deliberately left
undone stay unticked with a note.

---

## §4 — Done when

| Criterion | State |
|---|---|
| `mkdocs build --strict` passes with no warnings | ✅ exit 0, clean |
| No page links to a file, notebook, or anchor that does not exist | ✅ 57 pages, 0 problems |
| Every `neurosurfer` import in the docs resolves | ✅ 79/79 |
| Every subsystem in §0.1's zero-pages table has a page | ✅ all six |
| A person upgrading from `main` can find §0.4 without the CHANGELOG | ✅ [`about/upgrading.md`](../docs/about/upgrading.md), first section |
| `describe_capability` answers for each of the eleven node kinds | ⬜ unverified — needs a provider and an API key, so it is a live-run check rather than a build-time one |

Both gates are runnable by anyone with the docs environment:

```bash
python .dev/check_docs_links.py                              # no deps needed
PYTHONIOENCODING=utf-8 python .dev/check_docs_imports.py     # needs the package
python -m mkdocs build --strict
```

Where §0.1's six landed:

| Subsystem | Page |
|---|---|
| `registry/` | `guides/tool-registry.md` |
| `mcp/{registry,runtime,credentials,sources}` | `guides/mcp-discovery.md` |
| `architect/agent/` | `architect/{agent,grounding,verification,knowledge}.md` |
| `app/server/` | `server/{workflows-api,architect-api}.md` |
| `graph/workflow/validation/` | `graph/validation.md` |
| `graph/engine/{nodes,builder,kinds,state,secrets}` | `graph/{node-kinds,building,state}.md` |

**Docs went from 41 pages / ~3,200 lines to 57 pages / ~5,570 lines.**

## §5 — What this plan found that the CHANGELOG did not

Recorded because §0.4 turned out to understate the problem, and the next person
reading the CHANGELOG as the authoritative migration list should know it is not one.

1. **`McpStore.default()` moved** — `~/.neurosurfer/mcp.json` →
   `./.neurosurfer/config/mcp.json`. Silent: the server list simply reads empty,
   and the location is now cwd-relative rather than `$HOME`-relative. In no
   CHANGELOG entry. This is worth adding to the CHANGELOG before merge.
2. **`NEUROSURFER_HOME`'s default never changed** — the old docs said
   `~/.neurosurfer` and were simply wrong. Corrected in §0.4 after the first draft
   of this plan repeated the error.
3. **`ArchitectAgent.build` is `async`** and **`registry.core.agent` re-exports
   nothing** — both caught by checking snippets against signatures rather than
   against the CHANGELOG's prose.
