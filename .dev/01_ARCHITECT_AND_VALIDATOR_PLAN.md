# 01 — The Architect and the validator

**Goal:** the Architect designs workflows it has *grounded* — every capability it
names is one that exists, every graph it emits is one the engine will run — and
the validator is the gate that proves it, in language a person can act on. No
studio.

**Where the work comes from.** `main` is a strict ancestor of the studio branch:
`architect-v2` is `main` **+ 54 commits**, and zero commits exist on `main` that
are not on it. So this is not a merge or a reconciliation. It is a decision about
which of those 54 to bring, in what order — and the answer is *not* "the two
directories called `architect/` and `validation/`", for the reason §0.3 gives.

---

## §0 — The diagnosis

Measured on 2026-08-03 against `4065c2f`, by reading the code on this branch and
diffing it against `studio/architect-v2`. Every claim carries the line that proves
it.

**Baseline: 360 tests pass** (`pytest tests/ -q`). The studio branch runs 1496.

### §0.1 — The validator is one file, and severity is a place you append to

[`validate.py`](../neurosurfer/graph/workflow/validate.py) is 231 lines with five
checks. `ValidationReport` holds three flat lists, and how much a problem matters
is decided by **which list a check happened to append to** — ten call sites, each
choosing independently:

```python
report.errors.append(ValidationIssue(kind="tool_typo", …))
report.gaps.append(ValidationIssue(kind="tool_gap", …))
report.warnings.append(ValidationIssue(kind="structure", …))
```

So *"is a missing prompt an error or a warning"* is not a question you can answer
by reading anything — you have to find the one line that appends it. And nothing
declares which checks can fire for which kind of node, so *"what can go wrong
with a tool node"* is a careful read of the whole file.

**The messages are written for whoever wrote the code.** `output_schema
'my:Model' does not import` names two things the author never typed. There is no
`detail` field, so a message cannot be made plain without deleting the technical
half that a developer needs.

**One rule is a guess wearing a number.** A workflow with fewer than three LLM
nodes is warned as "almost certainly under-designed"
([`validate.py:127`](../neurosurfer/graph/workflow/validate.py#L127)). That is a
heuristic about *taste* sitting in the gate that decides whether a package may
register, and it fires on every correct two-step workflow.

### §0.2 — The Architect cannot see what it is building with

[`build.py:95`](../neurosurfer/architect/build.py#L95) runs the architect
workflow with `allowed_tools={"web_search", "write_workflow_node"}` and hands it
the catalog as **a formatted string interpolated into a prompt**
([`build.py:102`](../neurosurfer/architect/build.py#L102)):

```python
"available_tools": format_workflow_tool_catalog(),
```

That is the whole of its self-knowledge. There is no capability vocabulary, so
"this step needs to read a file" cannot be *matched* against a tool that declares
`file.read` — it can only be matched against words the tool's description happens
to share. There is no manifest, so nothing tells the Architect what a node kind
requires. And there is no verification that the thing it built runs: `assemble`
validates and registers, and a package that passes the gate is assumed to work.

The result is the failure the studio branch's build logs record repeatedly — a
model inventing tool names, or designing a step whose capability nothing
provides, and finding out at run time.

### §0.3 — The mature validator speaks about kinds this engine refuses to load

This is the finding that decides the shape of the whole plan.

[`schema.py:116`](../neurosurfer/graph/engine/schema.py#L116):

```python
_VALID_NODE_KINDS = {"base", "react", "function", "python", "tool"}
```

The matured validator's rules **declare which kinds they speak about**, and the
kinds they name include four this engine rejects at load:

```
nodes/io.py       kinds=("input",)   kinds=("output",)
nodes/_common.py  kinds=("base", "react", "router")
```

Same for the Architect: "plan first, ground every capability, refuse what can't
run" designs branches and loops, and `router` / `loop` / `map` / `subgraph` do
not exist here.

**So the two directories cannot be lifted across on their own.** Roughly half the
matured validator would be dead rules, and the Architect would plan control flow
the executor cannot run. The engine floor is a prerequisite, not a nice-to-have —
and it is independent of the studio, which is what makes this tractable.

### §0.4 — A defect that predates the split and is still here

[`manager.py:48`](../neurosurfer/graph/engine/manager.py#L48) builds every LLM
node's prompt from a hardcoded key:

```python
user_intent = str(graph_inputs.get("user_intent", "(not specified)"))
```

`user_intent` is set in exactly one place in the codebase —
[`build.py:104`](../neurosurfer/architect/build.py#L104), the Architect's own
graph. **Every other workflow's prompt opens with "User request: (not
specified)".** Reproduced on the studio branch against a live model; the run
succeeded and the answer was plausible, which is why it survived this long.

It is listed here rather than left for later because it is upstream of every
judgement the Architect makes about whether its output was any good.

### §0.5 — What is on the other branch, by category

The 54 commits, grouped by what this plan does with them:

| | Commits | Disposition |
|---|---|---|
| Studio (app, canvas, panels, redesign, accounts, uploads) | ~18 | **left behind** |
| Engine: control flow, typed state, expressions, error routing | 3 | Phase 1 |
| Self-knowledge: manifest, docs index, KnowledgeBase; tool registry | 3 | Phase 2 |
| Validator as a module | 2 | Phase 3 |
| Architect: agent harness, planner, grounding, verification | 6 | Phases 4–5 |
| MCP, tracing, gateway, tutorials, docs | ~10 | Phase 7, selectively |
| Node **kind specs** (`engine/kinds/`) | 1 | see §3.2 |

---

## §1 — The shape of the fix

**Bottom-up, and nothing lands before the thing it stands on.** The studio
branch's own roadmap records why: it ran the Architect first, on primitives
nobody had ever built by hand, and *every defect it found was a defect in the
primitive, not in the authoring.* Repeating that here would repeat the result.

So the order is: the engine can run it → the Architect can see it → the validator
can prove it → the Architect designs against the proof.

**Port, do not re-derive** — with one exception. These commits encode fixes each
found by a live build, and re-deriving loses them silently. Where a module is
studio-coupled the coupling is cut at the seam rather than the module rewritten;
where a module is clean (`validation/`, `knowledge/`, `capability.py`) it comes
across close to verbatim.

The test for whether this worked: **the Architect refuses a request it cannot
ground, and says which capability is missing** — rather than building something
that fails at run time.

---

## §2 — The phases

### Phase 0 — A baseline that can be diffed against ✅

- [x] Record the pre-port failure list, not the count. The studio branch learned
      this the hard way: comparing test *counts* hides a fixed failure and a new
      one cancelling out.
- [x] `studio` remote added and fetched, so every port is `git show
      studio/architect-v2:<path>` and reviewable as a diff rather than a retype.
- [x] Decide the Python/dependency floor: the engine work uses `StrEnum` and
      newer pydantic idioms than some of `main` assumes.

### Phase 1 — The engine floor ✅

The kinds the Architect designs with and the validator speaks about.

- [x] `router` / `loop` / `map` / `subgraph` / `input` / `output` node kinds,
      typed workflow state, and the safe expression evaluator (`a57034b`).
- [x] Error routing (`on_error`), retries, and the `GraphBuilder` fluent API.
- [x] `routes` router (classify-and-branch) and `until` loops with an exit judge
      (`d291be8`) — the plain-English control flow, which is what an Architect can
      actually plan with.
- [x] Engine fixes that came out of live builds: `on_error` not firing on the
      happy path (`fe91a75`), prompt vars that resolve and per-node providers
      (`38ce56d`).
- [x] The node *kind specs* (`engine/kinds/`) came across as well. This item was
      written as "**not** in Phase 1 — see §3.2", and that turned out to be a
      distinction the port could not honour: they live *inside* `graph/engine/`,
      so taking the directory takes them. §3.2 already recommended keeping them,
      so the outcome is the intended one and the phasing was wrong, not the
      decision. They are inert until Phase 2 reads them into the manifest.

**Done when:** a hand-written YAML workflow with a router and a loop loads, runs,
and routes.

### Phase 2 — The Architect can see what exists ✅

- [x] The capability manifest: content-hash versioned, auto-derived from the tool
      classes, with a **closed capability vocabulary** so a step's need is matched
      against a declared tag rather than against shared words (`6a3bd67`).
- [x] The docs index and `KnowledgeBase`, with its freshness gate.
- [x] The tool registry — foldered by domain, `Tool.title` / `icon` /
      `capabilities` / `secret_inputs` / `credential_help`, operations as a
      first-class thing (`80d7c45`). This is what replaces the formatted string in
      §0.2.
- [x] `capability.py` — what a node *claims* to do, checked against what it holds.

**Done when:** ~~`KnowledgeBase().render_context()` is what the Architect is
given, and `format_workflow_tool_catalog()` is no longer interpolated into a
prompt.~~ **Corrected after the fact.** The second half was a planning error, not
a shortfall: the studio branch's own `build.py` still interpolates that string,
and the function is byte-identical on both branches. The tip never rewired the
YAML architect — it built the ReAct agent beside it and left the old path alone,
because **Phase 4 replaces that path outright.** Doing it here would be work the
tip never did, on code about to be deleted.

**Done when (as shipped):** the manifest, the docs index and `KnowledgeBase`
derive from the live registry and engine, and a capability resolves against a
declared tag rather than against prose — `file.write` → `write_file`. Phase 4 is
the consumer.

### Phase 3 — The validator becomes a module ✅

Portable close to verbatim once Phase 1 lands.

- [x] `validation/` package: `models` · `registry` · `context` · `graph` ·
      `templates` · `nodes/` (`46ea9d1`).
- [x] A rule **declares its kinds and its severity**, so `rules_for_kind("input")`
      answers §0.1's question as a query rather than a read.
- [x] Severity travels **on the issue**; `errors` / `gaps` / `warnings` / `infos`
      become views over one list, which keeps the Architect's gap-routing working.
- [x] `message` / `detail` split, and the plain-language contract test — scoped to
      the swept rule ids, so the standard is enforced for new rules rather than
      asserted everywhere and skipped.
- [x] `validate.py` stays as a documented re-export, so every existing importer is
      untouched by the move.
- [x] **Move first, change second.** The move is proven by diffing the complete
      failure list before behaviour changes. On the studio branch the move alone
      surfaced four defects that would have been indistinguishable from intended
      changes had the rules changed in the same pass.
- [x] Retire the depth-floor guess (§0.1) — or demote it to `info`, which is what
      it always was.

**Done when:** the rule table is the answer to "what can go wrong with this kind",
and the plain-language test passes for every swept rule.

### Phase 4 — The Architect plans first, and refuses what it cannot ground ✅

- [x] The ReAct architect agent: toolbelt, session, JSON I/O harness (`309c94b`).
- [x] **Plan before design.** A written plan, checked against the capabilities
      that exist, before any node is emitted (`9843e87`).
- [x] **Refuse what cannot run** — a capability nothing provides is reported as a
      gap with what is missing, not built and discovered later.
- [x] Check the plan was actually built, and read the design back (`3bab542`).
- [x] The eleven fixes in `21e09a5`, each found by a live build. These are the
      highest-value part of the whole port and the easiest to lose.

**Done when:** a request needing a capability nothing provides comes back as a
refusal naming the capability, before any model call designs a node for it.

### Phase 5 — Verification that runs the workflow ✅ *(arrived with Phase 4)*

- [x] Closed-loop verification: the built workflow is *executed*, not just
      validated (`e18e33f`).
- [x] The A/B harness, so a prompt change can be measured rather than argued about.
- [x] Verification that remembers, so the same defect is not re-found each build
      (`38ce56d`).

**All three came in with Phase 4**, because they live in `architect/agent/` and
taking the package took them. The same thing happened to the kind specs in Phase
1 — twice now, a phase boundary has been drawn through a Python package, and a
package is not divisible by `git checkout`. Worth remembering when phasing the
*next* plan: **phase by package, or accept that the boundary is a reporting
convention rather than a sequencing one.**

### Phase 6 — The defects the port would otherwise carry across

Known, reproduced, and unfixed on both branches. They belong here because the
Architect's own judgement of its output is downstream of them.

- [ ] **`user_intent` is hardcoded** (§0.4). Lead with the request the graph
      actually declares; never print one value three times.
- [ ] **A `base` node with tools runs one round and stops silently.** When the
      budget is exhausted while the model still wants a tool, its last message is
      tool calls with no text, so the node returns `""` — and the run reports
      *succeeded*. Reproduced with a fake provider on the studio branch.
- [ ] **An empty output from an LLM node is not a failure anywhere**, so the above
      surfaces as a green run with a blank answer.
- [ ] Decide whether the tool-round budget becomes a declared property of the kind
      rather than a constant inside `run_base_node`.

### Phase 7 — What the port needs to keep working

- [ ] Tracing: per-run I/O on workflow and node spans (`6c2c33b`, `3432098`) —
      without it a build cannot be debugged, which is most of what Phases 4–5 need.
- [ ] Docs: control-flow engine and the self-verifying Architect (`0987dda`).
- [ ] Evals last, deliberately. Nine runs of two prompts is a bug-finder, not a
      benchmark, and nothing above is stable until Phase 5 lands.

---

## §3 — Decisions, with a recommendation

### §3.1 — Does the run/execution API come across?

The studio branch has a run store, REST + SSE streaming and durable run records
(`6c8214a`). Phase 5's verification needs *a* way to run a workflow and read what
happened; it does not need the HTTP surface.

~~**Recommendation: take the run store and the durable run record, leave the REST
and SSE routes.**~~ **Overturned in Phase 4.** The recommendation was right about
the Architect's *code* — its closure is clean — and wrong about the cost. Its
*tests* are written at the HTTP boundary, so leaving the routes left twenty tests
on the floor, including the only coverage of "a build that parks and asks a
person". `routes_architect`, `routes_workflows` and `workflow_runs` are in; the
studio, accounts and uploads are not. See the build log §4.

### §3.2 — Do the node kind specs come across?

`engine/kinds/` — one module per kind declaring its fields, types, help text and
which are required — was built so the studio could render a config panel from a
declaration rather than hand-coded forms.

But it also became the thing that made *the spec disagreeing with the executor*
visible, and it caught nine real defects that way, including three fields the
engine ignored entirely and one it read that nothing offered. The Architect's
manifest on that branch derives `key_fields`/`requires` from it rather than
hand-maintaining a second list that had already drifted.

**Recommendation: take it, in Phase 2, as manifest input rather than as UI
input.** It is the only thing on either branch that makes a kind's contract
checkable. Its studio renderer stays behind.

### §3.3 — Tool settings and path confinement?

The last commit on the studio branch gives a tool author-set configuration —
notably a filesystem root that `write_file` is confined to. It was built for the
studio, but the defect it fixes is not a studio defect: without it every
filesystem tool resolves against the process working directory, so an agent asked
to write a report writes into the checkout.

**Recommendation: Phase 6 or later, and not a blocker.** It is a real fix and a
small one, but the Architect does not depend on it and it is the newest, least
exercised code on that branch.

---

## §4 — What this plan does not own

- **The studio.** Not ported, not maintained here. It lives at
  `NaumanHSA/neurosurfer-studio`.
- **The gateway's account system, uploads and per-account workspaces.** Built for
  a hosted multi-user surface; this line has a CLI and a library.
- **MCP.** The engine can already carry MCP tools; the discovery/registry surfaces
  on the other branch are studio-facing. Revisit after Phase 4.

---

## §5 — How this lands

**`main` is not touched until the whole plan is done.** Every phase lands on
`architect-validator/enhancement`, which is pushed and is the only place this work
exists. `main` stays at `4065c2f` — the stable line people can use while a
fifty-commit port is in flight.

Two mechanical notes, because the branch was created in a way that makes one
mistake easy:

- It was cut with `git checkout -b … origin/main`, so its **upstream was
  `origin/main`** until it was first pushed. `push.default` is unset (`simple`),
  which refuses on a name mismatch — but under `upstream` a bare `git push` would
  have put the whole port on `main`. The upstream now points at the branch itself.
- Push explicitly — `git push origin architect-validator/enhancement` — rather than
  relying on whatever `push.default` happens to be on the machine.

### The merge, when the phases are done

Not before: the plan's own argument is that these pieces only make sense on top of
each other, and a half-ported Architect on `main` is the thing §1 exists to avoid.

- [ ] All phases ticked, and the build log has a section for each — a ticked box
      with an empty log section is the discrepancy the README's convention exists
      to catch.
- [ ] The **full suite including the live tests**, run once, deliberately, against
      OpenAI. Everything up to here has been run with the live provider pointed
      away so the loop stays at 24s; that is a working convenience and not a
      release check.
- [ ] Ruff clean, which it currently is.
- [ ] Merge to `main`, then **bump the version**.

### The bump

`1.0.0` today, in **two** places that must move together —
[`pyproject.toml`](../pyproject.toml) and
[`neurosurfer/__init__.py`](../neurosurfer/__init__.py) — plus the `[Unreleased]`
section of the [CHANGELOG](../CHANGELOG.md), which follows Keep a Changelog and
declares SemVer.

**Recommendation: `1.1.0`, a minor bump**, and the reasoning is worth settling
before the day of, because there is a real argument for major:

- *For minor:* everything the port adds is additive — six node kinds, a capability
  layer, a rules-based validator, two route groups. Nothing that worked on `1.0.0`
  stops working. The tool modules that **moved** kept package-level re-exports, so
  `from neurosurfer.tools.builtin import ReadFileTool` is unchanged.
- *For major:* **submodule** imports did break —
  `from neurosurfer.tools.builtin.search import SearchTool` no longer resolves, and
  nine files in this repo were relying on exactly that. If anyone outside is
  importing tools by submodule path, that is a breaking change to them.

Minor plus a prominent CHANGELOG note under *Changed* naming the move, unless
somebody knows of an external caller reaching in by submodule path — in which case
it is `2.0.0` and the note is not enough.
