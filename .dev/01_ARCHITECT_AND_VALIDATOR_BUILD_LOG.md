# 01 — Build log: the Architect and the validator

Companion to [01_ARCHITECT_AND_VALIDATOR_PLAN.md](01_ARCHITECT_AND_VALIDATOR_PLAN.md).
What shipped, what it cost, and — more usefully — what the work turned out to be
once it was underway.

**Started:** 2026-08-03. **Baseline on `4065c2f`:** 360 tests passing.
**Now:** 1134 passing, 4 skipped, ruff clean — Phases 0–4 in.

**Where it lives.** `architect-validator/enhancement`, pushed to `origin`.
**`main` is not touched until the plan is finished**, then merged with a version
bump — see the plan's §5. Nine commits were local-only for most of a day before
anybody noticed, which is its own small lesson: *push the branch when you cut it,
not when someone asks where it is.*

Fill a section in as each phase lands. A phase with nothing under it has not
started; a phase whose plan boxes are ticked but whose section here is empty is a
discrepancy worth chasing, which is the point of keeping both files.

---

## §0 — Setting up

- [x] Branch `architect-validator/enhancement` cut from `origin/main` (`4065c2f`).
- [x] `architect-v2` deleted from the `neurosurfer` remote. It is preserved in
      full at `NaumanHSA/neurosurfer-studio` (`ee00a6a`, 162 commits), verified
      byte-for-byte before the delete.
- [x] Working copy cleaned of that branch's remains: `studio/` (155 MB of
      `node_modules`/`dist`), `logs/`, and **twenty-seven orphaned package
      directories** — `neurosurfer/registry/`, `graph/workflow/validation/`,
      `graph/engine/kinds/`, `architect/agent/`, `architect/knowledge/`,
      `app/server/auth/` and the rest.

  Those survived the checkout because each held only `__pycache__`, and git
  cannot remove a directory whose remaining contents it never tracked. Worth
  writing down: **a branch switch does not leave a clean tree when the old branch
  had packages the new one lacks** — `import neurosurfer.registry` still
  succeeded on this branch afterwards, which is exactly the kind of thing that
  makes a port look further along than it is.

- [x] Baseline recorded: 360 passed, 0 failed, in 17s.
- [x] Branch pushed to `origin` (2026-08-03, after four phases — see above). Its
      upstream had been `origin/main`, from `git checkout -b … origin/main`; a
      bare `git push` under `push.default=upstream` would have put the port on
      `main`. Now tracks itself.

---

## §1 — Phase 1: the engine floor

**Shipped 2026-08-03** (`2fdd5b2`). 581 tests pass, from a 360 baseline — no
failures on either side, compared as *lists* rather than counts. Ruff clean.

### The decision that made it small: port the tip, not the commit

`a57034b` is the commit that introduced control flow, and its parent is exactly
`main`'s tip — so it would have cherry-picked cleanly. It was still the wrong
thing to take. It is 53 commits behind, and the fixes since are the whole reason
to port rather than rewrite. So the port is `git checkout studio/architect-v2 --
neurosurfer/graph/engine`: the *matured* engine, in one move.

`graph/engine/` goes 2.4k → 6.9k lines and **nothing was dropped** — every file on
`main` still exists at the tip. That is what made a wholesale take safe, and it
was worth checking before committing to it rather than after.

### The closure was three modules, and the scan that found them was wrong twice

Grepping the tip's engine for `from neurosurfer.…` said only two modules were
missing. Both misses were the same mistake in different clothes:

- **`from .registry import …`** — a *relative* import inside `mcp/credentials.py`,
  which a `from neurosurfer.` grep cannot see. Found by the test suite, not by the
  scan.
- **`neurosurfer/graph/__init__.py`** — the package re-export one level *above*
  the directory being ported. `GraphBuilder` was in the ported code and not in the
  namespace anybody imports it from.

Neither is exotic. Both say the same thing: **an import scan anchored on absolute
paths inside one directory is not a dependency closure.** The suite found both in
under a minute, which is the argument for porting against a green suite rather
than reading imports harder.

Two more arrived from outside `graph/`: `tests/fakes.py` (the tip's
`ScriptedProvider` records prompts) and an additive `on_usage` hook in
`structured_completion` — without which a structured node reports zero tokens no
matter how many repair attempts it took.

### One ported test belonged to a later phase

`test_workflow_template_vars.py` failed 23 times, and the failures were real:
it imports `validate_package` and tests the **template walk**, which is
Phase 3's. Deferred rather than fixed. Worth stating because the count looked
alarming and the cause was that a test had been filed under the wrong phase by
whoever ported it — me, ten minutes earlier.

### What the port bought, checked outside the suite

Two hand-written YAML packages, loaded through `load_package` and run:

- a `router` that takes the `bug` branch on *"there is an error in checkout"* and
  the `question` branch on *"how do I reset my password"* — the branch that did
  not run is genuinely absent from the results, not merely empty;
- a `loop` with `break_when: iteration >= 2`, iterating and returning its last
  draft.

The first attempt at the router YAML was rejected at load with *"router 'classify'
target 'bug' must list 'classify' in its depends_on (so it runs after the routing
decision)"*. That is the ported control-flow validation earning its place on its
first use — the error named the node, the rule and the fix.

### …and what it taught

**A branch switch does not leave a clean tree.** Twenty-seven directories from the
studio branch survived the checkout because each held only `__pycache__`, and git
cannot remove a directory whose remaining contents it never tracked.
`import neurosurfer.registry` still succeeded on this branch before the sweep —
so a Phase 2 test could have passed against code that is not supposed to exist
here yet, and the port would have looked further along than it was.

### Deliberately not done

- **`configured_tools.py` and `tool_settings`** came along inside `graph/engine/`
  rather than being chosen. They are inert without a tool declaring
  `settings_model`, which is §3.3's call and not Phase 1's.
- **`_check_output_schema` only guards.** `output_schema` is now `str | dict`,
  and the inline JSON-Schema form is skipped rather than checked — handing a dict
  to `import_string` would report "does not import" about a well-formed schema.
  The real rule is Phase 3's.
- **`tests/engine/` is a new directory beside `main`'s flat `tests/`.** The tip
  reorganised the suite; four of its files have counterparts still living at the
  top level here. Reconciling that is not Phase 1's job and would have made this
  diff unreadable.

---

## §2 — Phase 2: the Architect can see what exists

**Shipped 2026-08-03** (`fedbff3`). 641 pass, 1 skipped, from 620. Ruff clean.

### The closure scan was wrong a third time, in a third way

Phase 1 learned that an import scan must resolve *relative* imports and check the
package `__init__` above the ported directory. So this time the scan did both —
and still missed something, because the registry port is a **move**, not an
addition.

Six test files and three source files imported `neurosurfer.tools.builtin.<mod>`
by submodule path. `tools/builtin/__init__.py` survives as a re-export, so
`from neurosurfer.tools.builtin import ReadFileTool` keeps working; `from
neurosurfer.tools.builtin.search import SearchTool` does not.

The rule the three failures actually teach:

> A closure scan answers *"what does the ported code need?"*. A **move** also
> needs the reverse: *"who reaches into where it used to be?"* — and nothing about
> the first question surfaces the second.

`grep -rn "tools\.builtin\.[a-z_]"` answers it in one line. It should have been
the first thing run, not the thing run after the suite went red.

### Two more test files filed under the wrong phase

Exactly Phase 1's mistake, repeated with different files. `test_capability_grounding.py`
turned out to be **validator** tests (it asserts severities from rules that arrive
in Phase 3) and `test_capability_resolution.py` needs `architect/agent/`, which is
Phase 4. Both deferred rather than made to pass.

That is twice now. The pattern is worth naming: **a test file named after a
subject does not belong to the phase that ports that subject** — it belongs to
the phase that ports whatever it *asserts against*. Checking a ported test's
imports before adding it costs one command.

### The "done when" was wrong, and the tip proves it

Phase 2's stated finish line was *"`format_workflow_tool_catalog()` is no longer
interpolated into a prompt"*. It is still interpolated, and it will be until
Phase 4.

The reason is worth recording, because it was a bad piece of planning rather than
a shortfall in the work. **The tip's `build.py` still calls it too**, and
`format_workflow_tool_catalog` is byte-identical on both branches — a flat
`- name: description` list with no capabilities. The studio branch never rewired
the YAML architect: it built the ReAct agent beside it and left the old path
alone, because Phase 4 replaces that path outright.

So rewiring `build.py` here would be work the tip never did, on a path that is
about to be deleted. Not done, deliberately. The Phase 2 checklist has been
corrected rather than ticked around.

### What Phase 2 actually delivers

Measured, not asserted:

```
VOCAB    : 17 declared capability tags
MANIFEST : 11/20 tools declare capabilities
GROUND   : file.write     -> ['write_file']
           web.search     -> ['web_search']
           data.inspect   -> ['data']
KINDS    : base, function, input, loop, map, output, python, react,
           router, subgraph, tool
```

Eleven node kinds in the manifest is Phase 1 paying for itself — on `main` the
same derivation finds five. The capability lookup is the thing `main` could not
do at all: a need resolved against a **declared tag** instead of against words a
description happens to share.

The consumer of all this is Phase 4. Phase 2's job was to make it exist and be
derivable, and that is done.

### Deliberately not done

- **`build.py` still gets the flat catalog string.** See above.
- **9 of 20 tools declare no capabilities.** Untagged tools still work; they are
  simply never a *match*, which is the design. Tagging the rest is cheap and is
  better done when Phase 4 shows which gaps actually bite.
- **`test_execution_api_endpoints_derived` is skipped**, with the reason in the
  skip marker rather than in a commit message nobody will find: it walks the
  gateway's routes for `/v1/workflows` and `/v1/runs`, and §3.1 declines them.

---

## §3 — Phase 3: the validator becomes a module

**Shipped 2026-08-03** (`ad28104`). 796 pass, 6 skipped, from 641. **Ruff clean
across the whole tree**, for the first time on this line.

### The port was the easy half

`validate.py` becoming a 31-line re-export meant all twelve importers — four in
`architect/`, two in `registry/`, and six test modules — were untouched. That is
the whole argument for keeping a shim: a rename is not a reason to edit nineteen
files, and every one you edit is a chance to edit it wrong.

### Three failures, and only one of them was a bug

Worth separating, because they looked identical in the output:

1. **`test_unknown_depends_on_is_error`** asserted `"ghost" in e.message`. The
   rule survives with the same `kind="dag"`; the message became *"This step waits
   for a step that is not in the workflow."* and `ghost` moved to `detail`. That
   is the `message`/`detail` split working exactly as designed, and the tip's own
   version of the test asserts `"ghost" in (e.detail or "")`. Superseded, not
   fixed.

2. **`test_invalid_patch_is_rejected`** in refine. The fixture patched
   `tools: [nonexistent_xyz]` onto a **`function`** node and expected
   re-validation to fail. Under declared kinds, `tools_exist` speaks about
   base/react/tool — a function node has no tools, so the patch is inert and
   validation correctly passes. A stale fixture, and the tip had hit it and
   changed the same line. Fixed the fixture; the comment says why.

3. **Seven requirements failures** were a genuinely missing module. Real work,
   not a test artefact.

The distinction matters because the instinct on a red suite is to treat all of it
as breakage. Two of these three were the port *working*.

### `refine.py` could not come, and that is the plan holding

Its tip version needs `architect/agent/jsonio` — Phase 4. So `refine.py` stays at
`main`'s version here, and the one test that depended on new validator semantics
was fixed at the fixture rather than by dragging Phase 4 forward. The 527-line
refine diff waits its turn.

### The depth-floor guess retired itself

§0.1 called out *"a workflow with fewer than three LLM nodes is almost certainly
under-designed"* as a judgement about taste sitting in a registration gate. The
matured module had already deleted it — the studio branch reached the same
conclusion independently. A correct two-step workflow now validates silent, which
is checked rather than assumed.

### Eighteen lint findings came with the port

They had been sitting on the studio branch too; this is the first branch where
the whole tree is clean. Two were worth more than tidying:

- **Six annotations named `ValidationReport` without importing it.** Invisible at
  run time under `from __future__ import annotations`, and simply wrong — the
  annotation refers to a name not in scope. Fixed by importing it, not by
  deleting the annotation.
- **`Severity(str, Enum)`** where the engine's own `NodeMode` already uses
  `StrEnum`. Same behaviour, and the codebase now agrees with itself.

### What Phase 3 delivers

```
rules_for_kind('input')  -> edges_point_at_real_nodes, required_fields_present,
                            one_declared_field_could_be_free_text,
                            capability_is_available
two-step workflow issues -> none
```

*"What can go wrong with an input node"* is a query. §0.1 said it was a careful
read of a thousand lines.

### Deliberately not done

- **`refine.py` stays at `main`'s version.** Phase 4.
- **Five registration-gate tests are skipped at their fixture**, with
  `pytest.importorskip` and a reason, rather than five separate markers. They
  test the gate; the rules the gate consumes are covered by the thirty-odd tests
  above them in the same file.

---

## §4 — Phase 4: plan first, ground, refuse

**Shipped 2026-08-03** (`7ccc08c`). 1134 pass, 4 skipped, from 796. Ruff clean.

### §3.1 was wrong, and the reason is worth keeping

The plan declined the REST and SSE routes: *"the record is what verification
reads; the routes exist to feed a browser."* That was right about the Architect's
**code** — its closure was clean apart from `mcp.runtime` — and wrong about the
cost, because its **tests** are written at the HTTP boundary.

Twenty of the twenty-four failures after the code port were gateway surface. The
one that decided it was `test_architect_interaction.py`: ten tests covering *"a
build that parks and asks a person, is answered, and resumes"*. That is
behaviour, not transport, and it had no other coverage anywhere.

So `routes_architect` and `routes_workflows` came in, with `workflow_runs`
behind them. The studio, accounts and uploads did not.

### `workspaces.py` was rewritten, not ported

The per-account version keys every manager on a user id, and there are no users
on this line. Porting it would have brought a workspace dimension that is inert
by construction, plus the half of `paths.py` that exists to serve it.

Rewritten single-tenant instead — but **keeping the `user` parameter on every
signature**, which is the part worth explaining. It means the nine call sites in
the ported routes are untouched (nine chances to drop the wrong argument, not
taken), and it keeps one file as the seam if accounts ever return. `user` is
accepted and ignored, and the docstring says so rather than leaving the next
reader to work out why a single-tenant module talks about users.

### A real defect, not a missing module

Four MCP tests failed with `assert []` — no servers connected. The cause was two
layers down:

```
McpManager.connect_all() got an unexpected keyword argument 'publish'
```

`mcp/runtime.py` came across in Phase 4; `mcp/manager.py` had not, and its
`connect_all` predated the argument. Every server reported `connected: False`
with the `TypeError` **buried inside a status object** rather than raised — so
the test said "nothing connected" and the reason was one `getattr` away.

Worth noting as a class: a port can be import-clean and still signature-broken.
The closure scan cannot see this, and neither can the type checker when the
call is `connect_all(publish=...)` on a duck-typed manager.

### Proven twice, deliberately

Deterministically, with no model in the loop — which is the point, because a weak
model has no say in it:

```
'file.read'   -> read_file        grounded
'web.search'  -> web_search       grounded
'email.read'  -> nothing provides it
'chat.post'   -> nothing provides it
```

Then once **live**, on OpenAI `gpt-4o-mini`:
`test_agent_declares_blocked_with_real_llm` — 1 passed in 35.77s. A real model,
asked for something the install cannot do, declares it blocked instead of
building an unrunnable workflow. That is Phase 4's finish line.

### The suite got twelve minutes slower without saying so

`tests/_llm_test_provider.py` defaults to `qwen/qwen3.5-9b` at
`localhost:1234`. It is written to auto-skip when that is unreachable — and it
was reachable, because LM Studio was running. So the live architect tests
silently engaged the developer's GPU and took the suite from 22s to **12m16s**.

Nothing was wrong; it simply never announced itself. Offline runs now use
`NEUROSURFER_TEST_BASE_URL=http://127.0.0.1:9`, and live checks are named,
individual, and pointed at OpenAI.

### Deliberately not done

- **`test_architect_clarify_and_attachments.py` deleted**, not skipped. It drives
  `POST /v1/architect/attachments`; uploads are not on this line, so the tests are
  not pending, they are inapplicable.
- **`auth/`, `routes_uploads`, `routes_attachments`, `routes_fs`, `routes_catalog`,
  `routes_authored_tools`, `routes_mcp`, `routes_settings`** stay behind. The
  first three are accounts and uploads; the rest exist to render a studio.
- **`settings_store.py` came in** even though its routes did not — the run
  manager reads provider profiles and secrets from it, so it is a dependency of
  execution rather than of a settings screen.

---

## §5 — Phase 5: verification that runs

*Not started.*

---

## §6 — Phase 6: the carried defects

*Not started.* All three are reproduced — see the plan's §0.4 and Phase 6. The
repros are worth re-running rather than trusting: they were built against the
studio branch's engine, and Phase 1 changes that engine.

---

## §7 — Phase 7: tracing, docs, evals

*Not started.*
