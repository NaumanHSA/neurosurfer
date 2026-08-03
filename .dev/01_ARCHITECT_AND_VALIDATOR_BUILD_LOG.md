# 01 — Build log: the Architect and the validator

Companion to [01_ARCHITECT_AND_VALIDATOR_PLAN.md](01_ARCHITECT_AND_VALIDATOR_PLAN.md).
What shipped, what it cost, and — more usefully — what the work turned out to be
once it was underway.

**Started:** 2026-08-03. **Baseline on `4065c2f`:** 360 tests passing.

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

*Not started.*

---

## §4 — Phase 4: plan first, ground, refuse

*Not started.*

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
