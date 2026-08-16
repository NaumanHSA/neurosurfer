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

**Already in, as of Phase 4** (`7ccc08c`). Nothing new was written; this section
records the check that it is genuinely there, because a ticked box with no work
behind it is exactly the discrepancy the README's convention exists to catch.

All three items live in `architect/agent/`, and taking that package took them.

### 1. Verification executes the graph

`agent/verify.py` builds a `WorkflowRunner` and runs it — it does not re-validate
and call that verification. It tracks which nodes actually executed and reports
**branch coverage**:

> `COVERAGE WARNING — these nodes never executed in any test case`

which is the check that separates "the workflow ran" from "every path in the
workflow ran". A router whose second branch is never exercised is a workflow
half-verified, and it now says so.

### 2. Verification remembers, and stales correctly

`session.VerificationRecord` stores a **fingerprint** (graph + authored tools) and
an `inputs_key` alongside the verdict, rather than a bare `verified` flag. Its own
docstring gives the reason: verifying re-runs the whole graph and is *"by some
distance the most expensive thing a build does"*.

The tests name the behaviours precisely, and all pass:

```
test_repeat_test_does_not_re_run_the_graph
test_a_no_op_edit_does_not_stale_the_verification
test_reverting_an_edit_restores_the_verification
test_a_real_edit_forces_a_re_run
test_changing_the_output_set_stales_it
test_authoring_a_tool_stales_it
test_different_test_inputs_are_a_different_test
test_the_report_states_how_many_graph_runs_it_cost
```

*Reverting an edit restores the verification* is the one worth pointing at: a
fingerprint over content, not a dirty bit, so editing a node and putting it back
does not cost another full run.

### 3. The A/B harness

`agent/harness.py`, unchanged from the tip. Runs a fixed suite of intents through
named builder callables and compares success rate, validation status, node count
and wall time — *"the evidence that decides whether the ReAct agent replaces the
legacy pipeline — no pre-commitment."*

**Verified:** 21 verification tests pass; 1134 total, 4 skipped, ruff clean.

### …and what it taught

**Two phase boundaries have now been drawn through a Python package**, and a
package is not divisible by `git checkout`. Phase 1 was written as *"not the kind
specs"* and got them because they sit in `graph/engine/`; Phase 5 was written as
separate work and arrived with Phase 4 because it sits in `architect/agent/`.

Neither produced a wrong outcome — §3.2 wanted the kind specs, and Phase 5's items
were always going to come with the agent. But the plan claimed a sequencing it
could not enforce. For the next plan: **phase by package**, or say plainly that a
phase is a unit of *reporting* rather than of *delivery*.

---

## §6 — Phase 6: the carried defects

**Shipped 2026-08-04** (`bb8d789`). 1142 pass, 4 skipped, ruff clean. The first
phase that is code rather than a port.

### Re-reproducing first paid off — by confirming nothing had changed

The note above said the repros were written against the studio branch's engine
and were worth re-running rather than trusting. Re-run, all three reproduce
byte-for-byte on the ported engine.

For the prompt defect there was a stronger check available and it is the one
worth recording: `git diff origin/main studio/architect-v2 --
graph/engine/manager.py` is **zero lines**. The file is identical on both
branches, so no amount of porting could have fixed it, and the defect was never
the studio's — it was always here.

### 1. The prompt was written for the Architect's own graph

`compose_user_prompt` opened with a hardcoded `user_intent`, set in exactly one
place in the codebase. Every other workflow got:

```
User request: (not specified)          ← it was specified
Additional inputs:                      ← it is the only input, not an extra
  intake: visit https://example.com …
Context from previous nodes:            ← and now a third time
--- intake ---
visit https://example.com …
```

Three statements, two false. The third repetition is structural rather than
careless: `_run_input_node` writes the value it collected under its own id, so an
input node's output *is* a graph input, arriving twice through two different
doors.

Fixed by making the header conditional, renaming `Additional inputs` to `Inputs`
when there is nothing for them to be additional to, and dropping a dependency
whose value is already shown as an input.

**The de-duplication keys on the value, not on "it is a dependency"** — which is
the part that needed care. Keying on the relationship would have swallowed
genuine upstream context, so there is a test for exactly that
(`test_a_genuine_upstream_result_is_still_shown`).

### 2. A cut-short step stopped claiming success

`base` gets one tool round. "Fetch the page then write it to a file" needs two:
the model spends the round on the fetch, is refused the second, and its final
turn is tool calls with no prose. `response.text()` is `""`, and
`RunFinished("completed", "")` reported that as a finished run.

`OneShotAgent` now sets `cut_short`, and the flag is **keyed on pending tool
calls rather than on an empty string** — a model that returns nothing for its own
reasons is a different problem and must not be diagnosed as truncation.

### The line is *empty*, not *cut short*

The judgement worth defending. A truncated step that still produced text produced
a **partial answer**, and a partial answer is sometimes exactly what was wanted —
failing it would break runs that are useful today. What is never useful is
nothing at all. So:

| | outcome |
|---|---|
| cut short, no text | **error**, naming the tools it called and pointing at `react` |
| cut short, some text | kept |
| finished normally | kept, including a legitimately empty answer |

### …and what it taught

**Three defects, one shape: each produced a plausible success.** A prompt that
reads fine, a green run, a blank answer. Nothing raised, nothing logged, nothing
red — which is why all three survived a year and a full port. The tests for them
are named after the *symptom a person would notice*, not the function, because
the function was never the thing that was hard to find.

### Deliberately not done

- **The tool-round budget is still a constant** (`max_tool_rounds=1` inside
  `run_base_node`) rather than a declared property of the kind. Phase 6 made its
  exhaustion loud; it did not make the limit *visible*. As a spec field the card
  could say "one round of tools" and validation could warn on a `base` node
  holding two tools that must run in sequence. That is a change to the kind specs
  and a design question, not a bug fix — and the silent failure it was hiding
  behind is now gone, which was the urgent half.

---

## §7 — Phase 7: tracing, docs, evals

**Shipped 2026-08-04** (`7fb8972`). 1142 pass, 4 skipped, ruff clean,
`mkdocs build --strict` clean.

### A race that only became real in Phase 1

Tracing was mostly current — the engine port carried most of it — but five files
still differed, and one difference matters here in a way it did not on `main`:

`Tracer` handed out step ids from a bare `self._counter += 1`. Workflow nodes are
traced from executor worker threads, and a **`map` node runs its body
concurrently**, so two steps could be handed the same id. On `main` that was
theoretical, because `map` did not exist. Phase 1 brought it, which made a latent
race a live one — and nothing in the port would have flagged it, because the
file's diff looked like a tidy-up.

Checked rather than assumed: an eight-item `map`, eight concurrent body steps,
**eight distinct ids**, each attributed to its node.

The other change is `node_id` on a step, which is what turns a run's trace from a
flat list into a per-node tree.

### One doc was actively wrong, so it was rewritten

`docs/guides/configuration.md` gained a storage-layout section on the studio
branch, documenting `workspaces/` with one directory per signed-in account. True
there. **False here** — there are no accounts, for the same reason
`workspaces.py` was rewritten rather than ported in Phase 4.

Porting it would have shipped documentation describing a layout the code does not
have, which is worse than having none: nobody checks a doc against the
filesystem. Rewritten for the real layout, and the rewrite says which directories
are host-level *on purpose* rather than by omission — `config/mcp.json`, because
an MCP server is a process the gateway spawns as itself; authored tools, because
the registry loads them with no scoping argument, so a tool stored anywhere else
would exist and never be callable.

The control-flow guide and the Architect's self-verification note came across
unchanged; both were already true of this line once Phases 1 and 5 landed.

### Evals: still not done, and the reason changed

It was deferred because *"nothing above is stable until Phase 5"*. Phase 5 has
landed, so that reason has expired. It is now a real next piece of work rather
than a deferral, and it belongs to whatever plan follows this one — with
`agent/harness.py` as its starting point, since the A/B machinery already exists
and has never been pointed at anything.

---

## §8 — The release checklist (plan §5)

- [x] **All phases ticked, each with a section here.** Seven phases, seven
      sections. The two remaining unticked items in the plan are written-down
      decisions, not omissions: the tool-round budget stays a constant (Phase 6),
      and evals move to the next plan (above).
- [x] **The full suite including the live tests**, run once against OpenAI
      `gpt-4o-mini`: **1145 passed, 1 failed** in 3m54s. See below — the failure
      is a model-capability boundary, not a defect.
      **Superseded 2026-08-16** — that run predates plans 02 and 03 and ~50
      commits. Re-run in full; see *§11 — the pre-merge verification* at the end
      of this log.
- [x] **Ruff clean** across `neurosurfer/` and `tests/`; `mkdocs build --strict`
      clean.
- [x] **CHANGELOG drafted** under `[Unreleased]`, with the submodule-import break
      called out under *Changed* — the one thing that decides `1.1.0` vs `2.0.0`.
- [ ] **Merge to `main`, then bump the version.** Awaiting the call.

**Where it stands:** 1145 passing from a 360 baseline. The offline suite runs in
24 seconds, which is the number that made the whole port reviewable.

### The one live failure, and why it is not a blocker

`test_agent_designs_branching_workflow_with_real_llm` failed on `gpt-4o-mini`.
The other three live tests passed.

What happened is the system working:

```
VERIFICATION FAILED (run ok, 1 graph run)
  ✗ [urgent_ticket_response] The workflow did not generate an
    escalation notice for an urgent ticket.
```

The Architect designed a branching workflow, **ran it**, judged the result
against acceptance criteria it had derived, found the urgent branch did not
escalate, tried to repair it, ran out of step budget, and **refused to register a
workflow that does not work**. Every one of those steps is Phase 4 and Phase 5
doing their job. Before this port the same build would have validated clean and
registered.

The test asserts the *build succeeds*, which makes it a test of the model as much
as of the code — and `gpt-4o-mini` is under the bar for a branching design. The
Architect's own docs, ported in Phase 7, say so in as many words: *"generated-graph
quality still tracks the model you give it: strong tool-calling models (e.g.
`gpt-5-mini` and up) produce solid, branching designs; smaller models occasionally
emit a simpler-than-ideal graph."*

**Not fixed, and not silenced.** Making it pass would mean either running the
release check on a bigger model or weakening the assertion, and only the first is
honest. Recorded here so the next person meeting a red live suite knows to check
the model before the code.

---

## §9 — Past the checklist: classes, the prompt contract, and a package

**Shipped 2026-08-04 → 2026-08-06** (`9051112`..`0579b32`, 23 commits). **1190
collected — 1171 passed, 4 skipped, 15 failed**, ruff clean. The 15 are
environmental and fail identically on `main`; see *Running the suite on Windows*.

### Why this is a §9 and not a plan 02

The question was left open, and the answer is in what these commits are. None of
them starts from a new diagnosis; every one finishes a mechanism this plan
already owns. Node kinds became classes because §1 built the kinds. Validation
moved to the front of a run because §3 made it a module worth running. The
prompt contract changed because §6 fixed the same recitation defect in three
places and the fourth was the block itself. A plan 02 would need a §0 of its
own, and there is not one here — there is this plan's floor, finished.

### The prompt contract, and reading the two commits in the right order

`ad393c1` narrowed the ambient `Inputs:` block. `a4a5259` deleted it. **Half of
what the first commit adds is gone by the second**, and reading `ad393c1` alone
will mislead: `_hidden_inputs`, `_hidden_body_inputs`, `_input_root`,
`recited_names` and `NOTHING_FURTHER_PROMPT` all existed to decide what to omit
from a block that no longer exists.

What survives from `ad393c1` is `render_scope`, and it earned its place: four
sites assembled a node's template scope by hand, all four disagreed, and none saw
the container scope — which is why `map` had to smuggle `{item}` in through the
body's graph inputs.

The contract that holds:

> A node's turn is **what its task text names, plus the outputs of the steps it
> declared as dependencies.** Nothing ambient.

The task also moved from the system prompt into the user turn. Rendering it into
the system prompt made that prompt differ per node and, inside a `map`, per item
— so **prompt caching could never fire**. `NODE_SYSTEM_PROMPT` is now one
constant, byte-identical for every call.

### The executor is a package

`executor.py` (1,950 lines) → `executor/` (8 files, 2,322). Runners are functions
taking the executor; `GraphExecutor` keeps a one-line forwarder each, so nothing
outside the package changed — and nothing outside it imports the package's
internals by path, which is what made the move safe to publish.

**The previous attempt was reverted and cost a session.** The cause is now known:
26 relative imports, 14 of them inside methods where no import-time check reaches
them, which surfaced as thirteen unrelated test modules failing to import. One
anchored rewrite fixes it. `tests/engine/test_import_boundaries.py` is the only
thing that notices if the lazy `neurosurfer.agents.*` imports in `node_runner.py`
are ever "tidied" to the top level — do not delete it.

Ruff's `F` selector was the mechanical net for moving fourteen methods out of a
class: `F821` fired on six leftovers (`Usage`, `GraphExecutionError`, `React`,
`copy_context`, `FuturesTimeout`, `import_string`), most on failure paths no
offline test reaches.

**What F821 cannot see is an attribute access on a parameter**, and that is
exactly what it missed — see below.

### What the live run found — and it found two things

The handoff recorded the risk plainly: every prompt the framework emits changed,
and the entire verification was offline against a scripted provider. Running it
was the right first move, because it did not come back clean.

**1. A bound argument could not reach an agent node.**

```
AttributeError: 'GraphExecutor' object has no attribute '_render_tool_args'
  neurosurfer/graph/engine/executor/llm.py:96 in _agent_tools
```

`_render_tool_args` and `_render_tool_settings` became module-level functions
taking the executor, like everything else in the split. The two call sites in
`llm.py` kept calling them as methods. Three things hid it: no offline test
attaches `tool_args` to a node that calls a model, ruff sees an attribute access
on a parameter rather than an undefined name, and the path only runs when a node
carries bound arguments — which is the only way a credential reaches a tool
without passing through a prompt.

Fixed in `c01894a` with the regression tests that should have existed first;
they fail with the live error when the fix is reverted.

**2. The new contract silently broke the tutorials.**

Tutorial 03 §4 ran green and answered:

> *Please provide the specific topic you would like me to research and condense
> into five key bullet points.*

`content_pipeline` declares no inputs, its goals interpolate nothing, and it is
run with `{"user_intent": …}`. Under the old contract that arrived as
`Additional inputs:`; now nothing carries the topic. §6, §8, §10 and §11 reuse
that graph. The capstone's `db_analyst` asked for "the user's question" while
naming no placeholder for it.

**The validator could not have caught it.**
`declared_inputs_are_read_by_something` fires on *declared* inputs, and this
graph declares none — the input arrives at `run()` and is read by nobody. That
gap is worth a decision: today a run handed inputs no step reads is
indistinguishable from a correct one until you read the answer.

It is stale documentation, not a missing shim. The `User request:` header the
notebook described was introduced in `e748c9a` (2026-08-04), an ancestor of
`a4a5259` (2026-08-06). The prose documented behaviour that had been removed.
So the examples moved to the contract, in `0579b32`, and were re-run.

### The rule the Architect earned, on the model that could design the branch

The handoff's second open item — teach the Architect that a node which does not
name an input no longer receives it — was written as anticipation, and
`_BUILD_RULES` takes one entry per failure seen in a real transcript. On
`gpt-4o-mini` no such transcript appeared: the workflows it built interpolated
their inputs, and the live suite's only failure was the branching design.

On `gpt-5-mini` it appeared on the first build:

```
ticket_urgency_routing_and_reply: The workflow asks for 'ticket_text'
but no step uses it, so the value a caller passes is ignored.
```

Five steps, a router among them, and the graph input carrying the ticket named
by none of them. The warning is doing its job — and a warning is all it does,
so the workflow was still registerable. That is the evidence the rule needed.

### The measurement §3 of the handoff asked for

The map cell was reported at 51s for four small calls, with prompt bloat as the
suspect. `res.total_usage()` on that cell, on `gpt-4o-mini`:

```
input_tokens=1154  output_tokens=20      # four calls: 2 items × 2 body nodes
```

~290 input tokens per call. **Prompt bloat was not the cause**, and the standing
hypothesis — a local model's thinking tokens — survives.

### Running the suite on Windows

> **Superseded — 2026-08-10.** What follows was accurate when written and is no
> longer what happens: `D:\tmp` has since been created on that machine, so the
> `/tmp` fixtures now land somewhere real and those thirteen failures resolved
> into a different, smaller set. The current list is
> [WINDOWS_TEST_FAILURES.md](WINDOWS_TEST_FAILURES.md), where ten of sixteen
> turn out to be library bugs the `/tmp` failures were masking. Left here
> unedited: that the diagnosis depended on an undeclared directory existing is
> the useful part of it.

The offline suite is green on POSIX and shows **15 failures on Windows**. They
are environmental and every one of them fails identically on `main`, checked by
running the same modules in a worktree at `4065c2f`:

- `tests/tools/` (13) — fixtures hardcode `ToolContext(cwd=Path("/tmp"))`, which
  on Windows resolves to a drive-relative `\tmp` that does not exist, so the
  subprocess launch fails with `[WinError 267] The directory name is invalid`;
- `test_cli.py::test_file_permissions_are_owner_only` — `chmod` owner-only bits;
- `test_agent_loop.py::test_write_outside_scope_always_widens_and_persists`.

**Compare failure *lists*, not counts** (the §1 habit): the collected total is
the same on both platforms, so a count alone would have suggested the branch had
broken something.

Also: the offline suite is ~45s here, not the 24s recorded in §8, and
`NEUROSURFER_TEST_BASE_URL=http://127.0.0.1:9` is still what keeps it from
dialling LM Studio for twelve minutes.

### The bump, with the argument settled as far as evidence takes it

§5 of the plan recommends `1.1.0` unless something outside is reaching in by
submodule path. The prompt-contract work does **not** move that as much as the
handoff feared, and the reason is worth writing down:

- `_build_system_prompt` is private and gone — not a public break.
- `compose_user_prompt` changed its second parameter from `graph_inputs` to
  `task`, but `ManagerAgent` is not exported from `neurosurfer/__init__.py` and
  the method appears in no user-facing document. Internal.
- Nothing outside `executor/` imports the package's internals by path.

What *is* a real break is **behavioural, not API**: `main`'s `compose_user_prompt`
emitted an ambient `Additional inputs:` block, so a `1.0.0` workflow whose step
relied on being recited an input it never named will now run, go green, and
answer as though it had been passed nothing. That is the same class of break as
the submodule imports already recorded under *Changed* — silent rather than
loud, which is arguably worse for a caller.

**The call is still open**, and it is the last box. Minor is defensible if the
CHANGELOG entry is prominent — it now leads the *Changed* section and says what
to check. Major is defensible on the grounds that a silent behavioural change to
every workflow ever registered is exactly what a major version is for.

---

## §10 — The four open items, closed

**Shipped 2026-08-12.** Everything §9 and the roadmap left open on the graph
side, plus three defects the work turned up. **1235 passed, 7 skipped**, ruff
clean, on Linux — see *A red suite nobody had seen* below, because that number
was not what the branch actually had.

### The four that were planned

1. **The tool-round budget is a spec field.** `NodeKindSpec.tool_rounds`: `1` on
   `base`, `None` on `react`. `run_base_node` reads it rather than restating it,
   and a test asserts it does — one number, not two that drift. It buys the
   warning Phase 6 said it would: `agent.tools_exceed_rounds` fires on a `base`
   node holding two or more tools, because one round cannot chain one tool's
   result into the next. A **warning**, since two independent lookups answered in
   a single parallel round is a working step, and it stays quiet when
   `output_schema` is set because that is the other rule's finding.

2. **The Architect learned to name its inputs.** `_BUILD_RULES` gains the entry
   the `ticket_urgency_routing_and_reply` transcript earned — five steps, a
   router among them, and `ticket_text` named by none of them. Written from the
   transcript, per the one-entry-per-observed-failure rule the list is scoped by.
   `assemble.py`'s docstring is corrected with it: the goal suffix reads as an
   authored-tool convenience and is nothing of the sort under the current
   contract, where interpolation is the *only* route in for any node at all.

3. **The branching live test is two tests.** `test_agent_designs_a_branch…`
   runs with `verify="off"` and asks only whether the agent *designs* a branch;
   `test_agent_verifies_the_branch_it_designed…` keeps the default and is marked
   `slow`. The split matters because the assertion used to sit behind the repair
   loop, so a design that was right on the first plan was reported as a model
   that could not design a branch. **The structure half passes on
   `qwen/qwen3.5-9b` in 124s** — which is the whole point, and was previously
   invisible behind 17 non-converging graph runs.

4. **A code node's parameters count as reading an input.** Below.

### The three defects the work found

**`declared_inputs_are_read_by_something` had a false positive**, and it was
blocking item 2's real goal. The rule already knew that *"a tool node is handed
the inputs dict as kwargs, so a parameter name matching an input is a read"* —
and gated it on `kind == "tool"`. A `function` node gets identical treatment in
`executor/deterministic.py`, so the capstone tutorial was reported as ignoring
`db_path` and `artifacts_dir`, which its functions consume on every run.
Validation already imports the callable for `callable_resolves`, so the signature
was free. A callable declaring `**kwargs` reads whatever it is handed and
silences the rule entirely.

This mattered more than its size: the plan for this rule is to promote it from
warning to blocking, and promoting a rule with a known false positive would start
refusing correct graphs.

**A `react` node whose turn is all reasoning was a failed node.** `final_text`
accumulates `TextDelta`, so a local reasoning model ending a turn with only
`ThinkingDelta` — no tool call, no text — left it empty, and the node was
reported as having produced nothing, taking every node downstream with it.
`CanonicalResponse.text()` has resolved this the same way for the one-shot path
all along; the streamed path disagreed purely because it accumulates deltas.
`RunResult.final_thinking` is a *separate* channel — `TextDelta` is the answer
and `ThinkingDelta` is reasoning, and concatenating them would hand a caller
reasoning labelled as an answer — consulted last, after `report` and
`final_text`. **Measured on the capstone's vision node: 2 failures in 6 runs
before, 0 in 5 after.**

**An image named in a long prompt killed the run.** Every user turn is scanned
for image paths so `"explain /tmp/chart.png"` attaches the image without routing
through `read_file`. The scan tries the longest candidate first — for a node's
turn, the whole task text up to the extension. Past `NAME_MAX`, and
`Path.is_file()` only swallows `ENOENT`/`ENOTDIR`/`EBADF`/`ELOOP`, so the
`ENAMETOOLONG` escaped: the capstone's vision node died in **0 ms**, before the
model was asked anything. `AgenticLoop` runs this on every turn, so it hit the
CLI too.

### A red suite nobody had seen

The branch was recorded as *1207 pass, 16 fail only on Windows*. On Linux it was
**nine red**, and eight of them were one cause: `configure_logging` sets
`propagate = False` on the `neurosurfer` logger — correct for a library that owns
its handler — while `caplog` captures at the **root**. Records stopped one logger
short of it, so `caplog.records` was empty however loudly the code logged. The
message is right there in the captured stdout of every one of those failures.

Eight tests across `test_observability_exporters.py` and
`test_unread_run_inputs.py` were written against that gap and **could never have
passed**. `tests/conftest.py` now restores propagation for the duration of a
test; production behaviour is untouched.

The ninth was the documentation plan's fallout: `test_docs_index_finds_relevant_sections`
asserted on `guides/graph-workflows.md`, which plan 02 deliberately deleted when
it split that page into the `graph/` section. The retrieval is *better* now — the
query lands on `graph/packages.md` — so the assertion was updated to match the
section rather than one filename.

**Worth stating plainly: the suite was not being run green before this.** Both
numbers in §9 and the roadmap should be read as "on the author's machine, with
whatever logging state that shell had".


---

## §11 — The pre-merge verification, 2026-08-16

Run because the §8 tick was two plans and fifty commits stale. **It found two
real defects that 1500 passing tests did not**, which is the argument for doing
this deliberately rather than trusting the offline number.

### The gate

| Check | Result |
|---|---|
| Offline suite | **1521 passed, 5 skipped**, ~59s |
| Live suite, local `qwen/qwen3.5-9b` | 1521 passed, **1 failed** — `test_agent_designs_a_branch_with_real_llm` |
| Live suite, hosted `gpt-5-mini` | 1521 passed, **1 failed** — `test_agent_declares_blocked_with_real_llm` |
| ruff, `mkdocs build --strict`, both docs gates | clean |
| Tutorials 00–06 | **0 errors**, every one |
| Windows | **deferred** — see `WINDOWS_TEST_FAILURES.md` |

**Neither live failure is a defect, and each was checked rather than assumed.**
The branching one is the same model-capability boundary §8 recorded: the 9B built
two `base` nodes where a router was wanted, and the identical test passes on
`gpt-5-mini` in 274s. The blocked one is variance, not a boundary — `gpt-5-mini`
planned nine steps to do the Oracle request instead of refusing it, ran out of
nudges fixing validation, and the **same test passed on re-run** in 160s. Worth
knowing that it is variance and not a wall; worth not pretending it is a pass.

### The two defects, and why nothing caught them earlier

**A field named `title` was deleted from every schema it appeared in.**
`_strip_titles` filtered `k != "title"` at every level to remove pydantic's
annotation noise — and `properties` is keyed by *field names*. The model was shown
a schema without the field while `required` still demanded it, so structured
output failed on every attempt with "title Field required". Measured 0/3 before
and 3/3 after on the same prompt. It had been failing every run of tutorial 01,
and no test covered a field with that name.

**The OTLP dead-collector protection had been switched off by a library
upgrade.** `opentelemetry-exporter-otlp-proto-http` 1.44 retries internally and
returns `FAILURE` rather than raising, so a wrapper watching only for exceptions
never tripped: 6–7.6s per flush, on Linux, for anyone with `NEUROSURFER_EXPORTERS=otel`
and no collector. **The test that would have caught it had never run** — the
optional extra was not installed, so it skipped, silently, forever.

That second one is the transferable lesson: *an optional extra that is not
installed is a test that does not run*. The suite reported 7 skips and nobody read
them.

### The version

§5 recommended `1.1.0`, weighing only the `tools.builtin` submodule move. That
predates the cost removal. Four things now break on upgrade — the submodule paths,
`llm/pricing`, three public methods/fields, and a validation warning promoted to
an error that un-runs workflows already on disk. **`2.0.0`**, with the CHANGELOG's
upgrade table as the migration note.
