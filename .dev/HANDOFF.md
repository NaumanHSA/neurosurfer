# Handoff — 2026-08-06

Branch `architect-validator/enhancement`, pushed. `main` is still at `4065c2f`
and has not been touched.

This is a working handoff, not a build log: what changed, what is *not* verified,
and what to do first on the other machine. The build log (§9) for this work has
not been written — see "Documentation debt" below.

---

## 1. Where things stand

Ten commits this session, `ad393c1..a4a5259`, on top of the eleven
post-release-checklist commits that were already on the branch.

| | |
|---|---|
| Offline suite | **1183 passed, 4 skipped, ~22s** |
| Baseline before this session | 1159 |
| Ruff | clean across `neurosurfer/` and `tests/` |
| Live tests | **not run** — see §3 |
| `mkdocs build --strict` | not re-run since the doc edits |

Two things happened, in this order, and the second partly undid the first.

### Commit 1 — narrowing the inputs block (`ad393c1`)

Found by reading the prompts tutorial 03's `map` cell actually sends. Each of
four calls was handed both reviews, the index, the item, and the item again
inside the collection. Quadratic in the collection, and self-contradictory —
`reviews: [both]` beside `item: <one>` with nothing saying how they relate.

Fixed by separating **display** from **resolution**, keeping every value
resolvable and narrowing what got recited.

`render_scope` came out of this and **survives**: four sites used to assemble a
node's template scope by hand and all four disagreed, none of them seeing the
container scope. That omission is why `map` had to smuggle `{item}` in through
the body's graph inputs.

### Commit 10 — the contract that replaced it (`a4a5259`)

The narrowing kept arriving at the same place: any rule for *which inputs matter
to this node* is a worse version of a rule the author already wrote, in the
instruction's placeholders. Two of the rules were also each other's opposite —
"show only what the node references" and "hide what it already interpolated"
describe the same set, so together they show nothing. Which is the answer.

So the ambient `Inputs:` block is **gone**, and:

> A node's turn is **what its task text names, plus the outputs of the steps it
> declared as dependencies.** Nothing ambient.

And the task moved out of the system prompt into the user turn, because
rendering it into the system prompt made that prompt differ per node and per map
item, so **prompt caching could never fire**. `NODE_SYSTEM_PROMPT` is now one
constant, byte-identical for every call. This matches LangChain's `create_agent`,
which builds `system_message` once at construction and prepends it unchanged.

Deleted as subsumed: `_hidden_inputs`, `_hidden_body_inputs`, `_input_root`,
`recited_names`, `NOTHING_FURTHER_PROMPT`. All of them existed to decide what to
omit from a block that no longer exists. **If you are reading commit `ad393c1`
in isolation, half of what it adds was deleted eight commits later** — read
`a4a5259` for the contract that actually holds.

### The executor is a package (`5e2a8a5`..`613cfa8`)

`executor.py` (1,950 lines) → `executor/` (8 files, 2,322 lines):
`core.py` (scheduler + dispatch, 950), `iteration.py`, `routing.py`,
`deterministic.py`, `io_nodes.py`, `llm.py`, `_trace.py`, `__init__.py`.

Runners are **functions taking the executor** (`run_map_node(ex, node, state)`),
with `GraphExecutor` imported under `TYPE_CHECKING` only, so no runner module
has a runtime import of `core`. `GraphExecutor` keeps a one-line forwarder per
runner, so nothing outside the package changed.

---

## 2. If you touch the executor package, read this first

**The previous attempt at this split was reverted and cost a full session.** The
reason is now known and guarded, but the guard is easy to defeat.

### What killed it

Relative imports. `executor.py` sat *beside* its siblings, so `from .artifacts
import ArtifactStore` meant `engine.artifacts`. One directory down it means
`engine.executor.artifacts`, which does not exist. There were 26 of them, **14
inside methods** where no import-time check reaches them, and 12 on the
Architect's path — so it surfaced as thirteen unrelated test modules failing to
import, which looks nothing like a path problem.

One anchored rewrite fixed it. It is not hard; it is just invisible until it
isn't.

### The invariant that has no other guard

`node_runner.py` is the **only** engine module importing `neurosurfer.agents.*`,
and every use of it is a lazy import inside a function. That keeps
`import neurosurfer.graph.engine` from dragging in the agent stack. Hoist one to
a submodule's top level and the package `__init__` — which imports the
submodules — undoes it silently.

`tests/engine/test_import_boundaries.py` fails if that happens. It is the only
thing that would notice. **Do not delete it, and do not "tidy" the lazy
imports.**

### The mechanical net

Ruff selects `F`, so `F821 undefined name` fires on a leftover `self.` in a
module-level function. That is what made moving fourteen methods out of a class
a check a machine could finish — it caught six real omissions (`Usage`,
`GraphExecutionError`, `React`, `copy_context`, `FuturesTimeout`,
`import_string`), most on failure paths no offline test reaches. **Run
`ruff check` after every extraction, not just at the end.**

---

## 3. What is NOT verified — do this first

**Every prompt the framework emits changed, and the entire verification is
offline.** All 1183 tests drive a scripted provider. They prove the assembly.
They prove nothing about whether a real model behaves the same when the task
arrives in the user turn instead of the system prompt.

This repo's own convention — visible in the tutorial commit messages — is
"executed end to end against gpt-5-mini before committing". **The ten commits in
this session did not meet it.** That is the single biggest open risk.

To close it:

```bash
# tutorial 03, end to end, fresh kernel, outputs stripped afterwards
# then the four live tests:
conda run -n LLMs python -m pytest tests/architect/test_architect_agent_llm.py -q
```

- `test_agent_builds_simple_workflow_with_real_llm`
- `test_closed_loop_engine_with_real_llm`
- `test_agent_designs_branching_workflow_with_real_llm`
- `test_agent_declares_blocked_with_real_llm`

Note the build log's §8 already records the third of these failing on
`gpt-4o-mini` — that failure is a model-capability boundary, not a defect, and
predates this session. Check the model before the code.

There is also an open measurement worth taking while you are there: the map cell
was reported at 51s for four small calls. My view is that the prompt bloat was
not the cause and local-model thinking tokens were — `res.total_usage()` before
and after settles it.

---

## 4. Known-open items, in the order I would do them

1. **The live run above.** It can invalidate everything below it.
2. **Teach the Architect the new contract.** A generated node that does not name
   an input no longer receives it. `assemble.py:299` tells models to interpolate
   `{name}` only for *authored-tool* inputs — an understatement now. The new
   `declared_inputs_are_read_by_something` rule warns, but warnings do not
   block, so today the backstop is Phase 5 verification noticing the workflow
   ignores its parameter. That is an expensive way to find out, and the
   Architect runs on a weak model where correctness has to come from the prompt
   plus an enforced gate.
3. **Documentation debt** (below).
4. **The release checklist's last box** — merge to `main` and bump. It was
   already "awaiting the call" before this session; it now also waits on 1–3.

### Documentation debt

- `ROADMAP.md` still says "Phase 7 next". §7 and §8 both shipped on 2026-08-04.
- **The build log stops at §8.** There is no section for the eleven
  post-checklist commits (node classes, validation-on-run, the react `finish()`
  fix, the provider fix) *or* for this session's ten. Whether that is a §9 of
  plan 01 or the start of a plan 02 is a framing call nobody has made.
- `CHANGELOG.md`'s `[Unreleased]` mentions none of it. The prompt-contract change
  is the entry that decides `1.1.0` vs `2.0.0`: `compose_user_prompt` and
  `_build_system_prompt` both changed signature, and `_build_system_prompt` no
  longer exists under that name.

---

## 5. Environment notes

- **Always run Python in the conda env `LLMs`**, never base:
  `conda run -n LLMs python -m pytest ...`
- **Offline suite: set `NEUROSURFER_TEST_BASE_URL=http://127.0.0.1:9`.** Without
  it the live tests default to LM Studio on `:1234` and take twelve minutes if
  it happens to be up. With it: ~22 seconds.
- Importing the package prints a startup banner to stdout. Anything parsing
  subprocess output must take the last line (see `_fresh` in
  `test_import_boundaries.py`).
- `python-dotenv` warnings about unparseable `.env` lines are pre-existing noise.

### Reading a node's prompts

The `print`s used for the prompt debugging this session were committed by
accident in `a4a5259` and removed in the commit that carries this file. The view
is now logging, so it filters:

```python
import logging
logging.getLogger("neurosurfer.graph").setLevel(logging.DEBUG)
```

---

## 6. One thing that was investigated and is *not* a bug

In tutorial 03's map cell, the `verdict` node's prompt contains the original
review. That is not leakage — the node's own instruction says so:

```python
verdict = Base(
    id="verdict",
    depends_on=["summarise"],
    instructions="Reply with exactly one word — positive, negative or mixed:\n\n{item}",
)                                                                              # ← here
```

Remove `{item}` and its whole turn is the task plus
`Context from previous nodes: --- summarise ---`. `depends_on` was already doing
what it should. Judging sentiment from the original rather than a lossy 8-word
summary looks deliberate, so this is left as the tutorial has it.
