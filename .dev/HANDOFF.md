# Handoff — 2026-08-16

Branch `architect-validator/enhancement`, **pushed and clean**. `main` is still at
`4065c2f` and has not been touched.

**The branch is ready to merge.** The gate in
[plan 01 §5](01_ARCHITECT_AND_VALIDATOR_PLAN.md) is met and the evidence is in
[the build log §11](01_ARCHITECT_AND_VALIDATOR_BUILD_LOG.md). Nothing is merged
and the version is **not** bumped — both are the next person's first two moves.

---

## 1. Where things stand

80+ commits ahead of `main`; 380 files, +54k/−2k. This is a release, not an update.

| Check | Result |
|---|---|
| Offline suite | **1521 passed, 5 skipped**, ~59s |
| Live suite, local `qwen/qwen3.5-9b` | 1521 passed, **1 failed** |
| Live suite, hosted `gpt-5-mini` | 1521 passed, **1 failed** |
| ruff, `mkdocs build --strict` | clean |
| docs link + import gates | clean, 60 pages |
| Tutorials 00–06 | **0 errors, every one** |
| Windows | **deferred** — owner's call, see `WINDOWS_TEST_FAILURES.md` |

**Both live failures were checked, and neither is a defect.** The branching one is
the 9B not designing a router — the same test passes on `gpt-5-mini`. The blocked
one is variance: `gpt-5-mini` planned nine steps for the Oracle request instead of
refusing it, and the **same test passed on re-run**. Expect one or the other to be
red on any given live run; read §11 before treating it as a regression.

---

## 2. Do this first, on the other machine

```bash
git checkout architect-validator/enhancement && git pull
conda activate LLMs                       # never base — everything assumes this env
pip install -e ".[observability]"         # see §4; this is why two tests were skipping
NEUROSURFER_TEST_BASE_URL=http://127.0.0.1:9 python -m pytest -q
```

Expect **1521 passed**. If the number is lower, something did not come across;
diff the skip list before anything else.

---

## 3. The merge, in order

1. **Merge to `main`.** Push explicitly — `git push origin main` — the branch's
   upstream history makes a bare `git push` risky; see §5 of the plan.
2. **Bump to `2.0.0`** in *three* places that must move together:
   `pyproject.toml`, `neurosurfer/__init__.py`, and the CHANGELOG's
   `[Unreleased]` heading.
3. **Release notes**: the CHANGELOG's *Upgrading from 1.0.0* table is written and
   is the migration note. Add one sentence saying the release was **tested on
   Linux**.

### Why 2.0.0 and not 1.1.0

§5 recommended minor, weighing only the `tools.builtin` submodule move. That
predates the cost removal. Four things break on upgrade:

- `from neurosurfer.tools.builtin.<module> import …` — submodule paths (the
  package-level `from neurosurfer.tools.builtin import ReadFileTool` is fine)
- `neurosurfer.llm.pricing` — deleted outright
- `RunResult.cost()`, `RunResult.model`, `GraphExecutionResult.total_cost()`,
  `NodeExecutionResult.model`
- **a validation warning is now an error**, so a registered workflow declaring an
  input no step reads refuses to run — the only one of the four that breaks
  something already on disk rather than in source

---

## 4. Two things about the dev environment

**The `observability` extra was never installed here**, and that is not cosmetic.
The env had `opentelemetry-exporter-otlp-proto-grpc` (pulled in by langfuse) but
not `-proto-http`, which is what our code imports — so two tests skipped silently
for as long as they have existed, and one of them was guarding a real defect (§11).
`pip install -e ".[observability]"` on the new machine, and **read the skip list**
rather than the pass count.

**A version skew I introduced:** installing the http exporter directly brought in
`1.44.0` against a `1.42.1` SDK. `pyproject` pins nothing tighter than `>=1.20`,
so a fresh install gets the latest of everything and hits the same behaviour — the
finding stands for new users. Still worth installing the extra properly and
re-running `tests/test_observability_exporters.py` to confirm on a matched set.

---

## 5. Open decisions, both the owner's

- **`Operating System :: OS Independent`** in `pyproject.toml` is ahead of the
  evidence if this ships Linux-verified only. Recommendation: leave the
  classifier, say "tested on Linux" in the release notes. The code is
  cross-platform; it is the *verification* that is not.
- **A model that burns its turn budget still raises a bare `RuntimeError`**
  carrying the model's last rambling. Turning that into `WorkflowInfeasible` with
  the validation report would give the same *kind* of outcome on every model,
  which is the standard the Architect is now held to elsewhere. It is a change to
  the public terminal contract, so it was left alone deliberately.

---

## 6. Queued, not started

- **[04 — Plan review as a feature](04_PLAN_REVIEW_NOTES.md)** — notes only.
  Reviewing a plan is a callback the caller must write, off by default, showing a
  flat list, with no way to say *what* to change. Includes the four behaviours an
  implementation must not lose.
- **The repair loop's convergence number** on `gpt-5-mini`. The old figures (12
  and 17 rounds) predate the unread-input check blocking, so they are not a
  baseline any more. ~20 minutes and real spend when someone wants it.
- **The CLI still routes to `ArchitectBuilder`**, the older fixed 8-node pipeline,
  while everything else uses `ArchitectAgent`. Documented as a caveat in tutorial
  06; it is a real inconsistency, not a design choice.
- **Windows.** Ten documented failures, last measured ~50 commits ago. Seven are
  one upstream bug, three are `os.killpg` called unconditionally. Neither is this
  branch's work.
