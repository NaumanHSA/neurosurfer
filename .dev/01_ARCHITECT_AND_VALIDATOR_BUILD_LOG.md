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

*Not started.*

---

## §2 — Phase 2: the Architect can see what exists

*Not started.*

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
