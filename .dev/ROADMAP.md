# Roadmap — one plan, in order

**Status:** current · branch `architect-validator/enhancement`, cut from `main`
(`4065c2f`) and pushed · build logs live beside each plan.

> **`main` is not touched until this plan is finished.** Every phase lands on the
> branch; `main` stays at `4065c2f` as the stable line while a fifty-commit port
> is in flight. When the phases are done and the full suite — **including the live
> tests** — has been run once deliberately, it merges to `main` **with a version
> bump**. See the plan's §5 for the checklist and the `1.1.0`-vs-`2.0.0` argument.

Latest: [01's build log](01_ARCHITECT_AND_VALIDATOR_BUILD_LOG.md) — **Phases 1–6
are in**. The Architect grounds and refuses, verifies by running what it built,
and the three defects the port carried across are fixed: a step now says what it
was actually given, and one cut off mid-plan stops reporting success with a blank
answer. 1142 tests pass from a 360 baseline, ruff clean (2026-08-04). Phase 7
next — tracing and docs, then the release checklist in the plan's §5.

**Running the suite:** 24s offline with
`NEUROSURFER_TEST_BASE_URL=http://127.0.0.1:9`. Without it, the live tests
default to LM Studio on `:1234` and take twelve minutes if it happens to be up.

---

## 01 — The Architect and the validator ⬅ **current**

**[01_ARCHITECT_AND_VALIDATOR_PLAN.md](01_ARCHITECT_AND_VALIDATOR_PLAN.md)**

Bring the matured Architect and validator onto the stable line, with the engine
floor they stand on, and without the studio.

Done when the Architect refuses a request it cannot ground — naming the missing
capability — instead of building something that fails at run time, and the
validator is the gate that proves it in language a person can act on.

**Built bottom-up, on purpose.** The engine can run it → the Architect can see it
→ the validator can prove it → the Architect designs against the proof. The
previous round ran that order backwards and every defect it found was a defect in
the primitive rather than in the authoring.

---

## What is deliberately not a plan of its own

- **The studio.** It lives at `NaumanHSA/neurosurfer-studio` and is not
  maintained here.
- **Validation** belongs to whichever plan owns the mechanism being checked. The
  last round had rules arriving as prompt text ahead of the structure that would
  make them enforceable, which is how a weak model came to be told three
  different times not to write a step it kept writing.
- **Evals** wait for something stable to measure. Nine runs of two prompts is a
  bug-finder, not a benchmark, and nothing above is stable until Phase 5.
- **A release.** The version bump is the *last* step of plan 01, not a phase of
  its own — there is nothing to release until the plan is done, and cutting one
  midway would put a half-ported Architect on `main`, which is the outcome §1
  exists to prevent.
